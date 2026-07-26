"""Class-imbalance bias: SMOTE, ADASYN, Borderline-SMOTE, NearMiss (traditional); TabDDPM,
CTGAN, class-balanced focal loss, DeepSMOTE (modern).

Only the training split is resampled. Synthetic rows are flagged (meta_synthetic = 1) and inherit
metadata (sex/age/race group, region, IDs suffixed) from their nearest real minority row.
Benchmark: each method rebalances a training sub-sample; the same MLP classifier (class-balanced
focal loss for that method) is trained and scored on untouched validation rows.
Metrics: imbalance ratio and class entropy of the (weighted) training classes; minority-class
F1 ('balanced F1'), macro-F1, balanced accuracy, PR-AUC (average precision), G-mean and the
Kolmogorov-Smirnov separation of validation scores; TVD = mean total-variation distance between
the per-feature distributions of the resampled and original training class it changed
(fidelity / distortion; 0 when the data are unchanged).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import NearMiss
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from common import SEED, Standardizer, mlp, postprocess, predict_proba, select, stratified_sample, train_classifier

FAMILY = {"SMOTE": "Traditional", "ADASYN": "Traditional", "Borderline-SMOTE": "Traditional", "NearMiss": "Traditional",
          "TabDDPM": "Modern", "CTGAN": "Modern", "Class-Balanced Focal Loss": "Modern", "DeepSMOTE": "Modern"}
NOTES = {"TabDDPM": "Class-conditional Gaussian diffusion (100 steps, MLP denoiser) over all encoded columns; one-hot "
                    "columns are projected back to one category (the original uses multinomial diffusion for them).",
         "CTGAN": "Re-implemented conditional WGAN-GP with PacGAN (pac 10) and Gumbel-softmax one-hot outputs; "
                  "continuous columns are standardised instead of mode-specific normalised. The ctgan package is not installed.",
         "Class-Balanced Focal Loss": "Cui et al. 2019 effective-number weights (beta 0.999) with focal gamma 2. Changes the "
                                      "training loss, not the rows; exported as class-balanced sample weights.",
         "DeepSMOTE": "Autoencoder (Dablain et al. 2022) with SMOTE interpolation in the latent space, decoded back.",
         "NearMiss": "NearMiss-1 undersampling of the majority class."}
DIRECTIONS = {"Imbalance ratio": "min", "Class entropy": "max", "Balanced F1-score": "max", "Macro-F1": "max",
              "Balanced accuracy": "max", "PR-AUC": "max", "G-mean": "max", "KS statistic": "max", "TVD": "min"}


def detect(train):
    y = train.meta.target.to_numpy()
    w = train.meta.sample_weight.to_numpy()
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    ir = float(max(pos, neg) / max(min(pos, neg), 1e-9))
    return {"statistic": "imbalance ratio majority/minority (training, weighted)", "before": ir,
            "minority_share": float(min(pos, neg) / (pos + neg)), "rule": "present if ratio >= 2",
            "detected": bool(ir >= 2)}


def cb_weights(y, beta=0.999):
    counts = np.bincount(y.astype(int), minlength=2)
    eff = (1 - beta ** counts) / (1 - beta)
    w = 1 / eff
    return w / w.sum() * 2


# --------------------------------------------------------------------------- generators
def _time_emb(t, dim=32):
    from bias_missing import _time_embedding
    return _time_embedding(t, dim)


class TabDDPM:
    def __init__(self, T=100, steps=4000, seed=SEED):
        self.T, self.steps, self.seed = T, steps, seed
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.betas
        self.abar = torch.cumprod(self.alpha, 0)

    def fit(self, Z, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = Z.shape[1]
        self.net = mlp(d + 32 + 2, d, hidden=(256, 256, 256), act=nn.SiLU)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        Zt, yt = torch.from_numpy(Z.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
        for _ in range(self.steps):
            i = torch.from_numpy(rng.choice(len(Z), min(256, len(Z)), replace=False))
            x, c = Zt[i], F.one_hot(yt[i], 2).float()
            t = torch.randint(0, self.T, (len(i),))
            eps = torch.randn_like(x)
            ab = self.abar[t][:, None]
            loss = F.mse_loss(self.net(torch.cat([ab.sqrt() * x + (1 - ab).sqrt() * eps, _time_emb(t), c], 1)), eps)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def sample(self, n, cls, d):
        torch.manual_seed(self.seed + 11)
        out = []
        for s in range(0, n, 20000):
            m = min(20000, n - s)
            x = torch.randn(m, d)
            c = F.one_hot(torch.full((m,), cls, dtype=torch.long), 2).float()
            for t in reversed(range(self.T)):
                eps = self.net(torch.cat([x, _time_emb(torch.full((m,), t, dtype=torch.long)), c], 1))
                x = (x - self.betas[t] / (1 - self.abar[t]).sqrt() * eps) / self.alpha[t].sqrt()
                if t > 0:
                    x = x + ((1 - self.abar[t - 1]) / (1 - self.abar[t]) * self.betas[t]).sqrt() * torch.randn_like(x)
            out.append(x.numpy())
        return np.concatenate(out)


class CTGAN:
    def __init__(self, blocks, steps=3000, pac=10, seed=SEED):
        self.blocks, self.steps, self.pac, self.seed = blocks, steps, pac, seed

    def _activate(self, raw, hard=False):
        out = raw.clone()
        for b in self.blocks:
            out[:, b] = F.gumbel_softmax(raw[:, b], tau=0.2, hard=hard)
        return out

    def fit(self, Z, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d, z_dim = Z.shape[1], 64
        self.z_dim = z_dim
        self.G = mlp(z_dim + 2, d, hidden=(256, 256))
        self.D = mlp((d + 2) * self.pac, 1, hidden=(256, 256), act=lambda: nn.LeakyReLU(0.2))
        og = torch.optim.Adam(self.G.parameters(), 2e-4, betas=(0.5, 0.9))
        od = torch.optim.Adam(self.D.parameters(), 2e-4, betas=(0.5, 0.9))
        Zt = torch.from_numpy(Z.astype(np.float32))
        by_class = [np.flatnonzero(y == c) for c in (0, 1)]
        bs = 50 * self.pac
        for _ in range(self.steps):
            cls = rng.integers(0, 2, bs)                                  # training-by-sampling (balanced)
            real = Zt[np.where(cls == 0, rng.choice(by_class[0], bs), rng.choice(by_class[1], bs))]
            cond = F.one_hot(torch.from_numpy(cls), 2).float()
            fake = self._activate(self.G(torch.cat([torch.randn(bs, z_dim), cond], 1)))
            def critic(x):
                return self.D(torch.cat([x, cond], 1).reshape(bs // self.pac, -1))
            eps = torch.rand(bs // self.pac, 1, 1).expand(-1, self.pac, d + 2).reshape(bs, d + 2)
            mix = (eps * torch.cat([real, cond], 1) + (1 - eps) * torch.cat([fake.detach(), cond], 1)).requires_grad_(True)
            gp_out = self.D(mix.reshape(bs // self.pac, -1))
            grad = torch.autograd.grad(gp_out.sum(), mix, create_graph=True)[0].reshape(bs // self.pac, -1)
            gp = ((grad.norm(dim=1) - 1) ** 2).mean()
            ld = critic(fake.detach()).mean() - critic(real).mean() + 10 * gp
            od.zero_grad(); ld.backward(); od.step()
            fake = self._activate(self.G(torch.cat([torch.randn(bs, z_dim), cond], 1)))
            lg = -critic(fake).mean()
            og.zero_grad(); lg.backward(); og.step()
        return self

    @torch.no_grad()
    def sample(self, n, cls, d):
        torch.manual_seed(self.seed + 12)
        cond = F.one_hot(torch.full((n,), cls, dtype=torch.long), 2).float()
        return self._activate(self.G(torch.cat([torch.randn(n, self.z_dim), cond], 1)), hard=True).numpy()


class DeepSMOTE:
    def __init__(self, steps=3000, latent=16, seed=SEED):
        self.steps, self.latent, self.seed = steps, latent, seed

    def fit(self, Z, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = Z.shape[1]
        self.enc, self.dec = mlp(d, self.latent, hidden=(128,)), mlp(self.latent, d, hidden=(128,))
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3)
        Zt = torch.from_numpy(Z.astype(np.float32))
        for _ in range(self.steps):
            x = Zt[torch.from_numpy(rng.choice(len(Z), min(256, len(Z)), replace=False))]
            loss = F.mse_loss(self.dec(self.enc(x)), x)
            opt.zero_grad(); loss.backward(); opt.step()
        self.Z, self.y = Z, y
        return self

    @torch.no_grad()
    def sample(self, n, cls, d):
        rng = np.random.default_rng(self.seed + 13)
        H = self.enc(torch.from_numpy(self.Z[self.y == cls].astype(np.float32))).numpy()
        nn_ = NearestNeighbors(n_neighbors=min(6, len(H))).fit(H).kneighbors(H, return_distance=False)[:, 1:]
        base = rng.integers(0, len(H), n)
        mate = nn_[base, rng.integers(0, nn_.shape[1], n)]
        new = H[base] + rng.random((n, 1)) * (H[mate] - H[base])
        return self.dec(torch.from_numpy(new.astype(np.float32))).numpy()


GENERATIVE = {"TabDDPM": TabDDPM, "CTGAN": CTGAN, "DeepSMOTE": DeepSMOTE}


# --------------------------------------------------------------------------- resampling
def resample(name, X, y, schema, cols, seed=SEED):
    """Return (X_new, y_new, weights, source_index) where source_index = -1 for synthetic rows,
    otherwise the row position in X. Values are in the frame's units."""
    n = len(y)
    if name == "Class-Balanced Focal Loss":
        w = cb_weights(y)[y.astype(int)]
        return X, y, w / w.mean(), np.arange(n)
    if name == "NearMiss":
        sampler = NearMiss(version=1, n_jobs=-1).fit(X, y)
        sampler.fit_resample(X, y)
        idx = np.sort(sampler.sample_indices_)
        return X[idx], y[idx], np.ones(len(idx)), idx
    minority = int(np.bincount(y.astype(int)).argmin())
    need = int(np.ptp(np.bincount(y.astype(int), minlength=2)))
    if name in ("SMOTE", "ADASYN", "Borderline-SMOTE"):
        sampler = {"SMOTE": SMOTE, "ADASYN": ADASYN, "Borderline-SMOTE": BorderlineSMOTE}[name](random_state=seed)
        Xr, yr = sampler.fit_resample(X, y)
        synth = Xr[n:]
    else:
        st = Standardizer(X)
        blocks = [[cols.index(c) for c in b] for b in schema.onehot.values()]
        model = (CTGAN(blocks) if name == "CTGAN" else GENERATIVE[name]()).fit(st(X), y)
        synth = st.inverse(model.sample(need, minority, X.shape[1]))
    synth = postprocess(pd.DataFrame(synth, columns=cols), schema).to_numpy(float)
    ys = np.full(len(synth), minority)
    return np.vstack([X, synth]), np.r_[y, ys], np.ones(n + len(synth)), np.r_[np.arange(n), -np.ones(len(synth), int)]


def _tvd(a, b, bins=20):
    out = []
    for j in range(a.shape[1]):
        lo, hi = np.nanmin(np.r_[a[:, j], b[:, j]]), np.nanmax(np.r_[a[:, j], b[:, j]])
        if hi <= lo:
            continue
        ha = np.histogram(a[:, j], bins, (lo, hi))[0] / len(a)
        hb = np.histogram(b[:, j], bins, (lo, hi))[0] / len(b)
        out.append(0.5 * np.abs(ha - hb).sum())
    return float(np.mean(out)) if out else 0.0


def benchmark(frames, schema, log, n_train=12000, n_val=6000):
    cols = schema.columns()
    train, val = frames["train"], frames["validation"]
    ids = stratified_sample(None, train.meta.target.to_numpy(), n_train, seed=SEED + 8)
    vids = stratified_sample(None, val.meta.target.to_numpy(), n_val, seed=SEED + 9)
    X, y = train.X.iloc[ids][cols].to_numpy(float), train.meta.target.to_numpy().astype(int)[ids]
    w0 = train.meta.sample_weight.to_numpy()[ids]
    Xv, yv = val.X.iloc[vids][cols].to_numpy(float), val.meta.target.to_numpy().astype(int)[vids]
    minority = int(np.bincount(y).argmin())
    rows = {}
    for name in FAMILY:
        t0 = time.time()
        try:
            Xr, yr, wr, src = resample(name, X, y, schema, cols)
            wr = wr * np.where(src >= 0, w0[np.maximum(src, 0)], 1.0)
            if name == "Class-Balanced Focal Loss":
                net = train_classifier(Xr, yr, loss="focal_cb", class_weights=cb_weights(y), steps=1500)
            else:
                net = train_classifier(Xr, yr, weights=wr, steps=1500)
            p = predict_proba(net, Xv)
        except Exception as error:
            log(f"  imbalance/{name} FAILED: {error!r}")
            rows[name] = {m: np.nan for m in DIRECTIONS}
            continue
        pred = (p >= 0.5).astype(int)
        cw = np.array([wr[yr == c].sum() for c in (0, 1)])
        share = cw / cw.sum()
        changed = Xr[(yr == minority) & (src < 0)] if (src < 0).any() else None
        if name == "NearMiss":
            tvd = _tvd(X[y != minority], Xr[yr != minority])
        elif changed is not None:
            tvd = _tvd(X[y == minority], changed)
        else:
            tvd = 0.0
        rows[name] = {"Imbalance ratio": float(cw.max() / cw.min()),
                      "Class entropy": float(-(share * np.log2(share)).sum()),
                      "Balanced F1-score": f1_score(yv, pred, pos_label=minority, zero_division=0),
                      "Macro-F1": f1_score(yv, pred, average="macro", zero_division=0),
                      "Balanced accuracy": balanced_accuracy_score(yv, pred),
                      "PR-AUC": average_precision_score(yv == minority, p if minority == 1 else 1 - p),
                      "G-mean": float(np.sqrt(recall_score(yv, pred, pos_label=1, zero_division=0) *
                                              recall_score(yv, pred, pos_label=0, zero_division=0))),
                      "KS statistic": float(ks_2samp(p[yv == 1], p[yv == 0]).statistic),
                      "TVD": tvd}
        log(f"  imbalance/{name}: bal-acc {rows[name]['Balanced accuracy']:.3f} PR-AUC {rows[name]['PR-AUC']:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"{len(ids):,} training rows resampled; identical MLP trained on each result and scored at threshold 0.5 "
                f"on {len(vids):,} untouched validation rows. 'Balanced F1' = F1 of the minority class. TVD uses "
                "20-bin histograms per predictor.")
    return table, best, protocol


def apply(frames, schema, method, log, fit_rows=60000):
    cols = schema.columns()
    train = frames["train"]
    X, y = train.X[cols].to_numpy(float), train.meta.target.to_numpy().astype(int)
    if method in GENERATIVE and len(y) > fit_rows:
        # fit the generator on a stratified sample; generate enough rows to balance the full split
        ids = stratified_sample(None, y, fit_rows, seed=SEED + 10)
        st = Standardizer(X[ids])
        blocks = [[cols.index(c) for c in b] for b in schema.onehot.values()]
        model = (CTGAN(blocks) if method == "CTGAN" else GENERATIVE[method]()).fit(st(X[ids]), y[ids])
        minority = int(np.bincount(y).argmin())
        need = int(np.ptp(np.bincount(y, minlength=2)))
        synth = postprocess(pd.DataFrame(st.inverse(model.sample(need, minority, X.shape[1])), columns=cols), schema)
        Xr = np.vstack([X, synth.to_numpy(float)])
        yr, wr = np.r_[y, np.full(need, minority)], np.ones(len(y) + need)
        src = np.r_[np.arange(len(y)), -np.ones(need, int)]
    else:
        Xr, yr, wr, src = resample(method, X, y, schema, cols)
    synthetic = src < 0
    assert not synthetic.any() or synthetic[-int(synthetic.sum()):].all(), "synthetic rows must follow real rows"
    meta = train.meta.iloc[src[~synthetic]].copy()
    if synthetic.any():
        # metadata of a synthetic row = that of its nearest real minority row (standardised features)
        minority = int(np.bincount(y).argmin())
        real_min = np.flatnonzero(y == minority)
        st = Standardizer(X)
        nn_ = NearestNeighbors(n_neighbors=1).fit(st(X[real_min]))
        parent = real_min[nn_.kneighbors(st(Xr[synthetic]), return_distance=False)[:, 0]]
        extra = train.meta.iloc[parent].copy()
        for c in [c for c in extra.columns if c in ("meta_record_id", "meta_patientunitstayid")]:
            extra[c] = [f"synthetic-{i}-from-{v}" for i, v in enumerate(extra[c].astype(str))]
        extra["meta_synthetic"] = 1
        meta = pd.concat([meta, extra])
    meta["target"] = yr
    meta["sample_weight"] = meta["sample_weight"].to_numpy() * wr if method != "Class-Balanced Focal Loss" else wr
    meta = meta.reset_index(drop=True)
    new = {s: f.copy() for s, f in frames.items()}
    new["train"].X = pd.DataFrame(Xr, columns=cols)
    new["train"].meta = meta
    log(f"  {method}: training rows {len(y):,} -> {len(yr):,} (synthetic {int(synthetic.sum()):,})")
    return new, {"synthetic_rows": int(synthetic.sum()), "rows_after": len(yr)}
