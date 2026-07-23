"""Encoding / domain bias: DANN, CORAL, domain-conditional autoencoder (traditional);
IRM, optimal-transport domain adaptation (OTDA), domain-generalisation transformer (modern).

Domains are geographic: US census region (BRFSS state -> region, NHIS region, eICU hospital
region). NHANES has no domain identifier, so this step is not applicable to diabetes.
Encoding check: one-hot levels come from the training split; a validation/test category never
seen in training becomes an all-zero block (counted below).
Detection: 5-fold cross-validated one-vs-rest AUROC of a gradient-boosting domain classifier.
Benchmark (leave-one-domain-out on a training sub-sample): each region in turn is the unseen
target; methods that adapt (CORAL, OTDA, DANN, autoencoder) may use the target's unlabelled
features; IRM and the transformer only see source regions. Metrics per held-out region:
MMD, domain-classifier AUC, Wasserstein distance between source and target representations;
target AUROC; domain-generalisation gap (source-validation AUROC - target AUROC); worst-region
balanced accuracy and OOD performance (worst-region AUROC).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torch import nn
from torch.nn import functional as F

from common import SEED, Standardizer, mlp, predict_proba, rbf_mmd, select, stratified_sample, train_classifier

FAMILY = {"DANN": "Traditional", "CORAL": "Traditional", "Autoencoder (domain-conditional)": "Traditional",
          "IRM": "Modern", "OT domain adaptation": "Modern", "Domain-generalisation transformer": "Modern"}
DATA_LEVEL = ("CORAL", "OT domain adaptation", "Autoencoder (domain-conditional)")
NOTES = {"DANN": "Gradient-reversal domain-adversarial network (Ganin et al. 2016) with unlabelled target rows.",
         "CORAL": "Correlation alignment (Sun et al. 2016): source covariance whitened and re-coloured to the target.",
         "Autoencoder (domain-conditional)": "Encoder without domain input, decoder conditioned on the domain; the "
             "classifier uses the encoder code. Applied to data as reconstruction with the pooled domain code.",
         "IRM": "IRMv1 penalty (Arjovsky et al. 2019) over source regions as environments (penalty 100 after warm-up).",
         "OT domain adaptation": "POT Sinkhorn transport (entropic, median-normalised cost) with barycentric mapping "
             "(Courty et al. 2017).",
         "Domain-generalisation transformer": "Feature-token transformer encoder (2 layers) with a mean/covariance "
             "alignment penalty across source regions on the CLS representation."}
DIRECTIONS = {"MMD": "min", "Domain classifier AUC": "min", "Wasserstein distance": "min",
              "Worst-group accuracy": "max", "Domain generalization gap": "min",
              "Target-domain AUROC": "max", "OOD performance (worst AUROC)": "max"}


def domains_of(frame):
    return frame.meta["meta_domain_region"].fillna("Unknown").astype(str).to_numpy()


def unseen_levels(frames, schema):
    counts = {}
    for split in ("validation", "test"):
        X = frames[split].X
        counts[split] = {src: int((X[cols].fillna(-1).sum(axis=1) == 0).sum()) for src, cols in schema.onehot.items()}
    return counts


def detect(frames, schema):
    train = frames["train"]
    if schema.domain is None:
        return {"statistic": "no domain identifier in this dataset", "before": None, "detected": False,
                "applicable": False, "rule": "not applicable", "unseen_levels": unseen_levels(frames, schema)}
    d = domains_of(train)
    keep = d != "Unknown"
    ids = np.flatnonzero(keep)
    ids = ids[stratified_sample(None, d[ids], 60000, seed=SEED)]
    X = train.X.iloc[ids][schema.columns()].to_numpy(float)
    labels = pd.Categorical(d[ids])
    probs = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=SEED), X, labels.codes,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    auc = float(roc_auc_score(labels.codes, probs, multi_class="ovr", average="macro"))
    return {"statistic": "one-vs-rest AUROC of a domain classifier (training, 5-fold CV)", "before": auc,
            "rule": "present if AUROC > 0.60", "detected": bool(auc > 0.60), "applicable": True,
            "domains": {k: int(v) for k, v in pd.Series(d).value_counts().items()},
            "unseen_levels": unseen_levels(frames, schema)}


# --------------------------------------------------------------------------- models
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def balanced_weights(y):
    y = np.asarray(y)
    w = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))
    return w.astype(np.float32)


def _batches(rng, n, size):
    return torch.from_numpy(rng.choice(n, min(size, n), replace=n < size))


def train_dann(Xs, ys, Xt, steps=1500, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feat, head, disc = mlp(Xs.shape[1], 32, hidden=(64,)), nn.Linear(32, 1), mlp(32, 1, hidden=(32,))
    params = list(feat.parameters()) + list(head.parameters()) + list(disc.parameters())
    opt = torch.optim.Adam(params, 1e-3)
    Xs_, ys_, Xt_ = map(lambda a: torch.from_numpy(np.asarray(a, np.float32)), (Xs, ys, Xt))
    w = torch.from_numpy(balanced_weights(ys))
    for step in range(steps):
        i, j = _batches(rng, len(Xs), 256), _batches(rng, len(Xt), 256)
        lam = 2 / (1 + np.exp(-10 * step / steps)) - 1
        hs, ht = F.relu(feat(Xs_[i])), F.relu(feat(Xt_[j]))
        cls = (F.binary_cross_entropy_with_logits(head(hs).squeeze(-1), ys_[i], reduction="none") * w[i]).mean()
        dl = disc(GradReverse.apply(torch.cat([hs, ht]), lam)).squeeze(-1)
        dom = F.binary_cross_entropy_with_logits(dl, torch.cat([torch.zeros(len(i)), torch.ones(len(j))]))
        loss = cls + dom
        opt.zero_grad(); loss.backward(); opt.step()
    rep = lambda X: F.relu(feat(torch.from_numpy(np.asarray(X, np.float32)))).detach().numpy()
    prob = lambda X: torch.sigmoid(head(torch.from_numpy(rep(X)))).detach().numpy().ravel()
    return rep, prob


class CondAE(nn.Module):
    def __init__(self, d, n_dom):
        super().__init__()
        self.enc = mlp(d, 16, hidden=(64,))
        self.dec = mlp(16 + n_dom, d, hidden=(64,))

    def forward(self, x, dom):
        z = self.enc(x)
        return z, self.dec(torch.cat([z, dom], 1))


def train_cond_ae(X, dom_codes, n_dom, steps=2000, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ae = CondAE(X.shape[1], n_dom)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    Xt, D = torch.from_numpy(np.asarray(X, np.float32)), F.one_hot(torch.from_numpy(dom_codes), n_dom).float()
    for _ in range(steps):
        i = _batches(rng, len(X), 256)
        _, rec = ae(Xt[i], D[i])
        loss = F.mse_loss(rec, Xt[i])
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    return ae


def train_irm(X, y, env, steps=2000, penalty=100.0, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = mlp(X.shape[1], 1, hidden=(64, 32))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(np.asarray(X, np.float32)), torch.from_numpy(np.asarray(y, np.float32))
    w = torch.from_numpy(balanced_weights(y))
    envs = [np.flatnonzero(env == e) for e in np.unique(env) if (env == e).sum() >= 50]
    for step in range(steps):
        loss, pen = 0.0, 0.0
        for ids in envs:
            i = torch.from_numpy(rng.choice(ids, min(128, len(ids)), replace=len(ids) < 128))
            scale = torch.tensor(1.0, requires_grad=True)
            logits = net(Xt[i]).squeeze(-1) * scale
            l = (F.binary_cross_entropy_with_logits(logits, yt[i], reduction="none") * w[i]).mean()
            g = torch.autograd.grad(l, [scale], create_graph=True)[0]
            loss, pen = loss + l, pen + g.pow(2)
        lam = penalty if step >= 300 else 1.0
        total = (loss + lam * pen) / len(envs)
        if lam > 1:
            total = total / lam
        opt.zero_grad(); total.backward(); opt.step()
    net.eval()
    body = net[:-1]
    rep = lambda X: body(torch.from_numpy(np.asarray(X, np.float32))).detach().numpy()
    prob = lambda X: predict_proba(net, X)
    return rep, prob


class TokenTransformer(nn.Module):
    def __init__(self, d, width=32):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, width) * 0.1)
        self.b = nn.Parameter(torch.zeros(d, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, 64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(width, 1)

    def rep(self, x):
        tokens = x[:, :, None] * self.w[None] + self.b[None]
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], 1)
        return self.encoder(tokens)[:, 0]

    def forward(self, x):
        return self.head(self.rep(x))


def train_dg_transformer(X, y, env, steps=1000, penalty=1.0, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TokenTransformer(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(np.asarray(X, np.float32)), torch.from_numpy(np.asarray(y, np.float32))
    w = torch.from_numpy(balanced_weights(y))
    envs = [np.flatnonzero(env == e) for e in np.unique(env) if (env == e).sum() >= 50]
    for _ in range(steps):
        reps, loss = [], 0.0
        for ids in envs:
            i = torch.from_numpy(rng.choice(ids, min(64, len(ids)), replace=len(ids) < 64))
            r = net.rep(Xt[i])
            loss = loss + (F.binary_cross_entropy_with_logits(net.head(r).squeeze(-1), yt[i], reduction="none") * w[i]).mean()
            reps.append(r)
        align = 0.0
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                align = align + (reps[a].mean(0) - reps[b].mean(0)).pow(2).sum() + (torch.cov(reps[a].T) - torch.cov(reps[b].T)).pow(2).sum()
        total = loss / len(envs) + penalty * align
        opt.zero_grad(); total.backward(); opt.step()
    net.eval()

    def rep(X):
        with torch.no_grad():
            return np.concatenate([net.rep(torch.from_numpy(np.asarray(X[s:s + 4096], np.float32))).numpy()
                                   for s in range(0, len(X), 4096)])

    def prob(X):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(net(torch.from_numpy(np.asarray(X[s:s + 4096], np.float32)))).numpy().ravel()
                                   for s in range(0, len(X), 4096)])
    return rep, prob


def coral_map(Xs, Xt, eps=1.0):
    cs = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    ct = np.cov(Xt.T) + eps * np.eye(Xs.shape[1])
    W = np.real(np.linalg.inv(sqrtm(cs)) @ sqrtm(ct))
    return lambda X: (X - Xs.mean(0)) @ W + Xt.mean(0)


def ot_map(Xs, Xt, seed=SEED, n=2000):
    from ot.da import SinkhornTransport
    rng = np.random.default_rng(seed)
    a = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    b = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
    model = SinkhornTransport(reg_e=0.1, norm="median", max_iter=500).fit(Xs=a, Xt=b)
    return lambda X: np.concatenate([model.transform(Xs=X[s:s + 20000], batch_size=2048) for s in range(0, len(X), 20000)])


# --------------------------------------------------------------------------- benchmark
def _repr_metrics(rs, rt, seed=SEED):
    st = Standardizer(rs)
    a, b = st(rs), st(rt)
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b), 1500)
    ia, ib = rng.choice(len(a), n, replace=False), rng.choice(len(b), n, replace=False)
    Z, lab = np.vstack([a[ia], b[ib]]), np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(LogisticRegression(max_iter=1000), Z, lab, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    from scipy.stats import wasserstein_distance
    wd = np.mean([wasserstein_distance(a[ia, j], b[ib, j]) for j in range(a.shape[1])])
    return rbf_mmd(a, b, n=800, seed=seed), float(max(roc_auc_score(lab, p), 1 - roc_auc_score(lab, p))), float(wd)


def run_method(name, Xs, ys, ds, Xt, scal_idx):
    """Return (representation fn, probability fn, transformed-source-for-repr)."""
    Xs = np.asarray(Xs, float)
    if name in ("CORAL", "OT domain adaptation"):
        mapper = coral_map(Xs[:, scal_idx], Xt[:, scal_idx]) if name == "CORAL" else ot_map(Xs[:, scal_idx], Xt[:, scal_idx])
        Xs2 = Xs.copy()
        Xs2[:, scal_idx] = mapper(Xs[:, scal_idx])
        net = train_classifier(Xs2, ys, weights=balanced_weights(ys), steps=1200)
        return (lambda X: X), (lambda X: predict_proba(net, X)), Xs2
    if name == "DANN":
        rep, prob = train_dann(Xs, ys, Xt)
        return rep, prob, None
    if name == "Autoencoder (domain-conditional)":
        codes = pd.Categorical(np.r_[ds, np.full(len(Xt), "__target__")])
        ae = train_cond_ae(np.vstack([Xs, Xt]), codes.codes.astype(np.int64), len(codes.categories))
        enc = lambda X: ae.enc(torch.from_numpy(np.asarray(X, np.float32))).detach().numpy()
        net = train_classifier(enc(Xs), ys, weights=balanced_weights(ys), steps=1200)
        return enc, (lambda X: predict_proba(net, enc(X))), None
    if name == "IRM":
        rep, prob = train_irm(Xs, ys, ds)
        return rep, prob, None
    rep, prob = train_dg_transformer(Xs, ys, ds)
    return rep, prob, None


def benchmark(frames, schema, log, n_bench=12000, n_target=2000):
    train = frames["train"]
    d_all = domains_of(train)
    y_all = train.meta.target.to_numpy().astype(int)
    ids = stratified_sample(None, d_all, n_bench, seed=SEED + 7)
    X = train.X.iloc[ids][schema.columns()].to_numpy(float)
    y, d = y_all[ids], d_all[ids]
    scal_idx = [schema.columns().index(c) for c in schema.scalable]
    targets = [g for g in pd.unique(d) if g != "Unknown" and (d == g).sum() >= 200]
    rng = np.random.default_rng(SEED)
    per_fold, rows = {}, {}
    for name in FAMILY:
        t0 = time.time()
        fold = []
        try:
            for g in targets:
                tmask = d == g
                tid = np.flatnonzero(tmask)
                tid = rng.choice(tid, min(n_target, len(tid)), replace=False)
                sid = np.flatnonzero(~tmask)
                sval = rng.random(len(sid)) < 0.2
                s_tr, s_va = sid[~sval], sid[sval]
                rep, prob, Xs_rep = run_method(name, X[s_tr], y[s_tr], d[s_tr], X[tid], scal_idx)
                pt, pv = prob(X[tid]), prob(X[s_va])
                rs = rep(X[s_tr]) if Xs_rep is None else Xs_rep
                mmd, dauc, wd = _repr_metrics(rs, rep(X[tid]))
                fold.append({"target": g, "MMD": mmd, "Domain classifier AUC": dauc, "Wasserstein distance": wd,
                             "target_auroc": roc_auc_score(y[tid], pt), "source_auroc": roc_auc_score(y[s_va], pv),
                             "target_bacc": balanced_accuracy_score(y[tid], pt >= 0.5)})
        except Exception as error:
            log(f"  domain/{name} FAILED: {error!r}")
            rows[name] = {m: np.nan for m in DIRECTIONS}
            continue
        f = pd.DataFrame(fold)
        per_fold[name] = f
        rows[name] = {"MMD": f.MMD.mean(), "Domain classifier AUC": f["Domain classifier AUC"].mean(),
                      "Wasserstein distance": f["Wasserstein distance"].mean(),
                      "Worst-group accuracy": f.target_bacc.min(),
                      "Domain generalization gap": (f.source_auroc - f.target_auroc).mean(),
                      "Target-domain AUROC": f.target_auroc.mean(), "OOD performance (worst AUROC)": f.target_auroc.min()}
        log(f"  domain/{name}: target AUROC {rows[name]['Target-domain AUROC']:.3f} MMD {rows[name]['MMD']:.4f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table.insert(1, "output", ["data transform" if m in DATA_LEVEL else "model representation" for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"Leave-one-region-out over {len(targets)} regions on {len(ids):,} training rows (target capped at "
                f"{n_target:,} rows; 20% of source rows held out for the source AUROC). Classifiers use class-balanced "
                "weights; accuracy = balanced accuracy at 0.5. Representation metrics compare source and target after "
                "each method, z-scored with source statistics (MMD: RBF, median bandwidth).")
    return table, best, protocol, per_fold


# --------------------------------------------------------------------------- apply (data-level)
def apply(frames, schema, method, log, sample=3000):
    """Map every region's continuous/ordinal predictors onto the pooled training distribution."""
    train = frames["train"]
    cols = schema.scalable
    d_tr = domains_of(train)
    rng = np.random.default_rng(SEED)
    pooled = train.X[cols].to_numpy(float)
    ref = pooled[rng.choice(len(pooled), min(sample, len(pooled)), replace=False)]
    new = {s: f.copy() for s, f in frames.items()}
    mappers = {}
    if method == "Autoencoder (domain-conditional)":
        codes = pd.Categorical(d_tr)
        ids = rng.choice(len(pooled), min(40000, len(pooled)), replace=False)
        st = Standardizer(pooled)
        ae = train_cond_ae(st(pooled[ids]), codes.codes[ids].astype(np.int64), len(codes.categories))
        mean_code = torch.from_numpy(np.bincount(codes.codes, minlength=len(codes.categories)) / len(codes)).float()
        for split, frame in frames.items():
            Z = torch.from_numpy(st(frame.X[cols].to_numpy(float)).astype(np.float32))
            with torch.no_grad():
                rec = ae.dec(torch.cat([ae.enc(Z), mean_code.expand(len(Z), -1)], 1)).numpy()
            new[split].X[cols] = st.inverse(rec)
        log(f"  {method}: reconstructed every split with the pooled domain code")
        return new, {"domains": list(codes.categories)}
    for g in pd.unique(d_tr):
        if g == "Unknown":
            continue
        Xg = pooled[d_tr == g]
        mappers[g] = coral_map(Xg, ref) if method == "CORAL" else ot_map(Xg, ref, n=sample)
    for split, frame in frames.items():
        d = domains_of(frame)
        values = frame.X[cols].to_numpy(float).copy()
        for g, mapper in mappers.items():
            m = d == g
            if m.any():
                values[m] = mapper(values[m])
        new[split].X[cols] = values
    log(f"  {method}: aligned {len(mappers)} regions to the pooled training distribution in every split")
    return new, {"domains": list(mappers)}
