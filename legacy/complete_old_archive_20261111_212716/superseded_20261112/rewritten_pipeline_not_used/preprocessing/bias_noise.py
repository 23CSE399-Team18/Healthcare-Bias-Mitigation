"""Data-noise bias: Denoising autoencoder, Wavelet, Median filter, Savitzky-Golay
(traditional); Confident Learning (cleanlab), robust loss (GCE), diffusion denoising (modern).

Detection: Confident Learning on the real training split (5-fold out-of-fold probabilities
from gradient boosting) estimates the share of likely label errors.
Benchmark: a training sub-sample is treated as the reference; Gaussian noise (0.25 SD) is
added to its continuous predictors and 10% of labels are flipped. Feature methods return
denoised features; label methods return flagged rows. Metrics: residual variance, spectral
distortion, MAE, RMSE, signal-to-noise ratio (vs the reference values) and label-error
detection F1 (vs the flipped rows). Feature methods flag no labels (F1 = 0); label methods
leave features unchanged.
Wavelet / median / Savitzky-Golay are 1-D signal filters. Survey/ICU rows have no time order,
so each continuous column is filtered along the rows' order on the first principal component
(a declared tabular adaptation).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pywt
import torch
from scipy.signal import medfilt, savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torch import nn
from torch.nn import functional as F

from common import SEED, Standardizer, mlp, predict_proba, select, stratified_sample, train_classifier

FAMILY = {"Denoising autoencoder": "Traditional", "Wavelet denoising": "Traditional", "Median filter": "Traditional",
          "Savitzky-Golay": "Traditional", "Confident Learning (Cleanlab)": "Modern",
          "Robust loss (GCE)": "Modern", "Diffusion denoising": "Modern"}
KIND = {"Confident Learning (Cleanlab)": "label", "Robust loss (GCE)": "label"}
NOTES = {"Wavelet denoising": "db4 wavelet, soft universal threshold (noise SD from the finest-level MAD), along PC1 order.",
         "Median filter": "window 5 along PC1 order.", "Savitzky-Golay": "window 11, quadratic, along PC1 order.",
         "Denoising autoencoder": "MLP autoencoder trained to reconstruct inputs from masked/noised copies.",
         "Confident Learning (Cleanlab)": "cleanlab.filter.find_label_issues on 5-fold out-of-fold gradient-boosting probabilities; flagged rows are removed when applied.",
         "Robust loss (GCE)": "MLP trained with generalized cross-entropy (q = 0.7), 2-fold cross-fitted; a row is flagged when the "
                              "out-of-fold probability of the other class exceeds that class's mean self-confidence (confident-learning threshold). Flagged rows are removed when applied.",
         "Diffusion denoising": "DDPM (100 steps) on continuous predictors; noisy rows are treated as diffusion states at the step whose "
                                "noise level matches the wavelet-MAD noise estimate and mapped back with deterministic DDIM steps."}
DIRECTIONS = {"Residual variance": "min", "Spectral distortion": "min", "MAE": "min", "RMSE": "min",
              "SNR (dB)": "max", "Label-error detection F1": "max"}


def cl_probabilities(X, y, seed=SEED):
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    return cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")


def cl_issues(X, y, seed=SEED):
    from cleanlab.filter import find_label_issues
    probs = cl_probabilities(X, y, seed)
    return find_label_issues(labels=y.astype(int), pred_probs=probs, filter_by="prune_by_noise_rate", n_jobs=1)


def detect(train, schema):
    X = train.X[schema.columns()].to_numpy(float)
    y = train.meta.target.to_numpy().astype(int)
    issues = cl_issues(X, y)
    rate = float(issues.mean())
    return {"statistic": "Confident-Learning estimated label-issue share (training)", "before": rate,
            "rule": "present if share >= 5%", "detected": bool(rate >= 0.05), "issues": issues}


# --------------------------------------------------------------------------- feature denoisers
def _pc1_order(Z):
    return np.argsort(PCA(1, random_state=SEED).fit_transform(Z)[:, 0], kind="stable")


def wavelet_sigma(Z, order):
    sig = []
    for j in range(Z.shape[1]):
        d = pywt.wavedec(Z[order, j], "db4", level=1)[-1]
        sig.append(np.median(np.abs(d)) / 0.6745)
    return np.array(sig)


def filter_columns(Z, method):
    order = _pc1_order(Z)
    out = Z.copy()
    for j in range(Z.shape[1]):
        s = Z[order, j]
        if method == "Wavelet denoising":
            coeffs = pywt.wavedec(s, "db4", mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr = sigma * np.sqrt(2 * np.log(len(s)))
            coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
            r = pywt.waverec(coeffs, "db4", mode="symmetric")[: len(s)]
        elif method == "Median filter":
            r = medfilt(s, 5)
        else:
            r = savgol_filter(s, 11, 2)
        out[order, j] = r
    return out


class DAE:
    def __init__(self, steps=2500, seed=SEED):
        self.steps, self.seed = steps, seed

    def fit(self, Z):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = Z.shape[1]
        self.net = nn.Sequential(mlp(d, 16, hidden=(64,)), nn.ReLU(), mlp(16, d, hidden=(64,)))
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        Zt = torch.from_numpy(Z.astype(np.float32))
        for _ in range(self.steps):
            x = Zt[torch.from_numpy(rng.choice(len(Z), min(256, len(Z)), replace=False))]
            noisy = x * (torch.rand_like(x) > 0.1).float() + 0.2 * torch.randn_like(x)
            loss = F.mse_loss(self.net(noisy), x)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def transform(self, Z):
        return np.concatenate([self.net(torch.from_numpy(Z[s:s + 65536].astype(np.float32))).numpy()
                               for s in range(0, len(Z), 65536)])


class Diffusion:
    def __init__(self, T=100, steps=4000, seed=SEED):
        self.T, self.steps, self.seed = T, steps, seed
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.abar = torch.cumprod(1 - self.betas, 0)

    def fit(self, Z):
        from bias_missing import _time_embedding
        self.emb = _time_embedding
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = Z.shape[1]
        self.net = mlp(d + 32, d, hidden=(256, 256), act=nn.SiLU)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        Zt = torch.from_numpy(Z.astype(np.float32))
        for _ in range(self.steps):
            x = Zt[torch.from_numpy(rng.choice(len(Z), min(256, len(Z)), replace=False))]
            t = torch.randint(0, self.T, (len(x),))
            eps = torch.randn_like(x)
            ab = self.abar[t][:, None]
            pred = self.net(torch.cat([ab.sqrt() * x + (1 - ab).sqrt() * eps, self.emb(t)], 1))
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def transform(self, Z, sigma):
        """Noisy x = clean + sigma*eps  <=>  x_t / sqrt(abar_t) with sigma^2 = (1-abar_t)/abar_t."""
        ratio = ((1 - self.abar) / self.abar).sqrt().numpy()
        tstar = int(np.clip(np.searchsorted(ratio, sigma), 1, self.T - 1))
        out = []
        for s in range(0, len(Z), 65536):
            x = torch.from_numpy(Z[s:s + 65536].astype(np.float32)) * self.abar[tstar].sqrt()
            for t in range(tstar, 0, -1):
                tt = torch.full((len(x),), t, dtype=torch.long)
                eps = self.net(torch.cat([x, self.emb(tt)], 1))
                ab, abp = self.abar[t], self.abar[t - 1]
                x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                x = abp.sqrt() * x0 + (1 - abp).sqrt() * eps
            out.append(x.numpy() / self.abar[0].sqrt().item())
        return np.concatenate(out), tstar


def gce_flags(X, y, seed=SEED):
    probs = np.zeros(len(y))
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
        net = train_classifier(X[a], y[a], loss="gce", steps=1500, seed=seed)
        probs[b] = predict_proba(net, X[b])
    p = np.column_stack([1 - probs, probs])
    thresholds = np.array([p[y == c, c].mean() for c in (0, 1)])
    other = 1 - y
    return p[np.arange(len(y)), other] > thresholds[other]


def denoise_features(name, Zc, fit_Zc=None):
    if name in ("Wavelet denoising", "Median filter", "Savitzky-Golay"):
        return filter_columns(Zc, name), {}
    fit_Zc = Zc if fit_Zc is None else fit_Zc
    if name == "Denoising autoencoder":
        return DAE().fit(fit_Zc).transform(Zc), {}
    sigma = float(np.median(wavelet_sigma(Zc, _pc1_order(Zc))))
    out, tstar = Diffusion().fit(fit_Zc).transform(Zc, sigma)
    return out, {"estimated_noise_sd": sigma, "start_step": tstar}


# --------------------------------------------------------------------------- benchmark / apply
def benchmark(train, schema, log, n_bench=6000, sigma=0.25, flip=0.10):
    rng = np.random.default_rng(SEED)
    y_all = train.meta.target.to_numpy().astype(int)
    ids = stratified_sample(None, y_all, n_bench, seed=SEED + 3)
    cont = schema.continuous
    st = Standardizer(train.X[cont].to_numpy(float))
    clean_c = st(train.X.iloc[ids][cont].to_numpy(float))
    other = [c for c in schema.columns() if c not in cont]
    rest = train.X.iloc[ids][other].to_numpy(float)
    noisy_c = clean_c + sigma * rng.standard_normal(clean_c.shape)
    y_clean = y_all[ids]
    flipped = rng.random(len(ids)) < flip
    y_noisy = np.where(flipped, 1 - y_clean, y_clean)
    lam_clean = np.linalg.eigvalsh(np.cov(clean_c.T))
    rows = {}
    for name in FAMILY:
        t0 = time.time()
        try:
            if KIND.get(name) == "label":
                X = np.column_stack([noisy_c, rest])
                flags = cl_issues(X, y_noisy) if name.startswith("Confident") else gce_flags(X, y_noisy)
                den = noisy_c
            else:
                den, _ = denoise_features(name, noisy_c)
                flags = np.zeros(len(ids), bool)
        except Exception as error:
            log(f"  noise/{name} FAILED: {error!r}")
            rows[name] = {m: np.nan for m in DIRECTIONS}
            continue
        r = den - clean_c
        lam = np.linalg.eigvalsh(np.cov(den.T))
        rows[name] = {"Residual variance": float(r.var(0).mean()),
                      "Spectral distortion": float(np.abs(lam - lam_clean).sum() / np.abs(lam_clean).sum()),
                      "MAE": float(np.abs(r).mean()), "RMSE": float(np.sqrt((r ** 2).mean())),
                      "SNR (dB)": float(10 * np.log10((clean_c ** 2).sum() / max((r ** 2).sum(), 1e-12))),
                      "Label-error detection F1": f1_score(flipped, flags, zero_division=0)}
        log(f"  noise/{name}: RMSE {rows[name]['RMSE']:.4f} label-F1 {rows[name]['Label-error detection F1']:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table.insert(1, "acts_on", ["labels" if KIND.get(m) else "features" for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"{len(ids):,} training rows as reference. Injected: Gaussian noise SD {sigma} (standardised units) on "
                f"{len(cont)} continuous predictors and {flipped.sum():,} flipped labels ({flip:.0%}). Feature metrics compare "
                "denoised with reference continuous values; SNR = 10 log10(sum reference^2 / sum residual^2); spectral "
                "distortion = relative L1 change of covariance eigenvalues.")
    return table, best, protocol


def apply(frames, schema, method, detection, log, fit_rows=20000):
    train = frames["train"]
    new = {s: f.copy() for s, f in frames.items()}
    y = train.meta.target.to_numpy().astype(int)
    if KIND.get(method) == "label":
        X = train.X[schema.columns()].to_numpy(float)
        flags = detection["issues"] if method.startswith("Confident") else gce_flags(X, y)
        keep = ~flags
        new["train"].X, new["train"].meta = train.X.loc[keep], train.meta.loc[keep]
        log(f"  {method}: removed {int(flags.sum()):,} flagged training rows")
        return new, {"removed_training_rows": int(flags.sum()), "changed": "labels (rows removed)"}
    cont = schema.continuous
    st = Standardizer(train.X[cont].to_numpy(float))
    Zc = st(train.X[cont].to_numpy(float))
    ids = stratified_sample(None, y, fit_rows, seed=SEED + 4)
    den, info = denoise_features(method, Zc, Zc[ids])
    values = pd.DataFrame(st.inverse(den), columns=cont, index=train.X.index)
    lo, hi = train.X[cont].min(), train.X[cont].max()
    new["train"].X = train.X.copy()
    new["train"].X[cont] = values.clip(lo, hi, axis=1)
    log(f"  {method}: denoised {len(cont)} continuous predictors on {len(Zc):,} training rows")
    return new, {"changed": "continuous training features", **info}
