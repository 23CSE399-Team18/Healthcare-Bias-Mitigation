"""Missing-data bias: GMM, Random-Forest (missForest), GAIN, MICE, KNN (traditional);
HyperImpute, CSDI, MIP-VAE (modern).

Benchmark: on a training sub-sample, an extra 20% of observed cells is hidden completely at
random (one-hot blocks are hidden as a whole). Each method imputes the hidden cells and
returns M draws, so every metric in the method table can be computed on known truth:
Wasserstein distance, correlation preservation, MAE, RMSE, NRMSE, CRPS and NLL
(numeric predictors, standardised units). Methods without a native predictive distribution
are given one as stated in `DRAWS`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required before IterativeImputer)
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.nn import functional as F

from common import SEED, Standardizer, mlp, postprocess, select, stratified_sample, wasserstein_1d

FAMILY = {"GMM": "Traditional", "Random Forest": "Traditional", "GAIN": "Traditional", "MICE": "Traditional",
          "KNN": "Traditional", "HyperImpute": "Modern", "CSDI": "Modern", "MIP-VAE": "Modern"}
DRAWS = {"GMM": "samples from the fitted mixture's conditional distribution",
         "Random Forest": "predictions of individual trees (missForest, final iteration)",
         "GAIN": "generator outputs for independent noise draws",
         "MICE": "M chained-equation imputations with posterior sampling",
         "KNN": "values of the k=5 nearest donors",
         "HyperImpute": "draws of the column model chosen by cross-validation (Bayesian-ridge posterior, trees or neighbours)",
         "CSDI": "reverse-diffusion samples",
         "MIP-VAE": "importance-resampled decoder draws (MIWAE)"}
NOTES = {"HyperImpute": "Re-implementation of HyperImpute's column-wise AutoML iterative imputation (model pool: Bayesian ridge, random forest, k-NN; 3-fold CV per column). The hyperimpute package is not installed.",
         "CSDI": "Tabular CSDI: conditional score-based diffusion (50 steps, quadratic schedule) with an MLP denoiser instead of the time-series transformer.",
         "MIP-VAE": "'MIP-VAE' is not a published method name; implemented as MIWAE (Mattei & Frellsen 2019), the missing-data importance-weighted VAE.",
         "GAIN": "GAIN (Yoon et al. 2018): generator/discriminator with hint rate 0.9, alpha 100.",
         "Random Forest": "missForest-style iterative random-forest imputation (3 iterations).",
         "MICE": "scikit-learn IterativeImputer, Bayesian ridge with posterior sampling, 5 imputations.",
         "GMM": "5-component full-covariance Gaussian mixture, conditional imputation, refitted 3 times."}
DIRECTIONS = {"Wasserstein": "min", "Correlation preservation |dr|": "min", "MAE": "min", "RMSE": "min",
              "NRMSE": "min", "CRPS": "min", "NLL": "min"}
M_DRAWS = 10


# --------------------------------------------------------------------------- imputers
class GMMImputer:
    def __init__(self, k=5, iters=3, seed=SEED):
        self.k, self.iters, self.seed = k, iters, seed

    def fit(self, X):
        filled = np.where(np.isnan(X), 0.0, X)
        for _ in range(self.iters):
            self.gmm = GaussianMixture(self.k, covariance_type="full", reg_covar=1e-2, max_iter=200,
                                       random_state=self.seed).fit(filled)
            filled = self.sample(X, 0)
        return self

    def sample(self, X, m):
        """m=0 -> conditional mean; m>0 -> array (m, n, d) of conditional draws."""
        rng = np.random.default_rng(self.seed + 7)
        g = self.gmm
        miss = np.isnan(X)
        point = np.where(miss, 0.0, X)
        draws = np.repeat(point[None], max(m, 1), axis=0)
        keys = np.packbits(miss, axis=1)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        for p in range(inverse.max() + 1):
            rows = np.flatnonzero(inverse == p)
            mm = miss[rows[0]]
            if not mm.any():
                continue
            o = ~mm
            xo = X[np.ix_(rows, np.flatnonzero(o))]
            logr, cmeans, cchols = [], [], []
            for c in range(self.k):
                mu, S = g.means_[c], g.covariances_[c]
                Smm = S[np.ix_(mm, mm)]
                if o.any():
                    Soo, Smo = S[np.ix_(o, o)], S[np.ix_(mm, o)]
                    L = np.linalg.cholesky(Soo)
                    diff = xo - mu[o]
                    sol = np.linalg.solve(L, diff.T)                    # L^{-1}(x-mu)
                    A = np.linalg.solve(L.T, np.linalg.solve(L, Smo.T)).T  # Smo Soo^{-1}
                    cm = mu[mm] + diff @ A.T
                    cc = Smm - A @ Smo.T
                    ll = -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * o.sum() * np.log(2 * np.pi)
                else:
                    cm = np.repeat(mu[mm][None], len(rows), axis=0)
                    cc = Smm
                    ll = np.zeros(len(rows))
                logr.append(np.log(g.weights_[c]) + ll)
                cmeans.append(cm)
                cchols.append(np.linalg.cholesky(cc + 1e-6 * np.eye(len(cc))))
            logr = np.array(logr).T
            resp = np.exp(logr - logr.max(1, keepdims=True))
            resp /= resp.sum(1, keepdims=True)
            cmeans = np.array(cmeans)                                   # (k, rows, |m|)
            cols = np.flatnonzero(mm)
            if m == 0:
                draws[0][np.ix_(rows, cols)] = np.einsum("rk,krj->rj", resp, cmeans)
                continue
            for t in range(m):
                comp = (rng.random(len(rows))[:, None] > resp.cumsum(1)).sum(1).clip(0, self.k - 1)
                z = rng.standard_normal((len(rows), len(cols)))
                value = np.empty((len(rows), len(cols)))
                for c in range(self.k):
                    sel = comp == c
                    if sel.any():
                        value[sel] = cmeans[c][sel] + z[sel] @ cchols[c].T
                draws[t][np.ix_(rows, cols)] = value
        return draws[0] if m == 0 else draws


class RFImputer:
    """missForest: iterate random-forest regressions column by column."""
    def __init__(self, iters=3, trees=40, seed=SEED):
        self.iters, self.trees, self.seed = iters, trees, seed

    def _model(self, n):
        return RandomForestRegressor(self.trees, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                                     max_samples=min(1.0, 20000 / max(n, 1)), random_state=self.seed)

    def fit(self, X):
        miss = np.isnan(X)
        self.mean = np.zeros(X.shape[1])
        self.order = [j for j in np.argsort(miss.sum(0)) if miss[:, j].any()]
        filled = np.where(miss, 0.0, X)
        self.models = {}
        for _ in range(self.iters):
            for j in self.order:
                obs = ~miss[:, j]
                others = np.delete(np.arange(X.shape[1]), j)
                model = self._model(obs.sum()).fit(filled[np.ix_(obs, others)], X[obs, j])
                filled[miss[:, j], j] = model.predict(filled[np.ix_(miss[:, j], others)])
                self.models[j] = model
        return self

    def sample(self, X, m):
        miss = np.isnan(X)
        filled = np.where(miss, 0.0, X)
        for _ in range(2):
            for j in self.order:
                rows = miss[:, j]
                if rows.any():
                    others = np.delete(np.arange(X.shape[1]), j)
                    filled[rows, j] = self.models[j].predict(filled[np.ix_(rows, others)])
        if m == 0:
            return filled
        rng = np.random.default_rng(self.seed)
        draws = np.repeat(filled[None], m, axis=0)
        for j in self.order:
            rows = miss[:, j]
            if rows.any():
                others = np.delete(np.arange(X.shape[1]), j)
                trees = rng.choice(len(self.models[j].estimators_), m, replace=m > self.trees)
                Xr = filled[np.ix_(rows, others)]
                for t, tree in enumerate(trees):
                    draws[t, rows, j] = self.models[j].estimators_[tree].predict(Xr)
        return draws


class GAINImputer:
    def __init__(self, steps=3000, batch=256, hint=0.9, alpha=100.0, seed=SEED):
        self.steps, self.batch, self.hint, self.alpha, self.seed = steps, batch, hint, alpha, seed

    def fit(self, X):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.lo, self.hi = np.nanmin(X, 0), np.nanmax(X, 0)
        span = np.where(self.hi - self.lo > 1e-9, self.hi - self.lo, 1.0)
        self.span = span
        U = (X - self.lo) / span
        mask = (~np.isnan(U)).astype(np.float32)
        U = np.nan_to_num(U).astype(np.float32)
        d = X.shape[1]
        self.G = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Sigmoid())
        self.D = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Sigmoid())
        og, od = torch.optim.Adam(self.G.parameters(), 1e-3), torch.optim.Adam(self.D.parameters(), 1e-3)
        for _ in range(self.steps):
            ix = rng.choice(len(U), min(self.batch, len(U)), replace=False)
            x, m = torch.from_numpy(U[ix]), torch.from_numpy(mask[ix])
            z = torch.rand_like(x) * 0.01
            xt = m * x + (1 - m) * z
            b = (torch.rand_like(x) < self.hint).float()
            h = b * m + 0.5 * (1 - b)
            g = self.G(torch.cat([xt, m], 1))
            xh = m * x + (1 - m) * g
            dp = self.D(torch.cat([xh.detach(), h], 1))
            ld = -torch.mean(m * torch.log(dp + 1e-8) + (1 - m) * torch.log(1 - dp + 1e-8))
            od.zero_grad(); ld.backward(); od.step()
            dp = self.D(torch.cat([xh, h], 1))
            lg = -torch.mean((1 - m) * torch.log(dp + 1e-8)) + self.alpha * torch.sum((m * x - m * g) ** 2) / torch.sum(m)
            og.zero_grad(); lg.backward(); og.step()
        return self

    @torch.no_grad()
    def sample(self, X, m):
        U = (X - self.lo) / self.span
        mask = (~np.isnan(U)).astype(np.float32)
        U = np.nan_to_num(U).astype(np.float32)
        out = []
        for t in range(max(m, 1)):
            torch.manual_seed(self.seed + t)
            x, mk = torch.from_numpy(U), torch.from_numpy(mask)
            g = self.G(torch.cat([mk * x + (1 - mk) * torch.rand_like(x) * 0.01, mk], 1))
            out.append(((mk * x + (1 - mk) * g).numpy()) * self.span + self.lo)
        out = np.array(out)
        return out.mean(0) if m == 0 else out


class MICEImputer:
    def __init__(self, m=5, seed=SEED):
        self.m, self.seed = m, seed

    def fit(self, X):
        self.models = [IterativeImputer(BayesianRidge(), sample_posterior=True, max_iter=10, tol=1e-3,
                                        n_nearest_features=min(40, X.shape[1] - 1), random_state=self.seed + i,
                                        skip_complete=True, keep_empty_features=True).fit(X) for i in range(self.m)]
        return self

    def sample(self, X, m):
        imps = np.array([model.transform(X) for model in self.models])
        if m == 0:
            return imps.mean(0)
        return imps[np.arange(m) % self.m]


class KNNDonorImputer:
    def __init__(self, k=5, max_donors=20000, seed=SEED):
        self.k, self.max_donors, self.seed = k, max_donors, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        ids = rng.choice(len(X), min(self.max_donors, len(X)), replace=False)
        self.donors = X[ids]
        self.mean = np.nanmean(X, 0)
        return self

    def sample(self, X, m):
        miss = np.isnan(X)
        draws = np.repeat(np.where(miss, 0.0, X)[None], self.k, axis=0)
        rows = np.flatnonzero(miss.any(1))
        dobs = ~np.isnan(self.donors)
        for start in range(0, len(rows), 1000):
            r = rows[start:start + 1000]
            dist = nan_euclidean_distances(X[r], self.donors)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            for j in np.flatnonzero(miss[r].any(0)):
                need = miss[r, j]
                dj = np.where(dobs[:, j][None, :], dist[need], np.inf)
                kk = min(self.k, dobs[:, j].sum())
                if kk == 0:
                    draws[:, r[need], j] = self.mean[j]
                    continue
                nn_ = np.argpartition(dj, kk - 1, axis=1)[:, :kk]
                vals = self.donors[nn_, j]                              # (rows, kk)
                vals = np.where(np.isfinite(np.take_along_axis(dj, nn_, 1)), vals, self.mean[j])
                draws[:, r[need], j] = vals[:, np.arange(self.k) % kk].T
        return draws.mean(0) if m == 0 else draws[np.arange(m) % self.k]


class HyperImputeLike:
    """Column-wise model search (Bayesian ridge / random forest / k-NN) by 3-fold CV, then
    iterative imputation with the chosen model per column (HyperImpute, Jarrett et al. 2022)."""
    def __init__(self, iters=3, seed=SEED):
        self.iters, self.seed = iters, seed

    def _pool(self):
        return {"bayesian_ridge": BayesianRidge(),
                "random_forest": RandomForestRegressor(30, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                                                       max_samples=0.5, random_state=self.seed),
                "knn": KNeighborsRegressor(10)}

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        miss = np.isnan(X)
        self.order = [j for j in np.argsort(miss.sum(0)) if miss[:, j].any()]
        filled = np.where(miss, 0.0, X)
        self.choice, self.models, self.fit_y = {}, {}, {}
        for j in self.order:
            obs = np.flatnonzero(~miss[:, j])
            sub = rng.choice(obs, min(3000, len(obs)), replace=False)
            others = np.delete(np.arange(X.shape[1]), j)
            scores = {}
            for name, model in self._pool().items():
                err = []
                for a, b in KFold(3, shuffle=True, random_state=self.seed).split(sub):
                    model.fit(filled[np.ix_(sub[a], others)], X[sub[a], j])
                    err.append(np.mean((model.predict(filled[np.ix_(sub[b], others)]) - X[sub[b], j]) ** 2))
                scores[name] = float(np.mean(err))
            self.choice[j] = min(scores, key=scores.get)
        for _ in range(self.iters):
            for j in self.order:
                obs = ~miss[:, j]
                others = np.delete(np.arange(X.shape[1]), j)
                model = self._pool()[self.choice[j]]
                ids = np.flatnonzero(obs)
                if self.choice[j] == "knn" and len(ids) > 20000:
                    ids = rng.choice(ids, 20000, replace=False)
                model.fit(filled[np.ix_(ids, others)], X[ids, j])
                self.models[j], self.fit_y[j] = model, X[ids, j]
                filled[miss[:, j], j] = model.predict(filled[np.ix_(miss[:, j], others)])
        return self

    def sample(self, X, m):
        rng = np.random.default_rng(self.seed)
        miss = np.isnan(X)
        filled = np.where(miss, 0.0, X)
        for _ in range(2):
            for j in self.order:
                rows = miss[:, j]
                if rows.any():
                    others = np.delete(np.arange(X.shape[1]), j)
                    filled[rows, j] = self.models[j].predict(filled[np.ix_(rows, others)])
        if m == 0:
            return filled
        draws = np.repeat(filled[None], m, axis=0)
        for j in self.order:
            rows = miss[:, j]
            if not rows.any():
                continue
            others = np.delete(np.arange(X.shape[1]), j)
            Xr, model = filled[np.ix_(rows, others)], self.models[j]
            if self.choice[j] == "bayesian_ridge":
                mu, sd = model.predict(Xr, return_std=True)
                draws[:, rows, j] = mu[None] + sd[None] * rng.standard_normal((m, len(mu)))
            elif self.choice[j] == "random_forest":
                for t, tree in enumerate(rng.choice(len(model.estimators_), m, replace=False)):
                    draws[t, rows, j] = model.estimators_[tree].predict(Xr)
            else:
                nb = model.kneighbors(Xr, return_distance=False)
                draws[:, rows, j] = self.fit_y[j][nb][:, np.arange(m) % nb.shape[1]].T
        return draws


def _time_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], 1)


class CSDIImputer:
    def __init__(self, T=50, steps=4000, batch=256, seed=SEED):
        self.T, self.steps, self.batch, self.seed = T, steps, batch, seed
        betas = np.linspace(1e-4 ** 0.5, 0.5 ** 0.5, T) ** 2            # CSDI quadratic schedule
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alpha = 1 - self.betas
        self.abar = torch.cumprod(self.alpha, 0)

    def fit(self, X):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        obs = (~np.isnan(X)).astype(np.float32)
        x0 = np.nan_to_num(X).astype(np.float32)
        self.net = mlp(3 * d + 32, d, hidden=(256, 256, 256), act=nn.SiLU)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        for _ in range(self.steps):
            ix = rng.choice(len(X), min(self.batch, len(X)), replace=False)
            x, o = torch.from_numpy(x0[ix]), torch.from_numpy(obs[ix])
            ratio = torch.rand(len(ix), 1)
            target = o * (torch.rand_like(o) < ratio).float()
            cond = o - target
            t = torch.randint(0, self.T, (len(ix),))
            eps = torch.randn_like(x)
            ab = self.abar[t][:, None]
            xt = ab.sqrt() * x + (1 - ab).sqrt() * eps
            pred = self.net(torch.cat([xt * target, x * cond, cond, _time_embedding(t)], 1))
            loss = (((eps - pred) * target) ** 2).sum() / target.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def sample(self, X, m):
        obs = torch.from_numpy((~np.isnan(X)).astype(np.float32))
        x0 = torch.from_numpy(np.nan_to_num(X).astype(np.float32))
        target = 1 - obs
        out = []
        for s in range(m if m > 0 else 5):          # point imputation = mean of 5 samples
            torch.manual_seed(self.seed + 100 + s)
            x = torch.randn_like(x0)
            for t in reversed(range(self.T)):
                tt = torch.full((len(x0),), t, dtype=torch.long)
                eps = self.net(torch.cat([x * target, x0 * obs, obs, _time_embedding(tt)], 1))
                a, ab, b = self.alpha[t], self.abar[t], self.betas[t]
                x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                if t > 0:
                    sigma = ((1 - self.abar[t - 1]) / (1 - ab) * b).sqrt()
                    x = x + sigma * torch.randn_like(x)
            out.append((obs * x0 + target * x).numpy())
        out = np.array(out)
        return out.mean(0) if m == 0 else out


class MIWAEImputer:
    def __init__(self, latent=10, K=20, steps=4000, batch=128, seed=SEED):
        self.latent, self.K, self.steps, self.batch, self.seed = latent, K, steps, batch, seed

    def _parts(self, x, mask, K):
        h = self.enc(x)
        mu, logs = h[:, :self.latent], h[:, self.latent:].clamp(-6, 3)
        q = torch.distributions.Normal(mu, logs.exp())
        z = q.rsample((K,))                                               # (K, n, l)
        out = self.dec(z)
        xm, xs = out[..., :x.shape[1]], F.softplus(out[..., x.shape[1]:]) + 1e-3
        px = torch.distributions.Normal(xm, xs)
        logpx = (px.log_prob(x[None]) * mask[None]).sum(-1)
        logpz = torch.distributions.Normal(0, 1).log_prob(z).sum(-1)
        logq = q.log_prob(z).sum(-1)
        return logpx + logpz - logq, xm, xs

    def fit(self, X):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.enc = mlp(d, 2 * self.latent, hidden=(128, 128), act=nn.Tanh)
        self.dec = mlp(self.latent, 2 * d, hidden=(128, 128), act=nn.Tanh)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3)
        mask = (~np.isnan(X)).astype(np.float32)
        x0 = np.nan_to_num(X).astype(np.float32)
        for _ in range(self.steps):
            ix = rng.choice(len(X), min(self.batch, len(X)), replace=False)
            lw, _, _ = self._parts(torch.from_numpy(x0[ix]), torch.from_numpy(mask[ix]), self.K)
            loss = -(torch.logsumexp(lw, 0) - np.log(self.K)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def sample(self, X, m, L=50):
        torch.manual_seed(self.seed + 1)
        mask = torch.from_numpy((~np.isnan(X)).astype(np.float32))
        x0 = torch.from_numpy(np.nan_to_num(X).astype(np.float32))
        out_mean, out_draws = [], []
        for s in range(0, len(X), 2048):
            x, mk = x0[s:s + 2048], mask[s:s + 2048]
            lw, xm, xs = self._parts(x, mk, L)
            w = torch.softmax(lw, 0)                                      # (L, n)
            if m == 0:
                out_mean.append((mk * x + (1 - mk) * (w[..., None] * xm).sum(0)).numpy())
            else:
                idx = torch.multinomial(w.T, m, replacement=True)         # (n, m)
                ar = torch.arange(len(x))[:, None]
                draw = xm.permute(1, 0, 2)[ar, idx] + xs.permute(1, 0, 2)[ar, idx] * torch.randn(len(x), m, x.shape[1])
                out_draws.append((mk[:, None] * x[:, None] + (1 - mk[:, None]) * draw).permute(1, 0, 2).numpy())
        return np.concatenate(out_mean) if m == 0 else np.concatenate(out_draws, axis=1)


def make(name):
    return {"GMM": GMMImputer, "Random Forest": RFImputer, "GAIN": GAINImputer, "MICE": MICEImputer,
            "KNN": KNNDonorImputer, "HyperImpute": HyperImputeLike, "CSDI": CSDIImputer,
            "MIP-VAE": MIWAEImputer}[name]()


# --------------------------------------------------------------------------- step
def detect(train, schema):
    """Missing-cell share over original predictors (a one-hot block counts once)."""
    X = train.X
    per = {c: float(X[c].isna().mean()) for c in schema.numeric}
    per.update({src: float(X[cols[0]].isna().mean()) for src, cols in schema.onehot.items()})
    overall = float(np.mean(list(per.values()))) if per else 0.0
    detected = overall >= 0.01 or (max(per.values()) if per else 0) >= 0.05
    return {"statistic": "share of missing predictor cells (training)", "before": overall,
            "per_feature": per, "rule": "present if overall share >= 1% or any predictor >= 5%",
            "detected": bool(detected)}


def _mask_cells(X, schema_cols, onehot_idx, rate, rng):
    hide = (rng.random(X.shape) < rate) & ~np.isnan(X)
    for idx in onehot_idx:                                   # hide whole one-hot blocks
        row_hide = (rng.random(len(X)) < rate) & ~np.isnan(X[:, idx[0]])
        hide[:, idx] = row_hide[:, None]
    return hide


def benchmark(train, schema, log, n_bench=8000, rate=0.2):
    cols = schema.columns()
    y = train.meta.target.to_numpy()
    ids = stratified_sample(None, y, n_bench, seed=SEED)
    X = train.X.iloc[ids][cols].to_numpy(float)
    st = Standardizer(X)
    Z = st(X)
    rng = np.random.default_rng(SEED)
    onehot_idx = [[cols.index(c) for c in block] for block in schema.onehot.values()]
    hide = _mask_cells(Z, cols, onehot_idx, rate, rng)
    Zm = np.where(hide, np.nan, Z)
    numeric = [cols.index(c) for c in schema.numeric]
    spans = np.nanmax(Z, 0) - np.nanmin(Z, 0)
    truth_corr = pd.DataFrame(Z[:, numeric]).corr().to_numpy()
    rows, details = {}, {}
    for name in FAMILY:
        t0 = __import__("time").time()
        try:
            imp = make(name).fit(Zm)
            draws = imp.sample(Zm, M_DRAWS)
        except Exception as error:                     # keep the run going; report the failure
            log(f"  missing/{name} FAILED: {error!r}")
            rows[name] = {k: np.nan for k in DIRECTIONS}
            continue
        point = draws.mean(0)
        mae, rmse, nrmse, wd, crps, nll = [], [], [], [], [], []
        for j in numeric:
            cells = hide[:, j]
            if cells.sum() < 20:
                continue
            t, D = Z[cells, j], draws[:, cells, j]
            p = D.mean(0)
            err = p - t
            mae.append(np.abs(err).mean()); rmse.append(np.sqrt((err ** 2).mean()))
            nrmse.append(rmse[-1] / spans[j] if spans[j] > 0 else np.nan)
            wd.append(wasserstein_1d(p, t))
            term1 = np.abs(D - t[None]).mean(0)
            term2 = np.abs(D[:, None, :] - D[None, :, :]).mean((0, 1))
            crps.append((term1 - 0.5 * term2).mean())
            sd = np.maximum(D.std(0), 0.1)
            nll.append((0.5 * np.log(2 * np.pi * sd ** 2) + (t - p) ** 2 / (2 * sd ** 2)).mean())
        completed = np.where(np.isnan(Z), np.nan, point)          # evaluate only originally observed cells
        corr = pd.DataFrame(completed[:, numeric]).corr().to_numpy()
        rows[name] = {"Wasserstein": np.nanmean(wd), "Correlation preservation |dr|": np.nanmean(np.abs(corr - truth_corr)),
                      "MAE": np.nanmean(mae), "RMSE": np.nanmean(rmse), "NRMSE": np.nanmean(nrmse),
                      "CRPS": np.nanmean(crps), "NLL": np.nanmean(nll)}
        details[name] = {"seconds": round(__import__("time").time() - t0, 1)}
        log(f"  missing/{name}: RMSE {rows[name]['RMSE']:.4f} CRPS {rows[name]['CRPS']:.4f} ({details[name]['seconds']}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"{len(ids):,} training rows (class-stratified); {rate:.0%} of observed cells hidden completely at "
                f"random; {int(hide[:, numeric].sum()):,} hidden numeric cells scored in standardised units; "
                f"{M_DRAWS} draws per method for CRPS/NLL (NLL uses a Gaussian with SD floored at 0.1).")
    return table, best, protocol, details


def apply(frames, schema, method, log, fit_rows=20000):
    """Fit the selected imputer on (a sample of) training rows, impute every split."""
    cols = schema.columns()
    train = frames["train"]
    ids = stratified_sample(None, train.meta.target.to_numpy(), fit_rows, seed=SEED + 1)
    Xfit = train.X.iloc[ids][cols].to_numpy(float)
    st = Standardizer(Xfit)
    imp = make(method).fit(st(Xfit))
    observed = train.X[schema.continuous]
    clip = {c: (float(observed[c].min()), float(observed[c].max())) for c in schema.continuous}
    out = {}
    for split, frame in frames.items():
        X = frame.X[cols].to_numpy(float)
        Z = st(X)
        filled = Z.copy()
        rows = np.flatnonzero(np.isnan(Z).any(axis=1))           # only rows with a missing cell need imputing
        for s in range(0, len(rows), 50000):
            r = rows[s:s + 50000]
            filled[r] = imp.sample(Z[r], 0)
        filled = np.where(np.isnan(Z), filled, Z)
        new = pd.DataFrame(st.inverse(filled), columns=cols, index=frame.X.index)
        new = postprocess(new, schema, clip)
        new = new.where(frame.X[cols].isna(), frame.X[cols])      # observed cells unchanged
        assert np.isfinite(new.to_numpy(float)).all(), f"{split}: imputation left missing values"
        out[split] = new
        log(f"  imputed {split}: {int(frame.X[cols].isna().to_numpy().sum()):,} cells")
    return out, {"fit_rows": len(ids)}
