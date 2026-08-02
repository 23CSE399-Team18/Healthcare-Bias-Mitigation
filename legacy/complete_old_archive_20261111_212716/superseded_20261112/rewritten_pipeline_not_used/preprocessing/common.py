"""Shared data handling, neural utilities and metrics for the 8-bias preprocessing stage.

Conventions
-----------
* A dataset folder is D:/complete/<name>/ with raw/, preprocessed/ and plots/.
* Predictors are every raw column that is not `target`, `meta_*` or `sens_*`.
* Text predictors are one-hot encoded with training-split levels; a missing category keeps
  the whole one-hot block missing (NaN) until the missing-data step imputes it.
* Every fitted statistic, model or selection uses the training split (benchmarks use a
  training sub-sample and the validation split). The test split is never used for selection.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]          # D:/complete
SEED = 2026
SPLITS = ("train", "validation", "test")

# Census regions by state FIPS (BRFSS _STATE). Territories kept as their own domain.
_REGION = {
    "Northeast": [9, 23, 25, 33, 44, 50, 34, 36, 42],
    "Midwest": [17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46],
    "South": [10, 11, 12, 13, 24, 37, 45, 51, 54, 1, 21, 28, 47, 5, 22, 40, 48],
    "West": [4, 8, 16, 30, 32, 35, 49, 56, 2, 6, 15, 41, 53],
}
STATE_REGION = {fips: region for region, codes in _REGION.items() for fips in codes}

DATASETS = {
    "heart": {"title": "Heart disease", "source": "CDC BRFSS 2022", "group": None,
              "domain": "region_from_state", "target_meaning": "self-reported heart attack / coronary heart disease"},
    "diabetes": {"title": "Diabetes", "source": "CDC NHANES Aug 2021-Aug 2023", "group": None,
                 "domain": None, "target_meaning": "self-reported diagnosed diabetes"},
    "hypertension": {"title": "Hypertension", "source": "CDC NHIS 2023", "group": None,
                     "domain": "meta_domain_region", "target_meaning": "self-reported diagnosed hypertension"},
    "respiratory": {"title": "Respiratory ICU", "source": "PhysioNet eICU-CRD v2.0",
                    "group": "meta_split_group_uniquepid", "domain": "meta_domain_region",
                    "target_meaning": "in-hospital death after the 24 h ICU landmark"},
}
PROTECTED = ("sens_sex", "sens_race_ethnicity")       # fairness mitigation targets
AUDITED = ("sens_sex", "sens_age_group", "sens_race_ethnicity")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def now():
    return time.time()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, message):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time()-self.t0:7.1f}s] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


# --------------------------------------------------------------------------- data
@dataclass
class Frame:
    """Features (numeric, one-hot already expanded) plus aligned metadata for one split."""
    X: pd.DataFrame
    meta: pd.DataFrame          # target, sample_weight, meta_*, sens_*

    def copy(self):
        return Frame(self.X.copy(), self.meta.copy())

    def __len__(self):
        return len(self.X)


@dataclass
class Schema:
    name: str
    continuous: list = field(default_factory=list)   # numeric, >10 distinct training values
    ordinal: list = field(default_factory=list)      # numeric, 3..10 distinct values
    binary: list = field(default_factory=list)       # numeric 0/1
    onehot: dict = field(default_factory=dict)       # source categorical -> list of indicator columns
    levels: dict = field(default_factory=dict)       # ordinal column -> sorted valid values (current units)
    dropped: dict = field(default_factory=dict)      # column -> reason
    domain: str | None = None
    snap_ordinal: bool = True                         # False once domain alignment shifts levels per region

    @property
    def onehot_cols(self):
        return [c for cols in self.onehot.values() for c in cols]

    @property
    def numeric(self):
        return self.continuous + self.ordinal + self.binary

    @property
    def scalable(self):
        return self.continuous + self.ordinal

    def columns(self):
        return self.numeric + self.onehot_cols

    def kind(self, column):
        for name, cols in (("continuous", self.continuous), ("ordinal", self.ordinal), ("binary", self.binary)):
            if column in cols:
                return name
        return "onehot"

    def drop(self, columns, reason):
        for c in columns:
            for group in (self.continuous, self.ordinal, self.binary):
                if c in group:
                    group.remove(c)
            self.levels.pop(c, None)
            self.dropped[c] = reason


def load_raw(name):
    path = ROOT / name / "raw" / f"{name}_raw.csv"
    head = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "string" for c in head if c in ("meta_record_id", "meta_split_group_uniquepid")}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    if name == "heart":
        df["meta_domain_region"] = df.meta_domain_state.astype(int).map(STATE_REGION).fillna("Territories")
    return df, path


def predictor_columns(df):
    return [c for c in df.columns if c != "target" and not c.startswith(("meta_", "sens_"))]


def build_schema(name, train_raw):
    preds = predictor_columns(train_raw)
    schema = Schema(name=name, domain=DATASETS[name]["domain"] and "meta_domain_region")
    for c in preds:
        if pd.api.types.is_numeric_dtype(train_raw[c]):
            values = train_raw[c].dropna().unique()
            if set(np.unique(values)) <= {0, 1}:
                schema.binary.append(c)
            elif len(values) <= 10:
                schema.ordinal.append(c)
                schema.levels[c] = np.sort(values.astype(float))
            else:
                schema.continuous.append(c)
        else:
            levels = sorted(train_raw[c].dropna().astype(str).unique())
            schema.onehot[c] = [f"{c}={v}" for v in levels]
    return schema


def encode(raw, schema):
    """Numeric predictors as float; categorical -> one-hot with training levels.
    Missing category => NaN in the whole block; unseen level => all-zero block."""
    out = {c: pd.to_numeric(raw[c], errors="raise").astype(float).to_numpy() for c in schema.numeric}
    for source, cols in schema.onehot.items():
        value = raw[source].astype("string")
        missing = value.isna().to_numpy()
        for col in cols:
            level = col.split("=", 1)[1]
            indicator = (value == level).fillna(False).to_numpy().astype(float)
            indicator[missing] = np.nan
            out[col] = indicator
    return pd.DataFrame(out, index=raw.index)[schema.columns()]


def meta_frame(raw):
    meta = raw[[c for c in raw.columns if c.startswith(("meta_", "sens_")) or c == "target"]].copy()
    meta["sample_weight"] = 1.0
    meta["meta_synthetic"] = 0
    return meta


def postprocess(X, schema, clip=None):
    """Make values valid for their type: one-hot argmax, binary {0,1}, ordinal snapped to the
    nearest valid level, continuous clipped to the training range (clip = {col: (lo, hi)})."""
    X = X.copy()
    for cols in schema.onehot.values():
        block = X[cols].to_numpy(float)
        ok = np.isfinite(block).all(axis=1)
        hot = np.zeros_like(block)
        hot[np.arange(len(block)), np.nanargmax(np.where(np.isfinite(block), block, -np.inf), axis=1)] = 1
        # an all-zero (unseen level) block stays all-zero
        zero = ok & (np.abs(block).sum(axis=1) == 0)
        hot[zero] = 0
        block = np.where(ok[:, None], hot, block)
        X[cols] = block
    for c in schema.binary:
        v = X[c].to_numpy(float)
        X[c] = np.where(np.isfinite(v), (v >= 0.5).astype(float), np.nan)
    for c in (schema.ordinal if schema.snap_ordinal else []):
        levels = schema.levels[c]
        v = X[c].to_numpy(float)
        idx = np.abs(v[:, None] - levels[None, :]).argmin(axis=1)
        X[c] = np.where(np.isfinite(v), levels[idx], np.nan)
    if clip:
        for c, (lo, hi) in clip.items():
            if c in X:
                X[c] = X[c].clip(lo, hi)
    return X


def stratified_sample(index_or_n, y, n, seed=SEED):
    """Indices of a class-stratified sample of size <= n (positions, not labels)."""
    total = len(y)
    if total <= n:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    chosen = []
    for cls in np.unique(y):
        ids = np.flatnonzero(y == cls)
        k = max(1, int(round(n * len(ids) / total)))
        chosen.append(rng.choice(ids, min(k, len(ids)), replace=False))
    return np.sort(np.concatenate(chosen))


class Standardizer:
    def __init__(self, X):
        X = np.asarray(X, float)
        self.mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)

    def __call__(self, X):
        return (np.asarray(X, float) - self.mean) / self.sd

    def inverse(self, Z):
        return np.asarray(Z, float) * self.sd + self.mean


# --------------------------------------------------------------------------- neural nets
DEVICE = torch.device("cpu")


def mlp(d_in, d_out, hidden=(64, 32), act=nn.ReLU, dropout=0.0):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def focal_cb_loss(logits, y, class_weights, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    w = torch.where(y > 0.5, class_weights[1], class_weights[0])
    return (w * (1 - pt) ** gamma * -torch.log(pt)).mean()


def gce_loss(logits, y, q=0.7):
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1 - p).clamp(1e-6, 1.0)
    return ((1 - pt ** q) / q).mean()


def train_classifier(X, y, weights=None, loss="bce", class_weights=None, steps=1500, batch=256, lr=1e-3,
                     hidden=(64, 32), seed=SEED, act=nn.ReLU, record_gradients=False, model=None):
    """Small MLP classifier. Early stopping on a 10% internal split of the given training data."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    w = np.ones(len(y), np.float32) if weights is None else np.asarray(weights, np.float32)
    hold = rng.random(len(y)) < 0.1
    if hold.sum() < 20 or (~hold).sum() < 20:
        hold[:] = False
    tr = np.flatnonzero(~hold)
    net = model if model is not None else mlp(X.shape[1], 1, hidden, act)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xt, yt, wt = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(w)
    cw = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    best, best_state, stale, grads = np.inf, None, 0, []
    check = max(100, steps // 15)
    for step in range(steps):
        ix = torch.from_numpy(rng.choice(tr, min(batch, len(tr)), replace=len(tr) < batch))
        net.train()
        logits = net(Xt[ix]).squeeze(-1)
        if loss == "bce":
            value = (F.binary_cross_entropy_with_logits(logits, yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
        elif loss == "focal_cb":
            value = focal_cb_loss(logits, yt[ix], cw)
        elif loss == "gce":
            value = gce_loss(logits, yt[ix])
        else:
            raise ValueError(loss)
        opt.zero_grad()
        value.backward()
        if record_gradients:
            grads.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None))))
        opt.step()
        if hold.any() and (step + 1) % check == 0:
            net.eval()
            with torch.no_grad():
                h = torch.from_numpy(np.flatnonzero(hold))
                v = float(F.binary_cross_entropy_with_logits(net(Xt[h]).squeeze(-1), yt[h]))
            if v < best - 1e-5:
                best, stale = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                stale += 1
                if stale >= 4:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net.gradient_norms = grads
    return net


@torch.no_grad()
def predict_proba(net, X, batch=8192):
    net.eval()
    X = np.asarray(X, np.float32)
    out = [torch.sigmoid(net(torch.from_numpy(X[i:i + batch])).squeeze(-1)).numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------- metrics & ranking
def mean_rank(table, directions):
    """Average rank across metrics (1 = best). directions: {metric: 'max'|'min'}.
    Ties share the average rank. A method missing a metric gets the worst rank for it."""
    ranks = {}
    for metric, direction in directions.items():
        values = pd.to_numeric(table[metric], errors="coerce")
        filled = values.fillna(-np.inf if direction == "max" else np.inf)
        ranks[metric] = filled.rank(ascending=(direction == "min"), method="average")
    return pd.DataFrame(ranks).mean(axis=1)


def select(table, directions, eligible=None):
    table = table.copy()
    table["mean_rank"] = mean_rank(table, directions)
    order = table.sort_values("mean_rank", kind="stable")
    if eligible is not None:
        order = order[order.index.isin(eligible)]
    return table, (order.index[0] if len(order) else None)


def group_rates(y_true, y_pred, groups, min_rows=30):
    rows = {}
    for g in pd.unique(groups):
        if pd.isna(g) or str(g) in ("Unknown", "<NA>", "nan"):
            continue
        m = groups == g
        if m.sum() < min_rows:
            continue
        yt, yp = y_true[m], y_pred[m]
        rows[str(g)] = {"n": int(m.sum()), "selection_rate": float(yp.mean()),
                        "tpr": float(yp[yt == 1].mean()) if (yt == 1).sum() >= 10 else np.nan,
                        "fpr": float(yp[yt == 0].mean()) if (yt == 0).sum() >= 10 else np.nan,
                        "base_rate": float(yt.mean())}
    return rows


def fairness_metrics(y_true, y_pred, groups):
    """SPD (max-min selection rate), DI (min/max selection rate), equal opportunity difference
    (max TPR gap) and equalized-odds difference (max of TPR and FPR gaps) over supported groups."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups, dtype=object)
    rates = group_rates(y_true, y_pred, groups)
    if len(rates) < 2:
        return {"spd": np.nan, "di": np.nan, "eod": np.nan, "eodds": np.nan, "groups": rates}
    sel = np.array([r["selection_rate"] for r in rates.values()])
    tpr = np.array([r["tpr"] for r in rates.values()])
    fpr = np.array([r["fpr"] for r in rates.values()])
    eod = np.nanmax(tpr) - np.nanmin(tpr) if np.isfinite(tpr).sum() >= 2 else np.nan
    fgap = np.nanmax(fpr) - np.nanmin(fpr) if np.isfinite(fpr).sum() >= 2 else np.nan
    return {"spd": float(sel.max() - sel.min()), "di": float(sel.min() / sel.max()) if sel.max() > 0 else np.nan,
            "eod": float(eod), "eodds": float(np.nanmax([eod, fgap])), "groups": rates}


def wasserstein_1d(a, b):
    from scipy.stats import wasserstein_distance
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


def rbf_mmd(a, b, n=800, seed=SEED):
    """Unbiased squared MMD with an RBF kernel (median-distance bandwidth)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = np.asarray(b, float)[rng.choice(len(b), min(n, len(b)), replace=False)]
    z = np.vstack([a, b])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    bw = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    k = np.exp(-d2 / bw)
    na, nb = len(a), len(b)
    kaa, kbb, kab = k[:na, :na], k[na:, na:], k[:na, na:]
    return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
                 - 2 * kab.mean())
