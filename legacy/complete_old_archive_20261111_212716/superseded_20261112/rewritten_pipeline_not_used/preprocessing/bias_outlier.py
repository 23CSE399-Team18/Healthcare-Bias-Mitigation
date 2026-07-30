"""Outlier bias: Isolation Forest, LOF, One-Class SVM (traditional); ECOD, Deep SVDD,
Explainable Boosting Machine (modern).

Detection uses the skew-adjusted boxplot (Hubert & Vandervieren 2008) on continuous
predictors, so skewed laboratory values are not all called outliers.
Benchmark: detectors are fitted on a training sub-sample; a validation sample receives 5%
injected outliers (1-3 continuous values moved 3-6 IQR beyond the quartiles) whose labels
are known, so AUROC, AUPRC, Precision@K, F1 and Brier score can be computed. Outlier-score
stability (two disjoint training halves) and neighbourhood consistency need no labels.
Applying the selected detector removes the highest-scoring training rows only; validation
and test rows are never removed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from statsmodels.stats.stattools import medcouple

from common import SEED, Standardizer, select, stratified_sample

FAMILY = {"Isolation Forest": "Traditional", "LOF": "Traditional", "One-Class SVM": "Traditional",
          "ECOD": "Modern", "Deep SVDD": "Modern", "EBM": "Modern"}
NOTES = {"EBM": "An EBM is a classifier, so it is used as a classification-based density-ratio detector: it learns "
                "to separate real training rows from reference rows (half uniform over the expanded range, half "
                "independent marginals); anomaly score = -logit P(real).",
         "One-Class SVM": "RBF kernel, nu = 0.05, fitted on at most 5,000 training rows (quadratic cost).",
         "Deep SVDD": "pyod DeepSVDD (one-class hypersphere), 30 epochs.",
         "ECOD": "pyod ECOD (empirical cumulative distribution tails).",
         "LOF": "Local outlier factor in novelty mode, 35 neighbours.",
         "Isolation Forest": "200 trees."}
DIRECTIONS = {"Outlier score stability": "max", "Neighborhood consistency": "max", "AUROC": "max", "AUPRC": "max",
              "Precision@K": "max", "F1-score": "max", "Brier score": "min"}


def adjusted_fences(values, seed=SEED):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None
    sample = np.random.default_rng(seed).choice(v, min(2000, len(v)), replace=False)
    mc = float(medcouple(sample))
    if mc >= 0:
        return q1 - 1.5 * np.exp(-4 * mc) * iqr, q3 + 1.5 * np.exp(3 * mc) * iqr
    return q1 - 1.5 * np.exp(-3 * mc) * iqr, q3 + 1.5 * np.exp(4 * mc) * iqr


def fence_columns(train, schema):
    fences = {}
    for c in schema.continuous:
        if train.X[c].nunique() >= 20:
            f = adjusted_fences(train.X[c].to_numpy())
            if f is not None:
                fences[c] = f
    return fences


def flagged_cells(X, fences):
    return pd.DataFrame({c: ((X[c] < lo) | (X[c] > hi)).to_numpy() for c, (lo, hi) in fences.items()}, index=X.index)


def detect(train, schema):
    fences = fence_columns(train, schema)
    flags = flagged_cells(train.X, fences)
    cell = float(flags.to_numpy().mean()) if flags.size else 0.0
    rows = float(flags.any(axis=1).mean()) if flags.size else 0.0
    return {"statistic": "share of continuous cells beyond skew-adjusted boxplot fences (training)",
            "before": cell, "rows_with_any_flag": rows, "fences": {c: list(f) for c, f in fences.items()},
            "per_feature": {c: float(flags[c].mean()) for c in flags}, "rule": "present if share > 1%",
            "detected": bool(cell > 0.01), "contamination": float(np.clip(cell, 0.005, 0.05))}


# --------------------------------------------------------------------------- detectors
class EBMDetector:
    def __init__(self, seed=SEED):
        self.seed = seed

    def fit(self, X):
        from interpret.glassbox import ExplainableBoostingClassifier
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        lo, hi = X.min(0), X.max(0)
        span = hi - lo
        uniform = rng.uniform(lo - 0.1 * span, hi + 0.1 * span, size=(n // 2, d))
        marginal = np.column_stack([rng.choice(X[:, j], n - n // 2) for j in range(d)])
        ref = np.vstack([uniform, marginal])
        self.model = ExplainableBoostingClassifier(interactions=3, outer_bags=2, max_rounds=2000, n_jobs=-1,
                                                   random_state=self.seed)
        self.model.fit(np.vstack([X, ref]), np.r_[np.ones(n), np.zeros(len(ref))])
        return self

    def score(self, X):
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return -np.log(p / (1 - p))


class Sk:
    def __init__(self, name, seed=SEED):
        self.name, self.seed = name, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        if self.name == "Isolation Forest":
            self.m = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(X)
        elif self.name == "LOF":
            self.m = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(X)
        elif self.name == "One-Class SVM":
            ids = rng.choice(len(X), min(5000, len(X)), replace=False)
            self.m = OneClassSVM(nu=0.05, gamma="scale").fit(X[ids])
        elif self.name == "ECOD":
            from pyod.models.ecod import ECOD
            self.m = ECOD().fit(X)
        elif self.name == "Deep SVDD":
            import torch
            from pyod.models.deep_svdd import DeepSVDD
            torch.manual_seed(self.seed)
            self.m = DeepSVDD(n_features=X.shape[1], epochs=30, batch_size=256, hidden_neurons=[64, 32],
                              verbose=0, random_state=self.seed, preprocessing=False).fit(X)
        return self

    def score(self, X):
        out = []
        for s in range(0, len(X), 50000):
            part = X[s:s + 50000]
            out.append(self.m.decision_function(part) if self.name in ("ECOD", "Deep SVDD")
                       else -self.m.score_samples(part))
        return np.concatenate(out)


def make(name):
    return EBMDetector() if name == "EBM" else Sk(name)


# --------------------------------------------------------------------------- benchmark / apply
def _matrix(frame, schema, st=None):
    X = frame.X[schema.numeric].to_numpy(float)
    st = st or Standardizer(X)
    return st(X), st


def benchmark(frames, schema, detection, log, n_train=10000, n_eval=4000):
    rng = np.random.default_rng(SEED)
    train, val = frames["train"], frames["validation"]
    ids = stratified_sample(None, train.meta.target.to_numpy(), n_train, seed=SEED)
    Xtr, st = _matrix(train, schema)
    Xtr = Xtr[ids]
    vids = stratified_sample(None, val.meta.target.to_numpy(), n_eval, seed=SEED + 1)
    Xev = st(val.X[schema.numeric].to_numpy(float))[vids]
    cont = [schema.numeric.index(c) for c in detection["fences"]] or list(range(len(schema.numeric)))
    q1, q3 = np.percentile(Xtr, 25, axis=0), np.percentile(Xtr, 75, axis=0)
    iqr = np.where(q3 - q1 > 0, q3 - q1, Xtr.std(0) + 1e-6)
    k = max(10, int(0.05 * len(Xev)))
    inj = Xev[rng.choice(len(Xev), k)].copy()
    for row in inj:
        for j in rng.choice(cont, min(len(cont), rng.integers(1, 4)), replace=False):
            if rng.random() < 0.5:
                row[j] = q3[j] + rng.uniform(3, 6) * iqr[j]
            else:
                row[j] = q1[j] - rng.uniform(3, 6) * iqr[j]
    Xeval = np.vstack([Xev, inj])
    label = np.r_[np.zeros(len(Xev)), np.ones(k)]
    contamination = detection["contamination"]
    nbrs = NearestNeighbors(n_neighbors=11).fit(Xev).kneighbors(Xev, return_distance=False)[:, 1:]
    half = rng.permutation(len(Xtr))
    a, b = half[: len(half) // 2], half[len(half) // 2:]
    rows = {}
    for name in FAMILY:
        t0 = time.time()
        try:
            det = make(name).fit(Xtr)
            s_tr, s_ev = det.score(Xtr), det.score(Xeval)
            sa, sb = make(name).fit(Xtr[a]).score(Xev), make(name).fit(Xtr[b]).score(Xev)
        except Exception as error:
            log(f"  outlier/{name} FAILED: {error!r}")
            rows[name] = {m: np.nan for m in DIRECTIONS}
            continue
        thr = np.quantile(s_tr, 1 - contamination)
        topk = np.argsort(-s_ev)[:k]
        prob = np.searchsorted(np.sort(s_tr), s_ev, side="right") / len(s_tr)
        s_norm = s_ev[: len(Xev)]
        rows[name] = {"Outlier score stability": float(spearmanr(sa, sb).statistic),
                      "Neighborhood consistency": float(spearmanr(s_norm, s_norm[nbrs].mean(1)).statistic),
                      "AUROC": roc_auc_score(label, s_ev), "AUPRC": average_precision_score(label, s_ev),
                      "Precision@K": float(label[topk].mean()), "F1-score": f1_score(label, s_ev > thr),
                      "Brier score": float(np.mean((prob - label) ** 2))}
        log(f"  outlier/{name}: AUROC {rows[name]['AUROC']:.3f} stability {rows[name]['Outlier score stability']:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"Detectors fitted on {len(Xtr):,} training rows (numeric predictors, standardised). Evaluation: "
                f"{len(Xev):,} validation rows + {k} injected outliers (1-3 continuous values set 3-6 IQR beyond Q1/Q3). "
                f"F1 threshold = training-score quantile at contamination {contamination:.3f}; Brier uses the training-score "
                "percentile as the outlier probability. Stability = Spearman correlation of scores from detectors fitted on "
                "two disjoint training halves; neighbourhood consistency = Spearman correlation of each row's score with "
                "the mean score of its 10 nearest neighbours.")
    return table, best, protocol


def apply(frames, schema, method, detection, log, fit_rows=20000):
    train = frames["train"]
    X, st = _matrix(train, schema)
    ids = stratified_sample(None, train.meta.target.to_numpy(), fit_rows, seed=SEED + 2)
    det = make(method).fit(X[ids])
    scores = det.score(X)
    thr = np.quantile(scores, 1 - detection["contamination"])
    keep = scores <= thr
    log(f"  {method}: removing {int((~keep).sum()):,} of {len(keep):,} training rows (score > {thr:.3f})")
    new = {s: f.copy() for s, f in frames.items()}
    new["train"].X = train.X.loc[keep]
    new["train"].meta = train.meta.loc[keep]
    return new, {"removed_training_rows": int((~keep).sum()), "threshold": float(thr), "fit_rows": len(ids),
                 "scores": scores}
