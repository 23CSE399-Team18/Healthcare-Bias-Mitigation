"""Demographic bias: Optimal-transport repair, FairSMOTE, Reweighting (traditional);
Adversarial debiasing, counterfactual (DiCE-style) augmentation, fairness-aware supervised
contrastive learning (modern).

Protected attributes: sex and race/ethnicity (age is a clinical risk factor and is only
audited). Detection trains the reference MLP on the current training data and measures, on the
validation split, statistical parity difference (SPD), disparate impact (DI), equal opportunity
difference (EOD, TPR gap) and equalized-odds difference (max of TPR and FPR gaps) across
supported groups (>= 30 rows; >= 10 positives/negatives for TPR/FPR). The attribute with the
largest equalized-odds difference is mitigated.
Counterfactual metrics (all methods, DiCE-style random search on the method's own classifier):
for validation rows predicted positive, up to 200 candidates change 1-3 actionable predictors
(not age or sex) to values drawn from training marginals. Validity = share of rows with a
candidate predicted negative; proximity = standardised L1 distance of the closest valid
candidate; sparsity = number of predictors it changes.
Utility guard: a method whose validation AUROC is more than 0.05 below the reference cannot be
selected.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F

from common import (PROTECTED, SEED, fairness_metrics, mlp, postprocess, predict_proba, select,
                    stratified_sample, train_classifier)

FAMILY = {"Optimal Transport repair": "Traditional", "FairSMOTE": "Traditional", "Reweighting": "Traditional",
          "Adversarial debiasing": "Modern", "Counterfactual augmentation (DiCE-style)": "Modern",
          "Fair contrastive learning": "Modern"}
DATA_LEVEL = ("Optimal Transport repair", "FairSMOTE", "Reweighting", "Counterfactual augmentation (DiCE-style)")
NOTES = {"Optimal Transport repair": "Total repair (Feldman et al. 2015; Gordaliza et al. 2019): each group's 1-D distribution of every "
                                     "continuous/ordinal predictor is mapped to the Wasserstein barycenter of the groups. Applied to all splits.",
         "FairSMOTE": "Chakraborty et al. 2021: every (class x group) subgroup is grown towards the largest (synthetic rows capped at 50% of "
                      "the training rows) by differential-evolution crossover of subgroup neighbours; situation testing removes "
                      "training rows whose prediction flips when sex is flipped (only when sex is the attribute).",
         "Reweighting": "Kamiran & Calders 2012: weight = P(group) P(label) / P(group, label).",
         "Adversarial debiasing": "Zhang et al. 2018: adversary predicts the group from the predictor's logit and the label; gradient reversal.",
         "Counterfactual augmentation (DiCE-style)": "Each sampled training row gets a counterfactual twin in another group: continuous/ordinal "
                      "predictors are moved by group-to-group quantile mapping, the label is kept (twins capped at 50% of rows). The "
                      "DiCE package is not installed; recourse metrics use the DiCE-style random search described above.",
         "Fair contrastive learning": "Park et al. 2022 fair supervised contrastive loss: positives share the label, negatives share the "
                                      "group but not the label; linear head on the frozen encoder."}
DIRECTIONS = {"Statistical parity difference": "min", "Disparate impact difference": "min",
              "Equal opportunity difference": "min", "Equalized odds difference": "min",
              "Counterfactual validity": "max", "Proximity": "min", "Sparsity": "min"}


def group_values(frame, attribute):
    return frame.meta[attribute].fillna("Unknown").astype(str).to_numpy()


def reference_model(train, cols):
    X = train.X[cols].to_numpy(float)
    return train_classifier(X, train.meta.target.to_numpy(), weights=train.meta.sample_weight.to_numpy(), steps=1500)


def detect(frames, schema):
    cols = schema.columns()
    train, val = frames["train"], frames["validation"]
    net = reference_model(train, cols)
    p = predict_proba(net, val.X[cols].to_numpy(float))
    pred = (p >= 0.5).astype(int)
    yv = val.meta.target.to_numpy().astype(int)
    out, outcome = {}, {}
    for a in ("sens_sex", "sens_age_group", "sens_race_ethnicity"):
        if a in val.meta:
            out[a] = fairness_metrics(yv, pred, group_values(val, a))
            g = group_values(train, a)
            yt = train.meta.target.to_numpy()
            outcome[a] = {k: float(yt[g == k].mean()) for k in pd.unique(g) if k != "Unknown" and (g == k).sum() >= 30}
    flags = {a: (m["di"] < 0.8) or (m["spd"] > 0.1) or (m["eodds"] > 0.1) for a, m in out.items() if a in PROTECTED}
    attribute = max((a for a in PROTECTED if a in out), key=lambda a: np.nan_to_num(out[a]["eodds"]))
    return {"statistic": "reference-model fairness on validation (SPD, DI, EOD, equalized odds)",
            "before": out[attribute]["eodds"], "attribute": attribute, "metrics": out, "outcome_rates": outcome,
            "reference_auroc": float(roc_auc_score(yv, p)),
            "rule": "present if, for sex or race/ethnicity, DI < 0.8 or SPD > 0.1 or equalized-odds difference > 0.1",
            "detected": bool(any(flags.values())), "flags": flags}


# --------------------------------------------------------------------------- data-level methods
class OTRepair:
    def __init__(self, cols, grid=101):
        self.cols, self.u = cols, np.linspace(0, 1, grid)

    def fit(self, X, groups):
        self.groups = [g for g in pd.unique(groups) if g != "Unknown" and (groups == g).sum() >= 30]
        pi = np.array([(groups == g).sum() for g in self.groups], float)
        pi /= pi.sum()
        self.q = {}
        for j in range(X.shape[1]):
            qs = {g: np.quantile(X[groups == g, j], self.u) for g in self.groups}
            self.q[j] = (qs, sum(p * qs[g] for p, g in zip(pi, self.groups)))
        return self

    def transform(self, X, groups):
        X = X.copy()
        for j, (qs, bary) in self.q.items():
            for g in self.groups:
                m = groups == g
                if m.any():
                    q = qs[g]
                    # rank within the group's training quantiles (ties spread uniformly), then barycenter quantile
                    lo = np.searchsorted(q, X[m, j], side="left")
                    hi = np.searchsorted(q, X[m, j], side="right")
                    u = (lo + hi) / 2 / (len(q) - 1)
                    X[m, j] = np.interp(np.clip(u, 0, 1), self.u, bary)
        return X


def reweigh(y, groups, w0):
    w = np.asarray(w0, float).copy()
    total = w.sum()
    for g in pd.unique(groups):
        for c in (0, 1):
            m = (groups == g) & (y == c)
            if m.any():
                expected = w[groups == g].sum() * w[y == c].sum() / total
                w[m] *= expected / w[m].sum()
    return w / w.mean()


def fair_smote(X, y, groups, schema, cols, flip_col=None, cap=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = [(g, c) for g in pd.unique(groups) if g != "Unknown" for c in (0, 1) if ((groups == g) & (y == c)).sum() >= 6]
    sizes = {k: int(((groups == k[0]) & (y == k[1])).sum()) for k in keys}
    target = max(sizes.values())
    need = {k: target - n for k, n in sizes.items()}
    total = sum(need.values())
    scale = min(1.0, cap * len(y) / max(total, 1))
    new_X, new_y, new_g, parents = [], [], [], []
    for k, n_new in need.items():
        n_new = int(n_new * scale)
        if n_new <= 0:
            continue
        ids = np.flatnonzero((groups == k[0]) & (y == k[1]))
        nb = NearestNeighbors(n_neighbors=min(6, len(ids))).fit(X[ids]).kneighbors(X[ids], return_distance=False)[:, 1:]
        p = rng.integers(0, len(ids), n_new)
        n1, n2 = nb[p, rng.integers(0, nb.shape[1], n_new)], nb[p, rng.integers(0, nb.shape[1], n_new)]
        cross = rng.random((n_new, X.shape[1])) < 0.8
        child = np.where(cross, X[ids[p]] + 0.8 * (X[ids[n1]] - X[ids[n2]]), X[ids[p]])
        new_X.append(child); new_y.append(np.full(n_new, k[1])); new_g.append(np.full(n_new, k[0], object))
        parents.append(ids[p])
    if not new_X:
        return X, y, groups, np.arange(len(y)), np.ones(len(y), bool)
    synth = postprocess(pd.DataFrame(np.vstack(new_X), columns=cols), schema).to_numpy(float)
    Xa, ya = np.vstack([X, synth]), np.r_[y, np.concatenate(new_y)]
    ga, src = np.r_[groups, np.concatenate(new_g)], np.r_[np.arange(len(y)), -1 - np.concatenate(parents)]
    keep = np.ones(len(ya), bool)
    if flip_col is not None:                                        # situation testing
        net = train_classifier(Xa, ya, steps=1000)
        flipped = Xa.copy()
        flipped[:, flip_col] = 1 - flipped[:, flip_col]
        keep = (predict_proba(net, Xa) >= 0.5) == (predict_proba(net, flipped) >= 0.5)
    return Xa[keep], ya[keep], ga[keep], src[keep], keep


def counterfactual_twins(X, y, groups, scal_idx, flip_col=None, cap=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    supported = [g for g in pd.unique(groups) if g != "Unknown" and (groups == g).sum() >= 30]
    u = np.linspace(0, 1, 101)
    q = {g: np.quantile(X[groups == g][:, scal_idx], u, axis=0) for g in supported}
    rows = np.flatnonzero(np.isin(groups, supported))
    rows = rng.choice(rows, min(len(rows), int(cap * len(y))), replace=False)
    twins, tg = X[rows].copy(), np.empty(len(rows), object)
    for i, r in enumerate(rows):
        others = [g for g in supported if g != groups[r]]
        tg[i] = others[rng.integers(len(others))]
    for g_from in supported:
        for g_to in supported:
            m = (groups[rows] == g_from) & (tg == g_to)
            if not m.any():
                continue
            for k, j in enumerate(scal_idx):
                qf, qt = q[g_from][:, k], q[g_to][:, k]
                lo, hi = np.searchsorted(qf, twins[m, j], "left"), np.searchsorted(qf, twins[m, j], "right")
                twins[m, j] = np.interp((lo + hi) / 2 / 100, u, qt)
    if flip_col is not None:
        twins[:, flip_col] = 1 - twins[:, flip_col]
    return np.vstack([X, twins]), np.r_[y, y[rows]], np.r_[groups, tg], np.r_[np.arange(len(y)), -1 - rows]


# --------------------------------------------------------------------------- model-level methods
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def adversarial(X, y, groups, w, steps=2000, alpha=1.0, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cats = pd.Categorical(groups)
    pred, adv = mlp(X.shape[1], 1), mlp(2, len(cats.categories), hidden=(16,))
    opt = torch.optim.Adam(list(pred.parameters()) + list(adv.parameters()), 1e-3)
    Xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    gt, wt = torch.from_numpy(cats.codes.astype(np.int64)), torch.from_numpy(np.asarray(w, np.float32))
    known = torch.from_numpy(np.asarray(groups) != "Unknown")
    for step in range(steps):
        i = torch.from_numpy(rng.choice(len(X), min(256, len(X)), replace=False))
        logit = pred(Xt[i]).squeeze(-1)
        lp = (F.binary_cross_entropy_with_logits(logit, yt[i], reduction="none") * wt[i]).sum() / wt[i].sum()
        lam = alpha * min(1.0, step / 500)
        a = adv(torch.stack([GradReverse.apply(logit, lam), yt[i]], 1))
        k = known[i]
        la = F.cross_entropy(a[k], gt[i][k]) if k.any() else torch.tensor(0.0)
        opt.zero_grad(); (lp + la).backward(); opt.step()
    pred.eval()
    return lambda Z: predict_proba(pred, Z)


def fair_contrastive(X, y, groups, w, steps=1500, tau=0.1, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = mlp(X.shape[1], 32, hidden=(64,))
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    Xt = torch.from_numpy(X.astype(np.float32))
    g = pd.Categorical(groups).codes
    for _ in range(steps):
        i = rng.choice(len(X), min(256, len(X)), replace=False)
        z = F.normalize(enc(Xt[torch.from_numpy(i)]), dim=1)
        sim = z @ z.T / tau
        yi, gi = torch.from_numpy(y[i].astype(np.int64)), torch.from_numpy(g[i].astype(np.int64))
        eye = torch.eye(len(i), dtype=torch.bool)
        pos = (yi[:, None] == yi[None]) & ~eye
        neg = (yi[:, None] != yi[None]) & (gi[:, None] == gi[None])
        denom = torch.logsumexp(sim.masked_fill(~(pos | neg), -1e9), dim=1)
        logp = sim - denom[:, None]
        valid = pos.sum(1) > 0
        loss = -(logp.masked_fill(~pos, 0).sum(1)[valid] / pos.sum(1)[valid]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    embed = lambda Z: enc(torch.from_numpy(np.asarray(Z, np.float32))).detach().numpy()
    head = train_classifier(embed(X), y, weights=w, steps=1000, hidden=(16,))
    return lambda Z: predict_proba(head, embed(Z))


# --------------------------------------------------------------------------- recourse metrics
def recourse(prob, Xv, Xtr, actionable, n_rows=300, n_cand=200, seed=SEED):
    rng = np.random.default_rng(seed)
    p = prob(Xv)
    pos = np.flatnonzero(p >= 0.5)
    if len(pos) == 0:
        return np.nan, np.nan, np.nan
    rows = rng.choice(pos, min(n_rows, len(pos)), replace=False)
    sd = Xtr.std(0)
    sd[sd == 0] = 1
    valid, prox, spars = 0, [], []
    for r in rows:
        cand = np.repeat(Xv[r][None], n_cand, 0)
        k = rng.integers(1, 4, n_cand)
        for c in range(n_cand):
            feats = rng.choice(actionable, min(k[c], len(actionable)), replace=False)
            cand[c, feats] = Xtr[rng.integers(0, len(Xtr), len(feats)), feats]
        ok = prob(cand) < 0.5
        if ok.any():
            valid += 1
            dist = (np.abs(cand[ok] - Xv[r]) / sd).sum(1)
            best = np.argmin(dist)
            prox.append(dist[best]); spars.append(int((cand[ok][best] != Xv[r]).sum()))
    return valid / len(rows), (np.mean(prox) if prox else np.nan), (np.mean(spars) if spars else np.nan)


def benchmark(frames, schema, detection, log, n_train=12000, n_val=6000):
    cols = schema.columns()
    attr = detection["attribute"]
    train, val = frames["train"], frames["validation"]
    ids = stratified_sample(None, train.meta.target.to_numpy(), n_train, seed=SEED + 14)
    vids = stratified_sample(None, val.meta.target.to_numpy(), n_val, seed=SEED + 15)
    X, y = train.X.iloc[ids][cols].to_numpy(float), train.meta.target.to_numpy().astype(int)[ids]
    w = train.meta.sample_weight.to_numpy()[ids]
    g = group_values(train, attr)[ids]
    Xv, yv, gv = val.X.iloc[vids][cols].to_numpy(float), val.meta.target.to_numpy().astype(int)[vids], group_values(val, attr)[vids]
    scal_idx = [cols.index(c) for c in schema.scalable]
    flip_col = cols.index("male") if attr == "sens_sex" and "male" in cols else None
    actionable = [i for i, c in enumerate(cols) if c not in ("age", "male", "age_over89")]
    ref = train_classifier(X, y, weights=w, steps=1500)
    ref_auc = roc_auc_score(yv, predict_proba(ref, Xv))
    rows, eligible = {}, []
    for name in FAMILY:
        t0 = time.time()
        try:
            Xe = Xv
            if name == "Optimal Transport repair":
                rep = OTRepair(scal_idx).fit(X[:, scal_idx], g)
                Xr = X.copy(); Xr[:, scal_idx] = rep.transform(X[:, scal_idx], g)
                Xe = Xv.copy(); Xe[:, scal_idx] = rep.transform(Xv[:, scal_idx], gv)
                net = train_classifier(Xr, y, weights=w, steps=1500)
                prob = lambda Z, net=net: predict_proba(net, Z)
            elif name == "FairSMOTE":
                Xa, ya, _, src, _ = fair_smote(X, y, g, schema, cols, flip_col)
                wa = np.where(src >= 0, w[np.where(src >= 0, src, 0)], 1.0)
                net = train_classifier(Xa, ya, weights=wa, steps=1500)
                prob = lambda Z, net=net: predict_proba(net, Z)
            elif name == "Reweighting":
                net = train_classifier(X, y, weights=reweigh(y, g, w), steps=1500)
                prob = lambda Z, net=net: predict_proba(net, Z)
            elif name == "Counterfactual augmentation (DiCE-style)":
                Xa, ya, _, src = counterfactual_twins(X, y, g, scal_idx, flip_col)
                wa = w[np.where(src >= 0, src, -1 - src)]
                net = train_classifier(Xa, ya, weights=wa, steps=1500)
                prob = lambda Z, net=net: predict_proba(net, Z)
            elif name == "Adversarial debiasing":
                prob = adversarial(X, y, g, w)
            else:
                prob = fair_contrastive(X, y, g, w)
            p = prob(Xe)
            m = fairness_metrics(yv, (p >= 0.5).astype(int), gv)
            validity, prox, spars = recourse(prob, Xe, X, actionable)
        except Exception as error:
            log(f"  demographic/{name} FAILED: {error!r}")
            rows[name] = {k: np.nan for k in DIRECTIONS}
            continue
        auc = roc_auc_score(yv, p)
        rows[name] = {"Statistical parity difference": m["spd"], "Disparate impact difference": abs(1 - m["di"]),
                      "Equal opportunity difference": m["eod"], "Equalized odds difference": m["eodds"],
                      "Counterfactual validity": validity, "Proximity": prox, "Sparsity": spars,
                      "Validation AUROC (guard)": auc}
        if auc >= ref_auc - 0.05:
            eligible.append(name)
        log(f"  demographic/{name}: EOdds {m['eodds']:.3f} SPD {m['spd']:.3f} AUROC {auc:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table.insert(1, "output", ["data" if m in DATA_LEVEL else "model" for m in table.index])
    table, best = select(table, DIRECTIONS, eligible)
    table["eligible (AUROC guard)"] = table.index.isin(eligible)
    protocol = (f"Attribute: {attr}. {len(ids):,} training rows; classifiers scored at 0.5 on {len(vids):,} validation rows. "
                f"Reference AUROC {ref_auc:.3f}; guard = AUROC >= {ref_auc - 0.05:.3f}. Disparate impact difference = |1 - DI|.")
    return table, best, protocol, eligible


def apply(frames, schema, method, detection, log):
    cols = schema.columns()
    attr = detection["attribute"]
    train = frames["train"]
    X, y = train.X[cols].to_numpy(float), train.meta.target.to_numpy().astype(int)
    g, w = group_values(train, attr), train.meta.sample_weight.to_numpy()
    scal_idx = [cols.index(c) for c in schema.scalable]
    flip_col = cols.index("male") if attr == "sens_sex" and "male" in cols else None
    new = {s: f.copy() for s, f in frames.items()}
    if method == "Optimal Transport repair":
        rep = OTRepair(scal_idx).fit(X[:, scal_idx], g)
        for split, frame in frames.items():
            Z = frame.X[cols].to_numpy(float)
            Z[:, scal_idx] = rep.transform(Z[:, scal_idx], group_values(frame, attr))
            new[split].X = pd.DataFrame(Z, columns=cols, index=frame.X.index)
        log(f"  {method}: repaired {len(scal_idx)} predictors across {attr} groups in every split")
        return new, {"attribute": attr}
    if method == "Reweighting":
        new["train"].meta["sample_weight"] = reweigh(y, g, w)
        log(f"  {method}: sample weights updated for {attr} x label")
        return new, {"attribute": attr}
    if method == "FairSMOTE":
        Xa, ya, ga, src, _ = fair_smote(X, y, g, schema, cols, flip_col)
    else:
        Xa, ya, ga, src = counterfactual_twins(X, y, g, scal_idx, flip_col)
    parent = np.where(src >= 0, src, -1 - src)
    meta = train.meta.iloc[parent].copy().reset_index(drop=True)
    synthetic = src < 0
    meta.loc[synthetic, attr] = ga[synthetic]
    if attr == "sens_sex" and flip_col is not None and method != "FairSMOTE":
        pass                                                      # sex already flipped in features + metadata
    for c in [c for c in meta.columns if c in ("meta_record_id", "meta_patientunitstayid")]:
        values = meta[c].astype(str).to_numpy()
        values[synthetic] = [f"{'fairsmote' if method == 'FairSMOTE' else 'counterfactual'}-{i}-from-{v}"
                             for i, v in enumerate(values[synthetic])]
        meta[c] = values
    meta.loc[synthetic, "meta_synthetic"] = 1
    meta["target"] = ya
    new["train"].X = pd.DataFrame(Xa, columns=cols)
    new["train"].meta = meta
    log(f"  {method}: training rows {len(y):,} -> {len(ya):,}")
    return new, {"attribute": attr, "rows_after": len(ya), "synthetic_rows": int(synthetic.sum())}
