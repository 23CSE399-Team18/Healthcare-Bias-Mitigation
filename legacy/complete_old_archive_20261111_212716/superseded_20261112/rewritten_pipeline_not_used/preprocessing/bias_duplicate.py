"""Duplicate bias: Exact duplicate removal, Fuzzy matching, GBER (graph-based entity resolution)
(traditional); Ditto, GNN entity resolution, LSH (modern).

Detection (before splitting): records identical in every column except the record identifier
(predictors, fairness groups, label, survey design / weights, domain). Repeated ICU stays of one
eICU patient are different stays, not duplicates; they are kept together by the patient-grouped
split instead.
Benchmark (known truth): two disjoint sets of real records receive 10% injected duplicates
(30% exact copies, 70% perturbed: small numeric changes, dropped fields, changed categories).
Thresholds and learned matchers are fitted on the calibration set and scored on the evaluation
set. Metrics: duplicate rate after removal, redundancy reduction, pairwise precision / recall /
F1, pair completeness (share of true pairs among the method's candidates) and reduction ratio
(1 - candidates / all pairs).
Applying the selected method to the real data only compares records inside identical survey-
design / domain / fairness-group blocks, so different respondents cannot be merged.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from common import SEED, select

FAMILY = {"Exact duplicate removal": "Traditional", "Fuzzy matching": "Traditional", "GBER (graph-based ER)": "Traditional",
          "Ditto (transformer matcher)": "Modern", "GNN entity resolution": "Modern", "LSH (MinHash)": "Modern"}
NOTES = {"Fuzzy matching": "Mean per-field similarity (numeric: exp(-|a-b|/MAD-scale); categorical: equality) over blocked candidates; threshold tuned on calibration.",
         "GBER (graph-based ER)": "Similarity graph over candidates, edges above a tuned threshold, transitive closure (components of size <= 5).",
         "Ditto (transformer matcher)": "Li et al. 2020: pretrained DistilBERT (local cache) fine-tuned on serialised record pairs ('[COL] field [VAL] value'); candidates pre-filtered at fuzzy similarity >= 0.5.",
         "GNN entity resolution": "Two-layer GraphSAGE over the candidate graph (PyTorch Geometric) with an edge classifier on node embeddings and field similarities.",
         "LSH (MinHash)": "64 MinHash permutations of 'field=value' tokens (numeric values binned), 16 bands x 4 rows; Jaccard threshold tuned.",
         "Exact duplicate removal": "Identical values in every compared field."}
DIRECTIONS = {"Duplicate rate": "min", "Redundancy reduction": "max", "Precision": "max", "Recall": "max",
              "F1-score": "max", "Pair completeness": "max", "Reduction ratio": "max"}
ID_COLS = ("meta_record_id", "meta_patientunitstayid")


def detect(raw):
    cols = [c for c in raw.columns if c not in ID_COLS]
    dup = raw.duplicated(subset=cols, keep="first")
    return {"statistic": "records identical in every column except the record ID", "before": int(dup.sum()),
            "rate": float(dup.mean()), "rule": "present if any exact duplicate record", "detected": bool(dup.any()),
            "duplicate_mask": dup.to_numpy()}


# --------------------------------------------------------------------------- record encoding
class Fields:
    def __init__(self, df):
        self.cols = list(df.columns)
        self.numeric = [c for c in self.cols if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical = [c for c in self.cols if c not in self.numeric]
        x = df[self.numeric].to_numpy(float)
        med = np.nanmedian(x, 0)
        self.scale = np.nanmedian(np.abs(x - med), 0) * 1.4826
        sd = np.nanstd(x, 0)
        self.scale = np.where(self.scale > 0, self.scale, np.where(sd > 0, sd, 1.0))
        self.mean, self.sd = np.nanmean(x, 0), np.where(sd > 0, sd, 1.0)
        self.bins = {c: np.unique(np.nanquantile(df[c].to_numpy(float), np.linspace(0, 1, 41))) for c in self.numeric}

    def vectors(self, df):
        x = (df[self.numeric].to_numpy(float) - self.mean) / self.sd
        return np.nan_to_num(x)

    def similarity(self, df, pairs):
        """Per-field similarity matrix for pairs (n_pairs, n_fields)."""
        a, b = df.iloc[pairs[:, 0]], df.iloc[pairs[:, 1]]
        xa, xb = a[self.numeric].to_numpy(float), b[self.numeric].to_numpy(float)
        sn = np.exp(-np.abs(xa - xb) / self.scale)
        sn = np.where(np.isnan(xa) & np.isnan(xb), 1.0, np.where(np.isnan(xa) | np.isnan(xb), 0.5, sn))
        ca = a[self.categorical].astype("string").fillna("<missing>").to_numpy(str)
        cb = b[self.categorical].astype("string").fillna("<missing>").to_numpy(str)
        na, nb = ca == "<missing>", cb == "<missing>"
        sc = np.where(na & nb, 1.0, np.where(na | nb, 0.5, (ca == cb).astype(float)))
        return np.column_stack([sn, sc]).astype(float)

    def tokens(self, df):
        out = []
        num = {c: np.digitize(df[c].to_numpy(float), self.bins[c]) for c in self.numeric}
        nanm = {c: df[c].isna().to_numpy() for c in self.numeric}
        cat = {c: df[c].astype("string").fillna("<na>").to_numpy() for c in self.categorical}
        for i in range(len(df)):
            t = [f"{c}={'na' if nanm[c][i] else num[c][i]}" for c in self.numeric]
            t += [f"{c}={cat[c][i]}" for c in self.categorical]
            out.append(t)
        return out

    def serialise(self, df):
        parts = []
        for c in self.cols:
            v = df[c].to_numpy()
            parts.append([f"[COL] {c} [VAL] {('' if pd.isna(x) else (round(x, 2) if isinstance(x, float) else x))}" for x in v])
        return [" ".join(p[i] for p in parts) for i in range(len(df))]


def inject(base, rate=0.10, seed=SEED):
    """Return records = base + duplicates, and the true pair list (original_pos, duplicate_pos)."""
    rng = np.random.default_rng(seed)
    n_dup = max(10, int(rate * len(base)))
    src = rng.choice(len(base), n_dup, replace=False)
    dups = base.iloc[src].copy().reset_index(drop=True)
    numeric = [c for c in dups.columns if pd.api.types.is_numeric_dtype(dups[c])]
    categorical = [c for c in dups.columns if c not in numeric]
    perturbed = rng.random(n_dup) >= 0.3
    for c in numeric:
        v = dups[c].to_numpy(float).copy()
        levels = np.unique(base[c].dropna())
        few = len(levels) <= 10
        change = perturbed & (rng.random(n_dup) < (0.05 if few else 0.3))
        if few and len(levels) > 1:
            v[change] = rng.choice(levels, change.sum())
        else:
            v[change] = v[change] * (1 + rng.uniform(-0.03, 0.03, change.sum()))
        v[perturbed & (rng.random(n_dup) < 0.1)] = np.nan
        dups[c] = v
    for c in categorical:
        levels = base[c].dropna().unique()
        change = perturbed & (rng.random(n_dup) < 0.05)
        if len(levels) > 1 and change.any():
            dups.loc[change, c] = rng.choice(levels, change.sum())
    records = pd.concat([base.reset_index(drop=True), dups], ignore_index=True)
    pairs = np.column_stack([src, len(base) + np.arange(n_dup)])
    return records, pairs


def blocked_candidates(records, fields, k=5):
    block = records[[c for c in ("sens_sex", "sens_age_group") if c in records]].astype("string").fillna("na").agg("|".join, axis=1)
    X = fields.vectors(records)
    pairs = set()
    for _, ids in pd.Series(np.arange(len(records))).groupby(block.to_numpy()):
        ids = ids.to_numpy()
        if len(ids) < 2:
            continue
        nb = NearestNeighbors(n_neighbors=min(k + 1, len(ids))).fit(X[ids]).kneighbors(X[ids], return_distance=False)
        for i, row in enumerate(nb):
            for j in row:
                if ids[i] != ids[j]:
                    pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def minhash_candidates(tokens, perms=64, bands=16, seed=SEED):
    rng = np.random.default_rng(seed)
    vocab = {t: i for i, t in enumerate(sorted({t for ts in tokens for t in ts}))}
    a, b = rng.integers(1, 2 ** 31 - 1, perms), rng.integers(0, 2 ** 31 - 1, perms)
    prime = 2 ** 31 - 1
    sig = np.empty((len(tokens), perms), dtype=np.int64)
    for i, ts in enumerate(tokens):
        ids = np.array([vocab[t] for t in ts], dtype=np.int64)
        sig[i] = ((a[:, None] * ids[None] + b[:, None]) % prime).min(1)
    rows = perms // bands
    buckets = defaultdict(list)
    for i in range(len(tokens)):
        for band in range(bands):
            buckets[(band, sig[i, band * rows:(band + 1) * rows].tobytes())].append(i)
    pairs = set()
    for members in buckets.values():
        if 1 < len(members) <= 50:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    pairs = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
    jacc = (sig[pairs[:, 0]] == sig[pairs[:, 1]]).mean(1) if len(pairs) else np.zeros(0)
    return pairs, jacc


def exact_pairs(records):
    key = records.astype("string").fillna("<na>").agg("\x1f".join, axis=1)
    pairs = []
    for _, ids in pd.Series(np.arange(len(records))).groupby(key.to_numpy()):
        ids = ids.to_numpy()
        pairs += [(ids[0], j) for j in ids[1:]]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def components(n, pairs, max_size=5):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    size = np.ones(n, int)
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_size:
            parent[rb] = ra
            size[ra] += size[rb]
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = [(g[0], j) for g in groups.values() if len(g) > 1 for j in g[1:]]
    return np.array(out, dtype=int).reshape(-1, 2)


def best_threshold(scores, labels):
    best, best_t = -1, 0.5
    for t in np.unique(np.quantile(scores, np.linspace(0, 1, 101))):
        p = scores >= t
        tp = (p & labels).sum()
        f1 = 2 * tp / max(p.sum() + labels.sum(), 1)
        if f1 > best:
            best, best_t = f1, t
    return best_t


# --------------------------------------------------------------------------- learned matchers
class GNNMatcher:
    def fit(self, records, fields, cand, labels, steps=300, seed=SEED):
        from torch_geometric.nn import SAGEConv
        torch.manual_seed(seed)
        d = len(fields.numeric)
        self.c1, self.c2 = SAGEConv(d * 2, 32), SAGEConv(32, 32)
        n_sim = len(fields.cols)
        self.head = nn.Sequential(nn.Linear(64 + n_sim, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(list(self.c1.parameters()) + list(self.c2.parameters()) + list(self.head.parameters()), 5e-3)
        x, ei, sim = self._graph(records, fields, cand)
        y = torch.from_numpy(labels.astype(np.float32))
        pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1))
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(self._logits(x, ei, cand, sim), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _graph(self, records, fields, cand):
        v = records[fields.numeric].to_numpy(float)
        x = np.column_stack([np.nan_to_num((v - fields.mean) / fields.sd), np.isnan(v)])
        ei = np.vstack([np.r_[cand[:, 0], cand[:, 1]], np.r_[cand[:, 1], cand[:, 0]]])
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(ei.astype(np.int64)),
                torch.from_numpy(fields.similarity(records, cand).astype(np.float32)))

    def _logits(self, x, ei, cand, sim):
        h = F.relu(self.c2(F.relu(self.c1(x, ei)), ei))
        a, b = h[torch.from_numpy(cand[:, 0])], h[torch.from_numpy(cand[:, 1])]
        return self.head(torch.cat([(a - b).abs(), a * b, sim], 1)).squeeze(-1)

    @torch.no_grad()
    def score(self, records, fields, cand):
        x, ei, sim = self._graph(records, fields, cand)
        return torch.sigmoid(self._logits(x, ei, cand, sim)).numpy()


class DittoMatcher:
    NAME = "distilbert-base-uncased"

    def fit(self, texts, cand, labels, epochs=2, seed=SEED):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.tok = AutoTokenizer.from_pretrained(self.NAME, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.NAME, num_labels=2, local_files_only=True)
        pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
        neg = rng.choice(neg, min(len(neg), 3 * max(len(pos), 1)), replace=False)
        train = rng.permutation(np.r_[pos, neg])
        opt = torch.optim.AdamW(self.model.parameters(), 3e-5)
        self.model.train()
        for _ in range(epochs):
            for s in range(0, len(train), 16):
                ix = train[s:s + 16]
                enc = self._encode(texts, cand[ix])
                out = self.model(**enc, labels=torch.from_numpy(labels[ix].astype(np.int64)))
                opt.zero_grad(); out.loss.backward(); opt.step()
        self.model.eval()
        return self

    def _encode(self, texts, pairs):
        return self.tok([texts[a] for a in pairs[:, 0]], [texts[b] for b in pairs[:, 1]], truncation=True,
                        max_length=256, padding=True, return_tensors="pt")

    @torch.no_grad()
    def score(self, texts, pairs):
        out = []
        for s in range(0, len(pairs), 64):
            out.append(torch.softmax(self.model(**self._encode(texts, pairs[s:s + 64])).logits, -1)[:, 1].numpy())
        return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------- benchmark
def _truth(cand, true_pairs):
    t = {(min(a, b), max(a, b)) for a, b in true_pairs}
    return np.array([(min(a, b), max(a, b)) in t for a, b in cand], bool)


def _scores(records, true_pairs, pred_pairs, cand_count):
    n = len(records)
    t = {(min(a, b), max(a, b)) for a, b in true_pairs}
    p = {(min(a, b), max(a, b)) for a, b in pred_pairs}
    tp = len(t & p)
    prec, rec = tp / max(len(p), 1), tp / max(len(t), 1)
    dups = set(true_pairs[:, 1])
    removed = {max(a, b) for a, b in p}
    remaining = n - len(removed)
    rate_before = len(dups) / n
    rate_after = len(dups - removed) / max(remaining, 1)
    return {"Duplicate rate": rate_after, "Redundancy reduction": 1 - rate_after / rate_before if rate_before else 0.0,
            "Precision": prec, "Recall": rec, "F1-score": 2 * prec * rec / max(prec + rec, 1e-12),
            "Reduction ratio": 1 - cand_count / (n * (n - 1) / 2)}


def benchmark(raw, log, n_base=3000):
    rng = np.random.default_rng(SEED)
    fields_cols = [c for c in raw.columns if not c.startswith("meta_")]
    pool = raw.loc[~raw.duplicated(subset=[c for c in raw.columns if c not in ID_COLS]), fields_cols]
    ids = rng.permutation(len(pool))
    sets = {}
    for name, part in (("calibration", ids[:n_base]), ("evaluation", ids[n_base:2 * n_base])):
        rec, pairs = inject(pool.iloc[part], seed=SEED + (0 if name == "calibration" else 1))
        sets[name] = (rec, pairs)
    fields = Fields(sets["calibration"][0])
    prepared = {}
    for name, (rec, pairs) in sets.items():
        cand = blocked_candidates(rec, fields)
        sim = fields.similarity(rec, cand)
        prepared[name] = {"rec": rec, "pairs": pairs, "cand": cand, "fuzzy": sim.mean(1), "label": _truth(cand, pairs),
                          "tokens": fields.tokens(rec), "texts": fields.serialise(rec)}
    cal, ev = prepared["calibration"], prepared["evaluation"]
    rows, models = {}, {}
    for name in FAMILY:
        t0 = time.time()
        try:
            if name == "Exact duplicate removal":
                pred = exact_pairs(ev["rec"])
                cand_pairs, pc_label = pred, _truth(pred, ev["pairs"])
                models[name] = None
            elif name == "LSH (MinHash)":
                cc, cj = minhash_candidates(cal["tokens"])
                thr = best_threshold(cj, _truth(cc, cal["pairs"])) if len(cc) else 1.0
                ec, ej = minhash_candidates(ev["tokens"])
                pred, cand_pairs, pc_label = ec[ej >= thr], ec, _truth(ec, ev["pairs"])
                models[name] = thr
            else:
                cand_pairs, pc_label = ev["cand"], ev["label"]
                if name == "Fuzzy matching":
                    thr = best_threshold(cal["fuzzy"], cal["label"])
                    pred = ev["cand"][ev["fuzzy"] >= thr]
                    models[name] = thr
                elif name == "GBER (graph-based ER)":
                    best, thr = -1, 0.5
                    for t in np.linspace(0.5, 0.99, 50):
                        s = _scores(cal["rec"], cal["pairs"], components(len(cal["rec"]), cal["cand"][cal["fuzzy"] >= t]), 1)
                        if s["F1-score"] > best:
                            best, thr = s["F1-score"], t
                    pred = components(len(ev["rec"]), ev["cand"][ev["fuzzy"] >= thr])
                    models[name] = thr
                elif name == "GNN entity resolution":
                    model = GNNMatcher().fit(cal["rec"], fields, cal["cand"], cal["label"])
                    thr = best_threshold(model.score(cal["rec"], fields, cal["cand"]), cal["label"])
                    pred = ev["cand"][model.score(ev["rec"], fields, ev["cand"]) >= thr]
                    models[name] = (model, thr)
                else:
                    keep_c, keep_e = cal["fuzzy"] >= 0.5, ev["fuzzy"] >= 0.5
                    model = DittoMatcher().fit(cal["texts"], cal["cand"][keep_c], cal["label"][keep_c])
                    sub = ev["cand"][keep_e]
                    pred = sub[model.score(ev["texts"], sub) >= 0.5]
                    cand_pairs, pc_label = sub, ev["label"][keep_e]
                    models[name] = model
            scores = _scores(ev["rec"], ev["pairs"], pred, len(cand_pairs))
            scores["Pair completeness"] = pc_label.sum() / len(ev["pairs"])
            rows[name] = scores
        except Exception as error:
            log(f"  duplicate/{name} FAILED: {error!r}")
            rows[name] = {m: np.nan for m in DIRECTIONS}
            continue
        log(f"  duplicate/{name}: F1 {rows[name]['F1-score']:.3f} PC {rows[name]['Pair completeness']:.3f} ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows).T[list(DIRECTIONS)]
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table, best = select(table, DIRECTIONS)
    protocol = (f"Two disjoint sets of {n_base:,} real records (fields = predictors, fairness groups, label) each "
                f"received {len(sets['evaluation'][1]):,} injected duplicates (30% exact, 70% perturbed). Calibration set: "
                "thresholds / training of learned matchers; evaluation set: scores. Blocking for fuzzy, GBER, GNN and Ditto: "
                "same sex and age group, 5 nearest neighbours on standardised numeric fields.")
    return table, best, protocol, {"fields": fields, "models": models}


def apply(raw, method, context, log):
    """Remove duplicates inside blocks of identical design / domain / fairness metadata."""
    block_cols = [c for c in raw.columns if c.startswith(("meta_", "sens_")) and c not in ID_COLS]
    if method == "Exact duplicate removal" or method is None:
        dup = raw.duplicated(subset=[c for c in raw.columns if c not in ID_COLS], keep="first").to_numpy()
    else:
        fields, model = context["fields"], context["models"].get(method)
        cols = fields.cols
        key = raw[block_cols].astype("string").fillna("<na>").agg("\x1f".join, axis=1).to_numpy()
        order = pd.Series(np.arange(len(raw))).groupby(key)
        cand = np.array([(a, b) for _, ids in order for ia, a in enumerate(ids.to_numpy()) for b in ids.to_numpy()[ia + 1:]],
                        dtype=int).reshape(-1, 2)
        recs = raw[cols].reset_index(drop=True)
        if len(cand) == 0:
            pred = cand
        elif method in ("Fuzzy matching", "GBER (graph-based ER)"):
            sim = fields.similarity(recs, cand).mean(1)
            pred = cand[sim >= model] if method == "Fuzzy matching" else components(len(recs), cand[sim >= model])
        elif method == "LSH (MinHash)":
            sig = fields.tokens(recs)
            j = np.array([len(set(sig[a]) & set(sig[b])) / len(set(sig[a]) | set(sig[b])) for a, b in cand])
            pred = cand[j >= model]
        elif method == "GNN entity resolution":
            pred = cand[model[0].score(recs, fields, cand) >= model[1]]
        else:
            pred = cand[model.score(fields.serialise(recs), cand) >= 0.5]
        dup = np.zeros(len(raw), bool)
        dup[[max(a, b) for a, b in pred]] = True
    log(f"  duplicates removed: {int(dup.sum()):,}")
    return raw.loc[~dup].reset_index(drop=True), {"removed": int(dup.sum())}
