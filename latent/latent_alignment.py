"""Combined latent feature extraction across the four preprocessed datasets.

    python latent/latent_alignment.py      (or run latent_alignment.ipynb)

One common 16-dimensional latent space is learned for heart, diabetes, hypertension and
respiratory. Each dataset has its own input encoder (its predictors differ); the latent space
is shared. Methods (the team's plan): DGCCA, SAINT, MV-DAE, compared with a per-dataset PCA
baseline that has no alignment ("before").

Metrics (latent.png), all on validation rows:
  CKA (cross-dataset alignment, higher better): the four datasets contain different people,
      so CKA compares the latent centroids of shared demographic strata (age group x sex x BMI
      category, >= 20 rows) between every pair of datasets (linear CKA), averaged over pairs.
  Macro-F1 (clinical information preservation, higher better): class-balanced logistic probe
      trained on training latents, scored on validation latents, averaged over datasets.
  Silhouette (latent structure, higher better): silhouette of the clinical classes, <= 2,000
      stratified validation rows per dataset, averaged.
  NMI / ARI (domain invariance, lower better): k-means (k = 4) fitted on pooled training latents
      (equal rows per dataset), compared with dataset identity on pooled validation latents.
Selection: best mean rank over the five metrics; test rows are only encoded, never scored.
Neural encoders use the training split only (synthetic training rows included, flagged in the
outputs); checkpoints are chosen on a validation objective.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]          # D:/complete
SEED = 2026


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


class Logger:
    def __init__(self, path):
        self.path, self.t0 = Path(path), time.time()

    def __call__(self, message):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:7.1f}s] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def mlp(d_in, d_out, hidden=(64, 32), act=nn.ReLU):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def mean_rank(table, directions):
    ranks = {m: pd.to_numeric(table[m], errors="coerce").rank(ascending=(d == "min"), method="average")
             for m, d in directions.items()}
    return pd.DataFrame(ranks).mean(axis=1)


ID_COL = {"heart": "meta_record_id", "diabetes": "meta_record_id", "hypertension": "meta_record_id",
          "respiratory": "meta_patientunitstayid", "hospital": "meta_stay_id"}
# Cross-hospital ICU cohort (hospital/, built from eICU-CRD v2.0 + MIMIC-IV v3.1): one table, four recorded-condition labels.
# eICU training/validation/test hospitals are the usual splits; MIMIC (one other hospital) is the untouched "external" split,
# encoded with the trained encoder but never used for training, checkpointing, scaling or the metrics below.
HOSPITAL = "hospital"
HOSPITAL_LABELS = ("target_heart", "target_diabetes", "target_hypertension", "target_respiratory")
EXTRA_SPLITS = {HOSPITAL: ("external",)}
NO_BMI = {HOSPITAL}                      # no BMI recorded: stratum = age x sex


def splits_of(name):
    return SPLITS + EXTRA_SPLITS.get(name, ())


def bmi_category(name):
    """Shared BMI category (1 <18.5, 2 18.5-25, 3 25-30, 4 >=30) from the raw data, keyed by record ID."""
    raw = pd.read_csv(ROOT / name / "raw" / f"{name}_raw.csv", low_memory=False)
    if "bmi_category" in raw:
        cat = raw.bmi_category
    else:
        col = "admission_bmi" if "admission_bmi" in raw else "bmi"
        cat = pd.cut(raw[col], [0, 18.5, 25, 30, 1000], right=False, labels=[1, 2, 3, 4]).astype(float)
    return dict(zip(raw[ID_COL[name]].astype(str), cat))

DATASETS = ("heart", "diabetes", "hypertension", "respiratory", HOSPITAL)
SPLITS = ("train", "validation", "test")
DIM = 16
OUT = ROOT / "latent"
METHODS = ("PCA (no alignment)", "MV-DAE", "DGCCA", "SAINT")
FAMILY = {"PCA (no alignment)": "Baseline", "MV-DAE": "Traditional", "DGCCA": "Modern", "SAINT": "Modern"}
DIRECTIONS = {"CKA (alignment)": "max", "Macro-F1 (preservation)": "max", "Silhouette (structure)": "max",
              "NMI (domain)": "min", "ARI (domain)": "min"}
NOTES = {"PCA (no alignment)": "16 principal components fitted separately on each dataset's training rows: no shared space (the 'before').",
         "MV-DAE": "Multi-view denoising autoencoder: dataset-specific encoders/decoders (15% masking + Gaussian noise), "
                   "shared 16-D code, RBF-MMD alignment between datasets and a per-dataset label head.",
         "DGCCA": "Deep generalized CCA (Benton et al. 2019, MAXVAR objective) over five views formed by stratum-matched tuples "
                  "(one training row per dataset from the same age x sex x BMI stratum; the cross-hospital ICU view, which has "
                  "no BMI, is matched on age x sex), plus reconstruction and label heads; final linear GCCA projection fitted "
                  "on training tuples.",
         "SAINT": "SAINT-style transformer (Somepalli et al. 2021): dataset-specific feature tokens, shared column-attention "
                  "encoder, inter-sample attention to a fixed bank of training rows (inductive), contrastive + denoising "
                  "pre-training objective, MMD alignment and label heads."}
STEPS, BATCH = 3000, 128


# --------------------------------------------------------------------------- data
def load():
    """Final outputs of <dataset>/<dataset>_preprocessing.ipynb: every column except target, sens_* and meta_* is a
    model input; meta_sample_weight is the training weight; meta_synthetic = 1 marks generated rows."""
    data = {}
    for name in DATASETS:
        base = ROOT / name / "preprocessed"
        bmi = {} if name in NO_BMI else bmi_category(name)
        d = {}
        for split in splits_of(name):
            df = pd.read_csv(base / f"{name}_{split}_preprocessed.csv", low_memory=False)
            feats = [c for c in df.columns if not c.startswith(("meta_", "sens_", "target"))]
            d.setdefault("features", feats)
            assert feats == d["features"], f"{name}: feature columns differ between splits"
            meta = df[[c for c in df.columns if c not in feats]].copy()
            meta["sample_weight"] = meta.get("meta_sample_weight", 1.0)
            if "meta_synthetic" not in meta:
                meta["meta_synthetic"] = 0
            meta["meta_anchor_bmi_cat"] = meta[ID_COL[name]].astype(str).map(bmi).where(meta["meta_synthetic"] == 0)
            meta["meta_dataset"] = name
            y = df[list(HOSPITAL_LABELS)] if name == HOSPITAL else df.target   # hospital: (n, 4) multi-label
            d[split] = {"X": df[feats].to_numpy(np.float32), "y": y.to_numpy().astype(int), "meta": meta}
        sc = StandardScaler().fit(d["train"]["X"])
        sc.scale_ = np.where(sc.scale_ > 1e-6, sc.scale_, 1.0)
        for split in splits_of(name):
            d[split]["Z"] = sc.transform(d[split]["X"]).astype(np.float32)
        d["strata"] = {s: strata(d[s]["meta"]) for s in splits_of(name)}
        d["strata_as"] = {s: strata(d[s]["meta"], bmi=False) for s in splits_of(name)}
        data[name] = d
    return data


def strata(meta, bmi=True):
    """age x sex x BMI-category key ("NA" if any part is unknown); bmi=False gives the age x sex key."""
    age = meta["sens_age_group"].astype("string").fillna("NA")
    sex = meta["sens_sex"].astype("string").fillna("NA")
    key = age + "|" + sex
    if bmi:
        key = key + "|" + meta["meta_anchor_bmi_cat"].astype("string").fillna("NA")
    key = key.to_numpy().astype(str)
    key[np.char.find(key, "NA") >= 0] = "NA"
    key[np.char.find(key, "Unknown") >= 0] = "NA"
    return key


def stratum_keys(data, name, split, keys):
    """Each dataset's own stratum key for the shared age|sex|BMI keys (age|sex for datasets without BMI)."""
    if name in NO_BMI:
        return data[name]["strata_as"][split], [k.rsplit("|", 1)[0] for k in keys]
    return data[name]["strata"][split], list(keys)


def label_loss(logits, y, w):
    """Weighted BCE of the dataset's label head; multi-label views average their labels per row."""
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1) if y.ndim == 1 else logits, y, reduction="none")
    return ((loss if y.ndim == 1 else loss.mean(1)) * w).mean()


def label_columns(y):
    """Binary label vectors of a dataset: one for single-label datasets, one per recorded condition otherwise."""
    return [y] if y.ndim == 1 else [y[:, j] for j in range(y.shape[1])]


def real(d, split):
    return d[split]["meta"]["meta_synthetic"].to_numpy() == 0


# --------------------------------------------------------------------------- losses
def rbf_mmd_t(a, b):
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z).pow(2)
    bw = d2.detach().median().clamp(min=1e-3)
    k = torch.exp(-d2 / bw)
    n = len(a)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def mmd_alignment(codes):
    pooled = torch.cat(codes)
    return sum(rbf_mmd_t(c, pooled[torch.randperm(len(pooled))[: len(c)]]) for c in codes) / len(codes)


# --------------------------------------------------------------------------- models
def n_labels(name):
    return len(HOSPITAL_LABELS) if name == HOSPITAL else 1


class MVDAE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.enc = nn.ModuleDict({k: mlp(d, DIM, hidden=(128, 64)) for k, d in dims.items()})
        self.dec = nn.ModuleDict({k: mlp(DIM, d, hidden=(64, 128)) for k, d in dims.items()})
        self.head = nn.ModuleDict({k: nn.Linear(DIM, n_labels(k)) for k in dims})

    def encode(self, name, x, corrupt=False):
        if corrupt:
            x = x * (torch.rand_like(x) > 0.15).float() + 0.1 * torch.randn_like(x)
        return self.enc[name](x)

    def loss(self, batch):
        codes, total = [], 0.0
        for name, (x, y, w) in batch.items():
            z = self.encode(name, x, corrupt=True)
            codes.append(z)
            total = total + F.mse_loss(self.dec[name](z), x) + 0.5 * label_loss(self.head[name](z), y, w)
        return total / len(batch) + 1.0 * mmd_alignment(codes)


class DGCCA(nn.Module):
    def __init__(self, dims, hidden=32):
        super().__init__()
        self.enc = nn.ModuleDict({k: mlp(d, hidden, hidden=(128, 64)) for k, d in dims.items()})
        self.dec = nn.ModuleDict({k: mlp(hidden, d, hidden=(64,)) for k, d in dims.items()})
        self.head = nn.ModuleDict({k: nn.Linear(hidden, n_labels(k)) for k in dims})

    def loss(self, batch):
        whitened, total = [], 0.0
        for name, (x, y, w) in batch.items():
            h = self.enc[name](x)
            total = total + F.mse_loss(self.dec[name](h), x) + 0.5 * label_loss(self.head[name](h), y, w)
            hc = (h - h.mean(0)) / np.sqrt(len(h) - 1)
            chol = torch.linalg.cholesky(hc.T @ hc + 1e-2 * torch.eye(h.shape[1]))
            whitened.append(torch.linalg.solve_triangular(chol, hc.T, upper=False).T)
        sv = torch.linalg.svdvals(torch.cat(whitened, 1))
        correlation = sv[:DIM].pow(2).sum() / (DIM * len(batch))      # in [0, 1]
        return total / len(batch) - 1.0 * correlation

    @torch.no_grad()
    def hidden(self, name, x):
        return self.enc[name](x)


class SAINT(nn.Module):
    def __init__(self, dims, refs, width=32):
        super().__init__()
        self.w = nn.ParameterDict({k: nn.Parameter(torch.randn(d, width) * 0.1) for k, d in dims.items()})
        self.b = nn.ParameterDict({k: nn.Parameter(torch.zeros(d, width)) for k, d in dims.items()})
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.col = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, 4, 64, dropout=0.1, batch_first=True), 2)
        self.q, self.k, self.v = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, DIM)
        self.dec = nn.ModuleDict({k: mlp(DIM, d, hidden=(64,)) for k, d in dims.items()})
        self.head = nn.ModuleDict({k: nn.Linear(DIM, n_labels(k)) for k in dims})
        self.proj = mlp(DIM, DIM, hidden=(32,))
        for k, r in refs.items():
            self.register_buffer(f"ref_{k}", torch.as_tensor(r))

    def row_tokens(self, name, x):
        t = x[:, :, None] * self.w[name][None] + self.b[name][None]
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.col(t)[:, 0]

    def encode(self, name, x):
        h = self.row_tokens(name, x)
        r = self.row_tokens(name, getattr(self, f"ref_{name}"))            # fixed training bank (inductive)
        att = torch.softmax(self.q(h) @ self.k(r).T / np.sqrt(h.shape[1]), 1)
        return self.out(self.norm(h + att @ self.v(r)))

    def loss(self, batch):
        codes, total = [], 0.0
        for name, (x, y, w) in batch.items():
            mixed = torch.where(torch.rand_like(x) < 0.3, x[torch.randperm(len(x))], x)    # CutMix in input space
            z1, z2 = self.encode(name, x), self.encode(name, mixed)
            a, b = F.normalize(self.proj(z1), dim=1), F.normalize(self.proj(z2), dim=1)
            logits = a @ b.T / 0.7
            lab = torch.arange(len(x))
            contrast = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            total = total + contrast + F.mse_loss(self.dec[name](z2), x) + 0.5 * label_loss(self.head[name](z1), y, w)
            codes.append(z1)
        return total / len(batch) + 1.0 * mmd_alignment(codes)


# --------------------------------------------------------------------------- training
def sample_batch(data, split, rng, tuples=False):
    batch = {}
    if tuples:                                   # stratum-matched tuples for DGCCA
        common = data.setdefault("_common", {}).get(split)
        if common is None:
            full = [n for n in DATASETS if n not in NO_BMI]
            common = set.intersection(*[set(np.unique(data[n]["strata"][split][real(data[n], split)])) for n in full]) - {"NA"}
            for n in NO_BMI:                     # keep keys whose age|sex stratum also exists in the datasets without BMI
                have = set(np.unique(data[n]["strata_as"][split][real(data[n], split)])) - {"NA"}
                common = {k for k in common if k.rsplit("|", 1)[0] in have}
            common = data["_common"][split] = sorted(common)
        keys = rng.choice(common, BATCH)
    for name in DATASETS:
        d = data[name][split]
        if tuples:
            pools = data[name].get(f"_pool_{split}")
            if pools is None:
                st, own = stratum_keys(data, name, split, common)
                pools = data[name][f"_pool_{split}"] = {k: np.flatnonzero((st == o) & real(data[name], split)) for k, o in zip(common, own)}
            ids = np.array([rng.choice(pools[k]) for k in keys])
        else:
            ids = rng.choice(len(d["y"]), BATCH, replace=len(d["y"]) < BATCH)
        w = d["meta"]["sample_weight"].to_numpy(np.float32)[ids]
        batch[name] = (torch.from_numpy(d["Z"][ids]), torch.from_numpy(d["y"][ids].astype(np.float32)), torch.from_numpy(w / w.mean()))
    return batch


def train(method, data, log):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dims = {n: data[n]["features"].__len__() for n in DATASETS}
    if method == "MV-DAE":
        model = MVDAE(dims)
    elif method == "DGCCA":
        model = DGCCA(dims)
    else:
        refs = {n: data[n]["train"]["Z"][rng.choice(len(data[n]["train"]["y"]), 32, replace=False)] for n in DATASETS}
        model = SAINT(dims, refs)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    vrng = np.random.default_rng(SEED + 1)
    vbatches = [sample_batch(data, "validation", vrng, tuples=method == "DGCCA") for _ in range(8)]
    best, best_state, history = np.inf, None, []
    for step in range(STEPS):
        model.train()
        loss = model.loss(sample_batch(data, "train", rng, tuples=method == "DGCCA"))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if (step + 1) % 250 == 0:
            model.eval()
            with torch.no_grad():
                torch.manual_seed(SEED + 2)
                v = float(np.mean([float(model.loss(b)) for b in vbatches]))
            history.append({"step": step + 1, "train_loss": float(loss), "validation_objective": v})
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
            log(f"  {method}: step {step + 1}/{STEPS} train {float(loss):.4f} validation {v:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def encode(method, model, data, extra):
    out = {}
    for name in DATASETS:
        out[name] = {}
        for split in splits_of(name):
            Z = data[name][split]["Z"]
            if method == "PCA (no alignment)":
                out[name][split] = extra[name].transform(Z)
                continue
            parts = []
            for s in range(0, len(Z), 4096):
                x = torch.from_numpy(Z[s:s + 4096])
                if method == "DGCCA":
                    h = model.hidden(name, x).numpy().astype(np.float64)
                    parts.append((h - extra[name]["mean"]) @ extra[name]["U"])
                else:
                    parts.append(model.encode(name, x).numpy())
            out[name][split] = np.concatenate(parts)
    sc = StandardScaler().fit(np.vstack([out[n]["train"] for n in DATASETS]))
    return {n: {s: sc.transform(out[n][s]).astype(np.float32) for s in splits_of(n)} for n in DATASETS}


@torch.no_grad()
def gcca_projection(model, data, rng, n_tuples=40):
    """Global MAXVAR GCCA on training tuples: G = top-DIM eigvectors of sum_d P_d; U_d = (H_d'H_d + rI)^-1 H_d' G."""
    hs = {n: [] for n in DATASETS}
    for _ in range(n_tuples):
        b = sample_batch(data, "train", rng, tuples=True)
        for n in DATASETS:
            hs[n].append(model.hidden(n, b[n][0]).numpy().astype(np.float64))
    H = {n: np.vstack(v) for n, v in hs.items()}
    means = {n: h.mean(0) for n, h in H.items()}
    Hc = {n: H[n] - means[n] for n in DATASETS}
    r = 1e-2 * len(next(iter(Hc.values())))
    M = sum(h @ np.linalg.solve(h.T @ h + r * np.eye(h.shape[1]), h.T) for h in Hc.values())
    vals, vecs = np.linalg.eigh(M)
    G = vecs[:, np.argsort(vals)[::-1][:DIM]]
    return {n: {"mean": means[n], "U": np.linalg.solve(Hc[n].T @ Hc[n] + r * np.eye(Hc[n].shape[1]), Hc[n].T @ G)} for n in DATASETS}


# --------------------------------------------------------------------------- metrics
def linear_cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    num = np.linalg.norm(a.T @ b, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / den) if den > 0 else np.nan


def centroids(Z, keys, min_rows=20):
    return {k: Z[keys == k].mean(0) for k in np.unique(keys) if k != "NA" and (keys == k).sum() >= min_rows}


def evaluate(Zs, data, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    res = {"per_dataset": {}}
    k = len(DATASETS)
    va_real = {n: real(data[n], "validation") for n in DATASETS}
    cents = {n: centroids(Zs[n]["validation"][va_real[n]], data[n]["strata"]["validation"][va_real[n]]) for n in DATASETS}
    cents_as = {n: centroids(Zs[n]["validation"][va_real[n]], data[n]["strata_as"]["validation"][va_real[n]]) for n in DATASETS}
    cka = np.full((k, k), np.nan)
    support = np.zeros((k, k), int)
    for i, a in enumerate(DATASETS):
        for j, b in enumerate(DATASETS):
            ca, cb = (cents_as[a], cents_as[b]) if (a in NO_BMI or b in NO_BMI) else (cents[a], cents[b])
            keys = sorted(set(ca) & set(cb))
            support[i, j] = len(keys)
            if len(keys) >= 8:
                cka[i, j] = linear_cka(np.array([ca[q] for q in keys]), np.array([cb[q] for q in keys]))
    res["cka_matrix"], res["cka_support"] = cka.round(4).tolist(), support.tolist()
    res["cka_strata"] = "age group x sex x BMI category; pairs with the cross-hospital ICU view use age group x sex (it has no BMI)"
    res["CKA (alignment)"] = float(np.nanmean(cka[np.triu_indices(k, 1)]))
    f1s, sils = [], []
    for n in DATASETS:
        d = data[n]
        tr, va = real(d, "train"), va_real[n]
        ids = rng.choice(int(va.sum()), min(2000, int(va.sum())), replace=False)
        parts = []
        for yt, yv in zip(label_columns(d["train"]["y"][tr]), label_columns(d["validation"]["y"][va])):
            probe = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Zs[n]["train"][tr], yt)
            p = probe.predict_proba(Zs[n]["validation"][va])[:, 1]
            sil = silhouette_score(Zs[n]["validation"][va][ids], yv[ids]) if len(np.unique(yv[ids])) > 1 else np.nan
            parts.append((float(f1_score(yv, p >= 0.5, average="macro")), float(roc_auc_score(yv, p)), float(sil)))
        f1, auc, sil = (float(np.nanmean([q[i] for q in parts])) for i in range(3))
        res["per_dataset"][n] = {"macro_f1": f1, "auroc": auc, "silhouette": sil}
        if len(parts) > 1:                       # multi-label view: the mean above is over its recorded conditions
            res["per_dataset"][n]["per_condition"] = {c: {"macro_f1": q[0], "auroc": q[1], "silhouette": q[2]}
                                                      for c, q in zip(HOSPITAL_LABELS, parts)}
        f1s.append(f1); sils.append(sil)
    res["Macro-F1 (preservation)"], res["Silhouette (structure)"] = float(np.mean(f1s)), float(np.nanmean(sils))
    m = 2000
    tr = np.vstack([Zs[n]["train"][rng.choice(len(Zs[n]["train"]), m, replace=False)] for n in DATASETS])
    va_ids = {n: rng.choice(np.flatnonzero(va_real[n]), min(m, int(va_real[n].sum())), replace=False) for n in DATASETS}
    va = np.vstack([Zs[n]["validation"][va_ids[n]] for n in DATASETS])
    lab = np.concatenate([np.full(len(va_ids[n]), i) for i, n in enumerate(DATASETS)])
    km = KMeans(k, n_init=10, random_state=SEED).fit(tr)
    cl = km.predict(va)
    res["NMI (domain)"], res["ARI (domain)"] = float(normalized_mutual_info_score(lab, cl)), float(adjusted_rand_score(lab, cl))
    res["contingency"] = pd.crosstab(lab, cl).reindex(index=range(k), columns=range(k), fill_value=0).to_numpy().tolist()
    proj = PCA(2, random_state=SEED).fit(va)
    pts = proj.transform(va)
    sel = np.concatenate([np.flatnonzero(lab == i)[:300] for i in range(k)])
    any_label = lambda y: y if y.ndim == 1 else y.max(1)           # multi-label view: any recorded condition
    yv_all = np.concatenate([any_label(data[n]["validation"]["y"][va_ids[n]]) for n in DATASETS])
    res["scatter"] = {"x": pts[sel, 0].round(3).tolist(), "y": pts[sel, 1].round(3).tolist(),
                      "dataset": lab[sel].tolist(), "label": yv_all[sel].tolist()}
    return res


def uncompressed_f1(data):
    out = {}
    for n in DATASETS:
        d = data[n]
        tr, va = real(d, "train"), real(d, "validation")
        scores = []
        for yt, yv in zip(label_columns(d["train"]["y"][tr]), label_columns(d["validation"]["y"][va])):
            probe = LogisticRegression(class_weight="balanced", max_iter=2000).fit(d["train"]["Z"][tr], yt)
            scores.append(f1_score(yv, probe.predict_proba(d["validation"]["Z"][va])[:, 1] >= 0.5, average="macro"))
        out[n] = float(np.mean(scores))
    return out


# --------------------------------------------------------------------------- main
def clear_outputs(path):
    """Remove a regenerated output folder. On Windows another program (an editor's file watcher, antivirus) can hold a
    sub-folder open, and on exFAT drives that folder then stays 'delete pending' until the program lets go. Such
    leftovers are reported and skipped instead of stopping the run; every output is rewritten below anyway."""
    held = []
    shutil.rmtree(path, onexc=lambda func, p, err: held.append(str(p)))
    if held:
        print(f"note: left in place (held open by another program): {', '.join(sorted(set(held)))}")


def method_folder(method):
    """methods/<name>: the full method name, e.g. 'PCA (no alignment)' -> 'PCA_no_alignment'."""
    return OUT / "methods" / "_".join("".join(ch if ch.isalnum() or ch == "-" else " " for ch in method).split())


def main():
    for sub in ("data", "methods", "plots"):                 # regenerate outputs only; code files stay
        if (OUT / sub).exists():
            clear_outputs(OUT / sub)
    for f in ("run_log.txt", "latent_method_comparison.csv", "latent_report.json", "latent_scatter.json"):
        (OUT / f).unlink(missing_ok=True)
    OUT.mkdir(exist_ok=True)
    log = Logger(OUT / "run_log.txt")
    torch.set_num_threads(8)
    data = load()
    log("loaded " + ", ".join(f"{n}: {len(data[n]['features'])} features, train {len(data[n]['train']['y']):,}" for n in DATASETS))
    results, latents, histories = {}, {}, {}
    for method in METHODS:
        t0 = time.time()
        if method == "PCA (no alignment)":
            model = None
            extra = {n: PCA(DIM, random_state=SEED).fit(data[n]["train"]["Z"]) for n in DATASETS}
        else:
            model, histories[method] = train(method, data, log)
            extra = gcca_projection(model, data, np.random.default_rng(SEED + 3)) if method == "DGCCA" else None
        Zs = encode(method, model, data, extra)
        latents[method] = Zs
        results[method] = evaluate(Zs, data)
        r = results[method]
        log(f"{method}: CKA {r['CKA (alignment)']:.3f} F1 {r['Macro-F1 (preservation)']:.3f} silhouette {r['Silhouette (structure)']:.3f} "
            f"NMI {r['NMI (domain)']:.3f} ARI {r['ARI (domain)']:.3f} ({time.time() - t0:.0f}s)")
        folder = method_folder(method)
        folder.mkdir(parents=True, exist_ok=True)
        if model is not None:
            torch.save(model.state_dict(), folder / "encoder.pt")
        write_json(folder / "metrics.json", {k: v for k, v in r.items() if k != "scatter"} | {"history": histories.get(method)})
    table = pd.DataFrame({m: {k: results[m][k] for k in DIRECTIONS} for m in METHODS}).T
    table.insert(0, "family", [FAMILY[m] for m in table.index])
    table["mean_rank"] = mean_rank(table, DIRECTIONS)
    candidates = table.drop(index="PCA (no alignment)").sort_values("mean_rank", kind="stable")
    selected = candidates.index[0]
    table.to_csv(OUT / "latent_method_comparison.csv", index_label="method")
    data_dir = OUT / "data"
    data_dir.mkdir(exist_ok=True)
    cols = [f"z{i + 1:02d}" for i in range(DIM)]
    def latent_frame(n, split):
        meta = data[n][split]["meta"].reset_index(drop=True)
        keep = [c for c in meta.columns if (c.startswith(("meta_", "sens_", "target")) or c == "sample_weight") and c != "meta_dataset"]
        f = pd.concat([pd.DataFrame(latents[selected][n][split], columns=cols), meta[keep]], axis=1)
        f.insert(0, "dataset", n)
        return f

    for split in SPLITS:
        frames = []
        for n in DATASETS:
            f = latent_frame(n, split)
            f.to_csv(data_dir / f"{n}_latent_{split}.csv", index=False)
            frames.append(f)
        # one file for all five datasets; single-label datasets use `target`, the cross-hospital view its four target_* columns
        pd.concat(frames, ignore_index=True).to_csv(data_dir / f"combined_latent_{split}.csv", index=False)
    for n, extra_splits in EXTRA_SPLITS.items():          # MIMIC external hospital: encoded only, kept out of the combined files
        for split in extra_splits:
            latent_frame(n, split).to_csv(data_dir / f"{n}_latent_{split}.csv", index=False)
    unc = uncompressed_f1(data)
    write_json(OUT / "latent_report.json", {"selected": selected, "rule": "best mean rank over CKA, Macro-F1, Silhouette, NMI, ARI "
               "(PCA baseline excluded because it has no shared space)", "dimension": DIM, "datasets": DATASETS,
               "uncompressed_macro_f1": unc, "methods": {m: {k: v for k, v in results[m].items() if k != "scatter"} for m in METHODS},
               "rows": {n: {s: int(len(data[n][s]["y"])) for s in splits_of(n)} for n in DATASETS}})
    # the same validation records (300 per dataset) projected to 2-D in every method's space (website animation)
    write_json(OUT / "latent_scatter.json", {"datasets": DATASETS, "selected": selected,
                                             "methods": {m: results[m]["scatter"] for m in METHODS}})
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from latent_dashboard import build
    build(results, table, selected, unc, NOTES, DIRECTIONS)
    log(f"selected {selected}; outputs in {data_dir}")


if __name__ == "__main__":
    main()
