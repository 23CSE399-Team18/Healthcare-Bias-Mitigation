# %% [markdown]
# # __TITLE__ - in-processing bias mitigation
#
# **Input:** this disease's rows in the common latent space (`latent/data/__DATASET___latent_<split>.csv`, 16 latent features
# `z01..z16`, label, sample weight, fairness groups) and, for the cross-hospital bias, the cross-hospital ICU cohort in the same
# latent space (`latent/data/hospital_latent_<split>.csv`, eICU-CRD hospitals + MIMIC-IV, label = recorded __DATASET__ condition).
# **Output:** `__DATASET__/inprocessed/` (predictions, comparison tables, selected methods, models) and
# `__DATASET__/plots/inprocessing/` (one dashboard per in-processing bias).
#
# Method and metric table (`inprocessing.png`):
#
# | Bias problem | Methods | Metrics |
# |---|---|---|
# | Demographic prediction dependence | FairCare (adversarial fair representation), DANN, GroupDRO, Fair-ERM | CMI, PAD, WGER, EOV |
# | Intersectional / subgroup unfairness | FOMO, Intersectional debiasing, Fairness-constrained optimization, GroupDRO (intersectional) | HVI, DF-epsilon, MSFV, IWGR |
# | Predictive miscalibration | AvUC, CRL, Uncertainty-weighted gradient training, Calibration-regularized training | UCE, AURC, E-AURC, MMCE |
# | Cross-hospital generalization failure | Fishr, V-REx, EQRM, Fish | IEGVD, ERV, UQER, IDGA |
#
# **Cross-hospital data.** The cross-hospital bias needs many hospitals, so it uses the cross-hospital ICU cohort (`hospital/`):
# eICU-CRD v2.0 split by whole hospitals (141 training, 34 validation, 31 test hospitals) plus MIMIC-IV v3.1 (Beth Israel
# Deaconess, a different hospital system) kept untouched as the external hospital. Its label is the recorded __DATASET__
# condition. Environments = hospitals. Methods train on the eICU training hospitals, are selected on the eICU validation
# hospitals, and are reported on the eICU test hospitals + MIMIC. Its ERM reference is trained on the same cohort.
#
# **Protocol.** Every method trains on the training split only (synthetic training rows from pre-processing included, with
# their sample weights). Checkpoints, decision thresholds and the method choice use the validation split; the test split is
# only scored at the end. ERM (plain weighted cross-entropy, same network) is the reference ("before"). For each bias the
# method with the best mean rank over that bias's four metrics is selected ("after"), provided its validation AUROC is no more
# than 0.05 below ERM (accuracy guard). Protected attributes: sex and race/ethnicity; age is part of the intersectional groups.

# %%
from pathlib import Path
DATASET = "__DATASET__"
TITLE = "__TITLE__"
ENV_COL = "meta_domain_hospitalid"   # environment for the cross-hospital bias (cross-hospital ICU cohort)
ENV_NAME = "hospital"
HOSPITAL_LABEL = f"target_{DATASET}"    # recorded condition of this disease in the cross-hospital ICU cohort
EXTERNAL_HOSPITAL = "MIMIC-IV (BIDMC)"
ROOT = Path(r"D:/complete")
LATENT = ROOT / "latent" / "data"
OUT_DIR = ROOT / DATASET / "inprocessed"
PLOT_DIR = ROOT / DATASET / "plots" / "inprocessing"
SEED = 2026
EPOCHS = 8
BATCH = 1024
import torch
torch.set_num_threads(4)

# %%
import json, time, copy, math, warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
OUT_DIR.mkdir(parents=True, exist_ok=True); PLOT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)
PROGRESS = OUT_DIR / "run_progress.log"
PROGRESS.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start in-processing {DATASET}\n", encoding="utf-8")


def progress(msg):
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=1), encoding="utf-8")

# %% [markdown]
# ## 1. Load the latent features of this disease

# %%
SPLITS = ("train", "validation", "test")


def labels_of(df, env_col=None):
    g = {k: df[c].fillna("Unknown").astype(str).to_numpy() for k, c in
         (("sex", "sens_sex"), ("age", "sens_age_group"), ("race", "sens_race_ethnicity"))}
    g["protected"] = np.array([f"{a}|{b}" for a, b in zip(g["sex"], g["race"])])
    g["intersection"] = np.array([f"{a}|{b}|{c}" for a, b, c in zip(g["sex"], g["age"], g["race"])])
    if env_col:
        env = df[env_col].astype(str).to_numpy()
        if "meta_source" in df:                                  # the external hospital system is one environment
            env = np.where(df["meta_source"].astype(str).str.lower().to_numpy() == "mimic", EXTERNAL_HOSPITAL, env)
        g["env"] = env
    return g


def known(v):
    return not any(t in ("Unknown", "nan", "<NA>", "None", "") for t in str(v).split("|"))


def code_groups(groups, y, minimum=50):
    """Integer codes for training groups with >= `minimum` rows (others = -1)."""
    expanded = dict(groups)
    expanded["protected_label"] = np.array([f"{a}|{t}" for a, t in zip(groups["protected"], y)])
    expanded["intersection_label"] = np.array([f"{a}|{t}" for a, t in zip(groups["intersection"], y)])
    codes, levels = {}, {}
    for key, values in expanded.items():
        counts = pd.Series(values).value_counts()
        keep = sorted(k for k, n in counts.items() if n >= minimum and known(k))
        m = {v: i for i, v in enumerate(keep)}
        codes[key] = np.array([m.get(v, -1) for v in values], np.int64)
        levels[key] = keep
    return codes, levels


def make_bundle(frames, label_col, env_col=None):
    """Latent features standardised on the bundle's training rows, label, weights, groups and group codes."""
    zc = [c for c in frames["train"].columns if c.startswith("z") and c[1:].isdigit()]
    X = {s: f[zc].to_numpy(np.float32) for s, f in frames.items()}
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd[sd < 1e-6] = 1
    X = {s: ((v - mu) / sd).astype(np.float32) for s, v in X.items()}
    Y = {s: f[label_col].to_numpy().astype(int) for s, f in frames.items()}
    W = {s: f["sample_weight"].to_numpy(np.float32) if "sample_weight" in f else np.ones(len(f), np.float32) for s, f in frames.items()}
    W["train"] = W["train"] / W["train"].mean()
    G = {s: labels_of(f, env_col) for s, f in frames.items()}
    codes, levels = code_groups(G["train"], Y["train"])
    return {"DATA": frames, "ZCOLS": zc, "X": X, "Y": Y, "W": W, "G": G, "CODES": codes, "LEVELS": levels}


DIS = make_bundle({s: pd.read_csv(LATENT / f"{DATASET}_latent_{s}.csv", low_memory=False) for s in SPLITS}, "target")
HOS = make_bundle({s: pd.read_csv(LATENT / f"hospital_latent_{s}.csv", low_memory=False) for s in SPLITS + ("external",)},
                  HOSPITAL_LABEL, ENV_COL)
DATA, ZCOLS, X, Y, W, G, CODES, LEVELS = (DIS[k] for k in ("DATA", "ZCOLS", "X", "Y", "W", "G", "CODES", "LEVELS"))
ID_COL = "meta_patientunitstayid" if "meta_patientunitstayid" in DATA["train"] else "meta_record_id"
print(DATASET, {s: (len(Y[s]), round(float(Y[s].mean()), 4)) for s in SPLITS}, "| latent features:", len(ZCOLS))
print("training groups:", {k: len(v) for k, v in LEVELS.items()})
print("cross-hospital cohort,", HOSPITAL_LABEL, {s: (len(HOS["Y"][s]), round(float(HOS["Y"][s].mean()), 4)) for s in HOS["Y"]},
      "| hospitals:", {s: len(set(HOS["G"][s]["env"])) for s in HOS["G"]}, "| training hospitals with >= 50 rows:", len(HOS["LEVELS"]["env"]))

# %% [markdown]
# ## 2. Metrics (exactly the metrics of the in-processing table)
#
# Demographic dependence (protected attributes sex, race/ethnicity; groups with >= 30 rows):
# **CMI** = conditional mutual information I(prediction; group | label) in nats; **PAD** = proxy A-distance 2(1-2e) where e is
# the cross-validated error of a linear classifier separating two groups from the model's last hidden representation
# (largest value over group pairs); **WGER** = worst-group excess risk (group log-loss minus overall log-loss);
# **EOV** = equalized-odds violation (largest TPR or FPR gap). Lower is better for all four.
#
# Intersectional (sex x age x race groups with >= 30 rows): **HVI** = hypervolume of the (error = 1 - balanced accuracy,
# MSFV) points the method reaches over its training epochs, reference point (1, 1), higher is better; **DF-epsilon** =
# differential fairness (largest log-ratio of smoothed positive / negative prediction rates between groups); **MSFV** = maximum
# subgroup fairness violation (group share x |group positive rate - overall positive rate|); **IWGR** = largest group log-loss.
#
# Calibration (decision at 0.5 for the uncertainty metrics): **UCE** = uncertainty calibration error (10 bins of normalised
# entropy; |error rate - mean uncertainty|); **AURC** = area under the risk-coverage curve (rows ordered by confidence);
# **E-AURC** = AURC minus the optimal AURC for the same error rate; **MMCE** = maximum mean calibration error (Laplacian kernel,
# width 0.4). Lower is better for all four.
#
# Cross-hospital (environments = hospitals with >= 30 rows; for the final check the eICU test hospitals plus the MIMIC
# external hospital as one more environment): last-layer per-row gradients g = (p - y)[h, 1];
# **IEGVD** = mean squared deviation of each environment's gradient variance from the average (Fishr's quantity);
# **ERV** = variance of environment risks (V-REx); **UQER** = 0.9-quantile of environment risks (EQRM);
# **IDGA** = mean cosine similarity between environments' mean gradients (Fish), higher is better.

# %%
def supported(values, y, minimum=30):
    s = pd.Series(values)
    return [g for g, n in s.value_counts().items() if n >= minimum and known(g)]


def group_logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def threshold_for(y, p):
    grid = np.arange(0.05, 0.951, 0.025)
    return float(max(grid, key=lambda t: (f1_score(y, p >= t, average="macro", zero_division=0), -abs(t - 0.5))))


def cmi(yhat, a, y, groups):
    total = 0.0
    keep = np.isin(a, groups)
    for t in (0, 1):
        m = keep & (y == t)
        if m.sum() == 0:
            continue
        joint = pd.crosstab(yhat[m], a[m]).to_numpy() / m.sum()
        outer = np.outer(joint.sum(1), joint.sum(0))
        nz = joint > 0
        total += (m.sum() / keep.sum()) * float((joint[nz] * np.log(joint[nz] / outer[nz])).sum())
    return total


def pad(h, a, groups, n=1500):
    rng = np.random.default_rng(SEED)
    best = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ia, ib = np.flatnonzero(a == groups[i]), np.flatnonzero(a == groups[j])
            k = min(n, len(ia), len(ib))
            if k < 30:
                continue
            idx = np.r_[rng.choice(ia, k, replace=False), rng.choice(ib, k, replace=False)]
            lab = np.r_[np.zeros(k), np.ones(k)]
            pred = cross_val_predict(LogisticRegression(max_iter=500), h[idx], lab,
                                     cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
            err = float((pred != lab).mean())
            best = max(best, 2 * (1 - 2 * min(err, 0.5)))
    return best


def rates(y, yhat, mask):
    pos, neg = mask & (y == 1), mask & (y == 0)
    tpr = yhat[pos].mean() if pos.sum() >= 10 else np.nan
    fpr = yhat[neg].mean() if neg.sum() >= 10 else np.nan
    return tpr, fpr


def eov(y, yhat, a, groups):
    tprs, fprs = [], []
    for g in groups:
        t, f = rates(y, yhat, a == g)
        tprs.append(t); fprs.append(f)
    gap = lambda v: np.nanmax(v) - np.nanmin(v) if np.isfinite(v).sum() >= 2 else np.nan
    return float(np.nanmax([gap(np.array(tprs)), gap(np.array(fprs))]))


def hypervolume(points, ref=(1.0, 1.0)):
    pts = sorted((x, y) for x, y in points if np.isfinite(x) and np.isfinite(y) and x <= ref[0] and y <= ref[1])
    front, best_y = [], np.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y)); best_y = y
    hv = 0.0
    for i, (x, y) in enumerate(front):
        nxt = front[i + 1][0] if i + 1 < len(front) else ref[0]
        hv += (nxt - x) * (ref[1] - y)
    return float(hv)


def msfv(yhat, inter, groups):
    overall = yhat.mean()
    return float(max(((inter == g).mean() * abs(yhat[inter == g].mean() - overall) for g in groups), default=np.nan))


def df_epsilon(yhat, inter, groups, alpha=1.0):
    eps = 0.0
    probs = [((yhat[inter == g] == 1).sum() + alpha) / ((inter == g).sum() + 2 * alpha) for g in groups]
    for i in range(len(probs)):
        for j in range(len(probs)):
            if i != j:
                eps = max(eps, abs(math.log(probs[i]) - math.log(probs[j])), abs(math.log(1 - probs[i]) - math.log(1 - probs[j])))
    return float(eps)


def calibration_metrics(y, p, n_mmce=3000):
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(float)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    unc = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)) / np.log(2)
    bins = np.minimum((unc * 10).astype(int), 9)
    uce = sum((bins == b).mean() * abs(err[bins == b].mean() - unc[bins == b].mean()) for b in range(10) if (bins == b).any())
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    aurc = float(risk.mean())
    r = err.mean()
    optimal = r + (1 - r) * np.log(1 - r) if 0 < r < 1 else 0.0
    rng = np.random.default_rng(SEED)
    ids = rng.choice(len(y), min(n_mmce, len(y)), replace=False)
    c, rr = 1 - err[ids], conf[ids]
    k = np.exp(-np.abs(rr[:, None] - rr[None, :]) / 0.4)
    d = (c - rr)
    mmce = float(np.sqrt(max((d[:, None] * d[None, :] * k).sum() / len(ids) ** 2, 0.0)))
    return {"UCE": float(uce), "AURC": aurc, "E-AURC": float(aurc - optimal), "MMCE": mmce,
            "_risk_coverage": risk[np.linspace(0, len(risk) - 1, 60).astype(int)].tolist()}


def environment_metrics(y, p, h, env, minimum=30):
    envs = supported(env, y, minimum)
    if len(envs) < 2:
        return {"IEGVD": np.nan, "ERV": np.nan, "UQER": np.nan, "IDGA": np.nan, "_env": {}}
    grads = (p - y)[:, None] * np.c_[h, np.ones(len(h))]
    means, variances, risks, info = [], [], [], {}
    for e in envs:
        m = env == e
        means.append(grads[m].mean(0)); variances.append(grads[m].var(0))
        risks.append(group_logloss(y[m], p[m]))
        auc = roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) == 2 else np.nan
        info[e] = {"rows": int(m.sum()), "risk": risks[-1], "auroc": auc}
    V = np.array(variances)
    iegvd = float(((V - V.mean(0)) ** 2).sum(1).mean())
    M = np.array(means)
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    cos = (M / np.maximum(norm, 1e-12)) @ (M / np.maximum(norm, 1e-12)).T
    idga = float(cos[np.triu_indices(len(envs), 1)].mean())
    risks = np.array(risks)
    return {"IEGVD": iegvd, "ERV": float(risks.var()), "UQER": float(np.quantile(risks, 0.9)), "IDGA": idga, "_env": info}


FAMILIES = {
    "demographic": {"title": "Demographic prediction dependence",
                    "methods": ["FairCare", "DANN", "GroupDRO", "Fair-ERM"],
                    "metrics": {"CMI": "min", "PAD": "min", "WGER": "min", "EOV": "min"}},
    "intersectional": {"title": "Intersectional / subgroup unfairness",
                       "methods": ["FOMO", "Intersectional debiasing", "Fairness-constrained optimization", "GroupDRO (intersectional)"],
                       "metrics": {"HVI": "max", "DF-epsilon": "min", "MSFV": "min", "IWGR": "min"}},
    "calibration": {"title": "Predictive miscalibration",
                    "methods": ["AvUC", "CRL", "Uncertainty-weighted gradient", "Calibration-regularized"],
                    "metrics": {"UCE": "min", "AURC": "min", "E-AURC": "min", "MMCE": "min"}},
    "environment": {"title": "Cross-hospital generalization failure",
                    "methods": ["Fishr", "V-REx", "EQRM", "Fish"],
                    "metrics": {"IEGVD": "min", "ERV": "min", "UQER": "min", "IDGA": "max"}},
}


def family_metrics(family, y, p, h, groups, threshold, points=None):
    yhat = (p >= threshold).astype(int)
    if family == "demographic":
        out = {"CMI": 0.0, "PAD": 0.0, "WGER": -np.inf, "EOV": 0.0}
        base = group_logloss(y, p)
        for key in ("sex", "race"):
            gs = supported(groups[key], y)
            if len(gs) < 2:
                continue
            out["CMI"] = max(out["CMI"], cmi(yhat, groups[key], y, gs))
            out["PAD"] = max(out["PAD"], pad(h, groups[key], gs)) if h is not None else np.nan
            out["WGER"] = max(out["WGER"], max(group_logloss(y[groups[key] == g], p[groups[key] == g]) - base for g in gs))
            out["EOV"] = max(out["EOV"], eov(y, yhat, groups[key], gs))
        return out
    if family == "intersectional":
        gs = supported(groups["intersection"], y)
        return {"HVI": hypervolume(points) if points else np.nan,
                "DF-epsilon": df_epsilon(yhat, groups["intersection"], gs),
                "MSFV": msfv(yhat, groups["intersection"], gs),
                "IWGR": max((group_logloss(y[groups["intersection"] == g], p[groups["intersection"] == g]) for g in gs), default=np.nan)}
    if family == "calibration":
        return calibration_metrics(y, p)
    return environment_metrics(y, p, h, groups["env"])


def common_metrics(y, p, threshold):
    yhat = p >= threshold
    return {"AUROC": float(roc_auc_score(y, p)), "Balanced accuracy": float(balanced_accuracy_score(y, yhat)),
            "Macro-F1": float(f1_score(y, yhat, average="macro", zero_division=0)), "Log-loss": group_logloss(y, p),
            "Brier": float(np.mean((p - y) ** 2)), "threshold": threshold}

# %% [markdown]
# ## 3. Methods (one network for all: latent -> 32 -> 16 -> logit; the method changes the training objective)
#
# FairCare: label-conditioned adversaries recover sex and race from the hidden representation through gradient reversal
# (adversarial fair representation learning, equalized-odds form). DANN: domain-adversarial network with the protected
# groups as domains and the standard lambda schedule. GroupDRO: exponentiated-gradient weights on (sex x race x label) group
# risks. Fair-ERM: risk plus a conditional (label-wise) group-mean penalty. FOMO: Pareto search over intersectional
# group weights of a logistic model (error vs MSFV), refit on all training rows. Intersectional debiasing: adversary on
# sex x age x race groups. Fairness-constrained optimization: Lagrangian dual ascent on the intersectional penalty
# (target 0.0025). GroupDRO (intersectional): group = sex x age x race x label. AvUC: accuracy-versus-uncertainty
# calibration loss. CRL: correctness ranking loss using each row's correctness history. Uncertainty-weighted gradient:
# per-row weights that grow with predictive uncertainty / error. Calibration-regularized: cross-entropy plus a
# differentiable MMCE penalty. Fishr: matching of per-environment last-layer gradient variances. V-REx: risk mean plus
# variance of environment risks. EQRM: 0.9-quantile of environment risks. Fish: first-order inter-environment gradient
# matching (Reptile-style meta-step after every batch).

# %%
class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class Network(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.out = nn.Linear(16, 1)
        self.adv = nn.ModuleDict({k: nn.Sequential(nn.Linear(16 + extra, 16), nn.ReLU(), nn.Linear(16, n))
                                  for k, (n, extra) in heads.items()})

    def forward(self, x):
        h = self.body(x)
        return self.out(h).squeeze(-1), h


def group_means(values, codes, minimum=4):
    labels = [int(g) for g in torch.unique(codes) if g >= 0 and (codes == g).sum() >= minimum]
    return labels, (torch.stack([values[codes == g].mean(0) for g in labels]) if labels else None)


def soft_eo(p, y, code_list):
    penalties = []
    for codes in code_list:
        for t in (0, 1):
            m = y == t
            _, means = group_means(p[m], codes[m])
            if means is not None and len(means) > 1:
                penalties.append(means.var(unbiased=False))
    return torch.stack(penalties).mean() if penalties else p.sum() * 0


def mmce_penalty(p, y):
    conf = torch.maximum(p, 1 - p)
    correct = ((p >= 0.5).float() == y).float()
    d = correct - conf
    k = torch.exp(-torch.abs(conf[:, None] - conf[None, :]) / 0.4)
    return torch.sqrt(torch.clamp((d[:, None] * d[None, :] * k).mean(), min=1e-12))


ADVERSARY_HEADS = {"FairCare": {"sex": ("sex", 1), "race": ("race", 1)}, "DANN": {"protected": ("protected", 0)},
                   "Intersectional debiasing": {"intersection": ("intersection", 0)}}


def loss_for(method, model, x, y, w, g, state, ix, epoch, progress_frac):
    logits, h = model(x)
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    base = (bce * w).sum() / w.sum()
    if method in ADVERSARY_HEADS:
        lam = 0.3 if method != "DANN" else 0.3 * (2 / (1 + math.exp(-10 * progress_frac)) - 1)
        extra = []
        for head, (key, label_input) in ADVERSARY_HEADS[method].items():
            m = g[key] >= 0
            if m.sum() > 1:
                z = Reverse.apply(h[m], lam)
                if label_input:
                    z = torch.cat([z, y[m, None]], 1)
                extra.append(F.cross_entropy(model.adv[head](z), g[key][m]))
        return base + torch.stack(extra).mean() if extra else base
    if method in ("GroupDRO", "GroupDRO (intersectional)"):
        key = "protected_label" if method == "GroupDRO" else "intersection_label"
        labels, risks = group_means(bce, g[key])
        if risks is None:
            return base
        q = state["q"]
        q[labels] *= torch.exp(0.01 * risks.detach())
        q /= q.sum()
        wts = q[labels] / q[labels].sum()
        return 0.1 * base + 0.9 * (wts * risks).sum()
    if method == "Fair-ERM":
        return base + 10 * soft_eo(p, y, [g["sex"], g["race"]])
    if method == "Fairness-constrained optimization":
        pen = soft_eo(p, y, [g["intersection"]])
        state["dual"] = max(0.0, min(100.0, state["dual"] + 0.1 * (pen.detach().item() - 0.0025)))
        return base + state["dual"] * (pen - 0.0025)
    if method == "Calibration-regularized":
        return base + 2.0 * mmce_penalty(p, y)
    if method == "Uncertainty-weighted gradient":
        return ((bce * w) * (0.05 + 2 * (p.detach() - y) ** 2)).sum() / w.sum()
    if method == "CRL":
        conf = torch.maximum(p, 1 - p)
        hist = state["correct"][ix] / max(epoch, 1)
        other = torch.roll(torch.arange(len(y)), 1)
        sign = torch.sign(hist - hist[other])
        m = sign != 0
        rank = F.relu(0.05 - sign * (conf - conf[other]))[m].mean() if m.any() else base * 0
        state["correct"][ix] += (p.detach().ge(0.5) == y.bool()).float()
        return base + 0.1 * rank
    if method == "AvUC":
        ent = -(p * torch.log(p + 1e-7) + (1 - p) * torch.log(1 - p + 1e-7))
        conf = torch.maximum(p, 1 - p)
        if epoch == 0:
            state["ent_sum"] += ent.detach().sum().item(); state["ent_n"] += len(y)
            return base
        acc = (p.detach() >= 0.5) == y.bool()
        certain = ent.detach() <= state["ent_sum"] / state["ent_n"]
        u = torch.tanh(ent)
        good = (conf * (1 - u) * (acc & certain) + (1 - conf) * u * (~acc & ~certain)).sum()
        bad = (conf * u * (acc & ~certain) + (1 - conf) * (1 - u) * (~acc & certain)).sum()
        return base + 0.1 * torch.log1p(bad / (good + 1e-7))
    if method in ("V-REx", "EQRM", "Fishr"):
        labels, risks = group_means(bce, g["env"])
        if risks is None or len(risks) < 2:
            return base
        if method == "V-REx":
            return risks.mean() + 10 * risks.var(unbiased=False)
        if method == "EQRM":
            return torch.quantile(risks, 0.9)
        grads = (p - y)[:, None] * torch.cat([h, torch.ones(len(h), 1)], 1)
        variances = torch.stack([grads[g["env"] == e].var(0, unbiased=False) for e in labels])
        return base + 10 * variances.var(0, unbiased=False).mean()
    return base


@torch.no_grad()
def predict(model, x):
    model.eval()
    ps, hs = [], []
    for s in range(0, len(x), 8192):
        logit, h = model(torch.from_numpy(x[s:s + 8192]))
        ps.append(torch.sigmoid(logit).numpy()); hs.append(h.numpy())
    return np.concatenate(ps), np.concatenate(hs)


def epoch_point(y, p, groups, t):
    """(error = 1 - balanced accuracy, MSFV) at threshold t (always fitted on validation) - used for HVI."""
    yhat = (p >= t).astype(int)
    gs = supported(groups["intersection"], y)
    return (1 - balanced_accuracy_score(y, yhat), msfv(yhat, groups["intersection"], gs))


def train_neural(method, bundle=None, tag=""):
    """Train one method on a data bundle (default: this disease's latent rows; HOS = cross-hospital ICU cohort)."""
    X, Y, W, G, CODES, LEVELS = ((bundle or DIS)[k] for k in ("X", "Y", "W", "G", "CODES", "LEVELS"))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    heads = {}
    for head, (key, label_input) in ADVERSARY_HEADS.get(method, {}).items():
        heads[head] = (max(1, len(LEVELS[key])), label_input)
    model = Network(X["train"].shape[1], heads)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    state = {"dual": 0.0, "correct": torch.zeros(len(Y["train"])), "ent_sum": 0.0, "ent_n": 0}
    if method.startswith("GroupDRO"):
        key = "protected_label" if method == "GroupDRO" else "intersection_label"
        state["q"] = torch.ones(max(1, len(LEVELS[key]))) / max(1, len(LEVELS[key]))
    xt, yt, wt = torch.from_numpy(X["train"]), torch.from_numpy(Y["train"].astype(np.float32)), torch.from_numpy(W["train"])
    ct = {k: torch.from_numpy(v) for k, v in CODES.items()}
    best, best_state, history, points = np.inf, None, [], {"validation": [], "test": []}
    n_batches = math.ceil(len(Y["train"]) / BATCH)
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(Y["train"]))
        for b, s in enumerate(range(0, len(order), BATCH)):
            ix = torch.from_numpy(order[s:s + BATCH])
            gb = {k: v[ix] for k, v in ct.items()}
            loss = loss_for(method, model, xt[ix], yt[ix], wt[ix], gb, state, ix, epoch, (epoch * n_batches + b) / (EPOCHS * n_batches))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if method == "Fish" and "env" in gb:
                envs = [int(e) for e in torch.unique(gb["env"]) if e >= 0 and (gb["env"] == e).sum() >= 4]
                if len(envs) >= 2:
                    before = copy.deepcopy(model.state_dict())
                    inner = torch.optim.SGD(model.parameters(), lr=0.005)
                    for e in rng.choice(envs, min(4, len(envs)), replace=False):
                        m = gb["env"] == e
                        inner.zero_grad()
                        F.binary_cross_entropy_with_logits(model(xt[ix][m])[0], yt[ix][m]).backward()
                        inner.step()
                    after = model.state_dict()
                    model.load_state_dict({k: before[k] + 0.1 * (after[k] - before[k]) for k in before})
        pv, _ = predict(model, X["validation"])
        pt, _ = predict(model, X["test"])
        score = group_logloss(Y["validation"], pv)
        tv = threshold_for(Y["validation"], pv)
        points["validation"].append(epoch_point(Y["validation"], pv, G["validation"], tv))
        points["test"].append(epoch_point(Y["test"], pt, G["test"], tv))   # validation threshold; reporting only, never selection
        history.append({"epoch": epoch + 1, "validation_log_loss": score})
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "models" / f"{tag}{method.replace(' ', '_').replace('/', '_')}.pt")
    return model, history, points


def train_fomo():
    """FOMO: Pareto search over intersectional group log-weights (error vs MSFV on validation), logistic model."""
    rng = np.random.default_rng(SEED)
    labels = sorted(set(G["train"]["intersection"]))
    gid = np.array([labels.index(v) for v in G["train"]["intersection"]])
    sub = rng.choice(len(Y["train"]), min(20000, len(Y["train"])), replace=False)
    population = [np.zeros(len(labels))] + [rng.normal(0, 0.5, len(labels)) for _ in range(7)]
    pool, points = [], {"validation": [], "test": []}
    for generation in range(3):
        for wts in population:
            m = LogisticRegression(C=0.1, max_iter=400).fit(X["train"][sub], Y["train"][sub],
                                                            sample_weight=W["train"][sub] * np.exp(wts[gid[sub]]))
            pv = m.predict_proba(X["validation"])[:, 1]
            tv = threshold_for(Y["validation"], pv)
            e, f = epoch_point(Y["validation"], pv, G["validation"], tv)
            pool.append((e, f, wts.copy()))
            points["validation"].append((e, f))
            points["test"].append(epoch_point(Y["test"], m.predict_proba(X["test"])[:, 1], G["test"], tv))
        front = [a for a in pool if not any(b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1]) for b in pool)]
        population = [front[i % len(front)][2].copy() for i in range(2)] + \
                     [np.clip(front[rng.integers(len(front))][2] + rng.normal(0, 0.2, len(labels)), -2, 2) for _ in range(6)]
    best_e = min(a[0] for a in pool)
    winner = min((a for a in pool if a[0] <= best_e + 0.02), key=lambda a: (a[1], a[0]))
    model = LogisticRegression(C=0.1, max_iter=1000).fit(X["train"], Y["train"], sample_weight=W["train"] * np.exp(winner[2][gid]))
    return model, [{"generation": i + 1} for i in range(3)], points

# %% [markdown]
# ## 4. Train the reference (ERM) and every method
#
# Demographic, intersectional and calibration methods train on this disease's latent rows. The cross-hospital methods and
# their own ERM reference train on the cross-hospital ICU cohort (eICU training hospitals); MIMIC is only predicted.

# %%
RESULTS, RESULTS_H = {}, {}
DIS_METHODS = ["ERM"] + [m for f, s in FAMILIES.items() if f != "environment" for m in s["methods"]]
HOS_METHODS = ["ERM"] + FAMILIES["environment"]["methods"]
ALL_METHODS = DIS_METHODS
for method in DIS_METHODS:
    t0 = time.time()
    if method == "FOMO":
        model, history, points = train_fomo()
        out = {s: (model.predict_proba(X[s])[:, 1], None) for s in ("validation", "test")}
    else:
        model, history, points = train_neural(method)
        out = {s: predict(model, X[s]) for s in ("validation", "test")}
    threshold = threshold_for(Y["validation"], out["validation"][0])
    RESULTS[method] = {"p": {s: out[s][0] for s in out}, "h": {s: out[s][1] for s in out}, "threshold": threshold,
                       "points": points, "history": history, "seconds": round(time.time() - t0, 1)}
    progress(f"[{method}] trained in {RESULTS[method]['seconds']}s | validation AUROC "
             f"{roc_auc_score(Y['validation'], out['validation'][0]):.4f}")
for method in HOS_METHODS:
    t0 = time.time()
    model, history, points = train_neural(method, HOS, tag="hospital_")
    out = {s: predict(model, HOS["X"][s]) for s in ("validation", "test", "external")}
    threshold = threshold_for(HOS["Y"]["validation"], out["validation"][0])
    RESULTS_H[method] = {"p": {s: out[s][0] for s in out}, "h": {s: out[s][1] for s in out}, "threshold": threshold,
                         "points": points, "history": history, "seconds": round(time.time() - t0, 1)}
    progress(f"[cross-hospital {method}] trained in {RESULTS_H[method]['seconds']}s | AUROC eICU validation hospitals "
             f"{roc_auc_score(HOS['Y']['validation'], out['validation'][0]):.4f} | MIMIC "
             f"{roc_auc_score(HOS['Y']['external'], out['external'][0]):.4f}")

# %% [markdown]
# ## 5. Score every method on its bias's metrics and select the best (validation); report the test split
#
# For the cross-hospital bias the reported "test" split is the eICU test hospitals plus the MIMIC external hospital.

# %%
def family_arrays(fam, r, split):
    """Labels, probabilities, hidden layer and groups that a family is scored on."""
    if fam != "environment":
        return Y[split], r["p"][split], r["h"][split], G[split]
    if split == "validation":
        return HOS["Y"]["validation"], r["p"]["validation"], r["h"]["validation"], HOS["G"]["validation"]
    parts = ("test", "external")                                   # eICU test hospitals + MIMIC
    groups = {k: np.concatenate([HOS["G"][s][k] for s in parts]) for k in HOS["G"]["test"]}
    return (np.concatenate([HOS["Y"][s] for s in parts]), np.concatenate([r["p"][s] for s in parts]),
            np.concatenate([r["h"][s] for s in parts]), groups)


TABLES, TEST_TABLES, SELECTED, EXTRA, REF_AUC = {}, {}, {}, {}, {}
erm_auc = roc_auc_score(Y["validation"], RESULTS["ERM"]["p"]["validation"])
for fam, spec in FAMILIES.items():
    R = RESULTS_H if fam == "environment" else RESULTS
    REF_AUC[fam] = float(roc_auc_score(*family_arrays(fam, R["ERM"], "validation")[:2]))
    rows, trows = {}, {}
    for method in ["ERM"] + spec["methods"]:
        r = R[method]
        for split, store in (("validation", rows), ("test", trows)):
            y, pp, hh, gg = family_arrays(fam, r, split)
            fm = family_metrics(fam, y, pp, hh, gg, r["threshold"], r["points"][split])
            EXTRA[(fam, method, split)] = {k: v for k, v in fm.items() if k.startswith("_")}
            store[method] = {**{k: v for k, v in fm.items() if not k.startswith("_")}, **common_metrics(y, pp, r["threshold"])}
            if fam == "environment" and split == "test":
                store[method]["AUROC eICU test hospitals"] = float(roc_auc_score(HOS["Y"]["test"], r["p"]["test"]))
                store[method]["AUROC MIMIC (external)"] = float(roc_auc_score(HOS["Y"]["external"], r["p"]["external"]))
    tab = pd.DataFrame(rows).T
    ranks = pd.DataFrame({m: (-tab.loc[spec["methods"], m] if d == "max" else tab.loc[spec["methods"], m]).rank(method="average")
                          for m, d in spec["metrics"].items()})
    tab["mean rank"] = ranks.mean(axis=1)
    tab["eligible (AUROC guard)"] = [m in spec["methods"] and tab.loc[m, "AUROC"] >= REF_AUC[fam] - 0.05 for m in tab.index]
    cands = tab.loc[spec["methods"]]
    cands = cands[cands["eligible (AUROC guard)"]] if cands["eligible (AUROC guard)"].any() else cands
    SELECTED[fam] = cands["mean rank"].astype(float).fillna(np.inf).idxmin()
    TABLES[fam] = tab
    TEST_TABLES[fam] = pd.DataFrame(trows).T
    tab.to_csv(OUT_DIR / f"comparison_{fam}.csv", index_label="method")
    TEST_TABLES[fam].to_csv(OUT_DIR / f"test_metrics_{fam}.csv", index_label="method")
    progress(f"SELECTED {fam}: {SELECTED[fam]}")
    print(f"\n=== {spec['title']} (validation; ERM = reference) ===")
    print(tab[list(spec["metrics"]) + ["AUROC", "Balanced accuracy", "mean rank", "eligible (AUROC guard)"]].round(4).to_string())

# %% [markdown]
# ## 6. Selected method versus ERM on the untouched test split

# %%
summary = []
for fam, spec in FAMILIES.items():
    t = TEST_TABLES[fam]
    row = {"bias": spec["title"], "selected": SELECTED[fam],
           "data": "cross-hospital ICU cohort: eICU test hospitals + MIMIC" if fam == "environment" else f"{DATASET} test split"}
    for m in spec["metrics"]:
        row[f"{m} ERM"] = t.loc["ERM", m]; row[f"{m} selected"] = t.loc[SELECTED[fam], m]
    row["AUROC ERM"], row["AUROC selected"] = t.loc["ERM", "AUROC"], t.loc[SELECTED[fam], "AUROC"]
    if fam == "environment":
        for k in ("AUROC eICU test hospitals", "AUROC MIMIC (external)"):
            row[f"{k} ERM"], row[f"{k} selected"] = t.loc["ERM", k], t.loc[SELECTED[fam], k]
    summary.append(row)
SUMMARY = pd.DataFrame(summary).set_index("bias")
SUMMARY.to_csv(OUT_DIR / "inprocessing_summary.csv")
SUMMARY.T

# %%
pred_frames = {}
for split in ("validation", "test"):
    base = DATA[split][[c for c in DATA[split].columns if c.startswith(("meta_", "sens_")) or c == "target"]].copy()
    for method in DIS_METHODS:
        base[f"p_{method}"] = RESULTS[method]["p"][split]
    base.to_csv(OUT_DIR / f"predictions_{split}.csv", index=False)
    pred_frames[split] = base
hosp_frames = {}
for split in ("validation", "test", "external"):
    hd = HOS["DATA"][split]
    base = hd[[c for c in hd.columns if c.startswith(("meta_", "sens_"))]].copy()
    base["target"] = HOS["Y"][split]
    base["environment"] = HOS["G"][split]["env"]
    for method in HOS_METHODS:
        base[f"p_{method}"] = RESULTS_H[method]["p"][split]
    base.to_csv(OUT_DIR / f"hospital_predictions_{split}.csv", index=False)
    hosp_frames[split] = base
save_json(OUT_DIR / "selection.json", {
    "dataset": DATASET, "selected": SELECTED,
    "thresholds": {m: RESULTS[m]["threshold"] for m in DIS_METHODS},
    "thresholds_hospital": {m: RESULTS_H[m]["threshold"] for m in HOS_METHODS},
    "families": {f: {"title": s["title"], "methods": s["methods"], "metrics": s["metrics"],
                     "data": ("cross-hospital ICU cohort (eICU hospitals + MIMIC external), label " + HOSPITAL_LABEL)
                     if f == "environment" else f"{DATASET} latent rows"} for f, s in FAMILIES.items()},
    "erm_validation_auroc": erm_auc, "reference_validation_auroc": REF_AUC,
    "rule": "best mean rank over the bias metrics (validation), AUROC within 0.05 of that bias's ERM",
    "seconds": {**{m: RESULTS[m]["seconds"] for m in DIS_METHODS},
                **{f"cross-hospital {m}": RESULTS_H[m]["seconds"] for m in HOS_METHODS}},
    "test_used_for_selection": False, "external_used_for_selection": False})
rng = np.random.default_rng(SEED)
sample_ids = rng.choice(len(Y["test"]), 10, replace=False)
pred_frames["test"].iloc[sample_ids].to_csv(OUT_DIR / "sample10_test_predictions.csv", index=False)
hosp_sample = pd.concat([hosp_frames["test"].sample(5, random_state=SEED), hosp_frames["external"].sample(5, random_state=SEED)])
hosp_sample.to_csv(OUT_DIR / "sample10_hospital_predictions.csv", index=False)
print("saved predictions, selection and the 10-record samples to", OUT_DIR)

# %% [markdown]
# ## 7. Individual dashboards (one per in-processing bias): before = ERM, after = selected method (switchable)

# %%
def group_stats(y, p, t, values):
    yhat = (p >= t).astype(int)
    out = {}
    for g in supported(values, y):
        m = values == g
        tpr, fpr = rates(y, yhat, m)
        out[g] = {"n": int(m.sum()), "selection": float(yhat[m].mean()), "tpr": tpr, "fpr": fpr,
                  "risk": group_logloss(y[m], p[m]), "base_rate": float(y[m].mean())}
    return out


def reliability(y, p):
    bins = np.minimum((p * 10).astype(int), 9)
    return [{"bin": b, "n": int((bins == b).sum()), "mean_p": float(p[bins == b].mean()) if (bins == b).any() else None,
             "rate": float(y[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]


def payload_for(fam):
    spec = FAMILIES[fam]
    methods = ["ERM"] + spec["methods"]
    data = {}
    R = RESULTS_H if fam == "environment" else RESULTS
    for m in methods:
        r = R[m]
        p, t = family_arrays(fam, r, "test")[1], r["threshold"]
        d = {"test": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in TEST_TABLES[fam].loc[m].items()},
             "validation": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in TABLES[fam].loc[m].items()
                            if k != "eligible (AUROC guard)"}}
        if fam == "demographic":
            d["groups"] = {a: group_stats(Y["test"], p, t, G["test"][a]) for a in ("sex", "race", "age")}
        elif fam == "intersectional":
            d["groups"] = {"intersection": group_stats(Y["test"], p, t, G["test"]["intersection"])}
            d["points"] = [[round(a, 4), round(b, 4)] for a, b in r["points"]["validation"]]
        elif fam == "calibration":
            d["reliability"] = reliability(Y["test"], p)
            d["risk_coverage"] = EXTRA[(fam, m, "test")]["_risk_coverage"]
            d["confidence_hist"] = np.histogram(np.maximum(p, 1 - p), bins=10, range=(0.5, 1))[0].tolist()
        else:
            env = EXTRA[(fam, m, "test")].get("_env", {})
            d["env"] = dict(sorted(env.items(), key=lambda kv: -kv[1]["rows"])[:40])
        data[m] = d
    YY = HOS["Y"] if fam == "environment" else Y                  # cross-hospital: eICU hospitals + MIMIC external
    return {"dataset": DATASET, "title": TITLE, "family": fam, "family_title": spec["title"], "metrics": spec["metrics"],
            "methods": methods, "selected": SELECTED[fam], "data": data, "env_name": ENV_NAME,
            "data_note": ("Cross-hospital ICU cohort (eICU-CRD hospitals + MIMIC-IV), label: recorded " + DATASET + " condition"
                          if fam == "environment" else None),
            "rows": {s: int(len(v)) for s, v in YY.items()}, "positive": {s: float(v.mean()) for s, v in YY.items()},
            "table": {"columns": list(TABLES[fam].columns), "rows": [[m] + [None if isinstance(v, float) and not np.isfinite(v) else
                      (v if not isinstance(v, (np.bool_, bool)) else ("yes" if v else "no")) for v in TABLES[fam].loc[m].tolist()]
                      for m in TABLES[fam].index]}}


TEMPLATE = (ROOT / "code" / "inprocessing" / "dashboard_template.html").read_text(encoding="utf-8")
FILES = {"demographic": "01_demographic_prediction_dependence_dashboard.html",
         "intersectional": "02_intersectional_subgroup_unfairness_dashboard.html",
         "calibration": "03_predictive_miscalibration_dashboard.html",
         "environment": "04_cross_environment_generalization_dashboard.html"}
for fam, fname in FILES.items():
    if SELECTED.get(fam) is None:
        (PLOT_DIR / fname).unlink(missing_ok=True)
        continue
    payload = json.dumps(clean(payload_for(fam)), separators=(",", ":")).replace("</", "<\\/")
    (PLOT_DIR / fname).write_text(TEMPLATE.replace("__PAYLOAD__", payload).replace("__TITLE__", f"{TITLE} | {FAMILIES[fam]['title']}"),
                                  encoding="utf-8")
links = "".join(f"<li><a href='{FILES[f]}'>{FAMILIES[f]['title']}</a> &mdash; selected: <b>{SELECTED[f]}</b> (vs ERM)</li>"
                for f in FILES if SELECTED.get(f))
na = ""
(PLOT_DIR / "index.html").write_text(
    f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{TITLE} in-processing dashboards</title><style>body{{font-family:Segoe UI,Arial;"
    f"background:#EEF1F7;margin:0;color:#1B2333}}header{{background:linear-gradient(180deg,#0B1E3F,#132a52);color:#fff;padding:18px 28px;"
    f"text-align:center}}main{{padding:22px 28px;font-size:15px}}a{{color:#2563C8;font-weight:600}}li{{margin:8px 0}}</style></head><body>"
    f"<header><h2 style='margin:0'>{TITLE} &mdash; in-processing bias dashboards</h2><div style='color:#B9C6E6;font-size:13px'>before = ERM, "
    f"after = selected method; selection on validation, charts on the untouched test split</div></header><main><ul>{links}{na}</ul>"
    f"<p><a href='../index.html'>&larr; pre-processing dashboards</a></p></main></body></html>", encoding="utf-8")
print("dashboards ->", PLOT_DIR)
