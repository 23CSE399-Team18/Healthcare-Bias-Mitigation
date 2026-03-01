# CELL 2
from pathlib import Path
DATASET = "respiratory"
TITLE = "Respiratory ICU mortality — eICU-CRD v2.0"
DATA_FILE = Path(r"D:\complete\model_ready\respiratory_model_ready.csv")
OUT_DIR = Path(r"D:\complete\preprocessing\outputs\respiratory")
PLOT_DIR = Path(r"D:\plots\respiratory_preprocessing")
SEED = 42
FAST = False
FAST_CAP = 3000
FORCE_DETECT = False    # smoke-test switch only: run every method even if a bias is not detected
ORDINAL_MAPS = {}
NOMINAL = ["unit_type", "unit_admit_source", "hospital_admit_source", "unit_stay_type", "admission_dx_by24h"]
PROTECTED = ["sens_sex", "sens_race_ethnicity"]
AUDIT_ONLY = ["sens_age_group"]
PROTECTED_FEATURES = ["male"]
NON_ACTIONABLE = ["age", "age_over89"]
DOMAIN_COL = "meta_domain_hospitalid"
GROUP_COL = "meta_split_group_uniquepid"
ID_COLS = ["meta_patientunitstayid"]
CFG = {"bench": {"noise_detect": 60000, "domain_detect": 40000, "demog_detect": 80000, "dup_bench": 3000, "dup_train": 3000,
                 "missing": 20000, "outlier": 30000, "outlier_down": 60000, "noise": 20000, "scale": 50000,
                 "domain": 100000, "imbalance": 60000, "demographic": 60000}}

# CELL 4
import os
os.environ.setdefault("TQDM_DISABLE", "1")              # silence per-call progress bars of DiCE / CTGAN / HyperImpute
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys, json, time, math, random, warnings, itertools, re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, skew, kurtosis, spearmanr, ks_2samp
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, f1_score,
                             balanced_accuracy_score, recall_score, roc_curve)
from statsmodels.stats.stattools import medcouple

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = {}            # bias -> summary used for the final report
TIMINGS = {}
print("PyTorch", torch.__version__, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("Outputs ->", OUT_DIR, "| Dashboards ->", PLOT_DIR)


def S(key):
    """Benchmark sample size for a section (reduced in FAST mode)."""
    n = CFG["bench"][key]
    return min(n, FAST_CAP) if FAST else n


def E(n):
    """Training epochs / iterations (reduced in FAST mode)."""
    return max(1, n // 5) if FAST else n


PROGRESS = OUT_DIR / "run_progress.log"
PROGRESS.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {DATASET} (FAST={FAST})\n", encoding="utf-8")


def progress(msg):
    """Append a line to outputs/<disease>/run_progress.log so long runs can be followed live."""
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        TIMINGS[self.name] = round(time.time() - self.t, 1)
        print(f"  [{self.name}] {TIMINGS[self.name]:.1f}s")
        progress(f"[{self.name}] {TIMINGS[self.name]:.1f}s" + (f"  ERROR {a[0].__name__}: {a[1]}" if a and a[0] else ""))

# CELL 6
raw = pd.read_csv(DATA_FILE, low_memory=False)
META = [c for c in raw.columns if c.startswith("meta_")]
SENS = [c for c in raw.columns if c.startswith("sens_")]
FEATS_RAW = [c for c in raw.columns if c not in META + SENS + ["target"]]


def encode_features(df):
    out = {}
    for c in FEATS_RAW:
        if c in ORDINAL_MAPS:
            out[c] = df[c].map(ORDINAL_MAPS[c]).astype(float)
        elif c in NOMINAL:
            cats = sorted(df[c].dropna().astype(str).unique())
            miss = df[c].isna()
            for k in cats:
                col = (df[c].astype(str) == k).astype(float)
                col[miss] = np.nan
                out[f"{c}={k}"] = col
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return pd.DataFrame(out, index=df.index)


def build_schema(X):
    schema = {}
    for c in X.columns:
        base = c.split("=")[0]
        if "=" in c and base in NOMINAL:
            schema[c] = {"type": "onehot", "group": base}
            continue
        v = X[c].dropna().unique()
        if len(v) <= 2 and set(np.round(v, 6)) <= {0.0, 1.0}:
            t = "bin"
        elif len(v) <= 15 and np.allclose(v, np.round(v)):
            t = "ord"
        else:
            t = "cont"
        schema[c] = {"type": t}
        if t in ("bin", "ord"):
            schema[c]["allowed"] = sorted(float(a) for a in v)
    return schema


Xenc = encode_features(raw)
DATA = pd.concat([Xenc, raw[SENS + META + ["target"]]], axis=1)
DATA["target"] = DATA["target"].astype(int)
if FAST:   # smoke test only: work on a random 20,000-row subset
    DATA = DATA.sample(n=min(len(DATA), 20000), random_state=SEED).reset_index(drop=True)
    Xenc = DATA[list(Xenc.columns)]
FEATS = list(Xenc.columns)
SCHEMA = build_schema(Xenc)
ONEHOT_GROUPS = sorted({s["group"] for s in SCHEMA.values() if s["type"] == "onehot"})


WIDE = len(FEATS) > 60     # wide data (e.g. ICU): per-column learners use lighter settings to stay tractable


def cols_of(*types, feats=None):
    feats = FEATS if feats is None else feats
    return [c for c in feats if SCHEMA.get(c, {}).get("type") in types]


print(f"{len(DATA):,} rows | {len(FEATS)} encoded predictors | positive rate {DATA.target.mean():.3f}"
      + (" | wide data: lighter per-column learner settings" if WIDE else ""))
print(pd.Series({t: len(cols_of(t)) for t in ["cont", "ord", "bin", "onehot"]}, name="predictor types").to_frame().T)
print("Sensitive attributes:", SENS)
print("Domain column:", DOMAIN_COL, "| patient group column:", GROUP_COL)


def snap(df, feats=None):
    """Project generated / imputed values back onto valid values: binary and ordinal
    columns snap to their nearest allowed level, one-hot groups to a single 1."""
    df = df.copy()
    feats = FEATS if feats is None else feats
    for c in feats:
        s = SCHEMA.get(c, {})
        if s.get("type") in ("bin", "ord"):
            allowed = np.asarray(s["allowed"], float)
            v = df[c].to_numpy(float)
            m = ~np.isnan(v)
            if m.any():
                v[m] = allowed[np.abs(v[m, None] - allowed[None, :]).argmin(1)]
            df[c] = v
    for g in ONEHOT_GROUPS:
        cols = [c for c in feats if SCHEMA.get(c, {}).get("group") == g]
        if not cols:
            continue
        V = df[cols].to_numpy(float)
        m = ~np.isnan(V).any(1)
        if m.any():
            am = V[m].argmax(1)
            V[m] = 0.0
            V[np.where(m)[0], am] = 1.0
        df[cols] = V
    return df

# CELL 8
class Standardizer:
    def fit(self, A):
        A = np.asarray(A, float)
        self.mu = np.nan_to_num(np.nanmean(A, 0))
        sd = np.nanstd(A, 0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
        self.sd = sd
        return self

    def transform(self, A):
        return (np.asarray(A, float) - self.mu) / self.sd

    def inverse(self, Z):
        return np.asarray(Z, float) * self.sd + self.mu


def strat_sample(df, n, seed=SEED):
    if len(df) <= n:
        return df.copy()
    idx, _ = train_test_split(df.index, train_size=n, stratify=df["target"], random_state=seed)
    return df.loc[idx].copy()


def hgb(seed=SEED, iters=250):
    return HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(min(max(thr[j], 1e-6), 1 - 1e-6))


def fit_score(Xtr, ytr, Xte, w=None, seed=SEED, real=None, val=None):
    """HGB fitted on 80% of the training rows; decision threshold = Youden J on the other 20%.
    `real` marks genuine (non-synthetic) rows: the threshold is tuned on genuine rows only, because a
    threshold tuned on generated rows does not transfer to real patients.
    Returns test probabilities, the threshold and the fitted model."""
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ytr = np.asarray(ytr).astype(int)
    if val is not None:      # untouched validation patients: fit on all training rows, tune the threshold there
        Xv, yv = np.asarray(val[0], float), np.asarray(val[1]).astype(int)
        m = hgb(seed)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte)[:, 1], youden(yv, m.predict_proba(Xv)[:, 1]), m
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
    if real is not None:
        rb = b[np.asarray(real, bool)[b]]
        if len(rb) >= 50 and 0 < ytr[rb].sum() < len(rb):
            b = rb
    m = hgb(seed)
    m.fit(Xtr[a], ytr[a], sample_weight=None if w is None else np.asarray(w)[a])
    thr = youden(ytr[b], m.predict_proba(Xtr[b])[:, 1])
    return m.predict_proba(Xte)[:, 1], thr, m


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return np.nan


def clf_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tpr = recall_score(y, yhat, zero_division=0)
    tnr = recall_score(1 - y, 1 - yhat, zero_division=0)
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    return {"AUROC": safe_auc(y, p), "PR-AUC": float(average_precision_score(y, p)),
            "Brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "F1": float(f1_score(y, yhat, zero_division=0)),
            "Balanced F1": float(f1_score(y, yhat, sample_weight=w, zero_division=0)),
            "Macro-F1": float(f1_score(y, yhat, average="macro", zero_division=0)),
            "Balanced accuracy": float(balanced_accuracy_score(y, yhat)),
            "G-mean": float(np.sqrt(tpr * tnr)),
            "KS": float(ks_2samp(np.asarray(p)[y == 1], np.asarray(p)[y == 0]).statistic) if 0 < y.sum() < len(y) else np.nan}


def group_table(y, yhat, g, min_n=30):
    rows = []
    for k in pd.unique(g[~pd.isna(g)]):
        m = g == k
        if m.sum() < min_n:
            continue
        yk, hk = y[m], yhat[m]
        rows.append({"group": k, "n": int(m.sum()), "positive_rate": yk.mean(), "selection_rate": hk.mean(),
                     "TPR": hk[yk == 1].mean() if (yk == 1).sum() >= 10 else np.nan,
                     "FPR": hk[yk == 0].mean() if (yk == 0).sum() >= 10 else np.nan})
    return pd.DataFrame(rows).set_index("group")


def fairness_gaps(y, yhat, g):
    t = group_table(np.asarray(y), np.asarray(yhat), np.asarray(g, dtype=object))
    if len(t) < 2:
        return {"SPD": np.nan, "DI": np.nan, "DID": np.nan, "EOD": np.nan, "FPR gap": np.nan, "EqOdds": np.nan}
    sel = t["selection_rate"]
    di = sel.min() / sel.max() if sel.max() > 0 else np.nan
    eod = t["TPR"].max() - t["TPR"].min()
    fpr = t["FPR"].max() - t["FPR"].min()
    return {"SPD": float(sel.max() - sel.min()), "DI": float(di), "DID": float(1 - di),
            "EOD": float(eod), "FPR gap": float(fpr), "EqOdds": float(np.nanmax([eod, fpr]))}


SELECT_FAMILY = "Modern"   # project rule: traditional methods are compared, but only modern methods can be selected


def rank_methods(table, directions, families, reference_rows=()):
    """Rank every compared method (traditional and modern) per metric, 1 = best, and average the ranks.
    The selected method is the MODERN method with the lowest mean rank; the best traditional method is
    returned for comparison only. directions: metric -> 'max' | 'min' | ('target', value) | ('absmin',)."""
    tab = table.copy()
    elig = [i for i in tab.index if i not in reference_rows]
    R = pd.DataFrame(index=tab.index, columns=list(directions), dtype=float)
    for m, d in directions.items():
        col = pd.to_numeric(tab.loc[elig, m], errors="coerce")
        if d == "max":
            key = -col
        elif d == "min":
            key = col
        elif d[0] == "target":
            key = (col - d[1]).abs()
        else:
            key = col.abs()
        R.loc[elig, m] = key.rank(method="average", na_option="keep")
    tab["mean_rank"] = R.mean(axis=1, skipna=True)
    ranked = tab.loc[elig, "mean_rank"].astype(float).dropna()
    modern = [i for i in ranked.index if families.get(i) == SELECT_FAMILY]
    trad = [i for i in ranked.index if families.get(i) != SELECT_FAMILY]
    if modern:
        best = ranked[modern].idxmin()
    else:
        best = ranked.idxmin()
        print("WARNING: no modern method produced results; falling back to the best compared method.")
    best_trad = ranked[trad].idxmin() if trad else None
    order = ranked.sort_values().index.tolist() + [r for r in tab.index if r not in ranked.index]
    return tab.loc[order], best, best_trad


def arrow(d):
    return "↑" if d == "max" else ("↓" if d in ("min",) or (isinstance(d, tuple) and d[0] == "absmin") else "→" + str(d[1]))


def show_comparison(tab, directions, best, families, best_trad=None):
    disp = tab.copy()
    disp.insert(0, "family", [families.get(i, "reference") for i in disp.index])
    disp.insert(1, "eligible", ["yes" if families.get(i) == SELECT_FAMILY else "compared only" for i in disp.index])
    disp.columns = [c if c not in directions else f"{c} {arrow(directions[c])}" for c in disp.columns]
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is None:
            raise RuntimeError("not in a notebook")
        display(disp.style.format(precision=4, na_rep="–").apply(
            lambda r: ["background-color:#e6f6e7;font-weight:bold" if r.name == best else
                       ("background-color:#f3f6fc" if r.name == best_trad else "") for _ in r], axis=1))
    except Exception:
        with pd.option_context("display.max_columns", 30, "display.width", 250):
            print(disp.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Selected (modern methods only): {best} | best traditional method (comparison only): {best_trad}")
    progress(f"SELECTED {best} | best traditional {best_trad}")


def mean_rank_plot(ax, tab, best, title, families=None):
    t = tab["mean_rank"].dropna().sort_values()
    fam = families or {}
    cols = ["#4C9B50" if i == best else ("#9DB4D6" if fam.get(i) == SELECT_FAMILY else "#D3D8E2") for i in t.index[::-1]]
    ax.barh(t.index[::-1], t.values[::-1], color=cols)
    ax.set_xlabel("mean rank across metrics (lower = better)\ngreen = selected modern · blue = other modern · grey = traditional (compared only)")
    ax.set_title(title, fontsize=10)

# CELL 10
def _clean(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    return v


def _num(df, v, cap=200000, seed=SEED):
    if df is None or v not in df:
        return None
    a = pd.to_numeric(df[v], errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    if len(a) > cap:
        a = np.random.default_rng(seed).choice(a, cap, replace=False)
    return a if len(a) else None


def _box(a):
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    iqr = q3 - q1
    lo_w, hi_w = max(a.min(), q1 - 1.5 * iqr), min(a.max(), q3 + 1.5 * iqr)
    out = np.unique(a[(a < lo_w) | (a > hi_w)])
    if len(out) > 24:                                    # a readable set of outlier circles, extremes included
        out = out[np.unique(np.linspace(0, len(out) - 1, 24).round().astype(int))]
    return {"q1": q1, "med": med, "q3": q3, "loW": lo_w, "hiW": hi_w, "out": out.tolist(), "min": a.min(), "max": a.max()}


def _qq(a, n=60):
    p = (np.arange(n) + 0.5) / n
    sd = a.std()
    z = (a - a.mean()) / (sd if sd > 0 else 1)
    return [[float(t), float(s)] for t, s in zip(stats.norm.ppf(p), np.quantile(z, p))]


def var_payload(before, after, vars_, rel=False, nbins=10):
    """Histogram bins (10 bins, or one per value for discrete variables), box-plot statistics and Q-Q points.
    rel=True bins Before and After each across its own range (used when the step changes the units)."""
    dist, box, qq = {}, {}, {}
    for v in vars_:
        arrs = {k: _num(df, v) for k, df in (("b", before), ("a", after))}
        arrs = {k: a for k, a in arrs.items() if a is not None}
        if not arrs:
            continue
        d = {"rel": bool(rel)}
        if rel:
            for k, a in arrs.items():
                lo, hi = float(a.min()), float(a.max())
                e = np.linspace(lo, hi if hi > lo else lo + 1, nbins + 1)
                d[k] = np.histogram(a, e)[0].tolist()
                d[k + "r"] = [lo, hi]
        else:
            uniq = np.unique(np.concatenate(list(arrs.values())))
            if len(uniq) <= 12:
                for k, a in arrs.items():
                    d[k] = np.bincount(np.searchsorted(uniq, a), minlength=len(uniq)).tolist()
                lo, hi = float(uniq[0]), float(uniq[-1])
            else:
                lo = float(min(np.percentile(a, 0.5) for a in arrs.values()))
                hi = float(max(np.percentile(a, 99.5) for a in arrs.values()))
                if hi <= lo:
                    lo, hi = float(uniq[0]), float(uniq[-1])
                e = np.linspace(lo, hi, nbins + 1)
                for k, a in arrs.items():
                    d[k] = np.histogram(np.clip(a, lo, hi), e)[0].tolist()
            d["r"] = [lo, hi]
        dist[v] = d
        box[v] = {k: _box(a) for k, a in arrs.items()}
        qq[v] = {k: _qq(a) for k, a in arrs.items()}
    return dist, box, qq


def corr_payload(before, after, vars_, cap=100000):
    out = {}
    for key, n in (("core", 6), ("all", 12)):
        out[key] = {}
        for k, df in (("b", before), ("a", after)):
            vs = [v for v in vars_ if df is not None and v in df][:n]
            d = df[vs].apply(pd.to_numeric, errors="coerce") if vs else pd.DataFrame()
            if len(d) > cap:
                d = d.sample(cap, random_state=SEED)
            out[key][k + "v"] = vs
            out[key][k] = np.nan_to_num(d.corr().to_numpy(), nan=0.0).round(2).tolist() if vs else []
    return out


def comparison_payload(tab, directions, best, families, best_trad=None):
    if tab is None:
        return None
    metrics = list(directions)
    rows = [{"method": str(m), "family": families.get(m, "reference"), "values": [_clean(tab.loc[m, c]) for c in metrics],
             "mean_rank": _clean(tab.loc[m, "mean_rank"])} for m in tab.index]
    return {"metrics": metrics, "arrows": [arrow(directions[c]) for c in metrics],
            "dirs": [directions[c] if isinstance(directions[c], str) else list(directions[c]) for c in metrics],
            "rows": rows, "best": str(best), "best_trad": None if best_trad is None else str(best_trad),
            "select_family": SELECT_FAMILY}

# CELL 12
GRID_N = 30
REMOVED = {"v": 3, "label": "Removed by the selected method", "color": "#C9CFDA", "border": "#AEB6C6"}


def pct_txt(v, word):
    if v is None or not np.isfinite(v):
        return "—"
    if 0 < v < 0.01:
        return f"<0.01% {word}"
    s = f"{v:.0f}" if v >= 10 or v == 0 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")
    return f"{s}% {word}"


def grid_sample(flag_any, n=GRID_N, seed=SEED):
    """Positions of n example rows, in data order. Rare conditions (< 10%) are enriched to half of the examples."""
    flag_any = np.asarray(flag_any, bool)
    rng = np.random.default_rng(seed)
    if len(flag_any) <= n:
        return np.arange(len(flag_any)), False
    if 0 < flag_any.mean() < 0.10:
        on, off = np.where(flag_any)[0], np.where(~flag_any)[0]
        k = min(len(on), n // 2)
        return np.sort(np.r_[rng.choice(on, k, replace=False), rng.choice(off, n - k, replace=False)]), True
    return np.sort(rng.choice(len(flag_any), n, replace=False)), False


def grid_row(label, sb, sa, pb, pa):
    return {"label": str(label), "b": [int(x) for x in sb], "a": [int(x) for x in sa], "pb": pb, "pa": pa}


def grid_payload(title, states, rows, note, off_label, enriched=False, empty=None):
    if enriched:
        note += (" Example patients: the condition is rare, so half of the cells are patients that have it Before; "
                 "the % under each row is the true rate over all rows.")
    return {"title": title, "states": states, "rows": rows, "note": note, "off": off_label, "empty": empty}


def var_grid(title, before, after, cols, flag_fn, state, word, note, off_label, n=GRID_N, extra_states=()):
    """Rows = variables, cells = the same example patients Before and After.
    flag_fn(df, col) -> int array of cell states aligned with df rows (0 = off).
    Patients missing from `after` (removed by the step) are drawn as removed."""
    def flags(df, c):
        try:
            return np.asarray(flag_fn(df, c), int)
        except KeyError:                                 # variable not present on this side (e.g. replaced by a representation)
            return np.zeros(len(df), int)
    Fb = {c: flags(before, c) for c in cols}
    Fa = {c: flags(after, c) for c in cols}
    any_b = np.zeros(len(before), bool)
    for c in cols:
        any_b |= Fb[c] > 0
    pick, enriched = grid_sample(any_b, n)
    apos = after.index.get_indexer(before.index[pick]) if after.index.is_unique and before.index.is_unique else np.arange(len(pick))
    rows = []
    for c in cols:
        sa = [REMOVED["v"] if p < 0 else Fa[c][p] for p in apos]
        rows.append(grid_row(c, Fb[c][pick], sa, pct_txt(100 * (Fb[c] > 0).mean(), word), pct_txt(100 * (Fa[c] > 0).mean(), word)))
    return grid_payload(title, _used_states([state, *extra_states], rows), rows, note, off_label, enriched)


def _used_states(states, rows):
    used = {s for r in rows for s in r["b"] + r["a"]}
    keep = [s for i, s in enumerate(states) if i == 0 or s["v"] in used]
    return keep + ([REMOVED] if REMOVED["v"] in used and REMOVED not in keep else [])


def map_after(before, after, state_after):
    """After-state of each Before row (the same patient); REMOVED where the step dropped the row."""
    pos = after.index.get_indexer(before.index)
    st = np.asarray(state_after, int)
    return np.where(pos >= 0, st[np.clip(pos, 0, None)], REMOVED["v"])


def group_grid(title, groups, state_b, state_a, states, word, note, off_label, n=GRID_N):
    """Rows = patient groups; each cell = one example patient of that group, the same patient in Before and After.
    groups: list of (label, boolean mask over the Before rows); state_b / state_a: int arrays over the Before rows
    (state_a = the After state of the same patient, REMOVED if the step dropped it)."""
    state_b, state_a = np.asarray(state_b, int), np.asarray(state_a, int)
    rows, enriched = [], False
    for i, (label, m) in enumerate(groups):
        idx = np.where(np.asarray(m, bool))[0]
        if not len(idx):
            continue
        pick, e = grid_sample(state_b[idx] > 0, n, seed=SEED + i)
        enriched |= e
        p = idx[pick]
        present = state_a[idx] != REMOVED["v"]
        pa = 100 * (state_a[idx][present] > 0).mean() if present.any() else np.nan
        rows.append(grid_row(label, state_b[p], state_a[p], pct_txt(100 * (state_b[idx] > 0).mean(), word), pct_txt(pa, word)))
    return grid_payload(title, _used_states(states, rows), rows, note, off_label, enriched)


def array_grid(title, rows_b, rows_a, states, word, note, off_label, n=GRID_N):
    """Rows = labels of rows_b (dict label -> int array over the same patients); one shared set of example patients."""
    any_b = np.zeros(len(next(iter(rows_b.values()))), bool)
    for v in rows_b.values():
        any_b |= np.asarray(v) > 0
    pick, enriched = grid_sample(any_b, n)
    rows = [grid_row(k, np.asarray(rows_b[k], int)[pick], np.asarray(rows_a[k], int)[pick],
                     pct_txt(100 * (np.asarray(rows_b[k]) > 0).mean(), word), pct_txt(100 * (np.asarray(rows_a[k]) > 0).mean(), word))
            for k in rows_b]
    return grid_payload(title, _used_states(states, rows), rows, note, off_label, enriched)


def waffle_row(label, shares_b, shares_a, n=20):
    """Cells filled left to right in proportion to each share (state 1, 2, ... in order); used for class balance."""
    def cells(shares):
        out, used = [], 0
        for st, s in enumerate(shares, start=1):
            k = int(round(s * n)) if s > 0 else 0
            k = max(k, 1) if s > 0 else 0
            k = min(k, n - used)
            out += [st] * k
            used += k
        return out + [0] * (n - used)
    return cells(shares_b), cells(shares_a)

# CELL 13
DASH_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root{
    --navy:#0B1E3F;
    --navy2:#132a52;
    --bg:#EEF1F7;
    --card:#FFFFFF;
    --before:#6FA0DA;
    --before-line:#3E6FA8;
    --after:#7BC17E;
    --after-line:#4C9B50;
    --text:#1B2333;
    --muted:#6B7686;
    --border:#E1E5EE;
    --blue-hi:#2563C8;
    --red-hi:#D0433F;
    --shadow: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.08);
  }
  *{box-sizing:border-box;}
  body{
    margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:var(--bg); color:var(--text);
  }
  header.topbar{
    background:linear-gradient(180deg,var(--navy),var(--navy2));
    color:#fff; padding:18px 28px; text-align:center;
    box-shadow: var(--shadow);
  }
  header.topbar h1{margin:0; font-size:22px; letter-spacing:.2px; font-weight:700;}
  header.topbar .sub{margin-top:4px; font-size:12.5px; color:#B9C6E6; font-weight:400;}
  .layout{display:grid; grid-template-columns:270px minmax(0,1fr); gap:16px; padding:16px; align-items:start;}
  .main{min-width:0;}
  .card{
    background:var(--card); border:1px solid var(--border); border-radius:10px;
    box-shadow:var(--shadow); padding:16px;
  }
  .side .card{margin-bottom:14px;}
  .side-label{font-size:12.5px; font-weight:700; color:var(--blue-hi); margin:0 0 8px 0;}
  select.control{
    width:100%; padding:8px 10px; border:1px solid var(--border); border-radius:6px;
    background:#fff; font-size:13px; color:var(--text); appearance:none;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6'><path d='M0 0l5 6 5-6z' fill='%236B7686'/></svg>");
    background-repeat:no-repeat; background-position:right 10px center;
  }
  .varlist{
    max-height:230px; overflow-y:auto; border:1px solid var(--border); border-radius:6px; margin-top:8px;
  }
  .varlist .opt{
    padding:8px 10px; font-size:13px; cursor:pointer; border-bottom:1px solid #F0F2F7;
  }
  .varlist .opt:last-child{border-bottom:none;}
  .varlist .opt:hover{background:#F3F6FC;}
  .varlist .opt.active{background:var(--blue-hi); color:#fff; font-weight:600;}
  .info-title, .legend-title{
    font-size:13px; font-weight:700; margin:0 0 10px 0; display:flex; align-items:center; gap:6px;
  }
  .info-row{display:flex; justify-content:space-between; gap:10px; font-size:12.5px; padding:4px 0; color:var(--text);}
  .info-row span.k{color:var(--muted);}
  .info-row span.v{text-align:right;}
  .legend-row{display:flex; align-items:center; gap:8px; font-size:12.5px; padding:3px 0;}
  .swatch{width:13px; height:13px; border-radius:3px; display:inline-block;}
  .panel-title{
    font-size:14px; font-weight:700; color:var(--blue-hi); margin:0 0 12px 0;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
  }
  .panel-title .ctrls{display:flex; align-items:center; gap:6px; font-weight:400;}
  .panel-title select{padding:5px 8px; font-size:12px; border:1px solid var(--border); border-radius:5px;}
  .legend-inline{display:flex; gap:14px; font-size:11.5px; color:var(--muted); align-items:center; font-weight:400; flex-wrap:wrap;}
  .legend-inline .lg{display:flex; align-items:center; gap:5px;}
  .grid6{display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px;}
  @media (max-width:1500px){ .grid6{grid-template-columns:repeat(3,minmax(0,1fr));} }
  @media (max-width:1100px){ .grid6{grid-template-columns:repeat(2,minmax(0,1fr));} .layout{grid-template-columns:1fr;} }
  .chart-box{border:1px solid var(--border); border-radius:8px; padding:10px 8px 6px; background:#fbfcfe; cursor:pointer;}
  .chart-box .ct{font-size:12px; font-weight:700; text-align:center; margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .row2{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; margin-top:16px;}
  @media (max-width:1100px){ .row2{grid-template-columns:1fr;} }
  svg text{font-family:inherit;}
  .axis{stroke:#B7C0D1; stroke-width:1;}
  .gridline{stroke:#EDEFF4; stroke-width:1;}
  .corr-table{border-collapse:collapse; width:100%; font-size:10.5px;}
  .corr-table th, .corr-table td{border:1px solid #fff; padding:0; text-align:center; width:34px; height:26px;}
  .corr-table th{background:transparent; color:var(--text); font-weight:600; font-size:10px;}
  .corr-heading{text-align:center; font-size:12px; font-weight:700; margin-bottom:6px;}
  .mv-wrap{display:flex; flex-direction:column; overflow-x:auto;}
  .mv-inner{min-width:640px;}
  .mv-headrow{display:grid; grid-template-columns:170px minmax(0,1fr) minmax(0,1fr); gap:26px; margin-bottom:10px;}
  .mv-headrow .h{text-align:center; font-weight:700; font-size:14px; color:var(--text);}
  .mv-row{display:grid; grid-template-columns:170px minmax(0,1fr) minmax(0,1fr); gap:26px; align-items:center; margin-bottom:4px;}
  .mv-label{font-size:12.5px; font-weight:600; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .mv-grid{display:grid; grid-template-columns:repeat(var(--n), minmax(0,1fr)); gap:2px;}   /* 15px cells, shrinking to fit narrower screens */
  .mv-cell{width:100%; aspect-ratio:1/1; background:#FFFFFF; border:1px solid #D7DEEC; border-radius:2px; box-sizing:border-box;}
  .mv-cell.on{background:#C0533E; border-color:#9B4230;}
  .mv-pct{font-size:9.5px; color:var(--muted); text-align:right; margin-top:2px;}
  footer.note{padding:6px 28px 22px; font-size:11px; color:var(--muted); text-align:center;}
  .foot-legend{display:flex; gap:16px; justify-content:center; margin-top:8px; font-size:11px; color:var(--muted); flex-wrap:wrap;}
  .kpi-row{display:flex; gap:10px; flex-wrap:wrap; margin-top:2px;}
  .kpi{flex:1 1 100px; background:#F6F8FC; border:1px solid var(--border); border-radius:8px; padding:8px 10px; text-align:center;}
  .kpi .val{font-size:16px; font-weight:800; color:var(--navy);}
  .kpi .lbl{font-size:10px; color:var(--muted); margin-top:2px;}
  .method-table{width:100%; border-collapse:collapse; font-size:11.5px; margin-top:8px;}
  .method-table th, .method-table td{padding:5px 6px; text-align:center; border-bottom:1px solid var(--border);}
  .method-table th{color:var(--muted); font-weight:600; font-size:10.5px;}
  .method-table td.m{text-align:left; font-weight:600;}
  .method-table tr.winner td{background:#EAF3FF; font-weight:700; color:var(--blue-hi);}
  .method-table tr.ref td{color:var(--muted); font-style:italic;}
  .method-table td.top{color:var(--after-line); font-weight:700;}
  .muted-note{font-size:11.5px; color:var(--muted); margin:-4px 0 10px;}
</style>
</head>
<body>

<header class="topbar">
  <h1 id="ttl"></h1>
  <div class="sub" id="sub"></div>
</header>

<div class="layout">
  <!-- SIDEBAR -->
  <div class="side">
    <div class="card">
      <p class="side-label">1. Select Plot Type</p>
      <select id="plotType" class="control">
        <option value="hist">Histogram</option>
        <option value="density">Density Curve</option>
      </select>
    </div>

    <div class="card">
      <p class="side-label">2. Select Variable (Y Axis)</p>
      <select id="varDropdown" class="control"></select>
      <div class="varlist" id="varList"></div>
    </div>

    <div class="card">
      <p class="info-title">&#9432; Dataset Info</p>
      <div id="info"></div>
    </div>

    <div class="card">
      <p class="legend-title">&#128273; Legend</p>
      <div class="legend-row"><span class="swatch" style="background:var(--before)"></span>Before Preprocessing</div>
      <div class="legend-row"><span class="swatch" style="background:var(--after)"></span>After Preprocessing</div>
    </div>
  </div>

  <!-- MAIN -->
  <div class="main">

    <div class="card">
      <p class="panel-title">Distribution Comparison (Before vs After Preprocessing)
        <span class="legend-inline">
          <span class="lg"><span class="swatch" style="background:var(--before)"></span>Before</span>
          <span class="lg"><span class="swatch" style="background:var(--after)"></span>After</span>
        </span>
      </p>
      <div class="muted-note" id="distNote" style="display:none"></div>
      <div class="grid6" id="distGrid"></div>
    </div>

    <div class="card" style="margin-top:16px;">
      <p class="panel-title">Box Plot Comparison (Before vs After Preprocessing)</p>
      <div class="grid6" id="boxGrid"></div>
    </div>

    <div class="row2">
      <div class="card">
        <p class="panel-title">Q-Q Plot (Normality Check)
          <span class="ctrls"><select id="qqVar" class="control" style="width:170px;"></select></span>
        </p>
        <div id="qqChart"></div>
      </div>

      <div class="card">
        <p class="panel-title">Correlation Heatmap
          <span class="ctrls">
            <select id="corrSet" class="control" style="width:130px;">
              <option value="core">Core 6 Vars</option>
              <option value="all" id="corrAllOpt">All Vars</option>
            </select>
          </span>
        </p>
        <div id="corrChart" style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap; overflow-x:auto;"></div>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <p class="panel-title"><span id="heatTitle"></span>
        <span class="legend-inline" id="heatChips"></span>
      </p>
      <div class="muted-note" id="heatNote"></div>
      <div id="mvHeat"></div>
    </div>

    <div class="card" style="margin-top:16px;">
      <p class="panel-title">Method Comparison (every method &times; every metric)
        <span class="legend-inline"><span class="lg"><span class="swatch" style="background:#EAF3FF;border:1px solid var(--blue-hi)"></span>Selected (best-ranked modern method)</span></span>
      </p>
      <div id="cmp"></div>
    </div>

  </div>
</div>

<footer class="note">
  <span id="footNote"></span>
  <div class="foot-legend" id="footChips"></div>
</footer>

<script>
const D = __DATA__;
const ALL_VARS = D.vars;
const B_ONLY = ALL_VARS.filter(v=>D.box[v].b && !D.box[v].a), A_ONLY = ALL_VARS.filter(v=>D.box[v].a && !D.box[v].b);
const REPLACED = B_ONLY.length>0 && A_ONLY.length>0;   // the step replaced the predictors with learned features
const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs){ const e=document.createElementNS(NS, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }
function makeSVG(w,h){ return svgEl('svg', {viewBox:`0 0 ${w} ${h}`, width:'100%', height:h}); }
function esc(s){ return String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fmtAxis(val, range){
  if(range===0 || !isFinite(range)) return val.toFixed(2);
  if(Math.abs(val)>=1e5) return val.toExponential(1);
  if(range < 0.5) return val.toFixed(3);
  if(range < 5) return val.toFixed(2);
  if(range < 20) return val.toFixed(1);
  return val.toFixed(0);
}
function fmtCount(n){ if(n>=1e6) return (n/1e6).toFixed(1)+'M'; if(n>=1e4) return Math.round(n/1e3)+'k'; if(n>=1e3) return (n/1e3).toFixed(1)+'k'; return String(Math.round(n)); }
function fmtVal(v){ if(v===null||v===undefined||Number.isNaN(v)) return '–'; const a=Math.abs(v); if(a!==0&&(a<0.001||a>=1e6)) return v.toExponential(2); if(a>=1000) return v.toFixed(0); if(a>=100) return v.toFixed(1); if(a>=1) return v.toFixed(3); return v.toFixed(4); }

document.getElementById('ttl').textContent = 'Interactive EDA Dashboard — ' + D.title + ' (Before vs After Preprocessing)';
document.title = 'Interactive EDA Dashboard — ' + D.title + ' (Before vs After Preprocessing)';
document.getElementById('sub').textContent = D.subtitle;
document.getElementById('info').innerHTML = D.info.map(([k,v])=>`<div class="info-row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join('');
document.getElementById('footNote').innerHTML = D.footer;
document.getElementById('footChips').innerHTML = D.chips.map(c=>`<span>${c}</span>`).join('');

// ---------- sidebar variable list ----------
const varDropdown = document.getElementById('varDropdown');
const varList = document.getElementById('varList');
ALL_VARS.forEach(v=>{ const o=document.createElement('option'); o.value=v; o.textContent=v; varDropdown.appendChild(o); });
let selectedVar = ALL_VARS[0];
let gridVars = REPLACED ? B_ONLY.slice(0,5).concat(A_ONLY.slice(0,5)) : ALL_VARS.slice(0, 10);
if(REPLACED){
  const n=document.getElementById('distNote'); n.style.display='block';
  n.textContent='The selected method replaced the original predictors with learned features, so an original variable has no direct After value. Blue charts = original predictors (Before); green charts = learned features (After).';
}
function renderVarList(){
  varList.innerHTML = '';
  ALL_VARS.forEach(v=>{
    const d = document.createElement('div');
    d.className = 'opt' + (v===selectedVar?' active':'');
    d.textContent = v;
    d.onclick = ()=>choose(v);
    varList.appendChild(d);
  });
}
function choose(v){
  selectedVar=v; varDropdown.value=v; renderVarList();
  if(!gridVars.includes(v)){
    if(REPLACED){
      const top = list => [v].concat(list.filter(x=>x!==v)).slice(0,5);
      gridVars = B_ONLY.includes(v) ? top(B_ONLY).concat(A_ONLY.slice(0,5)) : B_ONLY.slice(0,5).concat(top(A_ONLY));
    } else { gridVars = [v].concat(ALL_VARS.filter(x=>x!==v).slice(0, 9)); }
    renderGrids();
  }
  highlightVar(v); document.getElementById('qqVar').value=v; drawQQ();
}
varDropdown.addEventListener('change', e=>choose(e.target.value));
renderVarList();

function highlightVar(v){
  document.querySelectorAll('.chart-box').forEach(el=>el.style.boxShadow='none');
  const el = document.getElementById('box_'+v);
  if(el){ el.style.boxShadow='0 0 0 2px var(--blue-hi)'; el.scrollIntoView({behavior:'smooth', block:'center'}); }
}

// ================= Histogram / Density =================
function drawDistChart(container, v){
  const d = D.dist[v];
  const nb = (d.b||d.a).length;
  const maxc = Math.max(...(d.b||[0]), ...(d.a||[0]), 1);
  const W=250, H=d.rel?160:150, padL=30, padB=d.rel?30:20, padT=6, padR=6;
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const bw = plotW/nb;
  const svg = makeSVG(W,H);
  for(let i=0;i<=4;i++){
    const y = padT + plotH - (i/4)*plotH;
    svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y,y2:y,class:'gridline'}));
    const t=svgEl('text',{x:padL-4,y:y+3,'text-anchor':'end','font-size':8,fill:'#8791A3'}); t.textContent=fmtCount((i/4)*maxc); svg.appendChild(t);
  }
  const plotType = document.getElementById('plotType').value;
  function barsOrLine(counts, color, line, alpha){
    if(!counts) return;
    if(plotType==='hist'){
      counts.forEach((c,i)=>{
        const x = padL + i*bw;
        const barH = (c/maxc)*plotH;
        svg.appendChild(svgEl('rect',{x:x+1, y:padT+plotH-barH, width:Math.max(bw-2,1), height:barH, fill:color, opacity:alpha, stroke:line, 'stroke-width':1}));
      });
    } else {
      let dd='';
      counts.forEach((c,i)=>{ const x = padL + i*bw + bw/2; const y = padT + plotH - (c/maxc)*plotH; dd += (i===0?'M':'L') + x + ' ' + y + ' '; });
      svg.appendChild(svgEl('path',{d:dd.trim(), fill:'none', stroke:line, 'stroke-width':2}));
      const dArea = dd.trim() + ` L ${padL+bw*(nb-0.5)} ${padT+plotH} L ${padL+bw/2} ${padT+plotH} Z`;
      svg.appendChild(svgEl('path',{d:dArea, fill:color, opacity:alpha*0.6}));
    }
  }
  barsOrLine(d.b, 'var(--before)', 'var(--before-line)', 0.55);
  barsOrLine(d.a, 'var(--after)', 'var(--after-line)', 0.55);
  svg.appendChild(svgEl('line',{x1:padL,x2:padL,y1:padT,y2:padT+plotH,class:'axis'}));
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:padT+plotH,y2:padT+plotH,class:'axis'}));
  if(d.rel){
    [['b','B','#3E6FA8',H-15],['a','A','#4C9B50',H-4]].forEach(([k,lab,col,y])=>{
      if(!d[k+'r']) return; const [lo,hi]=d[k+'r'];
      const t=svgEl('text',{x:padL,y:y,'font-size':8,fill:col}); t.textContent=`${lab}: ${fmtAxis(lo,hi-lo)}–${fmtAxis(hi,hi-lo)}`; svg.appendChild(t);
    });
  } else {
    const [lo,hi]=d.r;
    const xt1=svgEl('text',{x:padL,y:H-4,'font-size':8,fill:'#8791A3'}); xt1.textContent=fmtAxis(lo,hi-lo); svg.appendChild(xt1);
    const xt2=svgEl('text',{x:W-padR,y:H-4,'text-anchor':'end','font-size':8,fill:'#8791A3'}); xt2.textContent=fmtAxis(hi,hi-lo); svg.appendChild(xt2);
  }
  container.appendChild(svg);
}

// ================= Box plot =================
function drawBoxChart(container, v){
  const s = D.box[v], rel = D.dist[v].rel;
  const W=250, H=150, padL=32, padB=20, padT=8, padR=10;
  const plotH=H-padT-padB;
  const rng = st => [Math.min(st.loW, ...st.out), Math.max(st.hiW, ...st.out)];
  let lo=Infinity, hi=-Infinity;
  ['b','a'].forEach(k=>{ if(s[k]){ const [l,h]=rng(s[k]); lo=Math.min(lo,l); hi=Math.max(hi,h); } });
  const svg = makeSVG(W,H);
  for(let i=0;i<=3;i++){ const yy=padT+(i/3)*plotH; svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:yy,y2:yy,class:'gridline'})); }
  function box(cx, st, color, line){
    let L=lo, Hh=hi; if(rel){ [L,Hh]=rng(st); }
    const y = val => padT + plotH - ((val-L)/((Hh-L)||1))*plotH;
    const bw=46;
    svg.appendChild(svgEl('line',{x1:cx,x2:cx,y1:y(st.loW),y2:y(st.q1),stroke:line,'stroke-width':1.3}));
    svg.appendChild(svgEl('line',{x1:cx,x2:cx,y1:y(st.q3),y2:y(st.hiW),stroke:line,'stroke-width':1.3}));
    svg.appendChild(svgEl('line',{x1:cx-8,x2:cx+8,y1:y(st.loW),y2:y(st.loW),stroke:line,'stroke-width':1.3}));
    svg.appendChild(svgEl('line',{x1:cx-8,x2:cx+8,y1:y(st.hiW),y2:y(st.hiW),stroke:line,'stroke-width':1.3}));
    svg.appendChild(svgEl('rect',{x:cx-bw/2, y:y(st.q3), width:bw, height:Math.max(y(st.q1)-y(st.q3),1), fill:color, stroke:line, 'stroke-width':1.3, opacity:0.85}));
    svg.appendChild(svgEl('line',{x1:cx-bw/2,x2:cx+bw/2,y1:y(st.med),y2:y(st.med),stroke:line,'stroke-width':1.6}));
    st.out.forEach(o=>{ svg.appendChild(svgEl('circle',{cx:cx, cy:y(o), r:2, fill:'none', stroke:line, 'stroke-width':1})); });
  }
  const xb=padL+(W-padL-padR)*(s.a?0.28:0.5), xa=padL+(W-padL-padR)*(s.b?0.72:0.5);
  if(s.b) box(xb, s.b, 'var(--before)', 'var(--before-line)');
  if(s.a) box(xa, s.a, 'var(--after)', 'var(--after-line)');
  svg.appendChild(svgEl('line',{x1:padL,x2:padL,y1:padT,y2:padT+plotH,class:'axis'}));
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:padT+plotH,y2:padT+plotH,class:'axis'}));
  [['b',xb,'Before','#556'],['a',xa,'After','#556']].forEach(([k,x,lab,col])=>{
    if(!s[k]) return;
    let txt=lab, c=col;
    if(rel && s[k]){ const [l,h]=rng(s[k]); txt=`${lab} [${fmtAxis(l,h-l)}, ${fmtAxis(h,h-l)}]`; c = k==='b'?'#3E6FA8':'#4C9B50'; }
    const t=svgEl('text',{x:x,y:H-4,'text-anchor':'middle','font-size':rel?7.5:9,fill:c}); t.textContent=txt; svg.appendChild(t);
  });
  if(!rel){
    for(let i=0;i<=3;i++){
      const val = lo + (i/3)*(hi-lo); const yy=padT+plotH-(i/3)*plotH;
      const t=svgEl('text',{x:padL-4,y:yy+3,'text-anchor':'end','font-size':8,fill:'#8791A3'}); t.textContent=fmtAxis(val,hi-lo); svg.appendChild(t);
    }
  }
  container.appendChild(svg);
}

// ================= render grids =================
function renderGrids(){
  const distGrid=document.getElementById('distGrid'); distGrid.innerHTML='';
  const boxGrid=document.getElementById('boxGrid'); boxGrid.innerHTML='';
  gridVars.forEach(v=>{
    const cb = document.createElement('div'); cb.className='chart-box'; cb.id='box_'+v; cb.onclick=()=>choose(v);
    const t = document.createElement('div'); t.className='ct'; t.textContent=v; t.title=v; cb.appendChild(t);
    drawDistChart(cb, v);
    distGrid.appendChild(cb);
    const bb = document.createElement('div'); bb.className='chart-box'; bb.onclick=()=>choose(v);
    const t2 = document.createElement('div'); t2.className='ct'; t2.textContent=v; t2.title=v; bb.appendChild(t2);
    drawBoxChart(bb, v);
    boxGrid.appendChild(bb);
  });
}

// ================= Q-Q plot =================
function drawQQSub(svg, x0, w, pts, color, line, label, H, padT, padB){
  const container = svg;
  if(!pts){ const t=svgEl('text',{x:x0+w/2,y:H/2,'text-anchor':'middle','font-size':10,fill:'#8791A3'}); t.textContent='not present'; svg.appendChild(t); }
  else {
    const theo=pts.map(p=>p[0]), z=pts.map(p=>p[1]);
    const lo=Math.min(...theo,...z), hi=Math.max(...theo,...z);
    const plotH=H-padT-padB, plotW=w-40;
    const X = t => x0+30 + ((t-lo)/(hi-lo||1))*plotW;
    const Y = t => padT + plotH - ((t-lo)/(hi-lo||1))*plotH;
    svg.appendChild(svgEl('line',{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:'var(--red-hi)','stroke-width':1.3,'stroke-dasharray':'3,2'}));
    svg.appendChild(svgEl('line',{x1:X(lo),x2:X(lo),y1:padT,y2:padT+plotH,class:'axis'}));
    svg.appendChild(svgEl('line',{x1:X(lo),x2:X(hi),y1:padT+plotH,y2:padT+plotH,class:'axis'}));
    theo.forEach((t,i)=>{ svg.appendChild(svgEl('circle',{cx:X(t), cy:Y(z[i]), r:2.3, fill:color, stroke:line, 'stroke-width':0.5, opacity:0.85})); });
  }
  const lb=svgEl('text',{x:x0+w/2,y:H-2,'text-anchor':'middle','font-size':10,'font-weight':700,fill:'#334'}); lb.textContent=label; container.appendChild(lb);
}
function drawQQ(){
  const v = document.getElementById('qqVar').value;
  const q = D.qq[v] || {};
  const container = document.getElementById('qqChart'); container.innerHTML='';
  const W=520, H=210, padT=10, padB=24;
  const svg = makeSVG(W,H);
  let pb=q.b, pa=q.a, lb='Before Preprocessing', la='After Preprocessing';
  if(!pb && B_ONLY.length){ pb=D.qq[B_ONLY[0]].b; lb+=` (${B_ONLY[0]})`; }
  if(!pa && A_ONLY.length){ pa=D.qq[A_ONLY[0]].a; la+=` (${A_ONLY[0]})`; }
  drawQQSub(svg, 0, W/2, pb, 'var(--before)', 'var(--before-line)', lb, H, padT, padB);
  drawQQSub(svg, W/2, W/2, pa, 'var(--after)', 'var(--after-line)', la, H, padT, padB);
  container.appendChild(svg);
}

// ================= Correlation heatmap =================
function corrColor(r){
  const t=(r+1)/2;
  const c1=[47,94,181], c2=[255,255,255], c3=[192,57,43];
  let col;
  if(t<0.5){ const k=t/0.5; col=c1.map((c,i)=>Math.round(c+(c2[i]-c)*k)); }
  else { const k=(t-0.5)/0.5; col=c2.map((c,i)=>Math.round(c+(c3[i]-c)*k)); }
  return `rgb(${col[0]},${col[1]},${col[2]})`;
}
function short(v){ return v.length>8 ? v.slice(0,6)+'…' : v; }
function renderCorrTable(vars, mat, title){
  const wrap = document.createElement('div');
  const h = document.createElement('div'); h.className='corr-heading'; h.textContent=title; wrap.appendChild(h);
  if(!vars.length){ const p=document.createElement('div'); p.className='mv-pct'; p.style.textAlign='center'; p.textContent='no variables'; wrap.appendChild(p); return wrap; }
  const table = document.createElement('table'); table.className='corr-table';
  const thead = document.createElement('tr');
  thead.appendChild(document.createElement('th'));
  vars.forEach(v=>{ const th=document.createElement('th'); th.textContent=short(v); th.title=v; th.style.writingMode='vertical-rl'; th.style.transform='rotate(180deg)'; th.style.fontSize='9px'; thead.appendChild(th); });
  table.appendChild(thead);
  vars.forEach((v1,i)=>{
    const tr=document.createElement('tr');
    const th=document.createElement('th'); th.textContent=short(v1); th.title=v1; th.style.textAlign='right'; th.style.paddingRight='4px'; th.style.fontSize='9px'; tr.appendChild(th);
    vars.forEach((v2,j)=>{
      const td=document.createElement('td');
      const val=mat[i][j];
      td.style.background=corrColor(val);
      td.style.color = Math.abs(val)>0.6 ? '#fff':'#1B2333';
      td.textContent=val.toFixed(2); td.title=`${v1} × ${v2}: ${val.toFixed(2)}`;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  wrap.appendChild(table);
  return wrap;
}
function drawCorr(){
  const c = D.corr[document.getElementById('corrSet').value];
  const container = document.getElementById('corrChart'); container.innerHTML='';
  container.appendChild(renderCorrTable(c.bv, c.b, 'Before Preprocessing'));
  container.appendChild(renderCorrTable(c.av, c.a, 'After Preprocessing'));
}

// ================= Bias heatmap (cell grid) =================
function chip(st){ return `<span class="lg"><span class="swatch" style="background:${st.color};border:1px solid ${st.border||st.color}"></span>${esc(st.label)}</span>`; }
function drawMVHeat(){
  const h = D.heat;
  document.getElementById('heatTitle').textContent = h.title;
  document.getElementById('heatChips').innerHTML = h.states.map(chip).join('');
  document.getElementById('heatNote').innerHTML = h.note;
  const container = document.getElementById('mvHeat'); container.innerHTML='';
  if(!h.rows.length){ container.innerHTML = `<div style="font-size:12.5px;color:var(--muted);padding:6px 0;">${esc(h.empty||'Not applicable')}</div>`; return; }
  const colour = {}; h.states.forEach(s=>colour[s.v]=s);
  const outer = document.createElement('div'); outer.className='mv-wrap';
  const wrap = document.createElement('div'); wrap.className='mv-inner';
  outer.appendChild(wrap);
  const head = document.createElement('div'); head.className='mv-headrow';
  const gw = k => Math.max(...h.rows.map(r=>r[k].length))*17-2;
  head.innerHTML = `<div></div><div class="h" style="width:min(100%,${gw('b')}px)">Before</div><div class="h" style="width:min(100%,${gw('a')}px)">After</div>`;
  wrap.appendChild(head);
  h.rows.forEach(r=>{
    const row = document.createElement('div'); row.className='mv-row';
    const label = document.createElement('div'); label.className='mv-label'; label.textContent=r.label; label.title=r.label;
    row.appendChild(label);
    [['b','Before'],['a','After']].forEach(([k,lab])=>{
      const holder = document.createElement('div'); holder.style.width=`min(100%,${r[k].length*17-2}px)`;
      const grid = document.createElement('div'); grid.className='mv-grid'; grid.style.setProperty('--n', r[k].length);
      r[k].forEach((s,i)=>{
        const cell = document.createElement('div'); cell.className='mv-cell';
        if(s>0 && colour[s]){ cell.style.background=colour[s].color; cell.style.borderColor=colour[s].border||colour[s].color; }
        cell.title = `${r.label} · example ${i+1} · ${lab}: ` + (s>0 && colour[s] ? colour[s].label : h.off);
        grid.appendChild(cell);
      });
      const pct = document.createElement('div'); pct.className='mv-pct'; pct.textContent = k==='b' ? r.pb : r.pa;
      holder.appendChild(grid); holder.appendChild(pct);
      row.appendChild(holder);
    });
    row.style.marginBottom='10px';
    row.style.alignItems='start';
    wrap.appendChild(row);
  });
  container.appendChild(outer);
  const legend = document.createElement('div');
  legend.style.cssText='display:flex;gap:14px;justify-content:center;margin-top:14px;font-size:11px;color:var(--muted);flex-wrap:wrap;';
  legend.innerHTML = h.states.map(s=>`<span><span class="swatch" style="background:${s.color};border:1px solid ${s.border||s.color};display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px;"></span>${esc(s.label)}</span>`).join('') +
    `<span><span class="swatch" style="background:#FFFFFF;border:1px solid #D6DCEA;display:inline-block;width:11px;height:11px;border-radius:0;margin-right:5px;vertical-align:-1px;"></span>${esc(h.off)}</span>`;
  container.appendChild(legend);
}

// ================= Method comparison =================
function drawCmp(){
  const c = document.getElementById('cmp'); const d = D.comparison;
  if(!d){ c.innerHTML = `<div class="muted-note" style="margin:0;font-size:12.5px;">${esc(D.cmp_note)}</div>`; return; }
  const best=[];
  d.metrics.forEach((m,j)=>{
    const vals=d.rows.filter(x=>x.family!=='reference').map(x=>x.values[j]).filter(v=>v!==null);
    if(!vals.length){ best.push(null); return; }
    const dir=d.dirs[j]; let b;
    if(dir==='max') b=Math.max(...vals); else if(dir==='min') b=Math.min(...vals);
    else if(dir[0]==='target') b=vals.reduce((p,v)=>Math.abs(v-dir[1])<Math.abs(p-dir[1])?v:p);
    else b=vals.reduce((p,v)=>Math.abs(v)<Math.abs(p)?v:p);
    best.push(b);
  });
  const nm = d.rows.filter(x=>x.family!=='reference').length;
  let h = `<div class="kpi-row">
    <div class="kpi"><div class="val">${nm}</div><div class="lbl">methods compared</div></div>
    <div class="kpi"><div class="val">${d.metrics.length}</div><div class="lbl">metrics</div></div>
    <div class="kpi"><div class="val" style="font-size:13px">${esc(d.best)}</div><div class="lbl">selected (best-ranked ${esc(d.select_family.toLowerCase())})</div></div>
    <div class="kpi"><div class="val" style="font-size:13px;color:var(--muted)">${esc(d.best_trad||'—')}</div><div class="lbl">best traditional (comparison only)</div></div></div>`;
  h += `<div style="overflow-x:auto"><table class="method-table"><tr><th style="text-align:left">Method</th><th>Family</th>${d.metrics.map((m,j)=>`<th>${esc(m)} ${d.arrows[j]}</th>`).join('')}<th>Mean rank ↓</th></tr>`;
  d.rows.forEach(x=>{
    const cls = x.method===d.best ? 'winner' : (x.family==='reference' ? 'ref' : '');
    const fam = x.family==='reference' ? 'reference' : (x.family===d.select_family ? x.family : x.family + ' (compared only)');
    h += `<tr class="${cls}"><td class="m">${esc(x.method)}</td><td>${esc(fam)}</td>` +
         x.values.map((v,j)=>`<td class="${(cls!=='winner'&&x.family!=='reference'&&v!==null&&v===best[j])?'top':''}">${fmtVal(v)}</td>`).join('') +
         `<td>${x.mean_rank===null?'–':x.mean_rank.toFixed(2)}</td></tr>`;
  });
  h += `</table></div><div class="mv-pct" style="text-align:left;margin-top:8px;font-size:11px;">All methods are ranked together on every metric (green = best value of that metric). Project rule: only <b>${esc(d.select_family)}</b> methods can be selected and applied; traditional methods are compared only. Reference rows give context and are not ranked.</div>`;
  c.innerHTML = h;
}

// ================= wire up controls =================
document.getElementById('plotType').addEventListener('change', renderGrids);
document.getElementById('corrSet').addEventListener('change', drawCorr);
document.getElementById('corrAllOpt').textContent = `All ${Math.max(D.corr.all.bv.length, D.corr.all.av.length)} Vars`;
const qqVarSel = document.getElementById('qqVar');
ALL_VARS.forEach(v=>{ const o=document.createElement('option'); o.value=v; o.textContent=v; qqVarSel.appendChild(o); });
qqVarSel.value=ALL_VARS[0];
qqVarSel.addEventListener('change', drawQQ);

renderGrids();
drawQQ();
drawCorr();
drawMVHeat();
drawCmp();
</script>
</body>
</html>"""

import html as _html
STAGE = {"06": 1, "01": 2, "03": 3, "07": 4, "04": 5, "08": 6, "02": 7, "05": 8}


def write_dashboard(fname, title, detected, banner, detection, selected, before, after, vars_, heat,
                    comparison=None, cmp_note=None, footer="", info_extra=None, short=None, rel=False, chips=None):
    vars_ = list(dict.fromkeys(v for v in vars_ if (before is not None and v in before) or (after is not None and v in after)))
    dist, box, qq = var_payload(before, after, vars_, rel=rel)
    shown = [v for v in vars_ if v in dist]
    best_trad = (comparison or {}).get("best_trad")
    info = [("Total Patients (before)", f"{len(before):,}"), ("Rows after", f"{len(after):,}"),
            ("Variables shown", f"{len(shown)}"), ("Bias detected", "YES" if detected else "no")]
    info += [(k, str(v)) for k, v in detection.items()]
    info += [("Selected method", selected if detected and selected else "none (not detected)"),
             ("Best traditional", best_trad or "—")]
    if info_extra:
        info += [(k, str(v)) for k, v in info_extra.items()]
    info.append(("Generated", date.today().strftime("%d-%b-%Y")))
    stage = STAGE.get(fname[:2], "")
    short = short or (f"{title.split(' Bias')[0]} detected → {selected}" if detected else "not detected → data unchanged (After = Before)")
    chips = chips or []
    chips = [f"<b>{len(before):,}</b> rows (before)", f"<b>{len(after):,}</b> rows (after)",
             f"Bias detected: <b>{'YES' if detected else 'no'}</b>"] + chips + \
            [f"Selected method: <b>{_html.escape(str(selected)) if detected and selected else 'none'}</b>"]
    payload = {"title": title, "subtitle": f"{TITLE} · Stage {stage} of 8 · {short}", "info": info,
               "vars": shown, "dist": dist, "box": box, "qq": qq, "corr": corr_payload(before, after, shown),
               "heat": heat, "comparison": comparison,
               "cmp_note": cmp_note or "No methods were run: the bias was not detected, so the data are unchanged (After = Before).",
               "footer": f"<b>{_html.escape(banner.rstrip('.'))}.</b> {_html.escape(footer)}", "chips": chips}
    html = DASH_TEMPLATE.replace("__TITLE__", f"Interactive EDA Dashboard — {title} (Before vs After Preprocessing)") \
                        .replace("__DATA__", json.dumps(_clean(payload)))
    path = PLOT_DIR / fname
    path.write_text(html, encoding="utf-8")
    print("dashboard ->", path)
    return path


def display_vars(feats, cap=20):
    """Variables shown in dashboards: continuous and ordinal first, then binary (one-hot omitted)."""
    v = [c for c in feats if SCHEMA.get(c, {}).get("type") in ("cont", "ord")]
    v += [c for c in feats if SCHEMA.get(c, {}).get("type") == "bin"]
    v += [c for c in feats if c not in SCHEMA]           # latent dimensions after a representation change
    return v[:cap]

# CELL 15
def make_split(df, test_size=0.2, seed=SEED):
    if GROUP_COL:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, df["target"], df[GROUP_COL]))
    else:
        tr, te = train_test_split(np.arange(len(df)), test_size=test_size, stratify=df["target"], random_state=seed)
    return df.iloc[tr].copy(), df.iloc[te].copy()


def detect_duplicates(df):
    key = [c for c in df.columns if c not in ID_COLS and c != "w"]
    exact = df.duplicated(subset=key, keep="first")
    profile = df.duplicated(subset=[c for c in FEATS if c in df] + ["target"], keep="first")
    return {"statistic": "exact duplicate records (all columns except record ID)", "value": int(exact.sum()),
            "rate": float(exact.mean()), "profile_collisions": int(profile.sum()),
            "threshold": "> 0", "detected": bool(exact.sum() > 0), "mask": exact.to_numpy()}


def detect_missing(df, feats):
    rates = df[feats].isna().mean()
    cell = float(df[feats].isna().to_numpy().mean())
    return {"statistic": "share of missing predictor cells", "value": cell, "per_feature": rates,
            "threshold": "> 0", "detected": bool(cell > 0)}


def adjusted_fences(x, k=3.0, seed=SEED):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None
    s = np.random.default_rng(seed).choice(x, min(len(x), 3000), replace=False)
    try:
        mc = float(medcouple(s))
    except (IndexError, ValueError):
        # statsmodels' fast medcouple can fail on heavily tied samples; break ties with a negligible
        # jitter (1e-9 of the IQR), and fall back to symmetric fences (mc = 0) if it still fails.
        try:
            mc = float(medcouple(s + np.random.default_rng(seed).normal(0, 1e-9 * iqr, len(s))))
        except (IndexError, ValueError):
            mc = 0.0
    if mc >= 0:
        return q1 - k * np.exp(-4 * mc) * iqr, q3 + k * np.exp(3 * mc) * iqr
    return q1 - k * np.exp(-3 * mc) * iqr, q3 + k * np.exp(4 * mc) * iqr


def detect_outliers(df, feats, fences=None):
    cont = [c for c in feats if SCHEMA.get(c, {}).get("type") == "cont"]
    fences = fences or {c: adjusted_fences(df[c].to_numpy(float)) for c in cont}
    flag = np.zeros(len(df), bool)
    per = {}
    for c, f in fences.items():
        if f is None or c not in df:
            continue
        v = df[c].to_numpy(float)
        lo_f, hi_f = (v < f[0]), (v > f[1])
        per[c] = {"lower": f[0], "upper": f[1], "below_%": 100 * np.nanmean(lo_f), "above_%": 100 * np.nanmean(hi_f)}
        flag |= lo_f | hi_f
    rate = float(flag.mean())
    return {"statistic": "rows beyond skew-adjusted extreme fences", "value": rate, "per_feature": per,
            "fences": fences, "threshold": ">= 1% of rows", "detected": bool(rate >= 0.01), "mask": flag}


def oof_probs(X, y, folds=5, seed=SEED, iters=250):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = hgb(seed, iters)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def detect_label_noise(df, feats, n_max=None):
    from cleanlab.filter import find_label_issues
    d = strat_sample(df, n_max or S("noise_detect"))
    X, y = d[feats].to_numpy(float), d["target"].to_numpy(int)
    p = oof_probs(X, y)
    issues = find_label_issues(labels=y, pred_probs=np.c_[1 - p, p], filter_by="prune_by_noise_rate", n_jobs=1)
    rate = float(issues.mean())
    return {"statistic": "Confident-Learning label-issue rate", "value": rate, "threshold": ">= 2% of rows",
            "detected": bool(rate >= 0.02), "issues_by_class": {int(k): float(issues[y == k].mean()) for k in (0, 1)}}


def detect_scale(df, feats):
    cols = [c for c in feats if SCHEMA.get(c, {}).get("type") in ("cont", "ord")]
    sd = df[cols].std()
    sd = sd[sd > 0]
    sk = df[cols].apply(lambda s: skew(s.dropna()) if s.notna().sum() > 2 else 0.0)
    ratio = float(sd.max() / sd.min()) if len(sd) else 1.0
    msk = float(sk.abs().max()) if len(sk) else 0.0
    return {"statistic": "max/min predictor SD ratio; max |skewness|", "value": f"{ratio:.1f}; {msk:.2f}",
            "ratio": ratio, "max_abs_skew": msk, "threshold": "ratio > 10 or max|skew| > 1",
            "detected": bool(ratio > 10 or msk > 1), "skew": sk, "sd": df[cols].std()}


def domain_labels(df, min_rows=50):
    d = df[DOMAIN_COL].astype(str)
    counts = d.value_counts()
    keep = counts[counts >= min_rows].index
    return d.where(d.isin(keep), "other")


def detect_domain(df, feats, n_max=None, seed=SEED):
    if not DOMAIN_COL:
        return {"statistic": "no site/domain identifier in this dataset", "value": "n/a", "threshold": "n/a",
                "detected": False, "applicable": False}
    d = df.sample(min(len(df), n_max or S("domain_detect")), random_state=seed)
    X = d[feats].to_numpy(float)
    dom = domain_labels(d)
    top = [k for k in dom.value_counts().index if k != "other"][:20]
    aucs = []
    for k in top:
        yk = (dom == k).to_numpy().astype(int)
        if yk.sum() < 20:
            continue
        p = oof_probs(X, yk, folds=3, seed=seed, iters=100)
        aucs.append(safe_auc(yk, p))
    dom_auc = float(np.nanmean(aucs)) if aucs else np.nan
    rng = np.random.default_rng(seed)
    doms = np.array(sorted(df[DOMAIN_COL].astype(str).unique()))
    tgt = set(rng.choice(doms, max(1, int(round(0.2 * len(doms)))), replace=False))
    is_t = df[DOMAIN_COL].astype(str).isin(tgt).to_numpy()
    src, tg = df[~is_t], df[is_t]
    a, b = train_test_split(np.arange(len(src)), test_size=0.2, stratify=src["target"], random_state=seed)
    a = a[:200000]
    m = hgb(seed).fit(src[feats].to_numpy(float)[a], src["target"].to_numpy()[a])
    auc_in = safe_auc(src["target"].to_numpy()[b], m.predict_proba(src[feats].to_numpy(float)[b])[:, 1])
    auc_out = safe_auc(tg["target"], m.predict_proba(tg[feats].to_numpy(float))[:, 1])
    gap = auc_in - auc_out
    return {"statistic": "one-vs-rest domain-classifier AUC; AUROC drop on held-out domains",
            "value": f"{dom_auc:.3f}; {gap:+.3f}", "domain_auc": dom_auc, "dg_gap": gap,
            "threshold": "AUC >= 0.60 or drop >= 0.02", "detected": bool(dom_auc >= 0.60 or gap >= 0.02),
            "applicable": True, "n_domains": int(df[DOMAIN_COL].nunique())}


def protected_groups(df):
    out = {}
    for a in PROTECTED:
        out[a] = df[a].astype(object).where(df[a].notna(), "Unknown").to_numpy()
    return out


def detect_demographic(df, feats, seed=SEED):
    y = df["target"].to_numpy(int)
    d = strat_sample(df, S("demog_detect"))
    a, b = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d["target"], random_state=seed)
    if "meta_synthetic" in d:          # evaluate the baseline on genuine rows only
        b = b[d["meta_synthetic"].to_numpy()[b] == 0]
    X = d[feats].to_numpy(float)
    yd = d["target"].to_numpy(int)
    m = hgb(seed).fit(X[a], yd[a])
    pb = m.predict_proba(X[b])[:, 1]
    thr = youden(yd[b], pb)
    rows = []
    for attr in PROTECTED + AUDIT_ONLY:
        g_all = df[attr].astype(object).where(df[attr].notna(), "Unknown").to_numpy()
        rates = pd.Series(y).groupby(g_all).mean()
        rates = rates[[k for k in rates.index if k != "Unknown"]]
        gb = d[attr].astype(object).where(d[attr].notna(), "Unknown").to_numpy()[b]
        gaps = fairness_gaps(yd[b], (pb >= thr).astype(int), np.where(gb == "Unknown", None, gb))
        rows.append({"attribute": attr, "protected": attr in PROTECTED,
                     "outcome_rate_min": rates.min(), "outcome_rate_max": rates.max(),
                     "DI_outcome": rates.min() / rates.max() if rates.max() > 0 else np.nan,
                     "SPD_outcome": rates.max() - rates.min(), "EOD_baseline": gaps["EOD"], "EqOdds_baseline": gaps["EqOdds"]})
    t = pd.DataFrame(rows).set_index("attribute")
    prot = t[t["protected"]]
    det = bool((prot["DI_outcome"] < 0.8).any() or (prot["EOD_baseline"] > 0.10).any())
    return {"statistic": "min DI of outcome rate; max EOD of baseline model (protected attributes)",
            "value": f"{prot['DI_outcome'].min():.2f}; {prot['EOD_baseline'].max():.3f}",
            "threshold": "DI < 0.80 or EOD > 0.10", "detected": det, "table": t}


def detect_imbalance(df, weights=None):
    y = df["target"].to_numpy(int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    n1, n0 = w[y == 1].sum(), w[y == 0].sum()
    ir = float(max(n0, n1) / max(min(n0, n1), 1e-9))
    return {"statistic": "imbalance ratio (majority/minority)", "value": ir, "threshold": "> 1.5",
            "detected": bool(ir > 1.5), "counts": {0: float(n0), 1: float(n1)}}

# CELL 16
with Timer("raw detection"):
    raw_train, raw_test = make_split(DATA)
    feats_now = FEATS
    DETECT_RAW = {
        "Duplicate": detect_duplicates(DATA),
        "Missing data": detect_missing(raw_train, feats_now),
        "Outlier": detect_outliers(raw_train, feats_now),
        "Data noise": detect_label_noise(raw_train, feats_now),
        "Feature scale": detect_scale(raw_train, feats_now),
        "Encoding / domain": detect_domain(raw_train, feats_now),
        "Demographic": detect_demographic(raw_train, feats_now),
        "Class imbalance": detect_imbalance(raw_train),
    }
detect_table = pd.DataFrame([{"bias": k, "statistic": v["statistic"],
                              "value": (f"{v['value']:.4f}" if isinstance(v["value"], float) else v["value"]),
                              "threshold": v["threshold"], "detected": "YES" if v["detected"] else "no"}
                             for k, v in DETECT_RAW.items()]).set_index("bias")
print(f"Raw split: train {len(raw_train):,} | test {len(raw_test):,}")
detect_table

# CELL 18
DUP_FAMILY = {"Exact duplicate removal": "Traditional", "Fuzzy matching": "Traditional",
              "GBER (graph-based ER)": "Traditional", "Ditto (transformer matcher)": "Modern",
              "GNN entity resolution": "Modern", "LSH (MinHash)": "Modern"}
DUP_DIRS = {"Duplicate rate": "min", "Redundancy reduction": "max", "Precision": "max", "Recall": "max",
            "F1": "max", "Pair completeness": "max", "Reduction ratio": "max"}
ER_COLS = [c for c in DATA.columns if c not in ID_COLS and c != "w"]
GUARD_COLS = [c for c in META if c not in ID_COLS] + [c for c in SENS] + ["target"]   # same respondent => same design cell, demographics, outcome


def guard_matches(df, matches, max_diff=None):
    """Respondent-key guard: a matched pair is accepted as a duplicate only if both records agree on every
    respondent-level design column (survey weight, strata, PSU, site / patient). A true copy of a record
    shares them; two different people with similar answers almost never do. This prevents mass removal
    of look-alike records when real duplicates are rare. At most 15% of the predictors (minimum 1) may differ."""
    if not matches:
        return set()
    max_diff = max(1, int(0.15 * len(FEATS))) if max_diff is None else max_diff
    key = df[GUARD_COLS].astype(str).agg("|".join, axis=1).to_numpy() if GUARD_COLS else None
    Fm = df[[c for c in FEATS if c in df]].to_numpy(float)
    out = set()
    for a, b in matches:
        if key is not None and key[a] != key[b]:
            continue
        fa, fb = Fm[a], Fm[b]
        both = ~np.isnan(fa) & ~np.isnan(fb)
        diff = both & (np.abs(fa - fb) > 0.02 * np.maximum(np.maximum(np.abs(fa), np.abs(fb)), 1e-9))
        if diff.sum() <= max_diff:          # a copy of the same record with a few corrupted fields
            out.add((a, b))
    return out


def er_tokens(df):
    toks = []
    cont = set(cols_of("cont"))
    for row in df[ER_COLS].itertuples(index=False):
        t = []
        for c, v in zip(ER_COLS, row):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                t.append(f"{c}=∅")
            elif c in cont or (isinstance(v, float) and not float(v).is_integer()):
                t.append(f"{c}={float(v):.4g}")
            else:
                t.append(f"{c}={v}")
        toks.append(t)
    return toks


def er_vectors(df):
    num = df[ER_COLS].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any()]
    A = num.to_numpy(float)
    Z = Standardizer().fit(A).transform(A)
    return np.c_[np.nan_to_num(Z), np.isnan(A).astype(float)]


def attr_agreement(df, pairs):
    """Per-attribute agreement vector for record pairs (1 equal, 0 different, 0.5 if either missing)."""
    A = df[ER_COLS].to_numpy(object)
    out = np.zeros((len(pairs), len(ER_COLS)), np.float32)
    for k, (i, j) in enumerate(pairs):
        for c in range(len(ER_COLS)):
            a, b = A[i, c], A[j, c]
            na = a is None or (isinstance(a, float) and np.isnan(a))
            nb = b is None or (isinstance(b, float) and np.isnan(b))
            if na or nb:
                out[k, c] = 0.5
            elif isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
                out[k, c] = float(abs(float(a) - float(b)) <= 0.02 * max(abs(float(a)), abs(float(b)), 1e-9))
            else:
                out[k, c] = float(a == b)
    return out


def knn_pairs(V, k=10):
    k = min(k + 1, len(V))
    nn_ = NearestNeighbors(n_neighbors=k).fit(V)
    _, idx = nn_.kneighbors(V)
    pairs = {(min(i, j), max(i, j)) for i in range(len(V)) for j in idx[i, 1:] if i != j}
    return sorted(pairs)


def make_dup_benchmark(df, n_base, frac=0.10, seed=SEED):
    rng = np.random.default_rng(seed)
    base = df.drop_duplicates(subset=ER_COLS).sample(min(n_base, len(df)), random_state=seed).reset_index(drop=True)
    src = rng.choice(len(base), int(frac * len(base)), replace=False)
    copies = base.iloc[src].copy().reset_index(drop=True)
    cont = [c for c in cols_of("cont") if c in copies]
    disc = [c for c in cols_of("ord", "bin") if c in copies]
    for r in range(len(copies)):
        kind = rng.choice(["exact", "jitter", "missing", "flip"], p=[0.25, 0.35, 0.2, 0.2])
        if kind == "jitter" and cont:
            for c in rng.choice(cont, min(2, len(cont)), replace=False):
                v = copies.at[r, c]
                if pd.notna(v):
                    copies.at[r, c] = round(float(v) * (1 + rng.normal(0, 0.01)), 2)
        elif kind == "missing":
            c = rng.choice([c for c in FEATS if c in copies])
            copies.at[r, c] = np.nan
        elif kind == "flip" and disc:
            c = rng.choice(disc)
            allowed = SCHEMA[c]["allowed"]
            v = copies.at[r, c]
            if pd.notna(v) and len(allowed) > 1:
                i = allowed.index(float(v)) if float(v) in allowed else 0
                copies.at[r, c] = allowed[i + 1] if i + 1 < len(allowed) else allowed[i - 1]
    bench = pd.concat([base, copies], ignore_index=True)
    perm = rng.permutation(len(bench))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    bench = bench.iloc[perm].reset_index(drop=True)
    true_pairs = {(min(inv[s], inv[len(base) + r]), max(inv[s], inv[len(base) + r])) for r, s in enumerate(src)}
    return bench, true_pairs


def er_exact(df):
    key = df[ER_COLS].astype(str).agg("|".join, axis=1)
    pairs = set()
    for _, idx in key.groupby(key).groups.items():
        idx = sorted(idx)
        pairs |= {(a, b) for a, b in itertools.combinations(idx, 2)}
    return pairs, pairs


def er_fuzzy(df, window=10, threshold=97):
    from rapidfuzz import fuzz
    toks = er_tokens(df)
    strings = [" ".join(sorted(t)) for t in toks]
    cand = set()
    sort_cols = [c for c in (PROTECTED + cols_of("cont")) if c in df][:3]
    for keycols in (sort_cols, sort_cols[::-1]):
        order = df.sort_values(keycols, na_position="last").index.to_numpy()
        for i in range(len(order)):
            for j in order[i + 1:i + window]:
                a, b = int(order[i]), int(j)
                cand.add((min(a, b), max(a, b)))
    matches = {p for p in cand if fuzz.token_sort_ratio(strings[p[0]], strings[p[1]]) >= threshold}
    return cand, matches


def er_gber(df, k=10, threshold=0.9):
    cand = knn_pairs(er_vectors(df), k)
    agree = attr_agreement(df, cand)
    sim = agree.mean(1)
    edges = [p for p, s in zip(cand, sim) if s >= threshold]
    if not edges:
        return set(cand), set()
    r, c = zip(*edges)
    g = coo_matrix((np.ones(len(edges)), (r, c)), shape=(len(df), len(df)))
    _, lab = connected_components(g, directed=False)
    matches = set()
    for comp in pd.Series(np.arange(len(df))).groupby(lab).groups.values():
        if len(comp) > 1:
            matches |= {(a, b) for a, b in itertools.combinations(sorted(comp), 2)}
    return set(cand), matches


def _pair_training_set(df_train, seed):
    bench, tp = make_dup_benchmark(df_train, min(len(df_train), S("dup_train")), frac=0.15, seed=seed + 7)
    cand = knn_pairs(er_vectors(bench), 10)
    cand = sorted(set(cand) | tp)
    labels = np.array([int(p in tp) for p in cand])
    return bench, cand, labels


class TinyPairTransformer(nn.Module):
    def __init__(self, vocab, d=64, L=2, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.out = nn.Linear(d, 1)

    def forward(self, ids):
        mask = ids == 0
        h = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.enc(h, src_key_padding_mask=mask)
        return self.out(h[:, 0]).squeeze(-1)


def er_ditto(df, df_pool, seed=SEED, epochs=8):
    """Ditto-style matcher: records serialised as '[COL] name [VAL] value' and a transformer
    classifies each candidate pair. Uses the pre-trained BERT-tiny language model when it can
    be loaded (as in Ditto); otherwise a small transformer trained from scratch. Training pairs:
    every true pair plus 5x as many hard negatives (nearest-neighbour non-duplicates)."""
    tr_bench, tr_pairs, tr_lab = _pair_training_set(df_pool, seed)
    rng_d = np.random.default_rng(seed)
    pos, neg = np.where(tr_lab == 1)[0], np.where(tr_lab == 0)[0]
    keep = np.r_[pos, rng_d.choice(neg, min(len(neg), 5 * len(pos)), replace=False)]
    tr_pairs, tr_lab = [tr_pairs[i] for i in keep], tr_lab[keep]
    cand = knn_pairs(er_vectors(df), 10)

    def ser(d):
        return [" ".join(f"[COL] {t.split('=')[0]} [VAL] {t.split('=', 1)[1]}" for t in toks) for toks in er_tokens(d)]
    s_tr, s_ev = ser(tr_bench), ser(df)
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertConfig, BertForSequenceClassification
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()          # the new classifier head is expected to be untrained
        hf_logging.disable_progress_bar()
        cfg = BertConfig.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
        model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", config=cfg).to(DEVICE)
        wp = BertWordPieceTokenizer(hf_hub_download("prajjwal1/bert-tiny", "vocab.txt"), lowercase=True)
        wp.enable_truncation(max_length=512)
        wp.no_padding()
        CLS, SEP = wp.token_to_id("[CLS]"), wp.token_to_id("[SEP]")
        # Each record is tokenised once and kept to its first 254 word-pieces (Ditto's ~256-token budget per
        # record); pairs are assembled from cached ids instead of re-tokenising long records for every pair.
        tok_cache = {}

        def rec_ids(strings, key):
            if key not in tok_cache:
                tok_cache[key] = [o.ids[:254] for o in wp.encode_batch(strings, add_special_tokens=False)]
            return tok_cache[key]

        def enc(strings, pairs):
            ids = rec_ids(strings, id(strings))
            seqs = [[CLS] + ids[i] + [SEP] + ids[j] + [SEP] for i, j in pairs]
            typ = [[0] * (len(ids[i]) + 2) + [1] * (len(ids[j]) + 1) for i, j in pairs]
            L = max(len(s) for s in seqs)
            return {"input_ids": torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=DEVICE),
                    "attention_mask": torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=DEVICE),
                    "token_type_ids": torch.tensor([t + [0] * (L - len(t)) for t in typ], device=DEVICE)}
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        posw = torch.tensor([1.0, float((tr_lab == 0).sum() / max(tr_lab.sum(), 1))], device=DEVICE)
        model.train()
        for _ in range(E(epochs)):
            for b in np.array_split(np.random.permutation(len(tr_pairs)), max(1, len(tr_pairs) // 64)):
                x = enc(s_tr, [tr_pairs[i] for i in b])
                loss = F.cross_entropy(model(**x).logits, torch.tensor(tr_lab[b], device=DEVICE), weight=posw)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        probs = []
        with torch.no_grad():
            for b in np.array_split(np.arange(len(cand)), max(1, len(cand) // 256)):
                x = enc(s_ev, [cand[i] for i in b])
                probs.append(torch.softmax(model(**x).logits, -1)[:, 1].cpu().numpy())
        prob = np.concatenate(probs)
        ditto_backend = "BERT-tiny (pre-trained)"
    except Exception as e:  # offline fallback
        print("  Ditto: pre-trained LM unavailable -> transformer from scratch:", type(e).__name__)
        vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
        for s in s_tr + s_ev:
            for t in s.split():
                vocab.setdefault(t, len(vocab))

        def ids(strings, pairs):
            seqs = [([vocab["[CLS]"]] + [vocab[t] for t in strings[i].split()] + [vocab["[SEP]"]]
                     + [vocab[t] for t in strings[j].split()])[:512] for i, j in pairs]
            L = max(len(s) for s in seqs)
            return torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=DEVICE)
        model = TinyPairTransformer(len(vocab) + 1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pw = torch.tensor(float((tr_lab == 0).sum() / max(tr_lab.sum(), 1)), device=DEVICE)
        for _ in range(E(epochs * 2)):
            for b in np.array_split(np.random.permutation(len(tr_pairs)), max(1, len(tr_pairs) // 64)):
                loss = F.binary_cross_entropy_with_logits(model(ids(s_tr, [tr_pairs[i] for i in b])),
                                                          torch.tensor(tr_lab[b], device=DEVICE, dtype=torch.float32), pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            prob = np.concatenate([torch.sigmoid(model(ids(s_ev, [cand[i] for i in b]))).cpu().numpy()
                                   for b in np.array_split(np.arange(len(cand)), max(1, len(cand) // 256))])
        ditto_backend = "transformer from scratch"
    print("  Ditto backend:", ditto_backend)
    return set(cand), {p for p, s in zip(cand, prob) if s >= 0.5}


class SAGE(nn.Module):
    def __init__(self, d_in, d=64):
        super().__init__()
        self.l1 = nn.Linear(2 * d_in, d)
        self.l2 = nn.Linear(2 * d, d)

    def forward(self, x, adj):
        h = F.relu(self.l1(torch.cat([x, torch.sparse.mm(adj, x)], 1)))
        return self.l2(torch.cat([h, torch.sparse.mm(adj, h)], 1))


class EdgeHead(nn.Module):
    def __init__(self, d, d_agree):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d + d_agree, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, h, pairs, agree):
        i, j = pairs[:, 0], pairs[:, 1]
        return self.mlp(torch.cat([(h[i] - h[j]).abs(), h[i] * h[j], agree], 1)).squeeze(-1)


def _graph(df, pairs):
    V = torch.tensor(er_vectors(df), dtype=torch.float32, device=DEVICE)
    P = np.array(pairs)
    r = np.r_[P[:, 0], P[:, 1]]
    c = np.r_[P[:, 1], P[:, 0]]
    deg = np.bincount(r, minlength=len(df)).astype(np.float32)
    vals = 1.0 / np.maximum(deg[r], 1)
    adj = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(vals), (len(df), len(df))).coalesce().to(DEVICE)
    return V, adj, torch.tensor(P, device=DEVICE), torch.tensor(attr_agreement(df, pairs), device=DEVICE)


def er_gnn(df, df_pool, seed=SEED, epochs=200):
    """GNN entity resolution: GraphSAGE node embeddings on the k-NN record graph + an edge
    classifier on (|h_i-h_j|, h_i*h_j, attribute agreement). Trained on a disjoint labelled graph."""
    set_seed(seed)
    tr_bench, tr_pairs, tr_lab = _pair_training_set(df_pool, seed)
    Vt, At, Pt, Gt = _graph(tr_bench, tr_pairs)
    cand = knn_pairs(er_vectors(df), 10)
    Ve, Ae, Pe, Ge = _graph(df, cand)
    if Vt.shape[1] != Ve.shape[1]:
        d = max(Vt.shape[1], Ve.shape[1])
        Vt, Ve = F.pad(Vt, (0, d - Vt.shape[1])), F.pad(Ve, (0, d - Ve.shape[1]))
    enc, head = SAGE(Vt.shape[1]).to(DEVICE), EdgeHead(64, Gt.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=5e-3)
    y = torch.tensor(tr_lab, dtype=torch.float32, device=DEVICE)
    pw = torch.tensor(float((tr_lab == 0).sum() / max(tr_lab.sum(), 1)), device=DEVICE)
    for _ in range(E(epochs)):
        loss = F.binary_cross_entropy_with_logits(head(enc(Vt, At), Pt, Gt), y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(head(enc(Ve, Ae), Pe, Ge)).cpu().numpy()
    return set(cand), {p for p, s in zip(cand, prob) if s >= 0.5}


def er_lsh(df, threshold=0.7, verify=0.8, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    toks = er_tokens(df)
    mh = []
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, t in enumerate(toks):
        m = MinHash(num_perm=num_perm, seed=SEED)
        for s in t:
            m.update(s.encode("utf8"))
        mh.append(m)
        lsh.insert(str(i), m)
    cand = set()
    for i, m in enumerate(mh):
        for j in lsh.query(m):
            j = int(j)
            if j != i:
                cand.add((min(i, j), max(i, j)))
    return cand, {p for p in cand if mh[p[0]].jaccard(mh[p[1]]) >= verify}


def clusters_to_remove(n, matches):
    if not matches:
        return set()
    r, c = zip(*matches)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    remove = set()
    for comp in pd.Series(np.arange(n)).groupby(lab).groups.values():
        if len(comp) > 1:
            remove |= set(sorted(comp)[1:])
    return remove


def er_metrics(n, true_pairs, cand, matches):
    tp = len(matches & true_pairs)
    prec = tp / len(matches) if matches else 0.0
    rec = tp / len(true_pairs) if true_pairs else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    removed = clusters_to_remove(n, matches)
    remaining = sum(1 for a, b in true_pairs if a not in removed and b not in removed)
    return {"Duplicate rate": remaining / max(n - len(removed), 1),
            "Redundancy reduction": 1 - remaining / max(len(true_pairs), 1),
            "Precision": prec, "Recall": rec, "F1": f1,
            "Pair completeness": len(cand & true_pairs) / max(len(true_pairs), 1),
            "Reduction ratio": 1 - len(cand) / (n * (n - 1) / 2)}


ER_METHODS = {"Exact duplicate removal": lambda d, pool: er_exact(d), "Fuzzy matching": lambda d, pool: er_fuzzy(d),
              "GBER (graph-based ER)": lambda d, pool: er_gber(d), "Ditto (transformer matcher)": er_ditto,
              "GNN entity resolution": er_gnn, "LSH (MinHash)": lambda d, pool: er_lsh(d)}

# CELL 19
DUP = detect_duplicates(DATA)
print(f"Exact duplicate records: {DUP['value']:,} ({DUP['rate']:.4%}) | profile collisions (different people, kept): {DUP['profile_collisions']:,}")
dup_before = DATA.copy()
dup_tab, dup_best, dup_trad = None, None, None
dup_removed = []
if DUP["detected"] or FORCE_DETECT:
    pool = DATA.loc[~DUP["mask"]].reset_index(drop=True)
    bench_pool, eval_pool = train_test_split(pool, test_size=0.5, random_state=SEED)   # disjoint records for training learned matchers
    bench, true_pairs = make_dup_benchmark(eval_pool, S("dup_bench"))
    print(f"Benchmark: {len(bench):,} records with {len(true_pairs):,} injected duplicate pairs")
    rows = {}
    for name, fn in ER_METHODS.items():
        with Timer(f"duplicate:{name}"):
            cand, matches = fn(bench, bench_pool)
            raw_n = len(matches)
            matches = guard_matches(bench, matches)
            rows[name] = er_metrics(len(bench), true_pairs, cand, matches)
            rows[name]["matches rejected by guard (info)"] = raw_n - len(matches)
    dup_tab, dup_best, dup_trad = rank_methods(pd.DataFrame(rows).T, DUP_DIRS, DUP_FAMILY)
    show_comparison(dup_tab, DUP_DIRS, dup_best, DUP_FAMILY, dup_trad)
    with Timer("duplicate:apply"):
        real = DATA.reset_index(drop=True)
        _, matches_raw = ER_METHODS[dup_best](real, bench_pool)
        matches = guard_matches(real, matches_raw)
        print(f"{dup_best}: {len(matches_raw):,} candidate matches, {len(matches):,} pass the respondent-key guard")
        remove = clusters_to_remove(len(real), matches)
        exact_idx = set(np.where(DUP["mask"])[0])
        print(f"{dup_best} removed {len(remove):,} records; {len(remove & exact_idx):,} of the {len(exact_idx):,} exact duplicates")
        dup_removed = sorted(remove)
        DATA = real.drop(index=sorted(remove)).reset_index(drop=True)
else:
    print("Duplicate bias NOT detected -> no records removed (After = Before).")
dup_after = DATA
DUP_AFTER = detect_duplicates(DATA)
RUN_LOG["Duplicate"] = {"detected": DUP["detected"], "statistic": DUP["statistic"], "before": DUP["value"],
                        "after": DUP_AFTER["value"], "selected": dup_best or "—", "best_traditional": dup_trad or "—",
                        "methods_compared": len(ER_METHODS) if DUP["detected"] else 0}


def duplicate_grid(before, removed, method):
    """Rows = duplicate checks, cells = the same example records Before and After (full dataset, step precedes the split)."""
    b = before.reset_index(drop=True)
    a = b.drop(index=removed)
    rows = {"Exact duplicate": lambda df: detect_duplicates(df)["mask"],
            "Look-alike (kept)": lambda df: df.duplicated(subset=FEATS + ["target"], keep="first").to_numpy()
                                            & ~detect_duplicates(df)["mask"]}
    if method:
        rows[f"Removed by {method}"[:26]] = lambda df: df.index.isin(removed)
    st = {"v": 1, "label": "Duplicate candidate", "color": "#2563C8", "border": "#1D4FA0"}
    return var_grid("Duplicate Bias Heatmap", b, a, list(rows), lambda df, c: rows[c](df), st, "of records",
                    "Rows = duplicate checks; each cell = one record of the full dataset (the same records in Before and After). "
                    "Coloured = the record is a duplicate candidate under that check: an exact copy (all columns except the record ID) "
                    "or the same answers from a different respondent (look-alike), which the respondent-key guard keeps. "
                    "Only removed records disappear in After.",
                    "Not duplicate")


write_dashboard(
    "06_duplicate_bias_dashboard.html", "Duplicate Bias", DUP["detected"],
    (f"Detected: {DUP['value']:,} exact duplicate records — {dup_best} selected (best-ranked modern method; best traditional for comparison: {dup_trad}) and applied" if DUP["detected"]
     else f"Not detected: 0 exact duplicate records (threshold > 0) — no records removed, After = Before"),
    {"Statistic": DUP["statistic"], "Value (before)": f"{DUP['value']:,}", "Threshold": DUP["threshold"],
     "Value (after)": f"{DUP_AFTER['value']:,}", "Profile collisions (kept)": f"{DUP['profile_collisions']:,}"},
    dup_best, dup_before, dup_after, display_vars(FEATS),
    duplicate_grid(dup_before, dup_removed, dup_best),
    comparison=comparison_payload(dup_tab, DUP_DIRS, dup_best, DUP_FAMILY, dup_trad),
    footer="This step runs on the full dataset before the train/test split, so copies cannot leak across it. "
           "Rows shown are the full dataset.",
    info_extra={"Rows scope": "full dataset (step precedes split)"},
    chips=[f"Duplicates: <b>{DUP['value']:,} → {DUP_AFTER['value']:,}</b>", f"Records removed: <b>{len(dup_removed):,}</b>"])

# CELL 21
TR, TE = make_split(DATA)
TR, VA = make_split(TR, test_size=0.10, seed=SEED + 1)
TR["w"], TE["w"], VA["w"] = 1.0, 1.0, 1.0
TR["meta_role"], TE["meta_role"], VA["meta_role"] = "train", "test", "validation"
TE = pd.concat([TE, VA])          # validation rows travel with the test rows through every transform


def part(df, role):
    """Test or validation rows of TE (both always receive the same transforms)."""
    return df[df["meta_role"] == role]


TR_RAW, TE_RAW, FEATS_RAW_ENC = TR.copy(), TE.copy(), list(FEATS)      # kept for the end-to-end check
print(f"train {len(TR):,} (positive {TR.target.mean():.3f}) | validation {len(part(TE, 'validation')):,} | "
      f"test {len(part(TE, 'test')):,} (positive {part(TE, 'test').target.mean():.3f})")

# CELL 23
class BaseImputer:
    stochastic = False

    def sample(self, Z, m, seed=SEED):
        z = self.transform(Z)
        return [z] * m


class GMMImputer(BaseImputer):
    """Gaussian-mixture imputation: conditional expectation (or draw) of the missing block given the
    observed block, mixed with component responsibilities computed from the observed block."""
    stochastic = True

    def __init__(self, k=5, seed=SEED, fit_cap=80000):
        self.k, self.seed, self.cap = k, seed, fit_cap

    def fit(self, Z):
        rng = np.random.default_rng(self.seed)
        comp = Z[~np.isnan(Z).any(1)]
        if len(comp) < 20 * Z.shape[1]:
            comp = np.where(np.isnan(Z), 0.0, Z)
        if len(comp) > self.cap:
            comp = comp[rng.choice(len(comp), self.cap, replace=False)]
        self.diag = WIDE      # wide data: diagonal covariances (almost every row has its own missingness pattern)
        self.g = GaussianMixture(self.k, covariance_type="diag" if self.diag else "full", reg_covar=1e-2, max_iter=200,
                                 random_state=self.seed).fit(comp)
        return self

    def _run_diag(self, Z, rng=None):
        """Diagonal-covariance GMM: responsibilities from the observed dimensions; missing values = responsibility-
        weighted component means (or a draw from the sampled component). Vectorised over all rows."""
        out = Z.copy()
        M = np.isnan(Z)
        mu, var, w = self.g.means_, self.g.covariances_, self.g.weights_
        for s in range(0, len(Z), 20000):
            Zs, Ms = np.nan_to_num(Z[s:s + 20000]), ~M[s:s + 20000]
            ll = -0.5 * (((Zs[:, None, :] - mu[None]) ** 2 / var[None] + np.log(2 * np.pi * var[None])) * Ms[:, None, :]).sum(2) + np.log(w)
            ll -= ll.max(1, keepdims=True)
            R = np.exp(ll)
            R /= R.sum(1, keepdims=True)
            if rng is None:
                fill = R @ mu
            else:
                ks = (R.cumsum(1) > rng.random(len(R))[:, None]).argmax(1)
                fill = mu[ks] + rng.standard_normal(Zs.shape) * np.sqrt(var[ks])
            blk = out[s:s + 20000]
            blk[M[s:s + 20000]] = fill[M[s:s + 20000]]
            out[s:s + 20000] = blk
        return out

    def _run(self, Z, rng=None):
        if self.diag:
            return self._run_diag(Z, rng)
        out = Z.copy()
        M = np.isnan(Z)
        rows = np.where(M.any(1))[0]
        if not len(rows):
            return out
        pats, inv = np.unique(M[rows], axis=0, return_inverse=True)
        inv = inv.ravel()
        mu, S_, w = self.g.means_, self.g.covariances_, self.g.weights_
        for pi, p in enumerate(pats):
            r = rows[inv == pi]
            o, m = np.where(~p)[0], np.where(p)[0]
            Xo = Z[np.ix_(r, o)]
            lps, cms, ccs = [], [], []
            for k in range(self.k):
                if len(o):
                    Soo, Smo, Smm = S_[k][np.ix_(o, o)], S_[k][np.ix_(m, o)], S_[k][np.ix_(m, m)]
                    Sinv = np.linalg.pinv(Soo)
                    cms.append(mu[k][m] + (Xo - mu[k][o]) @ (Smo @ Sinv).T)
                    ccs.append(Smm - Smo @ Sinv @ Smo.T)
                    lps.append(np.atleast_1d(stats.multivariate_normal(mu[k][o], Soo, allow_singular=True).logpdf(Xo)) + np.log(w[k]))
                else:
                    cms.append(np.repeat(mu[k][m][None], len(r), 0))
                    ccs.append(S_[k][np.ix_(m, m)])
                    lps.append(np.full(len(r), np.log(w[k])))
            L = np.vstack(lps).T
            L -= L.max(1, keepdims=True)
            R = np.exp(L)
            R /= R.sum(1, keepdims=True)
            if rng is None:
                val = sum(R[:, [k]] * cms[k] for k in range(self.k))
            else:
                ks = (R.cumsum(1) > rng.random(len(r))[:, None]).argmax(1)
                val = np.empty((len(r), len(m)))
                for k in range(self.k):
                    idx = ks == k
                    if idx.any():
                        cov = (ccs[k] + ccs[k].T) / 2 + 1e-6 * np.eye(len(m))
                        val[idx] = cms[k][idx] + rng.multivariate_normal(np.zeros(len(m)), cov, int(idx.sum()), method="eigh")
            out[np.ix_(r, m)] = val
        return out

    def transform(self, Z):
        return self._run(Z)

    def sample(self, Z, m, seed=SEED):
        return [self._run(Z, np.random.default_rng(seed + i)) for i in range(m)]


def _cap_rows(Z, cap, seed):
    if len(Z) <= cap:
        return Z
    return Z[np.random.default_rng(seed).choice(len(Z), cap, replace=False)]


class RFImputer(BaseImputer):
    """missForest-style chained random-forest imputation (IterativeImputer + RandomForestRegressor)."""
    def __init__(self, seed=SEED, cap=30000):
        self.seed, self.cap = seed, (12000 if WIDE else cap)

    def fit(self, Z):
        self.imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=E(30 if WIDE else 60), max_depth=14, min_samples_leaf=3,
                                            max_features=0.3 if WIDE else 0.6, n_jobs=-1, random_state=self.seed),
            max_iter=(E(3) if WIDE else E(5)) if not FAST else 2, initial_strategy="median", random_state=self.seed, skip_complete=True, n_nearest_features=30 if WIDE else None,
            keep_empty_features=True)
        self.imp.fit(_cap_rows(Z, self.cap, self.seed))
        return self

    def transform(self, Z):
        return self.imp.transform(Z)


class MICEImputer(BaseImputer):
    """MICE: chained Bayesian-ridge regressions with posterior sampling; m independent chains."""
    stochastic = True

    def __init__(self, seed=SEED, m=5, cap=150000):
        self.seed, self.m, self.cap = seed, m, cap

    def fit(self, Z):
        Zf = _cap_rows(Z, self.cap, self.seed)
        self.imps = [IterativeImputer(BayesianRidge(), sample_posterior=True, max_iter=E(5) if WIDE else E(10), random_state=self.seed + i, n_nearest_features=30 if WIDE else None,
                                      skip_complete=True, keep_empty_features=True).fit(Zf) for i in range(self.m)]
        return self

    def transform(self, Z):
        return np.mean([imp.transform(Z) for imp in self.imps], 0)

    def sample(self, Z, m, seed=SEED):
        return [self.imps[i % self.m].transform(Z) for i in range(m)]


class KNNImp(BaseImputer):
    def __init__(self, seed=SEED, cap=25000, k=5):
        self.seed, self.cap, self.k = seed, cap, k

    def fit(self, Z):
        self.imp = KNNImputer(n_neighbors=self.k, weights="distance", keep_empty_features=True).fit(_cap_rows(Z, self.cap, self.seed))
        return self

    def transform(self, Z):
        return np.vstack([self.imp.transform(Z[i:i + 20000]) for i in range(0, len(Z), 20000)])


class HyperImputeImp(BaseImputer):
    """HyperImpute (Jarrett et al., 2022): column-wise iterative imputation with automatic model
    selection per column. Learner pool: random forest for categorical columns (HyperImpute's
    logistic-regression, XGBoost, CatBoost and LightGBM classifier plugins call a scikit-learn
    argument removed in scikit-learn >= 1.7), linear regression + random forest for numeric columns."""
    def __init__(self, seed=SEED, cap=20000):
        self.seed, self.cap = seed, (6000 if WIDE else cap)

    def fit(self, Z):
        from hyperimpute.plugins.imputers import Imputers
        self.plugin = Imputers().get("hyperimpute", optimizer="simple", random_state=self.seed,
                                     n_inner_iter=(E(5) if WIDE else E(20)) if not FAST else 3,
                                     classifier_seed=["random_forest"],
                                     regression_seed=["linear_regression", "random_forest_regressor"],
                                     class_threshold=5)
        self.Zfit = _cap_rows(Z, self.cap, self.seed)
        self.plugin.fit(pd.DataFrame(self.Zfit))
        return self

    def transform(self, Z):
        outs = []
        for i in range(0, len(Z), 20000):
            outs.append(np.asarray(self.plugin.transform(pd.DataFrame(Z[i:i + 20000])), float))
        return np.vstack(outs)


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, h=128, act=nn.ReLU, out_act=None, layers=2):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, h), act()]
            d = h
        mods.append(nn.Linear(d, d_out))
        if out_act is not None:
            mods.append(out_act)
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class GAINImp(BaseImputer):
    """GAIN (Yoon et al., 2018): generator imputes, discriminator guesses which entries were observed
    (with a hint mechanism); loss = adversarial + alpha * reconstruction of observed entries."""
    stochastic = True

    def __init__(self, seed=SEED, iters=4000, batch=256, hint=0.9, alpha=100):
        self.seed, self.iters, self.batch, self.hint, self.alpha = seed, iters, batch, hint, alpha

    def _norm(self, Z):
        return (Z - self.lo) / self.rg

    def fit(self, Z):
        set_seed(self.seed)
        self.lo, hi = np.nanmin(Z, 0), np.nanmax(Z, 0)
        self.lo = np.nan_to_num(self.lo)
        self.rg = np.where(np.nan_to_num(hi - self.lo) > 1e-6, np.nan_to_num(hi - self.lo), 1.0)
        N = self._norm(Z)
        M = (~np.isnan(N)).astype(np.float32)
        X = torch.tensor(np.nan_to_num(N), dtype=torch.float32, device=DEVICE)
        Mt = torch.tensor(M, device=DEVICE)
        d = Z.shape[1]
        self.G = _MLP(2 * d, d, h=max(d, 32), out_act=nn.Sigmoid()).to(DEVICE)
        Dn = _MLP(2 * d, d, h=max(d, 32), out_act=nn.Sigmoid()).to(DEVICE)
        og, od = torch.optim.Adam(self.G.parameters(), 1e-3), torch.optim.Adam(Dn.parameters(), 1e-3)
        for _ in range(E(self.iters)):
            b = torch.randint(0, len(X), (self.batch,), device=DEVICE)
            x, m = X[b], Mt[b]
            z = torch.rand_like(x) * 0.01
            xt = m * x + (1 - m) * z
            hb = (torch.rand_like(x) < self.hint).float()
            h = hb * m + 0.5 * (1 - hb)
            g = self.G(torch.cat([xt, m], 1))
            xh = m * x + (1 - m) * g
            dp = Dn(torch.cat([xh.detach(), h], 1))
            ld = -torch.mean(m * torch.log(dp + 1e-8) + (1 - m) * torch.log(1 - dp + 1e-8))
            od.zero_grad(); ld.backward(); od.step()
            dp = Dn(torch.cat([xh, h], 1))
            lg = -torch.mean((1 - m) * torch.log(dp + 1e-8)) + self.alpha * torch.mean((m * x - m * g) ** 2) / torch.mean(m)
            og.zero_grad(); lg.backward(); og.step()
        return self

    def _run(self, Z, seed):
        g_ = torch.Generator(device=DEVICE).manual_seed(int(seed))
        N = self._norm(Z)
        M = torch.tensor((~np.isnan(N)).astype(np.float32), device=DEVICE)
        X = torch.tensor(np.nan_to_num(N), dtype=torch.float32, device=DEVICE)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                x, m = X[i:i + 50000], M[i:i + 50000]
                z = torch.rand(x.shape, generator=g_, device=DEVICE) * 0.01
                g = self.G(torch.cat([m * x + (1 - m) * z, m], 1))
                outs.append((m * x + (1 - m) * g).cpu().numpy())
        return np.vstack(outs) * self.rg + self.lo

    def transform(self, Z):
        return self._run(Z, self.seed)

    def sample(self, Z, m, seed=SEED):
        return [self._run(Z, seed + i) for i in range(m)]


class CSDINet(nn.Module):
    """Tabular CSDI denoiser: one token per feature, conditioned on observed values, the condition
    mask, a learned feature embedding and the diffusion step; transformer attention across features."""
    def __init__(self, d, C=64, layers=2, heads=4, T=50):
        super().__init__()
        self.inp = nn.Linear(3, C)
        self.feat = nn.Parameter(torch.randn(d, C) * 0.02)
        self.step = nn.Embedding(T, C)
        layer = nn.TransformerEncoderLayer(C, heads, 4 * C, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(C, 1)

    def forward(self, xn, xc, cm, t):
        h = self.inp(torch.stack([xn * (1 - cm), xc * cm, cm], -1)) + self.feat[None] + self.step(t)[:, None, :]
        return self.out(self.tr(h)).squeeze(-1)


class CSDIImp(BaseImputer):
    """CSDI (Tashiro et al., 2021): conditional score-based diffusion imputation, trained by randomly
    hiding observed values and learning to denoise them given the rest; imputation = mean of samples."""
    stochastic = True

    def __init__(self, seed=SEED, T=50, iters=4000, batch=256, n_samples=5, cap=100000):
        self.seed, self.T, self.iters, self.batch, self.ns, self.cap = seed, T, iters, batch, n_samples, cap
        beta = torch.linspace(1e-4 ** 0.5, 0.5 ** 0.5, T) ** 2
        self.beta, self.alpha = beta.to(DEVICE), (1 - beta).to(DEVICE)
        self.abar = torch.cumprod(self.alpha, 0)

    def fit(self, Z):
        set_seed(self.seed)
        Zf = _cap_rows(Z, self.cap, self.seed)
        X = torch.tensor(np.nan_to_num(Zf), dtype=torch.float32, device=DEVICE)
        Mo = torch.tensor((~np.isnan(Zf)).astype(np.float32), device=DEVICE)
        self.net = CSDINet(Z.shape[1], T=self.T).to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        for _ in range(E(self.iters)):
            b = torch.randint(0, len(X), (self.batch,), device=DEVICE)
            x, mo = X[b], Mo[b]
            ratio = torch.rand(len(b), 1, device=DEVICE)
            tgt = (torch.rand_like(x) < ratio).float() * mo
            cm = mo - tgt
            t = torch.randint(0, self.T, (len(b),), device=DEVICE)
            eps = torch.randn_like(x)
            ab = self.abar[t][:, None]
            xn = ab.sqrt() * x + (1 - ab).sqrt() * eps
            pred = self.net(xn, x, cm, t)
            loss = (((eps - pred) ** 2) * tgt).sum() / tgt.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _run(self, Z, seed):
        g_ = torch.Generator(device=DEVICE).manual_seed(int(seed))
        out = Z.copy()
        rows = np.where(np.isnan(Z).any(1))[0]
        bs = int(max(128, min(8192, 2_000_000 // max(Z.shape[1], 1) ** 2 * 4)))   # keeps attention memory bounded for wide data
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r = rows[i:i + bs]
                x0 = torch.tensor(np.nan_to_num(Z[r]), dtype=torch.float32, device=DEVICE)
                cm = torch.tensor((~np.isnan(Z[r])).astype(np.float32), device=DEVICE)
                x = torch.randn(x0.shape, generator=g_, device=DEVICE)
                for t in range(self.T - 1, -1, -1):
                    tt = torch.full((len(r),), t, device=DEVICE, dtype=torch.long)
                    eps = self.net(x, x0, cm, tt)
                    x = (x - self.beta[t] / (1 - self.abar[t]).sqrt() * eps) / self.alpha[t].sqrt()
                    if t > 0:
                        sig = ((1 - self.abar[t - 1]) / (1 - self.abar[t]) * self.beta[t]).sqrt()
                        x = x + sig * torch.randn(x.shape, generator=g_, device=DEVICE)
                v = (cm * x0 + (1 - cm) * x).cpu().numpy()
                out[r] = np.where(np.isnan(Z[r]), v, Z[r])
        return out

    def transform(self, Z):
        return np.mean(self.sample(Z, self.ns, self.seed), 0)

    def sample(self, Z, m, seed=SEED):
        return [self._run(Z, seed + i) for i in range(m)]


class MIPVAEImp(BaseImputer):
    """MIP-VAE: missingness-aware VAE imputation. The encoder sees zero-filled values plus the mask;
    the Gaussian decoder likelihood is computed on observed entries only; imputation iterates
    encode→decode a few times (pseudo-Gibbs) and multiple imputations sample the latent code."""
    stochastic = True

    def __init__(self, seed=SEED, latent=8, epochs=40, batch=512, cap=200000):
        self.seed, self.k, self.epochs, self.batch, self.cap = seed, latent, epochs, batch, cap

    def fit(self, Z):
        set_seed(self.seed)
        d = Z.shape[1]
        Zf = _cap_rows(Z, self.cap, self.seed)
        X = torch.tensor(np.nan_to_num(Zf), dtype=torch.float32, device=DEVICE)
        M = torch.tensor((~np.isnan(Zf)).astype(np.float32), device=DEVICE)
        self.enc = _MLP(2 * d, 2 * self.k, h=128).to(DEVICE)
        self.dec = _MLP(self.k, d, h=128).to(DEVICE)
        self.logvar = nn.Parameter(torch.zeros(d, device=DEVICE))
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()) + [self.logvar], 1e-3)
        steps = max(1, len(X) // self.batch)
        for ep in range(E(self.epochs)):
            perm = torch.randperm(len(X), device=DEVICE)
            for s in range(steps):
                b = perm[s * self.batch:(s + 1) * self.batch]
                x, m = X[b], M[b]
                drop = (torch.rand_like(m) < 0.2).float() * m          # extra masking teaches imputation
                mi = m - drop
                mu, lv = self.enc(torch.cat([x * mi, mi], 1)).chunk(2, 1)
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                xr = self.dec(z)
                nll = 0.5 * (((x - xr) ** 2) / self.logvar.exp() + self.logvar) * m
                kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
                loss = (nll.sum(1) + kl).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _run(self, Z, seed, stochastic):
        g_ = torch.Generator(device=DEVICE).manual_seed(int(seed))
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 50000):
                Zi = Z[i:i + 50000]
                m = torch.tensor((~np.isnan(Zi)).astype(np.float32), device=DEVICE)
                x0 = torch.tensor(np.nan_to_num(Zi), dtype=torch.float32, device=DEVICE)
                x = x0.clone()
                for _ in range(5):
                    mu, lv = self.enc(torch.cat([x * m, m], 1)).chunk(2, 1)
                    z = mu + (torch.randn(mu.shape, generator=g_, device=DEVICE) * (0.5 * lv).exp() if stochastic else 0)
                    x = m * x0 + (1 - m) * self.dec(z)
                out.append(x.cpu().numpy())
        return np.vstack(out)

    def transform(self, Z):
        return self._run(Z, self.seed, False)

    def sample(self, Z, m, seed=SEED):
        return [self._run(Z, seed + i, True) for i in range(m)]


MISS_METHODS = {"GMM": GMMImputer, "Random Forest": RFImputer, "GAIN": GAINImp, "MICE": MICEImputer, "KNN": KNNImp,
                "HyperImpute": HyperImputeImp, "CSDI": CSDIImp, "MIP-VAE": MIPVAEImp}
MISS_FAMILY = {k: ("Modern" if k in ("HyperImpute", "CSDI", "MIP-VAE") else "Traditional") for k in MISS_METHODS}
MISS_DIRS = {"Wasserstein": "min", "Correlation preservation |Δr|": "min", "MAE": "min", "RMSE": "min",
             "NRMSE": "min", "CRPS": "min", "NLL": "min"}


def crps_ensemble(E_, y):
    """CRPS of an ensemble (m x n) against observations y (n)."""
    t1 = np.abs(E_ - y[None]).mean(0)
    t2 = np.abs(E_[:, None, :] - E_[None, :, :]).mean((0, 1))
    return t1 - 0.5 * t2


def imputation_metrics(Zobs, draws, newmask, eval_cols, seed=SEED):
    rng = np.random.default_rng(seed)
    E_ = np.stack(draws)
    mu = E_.mean(0)
    res = {k: [] for k in ["W", "abs", "sq", "nrmse", "crps"]}
    nll_parts = []
    for j in eval_cols:
        r = np.where(newmask[:, j])[0]
        if len(r) < 5:
            continue
        y, p, Ej = Zobs[r, j], mu[r, j], E_[:, r, j]
        res["W"].append(wasserstein_distance(p, y))
        res["abs"].append(np.abs(p - y))
        res["sq"].append((p - y) ** 2)
        sd = np.nanstd(Zobs[:, j]) or 1.0
        res["nrmse"].append(np.sqrt(np.mean((p - y) ** 2)) / sd)
        res["crps"].append(crps_ensemble(Ej, y))
        v = Ej.var(0)
        half = rng.random(len(r)) < 0.5
        if half.sum() < 2 or (~half).sum() < 2:
            continue
        delta = 1e-3
        c = np.mean((y[half] - p[half]) ** 2 / (v[half] + delta))
        var = np.maximum(c * (v[~half] + delta), 1e-6)
        nll_parts.append(0.5 * np.log(2 * np.pi * var) + (y[~half] - p[~half]) ** 2 / (2 * var))
    Zt = Zobs.copy()
    Zi = mu.copy()
    realnan = np.isnan(Zobs)
    Zi[realnan] = np.nan
    ct = pd.DataFrame(Zt[:, eval_cols]).corr().to_numpy()
    ci = pd.DataFrame(Zi[:, eval_cols]).corr().to_numpy()
    iu = np.triu_indices(len(eval_cols), 1)
    dcor = np.nanmean(np.abs(ct[iu] - ci[iu]))
    ab, sq = np.concatenate(res["abs"]), np.concatenate(res["sq"])
    return {"Wasserstein": float(np.mean(res["W"])), "Correlation preservation |Δr|": float(dcor),
            "MAE": float(ab.mean()), "RMSE": float(np.sqrt(sq.mean())), "NRMSE": float(np.mean(res["nrmse"])),
            "CRPS": float(np.concatenate(res["crps"]).mean()),
            "NLL": float(np.concatenate(nll_parts).mean()) if nll_parts else np.nan}


def missing_grid(before, after, feats):
    rates = before[feats].isna().mean().sort_values(ascending=False)
    cols = list(rates[rates > 0].index[:10]) or display_vars(feats, cap=10)
    return var_grid("Missing Value Heatmap", before, after, cols, lambda df, c: df[c].isna().to_numpy(),
                    {"v": 1, "label": "Missing", "color": "#2563C8", "border": "#1D4FA0"}, "missing",
                    "Rows = the predictors with the most missing values; each cell = one training patient (the same patients "
                    "in Before and After). Coloured = that patient's value is missing. After imputation every cell is filled.",
                    "Observed")

# CELL 24
MISS = detect_missing(TR, FEATS)
print(f"Missing predictor cells: {MISS['value']:.3%}")
print(MISS["per_feature"][MISS["per_feature"] > 0].sort_values(ascending=False).head(15).apply(lambda v: f"{v:.2%}").to_string())
miss_before = TR.copy()
miss_tab, miss_best, miss_trad = None, None, None
if MISS["detected"] or FORCE_DETECT:
    A = TR[FEATS].to_numpy(float)
    std_m = Standardizer().fit(A)
    Zall = std_m.transform(A)
    rng = np.random.default_rng(SEED)
    rows = rng.choice(len(Zall), min(S("missing"), len(Zall)), replace=False)
    Zobs = Zall[rows]
    real_pats = np.isnan(Zall)[np.isnan(Zall).any(1)]
    P = real_pats[rng.integers(0, len(real_pats), len(rows))]
    newmask = P & ~np.isnan(Zobs)
    Zmask = Zobs.copy()
    Zmask[newmask] = np.nan
    eval_cols = [j for j, c in enumerate(FEATS) if SCHEMA[c]["type"] != "onehot"]
    print(f"Benchmark: {len(rows):,} rows, {newmask.sum():,} hidden observed cells "
          f"({newmask.mean():.1%} of cells) + {np.isnan(Zobs).sum():,} genuinely missing cells")
    res = {}
    for name, cls in MISS_METHODS.items():
        with Timer(f"missing:{name}"):
            try:
                imp = cls().fit(Zmask)
                draws = imp.sample(Zmask, 5, SEED) if imp.stochastic else [imp.transform(Zmask)]
                res[name] = imputation_metrics(Zobs, draws, newmask, eval_cols)
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in MISS_DIRS}
    miss_tab, miss_best, miss_trad = rank_methods(pd.DataFrame(res).T, MISS_DIRS, MISS_FAMILY)
    show_comparison(miss_tab, MISS_DIRS, miss_best, MISS_FAMILY, miss_trad)
    with Timer(f"missing:apply {miss_best}"):
        imp = MISS_METHODS[miss_best]().fit(Zall)
        for df in (TR, TE):
            Z = std_m.transform(df[FEATS].to_numpy(float))
            filled = std_m.inverse(imp.transform(Z))
            orig = df[FEATS].to_numpy(float)
            df[FEATS] = np.where(np.isnan(orig), filled, orig)
        TR[FEATS] = snap(TR[FEATS]).to_numpy()
        TE[FEATS] = snap(TE[FEATS]).to_numpy()
        for df in (TR, TE):                      # safety net for any value an imputer could not produce
            for c in FEATS:
                if df[c].isna().any():
                    df[c] = df[c].fillna(TR[c].median())
else:
    print("Missing-data bias NOT detected -> unchanged.")
MISS_AFTER = detect_missing(TR, FEATS)
RUN_LOG["Missing data"] = {"detected": MISS["detected"], "statistic": MISS["statistic"], "before": MISS["value"],
                           "after": MISS_AFTER["value"], "selected": miss_best or "—", "best_traditional": miss_trad or "—",
                           "methods_compared": len(MISS_METHODS) if MISS["detected"] else 0}
print(f"Missing cells after: train {MISS_AFTER['value']:.3%} | test {TE[FEATS].isna().to_numpy().mean():.3%}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
top = MISS["per_feature"].sort_values(ascending=False).head(15)
ax[0].barh(top.index[::-1], 100 * top.values[::-1], color="#6FA0DA", label="before")
ax[0].barh(top.index[::-1], 100 * MISS_AFTER["per_feature"][top.index[::-1]].values, color="#7BC17E", label="after")
ax[0].set_xlabel("% missing (train)"); ax[0].legend(); ax[0].set_title("Missing values before vs after")
if miss_tab is not None:
    mean_rank_plot(ax[1], miss_tab, miss_best, "Imputer comparison", MISS_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "01_missing_data_bias_dashboard.html", "Missing Data Bias", MISS["detected"],
    (f"Detected: {MISS['value']:.2%} of predictor cells missing — {miss_best} selected (best-ranked modern method; best traditional for comparison: {miss_trad}) and applied"
     if MISS["detected"] else "Not detected: no missing predictor cells — unchanged, After = Before"),
    {"Statistic": MISS["statistic"], "Value (before)": f"{MISS['value']:.3%}", "Threshold": MISS["threshold"],
     "Value (after)": f"{MISS_AFTER['value']:.3%}"},
    miss_best, miss_before, TR, display_vars(FEATS),
    missing_grid(miss_before, TR, FEATS),
    comparison=comparison_payload(miss_tab, MISS_DIRS, miss_best, MISS_FAMILY, miss_trad),
    footer="Before = training split with genuine missing values (histograms use observed values only). "
           "After = the same rows imputed by the selected method (fitted on the training split; the test split is imputed with the same fitted model).",
    chips=[f"Missing predictor cells: <b>{MISS['value']:.2%} → {MISS_AFTER['value']:.2%}</b>"])

# CELL 26
class _Det:
    def __init__(self, c, seed=SEED):
        self.c, self.seed = c, seed


class IFDet(_Det):
    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, contamination=self.c, random_state=self.seed, n_jobs=-1).fit(_cap_rows(X, 200000, self.seed))
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class LOFDet(_Det):
    def fit(self, X):
        self.m = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=self.c, n_jobs=-1).fit(_cap_rows(X, 30000, self.seed))
        return self

    def score(self, X):
        return -np.concatenate([self.m.score_samples(X[i:i + 50000]) for i in range(0, len(X), 50000)])


class OCSVMDet(_Det):
    def fit(self, X):
        self.m = OneClassSVM(nu=max(self.c, 0.01), gamma="scale").fit(_cap_rows(X, 8000, self.seed))
        return self

    def score(self, X):
        return -np.concatenate([self.m.decision_function(X[i:i + 50000]) for i in range(0, len(X), 50000)])


class ECODDet(_Det):
    def fit(self, X):
        from pyod.models.ecod import ECOD
        self.m = ECOD(contamination=self.c).fit(_cap_rows(X, 200000, self.seed))
        return self

    def score(self, X):
        return np.concatenate([self.m.decision_function(X[i:i + 100000]) for i in range(0, len(X), 100000)])


class DeepSVDDDet(_Det):
    """Deep SVDD (Ruff et al., 2018): bias-free network maps normal data into a minimum-volume
    hypersphere around a fixed centre; distance to the centre is the anomaly score."""
    def fit(self, X, epochs=30):
        set_seed(self.seed)
        Xt = torch.tensor(_cap_rows(X, 200000, self.seed), dtype=torch.float32, device=DEVICE)
        d = X.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 64, bias=False), nn.LeakyReLU(), nn.Linear(64, 32, bias=False),
                                 nn.LeakyReLU(), nn.Linear(32, 16, bias=False)).to(DEVICE)
        with torch.no_grad():
            c = self.net(Xt[:20000]).mean(0)
            c[(c.abs() < 0.1) & (c < 0)] = -0.1
            c[(c.abs() < 0.1) & (c >= 0)] = 0.1
        self.cen = c
        opt = torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-6)
        for _ in range(E(epochs)):
            for b in torch.randperm(len(Xt), device=DEVICE).split(512):
                loss = ((self.net(Xt[b]) - self.cen) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def score(self, X):
        with torch.no_grad():
            return np.concatenate([((self.net(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE)) - self.cen) ** 2)
                                   .sum(1).cpu().numpy() for i in range(0, len(X), 100000)])


class EBMDet(_Det):
    """EBM contextual outliers: an Explainable Boosting Machine predicts each continuous variable from all
    others; the anomaly score is the largest robust z-score of the prediction residuals."""
    def __init__(self, c, seed=SEED, targets=None, cap=6000):
        super().__init__(c, seed)
        self.targets, self.cap = targets, (3000 if WIDE else cap)

    def fit(self, X):
        from interpret.glassbox import ExplainableBoostingRegressor
        Xs = _cap_rows(X, self.cap, self.seed)
        self.models = {}
        for j in self.targets:
            others = [k for k in range(X.shape[1]) if k != j]
            m = ExplainableBoostingRegressor(interactions=0, outer_bags=2, max_rounds=E(1000), random_state=self.seed, n_jobs=-1)
            m.fit(Xs[:, others], Xs[:, j])
            r = Xs[:, j] - m.predict(Xs[:, others])
            med = np.median(r)
            mad = np.median(np.abs(r - med)) * 1.4826 or (np.std(r) or 1.0)
            self.models[j] = (m, others, med, mad)
        return self

    def score(self, X):
        z = [np.abs(X[:, j] - m.predict(X[:, o]) - med) / mad for j, (m, o, med, mad) in self.models.items()]
        return np.max(np.vstack(z), 0)


OUT_FAMILY = {"Isolation Forest": "Traditional", "LOF": "Traditional", "One-Class SVM": "Traditional",
              "ECOD": "Modern", "Deep SVDD": "Modern", "EBM": "Modern"}
OUT_DIRS = {"Score stability": "max", "Neighbourhood consistency": "max", "AUROC": "max", "AUPRC": "max",
            "Precision@K": "max", "F1": "max", "Brier": "min"}


def make_detector(name, c, seed, ebm_targets):
    return {"Isolation Forest": IFDet, "LOF": LOFDet, "One-Class SVM": OCSVMDet, "ECOD": ECODDet,
            "Deep SVDD": DeepSVDDDet}.get(name, None)(c, seed) if name != "EBM" else EBMDet(c, seed, ebm_targets)


def outlier_grid(before, after, fences, method):
    rate = {c: np.nanmean((before[c] < f[0]) | (before[c] > f[1])) for c, f in fences.items() if f is not None and c in before}
    cols = [c for c in sorted(rate, key=lambda c: -rate[c]) if rate[c] > 0][:10] or sorted(rate, key=lambda c: -rate[c])[:10]
    def flag(df, c):
        v = df[c].to_numpy(float)
        return (v < fences[c][0]) | (v > fences[c][1])
    return var_grid("Outlier Heatmap", before, after, cols, flag,
                    {"v": 1, "label": "Outlier present (beyond extreme fence)", "color": "#E8836B", "border": "#C9624B"}, "outliers",
                    "Rows = the continuous predictors with the most extreme values; each cell = one training patient (the same "
                    f"patients in Before and After). Coloured = the value lies beyond the skew-adjusted extreme fence. "
                    f"Grey in After = the patient was among the most anomalous training rows removed by {method or 'the selected method'}.",
                    "None")

# CELL 27
OUT = detect_outliers(TR, FEATS)
print(f"Rows beyond extreme fences: {OUT['value']:.2%}")
print(pd.DataFrame(OUT["per_feature"]).T.sort_values("above_%", ascending=False).head(12).round(3).to_string())
out_before = TR.copy()
out_tab, out_best, out_trad = None, None, None
c_rate = float(np.clip(OUT["value"], 0.005, 0.03))
if OUT["detected"] or FORCE_DETECT:
    std_o = Standardizer().fit(TR[FEATS].to_numpy(float))
    Xall = std_o.transform(TR[FEATS].to_numpy(float))
    rng = np.random.default_rng(SEED)
    cont_idx = [FEATS.index(c) for c in cols_of("cont")]
    flagrate = {FEATS.index(c): v["above_%"] + v["below_%"] for c, v in OUT["per_feature"].items()}
    ebm_targets = sorted(cont_idx, key=lambda j: -flagrate.get(j, 0))[:(4 if WIDE else 8)]
    b_idx = rng.choice(len(Xall), min(S("outlier"), len(Xall)), replace=False)
    Xb = Xall[b_idx]
    n_inj = max(10, int(0.02 * len(Xb)))
    inj = Xb[rng.choice(len(Xb), n_inj)].copy()
    q1, q3 = np.percentile(Xall[:, cont_idx], [25, 75], axis=0)
    med = np.median(Xall[:, cont_idx], 0)
    rsd = np.where((q3 - q1) / 1.349 > 1e-6, (q3 - q1) / 1.349, 1.0)
    for r in range(n_inj):
        for k in rng.choice(len(cont_idx), rng.integers(1, min(3, len(cont_idx)) + 1), replace=False):
            inj[r, cont_idx[k]] = med[k] + rng.choice([-1, 1]) * rng.uniform(6, 12) * rsd[k]
    Xbench = np.vstack([Xb, inj])
    ylab = np.r_[np.zeros(len(Xb)), np.ones(n_inj)].astype(int)
    knn_idx = NearestNeighbors(n_neighbors=11).fit(Xbench).kneighbors(Xbench, return_distance=False)[:, 1:]
    down = strat_sample(TR, S("outlier_down"))
    Xd = std_o.transform(down[FEATS].to_numpy(float))
    yd = down["target"].to_numpy(int)
    TEt = part(TE, "test")
    Xte_o = std_o.transform(TEt[FEATS].to_numpy(float))
    p_base, _, _ = fit_score(Xd, yd, Xte_o)
    print(f"Benchmark: {len(Xbench):,} rows incl. {n_inj} injected anomalies | contamination c = {c_rate:.3%} | "
          f"downstream baseline Brier {brier_score_loss(TEt.target, p_base):.4f}")
    res = {}
    for name in OUT_FAMILY:
        with Timer(f"outlier:{name}"):
            try:
                det = make_detector(name, c_rate, SEED, ebm_targets).fit(Xbench)
                s = det.score(Xbench)
                k = n_inj
                top = np.argsort(-s)[:k]
                flags = s >= np.quantile(s, 1 - c_rate)
                reps = []
                for rep in range(3):
                    sub = np.random.default_rng(SEED + rep).choice(len(Xbench), int(0.8 * len(Xbench)), replace=False)
                    reps.append(make_detector(name, c_rate, SEED + rep, ebm_targets).fit(Xbench[sub]).score(Xbench))
                stab = np.mean([spearmanr(reps[a], reps[b]).correlation for a, b in itertools.combinations(range(3), 2)])
                cons = spearmanr(s, s[knn_idx].mean(1)).correlation
                det_real = make_detector(name, c_rate, SEED, ebm_targets).fit(Xd)
                sr = det_real.score(Xd)
                keep = sr < np.quantile(sr, 1 - c_rate)
                p, _, _ = fit_score(Xd[keep], yd[keep], Xte_o)
                res[name] = {"Score stability": stab, "Neighbourhood consistency": cons,
                             "AUROC": safe_auc(ylab, s), "AUPRC": average_precision_score(ylab, s),
                             "Precision@K": ylab[top].mean(), "F1": f1_score(ylab, flags.astype(int)),
                             "Brier": brier_score_loss(TEt.target, p), "downstream AUROC (info)": safe_auc(TEt.target, p)}
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in OUT_DIRS}
    out_tab, out_best, out_trad = rank_methods(pd.DataFrame(res).T, OUT_DIRS, OUT_FAMILY)
    show_comparison(out_tab, OUT_DIRS, out_best, OUT_FAMILY, out_trad)
    with Timer(f"outlier:apply {out_best}"):
        det = make_detector(out_best, c_rate, SEED, ebm_targets).fit(Xall)
        s_all = det.score(Xall)
        drop = s_all >= np.quantile(s_all, 1 - c_rate)
        removed = TR.loc[drop]
        TR = TR.loc[~drop].copy()
    print(f"Removed {drop.sum():,} training rows ({drop.mean():.2%}); removed positive rate {removed.target.mean():.3f} "
          f"vs train {out_before.target.mean():.3f}")
else:
    print("Outlier bias NOT detected -> unchanged.")
OUT_AFTER = detect_outliers(TR, FEATS, fences=OUT["fences"])
RUN_LOG["Outlier"] = {"detected": OUT["detected"], "statistic": OUT["statistic"], "before": OUT["value"],
                      "after": OUT_AFTER["value"], "selected": out_best or "—", "best_traditional": out_trad or "—",
                      "methods_compared": len(OUT_FAMILY) if OUT["detected"] else 0}
print(f"Rows beyond the same fences after treatment: {OUT_AFTER['value']:.2%}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
pf = pd.DataFrame(OUT["per_feature"]).T
pfa = pd.DataFrame(OUT_AFTER["per_feature"]).T
tot_b = (pf["above_%"] + pf["below_%"]).sort_values(ascending=False).head(12)
ax[0].barh(tot_b.index[::-1], tot_b.values[::-1], color="#6FA0DA", label="before")
ax[0].barh(tot_b.index[::-1], (pfa["above_%"] + pfa["below_%"])[tot_b.index[::-1]].values, color="#7BC17E", alpha=0.9, label="after")
ax[0].set_xlabel("% rows beyond extreme fences"); ax[0].legend(); ax[0].set_title("Outliers before vs after")
if out_tab is not None:
    mean_rank_plot(ax[1], out_tab, out_best, "Outlier-detector comparison", OUT_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "03_outlier_bias_dashboard.html", "Outlier Bias", OUT["detected"],
    (f"Detected: {OUT['value']:.2%} of training rows beyond extreme fences — {out_best} selected (best-ranked modern method; best traditional for comparison: {out_trad}); top {c_rate:.2%} most anomalous training rows removed"
     if OUT["detected"] else f"Not detected: {OUT['value']:.2%} rows beyond extreme fences (threshold 1%) — unchanged, After = Before"),
    {"Statistic": OUT["statistic"], "Value (before)": f"{OUT['value']:.3%}", "Threshold": OUT["threshold"],
     "Value (after)": f"{OUT_AFTER['value']:.3%}", "Contamination used": f"{c_rate:.2%}" if OUT["detected"] else "—"},
    out_best, out_before, TR, display_vars(FEATS),
    outlier_grid(out_before, TR, OUT["fences"], out_best),
    comparison=comparison_payload(out_tab, OUT_DIRS, out_best, OUT_FAMILY, out_trad),
    footer="Only the training split is filtered; the test split keeps every patient so evaluation reflects the real population. "
           "The 'downstream AUROC (info)' column is reported for context and is not part of the ranking.",
    chips=[f"Rows beyond extreme fences: <b>{OUT['value']:.2%} → {OUT_AFTER['value']:.2%}</b>",
           f"Training rows removed: <b>{len(out_before) - len(TR):,}</b>"])

# CELL 29
def gce_loss(logits, y, q=0.7, w=None):
    p = torch.sigmoid(logits)
    py = torch.where(y > 0.5, p, 1 - p).clamp(min=1e-6)
    l = (1 - py ** q) / q
    return (l * w).mean() if w is not None else l.mean()


def train_mlp(X, y, loss="bce", epochs=12, batch=512, seed=SEED, weights=None, hidden=(128, 64), act=nn.ReLU,
              focal=None, record=False, init=None):
    """Small MLP classifier used wherever a method needs a neural loss. Returns (model, history)."""
    set_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    wt = torch.tensor(weights, dtype=torch.float32, device=DEVICE) if weights is not None else None
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, 1))
    net = nn.Sequential(*layers).to(DEVICE)
    if init is not None:
        for m in net:
            if isinstance(m, nn.Linear):
                init(m)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    hist = {"grad_norm": []}
    for _ in range(E(epochs)):
        for b in torch.randperm(len(Xt), device=DEVICE).split(batch):
            lo = net(Xt[b]).squeeze(-1)
            if loss == "gce":
                l = gce_loss(lo, yt[b], w=None if wt is None else wt[b])
            elif loss == "focal":
                p = torch.sigmoid(lo)
                pt = torch.where(yt[b] > 0.5, p, 1 - p).clamp(1e-6, 1)
                l = (-(1 - pt) ** focal["gamma"] * torch.log(pt) * wt[b]).mean()
            else:
                l = F.binary_cross_entropy_with_logits(lo, yt[b], weight=None if wt is None else wt[b])
            opt.zero_grad(); l.backward()
            if record:
                hist["grad_norm"].append(float(torch.sqrt(sum((p_.grad ** 2).sum() for p_ in net.parameters() if p_.grad is not None))))
            opt.step()
    return net, hist


def mlp_predict(net, X):
    with torch.no_grad():
        return torch.sigmoid(torch.cat([net(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE)).squeeze(-1)
                                        for i in range(0, len(X), 100000)])).cpu().numpy()


def pc1_order(Z):
    return np.argsort(PCA(1, random_state=SEED).fit_transform(np.nan_to_num(Z))[:, 0], kind="stable")


def along_order(D, fn):
    o = pc1_order(D[:, :-1])
    out = np.empty_like(D)
    out[o] = np.column_stack([fn(D[o, j]) for j in range(D.shape[1])])
    return out


def wavelet_1d(x, wav="db4"):
    lvl = min(3, pywt.dwt_max_level(len(x), pywt.Wavelet(wav).dec_len))
    co = pywt.wavedec(x, wav, level=lvl)
    sigma = np.median(np.abs(co[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    if not np.isfinite(thr) or thr <= 0:          # no estimable noise (e.g. sparse 0/1 column): nothing to shrink
        return np.asarray(x, float).copy()
    co = [co[0]] + [pywt.threshold(c, thr, mode="soft") for c in co[1:]]
    return pywt.waverec(co, wav)[:len(x)]


def group_ids(df):
    """Intersection of the protected attributes (e.g. 'Female|Black'); missing -> 'Unknown'."""
    parts = [df[a].astype(object).where(df[a].notna(), "Unknown").astype(str) for a in PROTECTED]
    g = parts[0]
    for p_ in parts[1:]:
        g = g + "|" + p_
    return g.to_numpy()


def cl_issue_flags(y, p, groups=None, min_per_class=30):
    """Confident-Learning flags (prune by noise rate); with `groups` the thresholds are estimated inside
    each protected group, too-small groups pooled together."""
    from cleanlab.filter import find_label_issues
    y = np.asarray(y).astype(int)
    P = np.c_[1 - p, p]
    if groups is None:
        return find_label_issues(labels=y, pred_probs=P, filter_by="prune_by_noise_rate", n_jobs=1)
    flags = np.zeros(len(y), bool)
    small = np.zeros(len(y), bool)
    for k in np.unique(groups):
        m = groups == k
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) >= min_per_class:
            flags[m] = find_label_issues(labels=y[m], pred_probs=P[m], filter_by="prune_by_noise_rate", n_jobs=1)
        else:
            small |= m
    if small.any() and min((y[small] == 0).sum(), (y[small] == 1).sum()) >= 5:
        flags[small] = find_label_issues(labels=y[small], pred_probs=P[small], filter_by="prune_by_noise_rate", n_jobs=1)
    return flags


def noise_budget(y, p, groups=None):
    """k = Confident Learning's estimate of the number of mislabeled recorded non-cases (label 0)."""
    y = np.asarray(y).astype(int)
    return int((cl_issue_flags(y, p, groups) & (y == 0)).sum())


def budget_flags(score, y_given, k):
    """Prune the k recorded non-cases with the highest suspicion score (recorded cases are never pruned)."""
    flags = np.zeros(len(score), bool)
    cand = np.where(np.asarray(y_given) == 0)[0]
    if k > 0 and len(cand):
        flags[cand[np.argsort(-np.asarray(score)[cand], kind="stable")[:k]]] = True
    return flags


def _reconstruction_score(D, Dt):
    """Processed predictors (labels never rewritten) + suspicion = |reconstructed label - given label|."""
    score = np.abs(Dt[:, -1] - D[:, -1])
    Dt = Dt.copy()
    Dt[:, -1] = D[:, -1]
    return Dt, score


def nz_cleanlab(D, seed=SEED, groups=None, p=None):
    """Confident Learning (Northcutt et al., 2021), group-conditional: out-of-fold probabilities; a
    recorded non-case is suspicious by how far its predicted case-probability exceeds the confident
    threshold of its own protected group (mean probability among that group's recorded cases)."""
    y = D[:, -1].astype(int)
    p = oof_probs(D[:, :-1], y, seed=seed) if p is None else p
    g = np.full(len(y), "all", dtype=object) if groups is None else np.asarray(groups, dtype=object)
    thr1 = pd.Series(p[y == 1]).groupby(g[y == 1]).agg(["mean", "size"])
    pooled = p[y == 1].mean() if (y == 1).any() else 0.5
    t_map = {k: (r["mean"] if r["size"] >= 30 else pooled) for k, r in thr1.iterrows()}
    t1 = np.array([t_map.get(k, pooled) for k in g])
    score = np.where(y == 0, p - t1, (1 - p) - (1 - t1))
    return D.copy(), score


def nz_robust(D, seed=SEED, epochs=10, groups=None):
    """Robust loss: class-balanced MLP trained with Generalised Cross-Entropy (Zhang & Sabuncu, 2018), which does
    not fit flipped labels; out-of-fold predictions; suspicion = probability of the other class."""
    y = D[:, -1]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(D, y.astype(int)):
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        net, _ = train_mlp(D[tr, :-1], y[tr], loss="gce", epochs=epochs, seed=seed, weights=w)
        p[te] = mlp_predict(net, D[te, :-1])
    return D.copy(), np.where(y == 0, p, 1 - p)


def nz_dae(D, seed=SEED, epochs=20, groups=None):
    set_seed(seed)
    X = torch.tensor(D, dtype=torch.float32, device=DEVICE)
    d = D.shape[1]
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, d)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(E(epochs)):
        for b in torch.randperm(len(X), device=DEVICE).split(512):
            x = X[b]
            xc = x + 0.2 * torch.randn_like(x)
            keep = (torch.rand_like(x) > 0.2).float()
            xc = xc * keep
            lm = (torch.rand(len(b), device=DEVICE) < 0.5).float()
            xc[:, -1] = xc[:, -1] * (1 - lm) + 0.5 * lm
            loss = ((net(xc) - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xi = X.clone()
        Xi[:, -1] = 0.5
        R = torch.cat([net(Xi[i:i + 100000]) for i in range(0, len(Xi), 100000)]).cpu().numpy()
    return _reconstruction_score(D, R)


class TabDenoiser(nn.Module):
    def __init__(self, d, T, h=256):
        super().__init__()
        self.t = nn.Embedding(T, h)
        self.inp = nn.Linear(d, h)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, x, t):
        return self.net(self.inp(x) + self.t(t))


class GaussianDiffusion:
    """DDPM (Ho et al., 2020) on standardised tabular rows; also the core of the TabDDPM generator."""
    def __init__(self, d, T=200, seed=SEED):
        self.T, self.seed = T, seed
        b = torch.linspace(1e-4, 0.02, T, device=DEVICE)
        self.b, self.a = b, 1 - b
        self.ab = torch.cumprod(self.a, 0)
        self.net = TabDenoiser(d, T).to(DEVICE)

    def fit(self, X, iters=3000, batch=512):
        set_seed(self.seed)
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        for _ in range(E(iters)):
            x = Xt[torch.randint(0, len(Xt), (min(batch, len(Xt)),), device=DEVICE)]
            t = torch.randint(0, self.T, (len(x),), device=DEVICE)
            e = torch.randn_like(x)
            ab = self.ab[t][:, None]
            loss = ((self.net(ab.sqrt() * x + (1 - ab).sqrt() * e, t) - e) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def reverse(self, x, t_start):
        for t in range(t_start, -1, -1):
            tt = torch.full((len(x),), t, device=DEVICE, dtype=torch.long)
            e = self.net(x, tt)
            x = (x - self.b[t] / (1 - self.ab[t]).sqrt() * e) / self.a[t].sqrt()
            if t > 0:
                x = x + ((1 - self.ab[t - 1]) / (1 - self.ab[t]) * self.b[t]).sqrt() * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample(self, n, batch=20000):
        out = [self.reverse(torch.randn(min(batch, n - i), self.net.inp.in_features, device=DEVICE), self.T - 1).cpu().numpy()
               for i in range(0, n, batch)]
        return np.vstack(out) if out else np.zeros((0, self.net.inp.in_features))

    @torch.no_grad()
    def denoise(self, X, frac=0.3, batch=50000):
        ts = int(frac * self.T)
        out = []
        for i in range(0, len(X), batch):
            x0 = torch.tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            ab = self.ab[ts]
            out.append(self.reverse(ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0), ts).cpu().numpy())
        return np.vstack(out)


def nz_diffusion(D, seed=SEED, groups=None):
    st = Standardizer().fit(D)
    Z = st.transform(D)
    gd = GaussianDiffusion(D.shape[1], seed=seed).fit(Z)
    return _reconstruction_score(D, st.inverse(gd.denoise(Z)))


NOISE_METHODS = {
    "Denoising autoencoder": nz_dae,
    "Wavelet denoising": lambda D, seed=SEED, groups=None: _reconstruction_score(D, along_order(D, wavelet_1d)),
    "Median filter": lambda D, seed=SEED, groups=None: _reconstruction_score(D, along_order(D, lambda x: medfilt(x, 5))),
    "Savitzky–Golay": lambda D, seed=SEED, groups=None: _reconstruction_score(D, along_order(D, lambda x: savgol_filter(x, 11, 2) if len(x) > 11 else x)),
    "Confident Learning (Cleanlab)": nz_cleanlab,
    "Robust loss (GCE)": nz_robust,
    "Diffusion denoising": nz_diffusion,
}
NOISE_FAMILY = {k: ("Modern" if k in ("Confident Learning (Cleanlab)", "Robust loss (GCE)", "Diffusion denoising") else "Traditional")
                for k in NOISE_METHODS}
NOISE_DIRS = {"Residual variance": "min", "Spectral distortion (dB)": "min", "MAE": "min", "RMSE": "min",
              "SNR (dB)": "max", "Label-error detection F1": "max"}


def prune_gap(flags, groups, y_given, min_n=30):
    """Max - min (percentage points) of the share of recorded non-cases pruned, across protected groups."""
    rates = [100 * flags[(groups == k) & (y_given == 0)].mean() for k in np.unique(groups)
             if ((groups == k) & (y_given == 0)).sum() >= min_n]
    return float(max(rates) - min(rates)) if len(rates) > 1 else np.nan


def noise_metrics(Dc, Dt, flags, flipped):
    """Metrics on the rows the treatment keeps (flagged rows pruned), compared with their clean versions."""
    keep = ~np.asarray(flags, bool)
    C, T = Dc[keep], np.nan_to_num(Dt[keep], nan=0.0, posinf=0.0, neginf=0.0)
    res = T - C
    var_c = C.var(0)
    var_c[var_c < 1e-12] = 1.0
    sig = ((C - C.mean(0)) ** 2).sum()
    noise_pow = (res ** 2).sum()
    lc = np.clip(np.linalg.eigvalsh(np.cov(C.T)), 1e-8, None)
    lt = np.clip(np.linalg.eigvalsh(np.cov(T.T)), 1e-8, None)
    return {"Residual variance": float(np.mean(res.var(0) / var_c)),
            "Spectral distortion (dB)": float(np.sqrt(np.mean((10 * np.log10(lc) - 10 * np.log10(lt)) ** 2))),
            "MAE": float(np.abs(res).mean()), "RMSE": float(np.sqrt((res ** 2).mean())),
            "SNR (dB)": float(min(60.0, 10 * np.log10(sig / max(noise_pow, 1e-12)))),
            "Label-error detection F1": float(f1_score(flipped, flags, zero_division=0)),
            "rows pruned % (info)": 100 * float(np.mean(flags)),
            "label noise left in kept rows % (info)": 100 * float(flipped[keep].mean())}


def noise_grid(before, after, flags_b, flags_a, method):
    """Rows = all patients and each protected group; cells = example training patients of that group."""
    def states(df, fl):
        y = df["target"].to_numpy(int)
        return np.where(np.asarray(fl, bool), np.where(y == 1, 2, 1), 0)
    sb = states(before, flags_b)
    sa = map_after(before, after, states(after, flags_a))
    groups = [("All training patients", np.ones(len(before), bool))]
    for a_ in PROTECTED:
        for k in sorted(before[a_].dropna().astype(str).unique()):
            groups.append((f"{a_.replace('sens_', '').replace('_', ' ')} = {k}", (before[a_].astype(str) == k).to_numpy()))
    st = [{"v": 1, "label": "Suspected label error, recorded non-case", "color": "#F0B323", "border": "#C99212"},
          {"v": 2, "label": "Suspected label error, recorded case (kept by safeguard)", "color": "#D0433F", "border": "#A8322F"}]
    return group_grid("Noise Heatmap", groups, sb, sa, st, "flagged",
                      "Rows = all training patients and each protected group; each cell = one patient of that group (the same "
                      "patients in Before and After). Coloured = group-conditional Confident Learning says the recorded label "
                      "contradicts the predictors (same detector Before and After). Grey in After = pruned by "
                      f"{method or 'the selected method'}; recorded cases are never pruned.", "Label consistent")


def cl_report(df, feats, seed=SEED):
    """Group-conditional Confident-Learning flags and the out-of-fold probabilities."""
    X, y = df[feats].to_numpy(float), df["target"].to_numpy(int)
    p = oof_probs(X, y, seed=seed)
    return cl_issue_flags(y, p, group_ids(df)), p

# CELL 30
NZ = detect_label_noise(TR, FEATS)
print(f"Confident-Learning label-issue rate: {NZ['value']:.2%} (class 0: {NZ['issues_by_class'][0]:.2%}, class 1: {NZ['issues_by_class'][1]:.2%})")
nz_before = TR.copy()
with Timer("noise:CL on full training split"):
    nz_flags_before, P_BEFORE = cl_report(nz_before, FEATS)
g_before = group_ids(nz_before)
flag_share = pd.DataFrame({"group": g_before, "label": nz_before["target"].to_numpy(), "flagged": nz_flags_before}) \
    .groupby(["group", "label"]).flagged.mean().unstack().mul(100).round(2)
print("Group-conditional Confident Learning, flagged share (%) by protected group and recorded label:")
print(flag_share.to_string())
nz_tab, nz_best, nz_trad = None, None, None
k_full = 0
if NZ["detected"] or FORCE_DETECT:
    bench = strat_sample(TR, S("noise"))
    stn = Standardizer().fit(TR[FEATS].to_numpy(float))
    Dc = np.c_[stn.transform(bench[FEATS].to_numpy(float)), bench["target"].to_numpy(float)]
    rng = np.random.default_rng(SEED)
    flipped = (Dc[:, -1] == 1) & (rng.random(len(Dc)) < 0.10)      # under-reported cases
    Dn = Dc.copy()
    Dn[flipped, -1] = 0.0
    g_bench = group_ids(bench)
    y_n = Dn[:, -1].astype(int)
    p_bench = oof_probs(Dn[:, :-1], y_n)
    k_bench = noise_budget(y_n, p_bench, g_bench)
    print(f"Benchmark: {len(Dc):,} rows, {flipped.sum():,} true cases recorded as non-cases (10% of cases); "
          f"pruning budget k = {k_bench:,} recorded non-cases (Confident-Learning estimate)")
    res = {"No denoising (noisy input)": noise_metrics(Dc, Dn, np.zeros(len(Dn), bool), flipped)}
    for name, fn in NOISE_METHODS.items():
        with Timer(f"noise:{name}"):
            try:
                kw = {"p": p_bench} if name == "Confident Learning (Cleanlab)" else {}
                Dt, score = fn(Dn, seed=SEED, groups=g_bench, **kw)
                flags = budget_flags(score, y_n, k_bench)
                res[name] = noise_metrics(Dc, Dt, flags, flipped)
                res[name]["pruning-rate gap across groups, pp (info)"] = prune_gap(flags, g_bench, y_n)
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in NOISE_DIRS}
    nz_tab, nz_best, nz_trad = rank_methods(pd.DataFrame(res).T, NOISE_DIRS, NOISE_FAMILY, reference_rows=["No denoising (noisy input)"])
    show_comparison(nz_tab, NOISE_DIRS, nz_best, NOISE_FAMILY, nz_trad)
    with Timer(f"noise:apply {nz_best}"):
        y_tr = TR["target"].to_numpy(int)
        g_tr = group_ids(TR)
        k_full = noise_budget(y_tr, P_BEFORE, g_tr)
        Dfull = np.c_[stn.transform(TR[FEATS].to_numpy(float)), TR["target"].to_numpy(float)]
        kw = {"p": P_BEFORE} if nz_best == "Confident Learning (Cleanlab)" else {}
        _, score_full = NOISE_METHODS[nz_best](Dfull, seed=SEED, groups=g_tr, **kw)
        flags_full = budget_flags(score_full, y_tr, k_full)
        pruned_by_group = pd.Series(flags_full[y_tr == 0]).groupby(g_tr[y_tr == 0]).mean().mul(100).round(3)
        TR = TR.loc[~flags_full].copy()
    print(f"Pruning budget k = {k_full:,}; pruned {flags_full.sum():,} recorded non-cases ({flags_full.mean():.2%} of training rows). "
          "Recorded cases are never pruned.")
    print("Pruned share of recorded non-cases (%) by protected group:")
    print(pruned_by_group.to_string())
else:
    print("Label-noise bias NOT detected -> unchanged.")
with Timer("noise:CL after"):
    nz_flags_after, _ = cl_report(TR, FEATS)
NZ_AFTER = {"value": float(nz_flags_after.mean())}
RUN_LOG["Data noise"] = {"detected": NZ["detected"], "statistic": NZ["statistic"], "before": float(nz_flags_before.mean()),
                         "after": NZ_AFTER["value"], "selected": nz_best or "—", "best_traditional": nz_trad or "—",
                         "methods_compared": len(NOISE_METHODS) if NZ["detected"] else 0}
print(f"Group-conditional CL flag rate on the full training split: before {nz_flags_before.mean():.2%} -> after {nz_flags_after.mean():.2%} "
      f"(label 0: {nz_flags_before[nz_before.target.to_numpy() == 0].mean():.2%} -> {nz_flags_after[TR.target.to_numpy() == 0].mean():.2%})")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].bar(["label 0", "label 1", "all"], [100 * nz_flags_before[nz_before.target.to_numpy() == 0].mean(),
          100 * nz_flags_before[nz_before.target.to_numpy() == 1].mean(), 100 * nz_flags_before.mean()], color="#6FA0DA", width=0.4, label="before")
ax[0].bar(np.arange(3) + 0.4, [100 * nz_flags_after[TR.target.to_numpy() == 0].mean(), 100 * nz_flags_after[TR.target.to_numpy() == 1].mean(),
          100 * nz_flags_after.mean()], color="#7BC17E", width=0.4, label="after")
ax[0].set_ylabel("% rows flagged by Confident Learning"); ax[0].legend(); ax[0].set_title("Label issues before vs after (label-1 rows kept by design)")
if nz_tab is not None:
    mean_rank_plot(ax[1], nz_tab, nz_best, "Noise-method comparison", NOISE_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "07_data_noise_bias_dashboard.html", "Data Noise Bias", NZ["detected"],
    (f"Detected: Confident Learning flags {NZ['value']:.2%} of labels — {nz_best} selected (best-ranked modern method; best traditional for comparison: {nz_trad}); "
     f"its top {k_full:,} recorded non-cases pruned (recorded cases are never deleted)"
     if NZ["detected"] else f"Not detected: label-issue rate {NZ['value']:.2%} (threshold 2%) — unchanged, After = Before"),
    {"Statistic": NZ["statistic"], "Value (before)": f"{NZ['value']:.3%}", "Threshold": NZ["threshold"],
     "Pruning budget k": f"{k_full:,}", "CL rate full train before": f"{nz_flags_before.mean():.3%}",
     "CL rate full train after": f"{nz_flags_after.mean():.3%}"},
    nz_best, nz_before, TR, display_vars(FEATS),
    noise_grid(nz_before, TR, nz_flags_before, nz_flags_after, nz_best),
    comparison=comparison_payload(nz_tab, NOISE_DIRS, nz_best, NOISE_FAMILY, nz_trad),
    footer="Benchmark: 10% of true cases recorded as non-cases on a clean copy of real rows; every method prunes the same number k of recorded "
           "non-cases (Confident-Learning estimate). Clinical-risk safeguard: recorded cases are never pruned. Info columns are shown in the notebook.",
    chips=[f"Labels flagged: <b>{nz_flags_before.mean():.2%} → {nz_flags_after.mean():.2%}</b>",
           f"Non-cases pruned: <b>{len(nz_before) - len(TR):,}</b>"])

# CELL 32
TS_VARS = {"heartrate": ("heart_rate", 0, 300), "respiration": ("resp_rate", 0, 70), "sao2": ("spo2", 0, 100),
           "systemicmean": ("art_map", 0, 300)}
TS_FILE = Path(r"D:\complete\resp_physionet\resp\official_verified\vitalPeriodic.csv.gz")
NPTS = 289


def load_series(stays):
    stays = set(int(s) for s in stays)
    parts = []
    for ch in pd.read_csv(TS_FILE, usecols=["patientunitstayid", "observationoffset", *TS_VARS], chunksize=2_000_000):
        ch = ch[ch.patientunitstayid.isin(stays) & ch.observationoffset.between(0, 1440)]
        parts.append(ch)
    raw_ts = pd.concat(parts)
    raw_ts["k"] = (raw_ts.observationoffset // 5).astype(int)
    sid = np.array(sorted(raw_ts.patientunitstayid.unique()))
    pos = pd.Series(np.arange(len(sid)), index=sid)
    out = {}
    for src, (stem, lo, hi) in TS_VARS.items():
        M = np.full((len(sid), NPTS), np.nan, np.float32)
        v = raw_ts[src].to_numpy(float)
        ok = np.isfinite(v) & (v > lo) & (v <= hi)
        g = raw_ts.loc[ok].groupby(["patientunitstayid", "k"])[src].mean()
        M[pos[g.index.get_level_values(0)].to_numpy(), g.index.get_level_values(1).to_numpy()] = g.to_numpy()
        out[stem] = M
    return sid, out


def spike_mask(M):
    from numpy.lib.stride_tricks import sliding_window_view
    P = np.pad(M, ((0, 0), (2, 2)), constant_values=np.nan)
    med = np.nanmedian(sliding_window_view(P, 5, axis=1), axis=2)
    P2 = np.pad(M, ((0, 0), (5, 5)), constant_values=np.nan)
    W = sliding_window_view(P2, 11, axis=1)
    mad = 1.4826 * np.nanmedian(np.abs(W - np.nanmedian(W, axis=2, keepdims=True)), axis=2)
    return np.abs(M - med) > 4 * np.maximum(mad, 1.0)


def fill_gaps(M):
    out = M.copy()
    x = np.arange(M.shape[1])
    for i in range(len(M)):
        ok = np.isfinite(M[i])
        if ok.sum() >= 2:
            out[i] = np.interp(x, x[ok], M[i, ok])
    return out


class ConvDAE(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, ch, 7, padding=3), nn.ReLU(), nn.Conv1d(ch, ch, 7, padding=3), nn.ReLU(),
                                 nn.Conv1d(ch, ch, 7, padding=3), nn.ReLU(), nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x):
        return x - self.net(x)          # predicts the noise (residual learning)


def corrupt(C, rng, sd_frac=0.15, spike=0.01):
    sd = C.std(1, keepdims=True) + 1e-6
    N = C + rng.normal(0, 1, C.shape) * sd_frac * sd
    sp = rng.random(C.shape) < spike
    N[sp] += rng.choice([-1, 1], sp.sum()) * 6 * np.broadcast_to(sd, C.shape)[sp]
    return N


def ts_dae(train_clean, N, seed=SEED, epochs=15):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = train_clean.mean(1, keepdims=True), train_clean.std(1, keepdims=True) + 1e-6
    Ct = torch.tensor((train_clean - mu) / sd, dtype=torch.float32, device=DEVICE)[:, None]
    net = ConvDAE().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(E(epochs)):
        for b in torch.randperm(len(Ct), device=DEVICE).split(256):
            noisy = torch.tensor(corrupt(Ct[b, 0].cpu().numpy(), rng), dtype=torch.float32, device=DEVICE)[:, None]
            loss = ((net(noisy) - Ct[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m2, s2 = N.mean(1, keepdims=True), N.std(1, keepdims=True) + 1e-6
    with torch.no_grad():
        out = torch.cat([net(torch.tensor((N[i:i + 4096] - m2[i:i + 4096]) / s2[i:i + 4096], dtype=torch.float32, device=DEVICE)[:, None])[:, 0]
                         for i in range(0, len(N), 4096)]).cpu().numpy()
    return out * s2 + m2


def ts_diffusion(train_clean, N, seed=SEED):
    mu, sd = train_clean.mean(1, keepdims=True), train_clean.std(1, keepdims=True) + 1e-6
    gd = GaussianDiffusion(NPTS, seed=seed).fit((train_clean - mu) / sd, iters=4000)
    m2, s2 = N.mean(1, keepdims=True), N.std(1, keepdims=True) + 1e-6
    return gd.denoise((N - m2) / s2, frac=0.15) * s2 + m2


TS_METHODS = {"Denoising autoencoder": "dae", "Wavelet denoising": lambda x: wavelet_1d(x),
              "Median filter": lambda x: medfilt(x, 5), "Savitzky–Golay": lambda x: savgol_filter(x, 11, 2),
              "Diffusion denoising": "diff"}
TS_DIRS = {"Residual variance": "min", "Spectral distortion (dB)": "min", "MAE": "min", "RMSE": "min", "SNR (dB)": "max"}
# Confident Learning and robust loss act on labels, so they are listed as not applicable to signal denoising;
# among the signal methods only diffusion denoising is a modern method.
TS_FAMILY = {**{k: NOISE_FAMILY[k] for k in TS_METHODS},
             "Confident Learning (Cleanlab)": "reference", "Robust loss (GCE)": "reference"}


def ts_metrics(C, D):
    r = D - C
    pc = np.abs(np.fft.rfft(C - C.mean(1, keepdims=True), axis=1)) ** 2 + 1e-6
    pd_ = np.abs(np.fft.rfft(D - D.mean(1, keepdims=True), axis=1)) ** 2 + 1e-6
    return {"Residual variance": float(np.mean(r.var(1) / (C.var(1) + 1e-9))),
            "Spectral distortion (dB)": float(np.mean(np.sqrt(np.mean((10 * np.log10(pc) - 10 * np.log10(pd_)) ** 2, 1)))),
            "MAE": float(np.abs(r).mean()), "RMSE": float(np.sqrt((r ** 2).mean())),
            "SNR (dB)": float(10 * np.log10(((C - C.mean(1, keepdims=True)) ** 2).sum() / (r ** 2).sum()))}


def denoise_matrix(name, M, train_clean, seed=SEED):
    F_ = fill_gaps(M)
    ok = np.isfinite(F_).all(1)
    out = M.copy()
    if not ok.any():
        return out
    X = F_[ok]
    if TS_METHODS[name] == "dae":
        Dn = ts_dae(train_clean, X, seed)
    elif TS_METHODS[name] == "diff":
        Dn = ts_diffusion(train_clean, X, seed)
    else:
        Dn = np.vstack([TS_METHODS[name](x) for x in X])
    Dn = np.where(np.isfinite(M[ok]), Dn, np.nan)          # keep genuinely unobserved points unobserved
    out[ok] = Dn
    return out

# CELL 33
with Timer("noise-ts:load"):
    all_stays = np.r_[TR["meta_patientunitstayid"].to_numpy(), TE["meta_patientunitstayid"].to_numpy()]
    TS_SID, TS = load_series(all_stays)
tr_set = set(TR["meta_patientunitstayid"].astype(int))
is_tr = np.array([s in tr_set for s in TS_SID])
spk = {k: spike_mask(M[is_tr]) for k, M in TS.items()}
obs = {k: np.isfinite(M[is_tr]) for k, M in TS.items()}
TS_RATE = {k: float(spk[k][obs[k]].mean()) for k in TS}
ts_rate = float(sum(spk[k][obs[k]].sum() for k in TS) / sum(obs[k].sum() for k in TS))
TS_DET = ts_rate >= 0.01
TS_RUN = TS_DET or FORCE_DETECT
print(f"Monitor spike rate (train): {ts_rate:.2%} | per signal: " + ", ".join(f"{k} {v:.2%}" for k, v in TS_RATE.items()))
ts_tab, ts_best, ts_trad = None, None, None
ts_before = TR.copy()
if TS_RUN:
    rng = np.random.default_rng(SEED)
    Mhr = TS["heart_rate"][is_tr]
    full = np.where(np.isfinite(Mhr).mean(1) >= 0.9)[0]
    pick = rng.choice(full, min(E(2000), len(full)), replace=False)
    rest = np.setdiff1d(full, pick)[:E(6000)]
    clean = {k: fill_gaps(TS[k][is_tr][pick]) for k in TS}
    trainc = {k: fill_gaps(TS[k][is_tr][rest]) for k in TS}
    res = {}
    for name in TS_METHODS:
        with Timer(f"noise-ts:{name}"):
            parts = []
            for k in TS:
                okc = np.isfinite(clean[k]).all(1)
                okt = np.isfinite(trainc[k]).all(1)
                C = clean[k][okc]
                N = corrupt(C, np.random.default_rng(SEED))
                D = denoise_matrix(name, N, trainc[k][okt])
                parts.append(ts_metrics(C, D))
            res[name] = {m: float(np.mean([p[m] for p in parts])) for m in TS_DIRS}
    for name in ("Confident Learning (Cleanlab)", "Robust loss (GCE)"):
        res[name] = {m: np.nan for m in TS_DIRS}
    ts_tab, ts_best, ts_trad = rank_methods(pd.DataFrame(res).T, TS_DIRS, TS_FAMILY,
                                   reference_rows=["Confident Learning (Cleanlab)", "Robust loss (GCE)"])
    show_comparison(ts_tab, TS_DIRS, ts_best, TS_FAMILY, ts_trad)
    TS_RAW = {k: M.copy() for k, M in TS.items()}
    with Timer(f"noise-ts:apply {ts_best}"):
        pos_by_sid = pd.Series(np.arange(len(TS_SID)), index=TS_SID)
        for k, M in TS.items():
            okt = np.isfinite(trainc[k]).all(1)
            Dn = denoise_matrix(ts_best, M, trainc[k][okt])
            agg = {"mean": np.nanmean(Dn, 1), "min": np.nanmin(Dn, 1), "max": np.nanmax(Dn, 1)}
            for df in (TR, TE):
                p = pos_by_sid.reindex(df["meta_patientunitstayid"].astype(int)).to_numpy()
                has = ~np.isnan(p)
                for s, v in agg.items():
                    col = f"{k}_{s}"
                    if col in df:
                        vals = df[col].to_numpy(float)
                        newv = v[p[has].astype(int)]
                        vals[has] = np.where(np.isfinite(newv), newv, vals[has])
                        df[col] = vals
            TS[k] = Dn
    spk_after = {k: spike_mask(TS[k][is_tr]) for k in TS}
    ts_after = float(sum(spk_after[k][obs[k]].sum() for k in TS) / sum(obs[k].sum() for k in TS))
    print(f"Spike rate after denoising: {ts_after:.2%}")
else:
    ts_after = ts_rate
    spk_after = spk
    print("Monitor-signal noise NOT detected -> unchanged.")
RUN_LOG["Data noise"]["detected"] = RUN_LOG["Data noise"]["detected"] or TS_DET
RUN_LOG["Data noise"]["selected"] = f"labels: {RUN_LOG['Data noise']['selected']}; signals: {ts_best or '—'}"
RUN_LOG["Data noise"]["best_traditional"] = f"labels: {RUN_LOG['Data noise']['best_traditional']}; signals: {ts_trad or '—'}"
RUN_LOG["Data noise"]["statistic"] += "; monitor spike rate"
RUN_LOG["Data noise"]["before"] = f"{RUN_LOG['Data noise']['before']:.4f}; spikes {ts_rate:.4f}"
RUN_LOG["Data noise"]["after"] = f"{RUN_LOG['Data noise']['after']:.4f}; spikes {ts_after:.4f}"
RUN_LOG["Data noise"]["methods_compared"] += len(TS_METHODS) if TS_DET else 0

if TS_RUN:
    ex = np.where(np.isfinite(TS["heart_rate"][is_tr]).mean(1) > 0.9)[0][0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
    raw_one = TS_RAW["heart_rate"][is_tr][ex]
    ax[0].plot(np.arange(NPTS) * 5 / 60, raw_one, color="#6FA0DA", label="before (raw monitor)")
    ax[0].plot(np.arange(NPTS) * 5 / 60, TS["heart_rate"][is_tr][ex], color="#4C9B50", label=f"after ({ts_best})")
    ax[0].set_xlabel("hours since ICU admission"); ax[0].set_ylabel("heart rate"); ax[0].legend(); ax[0].set_title("Example series")
    mean_rank_plot(ax[1], ts_tab, ts_best, "Signal-denoising comparison", TS_FAMILY)
    plt.tight_layout(); plt.show()
write_dashboard(
    "07b_monitor_signal_noise_dashboard.html", "Data Noise Bias — monitor signals", TS_DET,
    (f"Detected: {ts_rate:.2%} of monitor points are spikes — {ts_best} selected (best-ranked modern method; best traditional for comparison: {ts_trad}); vital-sign summaries recomputed from denoised series"
     if TS_DET else f"Not detected: {ts_rate:.2%} of monitor points are spikes (threshold 1%) — signals unchanged, After = Before"),
    {"Statistic": "spike rate (|x − rolling median| > 4 rolling robust SD)", "Value (before)": f"{ts_rate:.3%}",
     "Threshold": ">= 1%", "Value (after)": f"{ts_after:.3%}", **{f"{k} spikes": f"{v:.3%}" for k, v in TS_RATE.items()}},
    ts_best, ts_before, TR, [c for c in FEATS if c.split("_")[0] in ("heart", "resp", "spo2", "art")][:15],
    array_grid("Noise Heatmap (monitor signals)",
               {k: np.where(np.nan_to_num(spk[k].sum(1) / np.maximum(obs[k].sum(1), 1)) > 0.02, 2, (spk[k].sum(1) > 0).astype(int)) for k in TS},
               {k: np.where(np.nan_to_num(spk_after[k].sum(1) / np.maximum(obs[k].sum(1), 1)) > 0.02, 2, (spk_after[k].sum(1) > 0).astype(int)) for k in TS},
               [{"v": 1, "label": "Some spikes (≤ 2% of points)", "color": "#F0B323", "border": "#C99212"},
                {"v": 2, "label": "Noisy signal (> 2% of points are spikes)", "color": "#D0433F", "border": "#A8322F"}],
               "of stays with spikes",
               "Rows = bedside-monitor signals; each cell = one training ICU stay (the same stays in Before and After). "
               "A spike = a 5-minute value more than 4 rolling robust SDs from the rolling median. After = the same series "
               f"after {ts_best or 'no denoising'}.", "No spikes"),
    comparison=comparison_payload(ts_tab, TS_DIRS, ts_best, TS_FAMILY, ts_trad),
    footer="Distributions show the vital-sign summary predictors (training split); they are recomputed from denoised series only when signal noise is detected.",
    chips=[f"Spike rate: <b>{ts_rate:.2%} → {ts_after:.2%}</b>"])

# CELL 35
SC_FAMILY = {"Quantile transform": "Traditional", "Min-Max scaler": "Traditional", "Robust scaler": "Traditional",
             "SNN (SELU)": "Modern", "FT-Transformer": "Modern", "PowerTransform (Yeo-Johnson)": "Modern"}
SC_DIRS = {"Wasserstein to N(0,1)": "min", "Correlation preservation |Δr|": "min", "Skewness reduction": "max",
           "Kurtosis |excess|": "min", "Activation variance |log σ²|": "min", "Gradient stability (CV)": "min"}


class ZScore:
    def fit(self, X):
        self.s = Standardizer().fit(X)
        return self

    def transform(self, X):
        return self.s.transform(X)


def make_scaler(name):
    return {"Quantile transform": lambda: QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                              subsample=200000, random_state=SEED),
            "Min-Max scaler": MinMaxScaler, "Robust scaler": RobustScaler,
            "PowerTransform (Yeo-Johnson)": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
            "SNN (SELU)": ZScore, "FT-Transformer": ZScore}[name]()


def lecun_normal(m):
    nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.in_features))
    nn.init.zeros_(m.bias)


def activation_logvar(net, X):
    """Mean |log variance| of every hidden activation layer of a Sequential MLP."""
    out, h = [], torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mods = list(net)
        for i, m in enumerate(mods[:-1]):
            h = m(h)
            if not isinstance(m, nn.Linear):
                out.append(abs(math.log(float(h.var()) + 1e-12)))
    return float(np.mean(out))


def ft_probe(Xtr, ytr, Xev, epochs=6, seed=SEED):
    from rtdl_revisiting_models import FTTransformer
    set_seed(seed)
    kw = FTTransformer.get_default_kwargs(n_blocks=2)
    model = FTTransformer(n_cont_features=Xtr.shape[1], cat_cardinalities=[], d_out=1, **kw).to(DEVICE)
    opt = model.make_default_optimizer()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    grads = []
    for _ in range(E(epochs)):
        model.train()
        for b in torch.randperm(len(Xt), device=DEVICE).split(512):
            loss = F.binary_cross_entropy_with_logits(model(Xt[b], None).squeeze(-1), yt[b])
            opt.zero_grad(); loss.backward()
            grads.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))))
            opt.step()
    model.eval()
    acts = []
    # blocks are ModuleDicts called piece by piece, so hook the attention and feed-forward sub-layers
    hooks = [blk[part].register_forward_hook(lambda m, i, o: acts.append(abs(math.log(float((o[0] if isinstance(o, tuple) else o).var()) + 1e-12))))
             for blk in model.backbone.blocks for part in ("attention", "ffn")]
    ev_bs = 20000 if Xev.shape[1] <= 100 else 256      # attention memory grows with the number of predictors (wide ICU data)
    with torch.no_grad():
        p = torch.sigmoid(torch.cat([model(torch.tensor(Xev[i:i + ev_bs], dtype=torch.float32, device=DEVICE), None).squeeze(-1).cpu()
                                     for i in range(0, len(Xev), ev_bs)])).numpy()
        acts.clear()
        n_act = min(4096, len(Xev))
        for i in range(0, n_act, min(ev_bs, 4096)):
            model(torch.tensor(Xev[i:min(i + ev_bs, n_act)], dtype=torch.float32, device=DEVICE), None)
    for h in hooks:
        h.remove()
    return float(np.mean(acts)), grads, p


def apply_scaler(df, scaler, cols):
    df = df.copy()
    df[cols] = scaler.transform(df[cols].to_numpy(float))
    return df


def update_ordinal_allowed(scaler, cols, ref):
    med = np.median(ref[cols].to_numpy(float), 0)
    for j, c in enumerate(cols):
        if SCHEMA[c]["type"] == "ord":
            a = np.asarray(SCHEMA[c]["allowed"], float)
            M = np.tile(med, (len(a), 1))
            M[:, j] = a
            SCHEMA[c]["allowed"] = [float(v) for v in scaler.transform(M)[:, j]]


def scale_grid(before, after, cols, method):
    """Rows = continuous/ordinal predictors (largest to smallest magnitude); coloured = |value| above one common line."""
    if not cols:
        return grid_payload("Feature Scale Heatmap", [], [], "", "Within common scale",
                            empty="No continuous or ordinal predictors.")
    p95 = before[cols].abs().quantile(0.95).sort_values(ascending=False)
    line = float(np.median(p95.values))
    order = list(p95.index)
    show = [order[i] for i in np.unique(np.linspace(0, len(order) - 1, min(10, len(order))).round().astype(int))]
    return var_grid("Feature Scale Heatmap", before, after, show, lambda df, c: (df[c].abs() > line).to_numpy(),
                    {"v": 1, "label": "Scale-dominant (|value| above the common line)", "color": "#2563C8", "border": "#1D4FA0"},
                    "above line",
                    f"Rows = predictors from the largest to the smallest magnitude; each cell = one training patient (the same "
                    f"patients in Before and After). Coloured = the value's magnitude exceeds one common line of {line:.3g} units "
                    f"(the median of the predictors' 95th-percentile |value| before scaling). Large-unit predictors light up "
                    f"and would dominate distance- and gradient-based models; after {method or 'scaling'} every predictor sits "
                    f"on a comparable scale.", "Within common scale")

# CELL 36
SC = detect_scale(TR, FEATS)
SC_COLS = [c for c in FEATS if SCHEMA.get(c, {}).get("type") in ("cont", "ord")]
print(f"SD ratio max/min = {SC['ratio']:.1f}; max |skewness| = {SC['max_abs_skew']:.2f} ({SC['skew'].abs().idxmax()})")
sc_before = TR.copy()
sc_tab, sc_best, sc_trad = None, None, None
if (SC["detected"] or FORCE_DETECT) and SC_COLS:
    bench = strat_sample(TR, S("scale"))
    a, b = train_test_split(np.arange(len(bench)), test_size=0.25, stratify=bench.target, random_state=SEED)
    other = [c for c in FEATS if c not in SC_COLS]
    Xs_raw = bench[SC_COLS].to_numpy(float)
    yb = bench["target"].to_numpy(int)
    sk_b = np.nanmean(np.abs(skew(Xs_raw, axis=0)))
    cor_b = np.nan_to_num(pd.DataFrame(Xs_raw).corr().to_numpy())
    ref = np.random.default_rng(SEED).standard_normal(20000)
    res = {}
    for name in SC_FAMILY:
        with Timer(f"scale:{name}"):
            try:
                scl = make_scaler(name).fit(Xs_raw[a])
                Xt_ = scl.transform(Xs_raw)
                Xfull = np.c_[Xt_, bench[other].to_numpy(float)] if other else Xt_
                sub = Xt_[np.random.default_rng(SEED).choice(len(Xt_), min(20000, len(Xt_)), replace=False)]
                w1 = np.mean([wasserstein_distance(sub[:, j], ref) for j in range(sub.shape[1])])
                cor_a = np.nan_to_num(pd.DataFrame(Xt_).corr().to_numpy())
                iu = np.triu_indices(len(SC_COLS), 1)
                if name == "FT-Transformer":
                    actv, grads, p = ft_probe(Xfull[a], yb[a], Xfull[b])
                else:
                    snn = name == "SNN (SELU)"
                    net, hist = train_mlp(Xfull[a], yb[a], epochs=6, record=True, act=nn.SELU if snn else nn.ReLU,
                                          init=lecun_normal if snn else None, hidden=(128, 128, 64))
                    actv, grads, p = activation_logvar(net, Xfull[b][:4096]), hist["grad_norm"], mlp_predict(net, Xfull[b])
                g = np.asarray(grads)
                res[name] = {"Wasserstein to N(0,1)": w1, "Correlation preservation |Δr|": float(np.mean(np.abs(cor_b[iu] - cor_a[iu]))),
                             "Skewness reduction": float(sk_b - np.nanmean(np.abs(skew(Xt_, axis=0)))),
                             "Kurtosis |excess|": float(np.nanmean(np.abs(kurtosis(Xt_, axis=0)))),
                             "Activation variance |log σ²|": actv, "Gradient stability (CV)": float(g.std() / g.mean()),
                             "probe AUROC (info)": safe_auc(yb[b], p)}
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in SC_DIRS}
    sc_tab, sc_best, sc_trad = rank_methods(pd.DataFrame(res).T, SC_DIRS, SC_FAMILY)
    show_comparison(sc_tab, SC_DIRS, sc_best, SC_FAMILY, sc_trad)
    scaler = make_scaler(sc_best).fit(TR[SC_COLS].to_numpy(float))
    update_ordinal_allowed(scaler, SC_COLS, TR)
    TR, TE = apply_scaler(TR, scaler, SC_COLS), apply_scaler(TE, scaler, SC_COLS)
    if sc_best in ("SNN (SELU)", "FT-Transformer"):
        print(f"{sc_best} selected: inputs z-scored as that architecture prescribes; use {sc_best} as the downstream model.")
else:
    print("Feature-scale bias NOT detected -> unchanged.")
SC_AFTER = detect_scale(TR, FEATS)
RUN_LOG["Feature scale"] = {"detected": SC["detected"], "statistic": SC["statistic"], "before": SC["value"],
                            "after": SC_AFTER["value"], "selected": sc_best or "—", "best_traditional": sc_trad or "—",
                            "methods_compared": len(SC_FAMILY) if SC["detected"] else 0}
print(f"After: SD ratio {SC_AFTER['ratio']:.2f}; max |skew| {SC_AFTER['max_abs_skew']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
skb, ska = SC["skew"].abs().sort_values(ascending=False).head(12), SC_AFTER["skew"].abs()
ax[0].barh(skb.index[::-1], skb.values[::-1], color="#6FA0DA", label="before")
ax[0].barh(skb.index[::-1], ska[skb.index[::-1]].values, color="#7BC17E", alpha=0.9, label="after")
ax[0].set_xlabel("|skewness|"); ax[0].legend(); ax[0].set_title("Skewness before vs after")
if sc_tab is not None:
    mean_rank_plot(ax[1], sc_tab, sc_best, "Scaling-method comparison", SC_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "04_feature_scale_bias_dashboard.html", "Feature Scale Bias", SC["detected"],
    (f"Detected: SD ratio {SC['ratio']:.1f}, max |skew| {SC['max_abs_skew']:.2f} — {sc_best} selected (best-ranked modern method; best traditional for comparison: {sc_trad}) and applied to continuous/ordinal predictors"
     if SC["detected"] else "Not detected: scales comparable and skew small — unchanged, After = Before"),
    {"Statistic": SC["statistic"], "Value (before)": SC["value"], "Threshold": SC["threshold"], "Value (after)": SC_AFTER["value"]},
    sc_best, sc_before, TR, display_vars(FEATS),
    scale_grid(sc_before, TR, SC_COLS, sc_best),
    comparison=comparison_payload(sc_tab, SC_DIRS, sc_best, SC_FAMILY, sc_trad),
    footer="Transforms are fitted on the training split and applied unchanged to the test split. Binary and one-hot predictors are not rescaled. "
           "Distribution and box plots place Before and After each on its own range (B: / A: labels), because the step changes the units.",
    rel=True,
    chips=[f"SD ratio max/min: <b>{SC['ratio']:.1f} → {SC_AFTER['ratio']:.2f}</b>",
           f"Max |skewness|: <b>{SC['max_abs_skew']:.2f} → {SC_AFTER['max_abs_skew']:.2f}</b>"])

# CELL 38
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def dom_index(d, vocab):
    return np.array([vocab.get(str(v), len(vocab) - 1) for v in d])


class DANNRep:
    """DANN (Ganin et al., 2016): encoder trained to predict the outcome while a gradient-reversed
    domain head tries to recognise the domain (labelled source + unlabelled target rows)."""
    latent = True

    def fit(self, X, y, d, Xu=None, du=None, epochs=12, seed=SEED):
        set_seed(seed)
        alld = list(d) + (list(du) if du is not None else [])
        self.vocab = {k: i for i, k in enumerate(sorted(set(map(str, alld))) + ["<unk>"])}
        K = len(self.vocab)
        self.enc = _MLP(X.shape[1], 32, h=128).to(DEVICE)
        self.cls = nn.Linear(32, 1).to(DEVICE)
        self.dh = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, K)).to(DEVICE)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.cls.parameters()) + list(self.dh.parameters()), 1e-3)
        Xa = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        ya = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        da = torch.tensor(dom_index(d, self.vocab), device=DEVICE)
        if Xu is not None:
            Xb_ = torch.tensor(Xu, dtype=torch.float32, device=DEVICE)
            db_ = torch.tensor(dom_index(du, self.vocab), device=DEVICE)
        steps = min(max(E(epochs) * max(1, len(Xa) // 512), E(1500)), E(3000))      # 1,500-3,000 steps at any data size
        for s in range(steps):
            lam = 2 / (1 + math.exp(-10 * s / steps)) - 1
            b = torch.randint(0, len(Xa), (512,), device=DEVICE)
            z = self.enc(Xa[b])
            loss = F.binary_cross_entropy_with_logits(self.cls(z).squeeze(-1), ya[b])
            zd, dd = z, da[b]
            if Xu is not None:
                bu = torch.randint(0, len(Xb_), (512,), device=DEVICE)
                zd, dd = torch.cat([z, self.enc(Xb_[bu])]), torch.cat([dd, db_[bu]])
            loss = loss + F.cross_entropy(self.dh(GradReverse.apply(zd, lam)), dd)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def transform(self, X, d=None):
        with torch.no_grad():
            return torch.cat([self.enc(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE))
                              for i in range(0, len(X), 100000)]).cpu().numpy()


def _sqrtm(C, inv=False):
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, 1e-6, None)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


class CORALRep:
    """CORAL (Sun et al., 2016), multi-domain: every domain is whitened with its own covariance and
    re-coloured with the pooled reference covariance (shrinkage for small domains)."""
    latent = False

    def fit(self, X, y, d, Xu=None, du=None, shrink=50):
        XA = np.vstack([X, Xu]) if Xu is not None else X
        dA = np.r_[np.asarray(d, str), np.asarray(du, str)] if Xu is not None else np.asarray(d, str)
        self.mu_r, self.C_r = X.mean(0), np.cov(X.T) + 1e-3 * np.eye(X.shape[1])
        self.sr = _sqrtm(self.C_r)
        self.maps = {}
        for k in np.unique(dA):
            Xk = XA[dA == k]
            n = len(Xk)
            Ck = (np.cov(Xk.T) * (n - 1) + shrink * self.C_r) / (n - 1 + shrink) if n > 1 else self.C_r
            self.maps[k] = (Xk.mean(0), _sqrtm(Ck + 1e-3 * np.eye(X.shape[1]), inv=True) @ self.sr)
        return self

    def transform(self, X, d):
        out = X.copy()
        d = np.asarray(d, str)
        for k in np.unique(d):
            if k in self.maps:
                mu, A = self.maps[k]
                m = d == k
                out[m] = (X[m] - mu) @ A + self.mu_r
        return out


class CondAERep:
    """Domain-conditional autoencoder: encoder sees predictors only, decoder also receives the domain
    code; translation = decode every row with the reference (largest) domain's code."""
    latent = False

    def fit(self, X, y, d, Xu=None, du=None, epochs=15, seed=SEED):
        set_seed(seed)
        XA = np.vstack([X, Xu]) if Xu is not None else X
        dA = np.r_[np.asarray(d, str), np.asarray(du, str)] if Xu is not None else np.asarray(d, str)
        self.vocab = {k: i for i, k in enumerate(sorted(set(dA)))}
        self.ref = pd.Series(np.asarray(d, str)).value_counts().index[0]
        K, p = len(self.vocab), X.shape[1]
        self.enc = _MLP(p, 16, h=128).to(DEVICE)
        self.dec = _MLP(16 + K, p, h=128).to(DEVICE)
        Xt = torch.tensor(XA, dtype=torch.float32, device=DEVICE)
        Dt = F.one_hot(torch.tensor([self.vocab[k] for k in dA], device=DEVICE), K).float()
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), 1e-3)
        for _ in range(E(epochs)):
            for b in torch.randperm(len(Xt), device=DEVICE).split(512):
                loss = ((self.dec(torch.cat([self.enc(Xt[b]), Dt[b]], 1)) - Xt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.K = K
        return self

    def transform(self, X, d=None):
        out = []
        code = F.one_hot(torch.tensor([self.vocab[self.ref]], device=DEVICE), self.K).float()
        with torch.no_grad():
            for i in range(0, len(X), 100000):
                z = self.enc(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE))
                out.append(self.dec(torch.cat([z, code.expand(len(z), -1)], 1)).cpu().numpy())
        return np.vstack(out)


class IRMRep:
    """IRMv1 (Arjovsky et al., 2019): representation whose optimal linear classifier is the same in
    every source domain (gradient penalty w.r.t. a dummy classifier scale)."""
    latent = True

    def fit(self, X, y, d, Xu=None, du=None, epochs=12, seed=SEED, lam=100.0):
        set_seed(seed)
        d = np.asarray(d, str)
        envs = [np.where(d == k)[0] for k in np.unique(d)]
        envs = [e for e in envs if len(e) >= 50 and 0 < y[e].sum() < len(e)]
        self.phi = _MLP(X.shape[1], 32, h=128).to(DEVICE)
        self.w = nn.Linear(32, 1).to(DEVICE)
        opt = torch.optim.Adam(list(self.phi.parameters()) + list(self.w.parameters()), 1e-3)
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        steps = min(max(E(epochs) * max(1, len(X) // 1024), E(1500)), E(3000))
        rng = np.random.default_rng(seed)
        for s in range(steps):
            pen_w = 1.0 if s < 0.3 * steps else lam
            loss, pen = 0.0, 0.0
            for e in rng.choice(len(envs), min(4, len(envs)), replace=False):
                b = torch.tensor(rng.choice(envs[e], min(256, len(envs[e]))), device=DEVICE)
                scale = torch.ones(1, device=DEVICE, requires_grad=True)
                l = F.binary_cross_entropy_with_logits(self.w(self.phi(Xt[b])).squeeze(-1) * scale, yt[b])
                g = torch.autograd.grad(l, [scale], create_graph=True)[0]
                loss, pen = loss + l, pen + (g ** 2).sum()
            tot = (loss + pen_w * pen) / (pen_w if pen_w > 1 else 1)
            opt.zero_grad(); tot.backward(); opt.step()
        return self

    def transform(self, X, d=None):
        with torch.no_grad():
            return torch.cat([self.phi(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE))
                              for i in range(0, len(X), 100000)]).cpu().numpy()


class OTDARep:
    """Optimal-transport domain adaptation (Courty et al., 2017; POT SinkhornTransport): each domain is
    transported onto the pooled reference distribution; new rows use the barycentric map extension."""
    latent = False

    def fit(self, X, y, d, Xu=None, du=None, n_ref=2000, n_dom=1500, seed=SEED):
        import ot
        rng = np.random.default_rng(seed)
        XA = np.vstack([X, Xu]) if Xu is not None else X
        dA = np.r_[np.asarray(d, str), np.asarray(du, str)] if Xu is not None else np.asarray(d, str)
        self.ref = X[rng.choice(len(X), min(n_ref, len(X)), replace=False)]
        self.maps = {}
        for k in np.unique(dA):
            Xk = XA[dA == k]
            if len(Xk) < 5:
                continue
            Xs = Xk[rng.choice(len(Xk), min(n_dom, len(Xk)), replace=False)]
            m = ot.da.SinkhornTransport(reg_e=1.0, norm="median", max_iter=500)
            m.fit(Xs=Xs, Xt=self.ref)
            self.maps[k] = m
        return self

    def transform(self, X, d):
        out = X.copy()
        d = np.asarray(d, str)
        for k in np.unique(d):
            if k in self.maps:
                m = d == k
                idx = np.where(m)[0]
                out[idx] = np.vstack([self.maps[k].transform(Xs=X[idx[i:i + 20000]]) for i in range(0, len(idx), 20000)])
        return out


class DGTransformerRep:
    """Domain-generalisation transformer: per-feature token embedding + [CLS] token + transformer
    encoder; trained on the outcome with an MMD penalty that aligns [CLS] embeddings across source
    domains in each batch. Representation = [CLS] embedding."""
    latent = True

    class Net(nn.Module):
        def __init__(self, p, d=32):
            super().__init__()
            self.W = nn.Parameter(torch.randn(p, d) * 0.1)
            self.B = nn.Parameter(torch.zeros(p, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(layer, 2)
            self.head = nn.Linear(d, 1)

        def embed(self, x):
            tok = x[:, :, None] * self.W[None] + self.B[None]
            h = self.tr(torch.cat([self.cls.expand(len(x), -1, -1), tok], 1))
            return h[:, 0]

    @staticmethod
    def mmd(a, b):
        z = torch.cat([a, b])
        d2 = torch.cdist(z, z) ** 2
        s = d2.detach().median().clamp(min=1e-6)
        K = torch.exp(-d2 / s)
        n = len(a)
        return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()

    def fit(self, X, y, d, Xu=None, du=None, epochs=6, seed=SEED, lam=1.0):
        set_seed(seed)
        self.net = self.Net(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), 1e-3, weight_decay=1e-5)
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        d = np.asarray(d, str)
        top = pd.Series(d).value_counts().index[:10]
        idx_by = {k: np.where(d == k)[0] for k in top}
        rng = np.random.default_rng(seed)
        steps = min(max(E(epochs) * max(1, len(X) // 512), E(1500)), E(3000))
        for _ in range(steps):
            k1, k2 = rng.choice(len(top), 2, replace=False) if len(top) > 1 else (0, 0)
            b1 = rng.choice(idx_by[top[k1]], 128)
            b2 = rng.choice(idx_by[top[k2]], 128)
            br = rng.integers(0, len(X), 256)
            b = torch.tensor(np.r_[b1, b2, br], device=DEVICE)
            z = self.net.embed(Xt[b])
            loss = F.binary_cross_entropy_with_logits(self.net.head(z).squeeze(-1), yt[b]) + lam * self.mmd(z[:128], z[128:256])
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def transform(self, X, d=None):
        self.net.eval()
        with torch.no_grad():
            return torch.cat([self.net.embed(torch.tensor(X[i:i + 2048], dtype=torch.float32, device=DEVICE))
                              for i in range(0, len(X), 2048)]).cpu().numpy()


DOM_METHODS = {"DANN": DANNRep, "CORAL": CORALRep, "Autoencoder (domain-conditional)": CondAERep,
               "IRM": IRMRep, "OT domain adaptation": OTDARep, "Domain-generalisation transformer": DGTransformerRep}
DOM_FAMILY = {k: ("Modern" if k in ("IRM", "OT domain adaptation", "Domain-generalisation transformer") else "Traditional")
              for k in DOM_METHODS}
DOM_DIRS = {"MMD": "min", "Domain-classifier AUC": "min", "Wasserstein (sliced)": "min", "Worst-group accuracy": "max",
            "DG gap": "min", "Target-domain AUROC": "max", "OOD performance": "max",
            "Source-domain AUROC (accuracy guard)": "max"}


def rbf_mmd(a, b, seed=SEED):
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(1500, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(1500, len(b)), replace=False)]
    z = np.vstack([a, b])
    d2 = cdist(z, z, "sqeuclidean")              # one matrix of distances, so wide data fits in memory
    K = np.exp(-d2 / np.median(d2[d2 > 0]))
    n = len(a)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def sliced_w1(a, b, n_proj=50, seed=SEED):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((a.shape[1], n_proj))
    P /= np.linalg.norm(P, axis=0)
    return float(np.mean([wasserstein_distance(a @ P[:, k], b @ P[:, k]) for k in range(n_proj)]))


def domain_eval(Rs_tr, ys_tr, Rs_te, ys_te, Rt, yt, dt, seed=SEED):
    st = Standardizer().fit(Rs_tr)
    Rs_tr, Rs_te, Rt = st.transform(Rs_tr), st.transform(Rs_te), st.transform(Rt)
    rng = np.random.default_rng(seed)
    a = Rs_te[rng.choice(len(Rs_te), min(3000, len(Rs_te)), replace=False)]
    b = Rt[rng.choice(len(Rt), min(3000, len(Rt)), replace=False)]
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)
    dc_auc = safe_auc(lab, oof_probs(np.vstack([a, b]), lab, folds=3, seed=seed))
    p_val, thr, m = fit_score(Rs_tr, ys_tr, Rs_te, seed=seed)
    pt = m.predict_proba(Rt)[:, 1]
    auc_s, auc_t = safe_auc(ys_te, p_val), safe_auc(yt, pt)
    per_auc, per_ba = [], []
    for k in np.unique(dt):
        mk = dt == k
        if mk.sum() >= 100 and 10 <= yt[mk].sum() <= mk.sum() - 10:
            per_auc.append(safe_auc(yt[mk], pt[mk]))
            per_ba.append(balanced_accuracy_score(yt[mk], (pt[mk] >= thr).astype(int)))
    return {"MMD": rbf_mmd(a, b), "Domain-classifier AUC": dc_auc, "Wasserstein (sliced)": sliced_w1(a, b),
            "Worst-group accuracy": float(np.min(per_ba)) if per_ba else np.nan, "DG gap": auc_s - auc_t,
            "Target-domain AUROC": auc_t, "OOD performance": float(np.mean(per_auc)) if per_auc else np.nan,
            "Source-domain AUROC (accuracy guard)": auc_s}


def domain_identifiable(X, dom, shown, seed=SEED):
    """1 where a cross-validated one-vs-rest domain classifier, seeing only the predictors, gives the patient's own
    domain at least twice its base share (capped at halfway to 1): the predictors reveal where the patient came from."""
    st = np.zeros(len(X), int)
    for k in shown:
        yk = (dom == k).astype(int)
        if yk.sum() < 20 or yk.sum() > len(yk) - 20:
            continue
        p = oof_probs(X, yk, folds=3, seed=seed, iters=60)
        share = yk.mean()
        st[(yk == 1) & (p >= min(2 * share, (1 + share) / 2))] = 1
    return st


def domain_grid(before, after, feats_b, feats_a, changed, method, n_max=20000):
    st = {"v": 1, "label": "Domain leaked", "color": "#C0533E", "border": "#9B4230"}
    if not DOMAIN_COL:
        return grid_payload("Domain Leakage Heatmap", [st], [], "", "No leak",
                            empty="This dataset has no site, region or hospital identifier, so encoding/domain bias cannot be "
                                  "measured; the data are unchanged (After = Before).")
    pos = np.sort(np.random.default_rng(SEED).choice(len(before), min(len(before), E(n_max)), replace=False))
    dom = domain_labels(before).to_numpy()[pos]
    shown = [k for k in pd.Series(dom).value_counts().index if k != "other"][:10]
    sb = domain_identifiable(before[feats_b].to_numpy(float)[pos], dom, shown)
    sa = domain_identifiable(after[feats_a].to_numpy(float)[pos], dom, shown) if changed else sb
    groups = [(f"{DOMAIN_COL.replace('meta_domain_', '')} {k}", dom == k) for k in shown]
    return group_grid("Domain Leakage Heatmap", groups, sb, sa, [st], "leak",
                      f"Rows = the {len(shown)} largest domains ({DOMAIN_COL.replace('meta_domain_', '')}); each cell = one training "
                      f"patient of that domain (the same patients in Before and After; {len(pos):,} patients evaluated). Coloured = "
                      f"a cross-validated classifier that sees only the predictors gives the patient's true domain at least twice its "
                      f"base share, i.e. the predictors are leaking domain identity. After = the same classifier refitted on the "
                      f"output of {method or 'the step'}.", "No leak")

# CELL 39
DOM = detect_domain(TR, FEATS)
print(f"Domain detection: {DOM['statistic']} = {DOM['value']} ({'applicable' if DOM.get('applicable') else 'not applicable'})")
dom_before, feats_before_dom = TR.copy(), list(FEATS)
dom_tab, dom_best, dom_trad = None, None, None
if DOM["detected"] or (FORCE_DETECT and DOM.get("applicable")):
    bench = TR.sample(min(S("domain"), len(TR)), random_state=SEED)
    dstr = bench[DOMAIN_COL].astype(str).to_numpy()
    doms = np.array(sorted(np.unique(dstr)))
    rng = np.random.default_rng(SEED)
    tgt = set(rng.choice(doms, max(1, int(round(0.2 * len(doms)))), replace=False))
    is_t = np.isin(dstr, list(tgt))
    stb = Standardizer().fit(bench[FEATS].to_numpy(float)[~is_t])
    Xb = stb.transform(bench[FEATS].to_numpy(float))
    yb = bench["target"].to_numpy(int)
    s_idx = np.where(~is_t)[0]
    s_tr, s_te = train_test_split(s_idx, test_size=0.2, stratify=yb[s_idx], random_state=SEED)
    t_idx = np.where(is_t)[0]
    print(f"Benchmark: {len(bench):,} rows | {len(doms) - len(tgt)} source / {len(tgt)} target domains | target rows {len(t_idx):,}")
    res = {"No adaptation (raw predictors)": domain_eval(Xb[s_tr], yb[s_tr], Xb[s_te], yb[s_te], Xb[t_idx], yb[t_idx], dstr[t_idx])}
    for name, cls in DOM_METHODS.items():
        with Timer(f"domain:{name}"):
            try:
                rep = cls().fit(Xb[s_tr], yb[s_tr], dstr[s_tr], Xb[t_idx], dstr[t_idx])
                R = {k: rep.transform(Xb[i], dstr[i]) for k, i in (("s_tr", s_tr), ("s_te", s_te), ("t", t_idx))}
                res[name] = domain_eval(R["s_tr"], yb[s_tr], R["s_te"], yb[s_te], R["t"], yb[t_idx], dstr[t_idx])
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in DOM_DIRS}
    dom_tab, dom_best, dom_trad = rank_methods(pd.DataFrame(res).T, DOM_DIRS, DOM_FAMILY, reference_rows=["No adaptation (raw predictors)"])
    show_comparison(dom_tab, DOM_DIRS, dom_best, DOM_FAMILY, dom_trad)
    with Timer(f"domain:apply {dom_best}"):
        stf = Standardizer().fit(TR[FEATS].to_numpy(float))
        Xtr_, Xte_ = stf.transform(TR[FEATS].to_numpy(float)), stf.transform(TE[FEATS].to_numpy(float))
        dtr, dte = TR[DOMAIN_COL].astype(str).to_numpy(), TE[DOMAIN_COL].astype(str).to_numpy()
        rep = DOM_METHODS[dom_best]().fit(Xtr_, TR["target"].to_numpy(int), dtr)   # fitted on training rows only
        Rtr, Rte = rep.transform(Xtr_, dtr), rep.transform(Xte_, dte)
        if rep.latent:
            if "FEATURE_SPACE" not in globals():
                FEATURE_SPACE = (TR.copy(), TE.copy(), list(FEATS), "after steps 1-7, before the encoding/domain step")
            new = [f"z{i + 1}" for i in range(Rtr.shape[1])]
            keep = [c for c in TR.columns if c not in FEATS]
            TR = pd.concat([TR[keep].reset_index(drop=True), pd.DataFrame(Rtr, columns=new)], axis=1)
            TE = pd.concat([TE[keep].reset_index(drop=True), pd.DataFrame(Rte, columns=new)], axis=1)
            FEATS = new
            for c in new:
                SCHEMA[c] = {"type": "cont"}
            print(f"{dom_best} selected: predictors replaced by its {len(new)}-dimensional domain-invariant representation.")
        else:
            TR[FEATS] = stf.inverse(Rtr)
            TE[FEATS] = stf.inverse(Rte)
            TR[FEATS] = snap(TR[FEATS]).to_numpy()
            TE[FEATS] = snap(TE[FEATS]).to_numpy()
            print(f"{dom_best} selected: predictors aligned across domains in the original feature space.")
else:
    print("Domain bias NOT detected" + ("" if DOM.get("applicable") else " (no domain identifier)") + " -> unchanged.")
DOM_AFTER = detect_domain(TR, FEATS) if DOM.get("applicable") else DOM
RUN_LOG["Encoding / domain"] = {"detected": DOM["detected"], "statistic": DOM["statistic"], "before": DOM["value"],
                                "after": DOM_AFTER["value"], "selected": dom_best or "—", "best_traditional": dom_trad or "—",
                                "methods_compared": len(DOM_METHODS) if DOM["detected"] else 0}
print("After:", DOM_AFTER["value"])

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
if DOM.get("applicable"):
    ax[0].bar(["domain AUC", "DG gap"], [DOM["domain_auc"], DOM["dg_gap"]], width=0.35, color="#6FA0DA", label="before")
    ax[0].bar(np.arange(2) + 0.35, [DOM_AFTER["domain_auc"], DOM_AFTER["dg_gap"]], width=0.35, color="#7BC17E", label="after")
    ax[0].axhline(0.6, ls="--", c="grey", lw=0.8); ax[0].legend(); ax[0].set_title("Domain encoding before vs after")
else:
    ax[0].text(0.5, 0.5, "No domain identifier in this dataset", ha="center"); ax[0].axis("off")
if dom_tab is not None:
    mean_rank_plot(ax[1], dom_tab, dom_best, "Domain-method comparison", DOM_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

dheat = domain_grid(dom_before, TR, feats_before_dom, FEATS, dom_best is not None, dom_best)
write_dashboard(
    "08_encoding_domain_bias_dashboard.html", "Encoding / Domain Bias", DOM["detected"],
    (f"Detected: {DOM['statistic']} = {DOM['value']} — {dom_best} selected (best-ranked modern method; best traditional for comparison: {dom_trad}) and applied"
     if DOM["detected"] else ("Not detected: " + f"{DOM['statistic']} = {DOM['value']} (thresholds {DOM['threshold']})" if DOM.get("applicable")
                              else "Not applicable: no domain identifier") + " — unchanged, After = Before"),
    {"Statistic": DOM["statistic"], "Value (before)": DOM["value"], "Threshold": DOM["threshold"], "Value (after)": DOM_AFTER["value"],
     "Domains": DOM.get("n_domains", "—")},
    dom_best, dom_before, TR, display_vars(feats_before_dom) + [c for c in display_vars(FEATS) if c not in feats_before_dom],
    dheat,
    comparison=comparison_payload(dom_tab, DOM_DIRS, dom_best, DOM_FAMILY, dom_trad),
    footer=("Domain = " + DOMAIN_COL + ". Target domains in the benchmark are whole held-out domains; their labels are never used for fitting."
            if DOMAIN_COL else "This dataset records no site, region or hospital, so there is nothing to align.")
           + (" The selected method replaced the predictors with a learned representation (z1, z2, ...), shown as After-only variables."
              if dom_best is not None and FEATS != feats_before_dom else ""),
    chips=([f"Domain-classifier AUC: <b>{DOM['domain_auc']:.3f} → {DOM_AFTER['domain_auc']:.3f}</b>",
            f"AUROC drop on unseen domains: <b>{DOM['dg_gap']:+.3f} → {DOM_AFTER['dg_gap']:+.3f}</b>"] if DOM.get("applicable") else
           ["No domain identifier"]))

# CELL 41
def cb_weights(y, beta=0.999):
    n = np.bincount(y.astype(int), minlength=2).astype(float)
    eff = (1 - beta ** n) / (1 - beta)
    w = 1 / eff
    w = w / w.sum() * 2
    return w[y.astype(int)]


def _gen_frame(X):
    return pd.DataFrame(X, columns=FEATS)


def rs_imblearn(sampler):
    def f(X, y, seed=SEED):
        s = sampler(seed)
        Xr, yr = s.fit_resample(X, y)
        keep_idx = getattr(s, "sample_indices_", None)
        return Xr, yr, None, keep_idx
    return f


def rs_tabddpm(X, y, seed=SEED):
    """TabDDPM (Kotelnikov et al., 2023): Gaussian diffusion model trained on the minority class;
    new minority rows are sampled and projected onto valid categorical / ordinal levels."""
    mi = int(np.argmin(np.bincount(y)))
    Xm = X[y == mi]
    need = int((y != mi).sum() - len(Xm))
    st = Standardizer().fit(Xm)
    gd = GaussianDiffusion(X.shape[1], seed=seed).fit(st.transform(Xm), iters=4000)
    Xs = snap(_gen_frame(st.inverse(gd.sample(need)))).to_numpy()
    return np.vstack([X, Xs]), np.r_[y, np.full(need, mi)], None, None


def rs_ctgan(X, y, seed=SEED):
    """CTGAN (Xu et al., 2019) trained on minority rows; discrete columns use mode-specific handling."""
    from ctgan import CTGAN
    mi = int(np.argmin(np.bincount(y)))
    dfm = _gen_frame(X[y == mi])
    disc = [c for c in FEATS if SCHEMA.get(c, {}).get("type") in ("bin", "ord", "onehot")]
    for c in disc:
        dfm[c] = dfm[c].round(6)
    need = int((y != mi).sum() - len(dfm))
    m = CTGAN(epochs=E(150), batch_size=500, enable_gpu=torch.cuda.is_available(), verbose=False)
    m.set_random_state(seed)
    m.fit(dfm, discrete_columns=disc)
    Xs = snap(m.sample(need)[FEATS].astype(float)).to_numpy()
    return np.vstack([X, Xs]), np.r_[y, np.full(need, mi)], None, None


def rs_cbfocal(X, y, seed=SEED):
    return X, y, cb_weights(y), None


def rs_deepsmote(X, y, seed=SEED, epochs=30):
    """DeepSMOTE (Dablain et al., 2022): encoder/decoder trained with reconstruction plus a
    permutation-penalty loss; SMOTE interpolation in the latent space; decoded rows are snapped."""
    set_seed(seed)
    st = Standardizer().fit(X)
    Z = torch.tensor(st.transform(X), dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    d = X.shape[1]
    enc, dec = _MLP(d, 16, h=128).to(DEVICE), _MLP(16, d, h=128).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
    for _ in range(E(epochs)):
        for b in torch.randperm(len(Z), device=DEVICE).split(512):
            h = enc(Z[b])
            loss = ((dec(h) - Z[b]) ** 2).mean()
            for k in (0, 1):                                    # permutation penalty within each class
                idx = (yt[b] == k).nonzero().squeeze(-1)
                if len(idx) > 1:
                    perm = idx[torch.randperm(len(idx), device=DEVICE)]
                    loss = loss + 0.5 * ((dec(h[perm]) - Z[b][perm]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    mi = int(np.argmin(np.bincount(y)))
    with torch.no_grad():
        H = enc(Z).cpu().numpy()
    need = int((y != mi).sum() - (y == mi).sum())
    Hm = H[y == mi]
    nn_ = NearestNeighbors(n_neighbors=6).fit(Hm)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(Hm), need)
    nb = nn_.kneighbors(Hm[base], return_distance=False)[np.arange(need), rng.integers(1, 6, need)]
    Hs = Hm[base] + rng.random((need, 1)) * (Hm[nb] - Hm[base])
    with torch.no_grad():
        Xs = st.inverse(dec(torch.tensor(Hs, dtype=torch.float32, device=DEVICE)).cpu().numpy())
    Xs = snap(_gen_frame(Xs)).to_numpy()
    return np.vstack([X, Xs]), np.r_[y, np.full(need, mi)], None, None


IMB_METHODS = {
    "SMOTE": rs_imblearn(lambda s: __import__("imblearn.over_sampling", fromlist=["SMOTE"]).SMOTE(random_state=s, k_neighbors=5)),
    "ADASYN": rs_imblearn(lambda s: __import__("imblearn.over_sampling", fromlist=["ADASYN"]).ADASYN(random_state=s, n_neighbors=5)),
    "Borderline-SMOTE": rs_imblearn(lambda s: __import__("imblearn.over_sampling", fromlist=["BorderlineSMOTE"]).BorderlineSMOTE(random_state=s, kind="borderline-1")),
    "NearMiss": rs_imblearn(lambda s: __import__("imblearn.under_sampling", fromlist=["NearMiss"]).NearMiss(version=1, n_neighbors=3)),
    "TabDDPM": rs_tabddpm, "CTGAN": rs_ctgan, "Class-Balanced Focal Loss": rs_cbfocal, "DeepSMOTE": rs_deepsmote}
IMB_FAMILY = {k: ("Modern" if k in ("TabDDPM", "CTGAN", "Class-Balanced Focal Loss", "DeepSMOTE") else "Traditional") for k in IMB_METHODS}
IMB_DIRS = {"Imbalance ratio": "min", "Class entropy": "max", "Balanced F1": "max", "Macro-F1": "max", "Balanced accuracy": "max",
            "PR-AUC": "max", "G-mean": "max", "KS": "max", "TVD": "min"}


def tvd(real, gen):
    """Mean total-variation distance between real and generated rows over predictors (quantile bins)."""
    if len(gen) == 0:
        return 0.0
    out = []
    for j in range(real.shape[1]):
        r, g = real[:, j], gen[:, j]
        u = np.unique(r)
        if len(u) <= 15:
            edges = np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
        else:
            edges = np.unique(np.r_[-np.inf, np.quantile(r, np.linspace(0.1, 0.9, 9)), np.inf])
        pr = np.histogram(r, edges)[0] / len(r)
        pg = np.histogram(g, edges)[0] / len(g)
        out.append(0.5 * np.abs(pr - pg).sum())
    return float(np.mean(out))


def imbalance_eval(name, X, y, Xte, yte, seed=SEED):
    Xr, yr, w, keep = IMB_METHODS[name](X, y, seed)
    mi = int(np.argmin(np.bincount(y)))
    st = Standardizer().fit(X)                       # same input scaling for the evaluator in every method
    if name == "Class-Balanced Focal Loss":
        net, _ = train_mlp(st.transform(Xr), yr, loss="focal", focal={"gamma": 2.0}, weights=w, epochs=8, seed=seed)
        mass = np.bincount(yr, weights=w, minlength=2)
        t = 0.0
    else:
        net, _ = train_mlp(st.transform(Xr), yr, epochs=8, seed=seed)
        mass = np.bincount(yr, minlength=2).astype(float)
        if keep is not None:      # under-sampling: kept vs original majority
            ma = 1 - mi
            kept = Xr[yr == ma]
            t = tvd(X[y == ma], kept)
        else:
            t = tvd(X[y == mi], Xr[len(X):])
    p = mlp_predict(net, st.transform(Xte))
    cm = clf_metrics(yte, p, 0.5)
    q = mass / mass.sum()
    ent = float(-(q * np.log2(np.clip(q, 1e-12, 1))).sum())
    return {"Imbalance ratio": float(mass.max() / mass.min()), "Class entropy": ent,
            "Balanced F1": cm["Balanced F1"], "Macro-F1": cm["Macro-F1"], "Balanced accuracy": cm["Balanced accuracy"],
            "PR-AUC": cm["PR-AUC"], "G-mean": cm["G-mean"], "KS": cm["KS"], "TVD": t,
            "AUROC (info)": cm["AUROC"], "train rows (info)": len(Xr)}


def class_grid(before, after, method):
    """Rows = all training rows and each protected group; 20 cells per grid, filled in proportion to the
    (weighted) share of cases, as in the reference class-imbalance heatmap."""
    def shares(df, m):
        w = df["w"].to_numpy(float)[m]
        y = df["target"].to_numpy(int)[m]
        syn = df["meta_synthetic"].to_numpy(int)[m] if "meta_synthetic" in df else np.zeros(m.sum(), int)
        tot = w.sum()
        return (w[(y == 1) & (syn == 0)].sum() / tot, w[(y == 1) & (syn == 1)].sum() / tot) if tot > 0 else (0.0, 0.0)
    groups = [("All training rows", lambda df: np.ones(len(df), bool))]
    for a_ in PROTECTED + AUDIT_ONLY:
        for k in sorted(before[a_].dropna().astype(str).unique()):
            groups.append((f"{a_.replace('sens_', '').replace('_', ' ')} = {k}", lambda df, a_=a_, k=k: (df[a_].astype(str) == k).to_numpy()))
    rows = []
    for label, fn in groups:
        mb, ma = fn(before), fn(after)
        if not mb.any():
            continue
        rb, ra = shares(before, mb), shares(after, ma)
        cb, ca = waffle_row(label, rb, ra)
        rows.append(grid_row(label, cb, ca, pct_txt(100 * sum(rb), "cases"), pct_txt(100 * sum(ra), "cases")))
    st = [{"v": 1, "label": "Minority (case)", "color": "#D9534F", "border": "#B23B37"},
          {"v": 2, "label": "Synthetic minority (generated)", "color": "#F2A49F", "border": "#D9776F"}]
    return grid_payload("Class Imbalance Heatmap", _used_states(st, rows), rows,
                        "Rows = all training rows and each protected group; the 20 cells of each grid are filled in proportion to "
                        "the share of cases (outcome = 1) in that group, weighted by the training sample weights. Before = the step "
                        f"input; After = the output of {method or 'the step'} (generated rows shown in light red).",
                        "Majority (non-case)")

# CELL 42
IMB = detect_imbalance(TR)
print(f"Imbalance ratio {IMB['value']:.2f} (class counts {IMB['counts']})")
imb_before = TR.copy()
imb_tab, imb_best, imb_trad = None, None, None
if IMB["detected"] or FORCE_DETECT:
    bench = strat_sample(TR, S("imbalance"))
    Xb, yb = bench[FEATS].to_numpy(float), bench["target"].to_numpy(int)
    Xte_i, yte_i = part(TE, "test")[FEATS].to_numpy(float), part(TE, "test")["target"].to_numpy(int)
    stI = Standardizer().fit(Xb)
    res = {}
    base_net, _ = train_mlp(stI.transform(Xb), yb, epochs=8)
    cmb = clf_metrics(yte_i, mlp_predict(base_net, stI.transform(Xte_i)), 0.5)
    res["No rebalancing (reference)"] = {"Imbalance ratio": IMB["value"], "Class entropy": float(-(np.bincount(yb) / len(yb) * np.log2(np.bincount(yb) / len(yb))).sum()),
                                         **{k: cmb[k] for k in ["Balanced F1", "Macro-F1", "Balanced accuracy", "PR-AUC", "G-mean", "KS"]},
                                         "TVD": 0.0, "AUROC (info)": cmb["AUROC"], "train rows (info)": len(Xb)}
    for name in IMB_METHODS:
        with Timer(f"imbalance:{name}"):
            try:
                r = imbalance_eval(name, Xb, yb, Xte_i, yte_i)
                res[name] = r
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in IMB_DIRS}
    imb_tab, imb_best, imb_trad = rank_methods(pd.DataFrame(res).T, IMB_DIRS, IMB_FAMILY, reference_rows=["No rebalancing (reference)"])
    show_comparison(imb_tab, IMB_DIRS, imb_best, IMB_FAMILY, imb_trad)
    with Timer(f"imbalance:apply {imb_best}"):
        X, y = TR[FEATS].to_numpy(float), TR["target"].to_numpy(int)
        Xr, yr, w, keep = IMB_METHODS[imb_best](X, y, SEED)
        other = [c for c in TR.columns if c not in FEATS]
        if imb_best == "Class-Balanced Focal Loss":
            TR["w"] = TR["w"].to_numpy() * w
            TR["meta_synthetic"] = 0
        elif keep is not None:
            TR = TR.iloc[keep].copy()
            TR["meta_synthetic"] = 0
        else:
            n0 = len(X)
            gen = Xr[n0:]
            mi = int(np.argmin(np.bincount(y)))
            real_min = np.where(y == mi)[0]
            stn = Standardizer().fit(X)
            nb = NearestNeighbors(n_neighbors=1).fit(stn.transform(X[real_min])).kneighbors(stn.transform(gen), return_distance=False)[:, 0]
            synth = pd.concat([TR.iloc[real_min[nb]][other].reset_index(drop=True), pd.DataFrame(gen, columns=FEATS)], axis=1)
            synth["target"] = mi
            synth["meta_synthetic"] = 1
            TR = TR.copy()
            TR["meta_synthetic"] = 0
            TR = pd.concat([TR, synth[TR.columns]], ignore_index=True)
    print(f"Training rows {len(imb_before):,} -> {len(TR):,}")
else:
    print("Class-imbalance bias NOT detected -> unchanged.")
if "meta_synthetic" not in TR:
    TR["meta_synthetic"] = 0
TE["meta_synthetic"] = 0
IMB_AFTER = detect_imbalance(TR, TR["w"])
RUN_LOG["Class imbalance"] = {"detected": IMB["detected"], "statistic": IMB["statistic"], "before": IMB["value"],
                              "after": IMB_AFTER["value"], "selected": imb_best or "—", "best_traditional": imb_trad or "—",
                              "methods_compared": len(IMB_METHODS) if IMB["detected"] else 0}
print(f"Imbalance ratio after (weighted): {IMB_AFTER['value']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].bar([0, 1], [IMB["counts"][0] / 1000, IMB["counts"][1] / 1000], width=0.35, color="#6FA0DA", label="before")
ax[0].bar([0.35, 1.35], [IMB_AFTER["counts"][0] / 1000, IMB_AFTER["counts"][1] / 1000], width=0.35, color="#7BC17E", label="after")
ax[0].set_xticks([0.17, 1.17], ["label 0", "label 1"]); ax[0].set_ylabel("rows (thousands, weighted)"); ax[0].legend()
ax[0].set_title("Class balance before vs after")
if imb_tab is not None:
    mean_rank_plot(ax[1], imb_tab, imb_best, "Imbalance-method comparison", IMB_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "02_class_imbalance_bias_dashboard.html", "Class Imbalance Bias", IMB["detected"],
    (f"Detected: imbalance ratio {IMB['value']:.2f} — {imb_best} selected (best-ranked modern method; best traditional for comparison: {imb_trad}) and applied to the training split"
     if IMB["detected"] else f"Not detected: imbalance ratio {IMB['value']:.2f} (threshold 1.5) — unchanged, After = Before"),
    {"Statistic": IMB["statistic"], "Value (before)": f"{IMB['value']:.3f}", "Threshold": IMB["threshold"],
     "Value (after)": f"{IMB_AFTER['value']:.3f}"},
    imb_best, imb_before, TR, display_vars(FEATS),
    class_grid(imb_before, TR, imb_best),
    comparison=comparison_payload(imb_tab, IMB_DIRS, imb_best, IMB_FAMILY, imb_trad),
    footer="Rebalancing is applied to the training split only. The test split keeps the real prevalence, so test metrics show real-world performance.",
    chips=[f"Imbalance ratio: <b>{IMB['value']:.2f} → {IMB_AFTER['value']:.2f}</b>",
           f"Generated rows: <b>{int(TR['meta_synthetic'].sum()):,}</b>"])

# CELL 44
def inter_groups(df):
    parts = [df[a].astype(object).where(df[a].notna(), "Unknown").astype(str) for a in PROTECTED]
    g = parts[0]
    for p in parts[1:]:
        g = g + "|" + p
    return g.to_numpy()


PROT_FEATS = [c for c in PROTECTED_FEATURES if c in FEATS]


def dm_reweight(Xtr, ytr, gtr, seed=SEED):
    df = pd.DataFrame({"g": gtr, "y": ytr})
    pg, py = df.g.value_counts(normalize=True), df.y.value_counts(normalize=True)
    pgy = df.groupby(["g", "y"]).size() / len(df)
    w = np.array([pg[g] * py[y] / pgy[(g, y)] for g, y in zip(gtr, ytr)])
    return Xtr, ytr, gtr, w, None


def dm_fairsmote(Xtr, ytr, gtr, seed=SEED, feats=None):
    """Fair-SMOTE (Chakraborty et al., 2021): every (group, class) subgroup is over-sampled to a common
    size by SMOTE-style interpolation inside the subgroup, then 'situation testing' removes rows whose
    prediction flips when the protected feature is flipped."""
    feats = feats or FEATS
    rng = np.random.default_rng(seed)
    keys = pd.Series(list(zip(gtr, ytr)))
    sizes = keys.value_counts()
    target = int(min(sizes.max(), 3 * len(ytr) / len(sizes)))
    Xs, ys, gs = [Xtr], [ytr], [gtr]
    for (g, y), n in sizes.items():
        if n >= target:
            continue
        idx = np.where((gtr == g) & (ytr == y))[0]
        need = target - n
        if len(idx) < 2:
            new = Xtr[rng.choice(idx, need)]
        else:
            k = min(5, len(idx) - 1)
            nb = NearestNeighbors(n_neighbors=k + 1).fit(Xtr[idx]).kneighbors(Xtr[idx], return_distance=False)
            base = rng.integers(0, len(idx), need)
            mate = nb[base, rng.integers(1, k + 1, need)]
            new = Xtr[idx][base] + rng.random((need, 1)) * (Xtr[idx][mate] - Xtr[idx][base])
        Xs.append(snap(pd.DataFrame(new, columns=feats), feats).to_numpy() if feats == FEATS else new)
        ys.append(np.full(need, y))
        gs.append(np.full(need, g, dtype=object))
    X2, y2, g2 = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    src = np.r_[np.arange(len(ytr)), np.full(len(y2) - len(ytr), -1)]      # original row index, -1 = synthetic
    pf = [feats.index(c) for c in PROT_FEATS if c in feats]
    if pf:
        m = hgb(seed).fit(X2, y2)
        Xf = X2.copy()
        for j in pf:
            vals = np.unique(Xtr[:, j])
            if len(vals) == 2:
                Xf[:, j] = np.where(np.isclose(X2[:, j], vals[0]), vals[1], vals[0])
        keep = m.predict(X2) == m.predict(Xf)
        X2, y2, g2, src = X2[keep], y2[keep], g2[keep], src[keep]
    return X2, y2, g2, None, src


class OTRepair:
    """Optimal-transport repair (Feldman-style total repair via POT): each protected group's predictor
    distribution is transported onto the pooled distribution; test rows use the fitted map of their group."""
    def fit(self, X, g, seed=SEED, n=1500):
        import ot
        rng = np.random.default_rng(seed)
        self.cols = [j for j, c in enumerate(CUR_FEATS) if c not in PROT_FEATS]
        ref = X[rng.choice(len(X), min(2000, len(X)), replace=False)][:, self.cols]
        self.maps = {}
        for k in np.unique(g):
            Xk = X[g == k][:, self.cols]
            if len(Xk) < 30:
                continue
            m = ot.da.SinkhornTransport(reg_e=1.0, norm="median", max_iter=500)
            m.fit(Xs=Xk[rng.choice(len(Xk), min(n, len(Xk)), replace=False)], Xt=ref)
            self.maps[k] = m
        return self

    def transform(self, X, g):
        out = X.copy()
        for k, m in self.maps.items():
            idx = np.where(g == k)[0]
            if len(idx):
                out[np.ix_(idx, self.cols)] = np.vstack([m.transform(Xs=X[idx[i:i + 20000]][:, self.cols]) for i in range(0, len(idx), 20000)])
        return out


class AdvDebias:
    """Adversarial debiasing (Zhang et al., 2018; representation form): encoder + outcome head, while a
    gradient-reversed adversary tries to predict the protected group from (z, y), i.e. equalised odds."""
    def fit(self, X, y, g, epochs=10, seed=SEED):
        set_seed(seed)
        self.vocab = {k: i for i, k in enumerate(sorted(set(g)))}
        K = len(self.vocab)
        self.enc = _MLP(X.shape[1], 32, h=128).to(DEVICE)
        head = nn.Linear(32, 1).to(DEVICE)
        adv = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, K)).to(DEVICE)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(head.parameters()) + list(adv.parameters()), 1e-3)
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        gt = torch.tensor([self.vocab[k] for k in g], device=DEVICE)
        steps = min(max(E(epochs) * max(1, len(X) // 512), E(1500)), E(3000))      # 1,500-3,000 steps at any data size
        for s in range(steps):
            lam = 2 / (1 + math.exp(-10 * s / steps)) - 1
            b = torch.randint(0, len(Xt), (512,), device=DEVICE)
            z = self.enc(Xt[b])
            l = F.binary_cross_entropy_with_logits(head(z).squeeze(-1), yt[b])
            l = l + F.cross_entropy(adv(torch.cat([GradReverse.apply(z, lam), yt[b][:, None]], 1)), gt[b])
            opt.zero_grad(); l.backward(); opt.step()
        return self

    def transform(self, X):
        with torch.no_grad():
            return torch.cat([self.enc(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE)) for i in range(0, len(X), 100000)]).cpu().numpy()


class FairContrastive:
    """Fair supervised contrastive learning (FSCL, Park et al., 2022): positives = same outcome but a
    different protected group; negatives = different outcome within the same group, so the embedding
    groups patients by outcome rather than by demographic group."""
    def fit(self, X, y, g, epochs=10, seed=SEED, tau=0.1):
        set_seed(seed)
        self.enc = _MLP(X.shape[1], 32, h=128).to(DEVICE)
        opt = torch.optim.Adam(self.enc.parameters(), 1e-3)
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, device=DEVICE)
        vocab = {k: i for i, k in enumerate(sorted(set(g)))}
        gt = torch.tensor([vocab[k] for k in g], device=DEVICE)
        steps = min(max(E(epochs) * max(1, len(X) // 512), E(1500)), E(3000))      # 1,500-3,000 steps at any data size
        for _ in range(steps):
            b = torch.randint(0, len(Xt), (512,), device=DEVICE)
            z = F.normalize(self.enc(Xt[b]), dim=1)
            sim = z @ z.T / tau
            sy = yt[b][:, None] == yt[b][None, :]
            sg = gt[b][:, None] == gt[b][None, :]
            eye = torch.eye(len(b), dtype=torch.bool, device=DEVICE)
            pos = sy & ~sg & ~eye
            neg = ~sy & sg
            den = torch.logsumexp(sim.masked_fill(~(pos | neg), -1e9), 1)
            if not pos.any():
                continue
            lp = (sim - den[:, None]).masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1)
            loss = -(lp[pos.any(1)]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def transform(self, X):
        with torch.no_grad():
            return F.normalize(torch.cat([self.enc(torch.tensor(X[i:i + 100000], dtype=torch.float32, device=DEVICE))
                                          for i in range(0, len(X), 100000)]), dim=1).cpu().numpy()


class ModelWrap:
    """Feature transform + fitted classifier, exposing predict_proba on a DataFrame (needed by DiCE)."""
    def __init__(self, model, transform=None):
        self.model, self.tf = model, transform

    def predict_proba(self, df):
        X = df[CUR_FEATS].to_numpy(float) if isinstance(df, pd.DataFrame) else np.asarray(df, float)
        return self.model.predict_proba(self.tf(X) if self.tf else X)

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


DICE_BACKEND = {"name": None}


def _random_cf(wrap, train_df, query_df, vary, seed=SEED, n_samples=600):
    """Same algorithm as DiCE's 'random' method: sample candidates that change 1-3 actionable features to
    values drawn from the training data, keep those that flip the prediction, return the closest one
    (MAD-normalised L1), then revert any change that is not needed (post-hoc sparsity)."""
    rng = np.random.default_rng(seed)
    T = train_df[CUR_FEATS].to_numpy(float)
    mad = np.median(np.abs(T - np.median(T, 0)), 0)
    mad[mad < 1e-6] = T.std(0)[mad < 1e-6] + 1e-6
    vidx = [CUR_FEATS.index(c) for c in vary]
    out = []
    for i in range(len(query_df)):
        q = query_df[CUR_FEATS].to_numpy(float)[i]
        C = np.repeat(q[None], n_samples, 0)
        for r in range(n_samples):
            for j in rng.choice(vidx, rng.integers(1, min(3, len(vidx)) + 1), replace=False):
                C[r, j] = T[rng.integers(0, len(T)), j]
        p = wrap.predict_proba(pd.DataFrame(C, columns=CUR_FEATS))[:, 1]
        ok = p >= 0.5
        if not ok.any():
            out.append(np.full(len(CUR_FEATS), np.nan))
            continue
        cand = C[ok]
        best = cand[np.argmin((np.abs(cand - q) / mad).sum(1))].copy()
        for j in np.where(np.abs(best - q) > 1e-9)[0]:
            trial = best.copy()
            trial[j] = q[j]
            if wrap.predict_proba(pd.DataFrame(trial[None], columns=CUR_FEATS))[0, 1] >= 0.5:
                best = trial
        out.append(best)
    return np.vstack(out)


def dice_counterfactuals(wrap, train_df, query_df, vary, n_cf=1, seed=SEED):
    """DiCE (Mothilal et al., 2020), model-agnostic random-sampling method via the dice-ml library.
    Returns one counterfactual row per query (NaN row when none was found). If dice-ml cannot run in
    this environment, the identical random-sampling algorithm above is used and reported."""
    try:
        import dice_ml
        data = dice_ml.Data(dataframe=train_df[CUR_FEATS + ["target"]], continuous_features=CUR_FEATS, outcome_name="target")
        exp = dice_ml.Dice(data, dice_ml.Model(model=wrap, backend="sklearn"), method="random")
        out, errors = [], 0
        for i in range(len(query_df)):
            try:
                r = exp.generate_counterfactuals(query_df.iloc[[i]][CUR_FEATS], total_CFs=n_cf, desired_class="opposite",
                                                 features_to_vary=vary, random_seed=seed, verbose=False,
                                                 posthoc_sparsity_algorithm="binary")
                cf = r.cf_examples_list[0].final_cfs_df
                out.append(cf[CUR_FEATS].iloc[0].to_numpy(float) if cf is not None and len(cf) else np.full(len(CUR_FEATS), np.nan))
            except Exception as e:
                errors += 1
                last = e
                out.append(np.full(len(CUR_FEATS), np.nan))
        if len(query_df) and errors == len(query_df):
            raise RuntimeError(f"dice-ml failed on every query: {type(last).__name__}: {last}")
        DICE_BACKEND["name"] = "dice-ml"
        return np.vstack(out) if out else np.zeros((0, len(CUR_FEATS)))
    except Exception as e:
        if DICE_BACKEND["name"] != "fallback":
            print(f"  dice-ml unavailable ({type(e).__name__}: {str(e)[:120]}) -> DiCE random-sampling re-implementation")
        DICE_BACKEND["name"] = "fallback"
        return _random_cf(wrap, train_df, query_df, vary, seed)


def recourse_metrics(wrap, train_df, query_df, vary):
    cfs = dice_counterfactuals(wrap, train_df, query_df, vary)
    Q = query_df[CUR_FEATS].to_numpy(float)
    ok = ~np.isnan(cfs).any(1)
    if ok.any():
        ok[ok] = wrap.predict_proba(pd.DataFrame(cfs[ok], columns=CUR_FEATS))[:, 1] >= 0.5
    mad = np.median(np.abs(train_df[CUR_FEATS].to_numpy(float) - np.median(train_df[CUR_FEATS].to_numpy(float), 0)), 0)
    mad[mad < 1e-6] = train_df[CUR_FEATS].to_numpy(float).std(0)[mad < 1e-6] + 1e-6
    diff = np.abs(cfs[ok] - Q[ok])
    return {"Counterfactual validity": float(ok.mean()),
            "Proximity": float((diff / mad).mean()) if ok.any() else np.nan,
            "Sparsity": float((diff > 1e-6).sum(1).mean()) if ok.any() else np.nan}


def demog_eval(name, d_tr, d_te, query_idx, d_va, seed=SEED):
    Xtr, ytr, gtr = d_tr[CUR_FEATS].to_numpy(float), d_tr["target"].to_numpy(int), inter_groups(d_tr)
    Xte, gte = d_te[CUR_FEATS].to_numpy(float), inter_groups(d_te)
    Xva, gva, yva = d_va[CUR_FEATS].to_numpy(float), inter_groups(d_va), d_va["target"].to_numpy(int)
    tf = None
    w = d_tr["w"].to_numpy(float)
    real = d_tr["meta_synthetic"].to_numpy() == 0 if "meta_synthetic" in d_tr else np.ones(len(d_tr), bool)
    if name == "Reweighting":
        Xtr, ytr, gtr, w2, _ = dm_reweight(Xtr, ytr, gtr)
        w = w * w2
    elif name == "FairSMOTE":
        Xtr, ytr, gtr, _, src = dm_fairsmote(Xtr, ytr, gtr, seed, feats=CUR_FEATS)
        real = np.where(src >= 0, real[np.maximum(src, 0)], False)
        w = None
    elif name == "Optimal Transport":
        ot_ = OTRepair().fit(Xtr, gtr)
        Xtr, Xte, Xva = ot_.transform(Xtr, gtr), ot_.transform(Xte, gte), ot_.transform(Xva, gva)
    elif name in ("Adversarial debiasing", "Fair contrastive learning"):
        st = Standardizer().fit(Xtr)
        rep = (AdvDebias() if name == "Adversarial debiasing" else FairContrastive()).fit(st.transform(Xtr), ytr, gtr)
        tf = lambda X, rep=rep, st=st: rep.transform(st.transform(X))
    elif name == "Counterfactual recourse (DiCE)":
        p0, thr0, m0 = fit_score(Xtr, ytr, Xtr, w=w, seed=seed, real=real)
        tpr = {k: ((p0 >= thr0)[(gtr == k) & (ytr == 1)]).mean() for k in np.unique(gtr) if ((gtr == k) & (ytr == 1)).sum() >= 20}
        worst = min(tpr, key=tpr.get)
        fn = np.where((gtr == worst) & (ytr == 1) & (p0 < thr0))[0][:E(150)]
        cfs = dice_counterfactuals(ModelWrap(m0), d_tr, d_tr.iloc[fn], VARY)
        ok = ~np.isnan(cfs).any(1)
        Xtr = np.vstack([Xtr, cfs[ok]])
        ytr = np.r_[ytr, np.ones(ok.sum(), int)]
        gtr = np.r_[gtr, np.full(ok.sum(), worst, dtype=object)]
        w = np.r_[w, np.ones(ok.sum())]
        real = np.r_[real, np.zeros(ok.sum(), bool)]
        print(f"  DiCE augmentation: {ok.sum()} counterfactuals added for group {worst}")
    Xfit = tf(Xtr) if tf else Xtr
    Xev = tf(Xte) if tf else Xte
    Xvv = tf(Xva) if tf else Xva
    p, thr, m = fit_score(Xfit, ytr, Xev, w=w, seed=seed, val=(Xvv, yva))
    yhat = (p >= thr).astype(int)
    yte = d_te["target"].to_numpy(int)
    gaps = {a: fairness_gaps(yte, yhat, np.where(d_te[a].isna(), None, d_te[a].astype(object))) for a in PROTECTED}
    out = {"SPD": max(v["SPD"] for v in gaps.values()), "DID": max(v["DID"] for v in gaps.values()),
           "EOD": max(v["EOD"] for v in gaps.values()), "Equalized-odds difference": max(v["EqOdds"] for v in gaps.values())}
    if name == "Optimal Transport":
        q = d_te.iloc[query_idx]
        wrap = ModelWrap(m, lambda X, o=ot_, gq=gte[query_idx[0]]: o.transform(X, np.full(len(X), gq)))
    else:
        wrap = ModelWrap(m, tf)
    q = d_te.iloc[query_idx]
    out.update(recourse_metrics(wrap, d_tr, q, VARY))
    out["AUROC (accuracy guard)"] = safe_auc(yte, p)
    return out, (p, thr)


DEM_METHODS = ["Optimal Transport", "FairSMOTE", "Reweighting", "Adversarial debiasing", "Counterfactual recourse (DiCE)",
               "Fair contrastive learning"]
DEM_FAMILY = {k: ("Traditional" if k in ("Optimal Transport", "FairSMOTE", "Reweighting") else "Modern") for k in DEM_METHODS}
DEM_DIRS = {"SPD": "min", "DID": "min", "EOD": "min", "Equalized-odds difference": "min", "Counterfactual validity": "max",
            "Proximity": "min", "Sparsity": "min", "AUROC (accuracy guard)": "max"}


def demog_grid(test_df, yhat_b, yhat_a, method):
    """Rows = protected (and audited) groups; cells = example test patients who ARE cases (outcome = 1), the same
    patients for the model trained on Before data and on After data. Coloured = missed by the model (false negative)."""
    y = test_df["target"].to_numpy(int)
    fn_b = ((y == 1) & (yhat_b == 0)).astype(int)
    fn_a = ((y == 1) & (yhat_a == 0)).astype(int)
    groups = []
    for a_ in PROTECTED + AUDIT_ONLY:
        for k in sorted(test_df[a_].dropna().astype(str).unique()):
            m = (test_df[a_].astype(str) == k).to_numpy() & (y == 1)
            if m.sum() >= 20:
                groups.append((f"{a_.replace('sens_', '').replace('_', ' ')} = {k}", m))
    st = [{"v": 1, "label": "Missed case (false negative)", "color": "#2563C8", "border": "#1D4FA0"}]
    return group_grid("Demographic Bias Heatmap", groups, fn_b, fn_a, st, "missed",
                      "Rows = protected groups (sex, race/ethnicity) and the audited age groups; each cell = one test patient of that "
                      "group who has the condition. Coloured = the classifier missed the case (false negative). Before = classifier "
                      f"trained on the step input; After = the same classifier trained on the output of {method or 'the step'}; "
                      "both are scored on the same untouched test patients. Equal miss rates across rows = equal opportunity.",
                      "Detected case")


def group_profile(train_df, test_all, feats):
    test_df, val_df = part(test_all, "test"), part(test_all, "validation")
    p, thr, _ = fit_score(train_df[feats].to_numpy(float), train_df["target"].to_numpy(int), test_df[feats].to_numpy(float),
                          w=train_df["w"].to_numpy(float), val=(val_df[feats].to_numpy(float), val_df["target"].to_numpy(int)))
    yhat = (p >= thr).astype(int)
    y = test_df["target"].to_numpy(int)
    rows = {}
    for a in PROTECTED + AUDIT_ONLY:
        for k in sorted(test_df[a].dropna().unique()):
            m = (test_df[a] == k).to_numpy()
            rows[f"{a.replace('sens_', '')}={k}"] = {"share %": 100 * m.mean(), "positive rate %": 100 * y[m].mean(),
                                                     "selection rate %": 100 * yhat[m].mean(),
                                                     "TPR %": 100 * yhat[m & (y == 1)].mean() if (m & (y == 1)).sum() else np.nan,
                                                     "FPR %": 100 * yhat[m & (y == 0)].mean() if (m & (y == 0)).sum() else np.nan}
    gaps = {a: fairness_gaps(y, yhat, np.where(test_df[a].isna(), None, test_df[a].astype(object))) for a in PROTECTED}
    worst = {k: max(v[k] for v in gaps.values()) for k in ["SPD", "DID", "EOD", "EqOdds"]}
    return pd.DataFrame(rows).T, worst, safe_auc(y, p), yhat

# CELL 45
CUR_FEATS = list(FEATS)
VARY = [c for c in CUR_FEATS if c not in PROT_FEATS and c not in NON_ACTIONABLE]
DEM = detect_demographic(TR, FEATS)
print(DEM["table"].round(3).to_string())
dem_before = TR.copy()
prof_b, worst_b, auc_b, yhat_b = group_profile(dem_before, TE, CUR_FEATS)      # Before: model trained on the step input
dem_tab, dem_best, dem_trad = None, None, None
if DEM["detected"] or FORCE_DETECT:
    bench = strat_sample(TR, S("demographic"))
    _, p0 = None, None
    TEt, TEv = part(TE, "test"), part(TE, "validation")
    base_p, base_thr, base_m = fit_score(bench[CUR_FEATS].to_numpy(float), bench.target.to_numpy(int), TEt[CUR_FEATS].to_numpy(float),
                                         w=bench["w"].to_numpy(float), val=(TEv[CUR_FEATS].to_numpy(float), TEv.target.to_numpy(int)))
    gte_all = inter_groups(TEt)
    yte_all = TEt.target.to_numpy(int)
    tpr_g = {k: ((base_p >= base_thr)[(gte_all == k) & (yte_all == 1)]).mean() for k in np.unique(gte_all)
             if ((gte_all == k) & (yte_all == 1)).sum() >= 20}
    worst_g = min(tpr_g, key=tpr_g.get)
    cand = np.where((gte_all == worst_g) & (base_p < base_thr))[0]
    query_idx = np.random.default_rng(SEED).choice(cand, min(E(25), len(cand)), replace=False)
    print(f"Disadvantaged group for recourse evaluation: {worst_g} (baseline TPR {tpr_g[worst_g]:.3f}); {len(query_idx)} DiCE queries per method")
    res = {}
    with Timer("demographic:baseline"):
        res["No mitigation (reference)"], _ = demog_eval("none", bench, TEt, query_idx, TEv)
    for name in DEM_METHODS:
        with Timer(f"demographic:{name}"):
            try:
                res[name], _ = demog_eval(name, bench, TEt, query_idx, TEv)
            except Exception as e:
                print(f"  {name} failed: {type(e).__name__}: {e}"); progress(f"FAILED {name}: {type(e).__name__}: {e}")
                res[name] = {k: np.nan for k in DEM_DIRS}
    dem_tab, dem_best, dem_trad = rank_methods(pd.DataFrame(res).T, DEM_DIRS, DEM_FAMILY, reference_rows=["No mitigation (reference)"])
    show_comparison(dem_tab, DEM_DIRS, dem_best, DEM_FAMILY, dem_trad)
    with Timer(f"demographic:apply {dem_best}"):
        X, y, g = TR[CUR_FEATS].to_numpy(float), TR.target.to_numpy(int), inter_groups(TR)
        other = [c for c in TR.columns if c not in CUR_FEATS]
        if dem_best == "Reweighting":
            TR["w"] = TR["w"].to_numpy() * dm_reweight(X, y, g)[3]
        elif dem_best == "Optimal Transport":
            ot_ = OTRepair().fit(X, g)
            TR[CUR_FEATS] = snap(pd.DataFrame(ot_.transform(X, g), columns=CUR_FEATS, index=TR.index), CUR_FEATS).to_numpy()
            TE[CUR_FEATS] = snap(pd.DataFrame(ot_.transform(TE[CUR_FEATS].to_numpy(float), inter_groups(TE)), columns=CUR_FEATS, index=TE.index), CUR_FEATS).to_numpy()
        elif dem_best == "FairSMOTE":
            X2, y2, g2, _, src = dm_fairsmote(X, y, g, feats=CUR_FEATS)
            base_rows = TR.reset_index(drop=True)
            kept = base_rows.iloc[src[src >= 0]]
            is_new = src < 0
            parts = [kept]
            if is_new.any():
                new = X2[is_new]
                nb = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(new, return_distance=False)[:, 0]
                syn = pd.concat([base_rows.iloc[nb][other].reset_index(drop=True), pd.DataFrame(new, columns=CUR_FEATS)], axis=1)
                syn["target"] = y2[is_new]
                for a, v in zip(PROTECTED, zip(*[s.split("|") for s in g2[is_new]])):
                    syn[a] = [None if x == "Unknown" else x for x in v]
                syn["meta_synthetic"] = 1
                parts.append(syn[base_rows.columns])
            TR = pd.concat(parts, ignore_index=True)
        elif dem_best in ("Adversarial debiasing", "Fair contrastive learning"):
            if "FEATURE_SPACE" not in globals():
                FEATURE_SPACE = (TR.copy(), TE.copy(), list(CUR_FEATS), "after steps 1-9, before the demographic step")
            st = Standardizer().fit(X)
            rep = (AdvDebias() if dem_best == "Adversarial debiasing" else FairContrastive()).fit(st.transform(X), y, g)
            new_cols = [f"f{i + 1}" for i in range(32)]
            keep = [c for c in TR.columns if c not in CUR_FEATS]
            TR = pd.concat([TR[keep].reset_index(drop=True), pd.DataFrame(rep.transform(st.transform(X)), columns=new_cols)], axis=1)
            TE = pd.concat([TE[keep].reset_index(drop=True), pd.DataFrame(rep.transform(st.transform(TE[CUR_FEATS].to_numpy(float))), columns=new_cols)], axis=1)
            FEATS = new_cols
            for c in new_cols:
                SCHEMA[c] = {"type": "cont"}
        elif dem_best == "Counterfactual recourse (DiCE)":
            p0, thr0, m0 = fit_score(X, y, X, w=TR["w"].to_numpy(float), real=TR["meta_synthetic"].to_numpy() == 0)
            tpr = {k: ((p0 >= thr0)[(g == k) & (y == 1)]).mean() for k in np.unique(g) if ((g == k) & (y == 1)).sum() >= 20}
            worst = min(tpr, key=tpr.get)
            fn = np.where((g == worst) & (y == 1) & (p0 < thr0))[0][:E(300)]
            cfs = dice_counterfactuals(ModelWrap(m0), TR, TR.iloc[fn], VARY)
            ok = ~np.isnan(cfs).any(1)
            syn = TR.iloc[fn[ok]].copy()
            syn[CUR_FEATS] = cfs[ok]
            syn["meta_synthetic"] = 1
            TR = pd.concat([TR, syn], ignore_index=True)
else:
    print("Demographic bias NOT detected -> unchanged.")
prof_a, worst_a, auc_a, yhat_a = group_profile(TR, TE, FEATS)                 # After: model trained on the step output
RUN_LOG["Demographic"] = {"detected": DEM["detected"], "statistic": DEM["statistic"], "before": DEM["value"],
                          "after": f"SPD {worst_a['SPD']:.3f}; EOD {worst_a['EOD']:.3f}", "selected": dem_best or "—", "best_traditional": dem_trad or "—",
                          "methods_compared": len(DEM_METHODS) if DEM["detected"] else 0}
print(f"Worst-gap fairness on test (model trained on Before vs After data): SPD {worst_b['SPD']:.3f} -> {worst_a['SPD']:.3f} | "
      f"EOD {worst_b['EOD']:.3f} -> {worst_a['EOD']:.3f} | AUROC {auc_b:.3f} -> {auc_a:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ks = ["SPD", "DID", "EOD", "EqOdds"]
ax[0].bar(np.arange(4), [worst_b[k] for k in ks], width=0.35, color="#6FA0DA", label="before")
ax[0].bar(np.arange(4) + 0.35, [worst_a[k] for k in ks], width=0.35, color="#7BC17E", label="after")
ax[0].set_xticks(np.arange(4) + 0.17, ks); ax[0].legend(); ax[0].set_title("Worst gap across protected attributes (test)")
if dem_tab is not None:
    mean_rank_plot(ax[1], dem_tab, dem_best, "Demographic-method comparison", DEM_FAMILY)
else:
    ax[1].axis("off")
plt.tight_layout(); plt.show()

write_dashboard(
    "05_demographic_bias_dashboard.html", "Demographic Bias", DEM["detected"],
    (f"Detected: {DEM['statistic']} = {DEM['value']} — {dem_best} selected (best-ranked modern method; best traditional for comparison: {dem_trad}) and applied"
     if DEM["detected"] else f"Not detected: {DEM['statistic']} = {DEM['value']} — unchanged, After = Before"),
    {"Statistic": DEM["statistic"], "Value (before)": DEM["value"], "Threshold": DEM["threshold"],
     "Protected attributes": ", ".join(a.replace("sens_", "") for a in PROTECTED),
     "Worst SPD before → after": f"{worst_b['SPD']:.3f} → {worst_a['SPD']:.3f}",
     "Worst EOD before → after": f"{worst_b['EOD']:.3f} → {worst_a['EOD']:.3f}",
     "Test AUROC before → after": f"{auc_b:.3f} → {auc_a:.3f}"},
    dem_best, dem_before, TR, display_vars(CUR_FEATS) + [c for c in display_vars(FEATS) if c not in CUR_FEATS],
    demog_grid(part(TE, "test"), yhat_b, yhat_a, dem_best),
    comparison=comparison_payload(dem_tab, DEM_DIRS, dem_best, DEM_FAMILY, dem_trad),
    footer="Mitigation groups = intersection of " + " × ".join(a.replace("sens_", "") for a in PROTECTED) +
           ". Age group is audited (heatmap rows) but not equalised. Fairness is always measured on the untouched test split."
           + (" The selected method replaced the predictors with a learned representation (f1, f2, ...), shown as After-only variables."
              if FEATS != CUR_FEATS else ""),
    chips=[f"Worst SPD: <b>{worst_b['SPD']:.3f} → {worst_a['SPD']:.3f}</b>", f"Worst EOD: <b>{worst_b['EOD']:.3f} → {worst_a['EOD']:.3f}</b>",
           f"Test AUROC: <b>{auc_b:.3f} → {auc_a:.3f}</b>"])

# CELL 47
summary = pd.DataFrame([{"bias": k, "detected": "YES" if v["detected"] else "no", "statistic": v["statistic"],
                         "before": (f"{v['before']:.4f}" if isinstance(v["before"], float) else v["before"]),
                         "after": (f"{v['after']:.4f}" if isinstance(v["after"], float) else v["after"]),
                         "methods compared": v["methods_compared"], "selected method (modern)": v["selected"],
                         "best traditional (comparison only)": v.get("best_traditional", "—")}
                        for k, v in RUN_LOG.items()]).set_index("bias")
summary

# CELL 48
def end_to_end(train_df, test_all, feats):
    test_df, val_df = part(test_all, "test"), part(test_all, "validation")
    w = train_df["w"].to_numpy(float) if "w" in train_df else None
    p, thr, _ = fit_score(train_df[feats].to_numpy(float), train_df["target"].to_numpy(int), test_df[feats].to_numpy(float), w=w,
                          val=(val_df[feats].to_numpy(float), val_df["target"].to_numpy(int)))
    y = test_df["target"].to_numpy(int)
    yhat = (p >= thr).astype(int)
    out = {"AUROC": safe_auc(y, p), "PR-AUC": average_precision_score(y, p),
           "Balanced accuracy": balanced_accuracy_score(y, yhat), "Brier": brier_score_loss(y, p)}
    for a in PROTECTED + AUDIT_ONLY:
        gaps = fairness_gaps(y, yhat, np.where(test_df[a].isna(), None, test_df[a].astype(object)))
        out[f"SPD {a.replace('sens_', '')}"] = gaps["SPD"]
        out[f"EOD {a.replace('sens_', '')}"] = gaps["EOD"]
    return out


e2e = pd.DataFrame({"raw data": end_to_end(TR_RAW, TE_RAW, FEATS_RAW_ENC), "pre-processed data": end_to_end(TR, TE, FEATS)})
e2e["change"] = e2e["pre-processed data"] - e2e["raw data"]
print("Note: Brier is not comparable when the training class balance changed (resampling shifts predicted probabilities).")
e2e

# CELL 49
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_cols = FEATS + ["target"] + [c for c in TR.columns if c.startswith("sens_")] + [c for c in TR.columns if c.startswith("meta_")]
tr_out = TR[save_cols].copy()
te_out = part(TE, "test")[[c for c in save_cols if c in TE]].copy()
va_out = part(TE, "validation")[[c for c in save_cols if c in TE]].copy()
va_out["meta_sample_weight"] = 1.0
tr_out["meta_sample_weight"] = TR["w"].to_numpy()
te_out["meta_sample_weight"] = 1.0
tr_out.to_csv(OUT_DIR / f"{DATASET}_train_preprocessed.csv", index=False)
te_out.to_csv(OUT_DIR / f"{DATASET}_test_preprocessed.csv", index=False)
va_out.to_csv(OUT_DIR / f"{DATASET}_validation_preprocessed.csv", index=False)
if "FEATURE_SPACE" in globals():
    # A selected modern method replaced the predictors with a learned latent representation; keep an
    # interpretable copy of the data in the original predictor space, from just before that step.
    fs_tr, fs_te, fs_feats, fs_note = FEATURE_SPACE
    keep_cols = fs_feats + ["target"] + [c for c in fs_tr.columns if c.startswith(("sens_", "meta_"))]
    a_ = fs_tr[keep_cols].copy(); a_["meta_sample_weight"] = fs_tr["w"].to_numpy()
    b_ = fs_te[[c for c in keep_cols if c in fs_te]].copy(); b_["meta_sample_weight"] = 1.0
    a_.to_csv(OUT_DIR / f"{DATASET}_train_feature_space.csv", index=False)
    part(b_, "test").to_csv(OUT_DIR / f"{DATASET}_test_feature_space.csv", index=False)
    part(b_, "validation").to_csv(OUT_DIR / f"{DATASET}_validation_feature_space.csv", index=False)
    print(f"Feature-space copy saved ({fs_note}).")
else:                                        # predictors stayed in the original space: remove copies from earlier runs
    for split_name in ("train", "test", "validation"):
        (OUT_DIR / f"{DATASET}_{split_name}_feature_space.csv").unlink(missing_ok=True)
for key, tab in [("missing_data", miss_tab), ("class_imbalance", imb_tab), ("outlier", out_tab), ("feature_scale", sc_tab),
                 ("demographic", dem_tab), ("duplicate", dup_tab), ("data_noise", nz_tab), ("encoding_domain", dom_tab)]:
    if tab is not None:
        tab.to_csv(OUT_DIR / f"comparison_{key}.csv")
    else:                                    # bias not detected: remove any table left by an earlier run
        (OUT_DIR / f"comparison_{key}.csv").unlink(missing_ok=True)
summary.to_csv(OUT_DIR / "bias_summary.csv")
e2e.to_csv(OUT_DIR / "end_to_end_check.csv")
(OUT_DIR / "pipeline_log.json").write_text(json.dumps(_clean({
    "dataset": DATASET, "title": TITLE, "source_file": str(DATA_FILE), "seed": SEED, "fast_mode": FAST,
    "train_rows": len(TR), "test_rows": len(part(TE, "test")), "validation_rows": len(part(TE, "validation")), "final_predictors": FEATS,
    "schema": {c: {k: v for k, v in s.items()} for c, s in SCHEMA.items() if c in FEATS},
    "biases": RUN_LOG, "timings_seconds": TIMINGS,
    "column_rule": "inputs = every column except meta_*, sens_* and target; use meta_sample_weight as sample weight"}), indent=2, default=str),
    encoding="utf-8")
print("Saved:", OUT_DIR)

# CELL 50
order = [("01_missing_data_bias_dashboard.html", "Missing data"), ("02_class_imbalance_bias_dashboard.html", "Class imbalance"),
         ("03_outlier_bias_dashboard.html", "Outlier"), ("04_feature_scale_bias_dashboard.html", "Feature scale"),
         ("05_demographic_bias_dashboard.html", "Demographic"), ("06_duplicate_bias_dashboard.html", "Duplicate"),
         ("07_data_noise_bias_dashboard.html", "Data noise"), ("08_encoding_domain_bias_dashboard.html", "Encoding / domain")]
rows_html = "".join(
    f"<tr><td><a href='{f}'>{b}</a></td><td class='{'y' if RUN_LOG[b]['detected'] else 'n'}'>{'Detected' if RUN_LOG[b]['detected'] else 'Not detected'}</td>"
    f"<td>{RUN_LOG[b]['statistic']}</td><td>{summary.loc[b, 'before']}</td><td>{summary.loc[b, 'after']}</td>"
    f"<td>{RUN_LOG[b]['methods_compared']}</td><td><b>{RUN_LOG[b]['selected']}</b></td><td>{RUN_LOG[b].get('best_traditional', '—')}</td></tr>" for f, b in order)
if (PLOT_DIR / "07b_monitor_signal_noise_dashboard.html").exists():
    rows_html += ("<tr><td><a href='07b_monitor_signal_noise_dashboard.html'>Data noise — monitor signals</a></td>"
                  "<td colspan='6'>Bedside-monitor spike detection and signal denoising (ICU vital-sign series); see the dashboard.</td></tr>")
e2e_html = e2e.to_html(float_format=lambda v: f"{v:.4f}", border=0)
(PLOT_DIR / "index.html").write_text(f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{TITLE} — pre-processing dashboards</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#EEF1F7;margin:0;color:#1B2333}}header{{background:linear-gradient(180deg,#0B1E3F,#132a52);color:#fff;padding:18px 28px;text-align:center}}
main{{padding:20px 28px}}table{{border-collapse:collapse;background:#fff;width:100%;font-size:13px;margin-bottom:22px}}td,th{{border:1px solid #E1E5EE;padding:7px 9px;text-align:left}}th{{background:#F3F6FC}}
td.y{{color:#8A1F17;font-weight:700}}td.n{{color:#1F5E23;font-weight:700}}a{{color:#2563C8;font-weight:600}}</style></head><body>
<header><h1 style='margin:0;font-size:22px'>{TITLE} — bias-aware pre-processing</h1><div style='font-size:12.5px;color:#B9C6E6'>Generated {date.today():%d-%b-%Y} · train {len(TR):,} rows · validation {len(part(TE, 'validation')):,} · test {len(part(TE, 'test')):,} rows</div></header>
<main><h3>Bias dashboards (Before vs After)</h3><p style='font-size:13px'>Traditional and modern methods are compared on every metric; only the best-ranked <b>modern</b> method is selected and applied.</p>
<table><tr><th>Bias</th><th>Result</th><th>Detection statistic</th><th>Before</th><th>After</th><th>Methods compared</th><th>Selected (modern)</th><th>Best traditional (comparison only)</th></tr>{rows_html}</table>
<h3>End-to-end check (same classifier, same test patients)</h3>{e2e_html}</main></body></html>""", encoding="utf-8")
print("Dashboard index ->", PLOT_DIR / "index.html")
print("Total runtime (s):", round(sum(TIMINGS.values()), 1))