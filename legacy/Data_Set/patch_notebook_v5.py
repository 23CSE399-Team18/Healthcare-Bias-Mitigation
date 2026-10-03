"""
patch_notebook_v5.py
====================
Patches multi_dataset_integration.ipynb so that:
  1. All preprocessing cells are correct (Stages 2 & 3)
  2. Stage 4 integration uses smart imputation — no empty cells
  3. Stage 5 validation checks for zero NaN
  4. Stage 6 exports imputed full_integrated_dataset.csv
  5. Stage 7 prints the clean final report
"""
import json, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")

def load():
    with open(NB, "r", encoding="utf-8") as f:
        return json.load(f)

def save(nb):
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[SAVED] {NB}")

def patch(nb, idx, src):
    cell = nb["cells"][idx]
    cell["source"] = [src]
    cell["outputs"] = []
    cell["execution_count"] = None
    print(f"  [PATCHED] Cell {idx}")

# ─────────────────────────────────────────────────────────────
nb = load()
print(f"Loaded {len(nb['cells'])} cells\n")

# ══════════════════════════════════════════════════════════════
# CELL 2 — Install packages
# ══════════════════════════════════════════════════════════════
patch(nb, 2, """\
# Stage 1: Install required packages
import subprocess, sys
required = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "openpyxl"]
for pkg in required:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("All packages ready.")
""")

# ══════════════════════════════════════════════════════════════
# CELL 3 — Imports
# ══════════════════════════════════════════════════════════════
patch(nb, 3, """\
import os, warnings, ast, json, gc, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.impute import SimpleImputer
from matplotlib.gridspec import GridSpec
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
print("Libraries loaded.")
""")

# ══════════════════════════════════════════════════════════════
# CELL 4 — Paths
# ══════════════════════════════════════════════════════════════
patch(nb, 4, """\
BASE     = Path(r"d:/Data_Set")
DS_DIR   = BASE / "Data_Set"
ND_DIR   = BASE / "ND"
OUT_DIR  = BASE / "integrated_output"
PP_DIR   = OUT_DIR / "preprocessed"
DOM_DIR  = OUT_DIR / "by_domain"
PLOT_DIR = OUT_DIR / "plots"

for d in [OUT_DIR, PP_DIR, DOM_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"BASE    : {BASE}")
print(f"DS_DIR  : {DS_DIR}")
print(f"ND_DIR  : {ND_DIR}")
print(f"OUT_DIR : {OUT_DIR}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 5 — Path registry
# ══════════════════════════════════════════════════════════════
patch(nb, 5, r"""
PATHS = {
    "heart_cleveland"   : DS_DIR / "UCI_HeartDisease" / "processed.cleveland.csv",
    "heart_hungarian"   : DS_DIR / "UCI_HeartDisease" / "processed.hungarian.csv",
    "heart_switzerland" : DS_DIR / "UCI_HeartDisease" / "processed.switzerland.csv",
    "heart_va"          : DS_DIR / "UCI_HeartDisease" / "processed.va.csv",
    "ckd"               : DS_DIR / "Chronic_Kidney_Disease" / "Chronic_Kidney_Disease" / "chronic_kidney_disease_full.csv",
    "eicu_patient"      : DS_DIR / "eLCU" / "patient.csv",
    "eicu_diagnosis"    : DS_DIR / "eLCU" / "diagnosis.csv",
    "eicu_lab"          : DS_DIR / "eLCU" / "lab.csv",
    "nhanes"            : DS_DIR / "National Center for health Statistics" / "NHANES_combined.csv",
    "cdc_diabetes"      : DS_DIR / "CDC diabetes" / "diabetes_binary_health_indicators_BRFSS2015.csv",
    "breast_cancer"     : DS_DIR / "breast+cancer+coimbra_UCL" / "dataR2.csv",
    "mimic_patients"    : ND_DIR / "mimic-iv-3.1" / "hosp" / "patients.csv.gz",
    "mimic_admissions"  : ND_DIR / "mimic-iv-3.1" / "hosp" / "admissions.csv.gz",
    "mimic_icustays"    : ND_DIR / "mimic-iv-3.1" / "icu"  / "icustays.csv.gz",
    "eicu2_patient"     : ND_DIR / "eicu-collaborative-research-database-2.0" / "patient.csv.gz",
    "eicu2_diagnosis"   : ND_DIR / "eicu-collaborative-research-database-2.0" / "diagnosis.csv.gz",
    "eicu2_lab"         : ND_DIR / "eicu-collaborative-research-database-2.0" / "lab.csv.gz",
    "ptbxl_db"          : ND_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptbxl_database.csv",
    "ptbxl_scp"         : ND_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "scp_statements.csv",
    "ham10000"          : ND_DIR / "dataverse_files" / "HAM10000_metadata",
    "hf_ehr"            : ND_DIR / "hospitalized-patients-with-heart-failure-integrating-electronic-healthcare-records-and-external-outcome-data-1.2" / "hospitalized-patients-with-heart-failure-integrating-electronic-healthcare-records-and-external-outcome-data-1.2" / "dat.csv",
}

print("Dataset Path Verification:")
print("=" * 65)
for name, path in PATHS.items():
    exists = path.exists()
    size   = f"{path.stat().st_size/1e6:.1f} MB" if exists else "---"
    print(f"  [{'OK' if exists else 'MISSING':7s}] {name:25s}  {size:>10s}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 35 — Unified schema + harmonize function
# ══════════════════════════════════════════════════════════════
patch(nb, 35, """\
COMMON_SCHEMA = ["dataset_source", "domain", "target", "age", "age_group", "gender"]

def _std_gender(series, mapping=None):
    s = series.astype(str).str.strip().str.lower()
    if mapping:
        lmap = {str(k).lower(): v for k, v in mapping.items()}
        out  = s.map(lmap)
        mask = out.isna()
        out[mask] = s[mask].map(
            lambda x: "Male"   if x in ("m","male","man","1","1.0") else
                      "Female" if x in ("f","female","woman","0","0.0","2","2.0") else
                      "Unknown"
        )
        return out
    return s.map(
        lambda x: "Male"   if x in ("m","male","man","1","1.0") else
                  "Female" if x in ("f","female","woman","0","0.0","2","2.0") else
                  "Unknown"
    )

def _bin_age(s):
    return (pd.cut(s, bins=[0,40,60,75,120],
                   labels=["young","middle","senior","elderly"],
                   include_lowest=True)
            .astype(str).replace("nan","Unknown"))

# Registry: how each preprocessed CSV maps to the unified schema
REGISTRY = {
    "heart_disease_preprocessed.csv" : dict(source="UCI_HeartDisease",    domain="cardiovascular",
        target_col="target",   age_col="age",       gender_col="sex",
        gender_map={"1":"Male","1.0":"Male","0":"Female","0.0":"Female"}),
    "ckd_preprocessed.csv"           : dict(source="CKD_UCI",             domain="renal",
        target_col="class", target_map={"ckd":1,"notckd":0,"1":1,"0":0,"1.0":1,"0.0":0},
        age_col="age",       gender_fixed="Unknown"),
    "eicu_original_preprocessed.csv" : dict(source="eICU_original",       domain="icu",
        target_col="target",   age_col="age",       gender_col="gender"),
    "nhanes_preprocessed.csv"        : dict(source="NHANES",              domain="population_health",
        target_col="target",   age_col="Age_Years", gender_col="Gender",
        gender_map={"1":"Male","1.0":"Male","2":"Female","2.0":"Female"}),
    "cdc_diabetes_preprocessed.csv"  : dict(source="CDC_Diabetes",        domain="metabolic",
        target_col="target",   age_col="Age",       gender_col="Sex",
        gender_map={"1":"Male","1.0":"Male","0":"Female","0.0":"Female"}),
    "breast_cancer_preprocessed.csv" : dict(source="BreastCancer_Coimbra",domain="oncology",
        target_col="Classification", target_map={"1":0,"1.0":0,"2":1,"2.0":1},
        age_col="Age",       gender_fixed="Female"),
    "mimic_iv_preprocessed.csv"      : dict(source="MIMIC_IV",            domain="icu",
        target_col="hospital_expire_flag", age_col="age", gender_col="gender"),
    "eicu_crd_v2_preprocessed.csv"   : dict(source="eICU_CRD_v2",         domain="icu",
        target_col="target",   age_col="age",       gender_col="gender"),
    "ptbxl_preprocessed.csv"         : dict(source="PTB_XL_ECG",          domain="ecg",
        target_col="target",   age_col="age",       gender_col="sex",
        gender_map={"0":"Male","0.0":"Male","1":"Female","1.0":"Female"}),
    "ham10000_preprocessed.csv"      : dict(source="HAM10000",            domain="dermatology",
        target_col="target",   age_col="age",       gender_col="sex"),
    "heart_failure_ehr_preprocessed.csv": dict(source="HeartFailure_EHR", domain="cardiovascular",
        target_col="death.within.28.days", age_col=None, gender_col="gender"),
}

def harmonize_to_schema(df, cfg):
    out = df.copy()
    out["dataset_source"] = cfg["source"]
    out["domain"]         = cfg["domain"]

    # target
    t = cfg.get("target_col")
    if t and t in out.columns:
        tm = cfg.get("target_map")
        if tm:
            out["target"] = out[t].astype(str).str.strip().str.lower().map(
                {str(k).lower():v for k,v in tm.items()})
        else:
            out["target"] = pd.to_numeric(out[t], errors="coerce")
        if t != "target":
            out.drop(columns=[t], inplace=True, errors="ignore")
    elif "target" not in out.columns:
        out["target"] = np.nan
    valid = out["target"].notna()
    out.loc[valid & (out["target"] > 1), "target"] = 1
    out["target"] = pd.to_numeric(out["target"], errors="coerce")

    # age
    a = cfg.get("age_col")
    if a and a in out.columns:
        out["age"] = pd.to_numeric(
            out[a].astype(str).str.replace("> 89","90",regex=False), errors="coerce")
        if a != "age":
            out.drop(columns=[a], inplace=True, errors="ignore")
    elif "age" not in out.columns:
        out["age"] = np.nan

    # age_group
    if "age_group" not in out.columns:
        out["age_group"] = _bin_age(out["age"])
    else:
        out["age_group"] = out["age_group"].astype(str).replace("nan","Unknown")

    # gender
    gf = cfg.get("gender_fixed")
    if gf:
        out["gender"] = gf
    else:
        g = cfg.get("gender_col")
        if g and g in out.columns:
            out["gender"] = _std_gender(out[g], cfg.get("gender_map"))
            if g != "gender":
                out.drop(columns=[g], inplace=True, errors="ignore")
        elif "gender" not in out.columns:
            out["gender"] = "Unknown"

    front = [c for c in COMMON_SCHEMA if c in out.columns]
    other = [c for c in out.columns if c not in front]
    print(f"  [{cfg['source']:<30}] -> {out.shape} | "
          f"target={out['target'].mean():.3f} | "
          f"age_ok={out['age'].notna().mean()*100:.0f}%")
    return out[front + other]

print("Schema helpers & registry defined.")
""")

# ══════════════════════════════════════════════════════════════
# CELL 36 — Harmonize all datasets
# ══════════════════════════════════════════════════════════════
patch(nb, 36, """\
print("Harmonizing all datasets...")
print("=" * 72)

harmonized = {}
for fname, cfg in REGISTRY.items():
    path = PP_DIR / fname
    if not path.exists():
        print(f"  [SKIP] {cfg['source']} — preprocessed file not found")
        continue
    df = pd.read_csv(path, low_memory=False)
    harmonized[cfg["source"]] = harmonize_to_schema(df, cfg)

print(f"\\n{len(harmonized)}/{len(REGISTRY)} datasets harmonized successfully")
if len(harmonized) < len(REGISTRY):
    missing = [cfg["source"] for f,cfg in REGISTRY.items() if cfg["source"] not in harmonized]
    print(f"WARNING - missing: {missing}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 38 — Domain grouping
# ══════════════════════════════════════════════════════════════
patch(nb, 38, """\
domain_groups = {}
for name, df in harmonized.items():
    dom = df["domain"].iat[0]
    domain_groups.setdefault(dom, []).append((name, df))

print("Domain Groups:")
print("=" * 60)
for dom, items in sorted(domain_groups.items()):
    total  = sum(len(d) for _,d in items)
    labels = [f"{n}({len(d):,})" for n,d in items]
    print(f"  {dom:<25}: {total:>8,} records  <-  {', '.join(labels)}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 39 — Domain integration helper
# ══════════════════════════════════════════════════════════════
patch(nb, 39, """\
def integrate_domain(datasets, domain_name):
    frames = [df for _,df in datasets]
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    front  = [c for c in COMMON_SCHEMA if c in merged.columns]
    other  = [c for c in merged.columns if c not in front]
    merged = merged[front + other]
    print(f"  [{domain_name:<20}] -> {merged.shape} | missing: {merged.isnull().mean().mean()*100:.1f}%")
    return merged

print("Integrating by domain...")
print("=" * 60)
domain_dfs = {}
for dom, items in sorted(domain_groups.items()):
    domain_dfs[dom] = integrate_domain(items, dom)
print(f"\\n{len(domain_dfs)} domain DataFrames created")
""")

# ══════════════════════════════════════════════════════════════
# CELL 41 — Full integration + SMART IMPUTATION
# ══════════════════════════════════════════════════════════════
patch(nb, 41, """\
# ── Concatenate all datasets ──────────────────────────────────────────────────
print("Concatenating all datasets...")
all_frames      = list(harmonized.values())
full_integrated = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)

front = [c for c in COMMON_SCHEMA if c in full_integrated.columns]
other = [c for c in full_integrated.columns if c not in front]
full_integrated = full_integrated[front + other]

print(f"Before imputation: {len(full_integrated):,} rows × {full_integrated.shape[1]} cols")
print(f"Overall NaN %    : {full_integrated.isnull().mean().mean()*100:.1f}%")

# ── Smart Imputation — fill every empty cell ──────────────────────────────────
print("\\nApplying smart imputation to fill all empty cells...")
print("-" * 60)

nan_before = full_integrated.isnull().sum().sum()
print(f"  NaN cells before: {nan_before:,}")

# Schema text columns
for col in ["dataset_source", "domain"]:
    if col in full_integrated.columns:
        full_integrated[col] = full_integrated[col].fillna("Unknown")
for col in ["gender", "age_group"]:
    if col in full_integrated.columns:
        full_integrated[col] = full_integrated[col].replace({"nan":"Unknown","":" Unknown"}).fillna("Unknown")

# Numeric columns
num_cols  = full_integrated.select_dtypes(include=[np.number]).columns.tolist()
text_cols = [c for c in full_integrated.select_dtypes(exclude=[np.number]).columns
             if c not in COMMON_SCHEMA]

filled_total = 0
for col in num_cols:
    n_null = full_integrated[col].isna().sum()
    if n_null == 0:
        continue
    non_null = full_integrated[col].dropna()
    if len(non_null) == 0:
        fill_val = 0
    else:
        unique_vals = set(non_null.unique())
        is_binary   = unique_vals.issubset({0, 1, 0.0, 1.0})
        fill_val    = 0 if is_binary else float(non_null.median())
    full_integrated[col] = full_integrated[col].fillna(fill_val)
    filled_total += n_null

for col in text_cols:
    n_null = full_integrated[col].isna().sum()
    if n_null == 0:
        continue
    vals = full_integrated[col].dropna()
    fill_val = vals.mode().iloc[0] if len(vals) > 0 else "Unknown"
    full_integrated[col] = full_integrated[col].fillna(fill_val)
    filled_total += n_null

# Re-derive age_group for HeartFailure rows where age is still NaN
mask_unk = full_integrated["age_group"] == "Unknown"
if mask_unk.any():
    full_integrated.loc[mask_unk, "age_group"] = _bin_age(
        full_integrated.loc[mask_unk, "age"]).values
    full_integrated["age_group"] = full_integrated["age_group"].replace("nan","Unknown")

nan_after = full_integrated.isnull().sum().sum()
print(f"  NaN cells after : {nan_after:,}  (filled {filled_total:,} cells)")
print(f"\\nAfter imputation: {len(full_integrated):,} rows × {full_integrated.shape[1]} cols")
print(f"Overall NaN %   : {full_integrated.isnull().mean().mean()*100:.2f}%  (target: 0.00%)")

print("\\n" + "="*70)
print("FULL INTEGRATED DATASET — SUMMARY")
print("="*70)
print(f"  Total records  : {len(full_integrated):>12,}")
print(f"  Total features : {full_integrated.shape[1]:>12,}")
print()
print("  Records per dataset source:")
for ds, cnt in full_integrated["dataset_source"].value_counts().items():
    print(f"    {ds:<32} {cnt:>10,}")
print()
print("  Records per clinical domain:")
for dom, cnt in full_integrated["domain"].value_counts().items():
    print(f"    {dom:<32} {cnt:>10,}")

# Verify sample looks clean
print("\\nSample (3 rows × 12 cols):")
print(full_integrated.head(3).iloc[:, :12].to_string(index=False))
""")

# ══════════════════════════════════════════════════════════════
# CELL 43 — Quality statistics summary
# ══════════════════════════════════════════════════════════════
patch(nb, 43, """\
summary_rows = []
for name, df in harmonized.items():
    t_rate = df["target"].mean() if df["target"].notna().any() else None
    summary_rows.append({
        "Dataset"     : name,
        "Domain"      : df["domain"].iat[0],
        "Records"     : len(df),
        "Columns"     : df.shape[1],
        "Missing %"   : round(df.isnull().mean().mean()*100, 1),
        "Target Rate" : round(t_rate, 3) if t_rate is not None else "N/A",
        "Age Avail %" : round(df["age"].notna().mean()*100, 1) if "age" in df.columns else 0,
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Records", ascending=False)
print("Dataset Quality Summary (before imputation):")
print(summary_df.to_string(index=False))
""")

# ══════════════════════════════════════════════════════════════
# CELL 44 — Dashboard
# ══════════════════════════════════════════════════════════════
patch(nb, 44, """\
fig = plt.figure(figsize=(22, 18))
gs  = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.3)

# 1. Records per dataset
ax1 = fig.add_subplot(gs[0, 0])
rec    = full_integrated["dataset_source"].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(rec)))
bars   = ax1.barh(range(len(rec)), rec.values, color=colors, edgecolor="white")
ax1.set_yticks(range(len(rec)))
ax1.set_yticklabels([t[:22] for t in rec.index], fontsize=8)
ax1.set_title("Records per Dataset", fontweight="bold")
ax1.set_xlabel("Count")
for bar, val in zip(bars, rec.values):
    ax1.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
             f"{val:,}", va="center", fontsize=7)

# 2. Missing % per dataset (before imputation)
ax2 = fig.add_subplot(gs[0, 1])
miss_by_ds   = {n: df.isnull().mean().mean()*100 for n, df in harmonized.items()}
names_sorted = sorted(miss_by_ds, key=miss_by_ds.get, reverse=True)
vals         = [miss_by_ds[n] for n in names_sorted]
max_miss     = max(vals) + 1e-6 if vals else 1
colors2      = [plt.cm.RdYlGn_r(v / max_miss) for v in vals]
ax2.barh(range(len(names_sorted)), vals, color=colors2, edgecolor="white")
ax2.set_yticks(range(len(names_sorted)))
ax2.set_yticklabels([n[:22] for n in names_sorted], fontsize=8)
ax2.set_title("Missing Value % per Dataset (before imputation)", fontweight="bold")
ax2.set_xlabel("Missing %")

# 3. Records per domain pie
ax3 = fig.add_subplot(gs[1, 0])
dom_counts = full_integrated["domain"].value_counts()
ax3.pie(dom_counts.values, labels=dom_counts.index, autopct="%1.1f%%",
        colors=plt.cm.Pastel1(np.linspace(0, 1, len(dom_counts))), startangle=90)
ax3.set_title("Records by Clinical Domain", fontweight="bold")

# 4. Target rate per dataset
ax4 = fig.add_subplot(gs[1, 1])
trates = {n: df["target"].mean() for n, df in harmonized.items()
          if "target" in df.columns and df["target"].notna().any()}
ax4.barh(list(trates.keys()), list(trates.values()), color="coral", edgecolor="white")
ax4.set_title("Target Rate per Dataset", fontweight="bold")
ax4.set_xlabel("Rate (0-1)")
ax4.set_yticklabels([k[:22] for k in trates.keys()], fontsize=8)

# 5. Gender distribution
ax5 = fig.add_subplot(gs[2, 0])
gc = full_integrated["gender"].value_counts()
ax5.bar(gc.index, gc.values, color=["royalblue","salmon","lightgrey"][:len(gc)])
ax5.set_title("Gender Distribution (All Datasets)", fontweight="bold")
ax5.set_ylabel("Count")
for i, (_, v) in enumerate(gc.items()):
    ax5.text(i, v + gc.max()*0.01, f"{v:,}", ha="center", fontsize=9)

# 6. Age distribution
ax6 = fig.add_subplot(gs[2, 1])
age_data = full_integrated["age"].dropna()
ax6.hist(age_data, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax6.set_title("Age Distribution (All Datasets)", fontweight="bold")
ax6.set_xlabel("Age (years)")
ax6.set_ylabel("Count")

fig.suptitle("Multi-Dataset Integration Dashboard", fontsize=16, fontweight="bold", y=1.01)
plt.savefig(PLOT_DIR / "dataset_overview.png", bbox_inches="tight", dpi=150)
plt.show()
print("Dashboard saved.")
""")

# ══════════════════════════════════════════════════════════════
# CELL 48 — Schema availability heatmap
# ══════════════════════════════════════════════════════════════
patch(nb, 48, """\
schema_cols_present = [c for c in COMMON_SCHEMA if c in full_integrated.columns]

# Use ORIGINAL harmonized data (before imputation) so we see true availability
avail_data = {}
for name, df in harmonized.items():
    avail_data[name] = {
        col: round(df[col].notna().sum() / len(df) * 100, 1)
        for col in schema_cols_present if col in df.columns
    }

avail_df = pd.DataFrame(avail_data).T.fillna(0)
fig, ax  = plt.subplots(figsize=(10, 7))
sns.heatmap(avail_df, ax=ax, cmap="YlGn", annot=True, fmt=".0f",
            linewidths=0.5, cbar_kws={"label":"% Available (non-null — before imputation)"},
            vmin=0, vmax=100)
ax.set_title("Schema Column Original Availability per Dataset (%)", fontsize=12, fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
plt.savefig(PLOT_DIR / "schema_availability.png", bbox_inches="tight")
plt.show()
print("Heatmap saved.")
""")

# ══════════════════════════════════════════════════════════════
# CELL 50 — Save preprocessed CSVs
# ══════════════════════════════════════════════════════════════
patch(nb, 50, """\
print("Saving individual preprocessed datasets...")
pp_dir = OUT_DIR / "preprocessed"
pp_dir.mkdir(exist_ok=True)

save_map = {
    "heart_disease_preprocessed.csv"    : heart_pp   if "heart_pp"   in dir() else None,
    "ckd_preprocessed.csv"              : ckd_pp     if "ckd_pp"     in dir() else None,
    "eicu_original_preprocessed.csv"    : eicu_pp    if "eicu_pp"    in dir() else None,
    "nhanes_preprocessed.csv"           : nhanes_pp  if "nhanes_pp"  in dir() else None,
    "cdc_diabetes_preprocessed.csv"     : cdc_pp     if "cdc_pp"     in dir() else None,
    "breast_cancer_preprocessed.csv"    : bc_pp      if "bc_pp"      in dir() else None,
    "mimic_iv_preprocessed.csv"         : mimic_pp   if "mimic_pp"   in dir() else None,
    "eicu_crd_v2_preprocessed.csv"      : eicu2_pp   if "eicu2_pp"   in dir() else None,
    "ptbxl_preprocessed.csv"            : ptbxl_pp   if "ptbxl_pp"   in dir() else None,
    "ham10000_preprocessed.csv"         : ham_pp     if "ham_pp"     in dir() else None,
    "heart_failure_ehr_preprocessed.csv": hf_pp      if "hf_pp"      in dir() else None,
}
for fname, df in save_map.items():
    if df is not None:
        p = pp_dir / fname
        df.to_csv(p, index=False)
        print(f"  {fname:<45} {df.shape}  {p.stat().st_size/1e3:>8.0f} KB")
    else:
        print(f"  {fname:<45} SKIPPED")
""")

# ══════════════════════════════════════════════════════════════
# CELL 51 — Save domain files
# ══════════════════════════════════════════════════════════════
patch(nb, 51, """\
print("\\nSaving domain-integrated datasets (from imputed full data)...")
domain_dir = OUT_DIR / "by_domain"
domain_dir.mkdir(exist_ok=True)

for dom, items in sorted(domain_groups.items()):
    # Pull rows from imputed full dataset, keep only relevant columns
    member_cols = set(COMMON_SCHEMA)
    for _, orig_df in items:
        member_cols |= set(orig_df.columns)
    dom_df   = full_integrated[full_integrated["domain"] == dom]
    dom_df   = dom_df[[c for c in dom_df.columns if c in member_cols]]
    fname    = f"{dom}_integrated.csv"
    out_path = domain_dir / fname
    dom_df.to_csv(out_path, index=False)
    size     = out_path.stat().st_size / 1e3
    labels   = [f"{n}({len(d):,})" for n, d in items]
    print(f"  {fname:<35} {dom_df.shape}  {size:>8.0f} KB  <-  {', '.join(labels)}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 52 — Save full + core CSV
# ══════════════════════════════════════════════════════════════
patch(nb, 52, """\
print("\\nSaving full integrated dataset (imputed — no empty cells)...")

# Full dataset
full_path = OUT_DIR / "full_integrated_dataset.csv"
full_integrated.to_csv(full_path, index=False)
size_mb   = full_path.stat().st_size / 1e6
print(f"  full_integrated_dataset.csv      -> {full_integrated.shape}  ({size_mb:.1f} MB)")

# Core (6 schema columns only)
core_cols = [c for c in COMMON_SCHEMA if c in full_integrated.columns]
core_path = OUT_DIR / "core_integrated_dataset.csv"
full_integrated[core_cols].to_csv(core_path, index=False)
core_mb   = core_path.stat().st_size / 1e6
print(f"  core_integrated_dataset.csv      -> {full_integrated[core_cols].shape}  ({core_mb:.1f} MB)")

# Summary schema (copy of core)
full_integrated[core_cols].to_csv(OUT_DIR / "integration_summary_schema.csv", index=False)
print(f"  integration_summary_schema.csv   -> {full_integrated[core_cols].shape}")

# Quick sanity check
remaining_nan = full_integrated.isnull().sum().sum()
print(f"\\n  Remaining NaN cells in full file : {remaining_nan}  (should be 0)")
""")

# ══════════════════════════════════════════════════════════════
# CELL 53 — Metadata JSON
# ══════════════════════════════════════════════════════════════
patch(nb, 53, """\
meta = {
    "run_timestamp"     : datetime.now().isoformat(timespec="seconds"),
    "pipeline_version"  : "5.0",
    "total_records"     : int(len(full_integrated)),
    "total_features"    : int(full_integrated.shape[1]),
    "imputation_applied": True,
    "nan_after_imputation": int(full_integrated.isnull().sum().sum()),
    "datasets_integrated": list(harmonized.keys()),
    "domains"           : sorted(domain_groups.keys()),
    "common_schema"     : COMMON_SCHEMA,
    "per_dataset": {
        name: {
            "records"        : int(len(df)),
            "original_columns": int(df.shape[1]),
            "missing_pct_before_imputation":
                round(float(df.isnull().mean().mean() * 100), 2),
            "has_target"     : bool(df["target"].notna().any()),
            "target_rate"    : round(float(df["target"].mean()), 4)
                               if df["target"].notna().any() else None,
            "domain"         : str(df["domain"].iat[0]),
            "age_pct"        : round(float(df["age"].notna().mean()*100), 1)
                               if "age" in df.columns else 0.0,
            "gender_counts"  : df["gender"].value_counts().to_dict(),
        }
        for name, df in harmonized.items()
    },
}
meta_path = OUT_DIR / "integration_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"integration_metadata.json saved -> {meta_path}")
""")

# ══════════════════════════════════════════════════════════════
# CELL 55 — Final report
# ══════════════════════════════════════════════════════════════
patch(nb, 55, """\
remaining = full_integrated.isnull().sum().sum()
print("=" * 78)
print("  MULTI-DATASET HEALTHCARE INTEGRATION — FINAL REPORT  (v5)")
print("=" * 78)
print(f"  Datasets integrated   : {len(harmonized)}")
print(f"  Clinical domains      : {len(domain_groups)}")
print(f"  Total patient records : {len(full_integrated):,}")
print(f"  Total feature columns : {full_integrated.shape[1]}")
print(f"  NaN cells remaining   : {remaining}  (0 = perfectly clean)")
print()
print(f"  {'Dataset':<30} {'Domain':<20} {'Records':>10} {'Cols':>5} {'Target%':>8}")
print("  " + "-" * 76)
for name, df in harmonized.items():
    dom  = df["domain"].iat[0]
    rate = f"{df['target'].mean()*100:.1f}%" if df["target"].notna().any() else "N/A"
    print(f"  {name:<30} {dom:<20} {len(df):>10,} {df.shape[1]:>5} {rate:>8}")
print("  " + "-" * 76)
print(f"  {'TOTAL':<30} {'':<20} {len(full_integrated):>10,} {full_integrated.shape[1]:>5}")
print()
print("  Output files:")
for p in [OUT_DIR / "full_integrated_dataset.csv",
          OUT_DIR / "core_integrated_dataset.csv",
          OUT_DIR / "integration_metadata.json"]:
    if p.exists():
        print(f"    {p.name:<45} {p.stat().st_size/1e6:.1f} MB")
dom_files = list(DOM_DIR.glob("*_integrated.csv"))
print(f"    by_domain/  -> {len(dom_files)} domain files")
print()
print("  PIPELINE COMPLETE — all empty cells filled, no commas without values.")
print("=" * 78)
""")

# ── Clear all outputs ─────────────────────────────────────────
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"]         = []
        cell["execution_count"] = None

save(nb)
print("\n[DONE] Notebook patched — ready for Kernel -> Restart & Run All")
