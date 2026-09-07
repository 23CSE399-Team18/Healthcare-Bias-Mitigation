"""Validate the four model-ready files and run a leakage smoke test.

Checks: expected rows, binary target, no label-source / identifier / survey-design
columns among predictors, harmonised fairness groups, and a quick baseline
(HistGradientBoosting, 80/20 split; patient-grouped for eICU). A full-model AUROC
above 0.97, or any single predictor with AUROC above 0.95, is flagged as possible
leakage. The baseline is a sanity check only, not a project result.

Run: python code/build_raw/validate_raw.py
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]   # D:/complete

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402

EXPECTED = {"heart": 440111, "hypertension": 29471, "diabetes": 7876, "respiratory": 28842}
FORBIDDEN = {  # must never appear as predictors (case-insensitive)
    "heart": ["_michd", "cvdinfr4", "cvdcrhd4", "heartdisease", "seqno", "_state", "_llcpwt", "_ststr", "_psu"],
    "hypertension": ["hypev_a", "hyp12m_a", "hypdif_a", "hypmed_a", "bplast_a", "hypertension", "hhx",
                     "wtfa_a", "pstrat", "ppsu", "region"],
    "diabetes": ["diq010", "diabetes_diagnosed", "lbxgh", "lbxglu", "lbdglusi", "seqn", "wtint2yr", "wtmec2yr",
                 "sdmvstra", "sdmvpsu"],
    "respiratory": ["hospital_mortality", "diedinhospital", "hospitaldischargestatus", "unitdischargeoffset",
                    "hospitaldischargeoffset", "dischargelocation", "uniquepid", "patientunitstayid",
                    "hospitalid", "patienthealthsystemstayid", "apachescore", "predictedhospitalmortality"],
}
SENS_VALUES = {"sens_sex": {"Male", "Female"}, "sens_age_group": {"18-39", "40-64", "65-79", "80+"},
               "sens_race_ethnicity": {"White", "Black", "Hispanic", "Other"}}


def single_feature_auc(x, y):
    if not pd.api.types.is_numeric_dtype(x):
        x = x.astype("category").cat.codes.replace(-1, np.nan)
    x = x.astype(float)
    if x.notna().sum() < 50 or x.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, x.fillna(x.median()))
    return max(auc, 1 - auc)


def check(name):
    df = pd.read_csv(ROOT / name / "raw" / f"{name}_raw.csv", low_memory=False)
    issues = []
    if len(df) != EXPECTED[name]:
        issues.append(f"row count {len(df)} != expected {EXPECTED[name]}")
    y = df.target
    if y.isna().any() or set(y.unique()) != {0, 1}:
        issues.append("target not binary / has missing")
    features = [c for c in df.columns if not c.startswith(("meta_", "sens_")) and c != "target"]
    bad = [f for f in features if f.lower() in FORBIDDEN[name]]
    if bad:
        issues.append(f"forbidden predictors present: {bad}")
    for col, allowed in SENS_VALUES.items():
        if col not in df:
            issues.append(f"missing {col}")
        elif not set(df[col].dropna().unique()) <= allowed:
            issues.append(f"{col} has unexpected values {set(df[col].dropna().unique()) - allowed}")

    X = df[features].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    single = {f: round(single_feature_auc(X[f], y), 4) for f in features}
    top_single = dict(sorted(single.items(), key=lambda kv: -kv[1])[:5])
    leaky = {f: a for f, a in single.items() if a > 0.95}
    if leaky:
        issues.append(f"single predictor AUROC > 0.95 (possible leakage): {leaky}")

    if name == "respiratory":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(splitter.split(X, y, groups=df.meta_split_group_uniquepid))
        overlap = set(df.meta_split_group_uniquepid.iloc[tr]) & set(df.meta_split_group_uniquepid.iloc[te])
        if overlap:
            issues.append("patient overlap between train and test")
    else:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features="from_dtype",
                                           random_state=42)
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    auroc = roc_auc_score(y.iloc[te], p)
    if auroc > 0.97:
        issues.append(f"baseline AUROC {auroc:.3f} suspiciously high")
    return {
        "rows": int(len(df)), "features": len(features), "positive_rate": round(float(y.mean()), 4),
        "predictor_missing_cells_pct": round(float(X.isna().to_numpy().mean() * 100), 2),
        "sens_group_counts": {c: df[c].value_counts(dropna=False).to_dict() for c in SENS_VALUES},
        "smoke_test_baseline": {"model": "HistGradientBoosting (native NaN handling), default settings",
                                "split": "patient-grouped 80/20" if name == "respiratory" else "stratified 80/20",
                                "test_auroc": round(float(auroc), 4),
                                "test_auprc": round(float(average_precision_score(y.iloc[te], p)), 4)},
        "top_single_feature_auroc": top_single,
        "issues": issues,
        "status": "PASS" if not issues else "CHECK",
    }


def main():
    report = {}
    for name in EXPECTED:
        report[name] = check(name)
        r = report[name]
        print(f"{name:13s} {r['status']}  rows={r['rows']} features={r['features']} "
              f"pos={r['positive_rate']:.3f} AUROC={r['smoke_test_baseline']['test_auroc']:.3f} "
              f"top1={next(iter(r['top_single_feature_auroc'].items()))} issues={r['issues']}")
    for name, record in report.items():
        path = ROOT / name / "raw" / f"{name}_raw_validation.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
