"""Correct the eICU race/ethnicity audit labels in the cross-hospital tables (run once; safe to re-run).

    python hospital/code/fix_eicu_race_labels.py

Bug: race_group() in prepare_cross_dataset.py tested the plain substring "ASIAN" after the White rule, and "CAUCASIAN"
contains "ASIAN", so every eICU Caucasian stay was labelled "Asian" (MIMIC never uses "Caucasian" and was not affected).
The function is fixed (whole-word match). This script recomputes sens_race_ethnicity for every eICU stay from the
official eICU patient table with the fixed function and rewrites only that column. It asserts that the only rows
that change are Caucasian stays moving from "Asian" to "White". Race is an audit attribute, never a model input, so
no feature, split, label or fitted transformation changes. Writes preprocessed/race_fix_audit.json.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATIENT = Path("D:/complete/respiratory/raw/source/official_verified/patient.csv.gz")   # official eICU-CRD v2.0 file

src = (HERE / "prepare_cross_dataset.py").read_text(encoding="utf-8")
ns = {"pd": pd}
exec(src[src.index("def race_group"):src.index("class HourlyAccumulator")], ns)   # the fixed function, nothing else
race_group = ns["race_group"]

patient = pd.read_csv(PATIENT, usecols=["patientunitstayid", "ethnicity"])
ethnicity = dict(zip(patient.patientunitstayid.astype(int), patient.ethnicity))
fixed = dict(zip(patient.patientunitstayid.astype(int), race_group(patient.ethnicity)))

audit = {"bug": "substring 'ASIAN' matched inside 'CAUCASIAN' (prepare_cross_dataset.race_group)",
         "fix": "whole-word regex \\bASIAN; eICU labels recomputed from patient.csv.gz", "files": {}}
files = sorted((ROOT / "raw" / "eicu").glob("*_model_ready.csv")) + \
    [ROOT / "preprocessed" / f"hospital_{s}_preprocessed.csv" for s in ("train", "validation", "test")]
for path in files:
    df = pd.read_csv(path, low_memory=False)
    assert (df.meta_source == "eicu").all(), path
    stay = df.meta_stay_id.astype(int)
    old = df.sens_race_ethnicity.astype("string")
    new = stay.map(fixed).astype("string")
    assert new.notna().all(), f"{path.name}: stays missing from the eICU patient table"
    changed = old != new
    eth = stay.map(ethnicity)
    assert ((old[changed] == "Asian") & (new[changed] == "White") & (eth[changed] == "Caucasian")).all(), path
    if changed.any():
        df["sens_race_ethnicity"] = new.astype(str)
        df.to_csv(path, index=False)
    audit["files"][str(path.relative_to(ROOT))] = {"rows": int(len(df)), "changed_asian_to_white": int(changed.sum()),
                                                   "before": old.value_counts().to_dict(), "after": new.value_counts().to_dict()}
    print(f"{path.relative_to(ROOT)}: {int(changed.sum()):,} Caucasian stays relabelled Asian -> White | after: {new.value_counts().to_dict()}")
for path in sorted((ROOT / "raw" / "mimic").glob("*_model_ready.csv")) + [ROOT / "preprocessed" / "hospital_external_preprocessed.csv"]:
    df = pd.read_csv(path, usecols=["sens_race_ethnicity"], low_memory=False)
    audit["files"][str(path.relative_to(ROOT))] = {"rows": int(len(df)), "changed_asian_to_white": 0, "note": "MIMIC, unaffected",
                                                   "after": df.sens_race_ethnicity.value_counts().to_dict()}
(ROOT / "preprocessed" / "race_fix_audit.json").write_text(json.dumps(audit, indent=1, default=int), encoding="utf-8")
print("audit ->", ROOT / "preprocessed" / "race_fix_audit.json")
