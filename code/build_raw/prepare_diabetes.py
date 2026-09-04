r"""Download and prepare a documented NHANES adult diagnosed-diabetes dataset.

Python 3 standard library downloads; preparation additionally requires pandas.
Run: python code/build_raw/prepare_diabetes.py   (files in diabetes/raw/source)
No imputation, resampling, train/test splitting, or model fitting is performed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / "diabetes" / "raw" / "source"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"
COMPONENTS = ("DEMO_L", "DIQ_L", "BMX_L", "SMQ_L", "GHB_L", "GLU_L")
SOURCE_PAGE = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_one(spec):
    component, ext = spec
    url = BASE + component + "." + ext
    path = ROOT / ("raw" if ext == "xpt" else "codebooks") / (component + "." + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            if not path.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; NHANES academic dataset download)", "Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    head = response.read(1024)
                    if ext == "xpt" and not head.startswith(b"HEADER RECORD"):
                        raise ValueError(f"Expected SAS transport header, got {head[:60]!r}")
                    temporary = path.with_suffix(path.suffix + ".part")
                    with temporary.open("wb") as f:
                        f.write(head)
                        while block := response.read(1024 * 1024):
                            f.write(block)
                    temporary.replace(path)
            if ext == "xpt" and not path.open("rb").read(80).startswith(b"HEADER RECORD"):
                raise ValueError(f"Invalid existing SAS transport file: {path}")
            result = {"component": component, "kind": "raw_data" if ext == "xpt" else "official_codebook", "url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path), "retrieved_utc": datetime.now(timezone.utc).isoformat()}
            print(f"VERIFIED {path.name}: {result['bytes']} bytes", flush=True)
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def download():
    specs = [(c, ext) for c in COMPONENTS for ext in ("xpt", "htm")]
    completed, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(download_one, spec): spec for spec in specs}
        for job in concurrent.futures.as_completed(jobs):
            try:
                completed.append(job.result())
            except Exception as e:
                failures.append({"component": jobs[job][0], "extension": jobs[job][1], "error": str(e)})
                print("DOWNLOAD FAILED " + str(jobs[job]) + ": " + str(e), flush=True)
    manifest = {"dataset": "NHANES August 2021-August 2023 adult diagnosed-diabetes task", "provider": "US CDC / National Center for Health Statistics", "official_listing": SOURCE_PAGE, "files": sorted(completed, key=lambda x: x["path"]), "failures": failures}
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; see download_manifest.json")


def prepare():
    import pandas as pd
    data = {c: pd.read_sas(ROOT / "raw" / (c + ".xpt"), format="xport", encoding="utf-8") for c in COMPONENTS}
    for name, df in data.items():
        if not df["SEQN"].is_unique:
            raise ValueError(f"Duplicate SEQN in {name}")
        df["SEQN"] = df["SEQN"].astype("int64")
    demo = data["DEMO_L"]
    selected_bmx = ["SEQN", "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT", "BMDSTATS"]
    selected_bmx = [c for c in selected_bmx if c in data["BMX_L"]]
    selected_smq = ["SEQN", "SMQ020", "SMQ040"]
    joined = demo.merge(data["DIQ_L"], on="SEQN", how="left", validate="one_to_one").merge(data["BMX_L"][selected_bmx], on="SEQN", how="left", validate="one_to_one").merge(data["SMQ_L"][selected_smq], on="SEQN", how="left", validate="one_to_one")
    joined["included_binary_task"] = joined.RIDAGEYR.ge(18) & joined.DIQ010.isin([1, 2])
    joined["exclusion_reason"] = "included"
    joined.loc[joined.RIDAGEYR.lt(18), "exclusion_reason"] = "under_18"
    adult = joined.RIDAGEYR.ge(18)
    for code, name in [(3, "borderline_diabetes"), (7, "refused_diabetes_answer"), (9, "unknown_diabetes_answer")]:
        joined.loc[adult & joined.DIQ010.eq(code), "exclusion_reason"] = name
    joined.loc[adult & joined.DIQ010.isna(), "exclusion_reason"] = "diabetes_answer_missing_or_unavailable"
    joined.loc[joined.RIDAGEYR.isna(), "exclusion_reason"] = "age_missing"
    joined.loc[adult & joined.DIQ010.notna() & ~joined.DIQ010.isin([1, 2, 3, 7, 9]), "exclusion_reason"] = "unrecognised_diabetes_code"
    joined.to_csv(ROOT / "diabetes_all_participants_label_audit.csv", index=False)
    joined.loc[~joined.included_binary_task].to_csv(ROOT / "diabetes_excluded_audit.csv", index=False)
    baseline_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT", "SMQ020", "SMQ040", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "RIDSTATR", "RIDEXPRG", "DIQ010"]
    baseline = joined.loc[joined.included_binary_task, baseline_cols].copy()
    baseline["education_level"] = baseline.DMDEDUC2.where(baseline.DMDEDUC2.isin([1, 2, 3, 4, 5]))
    # Missing SMQ040 for SMQ020=2 is an expected skip, not unknown smoking.
    baseline["smoking_status"] = pd.Series(pd.NA, index=baseline.index, dtype="string")
    baseline.loc[baseline.SMQ020.eq(2), "smoking_status"] = "never"
    baseline.loc[baseline.SMQ020.eq(1) & baseline.SMQ040.eq(3), "smoking_status"] = "former"
    baseline.loc[baseline.SMQ020.eq(1) & baseline.SMQ040.isin([1, 2]), "smoking_status"] = "current"
    baseline["diabetes_diagnosed"] = baseline.DIQ010.map({1: 1, 2: 0}).astype("int64")
    baseline.to_csv(ROOT / "diabetes_baseline_raw.csv", index=False)
    biomarkers = joined.loc[adult, ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDEXPRG", "DIQ010", "included_binary_task", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]].copy()
    biomarkers = biomarkers.merge(data["GHB_L"], on="SEQN", how="left", validate="one_to_one").merge(data["GLU_L"], on="SEQN", how="left", validate="one_to_one")
    biomarkers.to_csv(ROOT / "diabetes_biomarkers_adults.csv", index=False)
    labels = baseline.diabetes_diagnosed.value_counts().sort_index().to_dict()
    summary = {"source_component_rows": {k: len(v) for k, v in data.items()}, "all_participants": len(joined), "adults_18_plus": int(adult.sum()), "binary_cohort_rows": len(baseline), "class_counts": {"no_diabetes": int(labels.get(0, 0)), "diagnosed_diabetes": int(labels.get(1, 0))}, "excluded_rows": int((~joined.included_binary_task).sum()), "exclusion_reasons": joined.loc[~joined.included_binary_task, "exclusion_reason"].value_counts().to_dict(), "baseline_missing_counts": baseline.isna().sum().to_dict(), "biomarker_adult_rows": len(biomarkers), "adult_hba1c_observed": int(biomarkers.LBXGH.notna().sum()), "adult_fasting_glucose_observed": int(biomarkers.LBXGLU.notna().sum()), "unique_participants": bool(baseline.SEQN.is_unique), "target_definition": "age >=18; DIQ010=1 maps to1, DIQ010=2 maps to0; other answers excluded", "baseline_predictors": ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "education_level", "INDFMPIR", "BMXBMI", "BMXWAIST", "smoking_status"]}
    (ROOT / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"reason": k, "rows": int(v)} for k, v in joined.exclusion_reason.value_counts().items()]).to_csv(ROOT / "cohort_flow.csv", index=False)
    dictionary = {
        "SEQN": ("identifier", "NHANES participant ID; join key; never a predictor"),
        "RIDAGEYR": ("predictor / fairness", "Age at screening, years; age80 means80 or older"),
        "RIAGENDR": ("predictor / fairness", "CDC-labelled gender variable:1 male,2 female; retain for fairness evaluation"),
        "RIDRETH3": ("predictor / fairness", "Race/Hispanic origin:1 Mexican American,2 other Hispanic,3 non-Hispanic White,4 non-Hispanic Black,6 non-Hispanic Asian,7 other/multiracial"),
        "DMDEDUC2": ("source provenance, not baseline predictor", "Raw adult20+ education:1 below9th,2 grades9-11,3 high school/GED,4 some college/AA,5 college+,7 refused,9 unknown; structurally missing at18-19"),
        "education_level": ("predictor", "DMDEDUC2 valid1-5;7/9 converted to missing; no imputation"),
        "INDFMPIR": ("predictor", "Family income to poverty ratio, top coded5"),
        "BMXBMI": ("predictor", "Measured body mass index kg/m2"),
        "BMXWAIST": ("predictor", "Measured waist circumference cm"),
        "BMXWT": ("optional predictor, redundant with BMI", "Measured weight kg; excluded from default baseline feature list"),
        "BMXHT": ("optional predictor, redundant with BMI", "Measured standing height cm; excluded from default baseline feature list"),
        "SMQ020": ("source provenance, not baseline predictor", "100+ lifetime cigarettes:1 yes,2 no,7 refused,9 unknown"),
        "SMQ040": ("source provenance, not baseline predictor", "Current smoking:1 every day,2 some days,3 none; skipped for never-smokers"),
        "smoking_status": ("predictor", "never if SMQ020=2; former if020=1 and040=3; current if020=1 and040 in1/2; otherwise missing"),
        "DIQ010": ("label source, never predictor", "Diagnosed diabetes:1 yes,2 no,3 borderline,7 refused,9 unknown; pregnancy-only diabetes excluded by question wording"),
        "diabetes_diagnosed": ("target", "DIQ010=1 ->1; DIQ010=2 ->0"),
        "WTINT2YR": ("survey metadata", "Interview weight for interview-only analyses"),
        "WTMEC2YR": ("survey metadata", "MEC weight for baseline analyses incorporating measured BMI/waist; use appropriate examined population"),
        "SDMVSTRA": ("survey metadata", "Masked variance stratum; not predictor"),
        "SDMVPSU": ("survey metadata", "Masked variance PSU; not predictor and not hospital ID"),
        "RIDSTATR": ("survey metadata", "Interview/examination status;1 interview only,2 interviewed and examined"),
        "RIDEXPRG": ("cohort metadata", "Pregnancy status;1 pregnant,2 not pregnant,3 cannot determine; restricted reporting ages; not a baseline feature"),
        "LBXGH": ("biomarker extension only", "HbA1c percent; not included in baseline predictors"),
        "WTPH2YR": ("survey metadata", "Phlebotomy weight for HbA1c blood analyte analyses; positive weights only in relevant analytic population"),
        "LBXGLU": ("biomarker extension only", "Fasting plasma glucose mg/dL; not included in baseline predictors"),
        "LBDGLUSI": ("biomarker extension only", "Fasting glucose mmol/L, redundant unit conversion; not baseline"),
        "WTSAF2YR": ("survey metadata", "Fasting subsample weight for fasting glucose analyses; supersedes general MEC/interview weights for that subsample"),
        "included_binary_task": ("cohort metadata", "Whether adult with DIQ0101/2"),
        "exclusion_reason": ("cohort metadata", "Reason participant excluded; no excluded records discarded from audit")
    }
    with (ROOT / "data_dictionary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["column", "role", "definition"])
        writer.writerows((name, role, definition) for name, (role, definition) in dictionary.items())
    checks = {"one_row_per_participant": bool(baseline.SEQN.is_unique), "all_adults": bool(baseline.RIDAGEYR.ge(18).all()), "binary_labels_only": bool(baseline.diabetes_diagnosed.isin([0, 1]).all()), "cohort_and_exclusions_partition_source": len(baseline) + int((~joined.included_binary_task).sum()) == len(demo), "no_biomarkers_in_baseline": not any(c in baseline.columns for c in ["LBXGH", "LBXGLU", "LBDGLUSI"]), "no_insulin_or_diabetes_medications_in_baseline": not any(c in baseline.columns for c in ["DIQ050", "DIQ070", "DID040"])}
    if not all(checks.values()):
        raise AssertionError(checks)
    (ROOT / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    generated = [p for p in ROOT.glob("*.csv")] + [ROOT / "cohort_summary.json", ROOT / "validation.json"]
    (ROOT / "derived_files_manifest.json").write_text(json.dumps([{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in generated], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reader-dir", type=Path)
    args = parser.parse_args()
    if args.reader_dir:
        sys.path.insert(0, str(args.reader_dir.resolve()))
    if not args.prepare_only:
        download()
    if not args.download_only:
        prepare()
