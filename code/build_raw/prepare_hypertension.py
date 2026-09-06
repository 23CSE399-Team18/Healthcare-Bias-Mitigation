"""Prepare a transparent, unimputed NHIS 2023 hypertension classification cohort.

Run download_nhis2023.py first. No model training or train/test selection is done.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parents[2] / "hypertension" / "raw" / "source"
import pandas as pd

DICTIONARY = {
    "HHX": ("identifier", "Random household ID; group splits by HHX if ever combining adult and child or repeated household records"),
    "SRVY_YR": ("metadata", "NHIS survey year"),
    "WTFA_A": ("survey_weight", "Final annual sample-adult survey weight; do not use as predictor"),
    "PSTRAT": ("survey_design", "Public-use pseudo-stratum; do not use as predictor"),
    "PPSU": ("survey_design", "Public-use pseudo-PSU; do not use as predictor"),
    "REGION": ("domain_metadata", "1 Northeast; 2 Midwest; 3 South; 4 West; optional region-held-out evaluation domain"),
    "URBRRL": ("domain_metadata", "1 large central metro; 2 large fringe metro; 3 medium/small metro; 4 nonmetro"),
    "AGEP_A": ("candidate_predictor_and_fairness", "Age years: 18-84,85 means85+; 97/98/99 become missing"),
    "SEX_A": ("candidate_predictor_and_fairness", "Reported sex:1 male,2 female;7/8/9 become missing"),
    "HISPALLP_A": ("fairness_metadata", "Race/ethnicity recode:1 Hispanic;2 NH White;3 NH Black;4 NH Asian;5 NH AIAN;6 NH AIAN+other;7 other;includes CDC-imputed race/ethnicity"),
    "RACEALLP_A": ("fairness_metadata", "Race groups:1 White;2 Black;3 Asian;4 AIAN;5 AIAN+other;6 other/multiple;7/8/9 become missing;includes source imputation"),
    "HISP_A": ("fairness_metadata", "Hispanic ethnicity:1 yes,2 no;7/8/9 become missing;includes source imputation"),
    "EDUCP_A": ("candidate_predictor", "Education:0-10 per source codebook;97/98/99 become missing"),
    "BMICAT_A": ("candidate_predictor", "1 underweight;2 healthy weight;3 overweight;4 obese;9 becomes missing;CDC category uses internal untruncated height/weight"),
    "SMKCIGST_A": ("candidate_predictor", "1 daily;2 some-day;3 former;4 never;5 smoker with unknown current status;9 becomes missing"),
    "HYPEV_A": ("label_source_exclude_from_predictors", "Ever told by health professional of hypertension:1 yes,2 no;7 refused,8 not ascertained,9 unknown"),
    "hypertension": ("target", "1 when HYPEV_A=1;0 when HYPEV_A=2"),
}


def prepare():
    raw_path = HERE / "raw" / "adult23.csv"
    raw = pd.read_csv(raw_path, dtype={"HHX": "string"}, low_memory=False)
    required = list(DICTIONARY)[:-1]
    absent = sorted(set(required) - set(raw.columns))
    if absent:
        raise ValueError("Missing required columns: " + ",".join(absent))
    if len(raw) != 29522:
        raise ValueError(f"Unexpected NHIS2023 record count {len(raw)}; inspect source release")
    if not raw.HHX.is_unique:
        raise ValueError("Sample adult household IDs are not unique")
    if not raw.SRVY_YR.eq(2023).all() or not raw.HHSTAT_A.eq(1).all():
        raise ValueError("Unexpected survey year or non-adult sample record")
    # NHIS sample adult eligibility already establishes age>=18, including unknown exact ages.
    known_age = raw.AGEP_A.between(18,85)
    unknown_age = raw.AGEP_A.isin([97,98,99]) | raw.AGEP_A.isna()
    if not (known_age | unknown_age).all():
        raise ValueError("Unexpected age codes")
    eligible = raw.HYPEV_A.isin([1,2])
    excluded = raw.loc[~eligible, ["HHX", "SRVY_YR", "HYPEV_A"]].copy()
    excluded["exclusion_reason"] = "Hypertension label refused/not ascertained/unknown/missing"
    cohort = raw.loc[eligible, required].copy()
    cohort["hypertension"] = cohort.HYPEV_A.map({1:1,2:0}).astype("int8")
    sentinels = {
        "AGEP_A":[97,98,99], "SEX_A":[7,8,9], "HISPALLP_A":[97,98,99],
        "RACEALLP_A":[7,8,9], "HISP_A":[7,8,9], "EDUCP_A":[97,98,99],
        "BMICAT_A":[9], "SMKCIGST_A":[9],
    }
    for column, missing_codes in sentinels.items():
        cohort[column] = cohort[column].mask(cohort[column].isin(missing_codes))
    if cohort.WTFA_A.isna().any() or not cohort.WTFA_A.gt(0).all():
        raise ValueError("Invalid survey weights")
    derived = HERE / "derived"
    derived.mkdir(exist_ok=True)
    cohort.to_csv(derived / "hypertension_nhis2023_raw_cohort.csv", index=False)
    excluded.to_csv(derived / "excluded_labels.csv", index=False)
    pd.DataFrame([
        {"column": column, "role": role, "description": desc}
        for column, (role,desc) in DICTIONARY.items()
    ]).to_csv(derived / "data_dictionary.csv", index=False)
    summary = {
        "source": "CDC NHIS2023 sample adult public use",
        "source_records": len(raw), "retained_records": len(cohort),
        "hypertension_positive": int(cohort.hypertension.sum()),
        "hypertension_negative": int(cohort.hypertension.eq(0).sum()),
        "excluded_unknown_labels": len(excluded),
        "raw_label_counts": {str(k):int(v) for k,v in raw.HYPEV_A.value_counts(dropna=False).items()},
        "retained_with_unknown_exact_age": int(cohort.AGEP_A.isna().sum()),
        "raw_csv_bytes": raw_path.stat().st_size,
        "raw_csv_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "missing_after_documented_sentinel_recode": {k:int(v) for k,v in cohort.isna().sum().items()},
        "candidate_baseline_predictors": ["AGEP_A","SEX_A","EDUCP_A","BMICAT_A","SMKCIGST_A"],
        "prohibited_predictor_families": ["HYPEV_A","hypertension","HYPDIF_A","HYP12M_A","HYPMED_A","BPLAST_A","all identifiers and survey design fields"],
        "target": "Self-reported lifetime clinician-diagnosed hypertension; not measured BP or future incident hypertension",
        "processing": "Source missing codes converted to null; no imputation, resampling, scaling, model fitting, or test selection",
    }
    (HERE / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (HERE / "feature_schema.json").write_text(json.dumps({
        "target":"hypertension", "candidate_baseline_predictors":summary["candidate_baseline_predictors"],
        "fairness_attributes":["AGEP_A","SEX_A","HISPALLP_A"],
        "identifier_columns":["HHX"], "survey_design_columns":["WTFA_A","PSTRAT","PPSU"],
        "domain_metadata":["REGION","URBRRL"],
        "label_source_exclude_from_predictors":["HYPEV_A"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    prepare()
