"""Check completed clinical outputs; this does not re-extract raw measurements."""
from pathlib import Path
import argparse
import hashlib
import json

from prepare_mimic_clinical_features import (pd, np, ROOT, VITALS, BOUNDS,
    BASELINE_FEATURES, EXTENDED_FEATURES, atomic_json)


def validate(output_dir):
    processing = json.loads((output_dir / "clinical_preprocessing_validation.json").read_text(encoding="utf-8"))
    if processing.get("status") != "completed" or processing.get("passed") is not True:
        raise ValueError("Clinical extraction has not completed successfully")
    if processing.get("chartevents_gzip_crc_verified_by_full_stream") is not True:
        raise ValueError("Full raw-source gzip integrity check missing")
    for name, expected in processing["output_sha256"].items():
        with (output_dir / name).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Recorded output checksum differs: " + name)
    clinical = pd.read_csv(output_dir / "clinical_24h_features.csv").set_index("stay_id")
    metadata = pd.read_csv(output_dir / "cohort_metadata.csv").set_index("stay_id")
    observed = pd.read_csv(output_dir / "observation_audit.csv").set_index("stay_id")
    if clinical.columns.tolist() != BASELINE_FEATURES + EXTENDED_FEATURES:
        raise ValueError("Feature columns differ from explicit approved clinical schema")
    if not clinical.index.is_unique or not metadata.index.is_unique or not observed.index.is_unique:
        raise ValueError("Duplicate ICU stay key")
    if not clinical.index.equals(metadata.index) or not clinical.index.equals(observed.index):
        raise ValueError("Feature, metadata and observation audit membership/order differs")
    # Compare patient/hospital admission keys against the original actual ICU table.
    original = pd.read_csv(Path(processing["sources"]["icu/icustays.csv.gz"]["path"]),
                           usecols=["stay_id", "subject_id", "hadm_id"]).set_index("stay_id")
    linked = original.loc[clinical.index]
    if not metadata.subject_id.equals(linked.subject_id) or not metadata.hadm_id.equals(linked.hadm_id):
        raise ValueError("Patient/admission join keys differ from original ICU source")
    if not metadata.patient_id.eq("mimic:" + metadata.subject_id.astype("string")).all():
        raise ValueError("Patient grouping identifier is inconsistent")
    if len(clinical) != processing["rows"] or not clinical.age_group.isin(["18-39", "40-64", "65-79", "80+"]).all():
        raise ValueError("Adult-age representation or cohort count mismatch")
    for vital in VITALS:
        lower, upper, upper_inclusive = BOUNDS[vital]
        count = observed[f"{vital}_observed_hours"]
        if not count.between(0, 24).all():
            raise ValueError("Observed hour count outside24h: " + vital)
        values = clinical[[f"{vital}_min", f"{vital}_mean", f"{vital}_max"]]
        if not values.isna().all(axis=1).equals(count.eq(0)):
            raise ValueError("Missing clinical observations disagree with audit: " + vital)
        available = values.dropna()
        if not (available.gt(lower).all().all() and
                ((available.le(upper) if upper_inclusive else available.lt(upper)).all().all())):
            raise ValueError("Converted/aggregated feature outside source filter: " + vital)
        if not ((available.iloc[:, 0] <= available.iloc[:, 1] + 1e-10) &
                (available.iloc[:, 1] <= available.iloc[:, 2] + 1e-10)).all():
            raise ValueError("Hourly aggregation ordering inconsistent: " + vital)
    result = {
        "passed": True, "check_type": "Derived output consistency plus patient/admission joins against original ICU table",
        "not_a_second_independent_measurement_extraction": True,
        "rows": len(clinical), "patients": int(metadata.subject_id.nunique()),
        "hospitals": int(metadata.hospital_id.nunique()), "unique_stay_ids": True,
        "explicit_predictor_schema_matched": True, "no_label_or_discharge_or_identifier_predictors": True,
        "patient_admission_keys_match_original_icustays": True,
        "observed_hours_match_feature_missingness": True,
        "unit_converted_features_inside_bounds": True, "hourly_min_mean_max_ordering": True,
        "full_chartevents_gzip_crc_verified": True,
        "baseline_numeric_predictors": 9, "baseline_categorical_predictors": 2,
        "optional_mimic_only_numeric_predictors": len(EXTENDED_FEATURES),
    }
    atomic_json(output_dir / "clinical_output_consistency_validation.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "four_conditions/mimic")
    validate(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
