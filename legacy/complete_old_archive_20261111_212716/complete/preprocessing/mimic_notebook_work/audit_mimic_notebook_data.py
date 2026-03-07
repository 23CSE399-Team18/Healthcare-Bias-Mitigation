"""Read-only source data/environment audit for the MIMIC comparison notebook."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "cross_dataset/four_conditions/mimic"
OUTPUT = Path(__file__).resolve().parent
CONDITIONS = ["heart", "respiratory", "diabetes", "hypertension"]
PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn", "pyod": "pyod", "fairlearn": "fairlearn",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "nbformat": "nbformat",
    "nbclient": "nbclient", "jupyter_client": "jupyter_client", "ipykernel": "ipykernel",
    "torch": "torch", "cleanlab": "cleanlab", "hyperimpute": "hyperimpute",
    "ctgan": "ctgan", "sdv": "sdv", "diffusers": "diffusers",
    "denoising_diffusion_pytorch": "denoising-diffusion-pytorch",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost", "optuna": "optuna",
}


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment_inventory():
    result = {"python_executable": sys.executable, "python_version": sys.version,
              "packages": {}, "checked_utc": datetime.now(timezone.utc).isoformat()}
    for module, distribution in PACKAGES.items():
        spec = importlib.util.find_spec(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result["packages"][module] = {"available": spec is not None,
            "version": version, "origin": spec.origin if spec is not None else None}
    return result


def label_counts(values):
    positive, total = int(values.sum()), int(len(values))
    negative = total - positive
    probability = positive / total if total else None
    entropy = -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability)) if probability not in (None, 0, 1) else 0.
    return {"rows": total, "positive": positive, "not_documented": negative,
        "positive_rate": probability, "majority_minority_ratio": max(positive, negative) / min(positive, negative) if min(positive, negative) else None,
        "class_entropy_bits": float(entropy)}


def duplicate_audit(frame, features):
    x = frame[features]
    duplicate_rows = x.duplicated(keep=False)
    excess = x.duplicated(keep="first")
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(frame[name])]
    all_numeric_missing = x[numeric].isna().all(axis=1)
    sizes = frame.groupby(features, dropna=False, observed=True, sort=False).size()
    candidate = frame.loc[duplicate_rows]
    cross_splits = candidate.groupby(features, dropna=False, observed=True, sort=False).meta_split.nunique() if len(candidate) else pd.Series(dtype=int)
    contradictory = {}
    for condition in CONDITIONS:
        groups = candidate.groupby(features, dropna=False, observed=True, sort=False)["target_" + condition].nunique() if len(candidate) else pd.Series(dtype=int)
        contradictory[condition] = int(groups.gt(1).sum())
    return {"feature_count": len(features), "rows_sharing_a_feature_tuple": int(duplicate_rows.sum()),
        "excess_identical_feature_tuples": int(excess.sum()), "unique_feature_tuples": int(len(x) - excess.sum()),
        "duplicate_tuple_groups": int(sizes.gt(1).sum()), "largest_tuple_group": int(sizes.max()),
        "duplicate_tuple_rows_with_all_numeric_predictors_missing": int((duplicate_rows & all_numeric_missing).sum()),
        "duplicate_tuple_rows_with_some_numeric_measurements": int((duplicate_rows & ~all_numeric_missing).sum()),
        "feature_tuple_groups_crossing_splits": int(cross_splits.gt(1).sum()),
        "feature_tuple_groups_with_different_documented_labels": contradictory,
        "interpretation": "Matching measured features across distinct stays do not establish duplicated patient records or incorrect diagnoses; no rows removed"}


def feature_audit(frame, features):
    numerical = [name for name in features if pd.api.types.is_numeric_dtype(frame[name])]
    training = frame.loc[frame.meta_split.eq("train")]
    details, outliers, flags_by_split = {}, {}, {}
    for split in ["train", "validation", "test"]:
        rows = frame.loc[frame.meta_split.eq(split)]
        flags_by_split[split] = pd.Series(False, index=rows.index)
    for feature in numerical:
        values = frame[feature]
        train = training[feature].dropna()
        quantiles = values.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        details[feature] = {"observed": int(values.notna().sum()), "missing": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()), "unique_observed": int(values.nunique()),
            "mean": float(values.mean()), "std": float(values.std()), "skew": float(values.skew()),
            "excess_kurtosis": float(values.kurt()),
            "quantiles": {str(q): float(value) for q, value in quantiles.items()}}
        q1, q3, median = (float(train.quantile(q)) for q in [.25, .75, .5])
        iqr, mad = q3 - q1, float((train - median).abs().median())
        outliers[feature] = {"fitted_from": "MIMIC training rows only", "training_median": median,
            "training_iqr": iqr, "training_mad": mad, "iqr_rule_defined": iqr > 0,
            "mad_rule_defined": mad > 0, "partitions": {}}
        for split in ["train", "validation", "test"]:
            rows = frame.loc[frame.meta_split.eq(split)]
            vals = rows[feature]
            iqr_flags = (vals.lt(q1 - 1.5 * iqr) | vals.gt(q3 + 1.5 * iqr)) if iqr > 0 else pd.Series(False, index=vals.index)
            extreme_flags = (vals.lt(q1 - 3 * iqr) | vals.gt(q3 + 3 * iqr)) if iqr > 0 else pd.Series(False, index=vals.index)
            mad_flags = ((vals - median).abs() * .6744897501960817 / mad).gt(3.5) if mad > 0 else pd.Series(False, index=vals.index)
            flags_by_split[split] |= iqr_flags | mad_flags
            outliers[feature]["partitions"][split] = {"observed": int(vals.notna().sum()),
                "beyond_1_5_iqr": int(iqr_flags.sum()) if iqr > 0 else None,
                "beyond_3_iqr": int(extreme_flags.sum()) if iqr > 0 else None,
                "modified_z_over_3_5": int(mad_flags.sum()) if mad > 0 else None}
    missing = {}
    for split in ["all", "train", "validation", "test"]:
        rows = frame if split == "all" else frame.loc[frame.meta_split.eq(split)]
        missing[split] = {"rows": len(rows), "rows_with_any_missing_predictor": int(rows[features].isna().any(axis=1).sum()),
            "rows_with_all_numeric_predictors_missing": int(rows[numerical].isna().all(axis=1).sum()),
            "missing_counts": rows[features].isna().sum().astype(int).to_dict()}
    stds = {feature: details[feature]["std"] for feature in numerical if details[feature]["std"] > 0}
    medians = {feature: abs(details[feature]["quantiles"]["0.5"]) for feature in numerical if details[feature]["quantiles"]["0.5"] != 0}
    return {"predictors": features, "numeric_predictor_count": len(numerical),
        "categorical_predictor_count": len(features) - len(numerical), "missingness": missing,
        "numeric_summary": details, "robust_outlier_audit": outliers,
        "rows_flagged_by_any_defined_iqr_or_mad_rule": {name: int(flags.sum()) for name, flags in flags_by_split.items()},
        "scale": {"largest_smallest_nonzero_std_ratio": max(stds.values()) / min(stds.values()),
                  "largest_smallest_nonzero_abs_median_ratio": max(medians.values()) / min(medians.values()),
                  "interpretation": "Different physical units/scales motivate scaling; they do not establish collection or demographic bias"},
        "duplicate_features": duplicate_audit(frame, features),
        "outlier_interpretation": "Statistical tail candidates, not verified medical errors; zero-IQR/MAD rules are undefined and not used; no rows removed"}


def audit():
    reference, hashes = None, {}
    for condition in CONDITIONS:
        path = SOURCE / (condition + "_model_ready.csv")
        frame = pd.read_csv(path, dtype={"meta_stay_id": "string", "meta_patient_id": "string", "meta_domain_hospitalid": "string"})
        common_columns = [name for name in frame if name != "target"]
        if reference is None:
            reference = frame[common_columns].copy()
        else:
            assert reference[common_columns].equals(frame[common_columns]), "Condition files do not share identical rows/features"
        assert frame.target.isin([0, 1]).all()
        reference["target_" + condition] = frame.target.to_numpy()
        hashes[str(path.relative_to(PROJECT))] = sha256(path)
    frame = reference
    schema_path = SOURCE / "clinical_feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    core = schema["baseline_numeric_predictors"] + schema["baseline_categorical_predictors"]
    extended = core + schema["optional_extended_predictors_not_yet_harmonized_to_eicu"]
    clinical_path = SOURCE / "clinical_24h_features.csv"
    clinical = pd.read_csv(clinical_path, dtype={"stay_id": "string"})
    joined = frame.merge(clinical[["stay_id"] + schema["optional_extended_predictors_not_yet_harmonized_to_eicu"]],
                         left_on="meta_stay_id", right_on="stay_id", how="left", validate="one_to_one", indicator=True)
    assert joined._merge.eq("both").all()
    assert len(core) == 11 and len(extended) == 32 and len(frame) == 73990
    hashes[str(clinical_path.relative_to(PROJECT))] = sha256(clinical_path)
    repeat_counts = frame.meta_patient_id.value_counts()
    labels = {}
    for condition in CONDITIONS:
        column = "target_" + condition
        labels[condition] = {"all": label_counts(frame[column]),
            "by_split": {split: label_counts(rows[column]) for split, rows in frame.groupby("meta_split", sort=True)}}
    groups = {}
    for attribute in ["sens_sex", "sens_age_group", "sens_race_ethnicity"]:
        groups[attribute] = {}
        for value, rows in frame.groupby(attribute, dropna=False, observed=True, sort=True):
            groups[attribute][str(value)] = {"rows": len(rows), "by_split": rows.meta_split.value_counts().astype(int).to_dict(),
                "labels": {condition: label_counts(rows["target_" + condition]) for condition in CONDITIONS}}
    result = {"completed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "source": "MIMIC-IV v3.1; authenticated official downloads previously verified",
        "rows": len(frame), "patients": int(frame.meta_patient_id.nunique()),
        "clinical_source_rows_before_documentation_filter": len(clinical),
        "excluded_no_parseable_diagnosis_documentation": len(clinical) - len(frame),
        "source_csvs_modified": False, "models_trained": 0, "eicu_data_opened": False,
        "split_rows": frame.meta_split.value_counts().astype(int).to_dict(),
        "split_patient_overlap_groups": int(frame.groupby("meta_patient_id").meta_split.nunique().gt(1).sum()),
        "record_id_audit": {"duplicate_stay_ids": int(frame.meta_stay_id.duplicated().sum()),
            "duplicate_full_rows": int(frame.duplicated().sum()),
            "patients_with_multiple_icu_stays": int(repeat_counts.gt(1).sum()),
            "rows_belonging_to_patients_with_multiple_stays": int(repeat_counts[repeat_counts.gt(1)].sum()),
            "extra_stays_beyond_one_per_patient": int(len(frame) - len(repeat_counts))},
        "domain_audit": {"hospitals": int(frame.meta_domain_hospitalid.nunique()),
            "hospital_counts": frame.meta_domain_hospitalid.value_counts().astype(int).to_dict(),
            "single_center_mimic_cannot_measure_cross_hospital_domain_generalization": True,
            "eicu_external_data_reserved": True},
        "labels": labels, "demographics": groups,
        "feature_sets": {"core_11": feature_audit(frame, core), "extended_32": feature_audit(joined, extended)},
        "label_limits": "Hospital-discharge coded diagnosis; zero means not documented, not confirmed disease absence; not onset prediction",
        "input_sha256": hashes}
    save(OUTPUT / "data_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages-only", action="store_true")
    parser.add_argument("--package-output", type=Path, default=OUTPUT / "package_availability.json")
    args = parser.parse_args()
    addon = PROJECT / "_mimic_notebook_deps"
    if addon.exists():
        sys.path.insert(0, str(addon))
    save(args.package_output, environment_inventory())
    if not args.packages_only:
        result = audit()
        print(json.dumps({"rows": result["rows"], "patients": result["patients"],
            "labels": {name: values["all"] for name, values in result["labels"].items()},
            "core_missing_any": result["feature_sets"]["core_11"]["missingness"]["all"]["rows_with_any_missing_predictor"],
            "extended_missing_any": result["feature_sets"]["extended_32"]["missingness"]["all"]["rows_with_any_missing_predictor"],
            "core_tuple_duplicates": result["feature_sets"]["core_11"]["duplicate_features"],
            "extended_tuple_duplicates": result["feature_sets"]["extended_32"]["duplicate_features"]}), flush=True)
