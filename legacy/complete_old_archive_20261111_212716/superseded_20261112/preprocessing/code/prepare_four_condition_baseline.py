"""Fit source-training-only feature transformations for four diagnosis tasks.

The source model-ready CSVs retain their original missing values. This script
creates separate numerical matrices, aligned row metadata and labels for both
transfer directions. It trains no prediction model and reports no performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "cross_dataset"
sys.path.insert(0, str(ROOT))
for reader in [ROOT.parent / "_data_reader", ROOT.parent / "_model_reader"]:
    if reader.exists():
        sys.path.insert(0, str(reader))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download_mimic import save_json, utc_now
from prepare_cross_dataset import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from prepare_four_condition_tables import CONDITIONS

METADATA = ["meta_source", "meta_patient_id", "meta_stay_id",
            "meta_domain_hospitalid", "meta_split"]
SENSITIVE = ["sens_age_group", "sens_sex", "sens_race_ethnicity"]
TARGETS = ["target_" + name for name in CONDITIONS]
PARTITIONS = ("train", "validation", "test", "external")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def predictors(frame):
    """Select an explicit feature allowlist and normalize missing-value types."""
    result = frame.loc[:, FEATURES].copy()
    for feature in NUMERIC_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="raise").astype("float64")
        if np.isinf(result[feature].to_numpy()).any():
            raise ValueError("Infinite input predictor: " + feature)
    for feature in CATEGORICAL_FEATURES:
        result[feature] = result[feature].astype(object).where(result[feature].notna(), np.nan)
    return result


def load_source_tables(input_dir, source):
    """Require identical X, row order and partitions across four source tasks."""
    common = None
    input_hashes = {}
    expected = set(METADATA + SENSITIVE + FEATURES + ["target"])
    for condition in CONDITIONS:
        path = Path(input_dir) / source / (condition + "_model_ready.csv")
        frame = pd.read_csv(path, dtype={name: "string" for name in METADATA})
        if set(frame.columns) != expected:
            raise ValueError(f"Unexpected schema in {path}: missing={expected - set(frame.columns)}, "
                             f"extra={set(frame.columns) - expected}")
        if frame.empty or frame[METADATA].isna().any().any():
            raise ValueError("Missing metadata or empty source table: " + str(path))
        if set(frame.meta_source) != {source} or not frame.meta_stay_id.is_unique:
            raise ValueError("Wrong source or duplicate stay IDs in " + str(path))
        if not frame.target.isin([0, 1]).all():
            raise ValueError("Non-binary recorded-diagnosis label in " + str(path))
        baseline = frame[METADATA + SENSITIVE + FEATURES]
        if common is None:
            common = baseline.copy()
        elif not baseline.equals(common[METADATA + SENSITIVE + FEATURES]):
            raise ValueError("Four condition tables must share identical predictors, row order "
                             "and metadata: " + str(path))
        common["target_" + condition] = frame.target.astype("int8")
        input_hashes[str(path)] = file_sha256(path)
    if set(common.meta_split) != {"train", "validation", "test"}:
        raise ValueError("All three source partitions are required")
    if common.groupby("meta_patient_id").meta_split.nunique().max() != 1:
        raise ValueError("A source patient crosses train/validation/test partitions")
    if source == "eicu" and common.groupby("meta_domain_hospitalid").meta_split.nunique().max() != 1:
        raise ValueError("An eICU hospital crosses train/validation/test partitions")
    predictors(common)  # Validate numeric values before fitting any transformation.
    return common, input_hashes


def create_transformer():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)


def fit_transform_direction(source, external):
    """Only source train predictors enter fit; labels and metadata never do."""
    transformer = create_transformer()
    train = source.loc[source.meta_split.eq("train")]
    if train.empty:
        raise ValueError("Cannot fit transformations without source training rows")
    transformer.fit(predictors(train))
    if list(transformer.feature_names_in_) != list(FEATURES):
        raise AssertionError("Identifier, label or extra metadata entered the transformer")
    feature_names = transformer.get_feature_names_out().tolist()
    partitions = {}
    for name in PARTITIONS:
        frame = external if name == "external" else source.loc[source.meta_split.eq(name)]
        if frame.empty:
            raise ValueError("Empty transformation partition: " + name)
        matrix = np.asarray(transformer.transform(predictors(frame)), dtype="float64")
        if matrix.ndim != 2 or matrix.shape != (len(frame), len(feature_names)) or not np.isfinite(matrix).all():
            raise ValueError("Non-finite or inconsistent transformed matrix: " + name)
        partitions[name] = {"X": matrix,
            "rows": frame[METADATA + SENSITIVE + TARGETS].reset_index(drop=True).copy()}
    return transformer, feature_names, partitions


def transformer_statistics(transformer):
    numeric = transformer.named_transformers_["numeric"]
    categorical = transformer.named_transformers_["categorical"]
    return {
        "numeric_features": list(NUMERIC_FEATURES),
        "source_training_numeric_imputation_values": numeric.named_steps["imputer"].statistics_.tolist(),
        "numeric_missing_indicator_indices": numeric.named_steps["imputer"].indicator_.features_.tolist(),
        "numeric_scaler_mean": numeric.named_steps["scale"].mean_.tolist(),
        "numeric_scaler_scale": numeric.named_steps["scale"].scale_.tolist(),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "source_training_categorical_imputation_values": categorical.named_steps["imputer"].statistics_.tolist(),
        "source_training_categories": [values.tolist() for values in categorical.named_steps["encode"].categories_],
    }


def write_direction(source_name, external_name, source, external, output_dir, input_hashes):
    output = Path(output_dir) / (source_name + "_to_" + external_name)
    output.mkdir(parents=True, exist_ok=True)
    transformer, feature_names, partitions = fit_transform_direction(source, external)
    joblib.dump(transformer, output / "preprocessor.joblib")
    save_json(output / "feature_names.json", {"input_features": list(FEATURES), "output_features": feature_names})
    artifacts = {}
    for name, partition in partitions.items():
        matrix, rows = partition["X"], partition["rows"]
        x_path, y_path, row_path = output / ("X_" + name + ".npz"), output / ("y_" + name + ".npz"), output / ("rows_" + name + ".csv")
        np.savez_compressed(x_path, X=matrix)
        np.savez_compressed(y_path, **{condition: rows["target_" + condition].to_numpy(dtype="int8") for condition in CONDITIONS})
        rows.to_csv(row_path, index=False)
        artifacts[name] = {"rows": len(rows), "features": matrix.shape[1], "all_finite": True,
            "X_path": str(x_path), "y_path": str(y_path), "row_metadata_path": str(row_path),
            "sha256": {path.name: file_sha256(path) for path in [x_path, y_path, row_path]},
            "positive_labels": {condition: int(rows["target_" + condition].sum()) for condition in CONDITIONS}}
    details = {"completed": True, "completed_utc": utc_now(),
        "direction": source_name + "_to_" + external_name,
        "fit_source": source_name, "external_source": external_name,
        "fit_partition": "train", "fit_rows": int(source.meta_split.eq("train").sum()),
        "input_predictors": list(FEATURES), "output_feature_count": len(feature_names),
        "metadata_or_labels_used_as_predictors": False, "labels_used_to_fit_transformations": False,
        "source_validation_test_or_external_rows_used_for_fit": 0,
        "prediction_models_trained": 0, "performance_evaluation_completed": False,
        "raw_model_ready_csvs_modified": False,
        "numeric_method": "Training median imputation with missing indicators, then training StandardScaler",
        "categorical_method": "Training most-frequent imputation, then training one-hot categories; unknown categories ignored",
        "empty_training_numeric_feature": "Kept and imputed to zero by keep_empty_features=True",
        "missing_indicator_policy": "Indicators are learned for features missing in source training rows only",
        "external_rows": "All rows from the other dataset; its existing meta_split is preserved only as metadata",
        "row_alignment": "Each X/y/rows artifact shares identical row order; rows CSV includes source, patient and stay keys",
        "label_meaning": "Recorded diagnosis versus not documented among encounters with parseable diagnoses; not confirmed disease absence",
        "input_csv_sha256": input_hashes, "partitions": artifacts,
        "fit_statistics": transformer_statistics(transformer),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    save_json(output / "transformation_validation.json", details)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "four_conditions")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "preprocessing/outputs/cross_dataset_baseline")
    args = parser.parse_args()
    # Load and validate both sources before any fitted artifact is written.
    sources, hashes = {}, {}
    for name in ["mimic", "eicu"]:
        sources[name], source_hashes = load_source_tables(args.input_dir, name)
        hashes.update(source_hashes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"completed": False, "started_utc": utc_now(), "prediction_models_trained": 0, "directions": {}}
    save_json(args.output_dir / "baseline_preprocessing_status.json", summary)
    for source, external in [("mimic", "eicu"), ("eicu", "mimic")]:
        summary["directions"][source + "_to_" + external] = write_direction(
            source, external, sources[source], sources[external], args.output_dir, hashes)
        save_json(args.output_dir / "baseline_preprocessing_status.json", summary)
    summary.update(completed=True, completed_utc=utc_now())
    save_json(args.output_dir / "baseline_preprocessing_status.json", summary)
    print(json.dumps({"completed": True, "prediction_models_trained": 0, "output_dir": str(args.output_dir),
        "directions": {name: {"train_rows": record["fit_rows"], "feature_count": record["output_feature_count"]}
                       for name, record in summary["directions"].items()}}), flush=True)


if __name__ == "__main__":
    main()
