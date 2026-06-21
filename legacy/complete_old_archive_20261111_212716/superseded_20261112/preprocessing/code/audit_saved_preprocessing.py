"""Independently audit saved source-only preprocessing against actual data.

This checks existing fitted artifacts and matrices. It does not fit a disease
model, select a mitigation method, or modify any input/preprocessing artifact.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2] / "cross_dataset"
sys.path.insert(0, str(ROOT))
PROJECT = ROOT.parent
if sys.version_info >= (3, 14):
    sys.path[:0] = [str(PROJECT / "_model_reader"), str(PROJECT / "_data_reader")]
import joblib
import numpy as np
import pandas as pd

NUMERIC = [f"{v}_{s}" for v in ("heart_rate", "resp_rate", "spo2") for s in ("mean", "min", "max")]
CATEGORICAL = ["age_group", "sex"]
FEATURES = NUMERIC + CATEGORICAL
CONDITIONS = ("heart", "respiratory", "diabetes", "hypertension")
META = ["meta_source", "meta_patient_id", "meta_stay_id", "meta_domain_hospitalid", "meta_split"]
SENSITIVE = ["sens_age_group", "sens_sex", "sens_race_ethnicity"]
OUT = PROJECT / "preprocessing/outputs/audit_current"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_sources():
    sources, hashes = {}, {}
    for source in ("mimic", "eicu"):
        common = None
        for condition in CONDITIONS:
            path = ROOT / f"four_conditions/{source}/{condition}_model_ready.csv"
            frame = pd.read_csv(path, dtype={c: "string" for c in META})
            assert set(frame.columns) == set(META + SENSITIVE + FEATURES + ["target"])
            assert frame.meta_stay_id.is_unique and frame.target.isin([0, 1]).all()
            base = frame[META + SENSITIVE + FEATURES]
            if common is None:
                common = base.copy()
            else:
                pd.testing.assert_frame_equal(base, common[META + SENSITIVE + FEATURES])
            common["target_" + condition] = frame.target.to_numpy(dtype=np.int8)
            hashes[str(path)] = sha(path)
        assert common.groupby("meta_patient_id").meta_split.nunique().max() == 1
        if source == "eicu":
            assert common.groupby("meta_domain_hospitalid").meta_split.nunique().max() == 1
        sources[source] = common
    return sources, hashes


def input_features(frame):
    result = frame[FEATURES].copy()
    for name in NUMERIC:
        result[name] = result[name].astype(float)
    for name in CATEGORICAL:
        result[name] = result[name].astype(object).where(result[name].notna(), np.nan)
    return result


def audit_direction(source_name, external_name, sources, input_hashes):
    base = PROJECT / f"preprocessing/outputs/cross_dataset_baseline/{source_name}_to_{external_name}"
    record = json.loads((base / "transformation_validation.json").read_text())
    for path, expected in record["input_csv_sha256"].items():
        assert input_hashes[path] == expected, "Input file changed since fitting: " + path
    transformer = joblib.load(base / "preprocessor.joblib")
    assert transformer.feature_names_in_.tolist() == FEATURES
    source, external = sources[source_name], sources[external_name]
    train = source[source.meta_split.eq("train")]
    numeric = transformer.named_transformers_["numeric"]
    imputer, scaler = numeric.named_steps["imputer"], numeric.named_steps["scale"]
    raw = train[NUMERIC].to_numpy(float)
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    np.testing.assert_allclose(imputer.statistics_, medians, atol=1e-12, rtol=1e-12)
    indicator_indices = np.flatnonzero(np.isnan(raw).any(axis=0))
    np.testing.assert_array_equal(imputer.indicator_.features_, indicator_indices)
    independently_imputed = np.where(np.isnan(raw), medians, raw)
    independently_imputed = np.column_stack([independently_imputed, np.isnan(raw[:, indicator_indices]).astype(float)])
    expected_mean = independently_imputed.mean(axis=0)
    expected_scale = independently_imputed.std(axis=0, ddof=0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(scaler.scale_, expected_scale, atol=1e-10, rtol=1e-10)
    assert int(np.asarray(scaler.n_samples_seen_).max()) == len(train)
    cat = transformer.named_transformers_["categorical"]
    ci, encoder = cat.named_steps["imputer"], cat.named_steps["encode"]
    expected_categories = []
    for i, column in enumerate(CATEGORICAL):
        mode = train[column].dropna().mode().sort_values().iloc[0]
        assert ci.statistics_[i] == mode
        categories = sorted(train[column].fillna(mode).unique())
        assert encoder.categories_[i].tolist() == categories
        expected_categories.append(categories)
    assert encoder.handle_unknown == "ignore"
    feature_names = json.loads((base / "feature_names.json").read_text())["output_features"]
    assert transformer.get_feature_names_out().tolist() == feature_names
    partitions = {}
    for name in ("train", "validation", "test", "external"):
        frame = external if name == "external" else source[source.meta_split.eq(name)]
        frame = frame.reset_index(drop=True)
        paths = [base / f"X_{name}.npz", base / f"y_{name}.npz", base / f"rows_{name}.csv"]
        for path in paths:
            assert sha(path) == record["partitions"][name]["sha256"][path.name]
        with np.load(paths[0], allow_pickle=False) as archive:
            matrix = archive["X"]
        assert matrix.shape == (len(frame), len(feature_names)) and np.isfinite(matrix).all()
        recalculated = transformer.transform(input_features(frame))
        np.testing.assert_allclose(matrix, recalculated, atol=1e-10, rtol=1e-10)
        rows = pd.read_csv(paths[2], dtype={c: "string" for c in META})
        pd.testing.assert_frame_equal(rows[META + SENSITIVE], frame[META + SENSITIVE], check_dtype=False)
        positives = {}
        with np.load(paths[1], allow_pickle=False) as labels:
            for condition in CONDITIONS:
                target = frame["target_" + condition].to_numpy(dtype=np.int8)
                np.testing.assert_array_equal(labels[condition], target)
                np.testing.assert_array_equal(rows["target_" + condition], target)
                positives[condition] = int(target.sum())
        partitions[name] = {"rows": len(frame), "features": matrix.shape[1], "all_finite": True,
            "matrix_equals_frozen_transform": True, "all_four_labels_and_ids_aligned": True,
            "saved_artifact_hashes_match": True, "positive_counts": positives}
    return {"passed": True, "source_training_rows": len(train),
        "training_medians_recomputed": True, "training_scaler_statistics_recomputed": True,
        "training_categorical_modes_and_categories_recomputed": True,
        "trained_feature_names": FEATURES, "encoded_features": len(feature_names),
        "train_only_missing_indicators": indicator_indices.tolist(),
        "categorical_levels": expected_categories, "partitions": partitions,
        "saved_transformer_sha256": sha(base / "preprocessor.joblib")}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"passed": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "Actual saved artifact checks with independently recomputed training statistics",
        "predictive_models_trained": 0, "source_inputs_modified": False}
    try:
        sources, hashes = load_sources()
        state["source_rows"] = {name: len(frame) for name, frame in sources.items()}
        state["source_patients"] = {name: int(frame.meta_patient_id.nunique()) for name, frame in sources.items()}
        state["source_hospitals"] = {name: int(frame.meta_domain_hospitalid.nunique()) for name, frame in sources.items()}
        state["patient_partitions_disjoint"] = True
        state["eicu_hospital_partitions_disjoint"] = True
        state["directions"] = {f"{s}_to_{e}": audit_direction(s, e, sources, hashes)
                               for s, e in [("mimic", "eicu"), ("eicu", "mimic")]}
        clinical = json.loads((ROOT / "four_conditions/mimic/clinical_preprocessing_validation.json").read_text())
        assert clinical["passed"] and clinical["chartevents_gzip_crc_verified_by_full_stream"]
        for name, expected in clinical["output_sha256"].items():
            assert sha(ROOT / "four_conditions/mimic" / name) == expected
        state["clinical_output_hashes_match_full_stream_audit"] = True
        state["input_csv_sha256"] = hashes
        state["passed"] = True
        state["scope_status"] = {"baseline_preprocessing": "verified_complete",
            "eight_bias_mitigation_experiment": "not_assembled_or_run; fragments are development work",
            "previous_mimic_plot_preview": "test_fixture_only_not_patient_results"}
        state["limitations"] = [
            "Labels describe documentation; zeros are not confirmed disease absence",
            "MIMIC hospital-discharge coding differs from eICU ICU/history ascertainment",
            "MIMIC contains one institution; only eICU has internal hospital-held-out partitions",
            "MIMIC documentation-time exclusion has no exact eICU periodic-vital equivalent",
            "No fairness improvement or cross-hospital performance follows from clean preprocessing alone"]
    except BaseException as error:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (OUT / "mimic_audit.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"passed": state["passed"], "source_rows": state["source_rows"], "directions": list(state["directions"])}))
    return state


if __name__ == "__main__":
    run()
