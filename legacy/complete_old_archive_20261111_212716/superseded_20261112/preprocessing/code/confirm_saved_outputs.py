"""Read-only confirmation of saved baseline files and their plotting data.

Writes only confirmation.json. Does not fit, regenerate, or overwrite datasets.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "preprocessing/outputs/verified_baseline"
AUDIT = ROOT / "preprocessing/outputs/audit_current"
NAMES = ("heart", "respiratory", "diabetes", "hypertension")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(df, columns):
    return pd.MultiIndex.from_frame(df[columns]) if len(columns) > 1 else pd.Index(df[columns[0]])


def confirm_baseline(name):
    folder = BASE / name
    saved = json.loads((folder / "validation.json").read_text())
    source_path = ROOT / f"model_ready/{name}_model_ready.csv"
    assert digest(source_path) == saved["source_sha256"]
    ids = ["meta_patientunitstayid"] if name == "respiratory" else ["meta_record_id"]
    if name == "heart":
        ids.insert(0, "meta_domain_state")
    types = {c: "string" for c in ids}
    if name == "respiratory":
        types["meta_split_group_uniquepid"] = "string"
    raw = pd.read_csv(source_path, dtype=types)
    raw.index = key(raw, ids)
    assert raw.index.is_unique
    transformer = joblib.load(folder / "preprocessor.joblib")
    names = json.loads((folder / "feature_names.json").read_text())["features"]
    assert transformer.get_feature_names_out().tolist() == names == saved["output_predictors"]
    nums, cats = saved["numeric"], saved["categorical"]
    assert set(nums + cats) == {c for c in raw if c != "target" and not c.startswith(("meta_", "sens_"))}
    id_sets, group_sets, parts, report = {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        path = folder / f"{name}_{split}_preprocessed.csv"
        assert digest(path) == saved["partitions"][split]["sha256"]
        df = pd.read_csv(path, dtype=types)
        ix = key(df, ids)
        assert ix.is_unique and ix.isin(raw.index).all()
        original = raw.loc[ix].copy()
        parts[split] = original
        id_sets[split] = set(ix)
        if name == "respiratory":
            group_sets[split] = set(original.meta_split_group_uniquepid)
        np.testing.assert_array_equal(df.target, original.target)
        assert df.meta_split.eq(split).all() and df.meta_synthetic.eq(0).all()
        x_input = original[nums + cats].copy()
        for c in cats:
            x_input[c] = x_input[c].astype(object).where(x_input[c].notna(), np.nan)
        for c in nums:
            x_input[c] = x_input[c].astype(float)
        transformed = transformer.transform(x_input)
        with np.load(folder / f"X_{split}.npz", allow_pickle=False) as z:
            x = z["X"]
        with np.load(folder / f"y_{split}.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["y"], original.target)
        np.testing.assert_allclose(x, transformed, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(df[names], x, atol=1e-12, rtol=1e-12)
        assert np.isfinite(x).all()
        assert x.shape == (saved["partitions"][split]["rows"], len(names))
        report[split] = {"rows": len(df), "features": len(names), "file_hash_matches": True,
                         "csv_npz_and_frozen_transform_agree": True, "source_ids_and_labels_match": True,
                         "finite_predictors": True}
        if split == "train":
            train_x = x
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        assert not id_sets[a] & id_sets[b]
        if group_sets:
            assert not group_sets[a] & group_sets[b]
    assert set.union(*id_sets.values()) == set(raw.index)
    train = parts["train"]
    medians = train[nums].median().fillna(0).to_numpy()
    imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(medians, imputer.statistics_, atol=1e-12, rtol=1e-12)
    observed = train[nums].to_numpy(float)
    indices = np.flatnonzero(np.isnan(observed).any(axis=0))
    np.testing.assert_array_equal(indices, imputer.indicator_.features_)
    completed = np.where(np.isnan(observed), medians, observed)
    with_indicators = np.column_stack([completed, np.isnan(observed[:, indices]).astype(float)])
    np.testing.assert_allclose(scaler.mean_, with_indicators.mean(axis=0), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(scaler.var_, with_indicators.var(axis=0), atol=1e-10, rtol=1e-10)
    assert int(np.asarray(scaler.n_samples_seen_).max()) == len(train)
    encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
    for i, c in enumerate(cats):
        assert encoder.categories_[i].tolist() == sorted(train[c].fillna("__MISSING__").unique())
    plot = json.loads((folder / "plot_data.json").read_text())
    for split, d in parts.items():
        assert plot["partition_rows"][split] == len(d)
        np.testing.assert_allclose(plot["target_prevalence_percent"][split], 100*d.target.mean())
        assert plot["nonfinite_after_percent"][split] == 0
    for c, value in plot["missing_train_percent"].items():
        np.testing.assert_allclose(value, 100*train[c].isna().mean())
    for i, c in enumerate(nums):
        np.testing.assert_allclose(plot["numeric_train_std"][c], train_x[:, i].std(), atol=1e-12)
    c = plot["histogram_feature"]
    for label, values in (("observed", train[c].dropna().to_numpy(float)), ("imputed", completed[:, nums.index(c)])):
        counts = np.histogram(values, plot["histogram_edges"])[0]
        np.testing.assert_array_equal(counts, plot[f"{label}_counts"])
        assert len(values) == plot[f"{label}_denominator"] == counts.sum()
    return {"passed": True, "source_rows": len(raw), "partitions": report,
            "training_only_statistics_independently_verified": True, "all_source_rows_retained_once": True,
            "disjoint_records": True, "disjoint_patients": True if group_sets else "not applicable",
            "all_plotted_counts_and_rates_recomputed": True}


def confirm_mimic():
    root = ROOT / "cross_dataset/four_conditions"
    prior = json.loads((AUDIT / "mimic_validation.json").read_text())
    assert prior["passed"]
    count = 0
    for direction in ("mimic_to_eicu", "eicu_to_mimic"):
        folder = ROOT / "preprocessing/outputs/cross_dataset_baseline" / direction
        record = json.loads((folder / "transformation_validation.json").read_text())
        for full_path, expected in record["input_csv_sha256"].items():
            path = root.joinpath(*full_path.replace("\\", "/").split("/")[-2:])
            assert digest(path) == expected
        for split, part in record["partitions"].items():
            for file, expected in part["sha256"].items():
                assert digest(folder / file) == expected
                count += 1
            with np.load(folder / f"X_{split}.npz", allow_pickle=False) as z:
                assert np.isfinite(z["X"]).all() and z["X"].shape == (part["rows"], part["features"])
    clinical = json.loads((root / "mimic/clinical_preprocessing_validation.json").read_text())
    for file, expected in clinical["output_sha256"].items():
        assert digest(root / "mimic" / file) == expected
    return {"passed": True, "matrix_label_metadata_artifacts_checked": count,
            "inputs_and_derived_clinical_files_unchanged": True,
            "previous_independent_recomputation_still_applicable": True,
            "raw_archives_restreamed": False}


def main():
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "data_modified": False, "original_experimental_test_metrics": "not an independent evaluation"}
    try:
        result["datasets"] = {}
        for name in NAMES:
            print(f"Checking saved {name} files", flush=True)
            result["datasets"][name] = confirm_baseline(name)
        print("Checking saved MIMIC files", flush=True)
        result["mimic"] = confirm_mimic()
        result["passed"] = True
    finally:
        (AUDIT / "confirmation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CONFIRMED: all four fixed baselines and saved MIMIC preprocessing passed", flush=True)


if __name__ == "__main__":
    main()
