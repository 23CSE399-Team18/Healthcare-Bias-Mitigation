"""Check source-only fitting, feature exclusions and saved row alignment."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'preprocessing/code'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'preprocessing/code'))
from prepare_four_condition_baseline import (
    CONDITIONS, METADATA, SENSITIVE, FEATURES, NUMERIC_FEATURES,
    TARGETS, fit_transform_direction, load_source_tables,
    transformer_statistics, write_direction, np, pd,
)


def fixture(source="mimic", external=False):
    size = 8
    frame = pd.DataFrame({name: np.arange(size, dtype=float) + offset
                          for offset, name in enumerate(NUMERIC_FEATURES)})
    frame.loc[[0, 2], NUMERIC_FEATURES[0]] = np.nan
    frame.loc[:3, NUMERIC_FEATURES[-1]] = np.nan
    frame["age_group"] = ["18-39", "40-64", "18-39", "40-64", "65-79", "80+", "18-39", "65-79"]
    frame["sex"] = ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"]
    frame.loc[1, "sex"] = np.nan
    frame["meta_source"] = source
    frame["meta_patient_id"] = [source + "-p" + str(i) for i in range(size)]
    frame["meta_stay_id"] = [source + "-s" + str(i) for i in range(size)]
    frame["meta_domain_hospitalid"] = [source + "-h" + str(i) for i in range(size)]
    frame["meta_split"] = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    frame["sens_age_group"] = frame.age_group
    frame["sens_sex"] = frame.sex
    frame["sens_race_ethnicity"] = ["GroupA", "GroupB"] * 4
    for offset, condition in enumerate(CONDITIONS):
        frame["target_" + condition] = (np.arange(size) + offset) % 2
    if external:
        frame[NUMERIC_FEATURES] += 1000
        frame["age_group"] = "unseen-external-age"
    return frame


class FourConditionBaseline(unittest.TestCase):
    def test_external_heldout_outliers_and_labels_do_not_change_fit(self):
        source, external = fixture(), fixture("eicu", external=True)
        transformer, names, partitions = fit_transform_direction(source, external)
        changed_source, changed_external = source.copy(), external.copy()
        heldout = changed_source.meta_split.ne("train")
        changed_source.loc[heldout, NUMERIC_FEATURES] += 10**9
        changed_source.loc[heldout, "sex"] = "unseen-heldout-category"
        changed_source.loc[:, TARGETS] = 1 - changed_source[TARGETS]
        changed_source["sens_race_ethnicity"] = "metadata-must-not-be-a-feature"
        changed_external.loc[:, NUMERIC_FEATURES] += 10**9
        changed_external["sex"] = "unseen-external-category"
        changed_external.loc[:, TARGETS] = 1 - changed_external[TARGETS]
        second, second_names, second_partitions = fit_transform_direction(changed_source, changed_external)
        self.assertEqual(transformer_statistics(transformer), transformer_statistics(second))
        self.assertEqual(names, second_names)
        np.testing.assert_array_equal(partitions["train"]["X"], second_partitions["train"]["X"])
        self.assertEqual(list(transformer.feature_names_in_), FEATURES)
        self.assertFalse(any(token in name for name in names for token in ["meta_", "sens_", "target_"]))
        for partition in second_partitions.values():
            self.assertTrue(np.isfinite(partition["X"]).all())
            self.assertEqual(partition["X"].shape[1], len(names))
        # A training-empty numeric feature remains present and gets a zero fill.
        self.assertEqual(transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[-1], 0)

    def test_saved_matrices_and_four_targets_share_exact_row_order(self):
        source, external = fixture(), fixture("eicu", external=True)
        original = source.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            summary = write_direction("mimic", "eicu", source, external, directory, {})
            for name, record in summary["partitions"].items():
                rows = pd.read_csv(record["row_metadata_path"])
                expected = external if name == "external" else source.loc[source.meta_split.eq(name)]
                self.assertEqual(list(rows.meta_stay_id), list(expected.meta_stay_id))
                with np.load(record["X_path"]) as matrix, np.load(record["y_path"]) as labels:
                    self.assertEqual(len(rows), len(matrix["X"]))
                    for condition in CONDITIONS:
                        np.testing.assert_array_equal(labels[condition], rows["target_" + condition])
        pd.testing.assert_frame_equal(source, original)

    def test_mismatched_condition_features_are_rejected_before_fitting(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "mimic"
            source_dir.mkdir()
            for condition in CONDITIONS:
                single = frame[METADATA + SENSITIVE + FEATURES].copy()
                single["target"] = frame["target_" + condition]
                single.to_csv(source_dir / (condition + "_model_ready.csv"), index=False)
            combined, hashes = load_source_tables(directory, "mimic")
            self.assertEqual(len(combined), len(frame))
            self.assertEqual(len(hashes), 4)
            path = source_dir / "diabetes_model_ready.csv"
            tampered = pd.read_csv(path)
            tampered.loc[0, NUMERIC_FEATURES[1]] += 100
            tampered.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "identical predictors"):
                load_source_tables(directory, "mimic")


if __name__ == "__main__":
    unittest.main()
