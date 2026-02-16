"""Small scientific-integrity fixtures; no real patient data is used in tests."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from prepare_cross_dataset import (np, pd, HourlyAccumulator, age_group, eicu_cohort,
                                    mimic_cohort, FEATURES, NUMERIC_FEATURES)
from evaluate_generalization import patient_split, create_model, evaluate_direction


class ScientificIntegrityTests(unittest.TestCase):
    def test_window_and_bounds(self):
        accumulator = HourlyAccumulator([1])
        accumulator.add([1] * 7, [-1, 0, 60, 1440, 1441, 120, 180],
                        {"heart_rate": [99, 60, 120, 90, 99, 300, -10]})
        values, audit = accumulator.finish()
        self.assertEqual(audit.loc[1, "heart_rate_observed_hours"], 3)
        self.assertEqual(values.loc[1, "heart_rate_mean"], 90)
        self.assertEqual(accumulator.accepted["heart_rate"], 3)

    def test_equal_hour_weights_not_raw_observation_weights(self):
        accumulator = HourlyAccumulator([1])
        accumulator.add([1] * 4, [0, 10, 20, 60], {"heart_rate": [60, 60, 60, 120]})
        values, _ = accumulator.finish()
        self.assertEqual(values.loc[1, "heart_rate_mean"], 90)

    def test_chunking_does_not_change_aggregation(self):
        complete = HourlyAccumulator([1, 2])
        chunked = HourlyAccumulator([1, 2])
        ids, offsets, data = [1, 1, 2, 2], [0, 20, 0, 60], [60, 90, 70, 80]
        complete.add(ids, offsets, {"heart_rate": data})
        chunked.add(ids[:2], offsets[:2], {"heart_rate": data[:2]})
        chunked.add(ids[2:], offsets[2:], {"heart_rate": data[2:]})
        pd.testing.assert_frame_equal(complete.finish()[0], chunked.finish()[0])

    def test_source_age_topcodes_share_broad_band(self):
        bands = age_group(pd.Series([18, 39, 40, 64, 65, 79, 80, 90, 91]))
        self.assertEqual(list(bands), ["18-39", "18-39", "40-64", "40-64", "65-79", "65-79", "80+", "80+", "80+"])

    def test_eicu_landmark_and_unknown_age(self):
        base = {"patientunitstayid": 1, "uniquepid": "p1", "hospitalid": 1, "age": "60",
                "gender": "Female", "ethnicity": "Caucasian", "unitdischargeoffset": 2880,
                "hospitaldischargeoffset": 4000, "hospitaldischargestatus": "Alive"}
        records = [base, {**base, "patientunitstayid": 2, "age": ""},
                   {**base, "patientunitstayid": 3, "unitdischargeoffset": 1440},
                   {**base, "patientunitstayid": 4, "age": "17"},
                   {**base, "patientunitstayid": 5, "age": "> 89", "hospitaldischargestatus": "Expired"},
                   {**base, "patientunitstayid": 6, "hospitaldischargestatus": "Unknown"}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "patient.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            cohort, _ = eicu_cohort(path)
        self.assertEqual(set(cohort.stay_id), {1, 5})
        self.assertEqual(cohort.loc[5, "hospital_mortality"], 1)
        self.assertEqual(cohort.loc[5, "age_group"], "80+")

    def test_mimic_death_before_landmark_is_excluded(self):
        patients = pd.DataFrame({"subject_id": [1, 2], "gender": ["F", "M"],
                                 "anchor_age": [65, 91], "anchor_year": [2150, 2150]})
        admissions = pd.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22],
            "admittime": ["2150-01-01"] * 2, "dischtime": ["2150-01-05"] * 2,
            "deathtime": ["2150-01-01 12:00:00", "2150-01-04"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN"], "hospital_expire_flag": [1, 1]})
        stays = pd.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "stay_id": [111, 222],
                              "intime": ["2150-01-01"] * 2, "outtime": ["2150-01-05"] * 2})
        cohort, _ = mimic_cohort(patients, admissions, stays)
        self.assertEqual(list(cohort.stay_id), [222])
        self.assertEqual(cohort.iloc[0].age_group, "80+")

    def test_repeated_patient_stays_do_not_cross_split(self):
        frame = pd.DataFrame({"patient_id": [f"p{i}" for i in range(100) for _ in range(3)],
                              "stay_id": range(300)})
        train, test = patient_split(frame)
        self.assertFalse(set(train.patient_id) & set(test.patient_id))
        self.assertEqual(len(train) + len(test), 300)
        again = patient_split(frame)
        self.assertEqual(list(train.stay_id), list(again[0].stay_id))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed yet")
    def test_target_evaluation_cannot_change_source_imputation(self):
        source = pd.DataFrame({key: [10.0, 20.0, np.nan, 20.0] for key in NUMERIC_FEATURES})
        source["age_group"] = ["18-39", "40-64", "65-79", "80+"]
        source["sex"] = ["Female", "Male", "Female", "Male"]
        model = create_model()
        model.fit(source[FEATURES], [0, 1, 0, 1])
        imputer = model.named_steps["prepare"].named_transformers_["numeric"].named_steps["imputer"]
        before = imputer.statistics_.copy()
        target = source.copy()
        target[NUMERIC_FEATURES] = 1e9
        model.predict_proba(target[FEATURES])
        np.testing.assert_array_equal(before, imputer.statistics_)
        np.testing.assert_array_equal(before, np.full(len(NUMERIC_FEATURES), 20.0))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed yet")
    def test_direction_fits_only_source_patients_and_evaluates_target(self):
        def fixture(source, count, start):
            frame = pd.DataFrame({key: np.arange(count, dtype=float) % 30 + 50 for key in NUMERIC_FEATURES},
                                 index=np.arange(start, start + count))
            frame["patient_id"] = [f"{source}:p{i // 2}" for i in range(count)]
            frame["stay_id"] = np.arange(start, start + count)
            frame["source"] = source
            frame["hospital_mortality"] = np.arange(count) % 3 == 0
            frame["age_group"] = "40-64"
            frame["sex"] = "Female"
            frame["race_group"] = "Unknown"
            return frame
        source, target = fixture("source", 300, 0), fixture("target", 120, 1000)
        fitted_indices = []
        real_model = create_model()
        class SpyModel:
            def fit(self, x, y):
                fitted_indices.extend(x.index)
                real_model.fit(x, y)
                return self
            def predict_proba(self, x):
                return real_model.predict_proba(x)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch("evaluate_generalization.create_model", return_value=SpyModel()):
                metrics, groups, split = evaluate_direction(source, target, "source", "target", 42,
                                                            Path(directory), False)
        expected_train, expected_test = patient_split(source)
        self.assertEqual(set(fitted_indices), set(expected_train.index))
        self.assertFalse(set(fitted_indices).intersection(target.index))
        self.assertEqual(metrics[0]["n"], len(expected_test))
        self.assertEqual(metrics[1]["n"], len(target))
        self.assertEqual(split["target_rows_used_for_fit_or_tuning"], 0)
        self.assertEqual(split["source_patient_overlap"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
