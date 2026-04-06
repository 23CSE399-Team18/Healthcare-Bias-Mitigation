"""Scientific boundary/leakage tests using tiny, clearly artificial fixtures."""
from pathlib import Path
import gzip
import hashlib
import json
import tempfile
import unittest

from prepare_mimic_clinical_features import (pd, np, ITEMS, VITALS, BOUNDS,
    BASELINE_FEATURES, EXTENDED_FEATURES, filter_events, add_events,
    HourlyAccumulator, validate_items, prepare)


class MimicClinicalTests(unittest.TestCase):
    def cohort(self):
        return pd.DataFrame({"_intime": pd.to_datetime(["2150-01-01 00:00:00"])}, index=[1])

    def events(self, changes):
        base = {"stay_id": 1, "itemid": 220045, "charttime": "2150-01-01 01:00:00",
                "storetime": "2150-01-01 01:05:00", "valuenum": 75, "valueuom": "bpm"}
        return pd.DataFrame([{**base, **value} for value in changes])

    def test_documentation_window_units_and_unrelated_stays(self):
        events = self.events([
            {}, {"charttime": "2149-12-31 23:59:00"},
            {"charttime": "2150-01-02 00:00:00", "storetime": "2150-01-02 00:00:00"},
            {"charttime": "2150-01-02 00:00:01"}, {"charttime": None},
            {"storetime": None}, {"storetime": "2150-01-02 00:00:01"},
            {"valueuom": "mmHg"}, {"valueuom": None}, {"stay_id": 999}, {"itemid": 123},
        ])
        selected, offsets, audit = filter_events(events, self.cohort())
        self.assertEqual(list(selected.index), [0, 2, 8])
        self.assertEqual(list(offsets), [60, 1440, 60])
        self.assertEqual(audit["reviewed_item_and_eligible_stay_rows"], 9)
        self.assertEqual(audit["excluded_unknown_charttime"], 1)
        self.assertEqual(audit["excluded_charttime_outside_first24h"], 2)
        self.assertEqual(audit["excluded_unknown_storetime"], 1)
        self.assertEqual(audit["excluded_storetime_after24h"], 1)
        self.assertEqual(audit["excluded_explicit_unit_conflict"], 1)
        self.assertEqual(audit["retained_rows_unit_inferred_from_dictionary"], 1)

    def test_temperature_conversion_distinct_pressure_routes_and_bounds(self):
        events = self.events([
            {"itemid": 223761, "valueuom": "\u00b0F", "valuenum": 98.6},
            {"itemid": 223762, "valueuom": "\u00b0C", "valuenum": 37.0},
            {"itemid": 220179, "valueuom": "mmHg", "valuenum": 120.0},
            {"itemid": 220050, "valueuom": "mmHg", "valuenum": 130.0},
            {"itemid": 220179, "valueuom": "mmHg", "valuenum": 400.0},
        ])
        selected, offsets, _ = filter_events(events, self.cohort())
        accumulator = HourlyAccumulator([1], vitals=VITALS, bounds=BOUNDS)
        add_events(accumulator, selected, offsets)
        result, observed = accumulator.finish()
        self.assertAlmostEqual(result.loc[1, "temperature_mean"], 37.0)
        self.assertEqual(result.loc[1, "nibp_systolic_mean"], 120.0)
        self.assertEqual(result.loc[1, "arterial_systolic_mean"], 130.0)
        self.assertEqual(accumulator.accepted["nibp_systolic"], 1)
        self.assertTrue(np.isnan(result.loc[1, "spo2_mean"]))
        self.assertEqual(observed.loc[1, "spo2_observed_hours"], 0)

    def test_dictionary_label_or_unit_changes_are_rejected(self):
        items = pd.DataFrame([{"itemid": i, "label": value[1], "unitname": value[2]}
                              for i, value in ITEMS.items()])
        self.assertTrue(validate_items(items).label_and_unit_verified.all())
        items.loc[0, "unitname"] = "mmHg"
        with self.assertRaises(ValueError):
            validate_items(items)

    def make_sources(self, directory, corrupt_crc=False):
        tables = {
            "patients.csv.gz": pd.DataFrame({"subject_id": [10], "gender": ["F"],
                                              "anchor_age": [60], "anchor_year": [2150]}),
            "admissions.csv.gz": pd.DataFrame({"subject_id": [10], "hadm_id": [20],
                "admittime": ["2150-01-01 00:00:00"], "dischtime": ["2150-01-05 00:00:00"],
                "deathtime": [None], "race": ["WHITE"], "hospital_expire_flag": [0]}),
            "icustays.csv.gz": pd.DataFrame({"subject_id": [10], "hadm_id": [20], "stay_id": [1],
                "intime": ["2150-01-01 00:00:00"], "outtime": ["2150-01-04 00:00:00"]}),
            "d_items.csv.gz": pd.DataFrame([{"itemid": i, "label": value[1], "unitname": value[2]}
                                            for i, value in ITEMS.items()]),
            "chartevents.csv.gz": self.events([{}, {"valuenum": 95, "storetime": None},
                {"itemid": 223761, "valueuom": "\u00b0F", "valuenum": 98.6}]),
        }
        from prepare_mimic_clinical_features import REQUIRED
        sums = []
        for relative in REQUIRED:
            name = Path(relative).name
            content = gzip.compress(tables[name].to_csv(index=False).encode())
            if name == "chartevents.csv.gz" and corrupt_crc:
                content = content[:-8] + bytes([content[-8] ^ 255]) + content[-7:]
            (directory / name).write_bytes(content)
            sums.append(hashlib.sha256(content).hexdigest() + "  " + relative)
        (directory / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="ascii")

    def test_real_pipeline_keeps_ids_outcomes_and_labels_out_of_predictors(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            self.make_sources(root)
            result = prepare(root, root / "out", chunksize=1)
            table = pd.read_csv(root / "out/clinical_24h_features.csv")
            metadata = pd.read_csv(root / "out/cohort_metadata.csv")
            self.assertEqual(table.heart_rate_mean.iloc[0], 75)
            self.assertAlmostEqual(table.temperature_mean.iloc[0], 37.0)
            self.assertEqual(table.columns.tolist(), ["stay_id"] + BASELINE_FEATURES + EXTENDED_FEATURES)
            self.assertFalse(set(table.columns) & {"hadm_id", "subject_id", "hospital_mortality", "icd_code"})
            self.assertEqual(metadata.hadm_id.iloc[0], 20)
            self.assertEqual(result["event_audit"]["excluded_unknown_storetime"], 1)
            self.assertTrue(result["chartevents_gzip_crc_verified_by_full_stream"])

    def test_gzip_crc_error_fails_and_does_not_publish_feature_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            self.make_sources(root, corrupt_crc=True)
            with self.assertRaises((gzip.BadGzipFile, EOFError)):
                prepare(root, root / "out", chunksize=1)
            self.assertFalse((root / "out/clinical_24h_features.csv").exists())
            state = json.loads((root / "out/clinical_preprocessing_validation.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertFalse(state["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
