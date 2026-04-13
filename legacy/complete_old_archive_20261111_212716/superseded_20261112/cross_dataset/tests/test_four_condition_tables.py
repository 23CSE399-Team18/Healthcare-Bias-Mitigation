"""Boundary/version and group-separation checks for research diagnosis tables."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_four_condition_tables import code_flags, hospital_split, patient_split, eicu_code_lookup, pd


class FourConditionLabels(unittest.TestCase):
    def test_code_boundaries_versions_and_gestational_exclusions(self):
        records = [
            ("410.90", 9, "heart"), ("I25.10", 10, "heart"),
            ("401.9", 9, "hypertension"), ("I15.9", 10, "hypertension"),
            ("249.00", 9, "diabetes"), ("250.00", 9, "diabetes"),
            ("E09.9", 10, "diabetes"), ("E11.9", 10, "diabetes"),
            ("460", 9, "respiratory"), ("519.9", 9, "respiratory"), ("J96.00", 10, "respiratory"),
            ("459.9", 9, None), ("520.0", 9, None), ("I50.9", 10, None),
            ("O24.4", 10, None), ("648.0", 9, None), ("I16.9", 10, None),
            ("E849.9", 9, None), ("25000", 10, None), ("E119", 9, None), (None, 10, None),
        ]
        flags = code_flags(pd.Series([r[0] for r in records]), pd.Series([r[1] for r in records]))
        conditions = ["heart", "respiratory", "diabetes", "hypertension"]
        for idx, (_, _, expected) in enumerate(records):
            matched = [name for name in conditions if flags.loc[idx, name]]
            self.assertEqual(matched, [] if expected is None else [expected], records[idx])

    def test_mixed_eicu_tokens_with_duplicate_row_indices(self):
        codes = pd.Series(["250.00", "E11.9", "401.9", "I10"], index=[0, 0, 1, 1])
        versions = pd.Series([9, 10, 9, 10], index=codes.index)
        parsed = code_flags(codes, versions).groupby(level=0).max()
        self.assertTrue(parsed.loc[0, "diabetes"])
        self.assertTrue(parsed.loc[1, "hypertension"])
        self.assertFalse(parsed.loc[0, "hypertension"])

    def test_actual_dictionary_parent_recognition_and_version_ambiguity(self):
        dictionary = pd.DataFrame({"icd_code": ["E8499", "E119", "25000", "78720", "J9600"],
                                   "icd_version": [9, 10, 9, 9, 10]})
        lookup, exact = eicu_code_lookup(dictionary)
        self.assertTrue(lookup.loc["7872", "valid_code"])
        self.assertNotIn("7872", exact)
        self.assertFalse(lookup.loc["E8499", "diabetes"])
        self.assertTrue(lookup.loc["E11", "diabetes"])
        self.assertTrue(lookup.loc["250", "diabetes"])
        self.assertNotIn("J1A", lookup.index)
        self.assertNotIn("R999999", lookup.index)

    def test_patient_and_hospital_split_prevents_repeated_patient_leakage(self):
        rows = [{"hospital_id": str(hospital), "patient_id": f"p{hospital}"} for hospital in range(100)]
        # A patient visiting every hospital must be purged across partitions.
        rows += [{"hospital_id": str(hospital), "patient_id": "crosses-hospitals"} for hospital in range(100)]
        frame = pd.DataFrame(rows)
        result, audit = hospital_split(frame)
        self.assertEqual(result.groupby("hospital_id").split.nunique().max(), 1)
        self.assertEqual(result.groupby("patient_id").split.nunique().max(), 1)
        self.assertNotIn("crosses-hospitals", set(result.patient_id))
        self.assertEqual(audit["purged_patients"], 1)
        self.assertEqual(audit["purged_rows"], 100)
        repeated = pd.concat([frame, frame], ignore_index=True)
        patient_result, _ = patient_split(repeated)
        self.assertEqual(patient_result.groupby("patient_id").split.nunique().max(), 1)
        self.assertEqual(list(patient_result.split), list(patient_split(repeated)[0].split))


if __name__ == "__main__":
    unittest.main()
