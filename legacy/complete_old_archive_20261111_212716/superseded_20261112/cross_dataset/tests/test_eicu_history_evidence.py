"""Check clinical-concept boundaries and unknowns in structured history."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eicu_history_evidence import history_condition

BASE = "notes/Progress Notes/Past History/Organ Systems/"


class StructuredHistoryEvidence(unittest.TestCase):
    def test_explicit_chronic_disease_positives(self):
        examples = [
            ("Cardiovascular (R)/Hypertension Requiring Treatment/hypertension requiring treatment",
             "hypertension requiring treatment", "hypertension"),
            ("Endocrine (R)/Non-Insulin Dependent Diabetes/non-medication dependent",
             "non-medication dependent", "diabetes"),
            ("Endocrine (R)/Insulin Dependent Diabetes/insulin dependent diabetes",
             "insulin dependent diabetes", "diabetes"),
            ("Cardiovascular (R)/Myocardial Infarction/MI - remote", "MI - remote", "heart"),
            ("Cardiovascular (R)/Angina/angina - class II", "angina - class II", "heart"),
            ("Pulmonary/COPD/COPD  - no limitations", "COPD  - no limitations", "respiratory"),
            ("Pulmonary/Asthma/asthma", "asthma", "respiratory"),
        ]
        for path, value, expected in examples:
            self.assertEqual(history_condition(BASE + path, value), expected)

    def test_proxies_unknowns_and_nonmatching_families_do_not_label(self):
        examples = [
            ("Cardiovascular (R)/Coronary Artery Bypass/CABG - remote", "CABG - remote"),
            ("Cardiovascular (R)/Pulmonary Embolism/pulmonary embolism - remote", "pulmonary embolism - remote"),
            ("Pulmonary/Home Oxygen/home oxygen", "home oxygen"),
            ("Pulmonary/Respiratory Failure/respiratory failure - remote", "respiratory failure - remote"),
            ("Pulmonary/Sarcoidosis/sarcoidosis", "sarcoidosis"),
            ("Pulmonary/COPD/Not Obtainable", "Not Obtainable"),
            ("Pulmonary/COPD/COPD - severe", "No Health Problems"),
        ]
        for path, value in examples:
            self.assertIsNone(history_condition(BASE + path, value))
        self.assertIsNone(history_condition(BASE + "Pulmonary/COPD/COPD - severe",
                                            "COPD - severe", "Not Obtainable"))


if __name__ == "__main__":
    unittest.main()
