"""Independently compare prepared membership/labels with original patient records."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "_data_reader"))
import pandas as pd
from prepare_cross_dataset import FEATURES


def main():
    path = ROOT / "prepared/eicu_24h_features.csv"
    frame = pd.read_csv(path)
    expected = {}
    source = ROOT.parent / "resp_physionet/resp/official_verified/patient.csv.gz"
    with gzip.open(source, "rt", newline="") as inp:
        for row in csv.DictReader(inp):
            raw = row["age"].replace(" ", "")
            try:
                age = 90.0 if raw == ">89" else float(raw)
            except ValueError:
                continue
            if (age >= 18 and row["uniquepid"] and row["hospitaldischargestatus"] in ["Alive", "Expired"]
                and float(row["unitdischargeoffset"]) > 1440 and float(row["hospitaldischargeoffset"]) > 1440):
                expected[int(row["patientunitstayid"])] = (
                    row["uniquepid"], row["hospitalid"], int(row["hospitaldischargestatus"] == "Expired"))
    assert set(frame.stay_id) == set(expected)
    for row in frame.itertuples():
        match = expected[row.stay_id]
        assert row.patient_id == "eicu:" + match[0]
        assert int(row.hospital_id) == int(match[1])
        assert row.hospital_mortality == match[2]
    assert frame.stay_id.is_unique and frame.age_group.notna().all()
    assert not set(FEATURES).intersection(["hospital_mortality", "patient_id", "stay_id", "hospital_id", "race_group"])
    assert not set(frame.columns).intersection(["_age", "age", "unitdischargeoffset", "hospitaldischargeoffset",
                                               "hospitaldischargestatus", "temperature_mean"])
    audit = pd.read_csv(ROOT / "prepared/eicu_observation_audit.csv")
    temperature_missing = int(audit.temperature_observed_hours.eq(0).sum())
    result = {"passed": True, "independent_source_membership_check": True, "rows": len(frame),
              "patients": int(frame.patient_id.nunique()), "hospitals": int(frame.hospital_id.nunique()),
              "deaths": int(frame.hospital_mortality.sum()), "unknown_age_not_mapped_to_topcode": True,
              "no_outcome_or_future_columns_in_explicit_predictors": True,
              "temperature_missing_and_excluded_from_baseline": temperature_missing,
              "feature_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "mimic_evaluation_pending": not (ROOT / "results/generalization_metrics.csv").exists()}
    (ROOT / "prepared/eicu_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
