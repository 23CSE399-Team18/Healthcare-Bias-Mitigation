"""Independently validate actual exported four-condition research tables.

Reads source-linked labels and clinical tables, recomputes eICU structured
history evidence, verifies all eight exports and split membership, and writes
an aggregate validation report. No clinical records are printed or changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "_data_reader"))
import pandas as pd
import numpy as np
from eicu_history_evidence import load_eicu_history_evidence

OUT = ROOT / "four_conditions"
CONDITIONS = ("heart", "respiratory", "diabetes", "hypertension")
NUMERIC = [f"{vital}_{statistic}" for vital in ("heart_rate", "resp_rate", "spo2")
           for statistic in ("mean", "min", "max")]
FEATURES = NUMERIC + ["age_group", "sex"]
META = ["meta_source", "meta_patient_id", "meta_stay_id", "meta_domain_hospitalid",
        "meta_split", "sens_age_group", "sens_sex", "sens_race_ethnicity"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_name(identifier, domain):
    raw = hashlib.sha256(f"four-condition-v1-{domain}:{identifier}".encode()).digest()
    fraction = int.from_bytes(raw[:8], "big") / 2**64
    return "train" if fraction < .70 else "validation" if fraction < .85 else "test"


def check_features(actual, expected, message):
    require(actual.index.equals(expected.index), message + ": stay alignment")
    require(np.allclose(actual[NUMERIC].to_numpy(), expected[NUMERIC].to_numpy(),
                        equal_nan=True, atol=1e-12, rtol=1e-12), message + ": numeric values")
    for column in ("age_group", "sex"):
        require(actual[column].fillna("<missing>").equals(expected[column].fillna("<missing>")),
                message + ": " + column)


def expected_mimic():
    directory = OUT / "mimic"
    labels = pd.read_csv(directory / "admission_diagnosis_labels.csv")
    label_audit = json.loads((directory / "diagnosis_label_audit.json").read_text())
    require(digest(directory / "admission_diagnosis_labels.csv") == label_audit["output_label_sha256"],
            "MIMIC source-linked labels changed after source audit")
    require(labels.hadm_id.is_unique and labels.diagnosis_code_rows.gt(0).all(),
            "MIMIC documented admission label keys/eligibility")
    clinical = pd.read_csv(directory / "clinical_24h_features.csv").set_index("stay_id")
    metadata = pd.read_csv(directory / "cohort_metadata.csv")
    require(metadata.stay_id.is_unique and clinical.index.is_unique, "MIMIC input stay keys")
    require(set(metadata.stay_id) == set(clinical.index), "MIMIC clinical/metadata population differs")
    expected = metadata.merge(labels, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    expected["expected_split"] = expected.patient_id.map(lambda value: split_name(value, "patient"))
    expected = expected.set_index("stay_id").sort_index()
    return expected, clinical.sort_index(), {
        "eligible_clinical_stays": len(clinical), "excluded_no_documented_diagnoses": len(clinical) - len(expected),
        "source_labels_sha256": digest(directory / "admission_diagnosis_labels.csv")}


def expected_eicu():
    directory = OUT / "eicu"
    clinical = pd.read_csv(ROOT / "prepared/eicu_24h_features.csv", dtype={"hospital_id": "string"})
    labels = pd.read_csv(directory / "stay_diagnosis_labels.csv")
    label_audit = json.loads((directory / "diagnosis_label_audit.json").read_text())
    require(digest(directory / "stay_diagnosis_labels.csv") == label_audit["output_label_sha256"],
            "eICU source-linked labels changed after source audit")
    require(labels.patientunitstayid.is_unique and labels.diagnosis_code_rows.gt(0).all(),
            "eICU documented stay label keys/eligibility")
    history, history_audit = load_eicu_history_evidence()
    union = labels.merge(history, on="patientunitstayid", how="left", validate="one_to_one")
    saved_union = pd.read_csv(directory / "stay_diagnosis_and_history_labels.csv").set_index("patientunitstayid")
    require(set(saved_union.index) == set(labels.patientunitstayid), "eICU history-only controls entered label table")
    for condition in CONDITIONS:
        expected_flag = union[condition].eq(1) | union["history_" + condition].eq(1)
        aligned = saved_union.loc[union.patientunitstayid]
        require(np.array_equal(aligned[condition].to_numpy(), expected_flag.astype(int).to_numpy()),
                "eICU saved primary union differs from independent source-history union: " + condition)
        require(np.array_equal(aligned["icd_" + condition].to_numpy(), union[condition].to_numpy()),
                "eICU strict ICD sensitivity flag differs: " + condition)
        union[condition] = expected_flag.astype("int8")
    union = union.rename(columns={"patientunitstayid": "stay_id"})
    expected = clinical.merge(union, on="stay_id", how="inner", validate="one_to_one")
    before_purge = len(expected)
    expected["expected_split"] = expected.hospital_id.map(lambda value: split_name(value, "hospital"))
    crossing = expected.groupby("patient_id").expected_split.nunique()
    cross_patients = set(crossing[crossing.gt(1)].index)
    expected = expected[~expected.patient_id.isin(cross_patients)].set_index("stay_id").sort_index()
    return expected, clinical.set_index("stay_id").sort_index(), {
        "eligible_clinical_stays": len(clinical), "diagnosis_documented_before_purge": before_purge,
        "excluded_no_documented_diagnoses": len(clinical) - before_purge,
        "cross_partition_patients_purged": len(cross_patients), "cross_partition_stays_purged": before_purge - len(expected),
        "history_source_recomputed_from_verified_gzip": history_audit["source_gzip_read_to_eof"],
        "history_source_sha256": history_audit["source_sha256"],
        "source_labels_sha256": digest(directory / "stay_diagnosis_labels.csv")}


def validate_source(source, expected, clinical, details):
    report = dict(details)
    report["conditions"] = {}
    reference = None
    for condition in CONDITIONS:
        path = OUT / source / f"{condition}_model_ready.csv"
        frame = pd.read_csv(path, dtype={"meta_domain_hospitalid": "string"})
        require(set(frame) == set(META + FEATURES + ["target"]), source + ": predictor/metadata allowlist")
        require(frame.meta_stay_id.is_unique, source + ": repeated stay")
        require(frame.target.notna().all() and frame.target.isin([0, 1]).all(), source + ": missing/nonbinary target")
        frame = frame.set_index("meta_stay_id").sort_index()
        require(set(frame.index) == set(expected.index), source + ": final eligibility differs from source records")
        wanted = expected.loc[frame.index]
        require(np.array_equal(frame.target.to_numpy(), wanted[condition].to_numpy()),
                source + ": labels differ from source linkage for " + condition)
        require(frame.meta_source.eq(source).all(), source + ": mixed source marker")
        require(np.array_equal(frame.meta_patient_id.to_numpy(), wanted.patient_id.to_numpy()), source + ": patient linkage")
        require(np.array_equal(frame.meta_domain_hospitalid.astype(str).to_numpy(), wanted.hospital_id.astype(str).to_numpy()),
                source + ": hospital linkage")
        require(np.array_equal(frame.meta_split.to_numpy(), wanted.expected_split.to_numpy()), source + ": split assignment")
        require(frame.groupby("meta_patient_id").meta_split.nunique().max() == 1, source + ": patient split overlap")
        if source == "eicu":
            require(frame.groupby("meta_domain_hospitalid").meta_split.nunique().max() == 1, "eICU hospital split overlap")
        require(set(frame.meta_split) == {"train", "validation", "test"}, source + ": missing partition")
        # Index names differ between source keys and export metadata keys only.
        source_features = clinical.loc[frame.index, FEATURES].copy()
        source_features.index.name = frame.index.name
        check_features(frame[FEATURES], source_features, source + ": source clinical measurements")
        require(frame.sens_age_group.fillna("<missing>").equals(frame.age_group.fillna("<missing>")), source + ": sensitive age alignment")
        require(frame.sens_sex.fillna("<missing>").equals(frame.sex.fillna("<missing>")), source + ": sensitive sex alignment")
        require(np.array_equal(frame.sens_race_ethnicity.fillna("<missing>").to_numpy(),
                               wanted.race_group.fillna("<missing>").to_numpy()), source + ": sensitive race alignment")
        non_target = frame.drop(columns="target")
        if reference is None:
            reference = non_target
        else:
            pd.testing.assert_frame_equal(non_target, reference)
        report["conditions"][condition] = {
            "rows": len(frame), "positive": int(frame.target.sum()), "not_documented": int(frame.target.eq(0).sum()),
            "sha256": digest(path), "source_labels_match": True,
            "split_targets": frame.groupby("meta_split").target.agg(["count", "sum"]).to_dict("index")}
    report.update(rows=len(reference), patients=int(reference.meta_patient_id.nunique()),
                  hospitals=int(reference.meta_domain_hospitalid.nunique()),
                  same_features_metadata_and_rows_all_four_conditions=True,
                  exact_documented_population_and_split_purge=True, no_patient_split_overlap=True,
                  no_hospital_split_overlap=source == "eicu", shared_predictors=FEATURES,
                  source_measurements_preserved=True, labels_or_outcomes_in_predictors=False)
    return report


def main():
    report_path = OUT / "independent_validation.json"
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "validation": "Actual exports, source-linked labels, independently recomputed history union and split eligibility",
              "models_trained": False}
    try:
        report["sources"] = {}
        for source, builder in (("mimic", expected_mimic), ("eicu", expected_eicu)):
            expected, clinical, details = builder()
            report["sources"][source] = validate_source(source, expected, clinical, details)
        report.update(passed=True, completed_utc=datetime.now(timezone.utc).isoformat())
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"passed": True, "report": str(report_path),
                          "rows": {source: value["rows"] for source, value in report["sources"].items()}}))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
