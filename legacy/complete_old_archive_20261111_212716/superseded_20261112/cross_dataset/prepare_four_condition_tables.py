"""Create auditable retrospective ICD-record classification tables.

These are research-derived documentation labels, not clinical disease-negative
ground truth or onset prediction. No diagnosis, identifier or outcome is a feature.
No imputers, scalers, models or resampling are fitted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "_data_reader"))
sys.path.insert(0, str(ROOT))
import pandas as pd

from download_mimic import RAW_ROOT, save_json, utc_now
from prepare_cross_dataset import FEATURES
from streaming_sources import load_segment_manifest, read_official_csv

OUT = ROOT / "four_conditions"
CONDITIONS = {
    "heart": {"name": "Recorded ischemic heart disease", "icd9": ["410", "411", "412", "413", "414"],
              "icd10": ["I20", "I21", "I22", "I23", "I24", "I25"]},
    "respiratory": {"name": "Recorded respiratory-system disease", "icd9_range": [460, 519],
                    "icd10_range": ["J00", "J99"]},
    "diabetes": {"name": "Recorded non-gestational diabetes", "icd9": ["249", "250"],
                 "icd10": ["E08", "E09", "E10", "E11", "E13"]},
    "hypertension": {"name": "Recorded systemic hypertension", "icd9": ["401", "402", "403", "404", "405"],
                     "icd10": ["I10", "I11", "I12", "I13", "I15"]},
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize_codes(values):
    return values.astype("string").fillna("").str.strip().str.upper().str.replace(".", "", regex=False)


def code_flags(codes, versions):
    """Version-specific prefixes; prevent numeric/alphanumeric cross-matching."""
    codes = normalize_codes(codes)
    versions = pd.to_numeric(versions, errors="coerce")
    prefix = codes.str[:3]
    nine = versions.eq(9) & codes.str.fullmatch(r"[0-9]{3,5}")
    ten = versions.eq(10) & codes.str.fullmatch(r"[A-Z][0-9][A-Z0-9]{1,5}")
    result = pd.DataFrame(index=codes.index)
    for condition, definition in CONDITIONS.items():
        if condition == "respiratory":
            a = pd.to_numeric(prefix, errors="coerce").between(460, 519)
            b = prefix.str.fullmatch(r"J[0-9]{2}") & prefix.between("J00", "J99")
        else:
            a = prefix.isin(definition["icd9"])
            b = prefix.isin(definition["icd10"])
        result[condition] = ((nine & a) | (ten & b)).fillna(False)
    # V/E supplementary ICD-9 codes count as documentation but do not trigger
    # these four condition families. They must not be misread as ICD-10 E codes.
    result["valid_code"] = ((versions.eq(9) & codes.str.fullmatch(r"(?:[0-9]{3,5}|[VE][0-9]{2,4})")) | ten).fillna(False)
    return result


def partition(key, namespace):
    number = int.from_bytes(hashlib.sha256((namespace + ":" + str(key)).encode()).digest()[:8], "big") / 2**64
    return "train" if number < .70 else "validation" if number < .85 else "test"


def patient_split(frame):
    result = frame.copy()
    mapping = {key: partition(key, "four-condition-v1-patient") for key in result.patient_id.unique()}
    result["split"] = result.patient_id.map(mapping)
    return result, {"method": "Stable patient-group 70/15/15 hash split; one hospital only", "purged_rows": 0}


def hospital_split(frame):
    result = frame.copy()
    mapping = {str(key): partition(key, "four-condition-v1-hospital") for key in result.hospital_id.unique()}
    result["split"] = result.hospital_id.astype(str).map(mapping)
    crossing = result.groupby("patient_id").split.nunique().gt(1)
    purge = set(crossing[crossing].index)
    count = int(result.patient_id.isin(purge).sum())
    result = result[~result.patient_id.isin(purge)].copy()
    return result, {"method": "Whole-hospital 70/15/15 hash split; purge patients crossing partitions",
                    "purged_patients": len(purge), "purged_rows": count}


def verified_mimic(relative, **kwargs):
    path = RAW_ROOT / relative
    _, manifest = load_segment_manifest(path, verify=True)
    if manifest["sha256"] != manifest["expected_sha256"] or not manifest["completed"]:
        raise ValueError("Unverified source: " + relative)
    return read_official_csv(path, **kwargs), {"relative_name": relative, "sha256": manifest["sha256"]}


def prepare_mimic_labels(chunksize):
    dictionary, dictionary_source = verified_mimic("hosp/d_icd_diagnoses.csv.gz", dtype={"icd_code": "string"})
    dictionary["icd_code"] = normalize_codes(dictionary.icd_code)
    if dictionary.duplicated(["icd_code", "icd_version"]).any():
        raise ValueError("Duplicate source diagnosis dictionary keys")
    flags = code_flags(dictionary.icd_code, dictionary.icd_version)
    dictionary_evidence = dictionary.join(flags[list(CONDITIONS)])
    dictionary_evidence[dictionary_evidence[list(CONDITIONS)].any(axis=1)].to_csv(OUT / "mimic/condition_code_dictionary.csv", index=False)
    keys = pd.MultiIndex.from_frame(dictionary[["icd_code", "icd_version"]])
    chunks, diagnosis_source = verified_mimic("hosp/diagnoses_icd.csv.gz", chunksize=chunksize,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"], dtype={"icd_code": "string"})
    totals = {"source_rows": 0, "dictionary_unmatched_rows": 0, "malformed_code_rows": 0}
    groups = []
    for chunk in chunks:
        totals["source_rows"] += len(chunk)
        codes = normalize_codes(chunk.icd_code)
        matched = pd.MultiIndex.from_arrays([codes, chunk.icd_version]).isin(keys)
        totals["dictionary_unmatched_rows"] += int((~matched).sum())
        parsed = code_flags(codes, chunk.icd_version)
        totals["malformed_code_rows"] += int((~parsed.valid_code).sum())
        valid = parsed.valid_code & matched & chunk.hadm_id.notna()
        part = chunk.loc[valid, ["hadm_id", "subject_id"]].join(parsed.loc[valid, list(CONDITIONS)])
        part["diagnosis_code_rows"] = 1
        if not part.empty:
            agg = {name: "max" for name in CONDITIONS}
            agg.update(subject_id="first", diagnosis_code_rows="sum")
            groups.append(part.groupby("hadm_id").agg(agg))
    merged = pd.concat(groups)
    # A hospital admission belongs to exactly one patient, including across chunks.
    if merged.groupby(level=0).subject_id.nunique().max() != 1:
        raise ValueError("Admission-to-patient source relationship inconsistent")
    agg = {name: "max" for name in CONDITIONS}
    agg.update(subject_id="first", diagnosis_code_rows="sum")
    labels = merged.groupby(level=0).agg(agg).reset_index()
    for name in CONDITIONS:
        labels[name] = labels[name].astype("int8")
    labels.to_csv(OUT / "mimic/admission_diagnosis_labels.csv", index=False)
    totals.update(admissions_with_valid_documentation=len(labels),
        output_label_sha256=sha256(OUT / "mimic/admission_diagnosis_labels.csv"),
        all_admission_positive_counts={name: int(labels[name].sum()) for name in CONDITIONS},
        dictionary_matching_codes={name: int(flags[name].sum()) for name in CONDITIONS},
        source_diagnoses_gzip_read_to_eof=True, sources=[dictionary_source, diagnosis_source])
    save_json(OUT / "mimic/diagnosis_label_audit.json", totals)
    return labels


def prepare_eicu_labels(chunksize):
    sys.path.insert(0, str(PROJECT / "resp_physionet/resp"))
    from eicu_sources import get_eicu_table_path
    path = get_eicu_table_path("diagnosis", verify=True)
    dictionary, dictionary_source = verified_mimic("hosp/d_icd_diagnoses.csv.gz", dtype={"icd_code": "string"})
    lookup, exact_codes = eicu_code_lookup(dictionary)
    totals = {"source_rows": 0, "rows_without_parseable_codes": 0, "mixed_code_rows": 0,
              "exact_dictionary_code_tokens": 0, "recognized_parent_code_tokens": 0, "unmatched_code_tokens": 0}
    groups = []
    for chunk in pd.read_csv(path, usecols=["patientunitstayid", "icd9code"], dtype={"icd9code": "string"}, chunksize=chunksize):
        totals["source_rows"] += len(chunk)
        tokens = chunk.icd9code.fillna("").str.split(",").explode()
        codes = normalize_codes(tokens)
        # This field includes numeric ICD-9, supplementary ICD-9 V/E codes,
        # ICD-10 and legitimate category-parent tokens. Validate against known
        # version-specific codes/parents rather than guessing version by letter.
        parsed = pd.DataFrame({name: codes.map(lookup[name]).fillna(False).astype(bool)
                               for name in [*CONDITIONS, "valid_code"]}, index=codes.index)
        exact = codes.isin(exact_codes)
        recognized = codes.isin(lookup.index)
        totals["exact_dictionary_code_tokens"] += int(exact.sum())
        totals["recognized_parent_code_tokens"] += int((recognized & ~exact).sum())
        totals["unmatched_code_tokens"] += int((~recognized & codes.ne("")).sum())
        row_flags = parsed.groupby(level=0).max().reindex(chunk.index, fill_value=False)
        totals["rows_without_parseable_codes"] += int((~row_flags.valid_code).sum())
        totals["mixed_code_rows"] += int(chunk.icd9code.fillna("").str.contains(",", regex=False).sum())
        part = chunk[["patientunitstayid"]].join(row_flags)
        part = part[part.valid_code].drop(columns="valid_code")
        part["diagnosis_code_rows"] = 1
        agg = {name: "max" for name in CONDITIONS}
        agg["diagnosis_code_rows"] = "sum"
        groups.append(part.groupby("patientunitstayid").agg(agg))
    agg = {name: "max" for name in CONDITIONS}
    agg["diagnosis_code_rows"] = "sum"
    labels = pd.concat(groups).groupby(level=0).agg(agg).reset_index()
    for name in CONDITIONS:
        labels[name] = labels[name].astype("int8")
    labels.to_csv(OUT / "eicu/stay_diagnosis_labels.csv", index=False)
    totals.update(stays_with_parseable_documentation=len(labels),
                  output_label_sha256=sha256(OUT / "eicu/stay_diagnosis_labels.csv"),
                  all_stay_positive_counts={name: int(labels[name].sum()) for name in CONDITIONS},
                  source_file=str(path), source_sha256=sha256(path), source_gzip_read_to_eof=True,
                  timing="Any time during ICU stay; retrospective labels only",
                  past_history_added=False, code_dictionary_source=dictionary_source,
                  limitation="ICU clinician documentation differs from MIMIC hospital-discharge coding")
    save_json(OUT / "eicu/diagnosis_label_audit.json", totals)
    return labels


def eicu_code_lookup(dictionary):
    """Accept exact dictionary codes and real category prefixes, not arbitrary regexes.

    A token may occur in both versions; apply the union of the version-specific
    condition definitions. None of these four selected families has an ambiguous
    supplementary ICD-9 / ICD-10 positive prefix.
    """
    base = dictionary[["icd_code", "icd_version"]].copy()
    base["icd_code"] = normalize_codes(base.icd_code)
    exact = set(base.icd_code)
    parents = []
    for width in range(3, 8):
        part = base[base.icd_code.str.len().ge(width)].copy()
        part["icd_code"] = part.icd_code.str[:width]
        parents.append(part)
    known = pd.concat(parents, ignore_index=True).drop_duplicates()
    flags = code_flags(known.icd_code, known.icd_version)
    flags["valid_code"] = True  # All are actual dictionary entries or parents.
    flags["icd_code"] = known.icd_code
    return flags.groupby("icd_code").max(), exact


def assert_partitions(frame, hospital=False):
    assert frame.groupby("patient_id").split.nunique().max() == 1
    if hospital:
        assert frame.groupby("hospital_id").split.nunique().max() == 1
    assert set(frame.split) == {"train", "validation", "test"}


def export_model_tables(source, frame, split_audit):
    assert frame.stay_id.is_unique
    assert frame[list(CONDITIONS)].isin([0, 1]).all().all()
    assert_partitions(frame, hospital=source == "eicu")
    feature_names = list(FEATURES)
    absent = set(feature_names) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing predictors: {absent}")
    metadata = pd.DataFrame({"meta_source": frame.source, "meta_patient_id": frame.patient_id,
        "meta_stay_id": frame.stay_id, "meta_domain_hospitalid": frame.hospital_id,
        "meta_split": frame.split, "sens_age_group": frame.age_group, "sens_sex": frame.sex,
        "sens_race_ethnicity": frame.race_group})
    summary = {"source": source, "rows": len(frame), "patients": int(frame.patient_id.nunique()),
               "hospitals": int(frame.hospital_id.nunique()), "split_audit": split_audit,
               "split_rows": frame.split.value_counts().to_dict(), "conditions": {},
               "no_patient_overlap": True, "no_hospital_overlap": source == "eicu",
               "baseline_predictors": feature_names,
               "missing_predictors": {name: int(frame[name].isna().sum()) for name in feature_names},
               "no_preprocessing_fit_or_model_training": True}
    for condition, definition in CONDITIONS.items():
        result = pd.concat([metadata, frame[feature_names].reset_index(drop=True)], axis=1)
        result["target"] = frame[condition].astype("int8").to_numpy()
        target_path = OUT / source / f"{condition}_model_ready.csv"
        result.to_csv(target_path, index=False)
        summary["conditions"][condition] = {"label": definition["name"], "positive": int(result.target.sum()),
            "not_documented": int(result.target.eq(0).sum()), "path": str(target_path.relative_to(PROJECT)),
            "sha256": sha256(target_path),
            "split_counts": result.groupby("meta_split").target.agg(["count", "sum"]).to_dict("index")}
    save_json(OUT / source / "preprocessing_summary.json", summary)
    return summary


def build_mimic_tables():
    feature_path = OUT / "mimic/clinical_24h_features.csv"
    metadata_path = OUT / "mimic/cohort_metadata.csv"
    if not feature_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Complete prepare_mimic_clinical_features.py before --finalize")
    clinical_audit = json.loads((OUT / "mimic/clinical_preprocessing_validation.json").read_text())
    if not clinical_audit.get("passed") or not clinical_audit.get("chartevents_gzip_crc_verified_by_full_stream"):
        raise ValueError("Clinical extraction/source audit must pass before exporting datasets")
    for name, expected in clinical_audit["output_sha256"].items():
        if sha256(OUT / "mimic" / name) != expected:
            raise ValueError("Clinical output changed after validation: " + name)
    features = pd.read_csv(feature_path)
    mapping = pd.read_csv(metadata_path)
    # The clinical extractor supplies source/IDs/demographics either inline or
    # through metadata; merge only the needed linkage keys to avoid duplicate X.
    required_metadata = ["source", "patient_id", "hospital_id", "race_group", "subject_id", "hadm_id"]
    keys = ["stay_id"] + [name for name in required_metadata if name not in features]
    frame = features.merge(mapping[keys], on="stay_id", validate="one_to_one")
    label_path = OUT / "mimic/admission_diagnosis_labels.csv"
    label_audit = json.loads((OUT / "mimic/diagnosis_label_audit.json").read_text())
    if sha256(label_path) != label_audit["output_label_sha256"]:
        raise ValueError("MIMIC diagnosis labels changed after audit")
    labels = pd.read_csv(label_path)
    before = len(frame)
    frame = frame.merge(labels, on=["hadm_id", "subject_id"], how="inner", validate="many_to_one")
    frame, split_audit = patient_split(frame)
    split_audit["excluded_no_valid_diagnosis_documentation"] = before - len(frame)
    return export_model_tables("mimic", frame.reset_index(drop=True), split_audit)


def build_eicu_tables():
    feature_path = ROOT / "prepared/eicu_24h_features.csv"
    expected_features = json.loads((ROOT / "prepared/eicu_validation.json").read_text())
    if not expected_features.get("passed") or sha256(feature_path) != expected_features["feature_csv_sha256"]:
        raise ValueError("Previously audited eICU measurements changed")
    frame = pd.read_csv(feature_path, dtype={"hospital_id": "string"})
    label_path = OUT / "eicu/stay_diagnosis_labels.csv"
    label_audit = json.loads((OUT / "eicu/diagnosis_label_audit.json").read_text())
    if sha256(label_path) != label_audit["output_label_sha256"]:
        raise ValueError("eICU diagnosis labels changed after audit")
    labels = pd.read_csv(label_path)
    from eicu_history_evidence import load_eicu_history_evidence
    history, history_audit = load_eicu_history_evidence()
    audit = {"active_icd_documented_stays": len(labels), "conditions": {}}
    labels = labels.merge(history, on="patientunitstayid", how="left", validate="one_to_one")
    for condition in CONDITIONS:
        labels["icd_" + condition] = labels[condition].astype("int8")
        labels["history_" + condition] = labels["history_" + condition].fillna(0).astype("int8")
        added = labels[condition].eq(0) & labels["history_" + condition].eq(1)
        labels[condition] = (labels[condition].eq(1) | labels["history_" + condition].eq(1)).astype("int8")
        outside = history[~history.patientunitstayid.isin(labels.patientunitstayid)]
        audit["conditions"][condition] = {"icd_positive": int(labels["icd_" + condition].sum()),
            "explicit_history_added_positive": int(added.sum()), "primary_union_positive": int(labels[condition].sum()),
            "history_positive_outside_documented_control_population_excluded": int(outside["history_" + condition].sum())}
    labels.to_csv(OUT / "eicu/stay_diagnosis_and_history_labels.csv", index=False)
    save_json(OUT / "eicu/history_evidence_audit.json", history_audit)
    save_json(OUT / "eicu/primary_label_union_audit.json", audit)
    labels = labels.rename(columns={"patientunitstayid": "stay_id"})
    before = len(frame)
    frame = frame.merge(labels, on="stay_id", how="inner", validate="one_to_one")
    frame, split_audit = hospital_split(frame)
    split_audit.update(excluded_no_parseable_diagnosis_documentation=before - len(frame) - split_audit["purged_rows"],
                      feature_source=str(feature_path), feature_source_sha256=sha256(feature_path))
    return export_model_tables("eicu", frame.reset_index(drop=True), split_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Use already audited diagnosis labels, attach features and split")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    if args.labels_only and args.finalize:
        parser.error("Choose --labels-only or --finalize")
    for source in ["mimic", "eicu"]:
        (OUT / source).mkdir(parents=True, exist_ok=True)
    if not args.finalize:
        prepare_mimic_labels(args.chunksize)
        prepare_eicu_labels(args.chunksize)
    save_json(OUT / "phenotype_definitions.json", {"conditions": CONDITIONS,
        "label_1": "Matching diagnosis code; eICU also accepts explicit structured chronic-condition history positives",
        "label_0": "No matching diagnosis documented among encounters with parseable diagnosis documentation; not confirmed absence of disease",
        "unknown": "No parseable diagnosis documentation: excluded, not converted to zero",
        "mimic_label_window": "Entire hospital admission, coded on discharge",
        "eicu_label_window": "Entire ICU stay, clinician-entered diagnosis table OR explicit chronic history; no procedures/treatment proxies",
        "target_not_onset_or_early_diagnosis": True,
        "respiratory_original_mortality_task_preserved_separately": True,
        "research_definition_not_official_prepackaged_cohort": True})
    if args.labels_only:
        print("Diagnosis labels prepared; clinical-feature extraction/finalization remains.", flush=True)
        return
    eicu = build_eicu_tables()
    mimic = build_mimic_tables()
    save_json(OUT / "preprocessing_status.json", {"completed": True, "completed_utc": utc_now(),
        "task": "Four retrospective recorded-diagnosis classification tasks", "sources": {"mimic": mimic, "eicu": eicu},
        "cross_hospital_evaluation_completed": False,
        "fully_equivalent_clinical_label_ascertainment": False,
        "baseline_transformations_fitted": False})
    print(json.dumps({source: {"rows": data["rows"], "patients": data["patients"], "hospitals": data["hospitals"],
        "positives": {name: record["positive"] for name, record in data["conditions"].items()}}
        for source, data in [("mimic", mimic), ("eicu", eicu)]}), flush=True)


if __name__ == "__main__":
    main()
