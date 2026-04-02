"""Extract actual, publisher-verified MIMIC-IV3.1 first-day measurements.

No diagnoses, glucose, HbA1c, therapies, imputation, scaling, resampling, model
fitting, or evaluation are used. All measured predictors require charttime in
[ICU admission, ICU admission+24h] AND a known storetime no later than24h.
Unknown documentation times are excluded, unlike the older mortality extractor.
The entire chartevents gzip is streamed to EOF, validating its gzip CRC.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from prepare_cross_dataset import (pd, np, ROOT, mimic_cohort, HourlyAccumulator,
                                   NUMERIC_FEATURES, CATEGORICAL_FEATURES)
from import_browser_downloads import hash_stable_file, fingerprint
from download_mimic import BASE_URL, parse_publisher_checksums, utc_now


REQUIRED = (
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz",
    "icu/d_items.csv.gz", "icu/chartevents.csv.gz",
)
# The two blood-pressure measurement routes remain separate. They are optional
# MIMIC-only predictors until equivalent eICU routes have been harmonized.
ITEMS = {
    220045: ("heart_rate", "Heart Rate", "bpm"),
    220210: ("resp_rate", "Respiratory Rate", "insp/min"),
    224690: ("resp_rate", "Respiratory Rate (Total)", "insp/min"),
    220277: ("spo2", "O2 saturation pulseoxymetry", "%"),
    223761: ("temperature", "Temperature Fahrenheit", "\u00b0F"),
    223762: ("temperature", "Temperature Celsius", "\u00b0C"),
    220179: ("nibp_systolic", "Non Invasive Blood Pressure systolic", "mmHg"),
    220180: ("nibp_diastolic", "Non Invasive Blood Pressure diastolic", "mmHg"),
    220181: ("nibp_mean", "Non Invasive Blood Pressure mean", "mmHg"),
    220050: ("arterial_systolic", "Arterial Blood Pressure systolic", "mmHg"),
    220051: ("arterial_diastolic", "Arterial Blood Pressure diastolic", "mmHg"),
    220052: ("arterial_mean", "Arterial Blood Pressure mean", "mmHg"),
}
VITALS = tuple(dict.fromkeys(value[0] for value in ITEMS.values()))
BOUNDS = {"heart_rate": (0, 300, False), "resp_rate": (0, 70, False),
          "spo2": (0, 100, True), "temperature": (10, 50, False),
          "nibp_systolic": (0, 400, False), "nibp_diastolic": (0, 300, False),
          "nibp_mean": (0, 300, False), "arterial_systolic": (0, 400, False),
          "arterial_diastolic": (0, 300, False), "arterial_mean": (0, 300, False)}
EXTENDED_FEATURES = [f"{v}_{s}" for v in VITALS if v not in ("heart_rate", "resp_rate", "spo2")
                     for s in ("mean", "min", "max")]
BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, frame, **kwargs):
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def verify_sources(download_dir):
    checksums = parse_publisher_checksums((download_dir / "SHA256SUMS.txt").read_bytes(), REQUIRED)
    sources = {}
    for relative in REQUIRED:
        path = download_dir / Path(relative).name
        record = hash_stable_file(path, require_gzip=True)
        if record["sha256"] != checksums[relative]:
            raise ValueError("Source fails MIMIC-IV3.1 publisher SHA256: " + relative)
        sources[relative] = {"path": str(path.resolve()), "relative_name": relative,
                             "url": BASE_URL + relative, "bytes": record["bytes"],
                             "sha256": record["sha256"], "publisher_sha256": checksums[relative],
                             "fingerprint": list(record["fingerprint"]),
                             "publisher_sha256_match": True}
        print("Publisher SHA256 verified: " + relative, flush=True)
    return sources


def validate_items(items):
    if not items.itemid.is_unique:
        raise ValueError("MIMIC item dictionary is not unique")
    indexed = items.set_index("itemid")
    audit = []
    for itemid, (concept, label, unit) in ITEMS.items():
        if itemid not in indexed.index:
            raise ValueError(f"Missing reviewed MIMIC vital item {itemid}")
        actual = indexed.loc[itemid]
        if actual["label"] != label or actual.unitname != unit:
            raise ValueError(f"Reviewed MIMIC label/unit differs for item {itemid}")
        audit.append({"itemid": itemid, "concept": concept, "label": label,
                      "dictionary_unit": unit, "label_and_unit_verified": True,
                      "conversion": "(F-32)/1.8" if itemid == 223761 else "identity",
                      "role": "baseline" if concept in ("heart_rate", "resp_rate", "spo2") else "optional_extended"})
    return pd.DataFrame(audit)


def filter_events(chunk, cohort):
    """Return only actual observations available by the prediction landmark.

    Audit counts below are disjoint exclusions, applied in the stated order.
    Blank row units inherit the verified dictionary unit; explicit conflicts
    are rejected. No knowledge of diagnoses or outcomes enters this function.
    """
    audit = {"rows_read": len(chunk)}
    relevant = chunk.itemid.isin(ITEMS) & chunk.stay_id.isin(cohort.index)
    selected = chunk.loc[relevant].copy()
    audit["reviewed_item_and_eligible_stay_rows"] = len(selected)
    audit["excluded_unknown_charttime"] = 0
    audit["excluded_charttime_outside_first24h"] = 0
    audit["excluded_unknown_storetime"] = 0
    audit["excluded_storetime_after24h"] = 0
    audit["excluded_explicit_unit_conflict"] = 0
    audit["retained_rows_unit_inferred_from_dictionary"] = 0
    audit["retained_rows_before_numeric_bounds"] = 0
    if selected.empty:
        return selected, pd.Series(dtype=float), audit
    intime = selected.stay_id.map(cohort._intime)
    charttime = pd.to_datetime(selected.charttime, errors="coerce")
    storetime = pd.to_datetime(selected.storetime, errors="coerce")
    offsets = (charttime - intime).dt.total_seconds() / 60
    valid = charttime.notna()
    audit["excluded_unknown_charttime"] = int((~valid).sum())
    in_window = offsets.ge(0) & offsets.le(1440)
    audit["excluded_charttime_outside_first24h"] = int((valid & ~in_window).sum())
    valid &= in_window
    audit["excluded_unknown_storetime"] = int((valid & storetime.isna()).sum())
    valid &= storetime.notna()
    on_time = storetime.le(intime + pd.Timedelta(hours=24))
    audit["excluded_storetime_after24h"] = int((valid & ~on_time).sum())
    valid &= on_time
    recorded_unit = selected.valueuom.astype("string").fillna("").str.strip()
    expected_unit = selected.itemid.map({i: value[2] for i, value in ITEMS.items()})
    unit_ok = recorded_unit.eq("") | recorded_unit.eq(expected_unit)
    audit["excluded_explicit_unit_conflict"] = int((valid & ~unit_ok).sum())
    valid &= unit_ok
    audit["retained_rows_unit_inferred_from_dictionary"] = int((valid & recorded_unit.eq("")).sum())
    selected = selected.loc[valid].copy()
    audit["retained_rows_before_numeric_bounds"] = len(selected)
    return selected, offsets.loc[valid], audit


def add_events(accumulator, selected, offsets):
    if selected.empty:
        return
    concept = selected.itemid.map({i: value[0] for i, value in ITEMS.items()})
    numeric = pd.to_numeric(selected.valuenum, errors="coerce")
    for vital in VITALS:
        rows = concept.eq(vital)
        if not rows.any():
            continue
        values = numeric.loc[rows]
        if vital == "temperature":
            fahrenheit = selected.loc[rows, "itemid"].eq(223761)
            values = values.mask(fahrenheit, (values - 32) / 1.8)
        accumulator.add(selected.loc[rows, "stay_id"], offsets.loc[rows], {vital: values})


def prepare(download_dir, output_dir, chunksize=400000):
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "clinical_preprocessing_validation.json"
    if (output_dir / "clinical_24h_features.csv").exists():
        raise ValueError("Existing clinical output preserved. Choose a new output directory to rerun.")
    started = time.monotonic()
    state = {"started_utc": utc_now(), "status": "verifying_sources", "passed": False,
             "raw_source": "PhysioNet MIMIC-IV v3.1", "synthetic_medical_records": False,
             "no_model_training": True, "no_imputation_scaling_or_resampling": True}
    atomic_json(validation_path, state)
    try:
        sources = verify_sources(download_dir)
        state["sources"] = sources
        atomic_json(validation_path, state)
        paths = {key: Path(value["path"]) for key, value in sources.items()}
        patients = pd.read_csv(paths["hosp/patients.csv.gz"], usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
        admissions = pd.read_csv(paths["hosp/admissions.csv.gz"],
            usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "hospital_expire_flag"])
        stays = pd.read_csv(paths["icu/icustays.csv.gz"],
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
        items = pd.read_csv(paths["icu/d_items.csv.gz"], usecols=["itemid", "label", "unitname"])
        if not patients.subject_id.is_unique or not admissions.hadm_id.is_unique or not stays.stay_id.is_unique:
            raise ValueError("MIMIC source keys are not unique")
        item_audit = validate_items(items)
        cohort, flow = mimic_cohort(patients, admissions, stays)
        accumulator = HourlyAccumulator(cohort.index, vitals=VITALS, bounds=BOUNDS)
        totals = {}
        state.update(status="streaming_chartevents", cohort_rows=len(cohort),
                     cohort_patients=int(cohort.patient_id.nunique()), cohort_flow=flow)
        atomic_json(validation_path, state)
        columns = ["stay_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
        # Exhaust this gzip stream completely: GzipFile verifies CRC and uncompressed
        # length at EOF. No large decompressed file is ever written to FAT32.
        with gzip.open(paths["icu/chartevents.csv.gz"], "rt", encoding="utf-8", newline="") as stream:
            with pd.read_csv(stream, usecols=columns, chunksize=chunksize,
                             dtype={"valueuom": "string"}) as reader:
                for number, chunk in enumerate(reader, start=1):
                    selected, offsets, audit = filter_events(chunk, cohort)
                    add_events(accumulator, selected, offsets)
                    for key, count in audit.items():
                        totals[key] = totals.get(key, 0) + count
                    if number % 10 == 0:
                        state.update(event_audit=totals, elapsed_seconds=round(time.monotonic() - started, 1))
                        atomic_json(validation_path, state)
                        print(f"MIMIC clinical extraction: {totals['rows_read']:,} chartevents rows; "
                              f"{totals['retained_rows_before_numeric_bounds']:,} available first24h rows", flush=True)
            # An explicit read also ensures decompressor EOF/CRC if parser buffering
            # already consumed the final CSV bytes.
            if stream.read(1):
                raise ValueError("CSV parser did not consume the complete chartevents source")
        for relative, source in sources.items():
            if list(fingerprint(Path(source["path"]).stat())) != source["fingerprint"]:
                raise ValueError("Source changed during extraction: " + relative)
        features, observed = accumulator.finish()
        if not features.index.equals(cohort.index):
            raise ValueError("Feature/cohort index alignment failed")
        result = cohort[CATEGORICAL_FEATURES].join(features)
        result = result[BASELINE_FEATURES + EXTENDED_FEATURES]
        if not result.index.is_unique or result.age_group.isna().any():
            raise ValueError("Feature membership or adult-age grouping failed")
        metadata = cohort.rename(columns={"_intime": "icu_intime"}).copy()
        mapping = stays.set_index("stay_id")[["subject_id", "hadm_id"]]
        metadata = metadata.join(mapping)
        metadata["prediction_landmark"] = metadata.icu_intime + pd.Timedelta(hours=24)
        forbidden = {"subject_id", "hadm_id", "patient_id", "hospital_id", "race_group", "hospital_mortality",
                     "icd_code", "diagnosis", "glucose", "HbA1c", "outtime", "dischtime", "deathtime"}
        if forbidden.intersection(result.columns):
            raise ValueError("Forbidden outcome/diagnosis/identifier column in predictors")
        atomic_csv(output_dir / "clinical_24h_features.csv", result, index_label="stay_id")
        atomic_csv(output_dir / "cohort_metadata.csv", metadata, index=False)
        atomic_csv(output_dir / "observation_audit.csv", observed, index_label="stay_id")
        atomic_csv(output_dir / "item_mapping_validation.csv", item_audit, index=False)
        schema = {
            "source": "MIMIC-IV v3.1", "baseline_numeric_predictors": NUMERIC_FEATURES,
            "baseline_categorical_predictors": CATEGORICAL_FEATURES,
            "optional_extended_predictors_not_yet_harmonized_to_eicu": EXTENDED_FEATURES,
            "join_key_never_a_predictor": "stay_id",
            "metadata_never_predictors": [name for name in metadata.columns if name not in CATEGORICAL_FEATURES],
            "baseline_demographics_also_copied_into_metadata_for_auditing": CATEGORICAL_FEATURES,
            "window_minutes_inclusive": [0, 1440], "documentation_time_required": True,
            "unknown_storetime": "excluded", "storetime_after_24h": "excluded",
            "aggregation": "Mean per observed ICU hour, then mean/min/max of observed hourly means; missing hours remain missing",
            "bounds_lower_exclusive_upper_inclusive_only_if_true": BOUNDS,
            "unit_policy": "Explicit row units must match verified dictionary; missing units inherit dictionary",
            "source_text_encoding": "UTF-8, explicitly selected including on Windows",
            "missing_values": "preserved; no imputation, interpolation or filling",
            "important_limits": ["Optional BP/temperature are MIMIC-only until cross-source harmonization",
                                 "Discharge diagnoses must not be interpreted as disease onset or confirmed availability by24h",
                                 "ICU>24h and known mortality eligibility reused from documented original metadata cohort",
                                 "No treatment, glucose, HbA1c, diagnosis or discharge measurements used as baseline predictors"],
        }
        atomic_json(output_dir / "clinical_feature_schema.json", schema)
        output_hashes = {}
        for name in ("clinical_24h_features.csv", "cohort_metadata.csv", "observation_audit.csv"):
            with (output_dir / name).open("rb") as output_stream:
                output_hashes[name] = hashlib.file_digest(output_stream, "sha256").hexdigest()
        accepted_counts = {v: int(accumulator.counts[v].sum(dtype=np.uint64)) for v in VITALS}
        state.update(status="completed", passed=True, completed_utc=utc_now(),
                     elapsed_seconds=round(time.monotonic() - started, 1),
                     chartevents_gzip_crc_verified_by_full_stream=True,
                     source_files_unchanged_after_processing=True, rows=len(result),
                     hospitals=1, baseline_predictors=BASELINE_FEATURES,
                     optional_extended_predictors=EXTENDED_FEATURES,
                     event_audit=totals, accepted_observations=accumulator.accepted,
                     accepted_counter_consistency=accepted_counts == accumulator.accepted,
                     missing_predictor_counts={name: int(count) for name, count in result.isna().sum().items()},
                     output_sha256=output_hashes, diagnosis_labels_built_here=False)
        if not state["accepted_counter_consistency"]:
            raise ValueError("Accumulator observation counts are inconsistent")
        atomic_json(validation_path, state)
        print(json.dumps({"passed": True, "rows": len(result), "chartevents_rows": totals["rows_read"],
                          "gzip_crc_verified": True, "output_dir": str(output_dir)}), flush=True)
        return state
    except BaseException as error:
        state.update(status="failed", passed=False, error_type=type(error).__name__,
                     error=str(error), elapsed_seconds=round(time.monotonic() - started, 1))
        atomic_json(validation_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "four_conditions/mimic")
    parser.add_argument("--chunksize", type=int, default=400000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    prepare(args.download_dir.resolve(), args.output_dir.resolve(), args.chunksize)


if __name__ == "__main__":
    main()
