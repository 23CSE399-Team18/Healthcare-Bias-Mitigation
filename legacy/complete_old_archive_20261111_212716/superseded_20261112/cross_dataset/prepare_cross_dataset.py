"""Prepare harmonized, 24-hour adult ICU mortality cohorts from real sources.

This is a separate all-adult-ICU study, not respiratory phenotype validation.
No imputation or model fitting occurs here. Large event tables are streamed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "_data_reader"))
import numpy as np
import pandas as pd

VITALS = ["heart_rate", "resp_rate", "spo2", "temperature"]
# Temperature remains an extraction-quality audit. eICU monitor temperatures
# are absent in89.5% of this cohort and are not comparable to routine MIMIC
# nursing temperatures without an additional source-harmonization study.
BASELINE_VITALS = ["heart_rate", "resp_rate", "spo2"]
NUMERIC_FEATURES = [f"{v}_{s}" for v in BASELINE_VITALS for s in ["mean", "min", "max"]]
CATEGORICAL_FEATURES = ["age_group", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BOUNDS = {"heart_rate": (0, 300, False), "resp_rate": (0, 70, False),
          "spo2": (0, 100, True), "temperature": (10, 50, False)}
MIMIC_ITEMS = {220045: "heart_rate", 220210: "resp_rate", 224690: "resp_rate",
               220277: "spo2", 223762: "temperature", 223761: "temperature"}
EICU_COLUMNS = {"heartrate": "heart_rate", "respiration": "resp_rate", "sao2": "spo2",
                "temperature": "temperature"}
METADATA = ["source", "patient_id", "stay_id", "hospital_id", "race_group"]
TARGET = "hospital_mortality"


def age_group(age):
    age = pd.to_numeric(age, errors="coerce")
    return pd.cut(age, [18, 40, 65, 80, np.inf], right=False,
                  labels=["18-39", "40-64", "65-79", "80+"]).astype("string")


def normalize_sex(values):
    values = values.astype("string").str.strip().str.upper()
    return values.map({"M": "Male", "MALE": "Male", "F": "Female", "FEMALE": "Female"}).fillna("Unknown")


def race_group(values):
    """Broad reported race/ethnicity categories for auditing, not model inputs."""
    text = values.astype("string").fillna("").str.strip().str.upper()
    result = pd.Series("Other", index=values.index, dtype="string")
    result[text.str.contains("WHITE|CAUCASIAN", regex=True)] = "White"
    result[text.str.contains("BLACK|AFRICAN", regex=True)] = "Black"
    result[text.str.contains("ASIAN", regex=False)] = "Asian"
    result[text.str.contains("HISPANIC|LATINO", regex=True)] = "Hispanic"
    result[text.isin(["", "UNKNOWN", "OTHER/UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"])] = "Unknown"
    return result


class HourlyAccumulator:
    """Equal-weight observed hourly means reduce sampling-frequency differences.

    Each source observation contributes to its stay/hour mean, then min/max/mean
    are calculated over observed hourly means. Missing hours remain missing.
    """
    def __init__(self, stay_ids, *, vitals=None, bounds=None):
        self.ids = pd.Index(stay_ids)
        self.vitals = tuple(VITALS if vitals is None else vitals)
        self.bounds = BOUNDS if bounds is None else bounds
        if len(set(self.vitals)) != len(self.vitals) or any(v not in self.bounds for v in self.vitals):
            raise ValueError("Each unique accumulator vital requires explicit bounds")
        self.size = len(self.ids) * 24
        self.sums = {v: np.zeros(self.size, dtype=np.float64) for v in self.vitals}
        self.counts = {v: np.zeros(self.size, dtype=np.uint32) for v in self.vitals}
        self.accepted = {v: 0 for v in self.vitals}

    def add(self, stay_ids, minute_offsets, values):
        ids = self.ids.get_indexer(stay_ids)
        offsets = np.asarray(minute_offsets, dtype=float)
        valid_window = (ids >= 0) & np.isfinite(offsets) & (offsets >= 0) & (offsets <= 1440)
        safe_offsets = np.where(np.isfinite(offsets), offsets, 0)
        hours = np.minimum(np.maximum(np.floor(safe_offsets / 60), 0), 23).astype(np.int64)
        bins = ids * 24 + hours
        for vital, series in values.items():
            data = np.asarray(series, dtype=float)
            low, high, upper_inclusive = self.bounds[vital]
            good = valid_window & np.isfinite(data) & (data > low)
            good &= (data <= high) if upper_inclusive else (data < high)
            selected = bins[good]
            if selected.size:
                self.sums[vital] += np.bincount(selected, weights=data[good], minlength=self.size)
                self.counts[vital] += np.bincount(selected, minlength=self.size).astype(np.uint32)
                self.accepted[vital] += int(good.sum())

    def finish(self):
        features = pd.DataFrame(index=self.ids)
        observed = pd.DataFrame(index=self.ids)
        for vital in self.vitals:
            count = self.counts[vital].reshape(-1, 24)
            sums = self.sums[vital].reshape(-1, 24)
            hourly = np.full(sums.shape, np.nan)
            np.divide(sums, count, out=hourly, where=count > 0)
            table = pd.DataFrame(hourly, index=self.ids)
            features[f"{vital}_mean"] = table.mean(axis=1)
            features[f"{vital}_min"] = table.min(axis=1)
            features[f"{vital}_max"] = table.max(axis=1)
            observed[f"{vital}_observed_hours"] = (count > 0).sum(axis=1)
        return features, observed


def eicu_cohort(source):
    columns = ["patientunitstayid", "uniquepid", "hospitalid", "age", "gender", "ethnicity",
               "unitdischargeoffset", "hospitaldischargeoffset", "hospitaldischargestatus"]
    frame = pd.read_csv(source, usecols=columns, dtype={"uniquepid": "string", "age": "string"})
    flow = [{"stage": "all_icu_stays", "rows": len(frame)}]
    raw_age = frame.age.str.replace(" ", "", regex=False)
    # Nullable .mask conditions treat NA as replaceable; fill only the explicit
    # topcode condition so unknown age never becomes an artificial90-year-old.
    frame["_age"] = pd.to_numeric(raw_age.mask(raw_age.eq(">89").fillna(False), "90"), errors="coerce")
    frame = frame[frame._age.ge(18) & frame.patientunitstayid.notna() & frame.hospitalid.notna() & frame.uniquepid.notna()].copy()
    flow.append({"stage": "known_adult_and_identifiers", "rows": len(frame)})
    frame = frame[frame.hospitaldischargestatus.isin(["Alive", "Expired"])].copy()
    flow.append({"stage": "known_hospital_outcome", "rows": len(frame)})
    frame = frame[frame.unitdischargeoffset.gt(1440) & frame.hospitaldischargeoffset.gt(1440)].copy()
    flow.append({"stage": "still_in_icu_and_hospital_after_24h", "rows": len(frame)})
    assert frame.patientunitstayid.is_unique
    cohort = pd.DataFrame({"source": "eicu", "patient_id": "eicu:" + frame.uniquepid,
                           "stay_id": frame.patientunitstayid.astype("int64"),
                           "hospital_id": frame.hospitalid.astype("int64").astype("string"),
                           "age_group": age_group(frame._age), "sex": normalize_sex(frame.gender),
                           "race_group": race_group(frame.ethnicity),
                           TARGET: frame.hospitaldischargestatus.eq("Expired").astype(int)})
    return cohort.set_index("stay_id", drop=False), flow


def prepare_eicu(args):
    sys.path.insert(0, str(args.eicu_dir))
    from eicu_sources import get_eicu_table_path
    patient = get_eicu_table_path("patient", verify=True)
    periodic = get_eicu_table_path("vitalPeriodic", verify=True)
    cohort, flow = eicu_cohort(patient)
    accumulator = HourlyAccumulator(cohort.index)
    rows = 0
    columns = ["patientunitstayid", "observationoffset"] + list(EICU_COLUMNS)
    for chunk in pd.read_csv(periodic, usecols=columns, chunksize=args.chunksize):
        rows += len(chunk)
        accumulator.add(chunk.patientunitstayid, chunk.observationoffset,
                        {out: pd.to_numeric(chunk[col], errors="coerce") for col, out in EICU_COLUMNS.items()})
        if rows % (args.chunksize * 20) == 0:
            print(f"eICU: streamed {rows:,} periodic-vital rows", flush=True)
    features, observed = accumulator.finish()
    result = cohort.join(features)
    save_prepared("eicu", result, observed, flow,
                  {"event_rows_read": rows, "accepted_observations": accumulator.accepted,
                   "source_paths": [str(patient), str(periodic)]}, args.output_dir)


def mimic_cohort(patients, admissions, stays):
    stays = stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    frame = stays.merge(patients, on="subject_id", how="inner", validate="many_to_one")
    flow = [{"stage": "ICU_stays_linked_to_admission_and_patient", "rows": len(frame)}]
    for field in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    frame["_age"] = frame.anchor_age + (frame.intime.dt.year - frame.anchor_year)
    frame = frame[frame._age.ge(18) & frame.stay_id.notna() & frame.subject_id.notna()].copy()
    flow.append({"stage": "known_adult_and_identifiers", "rows": len(frame)})
    frame = frame[frame.hospital_expire_flag.isin([0, 1])].copy()
    flow.append({"stage": "known_hospital_outcome", "rows": len(frame)})
    landmark = frame.intime + pd.Timedelta(hours=24)
    alive = frame.deathtime.isna() | frame.deathtime.gt(landmark)
    frame = frame[frame.outtime.gt(landmark) & frame.dischtime.gt(landmark) & alive & frame.admittime.le(frame.intime)].copy()
    flow.append({"stage": "alive_and_still_in_icu_and_hospital_after_24h", "rows": len(frame)})
    assert frame.stay_id.is_unique
    cohort = pd.DataFrame({"source": "mimic", "patient_id": "mimic:" + frame.subject_id.astype("int64").astype("string"),
                           "stay_id": frame.stay_id.astype("int64"), "hospital_id": "BIDMC",
                           "age_group": age_group(frame._age), "sex": normalize_sex(frame.gender),
                           "race_group": race_group(frame.race), TARGET: frame.hospital_expire_flag.astype(int),
                           "_intime": frame.intime})
    return cohort.set_index("stay_id", drop=False), flow


def prepare_mimic(args):
    from streaming_sources import read_official_csv
    raw = args.mimic_dir
    patients = read_official_csv(raw / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    admissions = read_official_csv(raw / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "hospital_expire_flag"])
    stays = read_official_csv(raw / "icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    items = read_official_csv(raw / "icu/d_items.csv.gz")
    selected_items = items[items.itemid.isin(MIMIC_ITEMS)].copy()
    missing = set(MIMIC_ITEMS) - set(selected_items.itemid)
    if missing:
        raise ValueError(f"MIMIC vital item mappings missing from this release: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_items.to_csv(args.output_dir / "mimic_item_mapping.csv", index=False)
    cohort, flow = mimic_cohort(patients, admissions, stays)
    accumulator = HourlyAccumulator(cohort.index)
    rows = accepted_rows = 0
    columns = ["stay_id", "itemid", "charttime", "storetime", "valuenum"]
    for chunk in read_official_csv(raw / "icu/chartevents.csv.gz", usecols=columns, chunksize=args.chunksize):
        rows += len(chunk)
        relevant = chunk.itemid.isin(MIMIC_ITEMS) & chunk.stay_id.isin(cohort.index)
        selected = chunk.loc[relevant].copy()
        if not selected.empty:
            intime = selected.stay_id.map(cohort._intime)
            charttime = pd.to_datetime(selected.charttime, errors="coerce")
            storetime = pd.to_datetime(selected.storetime, errors="coerce")
            # Prevent late-entered observations from becoming available before the landmark.
            available = storetime.isna() | storetime.le(intime + pd.Timedelta(hours=24))
            selected = selected.loc[available]
            offsets = (charttime.loc[available] - intime.loc[available]).dt.total_seconds() / 60
            accepted_rows += len(selected)
            numeric = pd.to_numeric(selected.valuenum, errors="coerce")
            for vital in VITALS:
                values = numeric.where(selected.itemid.map(MIMIC_ITEMS).eq(vital))
                if vital == "temperature":
                    fahrenheit = selected.itemid.eq(223761)
                    values = values.mask(fahrenheit, (values - 32) / 1.8)
                accumulator.add(selected.stay_id, offsets, {vital: values})
        if rows % (args.chunksize * 20) == 0:
            print(f"MIMIC: streamed {rows:,} chartevents rows", flush=True)
    features, observed = accumulator.finish()
    result = cohort.drop(columns="_intime").join(features)
    save_prepared("mimic", result, observed, flow,
                  {"event_rows_read": rows, "relevant_available_rows_before_window_filter": accepted_rows,
                   "accepted_observations": accumulator.accepted, "source_version": "MIMIC-IV3.1"}, args.output_dir)


def save_prepared(source, frame, observed, flow, details, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame[["source", "patient_id", "stay_id", "hospital_id", "race_group", TARGET] + FEATURES]
    assert frame.stay_id.is_unique and frame[TARGET].isin([0, 1]).all()
    assert frame.age_group.notna().all()
    feature_path = output_dir / f"{source}_24h_features.csv"
    partial = feature_path.with_suffix(feature_path.suffix + ".partial")
    frame.to_csv(partial, index=False)
    partial.replace(feature_path)
    audit_path = output_dir / f"{source}_observation_audit.csv"
    partial_audit = audit_path.with_suffix(audit_path.suffix + ".partial")
    observed.to_csv(partial_audit, index_label="stay_id")
    partial_audit.replace(audit_path)
    summary = {"source": source, "rows": len(frame), "patients": int(frame.patient_id.nunique()),
               "hospitals": int(frame.hospital_id.nunique()), "positive": int(frame[TARGET].sum()),
               "negative": int(frame[TARGET].eq(0).sum()), "cohort_flow": flow,
               "missing_predictor_counts": {k: int(v) for k, v in frame[FEATURES].isna().sum().items()},
               "no_imputation_or_resampling": True, "no_model_fitting": True,
               "observation_window_minutes_inclusive": [0, 1440],
               "aggregation": "Hourly means, then equal-weight mean/min/max of observed hourly means",
               "feature_csv_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
               **details}
    (output_dir / f"{source}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def refresh_status(output_dir=ROOT / "prepared", status_path=ROOT / "status.json", results_dir=ROOT / "results"):
    status = {"evaluation_completed": (results_dir / "generalization_metrics.csv").exists(),
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for source in ["eicu", "mimic"]:
        features = output_dir / f"{source}_24h_features.csv"
        summary = output_dir / f"{source}_summary.json"
        prepared = features.exists() and summary.exists()
        status[f"{source}_prepared"] = prepared
        if prepared:
            details = json.loads(summary.read_text(encoding="utf-8"))
            for field, key in [("stays", "rows"), ("patients", "patients"), ("hospitals", "hospitals"), ("deaths", "positive")]:
                status[f"{source}_{field}"] = details[key]
    if not status.get("mimic_prepared"):
        status["next_required_action"] = "Complete the approved-account MIMIC download/import, then run --source mimic and evaluation."
    elif not status["evaluation_completed"]:
        status["next_required_action"] = "Use the current four-condition workflow in four_conditions/README.md; the historical mortality runner is archived."
    else:
        status["next_required_action"] = "Review descriptive transfer and demographic metrics; baseline is not clinically validated."
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def write_schema(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {"study": "Additional all-adult ICU 24h hospital-mortality generalization",
              "numeric_predictors": NUMERIC_FEATURES, "categorical_predictors": CATEGORICAL_FEATURES,
              "target": TARGET, "metadata_excluded_from_predictors": METADATA,
              "fairness_attributes": ["age_group", "sex", "race_group"],
              "label": "Subsequent in-hospital death among patients alive and still observed at24h",
              "mimic_items": {str(k): v for k, v in MIMIC_ITEMS.items()},
              "vitals_excluded_from_baseline_after_source_quality_audit": ["temperature"],
              "common_bounds": BOUNDS, "age_bands": ["18-39", "40-64", "65-79", "80+"],
              "units": {"heart_rate": "beats/min", "resp_rate": "breaths/min", "spo2": "%", "temperature": "Celsius"}}
    (output_dir / "common_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["eicu", "mimic", "both"], default="both")
    parser.add_argument("--chunksize", type=int, default=400000)
    parser.add_argument("--eicu-dir", type=Path, default=PROJECT / "resp_physionet/resp")
    parser.add_argument("--mimic-dir", type=Path, default=ROOT / "raw/mimiciv/3.1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    write_schema(args.output_dir)
    if args.source in ["eicu", "both"]:
        prepare_eicu(args)
    if args.source in ["mimic", "both"]:
        prepare_mimic(args)
    refresh_status(args.output_dir)


if __name__ == "__main__":
    main()
