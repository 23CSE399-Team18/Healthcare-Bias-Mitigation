"""Extract leakage-safe first-24h predictors for the eICU respiratory cohort.

Input : dataset_verified/respiratory_cohort_24h_landmark.csv (membership + target)
Output: D:/complete/respiratory/raw/respiratory_raw.csv
        D:/complete/respiratory/raw/respiratory_feature_audit.json

Every predictor is available at the 24-hour prediction landmark:
  * periodic/aperiodic vitals observed 0..1440 min after ICU admission
    (temperature also from nurse charting, observed and entered by 1440 min);
  * labs resulted -360..1440 min (6 h pre-ICU, the MIT-LCP first-day convention),
    excluding results revised after 1440 min;
  * APACHE day-1 *inputs* (eICU: collected during the first 24 h of the unit stay);
    never APACHE predictions, scores, discharge or outcome fields;
  * APACHE admission diagnosis only if admissionDx shows it entered <=1440 min.

Only physiologically impossible / unit-error values are removed (wide bounds below);
genuine extreme values are kept for the outlier-bias experiments. Missing values are
left missing for the missing-data experiments.

Run: python extract_respiratory_features.py
"""
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]   # D:/complete

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eicu_sources import get_eicu_table_path  # noqa: E402

LANDMARK = 1440
LAB_START = -360
CHUNK = 2_000_000
COHORT = PROJECT / "respiratory" / "raw" / "source" / "dataset_verified" / "respiratory_cohort_24h_landmark.csv"
OUT_DIR = PROJECT / "respiratory" / "raw"

# source column -> (feature stem, low, high); kept when low < value <= high
VITAL_PERIODIC = {
    "heartrate": ("heart_rate", 0, 300),
    "respiration": ("resp_rate", 0, 70),
    "sao2": ("spo2", 0, 100),
    "temperature": ("temperature_c", 25, 45),
    "systemicsystolic": ("art_sbp", 0, 300),
    "systemicdiastolic": ("art_dbp", 0, 250),
    "systemicmean": ("art_map", 0, 300),
}
VITAL_APERIODIC = {
    "noninvasivesystolic": ("nibp_sbp", 0, 300),
    "noninvasivediastolic": ("nibp_dbp", 0, 250),
    "noninvasivemean": ("nibp_map", 0, 300),
}
# eICU labname -> (feature stem, low, high); kept when low <= value <= high
LABS = {
    "sodium": ("sodium", 100, 200),
    "potassium": ("potassium", 1, 12),
    "chloride": ("chloride", 50, 160),
    "bicarbonate": ("bicarbonate", 2, 60),
    "anion gap": ("anion_gap", -10, 60),
    "BUN": ("bun", 1, 300),
    "creatinine": ("creatinine", 0.1, 25),
    "glucose": ("glucose", 10, 2000),
    "calcium": ("calcium", 3, 20),
    "magnesium": ("magnesium", 0.3, 10),
    "Hgb": ("hemoglobin", 2, 25),
    "WBC x 1000": ("wbc", 0, 500),
    "platelets x 1000": ("platelets", 1, 2000),
    "RDW": ("rdw", 8, 40),
    "-lymphs": ("lymphocytes_pct", 0, 100),
    "pH": ("ph", 6.5, 8.0),
    "paO2": ("pao2", 10, 700),
    "paCO2": ("paco2", 5, 200),
    "FiO2": ("fio2", 21, 100),
    "Base Excess": ("base_excess", -40, 40),
    "albumin": ("albumin", 0.5, 7),
    "total bilirubin": ("bilirubin", 0, 70),
    "AST (SGOT)": ("ast", 1, 20000),
    "ALT (SGPT)": ("alt", 1, 20000),
    "lactate": ("lactate", 0.1, 40),
    "PT - INR": ("inr", 0.5, 20),
}
APS_INPUTS = {  # apacheApsVar: APACHE day-1 inputs; -1 means not recorded
    "intubated": "apache_day1_intubated",
    "vent": "apache_day1_ventilated",
    "dialysis": "apache_day1_dialysis",
    "eyes": "gcs_eyes",
    "motor": "gcs_motor",
    "verbal": "gcs_verbal",
    "meds": "gcs_unable_due_to_meds",
    "urine": "urine_output_day1_ml",
}
PRED_COMORBIDITY = {  # apachePredVar chronic-health inputs only
    "aids": "comorb_aids",
    "hepaticfailure": "comorb_hepatic_failure",
    "lymphoma": "comorb_lymphoma",
    "metastaticcancer": "comorb_metastatic_cancer",
    "leukemia": "comorb_leukemia",
    "immunosuppression": "comorb_immunosuppression",
    "cirrhosis": "comorb_cirrhosis",
    "diabetes": "comorb_diabetes",
    "readmit": "icu_readmission",
}
EXCLUDED_SOURCES = {
    "apachePatientResult": "all columns: APACHE predicted/actual mortality, LOS and outcomes",
    "apachePredVar": "diedinhospital, dischargelocation, var03hspxlos, saps3*, admitdiagnosis (timing), ejectfx/amilocation (sparse)",
    "patient": "discharge offsets/status/location/weight, wardid, hospitalid (domain only), uniquepid (split group only)",
    "diagnosis/pastHistory/treatment": "not used as predictors: entry timing and documentation completeness vary",
}
TOP_ADMISSION_DX = 20


class StayAggregator:
    """Running sum/count/min/max per (stay, variable) without holding raw rows."""

    def __init__(self, stays, names):
        self.index = pd.Index(stays)
        n = len(stays)
        self.sum = {k: np.zeros(n) for k in names}
        self.count = {k: np.zeros(n, dtype=np.int64) for k in names}
        self.min = {k: np.full(n, np.inf) for k in names}
        self.max = {k: np.full(n, -np.inf) for k in names}

    def update(self, stays, name, values):
        pos = self.index.get_indexer(stays)
        if (pos < 0).any():
            raise ValueError("Row outside cohort reached aggregator")
        n = len(self.index)
        self.sum[name] += np.bincount(pos, weights=values, minlength=n)
        self.count[name] += np.bincount(pos, minlength=n)
        np.minimum.at(self.min[name], pos, values)
        np.maximum.at(self.max[name], pos, values)

    def frame(self, stats):
        out = {}
        for name in self.sum:
            has = self.count[name] > 0
            if "mean" in stats:
                out[f"{name}_mean"] = np.where(has, self.sum[name] / np.maximum(self.count[name], 1), np.nan)
            if "min" in stats:
                out[f"{name}_min"] = np.where(has, self.min[name], np.nan)
            if "max" in stats:
                out[f"{name}_max"] = np.where(has, self.max[name], np.nan)
        return pd.DataFrame(out, index=self.index)


def stream_vitals(table, spec, stays, audit):
    path = get_eicu_table_path(table, verify=False)
    names = [stem for stem, _, _ in spec.values()]
    agg = StayAggregator(stays, names)
    stay_set = set(stays)
    removed = dict.fromkeys(names, 0)
    rows_in_window = 0
    cols = ["patientunitstayid", "observationoffset", *spec]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK):
        chunk = chunk[chunk.patientunitstayid.isin(stay_set)
                      & chunk.observationoffset.between(0, LANDMARK)]
        rows_in_window += len(chunk)
        for source, (stem, low, high) in spec.items():
            values = chunk[source]
            present = values.notna()
            valid = present & (values > low) & (values <= high)
            removed[stem] += int((present & ~valid).sum())
            agg.update(chunk.loc[valid, "patientunitstayid"].to_numpy(), stem, values[valid].to_numpy(float))
    audit[table] = {"rows_in_cohort_window": rows_in_window, "window_minutes": [0, LANDMARK],
                    "implausible_values_removed": removed}
    return agg


def stream_nurse_temperature(agg, stays, audit):
    """Add nurse-charted Temperature (C) to the monitor temperature aggregate.

    Monitor temperature is absent for most stays; routine nursing temperatures are
    the main source. Each reading is charted as (C) and (F) duplicates, so only (C)
    is used. Readings must be observed 0..1440 min AND entered by 1440 min.
    """
    path = get_eicu_table_path("nurseCharting", verify=False)
    stem, low, high = VITAL_PERIODIC["temperature"]
    stay_set = set(stays)
    used = removed = 0
    cols = ["patientunitstayid", "nursingchartoffset", "nursingchartentryoffset",
            "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK, dtype={"nursingchartvalue": "string"}):
        chunk = chunk[(chunk.nursingchartcelltypevalname == "Temperature (C)")
                      & chunk.patientunitstayid.isin(stay_set)
                      & chunk.nursingchartoffset.between(0, LANDMARK)
                      & (chunk.nursingchartentryoffset <= LANDMARK)]
        values = pd.to_numeric(chunk.nursingchartvalue, errors="coerce")
        valid = values.notna() & (values > low) & (values <= high)
        removed += int((~valid).sum())
        used += int(valid.sum())
        agg.update(chunk.loc[valid, "patientunitstayid"].to_numpy(), stem, values[valid].to_numpy(float))
    audit["nurseCharting_temperature"] = {"readings_used": used, "invalid_or_nonnumeric_removed": removed,
                                          "rule": "Temperature (C) rows; observed 0..1440 and entered <=1440 min"}


def stream_labs(stays, audit):
    path = get_eicu_table_path("lab", verify=False)
    stems = [stem for stem, _, _ in LABS.values()]
    agg = StayAggregator(stays, stems + ["pf_ratio"])
    stay_set = set(stays)
    removed = dict.fromkeys(stems, 0)
    revised_after_landmark = 0
    rows_in_window = 0
    cols = ["patientunitstayid", "labresultoffset", "labname", "labresult", "labresultrevisedoffset"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK):
        chunk = chunk[chunk.patientunitstayid.isin(stay_set)
                      & chunk.labresultoffset.between(LAB_START, LANDMARK)
                      & chunk.labname.isin(LABS)]
        late = chunk.labresultrevisedoffset > LANDMARK
        revised_after_landmark += int(late.sum())
        chunk = chunk[~late & chunk.labresult.notna()].copy()
        rows_in_window += len(chunk)
        fio2 = chunk.labname == "FiO2"
        fraction = fio2 & chunk.labresult.between(0.21, 1.0)
        chunk.loc[fraction, "labresult"] = chunk.loc[fraction, "labresult"] * 100
        chunk["stem"] = chunk.labname.map(lambda n: LABS[n][0])
        low = chunk.labname.map(lambda n: LABS[n][1])
        high = chunk.labname.map(lambda n: LABS[n][2])
        valid = chunk.labresult.between(low, high)
        for stem, count in chunk.loc[~valid, "stem"].value_counts().items():
            removed[stem] += int(count)
        chunk = chunk[valid]
        for stem, part in chunk.groupby("stem"):
            agg.update(part.patientunitstayid.to_numpy(), stem, part.labresult.to_numpy(float))
        # PaO2/FiO2 from the same blood-gas draw (same stay and result offset).
        gas = chunk[chunk.stem.isin(["pao2", "fio2"])]
        if len(gas):
            wide = gas.pivot_table(index=["patientunitstayid", "labresultoffset"], columns="stem",
                                   values="labresult", aggfunc="median")
            if {"pao2", "fio2"} <= set(wide.columns):
                wide = wide.dropna(subset=["pao2", "fio2"])
                pf = wide.pao2 / (wide.fio2 / 100)
                pf = pf[(pf > 20) & (pf <= 1000)]
                agg.update(pf.index.get_level_values(0).to_numpy(), "pf_ratio", pf.to_numpy(float))
    audit["lab"] = {"rows_in_cohort_window": rows_in_window, "window_minutes": [LAB_START, LANDMARK],
                    "rows_dropped_revised_after_landmark": revised_after_landmark,
                    "implausible_values_removed": removed,
                    "fio2_rule": "values 0.21-1.0 treated as fractions and multiplied by 100"}
    frame = agg.frame(("min", "max"))
    frame = frame.drop(columns=["pf_ratio_max"])
    return frame


def harmonized_race(ethnicity):
    return {"Caucasian": "White", "African American": "Black", "Hispanic": "Hispanic",
            "Asian": "Other", "Native American": "Other"}.get(ethnicity, np.nan)


def age_group(age):
    return pd.cut(age, [17, 39, 64, 79, 200], labels=["18-39", "40-64", "65-79", "80+"]).astype(object)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    cohort = pd.read_csv(COHORT)
    stays = cohort.patientunitstayid.to_numpy()
    if len(set(stays)) != len(stays):
        raise ValueError("Duplicate stays in cohort")
    audit = {"cohort_rows": int(len(cohort)), "landmark_minutes": LANDMARK, "excluded_sources": EXCLUDED_SOURCES}

    patient = pd.read_csv(get_eicu_table_path("patient", verify=False),
                          usecols=["patientunitstayid", "apacheadmissiondx", "admissionheight", "admissionweight",
                                   "hospitaladmitoffset", "hospitaladmitsource", "unittype", "unitadmitsource",
                                   "unitvisitnumber", "unitstaytype"])
    patient = patient[patient.patientunitstayid.isin(set(stays))].set_index("patientunitstayid").reindex(stays)

    adx = pd.read_csv(get_eicu_table_path("admissionDx", verify=False),
                      usecols=["patientunitstayid", "admitdxenteredoffset", "admitdxname"])
    adx = adx[adx.patientunitstayid.isin(set(stays))]
    adx["admitdxname"] = adx.admitdxname.str.strip()
    entered = adx.groupby(["patientunitstayid", "admitdxname"]).admitdxenteredoffset.min()

    base = pd.DataFrame(index=pd.Index(stays, name="patientunitstayid"))
    base["age"] = cohort.set_index("patientunitstayid").age_years_or_topcode90.reindex(stays).to_numpy(float)
    base["age_over89"] = cohort.set_index("patientunitstayid").age_is_topcoded_over89.reindex(stays).to_numpy()
    gender = cohort.set_index("patientunitstayid").gender.reindex(stays)
    base["male"] = gender.map({"Male": 1, "Female": 0}).to_numpy(float)
    height = patient.admissionheight.where(patient.admissionheight.between(120, 230))
    weight = patient.admissionweight.where(patient.admissionweight.between(25, 350))
    bmi = weight / (height / 100) ** 2
    base["admission_weight_kg"] = weight.to_numpy()
    base["admission_bmi"] = bmi.where(bmi.between(10, 100)).to_numpy()
    offset = patient.hospitaladmitoffset
    base["pre_icu_hospital_hours"] = (-offset / 60).where(offset <= 0).to_numpy()
    for src, dst in [("unittype", "unit_type"), ("unitadmitsource", "unit_admit_source"),
                     ("hospitaladmitsource", "hospital_admit_source"), ("unitstaytype", "unit_stay_type")]:
        base[dst] = patient[src].astype("string").str.strip().replace("", pd.NA).to_numpy()
    base["unit_visit_number"] = patient.unitvisitnumber.to_numpy()
    dx = patient.apacheadmissiondx.astype("string").str.strip()
    dx_time = pd.Series([entered.get((s, d), np.nan) if isinstance(d, str) else np.nan
                         for s, d in zip(stays, dx)], index=stays)
    dx_by24 = dx.where(dx_time.to_numpy() <= LANDMARK)
    top = dx_by24.value_counts().head(TOP_ADMISSION_DX).index
    base["admission_dx_by24h"] = dx_by24.where(dx_by24.isin(top) | dx_by24.isna(), "Other").to_numpy()
    audit["patient"] = {
        "height_cm_valid": [120, 230], "weight_kg_valid": [25, 350], "bmi_valid": [10, 100],
        "hospital_admit_after_icu_admit_set_missing": int((offset > 0).sum()),
        "admission_dx_available_by24h": int(dx_by24.notna().sum()),
        "admission_dx_entered_after24h_or_missing": int(dx_by24.isna().sum()),
        "admission_dx_categories_kept": [str(x) for x in top],
    }

    aps = pd.read_csv(get_eicu_table_path("apacheApsVar", verify=False), usecols=["patientunitstayid", *APS_INPUTS])
    aps = aps[aps.patientunitstayid.isin(set(stays))].drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    aps = aps.where(aps >= 0).rename(columns=APS_INPUTS).reindex(stays)
    pred = pd.read_csv(get_eicu_table_path("apachePredVar", verify=False), usecols=["patientunitstayid", *PRED_COMORBIDITY])
    pred = pred[pred.patientunitstayid.isin(set(stays))].drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    pred = pred.where(pred >= 0).rename(columns=PRED_COMORBIDITY).reindex(stays)
    audit["apache"] = {"stays_with_apache_inputs": int(aps.notna().any(axis=1).sum())}

    print("streaming vitalPeriodic ...", flush=True)
    periodic_agg = stream_vitals("vitalPeriodic", VITAL_PERIODIC, stays, audit)
    print(f"  done {time.time() - t0:.0f}s; streaming nurseCharting temperature ...", flush=True)
    stream_nurse_temperature(periodic_agg, stays, audit)
    periodic = periodic_agg.frame(("mean", "min", "max"))
    print(f"  done {time.time() - t0:.0f}s; streaming vitalAperiodic ...", flush=True)
    aperiodic = stream_vitals("vitalAperiodic", VITAL_APERIODIC, stays, audit).frame(("mean", "min", "max"))
    print(f"  done {time.time() - t0:.0f}s; streaming lab ...", flush=True)
    labs = stream_labs(stays, audit)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    c = cohort.set_index("patientunitstayid").reindex(stays)
    meta = pd.DataFrame({
        "meta_patientunitstayid": stays,
        "meta_split_group_uniquepid": c.uniquepid.to_numpy(),
        "meta_domain_hospitalid": c.hospitalid.to_numpy(),
        "meta_domain_region": c.hospital_region.to_numpy(),
        "meta_route_apache_dx_by24h": c.respiratory_apache_admission_flag.to_numpy(),
        "meta_route_primary_pulmonary_by24h": c.primary_pulmonary_diagnosis_by24h_flag.to_numpy(),
    }, index=stays)
    sens = pd.DataFrame({
        "sens_sex": gender.map({"Male": "Male", "Female": "Female"}).to_numpy(),
        "sens_age_group": age_group(base["age"]).to_numpy(),
        "sens_race_ethnicity": c.ethnicity.map(harmonized_race).to_numpy(),
        "sens_race_detail": c.ethnicity.where(c.ethnicity.notna() & (c.ethnicity != "Other/Unknown")).to_numpy(),
    }, index=stays)
    target = pd.DataFrame({"target": c.hospital_mortality.to_numpy()}, index=stays)

    out = pd.concat([meta, sens, base, aps, pred, periodic, aperiodic, labs, target], axis=1)
    if len(out) != len(cohort) or out.target.isna().any():
        raise ValueError("Row alignment failure")
    path = OUT_DIR / "respiratory_raw.csv"
    out.to_csv(path, index=False)
    features = [col for col in out.columns if not col.startswith(("meta_", "sens_")) and col != "target"]
    audit["output"] = str(path)
    audit["rows"] = int(len(out))
    audit["feature_count"] = len(features)
    audit["target_positive"] = int(out.target.sum())
    audit["feature_missing_rate"] = {f: round(float(out[f].isna().mean()), 4) for f in features}
    audit["seconds"] = round(time.time() - t0)
    (OUT_DIR / "respiratory_feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {path} rows={len(out)} features={len(features)} in {audit['seconds']}s")


if __name__ == "__main__":
    main()
