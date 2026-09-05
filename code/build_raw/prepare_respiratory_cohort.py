"""Build a dataset-only 24h respiratory ICU cohort manifest from verified eICU.

No model, imputation, feature extraction, or old processed files are used.
Run from any directory: python prepare_respiratory_cohort.py
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

from eicu_sources import get_eicu_table_path

ROOT = Path(__file__).resolve().parents[2] / "respiratory" / "raw" / "source"
OUT = ROOT / "dataset_verified"
LANDMARK_MINUTES = 1440

# Same admission-diagnosis respiratory list as the existing notebook; this is a
# broad respiratory cohort, including pulmonary sepsis, vascular and surgical cases.
RESP_APACHE_DX = {
    "Sepsis, pulmonary", "Pneumonia, bacterial", "Pneumonia, aspiration",
    "Pneumonia, other", "Pneumonia, viral", "Pneumonia, fungal",
    "Pneumonia, parasitic (i.e., Pneumocystic pneumonia)",
    "Emphysema/bronchitis", "Asthma", "Arrest, respiratory (without cardiac arrest)",
    "Respiratory - medical, other",
    "ARDS-adult respiratory distress syndrome, non-cardiogenic pulmonary edema",
    "Obstruction-airway (i.e., acute epiglottitis, post-extubation edema, foreign body, etc)",
    "Weaning from mechanical ventilation (transfer from other unit or hospital only)",
    "Apnea, sleep", "Embolus, pulmonary", "Hypertension-pulmonary, primary/idiopathic",
    "Hemorrhage/hemoptysis, pulmonary", "Effusions, pleural", "Pneumothorax", "Hemothorax",
    "Restrictive lung disease (i.e., Sarcoidosis, pulmonary fibrosis)",
    "Chest pain, respiratory", "Cancer, lung", "Thoracotomy for lung cancer",
    "Thoracotomy for pleural disease", "Thoracotomy for thoracic/respiratory infection",
    "Thoracotomy for lung reduction", "Thoracotomy for bronchopleural fistula",
    "Respiratory surgery, other", "Biopsy, open lung",
    "Apnea-sleep; surgery for (i.e., UPPP - uvulopalatopharyngoplasty)",
    "Lung transplant, bilateral", "Lung transplant, single",
}

FIELDS = [
    "patientunitstayid", "patienthealthsystemstayid", "uniquepid", "hospitalid",
    "landmark_minutes", "age_raw", "age_years_or_topcode90", "age_is_topcoded_over89",
    "gender", "ethnicity", "hospital_region", "respiratory_apache_admission_flag",
    "primary_pulmonary_diagnosis_by24h_flag", "hospital_mortality",
]


def positive_integer(value):
    text = (value or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def age_metadata(value):
    raw = (value or "").strip()
    if raw.replace(" ", "") == ">89":
        return raw, 90, 1
    age = number(raw)
    return raw, age, 0


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def main():
    OUT.mkdir(exist_ok=True)
    paths = {name: get_eicu_table_path(name, verify=True)
             for name in ("patient", "hospital", "diagnosis", "admissionDx")}
    hospitals = {positive_integer(row["hospitalid"]): row for row in rows(paths["hospital"])}
    if None in hospitals:
        raise ValueError("Invalid hospital identifier in authoritative hospital table")
    # eICU documents that APACHE admission diagnoses may be entered up to 36 h after
    # ICU admission. patient.apacheadmissiondx has no timestamp, so the APACHE route
    # is accepted only when admissionDx shows that same diagnosis entered <=1440 min.
    apache_dx_entered = {}
    for row in rows(paths["admissionDx"]):
        name = row["admitdxname"].strip()
        if name not in RESP_APACHE_DX:
            continue
        stay = positive_integer(row["patientunitstayid"])
        offset = number(row["admitdxenteredoffset"])
        if stay is None or offset is None:
            continue
        key = (stay, name)
        if key not in apache_dx_entered or offset < apache_dx_entered[key]:
            apache_dx_entered[key] = offset
    pulmonary_by24 = set()
    diagnosis_counts = {"all_diagnosis_rows": 0, "primary_pulmonary_rows_any_time": 0,
                        "primary_pulmonary_rows_at_or_before24h": 0,
                        "apache_respiratory_dx_not_entered_by24h_stays": 0}
    for row in rows(paths["diagnosis"]):
        diagnosis_counts["all_diagnosis_rows"] += 1
        if row["diagnosispriority"].strip().casefold() != "primary":
            continue
        top = row["diagnosisstring"].split("|", 1)[0].strip().casefold()
        if top != "pulmonary":
            continue
        diagnosis_counts["primary_pulmonary_rows_any_time"] += 1
        offset = number(row["diagnosisoffset"])
        stay = positive_integer(row["patientunitstayid"])
        # Negative offsets are available before ICU admission and are retained.
        if stay is not None and offset is not None and offset <= LANDMARK_MINUTES:
            pulmonary_by24.add(stay)
            diagnosis_counts["primary_pulmonary_rows_at_or_before24h"] += 1

    stages = ["all_icu_stays", "valid_identifiers_and_known_hospital",
              "respiratory_evidence_available_by24h", "known_age_at_least18",
              "known_hospital_outcome_alive_or_expired", "still_in_icu_after24h",
              "still_in_hospital_after24h"]
    flow = dict.fromkeys(stages, 0)
    retained = []
    seen_stays = set()
    for row in rows(paths["patient"]):
        flow[stages[0]] += 1
        stay = positive_integer(row["patientunitstayid"])
        encounter = positive_integer(row["patienthealthsystemstayid"])
        hospital = positive_integer(row["hospitalid"])
        person = row["uniquepid"].strip()
        if stay is None or encounter is None or hospital not in hospitals or not person:
            continue
        if stay in seen_stays:
            raise ValueError("Duplicate patientunitstayid: expected one authoritative row per stay")
        seen_stays.add(stay)
        flow[stages[1]] += 1
        apache_name = row["apacheadmissiondx"].strip()
        by_apache = False
        if apache_name in RESP_APACHE_DX:
            entered = apache_dx_entered.get((stay, apache_name))
            by_apache = entered is not None and entered <= LANDMARK_MINUTES
            if not by_apache:
                diagnosis_counts["apache_respiratory_dx_not_entered_by24h_stays"] += 1
        by_pulmonary = stay in pulmonary_by24
        if not (by_apache or by_pulmonary):
            continue
        flow[stages[2]] += 1
        raw_age, age, topcoded = age_metadata(row["age"])
        if age is None or age < 18:
            continue
        flow[stages[3]] += 1
        outcome = row["hospitaldischargestatus"].strip().casefold()
        if outcome not in ("alive", "expired"):
            continue
        flow[stages[4]] += 1
        icu_discharge = number(row["unitdischargeoffset"])
        if icu_discharge is None or icu_discharge <= LANDMARK_MINUTES:
            continue
        flow[stages[5]] += 1
        hospital_discharge = number(row["hospitaldischargeoffset"])
        if hospital_discharge is None or hospital_discharge <= LANDMARK_MINUTES:
            continue
        flow[stages[6]] += 1
        retained.append({
            "patientunitstayid": stay, "patienthealthsystemstayid": encounter,
            "uniquepid": person, "hospitalid": hospital,
            "landmark_minutes": LANDMARK_MINUTES, "age_raw": raw_age,
            "age_years_or_topcode90": int(age) if age.is_integer() else age,
            "age_is_topcoded_over89": topcoded,
            "gender": row["gender"].strip(), "ethnicity": row["ethnicity"].strip(),
            "hospital_region": hospitals[hospital]["region"].strip(),
            "respiratory_apache_admission_flag": int(by_apache),
            "primary_pulmonary_diagnosis_by24h_flag": int(by_pulmonary),
            "hospital_mortality": int(outcome == "expired"),
        })
    retained.sort(key=lambda row: row["patientunitstayid"])
    output = OUT / "respiratory_cohort_24h_landmark.csv"
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(retained)
    dictionary = {
        "patientunitstayid": ("stay identifier; never a predictor", "join_key"),
        "patienthealthsystemstayid": ("hospital encounter identifier; never a predictor", "group_identifier"),
        "uniquepid": ("true eICU patient identifier for patient-disjoint splitting", "group_identifier"),
        "hospitalid": ("hospital identifier for held-out-hospital evaluation", "domain_identifier"),
        "landmark_minutes": ("1440 minutes after ICU admission; prediction time", "task_metadata"),
        "age_raw": ("original age string; >89 indicates deidentified top-coded age", "demographic_metadata"),
        "age_years_or_topcode90": ("age in years; >89 represented as 90 solely for age grouping, not exact age", "demographic_metadata"),
        "age_is_topcoded_over89": ("1 means raw age is >89; value 90 is not an exact observed age", "demographic_metadata"),
        "gender": ("source-recorded gender for demographic auditing, including unknown values", "demographic_metadata"),
        "ethnicity": ("source-recorded ethnicity for demographic auditing, including unknown values", "demographic_metadata"),
        "hospital_region": ("hospital region from hospital table; site metadata", "domain_metadata"),
        "respiratory_apache_admission_flag": ("APACHE admission diagnosis in documented respiratory list AND admissionDx.admitdxenteredoffset <=1440 min for that diagnosis; no final APACHE predictions used", "cohort_provenance"),
        "primary_pulmonary_diagnosis_by24h_flag": ("primary pulmonary diagnosis with known diagnosisoffset <=1440 minutes; negative offsets allowed", "cohort_provenance"),
        "hospital_mortality": ("1=expired during this hospitalization after 24h eligibility; 0=alive at hospital discharge", "target"),
    }
    with (OUT / "respiratory_cohort_24h_dictionary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column", "description", "role"])
        writer.writerows((name, *dictionary[name]) for name in FIELDS)
    summary = {
        "source": "eICU-CRD v2.0", "source_page": "https://physionet.org/content/eicu-crd/2.0/",
        "task": "Subsequent in-hospital mortality at a 24-hour ICU landmark in adult respiratory ICU patients",
        "landmark_minutes": LANDMARK_MINUTES,
        "eligibility": "Known adult age; valid identifiers; respiratory APACHE admission diagnosis entered <=1440 min (admissionDx timestamp) or Primary pulmonary diagnosis recorded <=1440 min; known Alive/Expired hospital outcome; ICU and hospital dischargeoffset strictly >1440 min",
        "age_topcode": ">89 retained with marker; 90 is a grouping surrogate, not exact age",
        "flow": [{"stage": name, "retained_stays": flow[name],
                  "excluded_from_previous": 0 if i == 0 else flow[stages[i-1]]-flow[name]}
                 for i, name in enumerate(stages)],
        "diagnosis_flow": diagnosis_counts,
        "final": {"stays": len(retained),
                  "patients": len({row["uniquepid"] for row in retained}),
                  "hospitals": len({row["hospitalid"] for row in retained}),
                  "hospital_deaths": sum(row["hospital_mortality"] for row in retained),
                  "hospital_survivors": sum(1-row["hospital_mortality"] for row in retained),
                  "topcoded_age_stays": sum(row["age_is_topcoded_over89"] for row in retained),
                  "apache_route_stays": sum(row["respiratory_apache_admission_flag"] for row in retained),
                  "pulmonary_primary_route_stays": sum(row["primary_pulmonary_diagnosis_by24h_flag"] for row in retained)},
        "sources": {name: {"path": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                            "checksum_verified_against_canonical_manifest": True}
                    for name, path in paths.items()},
        "cohort_manifest_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
        "no_features_extracted": True, "no_model_training": True,
        "split_status": "Not yet split. Retain repeated stays; group by uniquepid and hold out hospitalid for appropriate evaluations. Do not remove different patients by feature similarity.",
        "scope": "Cohort membership and target only; demographics are auditing metadata. No future diagnostic text, APACHE final predictions, outcomes or lengths of stay are predictor columns. This does not harmonize MIMIC/eICU or validate a prediction model.",
    }
    (OUT / "respiratory_cohort_24h_flow.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["final"], indent=2))
    print("Cohort manifest:", output)


if __name__ == "__main__":
    main()
