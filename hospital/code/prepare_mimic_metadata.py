"""Audit verified MIMIC metadata and prepare membership, not clinical features.

Does not read chartevents, fit a model, or mark the MIMIC feature table ready.
"""
from pathlib import Path
import hashlib
import json

from prepare_cross_dataset import ROOT, pd, mimic_cohort, MIMIC_ITEMS, BASELINE_VITALS
from streaming_sources import read_official_csv, load_segment_manifest

EXPECTED_ITEMS = {
    220045: ("Heart Rate", "bpm"),
    220210: ("Respiratory Rate", "insp/min"),
    224690: ("Respiratory Rate (Total)", "insp/min"),
    220277: ("O2 saturation pulseoxymetry", "%"),
    223761: ("Temperature Fahrenheit", "°F"),
    223762: ("Temperature Celsius", "°C"),
}


def main():
    raw = ROOT / "raw/mimiciv/3.1"
    output = ROOT / "prepared"
    output.mkdir(exist_ok=True)
    required = {
        "patients": ("hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year"]),
        "admissions": ("hosp/admissions.csv.gz", ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "hospital_expire_flag"]),
        "icustays": ("icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]),
        "d_items": ("icu/d_items.csv.gz", ["itemid", "label", "unitname"]),
    }
    tables, sources = {}, {}
    for table, (relative, columns) in required.items():
        path = raw / relative
        # Require publisher-verified segmented sources for this production audit.
        _, manifest = load_segment_manifest(path, verify=True)
        tables[table] = read_official_csv(path, usecols=columns)
        sources[table] = {"relative_name": relative, "rows": len(tables[table]),
                          "sha256": manifest["sha256"], "publisher_sha256": manifest["expected_sha256"],
                          "stored_bytes_rehashed": True, "acquisition": manifest.get("acquisition")}
    assert tables["patients"].subject_id.is_unique
    assert tables["admissions"].hadm_id.is_unique
    assert tables["icustays"].stay_id.is_unique
    assert tables["d_items"].itemid.is_unique

    items = tables["d_items"].set_index("itemid")
    mapping = []
    for itemid, (label, unit) in EXPECTED_ITEMS.items():
        if itemid not in items.index:
            raise ValueError(f"Required vital-sign item absent from MIMIC release: {itemid}")
        observed = items.loc[itemid]
        if observed["label"] != label or observed.unitname != unit:
            raise ValueError(f"MIMIC item dictionary differs from the reviewed label/unit mapping: {itemid}")
        mapping.append({"itemid": itemid, "source_label": observed["label"], "source_unit": observed.unitname,
                        "mapped_concept": MIMIC_ITEMS[itemid],
                        "in_common_baseline": MIMIC_ITEMS[itemid] in BASELINE_VITALS,
                        "unit_conversion": "(F - 32) / 1.8" if itemid == 223761 else "identity",
                        "label_and_unit_match": True})
    pd.DataFrame(mapping).to_csv(output / "mimic_item_mapping_validation.csv", index=False)

    cohort, flow = mimic_cohort(tables["patients"], tables["admissions"], tables["icustays"])
    manifest = cohort.rename(columns={"_intime": "icu_intime"}).copy()
    manifest["prediction_landmark"] = manifest.icu_intime + pd.Timedelta(hours=24)
    target_path = output / "mimic_24h_cohort_manifest.csv"
    temporary = target_path.with_suffix(".csv.partial")
    manifest.to_csv(temporary, index=False)
    temporary.replace(target_path)

    # Independently confirm every membership row and label against the joined
    # original metadata; this also exposes dropped/unmatched source records.
    joined = tables["icustays"].merge(tables["admissions"], on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    joined = joined.merge(tables["patients"], on="subject_id", how="inner", validate="many_to_one")
    for name in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
        joined[name] = pd.to_datetime(joined[name], errors="coerce")
    age = joined.anchor_age + joined.intime.dt.year - joined.anchor_year
    landmark = joined.intime + pd.Timedelta(hours=24)
    eligible = (age.notna() & age.ge(18) & joined.stay_id.notna() & joined.subject_id.notna()
                & joined.hospital_expire_flag.isin([0, 1]) & joined.outtime.gt(landmark)
                & joined.dischtime.gt(landmark) & joined.admittime.le(joined.intime)
                & (joined.deathtime.isna() | joined.deathtime.gt(landmark)))
    expected = joined.loc[eligible].set_index("stay_id")
    assert set(manifest.stay_id) == set(expected.index)
    aligned = expected.loc[manifest.stay_id]
    assert list(manifest.patient_id) == list("mimic:" + aligned.subject_id.astype("int64").astype("string"))
    assert list(manifest.hospital_mortality) == list(aligned.hospital_expire_flag.astype(int))
    assert manifest.stay_id.is_unique and manifest.age_group.notna().all()
    assert manifest.hospital_mortality.isin([0, 1]).all()
    forbidden = {"outtime", "dischtime", "deathtime", "hospital_expire_flag", "los"}
    assert not forbidden.intersection(manifest.columns)
    validation = {
        "passed": True, "source": "MIMIC-IV v3.1 metadata only", "sources": sources,
        "source_icu_stays": len(tables["icustays"]), "linked_icu_stays": len(joined),
        "rows": len(manifest), "patients": int(manifest.patient_id.nunique()), "hospitals": 1,
        "positive": int(manifest.hospital_mortality.sum()), "negative": int(manifest.hospital_mortality.eq(0).sum()),
        "cohort_flow": flow, "source_item_label_and_unit_checks": mapping,
        "unique_stay_ids": True, "independent_source_eligibility_and_label_check": True,
        "known_adult_age_required": True, "prediction_landmark_hours": 24,
        "outcome": "Subsequent in-hospital mortality among eligible adult ICU stays alive at24h",
        "clinical_features_extracted": False, "mimic_feature_table_prepared": False,
        "evaluation_completed": False,
        "manifest_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "next_required_source": "icu/chartevents.csv.gz, completed and publisher-checksum verified",
    }
    (output / "mimic_cohort_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    dictionary = [{"column": col, "role": "target" if col == "hospital_mortality" else "cohort/ID/demographic metadata",
                   "notes": "No measured clinical features in this membership manifest"} for col in manifest.columns]
    pd.DataFrame(dictionary).to_csv(output / "mimic_cohort_dictionary.csv", index=False)
    # Deliberately do not change production status.json or mimic_prepared flags.
    print(json.dumps({key: validation[key] for key in ["passed", "source_icu_stays", "linked_icu_stays", "rows", "patients", "positive", "negative", "clinical_features_extracted"]}), flush=True)


if __name__ == "__main__":
    main()
