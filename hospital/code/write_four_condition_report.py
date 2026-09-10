"""Publish current, evidence-based dataset status without training any models."""
from pathlib import Path
import csv
import json

from download_mimic import ROOT, save_json, utc_now

PROJECT = ROOT.parent


def read(relative):
    return json.loads((PROJECT / relative).read_text(encoding="utf-8"))


def main():
    state = read("cross_dataset/four_conditions/preprocessing_status.json")
    baseline = read("preprocessing/outputs/cross_dataset_baseline/baseline_preprocessing_status.json")
    clinical = read("cross_dataset/four_conditions/mimic/clinical_preprocessing_validation.json")
    integrity = read("cross_dataset/mimic_download_integrity.json")
    independent = read("cross_dataset/four_conditions/independent_validation.json")
    if not independent.get("passed"):
        raise ValueError("Independent diagnosis/feature/split validation must pass before publishing")
    if not state["completed"] or not baseline["completed"] or not clinical["passed"]:
        raise ValueError("Do not publish a completed preprocessing claim before all stages pass")
    mimic, eicu = state["sources"]["mimic"], state["sources"]["eicu"]
    # Close the delegated source-archive CRC evidence using the actual full read.
    for record in integrity["files"]:
        if record["relative_name"] == "icu/chartevents.csv.gz":
            if not clinical["chartevents_gzip_crc_verified_by_full_stream"]:
                raise ValueError("chartevents CRC evidence missing")
            if clinical["sources"]["icu/chartevents.csv.gz"]["sha256"] != record["sha256"]:
                raise ValueError("The clinical full-stream check refers to a different chartevents archive")
            record.update(gzip_crc="passed", gzip_crc_evidence="four_conditions/mimic/clinical_preprocessing_validation.json")
        if not record.get("sha256_matches_publisher") or not str(record.get("gzip_crc", "")).startswith("passed"):
            raise ValueError("All source checks must pass before publishing: " + record["relative_name"])
    if len(integrity["files"]) != 9:
        raise ValueError("The source report must cover exactly the nine selected archives")
    integrity.update(status="all_nine_sha256_and_gzip_crc_passed", all_nine_gzip_crc_verified=True,
                     completed_utc=utc_now())
    save_json(ROOT / "mimic_download_integrity.json", integrity)
    state["baseline_transformations_fitted"] = True
    state["baseline_fit_partition"] = "Source training rows only; no labels used"
    state["baseline_artifacts"] = "../../preprocessing/outputs/cross_dataset_baseline/baseline_preprocessing_status.json"
    save_json(ROOT / "four_conditions/preprocessing_status.json", state)
    lines = [
        "# MIMIC and eICU: four-condition preprocessing\n",
        "**Source check passed. Preprocessing completed on 9 November 2026.** The nine MIMIC-IV v3.1 "
        "downloads match the saved PhysioNet publisher SHA256 manifest and pass full gzip checks. "
        "These are real deidentified records. The four analysis tables are research-derived, with the explicit definitions below.\n",
        f"MIMIC final population: **{mimic['rows']:,} ICU stays / {mimic['patients']:,} patients / one institution (BIDMC)**. "
        f"eICU comparison population after documentation eligibility and split purging: **{eicu['rows']:,} stays / "
        f"{eicu['patients']:,} patients / {eicu['hospitals']} hospitals**.\n",
        "| Application | Recorded diagnosis definition | MIMIC positives | eICU positives | Files |",
        "|---|---|---:|---:|---|",
    ]
    definitions = {
        "heart": "Ischemic heart disease: ICD-9 410-414 / ICD-10 I20-I25",
        "respiratory": "Respiratory-system disease: 460-519 / J00-J99",
        "diabetes": "Non-gestational diabetes: 249/250 / E08,E09,E10,E11,E13",
        "hypertension": "Systemic hypertension: 401-405 / I10,I11,I12,I13,I15",
    }
    for condition, definition in definitions.items():
        m, e = mimic["conditions"][condition], eicu["conditions"][condition]
        lines.append(f"| {condition.title()} | {definition} | {m['positive']:,} | {e['positive']:,} | "
                     f"[MIMIC](mimic/{condition}_model_ready.csv), [eICU](eicu/{condition}_model_ready.csv) |")
    lines += [
        "\neICU labels combine matching active-diagnosis ICD codes with explicit "
        "[structured chronic-history](https://eicu.mit.edu/eicutables/pasthistory/) positives "
        "(MI/angina; asthma/COPD/restrictive disease; diabetes; treated hypertension). No diagnosis is inferred from a procedure or treatment alone. "
        "The code-only flags and exact history rules remain in the label audits for sensitivity analyses.\n",
        "## Meaning and suitability\n",
        "These data support four separate **retrospective recorded-diagnosis classification** experiments for the project's fairness framework. "
        "A positive means that the selected condition was documented. Zero means no matching documentation among encounters with other parseable diagnoses; "
        "it is **not confirmed absence of disease**. Encounters without parseable ICD documentation are excluded, including history-only patients without documented-control eligibility. "
        "The conditions overlap and are not four mutually exclusive patient groups.\n",
        "Heart here means ischemic heart disease, not every cardiac condition. Respiratory uses the stated ICD chapter families, "
        "not every respiratory symptom or COVID-only U07.1 record. MIMIC discharge coding and eICU ICU/history documentation differ. "
        "The study measures both clinical and documentation shifts; these labels are not proven equivalent clinical ground truth.\n",
        "[MIMIC diagnosis documentation](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html) specifies discharge billing codes. "
        "They do not establish diagnosis onset or availability by 24 hours. This is not a future-onset, early-detection or causal study. "
        "Your original eICU respiratory mortality task remains separate in `D:/complete/model_ready/respiratory_model_ready.csv`.\n",
        "## Preprocessing completed\n",
        "- Adults alive and still in ICU/hospital beyond 24 hours; original known-hospital-outcome eligibility retained in both sources.",
        f"- Streamed all {clinical['event_audit']['rows_read']:,} MIMIC chart-event rows. Measurements must be charted during the first 24 hours "
        "and have a known documentation time no later than the landmark. Late records, out-of-range values and unit conflicts are excluded/audited.",
        "- Shared baseline: nine hourly-derived heart-rate, respiratory-rate and oxygen-saturation summaries plus age group and sex. "
        "MIMIC blood-pressure and temperature summaries are available separately as 21 optional features, outside the shared baseline.",
        "- Explicit feature allowlist excludes diagnosis codes/history, identifiers, other condition labels, discharge variables, treatments, glucose and HbA1c. "
        "Race/ethnicity is retained for auditing. Age and sex are both baseline predictors and available subgroup attributes.",
        "- MIMIC splits group all stays of a patient together. eICU splits hold out whole hospitals and purge patients crossing partitions.",
        "- Separate numerical matrices use source-training medians, missing indicators, scaling and one-hot categories. "
        "No validation, test or external records fit these transformations. Missing values remain available in the original model-ready CSVs.",
        "- Both transfer directions have saved transformers, aligned labels/metadata and matrices. This preprocessing stage fits no prediction model. Subsequent latent and in-processing results are saved in `latent/outputs/hospital` and `inprocessing/outputs/REPORT.md` at the project root.\n",
        "eICU has observation offsets but lacks a matching entry-time field for periodic vitals; its measurement availability cannot be checked exactly like MIMIC's storetime. "
        "Source missingness and acquisition differences must be reported.\n",
        "## Cross-hospital use\n",
        "Use eICU training hospitals, tune only on eICU validation hospitals, and evaluate on eICU test hospitals and untouched MIMIC records. "
        "The reverse direction is a separately specified experiment using a MIMIC-training-fitted transformer. "
        "MIMIC itself contains one institution, so its internal split is patient-held-out, not hospital-held-out. "
        "Public anonymised eICU hospital IDs cannot establish institutional non-overlap with BIDMC across databases. "
        "The CDC survey datasets have different populations/features and should remain separate benchmarks; do not concatenate them with ICU tables.\n",
        "## Files and evidence\n",
        "- `mimic/` and `eicu/`: four `*_model_ready.csv` files each; `target` is the binary label.",
        "- [MIMIC-source matrices](../../preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu) and [eICU-source matrices](../../preprocessing/outputs/cross_dataset_baseline/eicu_to_mimic): fitted preprocessing, X/y NPZ files and aligned rows CSVs.",
        "- [Definitions](phenotype_definitions.json), [cohort/split summary](preprocessing_status.json), "
        "[source integrity](../mimic_download_integrity.json), [clinical extraction](mimic/clinical_preprocessing_validation.json).",
        "- [Independent validation of all eight final datasets](independent_validation.json): exact labels, eligibility, clinical-feature values and patient/hospital separation passed.",
        "- [Independent source review](../../dataset_audit/MIMIC_FOUR_PHENOTYPE_REVIEW.md), "
        "[eICU history evidence](eicu/history_evidence_audit.json), [label union audit](eicu/primary_label_union_audit.json).\n",
        "Official sources: [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and "
        "[PhysioNet eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/). "
        "The original downloaded archives are preserved; compressed storage is split into FAT32-safe parts on D:.\n",
        "## Reproduce\n",
        "```powershell\npython cross_dataset/prepare_four_condition_tables.py --labels-only\n"
        "python cross_dataset/prepare_mimic_clinical_features.py\n"
        "python cross_dataset/prepare_four_condition_tables.py --finalize\n"
        "python preprocessing/code/prepare_four_condition_baseline.py\n"
        "python cross_dataset/validate_four_condition_tables.py\n"
        "python cross_dataset/write_four_condition_report.py\n```\n",
        "The clinical extractor preserves an existing output rather than overwriting it; use a fresh `--output-dir` for a deliberate rerun. "
        "Use the declared feature list or numerical matrices. Never load all CSV columns as predictors.\n",
    ]
    (ROOT / "four_conditions/README.md").write_text("\n".join(lines), encoding="utf-8")
    source_state = {
        "mode": "four_condition_sources_verified_and_preprocessed", "recorded_utc": utc_now(),
        "requested_applications": list(definitions), "raw_records_synthetic": False,
        "new_cohort_construction_authorized_to_continue": True,
        "authorization": "User requested preprocessing if source checks passed; delegated respiratory study outcome choice",
        "mimic_nine_sources_sha256_and_crc_verified": True, "preprocessing_complete": True,
        "baseline_transformations_fitted_on_source_train_only": True, "prediction_models_trained": 0,
        "cross_hospital_evaluation_completed": False, "clinically_equivalent_label_ascertainment_verified": False,
        "background_mortality_runner_still_stopped": True,
        "current_report": "cross_dataset/four_conditions/README.md",
        "important_limitations": ["Recorded diagnoses, not confirmed disease absence or onset",
            "MIMIC hospital coding and eICU ICU/history ascertainment differ", "MIMIC is one institution",
            "Original respiratory mortality task is separate", "Public eICU identities cannot prove institution non-overlap with MIMIC"]}
    save_json(PROJECT / "dataset_audit/source_verification_status.json", source_state)
    root_lines = ["# Current project dataset status\n",
        "**MIMIC source verification and four-condition preprocessing are complete.** "
        "All nine downloaded archives match the saved PhysioNet publisher checksums and pass full gzip checks.\n",
        "The additional MIMIC-eICU study now contains four separate recorded-diagnosis classification tasks: "
        "ischemic heart disease, respiratory-system disease, non-gestational diabetes and systemic hypertension. "
        "These are research-derived tables with documentation-based labels, not four official prepackaged datasets.\n",
        f"MIMIC: **{mimic['rows']:,} stays / {mimic['patients']:,} patients**. eICU comparison: "
        f"**{eicu['rows']:,} stays / {eicu['hospitals']} hospitals** after eligibility and split purging.\n",
        "[Open the four-condition datasets, counts, definitions and validation](cross_dataset/four_conditions/README.md).\n",
        "## Original application datasets\n",
        "| Application | Trusted source | Existing task | File |",
        "|---|---|---|---|",
        "| Heart | CDC BRFSS 2022 | Self-reported heart attack/CHD/angina | [Dataset](model_ready/heart_model_ready.csv) |",
        "| Hypertension | CDC NHIS 2023 | Self-reported diagnosed hypertension | [Dataset](model_ready/hypertension_model_ready.csv) |",
        "| Diabetes | CDC NHANES 2021-2023 | Self-reported diagnosed diabetes | [Dataset](model_ready/diabetes_model_ready.csv) |",
        "| Respiratory | Full eICU-CRD v2.0 | Subsequent in-hospital mortality, 24h respiratory cohort | [Dataset](model_ready/respiratory_model_ready.csv) |\n",
        "The original respiratory dataset has **28,842 stays** after the APACHE entry-time correction. "
        "It is preserved separately from the new respiratory recorded-diagnosis classification task.\n",
        "Use eICU hospital partitions for a hospital-held-out evaluation, and MIMIC as a separate external database. "
        "MIMIC has one institution. MIMIC discharge coding and eICU ICU/history documentation differ; "
        "missing codes do not establish disease absence. Surveys remain separate population benchmarks.\n",
        "Preprocessing is complete, including source-train-only imputation/scaling/encoding. "
        "No four-condition prediction models, cross-hospital results or bias-mitigation results have been produced. "
        "The older general ICU mortality runner remains stopped.\n",
        "[Source review](dataset_audit/MIMIC_FOUR_PHENOTYPE_REVIEW.md) | "
        "[Download integrity](cross_dataset/mimic_download_integrity.json) | "
        "[Historical authorised cleanup](dataset_audit/cleanup_completed.json).\n"]
    (PROJECT / "DATASET_STATUS.md").write_text("\n".join(root_lines), encoding="utf-8")
    scope = ["# Current four-condition generalization scope\n",
        "The user requested heart, respiratory, diabetes and hypertension, and authorised source checks followed by preprocessing. "
        "They delegated the respiratory outcome choice for this additional study.\n",
        "The additional hospital study uses four separate retrospective recorded-diagnosis classification targets. "
        "The original respiratory mortality application is preserved separately.\n",
        "[Current datasets, phenotype rules, preprocessing and limitations](four_conditions/README.md).\n",
        "Train/validate/test hospitals are separated in eICU; repeated patients crossing these partitions are purged. "
        "MIMIC is an external database from one institution, with its own patient-separated partitions for the reverse direction. "
        "Harmonised predictors and code families do not prove identical clinical label ascertainment. "
        "The CDC survey tasks remain population benchmarks, separate from these ICU classification tasks.\n",
        "Preprocessing is complete. Predictive model fitting, four-condition cross-hospital evaluation and mitigation comparisons remain to be done.\n"]
    (ROOT / "DISEASE_GENERALIZATION_SCOPE.md").write_text("\n".join(scope), encoding="utf-8")
    approved_path = PROJECT / "dataset_audit/approved_datasets.json"
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    for item in approved:
        if item.get("application", "").casefold() == "respiratory":
            item["rows"] = 28842
    save_json(approved_path, approved)
    bundle_path = PROJECT / "dataset_audit/dataset_bundle_summary.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.update(mimic_status="Nine sources verified; four-condition preprocessing complete; evaluation pending",
                  mimic_downloaded=True, mimic_verified_source_files=9, mimic_four_condition_rows=mimic["rows"],
                  eicu_four_condition_rows=eicu["rows"], cross_hospital_evaluation_completed=False,
                  current_four_condition_report="cross_dataset/four_conditions/README.md")
    save_json(bundle_path, bundle)
    sources_path = PROJECT / "dataset_audit/dataset_sources.csv"
    with sources_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames, records = reader.fieldnames, list(reader)
    for record in records:
        if "MIMIC" in record.get("source", ""):
            record["application"] = "heart, respiratory, diabetes, hypertension recorded-diagnosis study"
            record["acquisition_status"] = "9 official archives SHA256/gzip verified; four-condition preprocessing complete"
    with sources_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    mortality_status = read("cross_dataset/status.json")
    mortality_status.update(scope="Historical general ICU mortality study; current four-condition data are separate",
                            four_condition_preprocessing_completed=True,
                            current_four_condition_status="four_conditions/preprocessing_status.json",
                            next_required_action="Four-condition model/evaluation work remains; preprocessing complete")
    save_json(ROOT / "status.json", mortality_status)
    save_json(ROOT / "four_conditions/report_status.json", {"completed": True, "written_utc": utc_now()})
    print("Current source, cohort and preprocessing reports updated.")


if __name__ == "__main__":
    main()
