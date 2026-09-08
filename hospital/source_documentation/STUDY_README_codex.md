# MIMIC and eICU: four-condition preprocessing

**Source check passed. Preprocessing completed on 9 November 2026.** The nine MIMIC-IV v3.1 downloads match the saved PhysioNet publisher SHA256 manifest and pass full gzip checks. These are real deidentified records. The four analysis tables are research-derived, with the explicit definitions below.

MIMIC final population: **73,990 ICU stays / 53,931 patients / one institution (BIDMC)**. eICU comparison population after documentation eligibility and split purging: **109,378 stays / 88,955 patients / 206 hospitals**.

| Application | Recorded diagnosis definition | MIMIC positives | eICU positives | Files |
|---|---|---:|---:|---|
| Heart | Ischemic heart disease: ICD-9 410-414 / ICD-10 I20-I25 | 26,632 | 19,791 | [MIMIC](mimic/heart_model_ready.csv), [eICU](eicu/heart_model_ready.csv) |
| Respiratory | Respiratory-system disease: 460-519 / J00-J99 | 43,428 | 53,639 | [MIMIC](mimic/respiratory_model_ready.csv), [eICU](eicu/respiratory_model_ready.csv) |
| Diabetes | Non-gestational diabetes: 249/250 / E08,E09,E10,E11,E13 | 23,580 | 33,882 | [MIMIC](mimic/diabetes_model_ready.csv), [eICU](eicu/diabetes_model_ready.csv) |
| Hypertension | Systemic hypertension: 401-405 / I10,I11,I12,I13,I15 | 48,297 | 62,817 | [MIMIC](mimic/hypertension_model_ready.csv), [eICU](eicu/hypertension_model_ready.csv) |

eICU labels combine matching active-diagnosis ICD codes with explicit [structured chronic-history](https://eicu.mit.edu/eicutables/pasthistory/) positives (MI/angina; asthma/COPD/restrictive disease; diabetes; treated hypertension). No diagnosis is inferred from a procedure or treatment alone. The code-only flags and exact history rules remain in the label audits for sensitivity analyses.

## Meaning and suitability

These data support four separate **retrospective recorded-diagnosis classification** experiments for the project's fairness framework. A positive means that the selected condition was documented. Zero means no matching documentation among encounters with other parseable diagnoses; it is **not confirmed absence of disease**. Encounters without parseable ICD documentation are excluded, including history-only patients without documented-control eligibility. The conditions overlap and are not four mutually exclusive patient groups.

Heart here means ischemic heart disease, not every cardiac condition. Respiratory uses the stated ICD chapter families, not every respiratory symptom or COVID-only U07.1 record. MIMIC discharge coding and eICU ICU/history documentation differ. The study measures both clinical and documentation shifts; these labels are not proven equivalent clinical ground truth.

[MIMIC diagnosis documentation](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html) specifies discharge billing codes. They do not establish diagnosis onset or availability by 24 hours. This is not a future-onset, early-detection or causal study. Your original eICU respiratory mortality task remains separate in `D:/complete/model_ready/respiratory_model_ready.csv`.

## Preprocessing completed

- Adults alive and still in ICU/hospital beyond 24 hours; original known-hospital-outcome eligibility retained in both sources.
- Streamed all 432,997,491 MIMIC chart-event rows. Measurements must be charted during the first 24 hours and have a known documentation time no later than the landmark. Late records, out-of-range values and unit conflicts are excluded/audited.
- Shared baseline: nine hourly-derived heart-rate, respiratory-rate and oxygen-saturation summaries plus age group and sex. MIMIC blood-pressure and temperature summaries are available separately as 21 optional features, outside the shared baseline.
- Explicit feature allowlist excludes diagnosis codes/history, identifiers, other condition labels, discharge variables, treatments, glucose and HbA1c. Race/ethnicity is retained for auditing. Age and sex are both baseline predictors and available subgroup attributes.
- MIMIC splits group all stays of a patient together. eICU splits hold out whole hospitals and purge patients crossing partitions.
- Separate numerical matrices use source-training medians, missing indicators, scaling and one-hot categories. No validation, test or external records fit these transformations. Missing values remain available in the original model-ready CSVs.
- Both transfer directions have saved transformers, aligned labels/metadata and matrices. This preprocessing stage fits no prediction model. Subsequent latent and in-processing results are saved in `latent/outputs/hospital` and `inprocessing/outputs/REPORT.md` at the project root.

eICU has observation offsets but lacks a matching entry-time field for periodic vitals; its measurement availability cannot be checked exactly like MIMIC's storetime. Source missingness and acquisition differences must be reported.

## Cross-hospital use

Use eICU training hospitals, tune only on eICU validation hospitals, and evaluate on eICU test hospitals and untouched MIMIC records. The reverse direction is a separately specified experiment using a MIMIC-training-fitted transformer. MIMIC itself contains one institution, so its internal split is patient-held-out, not hospital-held-out. Public anonymised eICU hospital IDs cannot establish institutional non-overlap with BIDMC across databases. The CDC survey datasets have different populations/features and should remain separate benchmarks; do not concatenate them with ICU tables.

## Files and evidence

- `mimic/` and `eicu/`: four `*_model_ready.csv` files each; `target` is the binary label.
- [MIMIC-source matrices](../../preprocessing/outputs/cross_dataset_baseline/mimic_to_eicu) and [eICU-source matrices](../../preprocessing/outputs/cross_dataset_baseline/eicu_to_mimic): fitted preprocessing only, X/y NPZ files and aligned rows CSVs.
- [Definitions](phenotype_definitions.json), [cohort/split summary](preprocessing_status.json), [source integrity](../mimic_download_integrity.json), [clinical extraction](mimic/clinical_preprocessing_validation.json).
- [Independent validation of all eight final datasets](independent_validation.json): exact labels, eligibility, clinical-feature values and patient/hospital separation passed.
- [Independent source review](../../dataset_audit/MIMIC_FOUR_PHENOTYPE_REVIEW.md), [eICU history evidence](eicu/history_evidence_audit.json), [label union audit](eicu/primary_label_union_audit.json).

Official sources: [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and [PhysioNet eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/). The original downloaded archives are preserved; compressed storage is split into FAT32-safe parts on D:.

## Reproduce

```powershell
python cross_dataset/prepare_four_condition_tables.py --labels-only
python cross_dataset/prepare_mimic_clinical_features.py
python cross_dataset/prepare_four_condition_tables.py --finalize
python preprocessing/code/prepare_four_condition_baseline.py
python cross_dataset/validate_four_condition_tables.py
python cross_dataset/write_four_condition_report.py
```

The clinical extractor preserves an existing output rather than overwriting it; use a fresh `--output-dir` for a deliberate rerun. Use the declared feature list or numerical matrices. Never load all CSV columns as predictors.
