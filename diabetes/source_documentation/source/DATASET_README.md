# Diabetes dataset — CDC NHANES August 2021–August 2023

The selected task is **classification of self-reported, previously diagnosed diabetes among adults aged 18 or older**. The source is the US CDC / National Center for Health Statistics. It is a distinct source from the BRFSS heart and NHIS hypertension tasks. It is not a type-2-only dataset, a prospective diabetes-incidence dataset, or a validated clinical diagnosis from a single lab measurement.

Official listing: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023

## What is included

| Component | Purpose | Official data | Official documentation |
|---|---|---|---|
| DEMO_L | Demographics, fairness attributes, sample design and weights | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm |
| DIQ_L | Diabetes diagnosis question and original answers | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm |
| BMX_L | Measured BMI, waist, height and weight | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm |
| SMQ_L | Smoking history | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.htm |
| GHB_L | Separate HbA1c biomarker analysis | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm |
| GLU_L | Separate fasting-glucose biomarker analysis | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.xpt | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm |

The original files are in `raw/`. The corresponding full official HTML codebooks are in `codebooks/`. `download_manifest.json` records direct URLs, file sizes, retrieval timestamps and locally calculated SHA-256 hashes. These hashes record the downloaded file versions; they are not presented as publisher-supplied checksums.

## Cohort and label

All components are joined one-to-one by `SEQN`, beginning from the demographic participants. The binary task requires `RIDAGEYR >=18` and a valid `DIQ010` answer:

- `DIQ010=1` (yes) maps to `diabetes_diagnosed=1`.
- `DIQ010=2` (no) maps to `diabetes_diagnosed=0`.
- Borderline (`3`), refused (`7`), unknown (`9`) and missing answers are excluded from this binary cohort, with their original answers preserved in the audit.

The questionnaire asks about diabetes other than during pregnancy. A negative response indicates no reported diagnosis, and can include undiagnosed diabetes. No prediabetes class or laboratory-based diagnosis was invented.

The downloaded source contains 11,933 demographic participants and 8,153 adults. The binary cohort has **7,876 adults: 1,073 yes and 6,803 no**. Exclusions contain 3,780 participants under18, 272 adults reporting borderline diabetes, 3 adults with unavailable answers, and 2 adults answering unknown. All 4,057 excluded participants remain in `diabetes_excluded_audit.csv`.

## Files to use

- **`diabetes_baseline_raw.csv`**: the 7,876-person binary cohort. Original observed predictors and missingness are retained. There is no imputation, scaling, synthetic sampling, outlier removal, learned representation, model fitting or train/test split.
- `diabetes_all_participants_label_audit.csv`: all demographic participants joined to source diagnosis, anthropometry and smoking fields, including original diabetes questionnaire fields. It is an audit file and must not be passed to a model by selecting every non-target column.
- `diabetes_excluded_audit.csv`: complete excluded audit records and reasons.
- `diabetes_biomarkers_adults.csv`: all 8,153 adults with label status, separate HbA1c/fasting-glucose observations and their special weights. Some of these adults are excluded from the binary label task. Observed counts: 6,002 HbA1c and 3,329 fasting-glucose values.
- `data_dictionary.csv`: column definitions and feature/label/metadata roles.
- `cohort_summary.json`, `cohort_flow.csv`, `validation.json`: cohort counts, missingness and basic integrity checks.
- `derived_files_manifest.json`: hashes for the derived CSV and validation outputs.

Suggested initial predictor list, also stored in `cohort_summary.json`: `RIDAGEYR`, `RIAGENDR`, `RIDRETH3`, `education_level`, `INDFMPIR`, `BMXBMI`, `BMXWAIST`, `smoking_status`. Age, sex/gender and race/ethnicity are also retained for fairness evaluation; their inclusion as model inputs is an experimental choice that must be recorded consistently. `RIAGENDR` follows the provider's field label. `RIDAGEYR=80` means80 years or older.

`education_level` retains valid `DMDEDUC2` codes1–5 and converts refusal/unknown codes7/9 to missing. The source question is for ages20+, so missingness for ages18–19 is structural. The raw `DMDEDUC2` field is preserved. `smoking_status` respects skip patterns: `SMQ020=2` is never-smoker, even when follow-up `SMQ040` is blank; among lifetime100+ smokers, `SMQ040=3` means former and1/2 means current. Unresolved smoking answers remain missing. Raw smoking fields are retained for provenance rather than included alongside the derived category in the default feature list.

Identifiers, sample weights, stratum/PSU, diagnosis-source fields and audit flags are metadata. `DIQ010`, diabetes duration, insulin use and diabetes medicines must not become predictors for this diagnosis task. HbA1c and glucose are excluded from the baseline and stored separately. Any later experiment using biomarkers must state its target and observation setting explicitly; never include a biomarker as a predictor when it directly defines that experiment's outcome.

## Survey design and missingness

NHANES is a complex population survey and has no public hospital identifier. `SDMVSTRA` and `SDMVPSU` are masked sampling-design variables, not hospitals. This dataset can support population/fairness or cross-source evaluations, but cannot alone establish hospital generalization.

Retain `SDMVSTRA` and `SDMVPSU` for design-aware uncertainty estimates. Interview-only analyses use the relevant `WTINT2YR` interview weights. The initial predictors include measured BMI/waist, so population analyses that use those examination measurements need the relevant examined population and `WTMEC2YR` weights. Some interview-only participants have zero examination weight and structurally missing anthropometry; they were deliberately preserved to document missingness, rather than silently dropped or filled. Imputing measurements does not make a zero examination weight valid.

For HbA1c blood-analyte analyses, current `GHB_L` documentation specifies **`WTPH2YR`** phlebotomy weights. For fasting glucose analyses, use **`WTSAF2YR`**, the fasting subsample weights, rather than the full-sample weight. Use the population and nonzero weight applicable to each analysis. Combining components requires the corresponding restrictive component's design and sample definition; do not multiply survey weights or mix them across rows as interchangeable feature values.

The raw binary cohort has 1,862 missing BMI and2,069 missing waist values. Missing observations include component eligibility/participation and question skips as well as ordinary item nonresponse. Review mechanisms and subgroups before applying the later preprocessing methods. Raw class proportions should remain visible before any training-only balancing.

## Reproduce

From `D:\complete`, run:

```powershell
python .\diabetes_correct\prepare_diabetes.py --reader-dir D:\complete\_data_reader
```

The downloader uses Python's standard library, HTTPS and only the official CDC host. Preparation needs pandas and its dependencies; the existing project reader directory supplies them. Standalone environments may install pandas using their normal package setup and omit `--reader-dir`. `--download-only` downloads/validates source files; `--prepare-only` regenerates data from local sources. Existing raw files are reused; use a separate destination/version before intentionally refreshing a changed release. No private login or credential is needed for these public files.

Validation checks confirm one row per participant, valid adult ages and binary labels, complete partition into included/excluded records, and absence of biomarkers or diabetes-treatment fields from the baseline.

This folder completes dataset acquisition and transparent cohort construction. Method comparisons from the project's preprocessing, latent and in-processing tables have not been run here.
