# Dataset correction report — 7 November 2026

The dataset stage is complete: source inspection, official public downloads,
integrity checks, explicit labels, and raw cohort preparation. New downloads use
CDC servers. Existing eICU originals are retained and locally verified. No Kaggle
data was used. Original datasets and previous model outputs are preserved.

## Decisions

| Application | Original data verdict | Selected source | Labelled cohort |
|---|---|---|---:|
| Heart | Keep BRFSS 2022; correct the PDF's UCI description | [CDC BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html) | 440,111 |
| Hypertension | Replace BRFSS 2024: supplied file has no hypertension diagnosis label | [CDC NHIS 2023 Sample Adult](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) | 29,471 |
| Diabetes | Newly downloaded and prepared | [CDC NHANES August 2021–August 2023](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023) | 7,876 |
| Respiratory | Full eICU is suitable; bypass two truncated CSVs using complete compressed sources | [PhysioNet eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) | 29,126 eligible ICU stays at 24 hours |

These are four clinical applications from four source families. The three survey
tasks classify diagnosis history; they do not establish future disease onset.
The respiratory task predicts mortality within a respiratory ICU cohort.

## Files to use next

1. **Heart:** [heart_2022_baseline_raw.csv](Heart_correct/dataset_verified/heart_2022_baseline_raw.csv).
   The target is `HeartDisease`, derived from `_MICHD`. There are
   39,751 positives and 400,360 negatives;
   5,021 unknown labels are audited separately.
   [Heart notes and source links](Heart_correct/DATASET_README.md).
2. **Hypertension:** [hypertension_nhis2023_raw_cohort.csv](hypertension_correct/nhis2023/derived/hypertension_nhis2023_raw_cohort.csv).
   Target `hypertension` comes from `HYPEV_A` yes/no. There are 11,094 positives,
   18,377 negatives and 51 excluded unknown/refused labels.
   [Hypertension notes](hypertension_correct/nhis2023/DATASET_README.md).
3. **Diabetes:** [diabetes_baseline_raw.csv](diabetes_correct/diabetes_baseline_raw.csv).
   Target `diabetes_diagnosed` comes from `DIQ010` yes/no. There are 1,073 positives
   and 6,803 negatives. Borderline/unknown answers are not labelled as negative.
   Six original NHANES components and their codebooks are in the folder; HbA1c
   and glucose are available in a separate biomarker table.
   [Diabetes notes and all six download links](diabetes_correct/DATASET_README.md).
4. **Respiratory:** [24h cohort manifest](resp_physionet/resp/dataset_verified/respiratory_cohort_24h_landmark.csv)
   defines eligible stays and labels for subsequent extraction. The raw clinical
   tables are [complete compressed sources](resp_physionet/resp/official_verified).
   Use `eicu_sources.py` to load them. This manifest contains membership,
   demographics and labels; clinical features still need extraction from the
   first 24 hours before modelling. [Respiratory notes](resp_physionet/resp/DATASET_README.md).

No new imputation, balancing, feature scaling, latent learning, model training or
performance comparison has been performed. Missingness and exclusions are
documented. Read each dictionary and explicit predictor list: identifiers,
survey weights/design variables, source-label aliases and outcomes must not be
automatically included as predictors. Survey weights are specific to each
survey/component; strata and PSUs are not hospitals.

## Original download links you were missing

- **Your heart source:** [BRFSS 2022 XPT ZIP](https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip).
  Your 445,132-row XPT matches your archived ZIP member and passes local archive
  checks. Its archived 328-variable schema differs from the current page's count;
  the saved hashes identify the version actually used.
- **Replacement hypertension:** [NHIS 2023 adult CSV ZIP](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip).
- **Alternative hypertension, if BRFSS is required:** [BRFSS 2023 XPT ZIP](https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip)
  includes `BPHIGH6`. Its download returned HTTP403 here, so it is not the
  prepared replacement. The NHIS files were downloaded successfully.
- **Diabetes:** [NHANES diabetes questionnaire data](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt)
  must be joined to the other downloaded components by `SEQN`; all links are in
  the diabetes README and [source index](dataset_audit/dataset_sources.csv).
- **Respiratory:** [Full eICU](https://physionet.org/content/eicu-crd/2.0/).
  Replace the PDF's demo link with this full-resource link.

## Respiratory integrity and cohort correction

The original patient table contains 200,859 stays from 208 hospitals. Thirty
compressed tables have been copied into this project and checked against the
supplied checksum manifest; gzip decompression/CRC was also verified. Twenty-eight
raw CSVs match those originals. `vitalPeriodic.csv` and `nurseCharting.csv` stop
mid-row and must not be used as complete tables.

Drive D uses FAT32. The full CSVs would be 7.94 GB and 11.47 GB, above its 4 GB
single-file limit. Their complete compressed files are valid and readable
directly. The notebook's data paths and integrity handling now use these verified
sources, and the previous notebook is backed up. Existing outputs were not rerun.
`admissionDrug.csv` has no local compressed original and remains unverified; the
respiratory pipeline does not use it.

PhysioNet's unauthenticated live-checksum request returned HTTP403. Local
checksums demonstrate consistency with the supplied manifest, not an independent
new comparison with the authenticated publisher file.

The new cohort manifest fixes adult eligibility at 24 hours, requires the patient to
remain in ICU/hospital beyond that landmark, and restricts the primary pulmonary
diagnosis route to evidence recorded by 24 hours. It uses in-hospital mortality, not
ICU mortality. Previous processed cohorts should be treated as exploratory:
they admitted stays as short as 4 hours with 24-hour features and used diagnoses without a
time cutoff. Regenerate clinical features/results using the corrected definition.

## Your guide's MIMIC + eICU suggestion

Your guide's suggestion is appropriate as an **additional** generalization
experiment. eICU itself supports held-out-hospital evaluation. Its old cleaned
train/test files share 196 hospitals despite having no patient overlap; they do
not establish cross-hospital performance.

For an external test, use [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
and eICU with harmonized respiratory inclusion rules, feature meanings/units,
prediction landmarks, availability windows and the same mortality outcome.
MIMIC has not been downloaded. Approved access to that specific project and its
DUA are needed; a PhysioNet account alone is not equivalent to project approval.
No password or browser login was read or stored.

Evaluate the fairness framework separately on the four tasks. Combining their
different labels does not produce one valid prediction target. Cross-dataset
latent alignment needs an explicit shared feature/anchor definition; standard
CKA cannot pair arbitrary unrelated people across sources.

## Project-document correction

The PDF's motivation, gaps and objectives were reviewed (pages 6 and 13–15).
These data support a retrospective multi-stage fairness benchmark. They do not
alone prove equitable new data collection, clinical safety, deployed continuous
auditing, or formal privacy protection. Those require separate implementations
and evaluations.

[Replacement dataset-slide text](dataset_audit/PDF_DATASET_REPLACEMENT.md) records
the corrected sources, outcomes and optional MIMIC extension. The original PDF
has not been edited. Audit scripts, source manifests and counts are in
`dataset_audit/`; each disease folder has its own documentation.
