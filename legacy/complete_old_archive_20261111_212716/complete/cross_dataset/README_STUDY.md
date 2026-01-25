# MIMIC–eICU generalization study

**Historical mortality-only study setup.** The current four-condition study is documented in
[four_conditions/README.md](four_conditions/README.md). Its four recorded-diagnosis
classification tasks supersede this narrower plan for the requested cross-hospital work.
The mortality code below is preserved separately and has not been run as a four-condition evaluation.

This was the additional cross-dataset experiment requested by your guide. It uses
**the same adult ICU mortality task in both databases**. The four disease
applications remain separate; the respiratory application remains eICU-only.

## Current status

- The real eICU half has been extracted: **131,232 eligible ICU stays**, 103,341
  patients and 207 hospitals; 11,977 hospital deaths and 119,255 survivors.
- `prepared/eicu_24h_features.csv` contains observed common predictors and labels.
  Missing values are preserved. All 146,671,642 periodic-vital rows were streamed.
- MIMIC-IV v3.1 requires an approved PhysioNet account and authenticated download.
  Using a different approved email/account is fine. Passwords must not be sent in
  chat. See `README_DOWNLOAD.md` for the local, password-hidden download step.
- The four actual MIMIC metadata tables have now been imported from official
  browser downloads and rehashed against the publisher manifest. They contain
  364,627 patients, 546,028 hospital admissions, 94,458 ICU stays and 4,095 item
  definitions. The dataset-only 24-hour cohort has **74,009 stays**, 53,945 patients,
  8,621 hospital deaths and 65,388 survivors. Its membership and labels were
  independently checked against all joined original metadata records.
- `prepared/mimic_24h_cohort_manifest.csv` contains only cohort membership,
  demographic auditing fields and the target; it is not a clinical feature table.
  All six reviewed item labels and units match the actual item dictionary.
  See `prepared/mimic_cohort_validation.json` and
  `prepared/mimic_item_mapping_validation.csv`.
- The MIMIC feature table, transfer results and fairness results are **not yet
  produced**. `status.json` records the state; no result values are fabricated.

## Exact shared task

Include known adults who are still in ICU and hospital after 24 hours, alive at
that landmark where an explicit death timestamp is available, and have a known
hospital outcome. Predict **subsequent in-hospital death** using observations
from ICU admission through 24 hours inclusive. Eligibility uses the 24-hour
landmark; it is not a model that can predict deaths occurring before that time.

This additional cohort includes **all eligible adult ICU stays**, rather than
trying to infer a harmonized respiratory diagnosis from different coding
systems. MIMIC hospital ICD diagnoses are not timestamped as available during
the first 24 hours, so using discharge diagnoses to select a prospective
respiratory cohort would require a different, explicitly retrospective design.

Keep repeated ICU stays, but assign all stays belonging to one true patient to
the same within-source split. The eICU patient key is `uniquepid`; MIMIC uses
`subject_id`. Source-prefixed keys prevent accidental collisions. Identifiers,
hospital identity, outcome, diagnosis text and discharge/length-of-stay variables
are not model inputs.

## Harmonization

| Concept | eICU source | MIMIC-IV source | Shared representation |
|---|---|---|---|
| Heart rate | `vitalPeriodic.heartrate` | `chartevents` item 220045 | beats/min |
| Respiratory rate | `vitalPeriodic.respiration` | items 220210 and 224690 | breaths/min |
| Peripheral oxygen saturation | `vitalPeriodic.sao2` | item 220277 | percent |
| Age | `patient.age` | `anchor_age + ICU year − anchor_year` | 18–39,40–64,65–79,80+ |
| Sex/gender field | `patient.gender` | `patients.gender` | Male/Female/Unknown |
| Outcome | hospital discharge status Alive/Expired | `hospital_expire_flag` | 0/1 |

Values use identical range filters across sources: heart rate0<value<300,
respiratory rate0<value<70 and oxygen saturation0<value<=100. Observations are
aggregated within each ICU hour, then mean/min/max are computed across observed
hourly means. This gives nine numeric predictors plus age band and sex. Missing
hours are not invented and charting frequency is not used as a predictor.

eICU's monitor archive and MIMIC's charted records still have different sampling
and verification processes. Hourly aggregation reduces unequal observation
weighting; it does not prove measurement equivalence. Monitor temperature was
audited but excluded from this conservative baseline because 117,502 of 131,232
eICU stays lack it (89.5%). Routine nursing temperatures would need additional
extraction/harmonization before a temperature-based comparison.

MIMIC events use `charttime` relative to ICU `intime`; rows with a nonmissing
`storetime` after the landmark are also excluded. eICU uses
`observationoffset`, which does not expose an equivalent documentation timestamp.
This availability limitation must remain visible when interpreting transfer.

Age topcoding differs between sources; broad 80+ bands avoid pretending their
oldest patients have known exact ages. Broad race/ethnicity groups are preserved
for auditing, not used as baseline predictors. Their coding is imperfectly
equivalent, and unknown values remain unknown. All feature/metadata roles are
specified in `prepared/common_feature_schema.json`.

## Evaluation

`evaluate_generalization.py` runs two fixed directions:

1. Fit on approximately 80% of eICU patients; evaluate on the untouched 20% and on
   the entire MIMIC cohort.
2. Fit on approximately 80% of MIMIC patients; evaluate on the untouched 20% and on
   the entire eICU cohort.

The patient assignment is a reproducible hash split with seed 42. Each direction
fits median imputation, missingness indicators, numeric scaling and categorical
encoding using only its source-training rows. A fixed logistic-regression
baseline and probability threshold 0.5 are used without target-source tuning.
The two directions are separate experiments, not a source-adaptation procedure.

Reported metrics include AUROC, AUPRC, balanced accuracy, macro-F1, Brier score,
calibration error and false-negative/false-positive counts. Demographic output
includes counts and TPR/FPR for age, sex and broad race/ethnicity. Small-group
rates are suppressed below 100 records or 20 relevant outcome cases. These are
descriptive point estimates; no statistical-significance or clinical-safety
claim is made. This baseline does not establish successful bias mitigation.

MIMIC is from one health system; eICU covers many hospitals. Cross-dataset
differences also reflect calendar period, case mix, measurement and missingness.
They must not all be attributed causally to hospital identity. Existing
eICU respiratory results with shared train/test hospitals do not substitute for
this external test or a separate held-out-hospital test.

## Run after the MIMIC download finishes

From `D:\complete`:

```powershell
python cross_dataset\prepare_cross_dataset.py --source mimic
python cross_dataset\evaluate_generalization.py
```

The eICU extraction is already complete. To reproduce it from canonical sources:

```powershell
python cross_dataset\prepare_cross_dataset.py --source eicu
python cross_dataset\test_study_integrity.py
```

Preparation requires pandas and numpy. Evaluation also requires scikit-learn.
The scripts use the project's `_data_reader` and `_model_reader` environments
when present. MIMIC files larger than FAT32's 4 GB single-file limit are streamed
through the downloader's segmented gzip reader, without assembling an oversized
file. Original credentialed data remains local.

## Primary sources

- Full eICU v2.0: https://physionet.org/content/eicu-crd/2.0/
- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- Official MIT-LCP vital-sign concept and item mappings:
  https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/vitalsign.sql
- Official first-day concept (our harmonized window deliberately starts at ICU
  admission, rather than including its optional six pre-ICU hours):
  https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/firstday/first_day_vitalsign.sql
- eICU table documentation: https://eicu.mit.edu/eicutables/vitalperiodic/
- Original database publications explain source acquisition and deidentification:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6132188/ and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9810617/
