# Model-ready datasets

These are the original application benchmarks. The additional MIMIC-eICU hospital
study has separate [four-condition datasets and preprocessing](../cross_dataset/four_conditions/README.md).
Its respiratory recorded-diagnosis target differs from the respiratory mortality
target in this folder; do not substitute one for the other without changing the study question.

Four files, one per application, built from the verified official sources. Use these
for every pre-processing, in-processing, post-processing and latent experiment.

| File | Source | Rows | Predictors | Target | Positive |
|---|---|---:|---:|---|---:|
| `heart_model_ready.csv` | CDC BRFSS 2022 | 440,111 | 16 | heart attack or coronary heart disease (self-report) | 9.0% |
| `hypertension_model_ready.csv` | CDC NHIS 2023 | 29,471 | 32 | diagnosed hypertension (self-report) | 37.6% |
| `diabetes_model_ready.csv` | CDC NHANES 2021–2023 | 7,876 | 7 | diagnosed diabetes (self-report) | 13.6% |
| `respiratory_model_ready.csv` | eICU-CRD v2.0 | 28,842 | 112 | in-hospital death after 24 h in ICU | 13.6% |

## Column rule (same in all four files)

```python
X = df[[c for c in df.columns if not c.startswith(("meta_", "sens_")) and c != "target"]]
y = df["target"]
```

- `meta_*`: IDs, survey weights/strata/PSU, domain columns (`meta_domain_state`,
  `meta_domain_region`, `meta_domain_hospitalid`) and the eICU patient group
  (`meta_split_group_uniquepid`). **Never inputs.** Use them for splitting, domains
  and survey-weighted evaluation.
- `sens_*`: harmonised fairness groups for evaluation: `sens_sex`, `sens_age_group`
  (18-39/40-64/65-79/80+), `sens_race_ethnicity` (White/Black/Hispanic/Other) and the
  dataset's own finer `sens_race_detail`. Age and sex are also predictors; race is
  not, unless an experiment adds it on purpose.
- Text columns are categorical predictors: encode them inside your training pipeline.

## What was and was not done

Done: label-source and leakage columns removed; survey "don't know/refused" codes set
to missing; structural skips recoded; impossible values (unit or entry errors) removed
in eICU; every eICU predictor restricted to information available at 24 h.

Not done, on purpose (these are your pre-processing experiments): imputation,
scaling, encoding, oversampling, outlier removal, de-duplication, denoising.

## Splitting rules

- Respiratory: split by `meta_split_group_uniquepid` (repeat ICU stays of one patient
  must stay together). For cross-hospital methods, hold out whole
  `meta_domain_hospitalid` groups.
- Surveys: stratified random split; `meta_domain_state` (BRFSS) or
  `meta_domain_region` (NHIS) for cross-region domain experiments.
- Fit every pre-processing step on the training split only.

## Rebuild

```powershell
python resp_physionet\resp\prepare_respiratory_cohort.py     # eICU cohort (24 h APACHE timing check)
python resp_physionet\resp\extract_respiratory_features.py   # eICU first-24 h features
python model_ready\build_survey_model_ready.py               # BRFSS, NHIS, NHANES
python model_ready\validate_model_ready.py                   # checks + leakage smoke test
```

`schema_survey.json` and `respiratory_feature_audit.json` list every feature, its
missing rate and the excluded source columns. `validation_report.json` holds the
check results.
