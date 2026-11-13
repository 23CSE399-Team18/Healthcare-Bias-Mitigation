# Cross-hospital ICU cohort (eICU-CRD v2.0 + MIMIC-IV v3.1)

Used by the **latent** stage (fifth view of the common latent space) and by the **in-processing cross-hospital
generalization bias** of every disease. Built earlier by the team's Codex session for the guide's cross-hospital request;
restored here on 12 Nov 2026 from `D:/complete_old_archive_20261111_212716/superseded_20261112/` (the archive copy is kept).

| Part | Rows | Hospitals | Role |
|---|---:|---:|---|
| eICU training hospitals | 80,606 stays | 141 | training (environments = hospitals) |
| eICU validation hospitals | 20,318 | 34 | checkpoint / threshold / method selection |
| eICU test hospitals | 8,454 | 31 | final check |
| MIMIC-IV (Beth Israel Deaconess) | 73,990 | 1 (other hospital system) | external final check, never trained or tuned on |

Features (25): heart-rate, respiratory-rate and SpO2 mean/min/max over the first 24 ICU hours, their missing indicators, age
group and sex. Labels: recorded heart (ischaemic heart disease), diabetes, hypertension and respiratory-system disease
(`target_heart`, `target_diabetes`, `target_hypertension`, `target_respiratory`). Race/ethnicity is an audit attribute only.
Definitions, cohort rules and validation: `raw/STUDY_README_codex.md`, `raw/phenotype_definitions.json`,
`raw/preprocessing_status.json`, `raw/independent_validation.json`.

```text
raw/eicu/, raw/mimic/      <condition>_model_ready.csv - cohort tables before preprocessing (missing values kept)
code/                      Codex's extraction and preprocessing scripts (as used; paths refer to the old cross_dataset/ layout)
                           + fix_eicu_race_labels.py
preprocessed/              hospital_{train,validation,test,external}_preprocessed.csv (eICU-training-fitted medians,
                           missing indicators, scaling, one-hot), preprocessor.joblib, feature_names.json, race_fix_audit.json
```

**Correction made on restore (12 Nov 2026).** Codex's `race_group()` tested the substring "ASIAN" after the White rule, and
"CAUCASIAN" contains "ASIAN", so every eICU Caucasian stay had been labelled Asian (79% "Asian", no "White"). The function now
uses a whole-word match and `code/fix_eicu_race_labels.py` recomputed the eICU labels from the official eICU patient table
(83,473 stays Asian -> White; eICU is now 76% White, 1.7% Asian). It asserts that only Caucasian stays changed. Race is never
a model input, so features, labels, splits and the fitted preprocessing are unchanged. MIMIC was not affected.
