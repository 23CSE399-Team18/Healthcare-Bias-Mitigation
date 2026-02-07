# Rules for individual bias dashboards

These are descriptive project screening rules, declared before generating the new dashboards. They are not universal clinical, legal or statistical definitions of bias. A green status means **not detected by the displayed rule**, not proven absence. An orange status means insufficient evidence. The dashboards do not retrain models, alter thresholds, or select a new model using test results.

## Preprocessing

Compare the same saved partition and row IDs before and after the fitted baseline transform. Labels and audit groups remain unchanged. Original numeric units are reconstructed after imputation, before scaling, for distribution comparisons. Scale diagnostics instead compare original and standardized units explicitly.

| Separate page | Screen |
|---|---|
| Missing data | Any missing input cell; after values are from the saved fitted imputer and finite transformed matrix. |
| Class imbalance | Any binary task with minority-class share below 20%. No resampling was performed by the fixed baseline. |
| Outliers | More than 1% of observed values beyond training-only Q1 - 1.5 IQR / Q3 + 1.5 IQR, for numeric variables with at least 20 distinct training values and positive IQR. Statistical tails are not confirmed clinical errors. |
| Feature scale | Ratio of largest to smallest positive standard deviation greater than 10, among the same continuous numeric variables. Scaling does not remove clinical differences. |
| Demographic | Outcome-prevalence range greater than 0.10 across supported known groups of any audited attribute (minimum 30 rows). This describes case mix/label differences, not predictive discrimination. |
| Duplicates | Repeated record identifiers within a partition. Exact repeated feature patterns and inconsistent labels within those patterns are shown separately; they do not prove duplicate people or erroneous labels. |
| Noise | Not assessable without clean reference measurements/adjudicated labels. Conflicting labels in repeated feature patterns are an ambiguity diagnostic only. No claimed denoising. |
| Encoding/domain | Any unseen categorical value, or absolute standardized numeric mean shift greater than 0.5 relative to source training. No hospital claim for survey states/regions. Training self-comparison is not a domain-shift evaluation. |

Raw clinical extraction already applied its documented validity filters. Here, 'before' means the model-ready rows immediately before the fixed numerical preprocessing, not unfiltered source downloads. All distribution summaries use the full partition. Histograms use shared untrimmed bins. Box and Q-Q displays use precomputed full-data quantiles and do not expose individual records.

## In-processing

Before = saved latent ERM baseline. After = the comparison model selected in the page. The default remains the original frozen overall validation-selected model; other methods can be inspected without selecting a new winner. Classification thresholds stay at each model's original validation-fitted value.

| Separate page | Screen |
|---|---|
| Demographic prediction | Largest supported equalized-odds gap across sex, age and ethnicity greater than 0.10. |
| Intersectional/subgroup | Supported intersectional equalized-odds gap greater than 0.10. |
| Predictive miscalibration | Full-partition 10-bin ECE greater than 0.05. This depends on binning and sampling; reliability-bin counts are shown. |
| Cross-hospital generalization | On hospital holdouts/external evaluations: AUROC deterioration greater than 0.05 from that model's source validation AUROC, or Brier range greater than 0.05 across at least two hospitals with 30 rows. These are risk signals, not proof of universal generalization failure. Surveys, source validation, and one-institution internal MIMIC tests are not applicable to the hospital screen. |

Use the saved support rules: minimum 30 rows and, for TPR/FPR, at least 10 positives/negatives respectively. Unknown groups remain in overall metrics and are excluded from disparity summaries. Missing comparisons remain NA. MIMIC/eICU transfer also includes documentation shift; anonymous institutional overlap across databases cannot be excluded.

## Latent pages

Latent metrics are not all bias tests. Make one separate page each for information preservation, structure, input/latent alignment, and domain association. The original validation Macro-F1 guard is a decrease no greater than 0.02 relative to the uncompressed probe. Silhouette, CKA, NMI and ARI are displayed without inventing a universal pass/fail cutoff. Aggregate cross-source demographic alignment does not prove paired-patient or clinical equivalence. Hospital branches have additional validation diagnostics using the unchanged source encoders and fixed training-only logistic probes. Silhouette/CKA use up to 2,000 validation rows; probe and clustering association metrics use the complete validation partition. MIMIC domain association is unavailable because its source has one institution. No test labels are used to add model selection.
