## Complete-folder recheck — 11 November 2026

The current verified preprocessing and latent artifacts passed a fresh recheck for all five datasets. The review covered all 24 latent candidates, 45 method/task metric rows and 15 selected split CSVs. Overlapping silhouette markers were separated in the plots, and historical notebook notices now appear first. [Read the recheck and remaining limitations](dataset_audit/recheck_current/REPORT.md).

## Latent-stage update — 11 November 2026

The five verified baseline datasets now have trained latent representations, saved encoders, validation comparisons and checked exports. Use [the single notebook](latent/latent_pipeline.ipynb), [the results and plots](latent/index.html), and [the guide to all 15 split CSVs](latent/README.md).

Selected widths are heart 8, respiratory 89, diabetes 8, hypertension 8 and MIMIC 16. MIMIC retains its four overlapping outcome labels. These remain separate dataset-specific representations. Validation probes were trained for latent selection; in-processing fairness methods and final test evaluation have not been run. [Read the measured results and method adaptations](latent/REPORT.md).

## Verification update — 11 November 2026

The original heart, respiratory, diabetes and hypertension preprocessing files passed checks for finite predictors, label/ID alignment and disjoint splits. Their historical notebooks used test scores during method selection, so their reported test performance is exploratory.

New fixed train-only baseline datasets and verified plots are available in [preprocessing/verified_baseline](preprocessing/verified_baseline/README.md). All source records and labels are retained. [Open the plots](preprocessing/verified_baseline/index.html) or [read the full audit](preprocessing/audit_current/REPORT.md).

MIMIC baseline preprocessing was independently verified against source-training statistics and all saved matrices, labels, IDs and hashes; no rebuild was needed. Its eight-bias mitigation experiment remains incomplete.

---

# Current project dataset status

**MIMIC source verification and four-condition preprocessing are complete.** All nine downloaded archives match the saved PhysioNet publisher checksums and pass full gzip checks.

The additional MIMIC-eICU study now contains four separate recorded-diagnosis classification tasks: ischemic heart disease, respiratory-system disease, non-gestational diabetes and systemic hypertension. These are research-derived tables with documentation-based labels, not four official prepackaged datasets.

MIMIC: **73,990 stays / 53,931 patients**. eICU comparison: **109,378 stays / 206 hospitals** after eligibility and split purging.

[Open the four-condition datasets, counts, definitions and validation](cross_dataset/four_conditions/README.md).

## Original application datasets

| Application | Trusted source | Existing task | File |
|---|---|---|---|
| Heart | CDC BRFSS 2022 | Self-reported heart attack/CHD/angina | [Dataset](model_ready/heart_model_ready.csv) |
| Hypertension | CDC NHIS 2023 | Self-reported diagnosed hypertension | [Dataset](model_ready/hypertension_model_ready.csv) |
| Diabetes | CDC NHANES 2021-2023 | Self-reported diagnosed diabetes | [Dataset](model_ready/diabetes_model_ready.csv) |
| Respiratory | Full eICU-CRD v2.0 | Subsequent in-hospital mortality, 24h respiratory cohort | [Dataset](model_ready/respiratory_model_ready.csv) |

The original respiratory dataset has **28,842 stays** after the APACHE entry-time correction. It is preserved separately from the new respiratory recorded-diagnosis classification task.

Use eICU hospital partitions for a hospital-held-out evaluation, and MIMIC as a separate external database. MIMIC has one institution. MIMIC discharge coding and eICU ICU/history documentation differ; missing codes do not establish disease absence. Surveys remain separate population benchmarks.

Preprocessing is complete, including source-train-only imputation/scaling/encoding. The latent-stage validation probes are described above. No final four-condition test results, cross-hospital results or bias-mitigation results have been produced. The older general ICU mortality runner remains stopped.

[Source review](dataset_audit/MIMIC_FOUR_PHENOTYPE_REVIEW.md) | [Download integrity](cross_dataset/mimic_download_integrity.json) | [Historical authorised cleanup](dataset_audit/cleanup_completed.json).
