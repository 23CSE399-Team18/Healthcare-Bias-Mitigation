# In-processing results

Training, validation selection and locked test/external evaluation are complete. Use the [plots](../index.html) and [all method metrics](method_comparison.csv).

| Dataset branch | Task | Selected method | Test Macro-F1 | Test AUROC | Test max EO gap | ERM max EO gap |
|---|---|---|---:|---:|---:|---:|
| heart | heart | FairERM_softEO | 0.6301 | 0.8042 | 0.7391 | 0.7826 |
| diabetes | diabetes | FairERM_softEO | 0.5981 | 0.6869 | 0.3529 | 0.3529 |
| hypertension | hypertension | FairERM_softEO | 0.7183 | 0.7995 | 0.6979 | 0.8875 |
| respiratory_hospital | respiratory | FairERM_softEO | 0.6555 | 0.8017 | 0.4762 | 0.4790 |
| eicu_hospital | heart | DANN_source_domains | 0.5499 | 0.6358 | 0.7705 | 0.8033 |
| eicu_hospital | respiratory | GroupDRO_intersection_label | 0.6070 | 0.6486 | 0.4102 | 0.7624 |
| eicu_hospital | diabetes | CRL_history | 0.5576 | 0.5952 | 0.9167 | 0.9167 |
| eicu_hospital | hypertension | FairERM_softEO | 0.6106 | 0.6511 | 0.9612 | 1.0000 |
| mimic_transfer | heart | FairERM_softEO | 0.6396 | 0.7070 | 0.9048 | 1.0000 |
| mimic_transfer | respiratory | FairERM_softEO | 0.6218 | 0.6776 | 0.4399 | 0.6446 |
| mimic_transfer | diabetes | GroupDRO_intersection_label | 0.5479 | 0.5849 | 1.0000 | 0.9167 |
| mimic_transfer | hypertension | CRL_history | 0.6540 | 0.7108 | 1.0000 | 1.0000 |

## Frozen external-database evaluation

| Source branch | Task | Selected method | External Macro-F1 | External AUROC | External max EO gap |
|---|---|---|---:|---:|---:|
| eicu_hospital | heart | DANN_source_domains | 0.6178 | 0.6819 | 0.8308 |
| eicu_hospital | respiratory | GroupDRO_intersection_label | 0.6019 | 0.6486 | 0.3752 |
| eicu_hospital | diabetes | CRL_history | 0.5493 | 0.5843 | 0.8320 |
| eicu_hospital | hypertension | FairERM_softEO | 0.6241 | 0.6813 | 0.9737 |
| mimic_transfer | heart | FairERM_softEO | 0.5461 | 0.5863 | 0.8333 |
| mimic_transfer | respiratory | FairERM_softEO | 0.5691 | 0.6037 | 0.2252 |
| mimic_transfer | diabetes | GroupDRO_intersection_label | 0.5250 | 0.5483 | 0.7547 |
| mimic_transfer | hypertension | CRL_history | 0.5986 | 0.6371 | 1.0000 |

Selection uses validation only, not the test values in this table. A selected method can perform worse than ERM on the test set. The full method tables and support counts are retained.

## Cross-hospital scope

- Respiratory mortality uses hospital-disjoint preprocessing and a newly fitted latent encoder: 137 training, 34 validation and 29 test hospitals. Cross-partition patients are excluded and recorded.
- eICU recorded-diagnosis tasks use 141 training, 34 validation and 31 test hospitals, with patient purging across partitions. The frozen eICU models are also evaluated on MIMIC.
- MIMIC models use a single source institution and are evaluated externally on eICU. Its internal test is patient-held-out. Anonymous eICU hospital IDs do not establish institutional non-overlap with BIDMC across databases.
- Survey heart, diabetes and hypertension benchmarks have no hospital IDs. Their results are not cross-hospital evidence.
- Mortality and recorded respiratory diagnosis are different targets. MIMIC discharge coding and eICU ICU/history labels have ascertainment differences. These are retrospective recorded-label evaluations, not future disease prediction.

## Files and interpretation

Selected complete CSVs are in `outputs/<branch>/selected/`. They contain the original latent columns, labels, metadata and task-specific probabilities/predictions. In-processing trains classifiers; these files are prediction results, not newly cleaned patient data. Each task directory retains every fitted model, scaler, training history, frozen selection record, probabilities, group counts and reliability bins.
The methods are documented adaptations from the supplied image. FairCare graph learning is unavailable in flat latent inputs; the tabular adversarial model must not be called an exact FairCare replication. FOMO is bounded weight search, Fish is an ERM-plus-first-order variant, and EQRM uses empirical quantiles. Methods needing multiple source domains are NA for single-domain MIMIC and the no-domain diabetes survey.
Group rates are unweighted. Raw counts/rates and support flags are retained in group tables; unsupported groups are excluded from disparity calculations and comparison plots. Undefined rates or insufficient comparisons remain NA. Age and sex are present in source features as documented upstream. The group gap metric is descriptive, not a causal discrimination estimate. Single-seed comparisons and fixed budgets do not establish clinical safety, statistical significance or universal generalization. Independent replication and uncertainty analysis remain necessary before publication or deployment.

## Reproduce

`python inprocessing/code/run_inprocessing.py` trains unfinished tasks. Completed tasks are preserved. `python inprocessing/code/make_report.py` refreshes plots and prediction CSVs. `python inprocessing/code/validate.py` checks saved results.
