# Final project review

The requested **preprocessing -> latent -> in-processing** artifacts are present in `D:/complete`. Each stage uses `code`, `outputs` and `plots` folders. There are **144 individual bias/metric HTML pages**: 64 preprocessing, 32 latent and 48 in-processing. Each file covers one bias or latent metric for a current dataset/task.

[Open the individual dashboard catalogue](../BIAS_DASHBOARDS.md). [Read the code/method review](CODE_REVIEW.md), including the difference between the fixed preprocessing baseline and the algorithms shown in the supplied image. [Every current stage file is inventoried here](active_recheck/ALL_STAGE_FILES.csv).

This verification confirms saved transformations, data alignment, reproducible model predictions, metric calculations and local file integration. It does not certify strong predictive performance, statistical significance or universal hospital generalization.

| Stage | Complete CSV exports | Contents |
|---|---:|---|
| Preprocessing | 23 | Original five baselines plus hospital/external branches |
| Latent | 26 | Selected representations for the five baselines and three hospital/transfer branches |
| In-processing | 20 | Selected predictions for six branches and 12 tasks |

The pipeline retains 191 fitted in-processing candidates/references, nine executed notebooks, 49 plot images and the original frozen selection records. Each complete CSV contains its feature/latent columns, labels and metadata; prediction exports also include task probabilities and decisions.

## Verification evidence

| Check | Result | Evidence |
|---|---|---|
| Original preprocessing | Passed | [Saved report](../preprocessing/outputs/audit_current/confirmation.json) |
| MIMIC/eICU preprocessing | Passed | [Saved report](../preprocessing/outputs/audit_current/mimic_audit.json) |
| Latent artifacts | Passed | [Saved report](../latent/outputs/validation.json) |
| Latent plot metrics | Passed | [Saved report](recheck_current/verification.json) |
| Latent loader | Passed | [Saved report](../latent/outputs/handoff_check.json) |
| Hospital preprocessing and latent | Passed | [Saved report](../latent/outputs/hospital/independent_validation.json) |
| In-processing models and metrics | Passed | [Saved report](../inprocessing/outputs/validation.json) |
| Final prediction CSVs | Passed | [Saved report](../inprocessing/outputs/export_validation.json) |
| All executed notebooks | Passed | [Saved report](notebook_execution_all.json) |
| Folder structure and links | Passed | [Saved report](layout_validation.json) |
| Fresh full pipeline recheck | Passed | [Saved report](active_recheck/checks.json) |
| Individual chart data and file inventory | Passed | [Saved report](active_recheck/dashboard_validation.json) |
| Individual dashboard control logic | Passed | [Saved report](active_recheck/dashboard_control_tests.json) |

Preprocessing statistics were fitted on training rows and checked against saved matrices/CSVs. Hospital partitions were checked for hospital/patient overlap. Encoders reproduce saved latent values, and all 45 latent method/task comparison rows were recomputed. Saved in-processing models reproduce sampled predictions; all 506 validation/test/external metric rows and the complete prediction CSVs were reconciled. Model and prediction artifacts were not retrained during this folder cleanup.

## Dataset and study scope

The five sources are CDC BRFSS 2022, eICU-CRD v2.0, NHANES August 2021-August 2023, NHIS 2023 and MIMIC-IV v3.1. [The full catalogue](../DATASETS.md) records targets, cohort counts, hospital counts, feature widths, split sizes and local source paths. [The complete CSV inventory](../DATASET_FILES.csv) lists all 69 files.

- Respiratory mortality uses 29 held-out eICU test hospitals; the four recorded-diagnosis tasks use 31 held-out eICU test hospitals.
- Source-fitted MIMIC/eICU external transfer is also evaluated. Anonymous hospital IDs cannot prove institutional non-overlap across the two databases.
- Survey tasks have no hospital IDs. They remain separate population benchmarks.
- Respiratory mortality and recorded respiratory diagnosis are different outcomes. ICU diagnosis labels overlap and reflect documentation.
- Methods requiring adaptation are named in [the protocol](../inprocessing/code/PLAN.md). Performance and fairness effects are mixed. These are bounded-budget single-seed experiments.

## Remaining limitations

A live browser visual review could not be completed: browser access is unavailable, and the prior native browser attempt was stopped because its URL could not be verified. Local HTML structure, tab-to-panel wiring, figures and links passed checks. This report does not claim browser screenshots or interaction testing.

The original MIMIC/eICU preprocessing pickle reports scikit-learn 1.9.1 while the Python 3.12 validation environment uses 1.8.0. Its training statistics and transformation outputs matched independent reconstruction. The numeric matrices used by latent/in-processing do not depend on loading that pickle. Preserve the recorded dependencies when reusing it.

## Open the results

- [Start page](../START_HERE.html)
- [Individual in-processing dashboards](../inprocessing/index.html)
- [Preprocessing dashboards](../preprocessing/plots/index.html)
- [Latent dashboards](../latent/plots/index.html)
- [Measured in-processing results](../inprocessing/outputs/REPORT.md)

Old experiments remain outside this project in `D:/complete_old_archive_20261111_212716`. The original archive manifest, additional historical-audit archive record and layout migration hashes are retained in this audit folder.
