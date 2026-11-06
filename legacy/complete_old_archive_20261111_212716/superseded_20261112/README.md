# Complete project: current organized files

**[Open START_HERE.html](START_HERE.html)** or [the complete dataset catalogue](DATASETS.md).

```text
preprocessing/   code/   outputs/   plots/
latent/          code/   outputs/   plots/
inprocessing/    code/   outputs/   plots/
```

Each stage has separate code, outputs and plots. In-processing has six individual branch dashboards and separate disease pages for the MIMIC/eICU diagnosis tasks.

| Stage | Code | Outputs | Dashboards |
|---|---|---|---|
| preprocessing | [Code](preprocessing/code/README.md) | [Outputs](preprocessing/outputs/README.md) | [Dashboards](preprocessing/plots/index.html) |
| latent | [Code](latent/code/README.md) | [Outputs](latent/outputs/README.md) | [Dashboards](latent/plots/index.html) |
| inprocessing | [Code](inprocessing/code/README.md) | [Outputs](inprocessing/outputs/README.md) | [Dashboards](inprocessing/plots/index.html) |

Five source datasets produce 12 prediction tasks. The respiratory hospital experiment uses 29 test hospitals; eICU diagnosis experiments use 31 test hospitals. MIMIC/eICU external transfer is reported separately. Survey results are not hospital evidence. Methods are explicit adaptations; performance/fairness effects are mixed.

[Measured in-processing results](inprocessing/outputs/REPORT.md) | [All complete CSVs](DATASET_FILES.csv) | [Current organization and validation checks](dataset_audit/layout_validation.json)

Old experiments are outside this folder in `D:/complete_old_archive_20261111_212716`, including the five historical audit/code copies found during this cleanup. [Original archive](dataset_audit/archive_manifest.json) / [Additional archive record](dataset_audit/remaining_old_audit_archive.json). Frozen selections and model/prediction files are preserved; [layout move hashes](dataset_audit/layout_migration.json) record the relocation.

[One file per bias: open the dashboard catalogue](BIAS_DASHBOARDS.md). Each stage plot index now lists the individual bias/metric HTML files. Summary PNGs are current supporting diagnostics.

[Final verification report](dataset_audit/FINAL_REVIEW.md) | [Code and algorithm review](dataset_audit/CODE_REVIEW.md) | [Every current stage file](dataset_audit/active_recheck/ALL_STAGE_FILES.csv)
