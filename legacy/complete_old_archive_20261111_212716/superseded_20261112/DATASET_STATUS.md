# Current dataset status - 11 November 2026

Preprocessing, latent representations, and in-processing evaluation are complete for the defined tasks. [Open the project file guide](README.md) or [all results](inprocessing/outputs/REPORT.md).

All 12 in-processing tasks are complete. 191 fitted candidates/references, 20 final prediction CSVs and 32 comparison figures are saved. Models reproduce their saved predictions; reported metrics and CSV values passed read-back checks.

| Branch | Target | Internal evaluation | Source training / validation / test rows |
|---|---|---|---|
| Heart survey | Self-reported heart attack/CHD/angina | Record holdout | 308077 / 66017 / 66017 |
| Diabetes survey | Self-reported diagnosed diabetes | Record holdout | 5513 / 1181 / 1182 |
| Hypertension survey | Self-reported diagnosed hypertension | Record holdout | 20629 / 4421 / 4421 |
| Respiratory eICU | Subsequent hospital mortality after 24h landmark | 137 / 34 / 29 disjoint hospitals | 21082 / 4909 / 2301 |
| eICU four diagnoses | Recorded IHD / respiratory disease / diabetes / hypertension | 141 / 34 / 31 disjoint hospitals | 80606 / 20318 / 8454 |
| MIMIC four diagnoses | Same documented-diagnosis definitions, source differences retained | Patient holdout within one institution | 52229 / 10829 / 10932 |

Frozen external evaluation uses 73,990 MIMIC stays for eICU-trained models and 109,378 eICU stays from 206 hospitals for MIMIC-trained models. The diagnosis targets overlap. Missing matching diagnosis documentation is not confirmed disease absence. Respiratory mortality and recorded respiratory diagnosis are separate targets.

All preprocessing and latent transforms fit source-training rows only. Neural checkpoints, decision thresholds and method selection use validation only. Test and external results do not change selections. Each task retains an uncompressed logistic reference and a latent ERM baseline.

The original five latent widths are 8 / 89 / 8 / 8 / 16 for heart / respiratory reference / diabetes / hypertension / MIMIC. The hospital branches use 178 respiratory components and 16 components for each diagnosis direction. The original respiratory patient-split reference remains available, but is not the hospital experiment.

Survey performance is not cross-hospital evidence. Cross-database institutional non-overlap cannot be verified from anonymous eICU IDs. Performance and fairness effects are mixed; single-seed evaluations do not prove universal generalization or statistical significance. [Method adaptations and support rules](inprocessing/code/PLAN.md).

Old work is archived outside this folder: `D:/complete_old_archive_20261111_212716`. [Archive manifest](dataset_audit/archive_manifest.json).
