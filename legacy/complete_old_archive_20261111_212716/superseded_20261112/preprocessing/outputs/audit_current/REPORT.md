# Verified preprocessing baseline

Current baseline preprocessing is complete for heart, respiratory, diabetes, hypertension and MIMIC.

The four application baselines use training-only imputation, scaling and encoding. MIMIC uses its independently checked source-training preprocessor.

| Dataset | Train / validation / test | Features |
|---|---|---:|
| heart | 308,077 / 66,017 / 66,017 | 36 |
| respiratory | 20,150 / 4,360 / 4,332 | 274 |
| diabetes | 5,513 / 1,181 / 1,182 | 14 |
| hypertension | 20,629 / 4,421 / 4,421 | 69 |
| mimic | 52,229 / 10,829 / 10,932 | 24 |

[Open current plots](../../plots/index.html) and [the file guide](../verified_baseline/README.md).

Saved outputs are checked for source identity, label alignment, finite values and disjoint records/patients. Original survey weights remain metadata. These checks do not establish better fairness or clinical performance.

The respiratory mortality target differs from MIMIC's recorded respiratory diagnosis. Keep the tasks separate.

Historical experiments have been moved outside D:/complete. See [the archive manifest](../../../dataset_audit/archive_manifest.json). Current in-processing methods, completed evaluations and limitations are documented in `inprocessing/outputs/REPORT.md` at the project root.

Run `python preprocessing/code/confirm_saved_outputs.py` to verify saved files without rebuilding. Run `python preprocessing/code/audit_and_prepare.py` only to rebuild the fixed baselines; refresh dependent latent artifacts after any data change.
