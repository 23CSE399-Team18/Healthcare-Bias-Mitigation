# Complete-folder recheck — 11 November 2026

**The current verified preprocessing and latent artifacts pass the recheck for all five datasets.** This conclusion applies to the fixed baselines and current latent outputs. The older bias-mitigation notebooks and dashboards remain exploratory research artifacts with known evaluation and plotting problems.

| Dataset | Preprocessed features | Selected latent method | Latent features | Current checks |
|---|---:|---|---:|---|
| Heart | 36 | DGCCA adaptation | 8 | Pass |
| Respiratory | 274 | PCA, 95% training variance | 89 | Pass |
| Diabetes | 14 | PCA | 8 | Pass |
| Hypertension | 69 | PCA | 8 | Pass |
| MIMIC | 24 | PCA, 99.9% training variance | 16 | Pass |

## What was verified

- Recomputed training-only preprocessing statistics and compared the saved transforms, CSVs and arrays. Source IDs, labels, finite features and partition separation agree. Respiratory and MIMIC patients do not overlap between partitions. Both saved MIMIC/eICU preprocessing directions passed.
- Recomputed the preprocessing plot counts, missingness, target rates, scaling statistics and histogram bins/denominators, including MIMIC's four labels. Visually reviewed all five preprocessing figures.
- Rechecked all **24 latent candidates**, the **15 selected split CSVs**, source hashes, metadata and label alignment. Saved encoders reproduce sampled exported rows; complete arrays and CSVs pass integrity checks.
- Recomputed Macro-F1, silhouette and same-row CKA with the latent validator. The supplemental review checked AUROC, average precision, domain NMI/ARI, all **45 method/task rows** in the comparison CSV, and the per-target utility guard. The fixed uncompressed validation probes were refitted and their saved reference scores reproduced.
- Independently recomputed the cross-dataset group-centroid CKA using sample Gram matrices, and reproduced the pooled diagnostic NMI/ARI and permutation reference. These remain demographic diagnostics, not proof of a common clinical representation.
- Checked notebook structure, Python syntax and saved error outputs. The latent notebook's result-viewing cells and loader were executed successfully. All four model contract tests passed, including inference independent of query batch and order. Historical training notebooks were not reexecuted.

## Corrections made during this recheck

The latent silhouette figures placed methods at the same x coordinate, hiding methods with similar scores. All five figures were regenerated with separated markers. Single-target plots now show each method and its value; MIMIC uses offset markers for its four categorical tasks. All five updated latent figures and the aggregate CKA figure were visually reviewed. The numerical results and selected methods are unchanged.

The four historical preprocessing notebooks now begin with a clear notice and links to the verified baseline and current latent pipeline. Their introductions no longer describe the historical experiment as the current complete preprocessing stage. Every code cell and saved output was preserved; the preservation hashes are in [notebook_notice_changes.json](notebook_notice_changes.json).

## Remaining limitations

1. **The historical experiments are not a clean test evaluation.** Their test scores influenced method selection. Their old dashboards also use clipped tails, frequency polygons labeled as density, and zero values for undefined correlations. Their saved numbers were checked against their source tables where comparisons were available, but the historical charts are not certified as statistically correct. Use the verified baseline plots below.
2. **MIMIC model-persistence compatibility:** loading its original preprocessing pickle reports scikit-learn 1.9.1 versus the current Python 3.12 environment's scikit-learn 1.8.0. Its transform outputs and training statistics matched in this audit. Reusing that pickle in a different environment still needs compatible dependencies; the current latent loader uses the checked numeric arrays and does not load it. Dependencies and original artifacts were preserved.
3. The eight-bias mitigation experiment, in-processing fairness methods and final test-performance study are not completed by these checks. No test predictions were generated during the recheck. Valid file structure and reproducible metrics do not establish clinical safety or better fairness.
4. The selected latent widths intentionally differ. Respiratory and MIMIC needed wider PCA representations to satisfy the validation utility guard; this is not an equal-capacity win over the eight-dimensional neural candidates. The guard was an exploratory, validation-driven refinement.
5. Original respiratory predicts mortality in a respiratory ICU cohort; MIMIC respiratory predicts a recorded diagnosis. Keep the five datasets separate. The current latent run did not score test outcomes, but these source datasets have a history of exploratory analysis and should not be described as a newly collected external validation cohort.

## Files to use

- [Verified preprocessing guide](../../preprocessing/verified_baseline/README.md) and [preprocessing plots](../../preprocessing/verified_baseline/index.html).
- [Current latent notebook](../../latent/latent_pipeline.ipynb), [latent plots](../../latent/index.html), and [all 15 selected CSVs with loader instructions](../../latent/README.md).
- [Fresh preprocessing confirmation](../../preprocessing/audit_current/confirmation.json), [MIMIC matrix audit](../../preprocessing/audit_20261109/mimic_audit.json), [latent artifact validation](../../latent/outputs/validation.json), [loader/notebook checks](../../latent/outputs/handoff_check.json), and [supplemental verification](verification.json).

## Repeat the checks

From `D:/complete`, using the environment recorded in `latent/environment.json`:

```powershell
python preprocessing/confirm_saved_outputs.py
python cross_dataset/audit_saved_preprocessing.py
python latent/validate_latent.py
python dataset_audit/recheck_complete.py
python latent/test_latent.py
python latent/check_handoff.py
```

These commands preserve the dataset files and latent encoders. The supplemental review refits only the fixed uncompressed diagnostic probes to reproduce their validation reference scores. It does not save new prediction models or change method selection. Raw MIMIC archives were not restreamed; their derived-file hashes were checked against the previous full-stream audit.
