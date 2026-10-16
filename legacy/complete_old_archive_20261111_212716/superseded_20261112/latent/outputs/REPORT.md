Stage-specific latent validation report. For completed prediction evaluations, use [in-processing results](../../inprocessing/outputs/REPORT.md) and the [hospital branches](hospital/README.md).

# Latent stage: results and in-processing handoff

The latent stage ran on the verified baseline preprocessing. It did not use the historical eight-bias outputs. All method selection below uses validation data; no test performance was calculated.

| Dataset | Training / validation / test rows | Selected latent method | Latent width | Validation Macro-F1 | Uncompressed reference | Utility guard |
|---|---|---|---:|---:|---:|---|
| heart | 308,077 / 66,017 / 66,017 | DGCCA | 8 | 0.5828 | 0.5854 | pass |
| respiratory | 20,150 / 4,360 / 4,332 | PCA_95pct_variance | 89 | 0.6232 | 0.6275 | pass |
| diabetes | 5,513 / 1,181 / 1,182 | PCA | 8 | 0.5923 | 0.5952 | pass |
| hypertension | 20,629 / 4,421 / 4,421 | PCA | 8 | 0.7473 | 0.7494 | pass |
| mimic | 52,229 / 10,829 / 10,932 | PCA_99.9pct_variance | 16 | 0.6104 | 0.6104 | pass |

## Which files to use

Use the complete split CSVs in `outputs/<dataset>/selected/` for in-processing. Only columns matching `^z[0-9]+$` are model inputs; the table above gives the actual width. `target` is the outcome for each original dataset; MIMIC instead has `target_heart`, `target_respiratory`, `target_diabetes`, and `target_hypertension`. Choose a separate prediction head/task for each MIMIC label. All `meta_*` and `sens_*` columns are metadata, not inputs.

The authoritative compact arrays are `outputs/<dataset>/selected/Z_<split>.npz` (key `Z`). They align row-for-row with `outputs/<dataset>/rows_<split>.csv` and `y_<split>.npz`. Every output manifest records source hashes and row counts. All four candidate methods remain available in their own folders.

**One pipeline does not mean one combined patient table.** Different datasets have different labels, populations and feature meanings. Keep their splits and tasks separate. MIMIC uses one feature population with four overlapping labels, not four independent copies of patients. The latent spaces are dataset-specific; a matching number of columns does not prove semantic alignment.

## What was implemented

- DGCCA: within-patient paired feature views, MAXVAR correlation objective with a declared reconstruction auxiliary, followed by a global training-only GCCA projection. No arbitrary cross-database patient pairing.
- SAINT-style: feature-token attention and intersample attention with a fixed bank of training rows, plus denoising and contrastive training. The fixed-bank induction and feature grouping are explicit adaptations, not an exact published benchmark replication.
- MV-DAE: multiple feature-view encoders, masking and view dropout, a fused bottleneck, and reconstruction of all input views.
- PCA: a linear baseline eligible for selection; a modern method is not forced to win.

## Metrics and limits

- Macro-F1 is from fixed class-balanced logistic probes on validation labels. It measures predictive information for these recorded targets, not clinical truth. MIMIC selection averages its four task scores. Survey estimates here are unweighted; metadata preserves the original weights.
- Silhouette uses clinical validation labels on a fixed stratified sample of up to 2,000 rows per task. CKA compares input and latent coordinates of the same validation people on up to 2,000 rows.
- Cross-source CKA compares up to eight supported age/sex group means. It is a weak aggregate demographic diagnostic, not evidence of individual-patient clinical alignment. Do not use disease labels to match different tasks.
- Domain NMI/ARI compare training-fitted clusters against validation domain IDs. Diabetes has no suitable domain ID; MIMIC has one institution, so these within-dataset metrics are NA. The separate pooled diagnostic uses a four-dimensional training-fitted demographic-anchor map, equal source sample sizes, and a permutation reference. Anchor coordinates are not the default next-stage features.
- A single fixed seed and bounded training budget were used. This is a reproducible experiment, not proof of best possible model performance. All training rows were visited; validation selected checkpoints. Hyperparameter optimization, multi-seed confidence intervals and final in-processing performance remain separate work.
- The final utility guard checks every target separately: no validation Macro-F1 decrease exceeding 0.02 versus its uncompressed reference. The initial eight-dimensional comparison was followed by a declared validation-only rescue using PCA retaining 95%, 99%, then 99.9% of training variance when necessary. This preserves useful information at a larger width. It is not a capacity-matched victory over the eight-dimensional neural candidates, nor an independent confirmatory study. Retain a no-latent comparator in the in-processing study. No fairness or clinical-safety improvement is claimed.
- Original respiratory predicts ICU-cohort mortality; MIMIC respiratory predicts recorded respiratory diagnoses. Do not interchange these labels.

## Reproduce and validate

`python latent/code/run_latent.py` trains, applies the utility guard and exports all five packages. `python latent/code/validate_latent.py` checks the handoff without fitting new models. The notebook `latent_pipeline.ipynb` is a single entry point.

Method definitions and primary references are in [PLAN.md](../code/PLAN.md). Exact metrics, supports, checkpoints and hashes are in the dataset manifests and method folders.
