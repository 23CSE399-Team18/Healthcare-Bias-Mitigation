# Latent-stage protocol

This run follows `preprocessing -> latent -> in-processing`. It uses the verified
fixed preprocessing baseline, not the historical eight-bias outputs whose test
scores influenced selection. It cannot be described as an eight-bias-mitigated
experiment.

## Deliverables and task boundaries

- One executable pipeline and one notebook entry point.
- Five dataset packages, each with unchanged train/validation/test membership,
  aligned IDs, sensitive-group metadata, labels, fitted encoders and latent arrays.
- A complete latent CSV for each selected dataset/split for the in-processing
  stage. There is no pooled binary target across the five datasets.
- MIMIC is one feature population with four separate, overlapping diagnosis
  targets; all four labels remain attached to the same latent rows.
- This stage trains unsupervised representation learners and diagnostic linear
  probes. It does not run the fairness methods in `in.jpeg`.

## Methods and safeguards

The project paper names DGCCA, SAINT and MV-DAE. Compare explicit research
implementations with a PCA baseline and an uncompressed linear-probe reference.
Use eight latent dimensions, a fixed random seed and declared training budgets.
Fit neural weights, normalization, GCCA projections and clustering on training
rows only. Validation reconstruction selects checkpoints; validation Macro-F1
selects the next-stage method separately for each task population. MIMIC selection
averages its four target-specific Macro-F1 values. Do not score test targets here.

DGCCA and MV-DAE use feature views from the **same patient's row within each
dataset**, not arbitrary patients paired across databases. DGCCA adds an explicit
reconstruction auxiliary loss. SAINT uses feature attention and intersample
attention to a fixed training reference bank, giving deterministic inductive
embeddings that do not change with the test batch. These adaptations are stated
in the report; they are not claimed as exact reproductions of published benchmarks.

The primary eight-dimensional representations remain dataset-specific. Same
width does not imply common clinical semantics. An optional four-dimensional
demographic-anchor projection is fitted on training age/sex group means solely
for pooled dataset-identity diagnostics. It is not the default in-processing
input or proof of clinical alignment. Outcomes and race are not used to construct
the anchor map. The current datasets do not support patient-paired cross-source
DGCCA or a single common disease label.

## Metrics from latent.png

- Macro-F1: fixed, class-balanced logistic probes trained on training embeddings
  and measured on validation labels, with the uncompressed baseline beside them.
- Silhouette: validation clinical labels, separately for each task; fixed random
  stratified sample and full sample sizes disclosed.
- CKA: input and latent representation on the **same validation rows**. Additional
  cross-source CKA uses matching age/sex group means with adequate support, and
  is labelled an aggregate demographic comparison with only eight possible rows.
- NMI/ARI: training-fitted latent clusters versus observed validation domain IDs;
  not applicable where a dataset has no domain ID or only one domain. A separate
  pooled source-identity check uses the optional demographic-anchor space with
  equal dataset samples and a label-permutation reference.

These metrics measure different properties. Low NMI/ARI alone does not establish
fairness; high CKA does not establish clinical equivalence. Do not average unrelated
metrics into an unexplained score, force a modern method to win, or claim accuracy
improvements that are not measured. Retain all candidate arrays for inspection.

## Utility-triggered dimension refinement

The initial four-way comparison fixes eight dimensions. If the initial best
candidate loses more than 0.02 validation Macro-F1 on any target relative to the uncompressed probe, run a
separate preservation check with PCA retaining 95%, then 99%, then 99.9% of
training variance, stopping when the utility guard passes. Record every attempt
and its actual dimension. This rule was added after the initial respiratory
compression showed loss, before fitting the variance-retention candidates.
It is a validation-driven refinement, not an independent confirmatory result.
The higher-dimensional fallback is a next-stage preservation option; it must not
be presented as a capacity-matched victory over the eight-dimensional neural
methods. The final files may therefore have different latent widths by dataset.
Test outcomes remain locked throughout.

## Primary method references

- DGCCA: https://aclanthology.org/W19-4301/
- SAINT and its official implementation: https://github.com/somepago/saint
- CKA: https://proceedings.mlr.press/v97/kornblith19a.html
- Denoising autoencoders: https://www.jmlr.org/papers/v11/vincent10a.html
- Metric definitions: https://scikit-learn.org/stable/modules/clustering.html
