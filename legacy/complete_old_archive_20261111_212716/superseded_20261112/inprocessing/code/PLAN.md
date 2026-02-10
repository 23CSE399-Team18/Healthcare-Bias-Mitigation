# In-processing protocol

Inputs are the frozen selected latent matrices in `latent/outputs` for the three survey benchmarks, and `latent/outputs/hospital` for the ICU branches. Following the user's cross-hospital requirement, twelve tasks are fitted: heart/diabetes/hypertension survey benchmarks, respiratory mortality on a new hospital-disjoint partition, four eICU recorded-diagnosis tasks on hospital-disjoint partitions, and four MIMIC tasks with external eICU evaluation. No datasets or labels are pooled. The verified uncompressed inputs provide a separate logistic reference. No archived experiment is an input.

All models fit training rows only. Six full-data epochs, batch size 1024, Adam learning rate 0.001, hidden widths 32/16, seed 20261111, and weight decay 0.0001 are fixed before training. Every training row is visited in each epoch. Validation log loss chooses neural checkpoints. Each candidate's decision threshold is chosen on validation only from 0.05 through 0.95 in steps of 0.025 to maximize Macro-F1 (ties prefer 0.5). Probabilities are never modified by threshold selection. Class weights and survey weights are not applied; survey estimates are explicitly unweighted.

The baseline is latent ERM (binary cross-entropy). Overall selection minimizes the largest equalized-odds gap across sex, age, ethnicity and their intersection, subject to validation Macro-F1 and AUROC being no more than 0.02 below ERM and positive-class recall no more than 0.05 below ERM. ERM stays eligible. Selection is frozen and hashed before test scoring. All frozen candidates are evaluated once for transparent comparisons; test results cannot change models, thresholds or selection. No benefit is guaranteed. No clinical deployment claim is made.

Groups come from `sens_sex`, `sens_age_group`, `sens_race_ethnicity` and their intersection. Unknown values remain in overall evaluation and receive their own reported group, but are excluded from protected-group disparity summaries. Group tables retain raw counts/rates and support flags. Inclusion in disparities or comparison plots needs 30 rows; TPR/FPR comparisons additionally require 10 positive/negative labels, respectively. Training group definitions need at least 50 known training rows. Group-mean penalties need minibatch support at least 4; adversarial heads use all rows assigned a known training group. VREx/EQRM optimize supported domain risks when two or more exist and fall back to ordinary BCE otherwise. Undefined rates or insufficient comparisons are NA, never zero. Small subgroup estimates remain noisy; these are exploratory single-seed experiments.

## Image-to-implementation mapping

The image supplies method families, not complete reproducible specifications. Adapters are named explicitly, and none is described as an exact published benchmark replication.

| Image method | Implementation |
|---|---|
| FairCare | `AdversarialFair_tabular`: three gradient-reversal demographic heads on a tabular MLP. The original FairCare heterogeneous EHR graph is unavailable in latent tables; this is an adversarial-fairness adaptation, not graph FairCare. |
| DANN | `DANN_source_domains`: gradient reversal on actual training state/region/hospital IDs. No test-domain examples are used. NA when fewer than two training domains exist. |
| GroupDRO | `GroupDRO_sex_label`: exponentiated adversarial weights over sex-by-label training groups, mixed with 0.1 ordinary BCE to keep unsupported rows in the objective. |
| Fair-ERM | `FairERM_softEO`: BCE plus mean squared differences in group soft positive rates conditional on each true label, across the three attributes. This image label is treated as a generic regularized ERM family. |
| FOMO | `FOMO_weight_search`: bounded evolutionary, Pareto-based search over intersectional sample weights for logistic regression (population 8, three generations); training search uses a fixed maximum of 20,000 rows, and the validation-chosen weighting is refitted on all training rows. This is a small FOMO-inspired implementation, not the original package's default experiment. |
| Intersectional debiasing | `Intersectional_adversary`: gradient reversal on known sex/age/ethnicity intersections. |
| Fairness-constrained optimization | `Constrained_softEO`: projected dual ascent on differentiable conditional group-mean probability variance, tolerance 0.0025. A surrogate constraint is not a guarantee of hard-decision equalized odds. |
| Intersectional GroupDRO | `GroupDRO_intersection_label`: robust loss over intersection-by-label groups. |
| AvUC | `AvUC_deterministic`: BCE plus differentiable accurate/certain and inaccurate/uncertain loss; training-epoch-one mean predictive entropy freezes the certainty threshold. This is a deterministic binary adaptation, not Bayesian inference. |
| CRL | `CRL_history`: confidence ranking against running per-training-row correctness history; BCE remains the main objective. |
| Uncertainty-weighted gradients | `UWG_Brier`: detached binary multiclass Brier weighting of BCE gradients, with a 0.05 floor for stability. |
| Calibration-regularized training | `Brier_regularized`: BCE plus binary Brier loss. |
| Fishr | `Fishr_last_layer`: match per-domain variances of analytic per-sample final-layer BCE gradients; no moving-average estimator. |
| V-REx | `VREx`: mean training-domain risk plus variance of training-domain risks. |
| EQRM | `EQRM_empirical`: minimize the empirical 0.9 quantile of observed training-domain risks; a discrete-quantile adaptation, not smoothed distribution extrapolation. |
| Fish | `Fish_first_order`: ordinary BCE on each full batch, followed by sequential SGD steps on up to four training domains and a first-order parameter interpolation toward the resulting weights. This ERM-plus-Fish hybrid ensures every training row contributes. |

Domain methods require at least two observed training domains with 50 training rows. Heart states and hypertension regions are not hospitals. MIMIC has one institution; diabetes has no suitable domain ID. The original respiratory baseline is preserved, while its in-processing branch now uses new hospital-disjoint preprocessing and PCA fitted on training hospitals only. It has 137/34/29 train/validation/test hospitals. The eICU diagnosis branch has 141/34/31 disjoint hospitals. Both purge patients crossing partitions. Source-fitted transformations are frozen for external evaluation. Cross-database institutional overlap cannot be ruled out from anonymous eICU IDs; diagnosis documentation differs across sources. Internal eICU hospital separation is directly verified.

## Plots and artifacts

Save every model, normalization, group mappings, training history, validation/test/external probability arrays, support-aware metrics, subgroup tables, reliability bins and provenance hashes. Export 20 selected complete prediction CSVs across six branches (three splits for each of three surveys and respiratory mortality; four splits including external for eICU and MIMIC). Retain source latent features, labels and metadata. MIMIC/eICU add predictions for all four tasks. Produce before/after ERM-versus-selected utility, fairness, calibration and domain plots, complete method comparison plots, a local HTML dashboard, one notebook, and read-back validation. The comparison is between prediction models; it is not a before/after change to patients or their labels.

## Primary references

- FairCare: https://doi.org/10.1016/j.ipm.2024.103682
- Adversarial fairness: https://arxiv.org/abs/1802.06309
- DANN: https://www.jmlr.org/beta/papers/v17/15-239.html
- GroupDRO: https://arxiv.org/abs/1911.08731
- FOMO: https://cavalab.org/projects/fomo/
- AvUC: https://arxiv.org/abs/2012.07923
- UWG: https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Uncertainty_Weighted_Gradients_for_Model_Calibration_CVPR_2025_paper.html
- Fishr: https://proceedings.mlr.press/v162/rame22a.html
- DomainBed reference implementations (Fish, V-REx, EQRM): https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
