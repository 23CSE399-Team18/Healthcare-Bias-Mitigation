# MIMIC preprocessing comparison: independent design review

Reviewed 9 November 2026 against `pre_methods_metrics.jpeg`, the verified MIMIC
tables, and the existing four preprocessing notebooks. This report does not
claim that a modern method must outperform a traditional baseline.

## Recommended experiment

Run four separate recorded-diagnosis tasks on the existing patient-disjoint
MIMIC train/validation/test assignments. Fit every imputer, scaler, sampler,
anomaly threshold, feature selector, weighting rule and learned representation
on training patients only. Use group-disjoint folds within training for noise
scores and internal tuning. Select methods and decision thresholds on validation
patients; keep the final test population unchanged and locked until selection.
Include the unmodified baseline and no-operation option in every comparison.

Do not force the winner to be modern. Display the strongest measured traditional
and modern candidates and the overall validation-selected winner. If modern
methods lose or tie within uncertainty, report that result. An all-metric mean
rank is not automatically defensible: duplicate or redundant metrics, missing
values, unclear directions and surrogate objectives can change rankings.
Predeclare a primary task metric and utility/fairness guards; show the remaining
metrics as diagnostics. A method with missing required comparison outcomes is
not a winner by averaging fewer available metrics.

Rare ICU physiology can be clinically meaningful. Anomaly flags, class imbalance,
missingness and demographic differences are audit signals, not proof of bias or
measurement error. Missing-code labels are documentation outcomes, not verified
disease absence. Avoid claiming that preprocessing corrects clinical diagnoses.

## Practical comparisons for all eight rows

| Image row | Practical traditional/reference methods | Appropriate additional candidates | Main restriction |
|---|---|---|---|
| Missing data | Median plus missing indicators; KNN; Bayesian-ridge IterativeImputer | Random-forest/ExtraTrees IterativeImputer; genuine HyperImpute if installed | Call the iterative RF variant by its actual implementation, not exact MissForest. Evaluate held-out observed values with fixed masks; true values at naturally missing cells remain unknown. |
| Class imbalance | No resampling; random oversampling; SMOTENC; SMOTE/ADASYN only with a justified numeric representation | KMeansSMOTE on a declared numeric representation; genuine CTGAN/TabDDPM as optional generation experiments | Resample training only. Preserve categorical validity and clinical min<=mean<=max constraints. Synthetic rows must be flagged. Newer generators do not guarantee fairness or better accuracy. |
| Outlier | No removal; robust bounds/winsorization; IsolationForest; LOF novelty; OneClassSVM | ECOD, COPOD; DeepSVDD only if its actual training is implemented | Fit/score training records for any training-only removal; untouched validation/test cases remain eligible. Ground-truth outlier metrics are unavailable without a separately identified perturbation benchmark. |
| Feature scale | Standard, MinMax, Robust and Quantile transforms | Yeo-Johnson PowerTransformer | PowerTransformer is a conventional statistical transform, not evidence of present state of the art. Do not transform category codes as continuous physiology. |
| Demographic | No mitigation; Kamiran-Calders reweighing | Fairlearn CorrelationRemover, with declared sensitive features and fitted alpha | Removing linear correlation is not proof of fairness. Check subgroup TPR/FPR and utility. Reweighing uses labels only in training. |
| Duplicate | Exact repeated stay-key/payload audit; verified duplicate removal | LSH candidate retrieval followed by identity/provenance verification | Repeated patient visits are valid stays, and equal physiological vectors across different people are not proven duplicates. The existing unique stay keys mean no forced deduplication is needed. |
| Data noise | No modification; robust plausibility checks; cross-fitted model-disagreement flags | Genuine cleanlab confident-learning flags; optional bounded training weights | Model disagreement is not clinical ground truth. No automatic relabeling of ICD records. Median/wavelet/Savitzky-Golay filters need a real ordered signal, not arbitrary patient-row order. |
| Encoding/domain | Train-fitted one-hot encoding with unknown-category handling; no adaptation | CORAL/OTDA only in a separately declared target-covariate adaptation experiment | MIMIC has one hospital. Domain adaptation metrics and cross-hospital claims require an actual other domain, such as a separately partitioned eICU source. |

The image also contains methods from other stages. Focal/class-balanced losses,
robust losses, SELU/SNN, FT-Transformer, adversarial debiasing, fairness-aware
contrastive learning, DANN and IRM involve model/representation learning. They
must be classified accordingly, not reported as ordinary data cleaning. DiCE
generates post-hoc counterfactual explanations and is not itself a fairness
mitigation. EBM is a supervised predictor; an EBM-residual anomaly detector is a
custom research construction, not an established detector named EBM. A bare
autoencoder is not automatically a domain adaptation method.

## Every image metric: what can be reported honestly

| Row | Metrics that can be computed under an explicit protocol | Metrics requiring additional evidence or marked unavailable |
|---|---|---|
| Missing | Wasserstein distance, correlation preservation, MAE, RMSE, NRMSE on the same held-out observed-value mask; normalize across features with training scales | CRPS needs a declared predictive distribution/ensemble; a deterministic point-mass version reduces to MAE and is not uncertainty evidence. NLL needs a predictive density; do not invent variance. Naturally missing values have no known reconstruction error. |
| Class imbalance | Imbalance ratio, class entropy, macro-F1, balanced accuracy, PR-AUC, G-mean; KS/TVD on declared, comparable class-conditional feature distributions | “Balanced F1-score” has no single independently specified formula in the image. Define it explicitly; if macro-F1 is used, label it an alias and do not count it twice in ranking. PR-AUC must distinguish average precision from trapezoidal area. |
| Outlier | Score stability and neighborhood consistency as defined surrogate diagnostics; downstream disease-classifier AUROC/AUPRC/F1/Brier with their correct task names | Outlier-detection AUROC/AUPRC/Precision@K/F1/Brier require true anomaly labels. Disease diagnosis labels are not outlier labels. Uncalibrated anomaly scores are not probabilities for Brier scoring. |
| Scale | Skewness, kurtosis, rank/linear correlation diagnostics; downstream task metrics | Before/after Wasserstein is not comparable across different units; use a common declared coordinate system. Activation variance/gradient stability require an actual common neural probe and training protocol. They cannot be obtained from preprocessing tables alone. |
| Demographic | Statistical parity difference, disparate impact ratio/difference, equal opportunity difference and equalized odds with declared reference groups, thresholds and denominator support | Counterfactual validity, proximity and sparsity need actual counterfactual outputs, constraints and distance definitions. They do not follow from fairness gaps. Zero denominators mean unavailable, not zero disparity. |
| Duplicate | Exact-key duplicate rate and verified redundancy reduction; candidate/reduction ratios with definitions | Pair precision/recall/F1 and pair completeness need identity ground truth. A controlled duplicate-copy benchmark may provide engineering metrics, but must be separate from real-patient duplicate claims. |
| Noise | Reconstruction residual variance/MAE/RMSE only for an explicit observed-value corruption/reconstruction experiment; label-issue rates as unverified flags | Spectral distortion and temporal SNR require actual ordered signals. Label-error detection F1 needs independently known errors or clearly labeled controlled flips. Original billing labels are not known clinical truth. |
| Domain | With two actual domains: MMD, cross-fitted domain-classifier AUC, Wasserstein distance; worst-group task accuracy with support thresholds | Domain generalization gap, target-domain AUROC and OOD performance need a prespecified source/target evaluation and cannot be estimated within a single hospital merely by renaming demographic groups. |

Use a status/reason field for each metric and method: measured, not applicable,
dependency unavailable, not implemented, insufficient subgroup support, or
requires ground truth. Preserve NaN/NA in exported tables. Never replace an
unavailable result by zero or a favorable constant. Controlled masks/corruptions
are evaluation perturbations of observed training/validation records, not new
clinical ground truth, and must be labeled as such.

For binary fairness, state whether SPD is a signed reference-group difference
or a multigroup max-minus-min range. Define DI and its difference consistently.
Equalized odds can be max(TPR range, FPR range). Report subgroup denominators and
suppressed/unknown results. Use patient-cluster bootstrap intervals when stays
repeat; row bootstrap would ignore this dependence. A 0.8 impact-ratio heuristic
is not a clinical or legal certification of fairness.

## Important implementation traps found in existing notebooks

- `rank_methods` explicitly sets `SELECT_FAMILY="Modern"` and selects the best
  modern method even when a traditional method ranks higher. Do not reuse that
  selection rule for a claimed objective winner.
- `pc1_order` and `along_order` manufacture a patient sequence using principal
  component order, then apply temporal denoisers across unrelated patients.
  This is not a clinical time series and must not be copied into the MIMIC
  aggregate-feature notebook.
- `rs_tabddpm` uses a custom all-Gaussian diffusion generator followed by rounding
  categories. The original TabDDPM combines distributions appropriate for
  continuous and categorical features. The custom variant must not be labeled
  a faithful original TabDDPM implementation.
- Genuine package calls exist for HyperImpute and CTGAN, and standard sklearn
  imputers/scalers and imbalanced-learn samplers can be reused after checking
  fit/transform boundaries. Do not reuse earlier result CSVs as MIMIC results.
- The existing EBM detector is explicitly a custom contextual residual score.
  If retained, describe that construction and its validation rather than
  presenting EBM alone as a standard modern outlier method.
- PyOD ECOD/COPOD `decision_function` currently concatenates query records with
  training records when recomputing empirical distributions. For a strict
  frozen-training reference, either score/remove training records only or use
  an explicitly named frozen-reference ECDF variant. Do not claim the stock
  query scoring uses only training distribution statistics.

CORAL needs unlabeled target features to align covariance. Fitting it on the
external test population is a transductive adaptation setting, not untouched
external generalization. If adaptation is studied, reserve distinct target
adaptation and target evaluation patients. MIMIC shifted calendar dates must
not be silently treated as a real common calendar-era domain.

## Primary sources checked

- [scikit-learn leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
  and [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html).
- [MissForest original paper](https://arxiv.org/abs/1105.0828),
  [HyperImpute original ICML paper](https://proceedings.mlr.press/v162/jarrett22a.html),
  [CSDI original paper](https://arxiv.org/abs/2107.03502).
- [SMOTENC official API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html),
  [CTGAN original NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html),
  [TabDDPM original paper](https://arxiv.org/abs/2209.15421).
- [ECOD original paper](https://arxiv.org/abs/2201.00382),
  [COPOD original paper](https://arxiv.org/abs/2009.09463),
  [PyOD ECOD implementation](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ecod.html),
  [PyOD COPOD implementation](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/copod.html).
- [scikit-learn feature preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html),
  [AIF360 reweighing API and original reference](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.preprocessing.Reweighing.html),
  [Fairlearn preprocessing](https://fairlearn.org/main/user_guide/mitigation/preprocessing.html).
- [Confident learning original paper](https://arxiv.org/abs/1911.00068),
  [cleanlab filter API: out-of-sample probabilities](https://docs.cleanlab.ai/stable/cleanlab/filter.html),
  [CORAL original author paper](https://arxiv.org/abs/1612.01939),
  [DiCE original explanation paper](https://arxiv.org/abs/1905.07697).

The image does not identify a unique source for the terms MIP-VAE or GBER. Those
labels should remain source-unverified until a precise paper or implementation
is supplied; a similarly named homemade algorithm is not a verified substitute.
CSDI can be adapted beyond time series, but the exact tabular adaptation must
be stated and implemented rather than treating heterogeneous feature columns
or arbitrary patient order as a temporal signal.
