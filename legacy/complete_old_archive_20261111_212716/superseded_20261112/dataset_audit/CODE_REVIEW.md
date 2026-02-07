# Current pipeline code and method review

Reviewed for the user's preprocessing -> latent -> in-processing recheck on 12 November 2026.

**The saved baseline pipeline is reproducible, but it is not an exact implementation of every algorithm shown in the method images.** Separate dashboard pages now expose the actual calculations. A passing integrity test does not mean that every bias was removed or that a method improves performance.

## Corrections in this review

- Replaced the combined dashboard entry pages with one HTML file per preprocessing bias, latent diagnostic, or in-processing bias family. The layout follows the old examples in `D:/plots`, while values come exclusively from the current saved cohorts and models.
- Corrected the individual-page builder's family comparison: demographic pages use demographic EO gaps and intersectional pages use intersectional EO gaps. They no longer both use the overall maximum across all protected groups.
- Added missing dashboard JavaScript/CSS, local Plotly assets, separate-file navigation, model/partition/variable controls and downloadable aggregate plot data.
- Explicitly labelled histogram density, full-range box whiskers, before/after empirical Q-Q comparisons, missing measurements, unsupported groups and diagnostic screening rules.
- Corrected stale reproduction commands and report locations following the folder move. The preprocessing `--finalize` path now reads current dataset validation records rather than requiring an archived mixed audit snapshot.
- Corrected the latent protocol wording: the variance-retention refinement triggers when the initially selected candidate fails an individual target's guard. The former wording incorrectly said that every candidate must fail. Saved selections and results were preserved.

## What the code actually does

| Stage/family | Review finding |
|---|---|
| Preprocessing | Training-only median imputation, missing indicators, scaling and one-hot encoding. Original survey/respiratory reference cohorts are retained; hospital experiments use separately fitted hospital-disjoint inputs. This is a fixed baseline. |
| Eight preprocessing biases | Eight separate diagnostic pages exist per branch. The current baseline does not run all methods in `pre_methods_metrics.jpeg`. It does not resample classes, remove statistical outliers, adjudicate duplicate people, or establish denoising. Unchanged or unavailable results are shown explicitly. |
| DGCCA | Paired feature views within each record, a regularized correlation objective with reconstruction, and a training-fitted final projection. It does not pair unrelated people across databases. |
| SAINT | Feature attention plus row attention to a fixed training reference bank. This is an inductive SAINT-style adaptation; it is not the original implementation's complete experiment. |
| MV-DAE / PCA | View/input corruption occurs only during training. PCA, input normalization and latent scaling fit training rows. Wider PCA alternatives preserve validation utility when the initial bottleneck loses information. |
| Latent metrics | Validation probes, sampled same-row CKA/silhouette and training-fitted domain clustering have distinct meanings. Missing domain metrics stay unavailable. Demographic-anchor comparisons do not prove common clinical semantics. |
| Adversarial fairness / DANN | Gradient reversal on supported demographic or actual source-domain labels. The tabular fairness model is not graph FairCare. DANN does not use held-out/external-domain data for training here. |
| GroupDRO / intersectional GroupDRO | Exponentially updated group-risk weights, with ordinary BCE mixed in. Groups are constructed from training metadata and labels. This is a declared regularized adaptation. |
| Fair-ERM / constrained fairness | Conditional group-mean probability penalties and projected dual ascent. These optimize soft surrogates, not guaranteed hard-decision equalized odds. |
| FOMO | A bounded Pareto search over intersectional logistic-regression sample weights, followed by full-training-data refitting. It is FOMO-inspired, not the original package's default implementation. |
| AvUC / CRL / UWG / Brier | Deterministic uncertainty/correctness-history/Brier-based loss adaptations. Probability calibration is measured on saved holdouts. Good calibration is not guaranteed. |
| Fishr / V-REx / EQRM / Fish | Final-layer gradient-variance matching, supported domain-risk variance, empirical risk quantiles, and an ERM-plus-first-order Fish hybrid. They are deliberately simplified. Multiple source domains are required. |
| Selection and metrics | Validation chooses checkpoints, classification thresholds and the overall model. Frozen selection hashes precede test/external scoring. Prediction probabilities, thresholded decisions, subgroup support, reliability bins and summary metrics are rechecked. |

Primary method descriptions: [DGCCA paper](https://aclanthology.org/W19-4301/), [official SAINT implementation](https://github.com/somepago/saint), [DomainBed reference algorithms](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py). The detailed project mapping is in [the in-processing protocol](../inprocessing/code/PLAN.md) and [the latent protocol](../latent/code/PLAN.md).

## Code-file coverage

All stage Python files and notebook code cells are syntax-checked. The [complete file inventory](active_recheck/ALL_STAGE_FILES.csv) records every non-cache stage file, its role, size and SHA256. The [dashboard validation](active_recheck/dashboard_validation.json) lists the exact Python files checked.

The computational review covers preprocessing fit/split/feature rules, all latent model objectives and inference paths, utility refinement, hospital encoding, safe input loaders, in-processing loss functions, training/selection/scoring, exports, report generators and validation code. Rendering/extraction helper files are checked for current paths and syntax. The nine notebooks are viewers of saved results; their outputs are not evidence of newly retrained models.

Fresh artifact checks: [pipeline check results](active_recheck/checks.json). Eight in-processing contract tests cover finite gradients, query independence, group support, excluded unknown groups, gradient reversal, analytic Fishr gradients, conditional soft-EO penalties and ECE endpoint/empty-bin behavior. Four latent contract tests cover finite gradients, batch-independent inference, feature-view coverage, CKA invariance and anchor inputs. The separate [dashboard control tests](active_recheck/dashboard_control_tests.json) exercise the JavaScript using a minimal DOM and a Plotly stub; this is not a browser-rendering test.

## Files retained and excluded

Known superseded experiment files are checked against the cleanup list and must be absent. The earlier archive remains at `D:/complete_old_archive_20261111_212716`. Current non-selected models, candidate embeddings, fixed reference baselines and summary PNG diagnostics are retained because they support the current comparisons. They are not the superseded experiments. Original source downloads and research specifications remain inputs/provenance.

## Limits that remain

- Only the eICU hospital holdouts establish directly verified internal hospital separation. Survey tasks have no hospital IDs. MIMIC has one institution; MIMIC/eICU transfer is reported separately, with unknown cross-database institutional overlap.
- These are short, single-seed research experiments. Strong performance, improved fairness, uncertainty bounds and universal generalization are not established by code/integrity checks.
- The original MIMIC/eICU preprocessing pickle records scikit-learn 1.9.1, whereas validation runs under 1.8.0. Independent reconstruction and actual transformed values match, but compatibility across versions is not certified.
- Browser access is unavailable. Local HTML, chart-schema, numerical-data and control-logic checks pass only as documented in their reports; live browser visual review remains incomplete.
