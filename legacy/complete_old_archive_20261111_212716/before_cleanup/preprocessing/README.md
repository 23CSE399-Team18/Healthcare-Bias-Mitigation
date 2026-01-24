## Verification update — 11 November 2026

These saved outputs are historical preprocessing experiments. Some methods were selected using test-set scores; those test metrics are exploratory and cannot be treated as an independent evaluation. The plots also contain presentation limitations documented in [the audit](audit_current/REPORT.md).

Use the [verified baseline plots](verified_baseline/index.html) and the split CSVs under `verified_baseline/<dataset>/` for fixed train-only preprocessing. Reproduce them with `python preprocessing/audit_and_prepare.py`. The baseline does not claim completion of the eight-bias comparison or fairness improvement.

---

# Historical bias-aware preprocessing notebooks

The notebooks below preserve the earlier bias-mitigation experiments and their saved outputs.
The current latent pipeline reads the verified baseline files described above. See the
[complete-folder recheck](../dataset_audit/recheck_current/REPORT.md) for the current status.

| Notebook | Input (model-ready) | Dashboards (before vs after) | Outputs |
|---|---|---|---|
| `heart_preprocessing.ipynb` | `model_ready/heart_model_ready.csv` | `D:/plots/heart_preprocessing/` | `outputs/heart/` |
| `hypertension_preprocessing.ipynb` | `model_ready/hypertension_model_ready.csv` | `D:/plots/hypertension_preprocessing/` | `outputs/hypertension/` |
| `diabetes_preprocessing.ipynb` | `model_ready/diabetes_model_ready.csv` | `D:/plots/diabetes_preprocessing/` | `outputs/diabetes/` |
| `respiratory_preprocessing.ipynb` | `model_ready/respiratory_model_ready.csv` | `D:/plots/respiratory_preprocessing/` | `outputs/respiratory/` |

Open `D:/plots/<disease>_preprocessing/index.html` first: it links the eight bias dashboards and shows
the end-to-end check.

## What each notebook does

For each of the 8 biases in the project table (`pre_methods_metrics`):

1. **Detect** the bias with a stated statistic and threshold, on the data at that stage.
2. If detected, run **every listed method**, traditional and modern.
3. Score every method with **every listed metric**, rank all methods together on each metric, and average the ranks.
4. **Select the best-ranked modern method** and apply it. Traditional methods are compared as baselines
   only, and the best traditional method is reported for reference.
5. Write an interactive **before vs after dashboard**. If the bias is not detected, nothing changes and
   the dashboard shows identical before/after distributions.

## Dashboard layout

Every dashboard uses the same layout as the reference dashboards in `D:/plots`: a sidebar with plot type,
variable selector, dataset info (including the detection result and selected method) and legend; then the
distribution comparison (overlaid Before/After bars), box-plot comparison, Q-Q plot, Before/After
correlation heatmaps, the **bias heatmap**, the method-comparison table, and a footer note.

In the bias heatmap each row is a variable or patient group and each cell is one example patient, shown in
a Before grid and an After grid (the same patients in both). The percentage under each grid is the rate over
all rows of the split. When a condition is rare (under 10% of rows), half of the example cells are drawn
from patients who have it, so it stays visible; the heatmap note says when this applies.

| Bias | Heatmap rows | Coloured cell |
|---|---|---|
| Missing data | predictors with most missing values | value missing |
| Duplicate | duplicate checks | record is a duplicate candidate (grey in After = removed) |
| Outlier | predictors with most extreme values | value beyond the skew-adjusted extreme fence (grey = removed) |
| Data noise | all patients and each protected group | label contradicted by Confident Learning (grey = pruned) |
| Monitor signals (ICU) | vital-sign signals | stay has spikes (red = more than 2% of points) |
| Feature scale | predictors, largest to smallest magnitude | value above one common magnitude line |
| Encoding / domain | largest domains (state, region or hospital) | predictors reveal the patient's domain |
| Class imbalance | all rows and each group (20 cells) | share of cases (light red = generated cases) |
| Demographic | protected and audited groups | test case missed by the classifier (false negative) |

Pipeline order: duplicate → train/validation/test split → missing → outlier → noise → feature scale →
encoding/domain → class imbalance → demographic.

## Safeguards (why some treatments are deliberately conservative)

| Step | Safeguard | Reason found while testing |
|---|---|---|
| Split | A **validation split** (10% of training patients) is transformed like the test split but never pruned or resampled; every decision threshold is tuned on it | Thresholds tuned on rows that fed an oversampler were over-confident and lowered balanced accuracy |
| Duplicate | A match counts only if both records share the respondent-level design columns, demographics and outcome, and differ in at most 15% of predictors | Real duplicates are rare (0-3 per dataset); without this, matchers removed hundreds of different people with look-alike answers |
| Data noise | Recorded cases are never deleted; every method prunes at most *k* recorded non-cases (Confident Learning's estimate); thresholds are estimated within each protected group | Pooled Confident Learning flagged 37% of women's heart-disease cases vs 19% of men's as "noise"; pruning them widened the sex gap |
| Domain, demographic | Representation learners get a fixed 1,500-3,000-step training budget; an AUROC accuracy guard is ranked with the fairness/invariance metrics | Prevents a method winning by making every group or hospital equally poor |
| Wide ICU data | Lighter settings for per-column learners (HyperImpute, random-forest/MICE imputation, EBM, GMM) | Keeps 170-predictor data tractable |

## Outputs per disease

- `<disease>_train_preprocessed.csv`, `<disease>_validation_preprocessed.csv`, `<disease>_test_preprocessed.csv`:
  inputs are every column except `meta_*`, `sens_*` and `target`; use `meta_sample_weight` as the
  sample weight; `meta_synthetic = 1` marks generated training rows.
- `<disease>_*_feature_space.csv`: written only when a selected modern method replaced the predictors
  with a learned representation; an interpretable copy from just before that step.
- `comparison_<bias>.csv`: every method × every metric, with mean ranks.
- `bias_summary.csv`, `end_to_end_check.csv`, `pipeline_log.json`, `run_progress.log`.

## Running

Kernel: **Python (biasprep)** (conda env `C:\Users\ganap\envs\biasprep`, Python 3.11, PyTorch 2.6 + CUDA).
Open a notebook and *Run All*. Approximate runtimes on the RTX 3050 laptop: diabetes 7 min,
hypertension 20 min, heart 15-20 min, respiratory about 1 hour.
