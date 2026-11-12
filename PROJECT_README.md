# Healthcare bias mitigation — Group 18 (pre-processing, latent and in-processing stages)

Four diseases, four official datasets, eight data biases, every traditional and modern method from the
method table compared on the table's metrics, the best **modern** method applied — then one common latent
space for the four diseases and a cross-hospital ICU cohort, and in-processing methods that train the model
itself to be fair, calibrated and robust across hospitals.

## Folder layout

```text
heart/  diabetes/  hypertension/  respiratory/      (same layout for every disease)
    raw/<disease>_raw.csv                 data BEFORE preprocessing (biases left in)
    raw/source/                           official source files + cohort extracts
    <disease>_preprocessing.ipynb         complete 8-bias preprocessing notebook (the code)
    <disease>_sample10_comparison.ipynb   10 real records followed through every bias step
    preprocessed/                         <disease>_{train,validation,test}_preprocessed.csv,
                                          comparison_<bias>.csv (every method x every metric),
                                          modern_vs_traditional.csv, bias_summary.csv,
                                          end_to_end_check.csv, pipeline_log.json, sample10_trace.json
    plots/                                one interactive dashboard per bias + index.html
    <disease>_inprocessing.ipynb          in-processing notebook (4 biases x 4 methods x 4 metrics, ERM = before)
    <disease>_inprocessing_sample10.ipynb 10 test records: ERM vs the selected method for every in-processing bias
    inprocessed/                          comparison_<bias>.csv, test_metrics_<bias>.csv, inprocessing_summary.csv,
                                          predictions, selection.json, models/
    plots/inprocessing/                   one dashboard per in-processing bias + index.html

hospital/                                 cross-hospital ICU cohort (eICU-CRD hospitals + MIMIC-IV), see hospital/README.md
    raw/, code/, preprocessed/            prepared by the Codex session; restored and race labels corrected 12 Nov 2026

latent/
    latent_alignment.ipynb, latent_alignment.py, latent_dashboard.py    latent code
    data/combined_latent_<split>.csv      four diseases + the cross-hospital cohort in one common 16-D latent space
    data/<dataset>_latent_<split>.csv     the same rows per dataset (hospital_latent_external.csv = MIMIC)
    latent_method_comparison.csv, latent_report.json
    plots/                                one dashboard per latent metric + index.html

code/build_raw/      scripts that build <disease>/raw/<disease>_raw.csv from the official files
code/build_website.py  refreshes the Team-18 website (D:/Team-18/Team-18) from these results
code/recheck.py      read-only check of every output above + the website -> RECHECK_REPORT.md
code/inprocessing/   in-processing source (inprocessing_cells.py), dashboard template, make_notebooks.py
docs/                project report, slides, earlier copies of the method/metric tables

pre_methods_metrics.jpeg, latent methods.png, latent metrics.png, inprocessing.png
                     the method/metric tables the selections follow
```

## Datasets

| Disease | Source | Target |
|---|---|---|
| Heart | CDC BRFSS 2022 | ever told: heart attack or coronary heart disease / angina |
| Diabetes | CDC NHANES Aug 2021 - Aug 2023 | doctor-diagnosed diabetes |
| Hypertension | CDC NHIS 2023 Sample Adult | clinician-diagnosed hypertension |
| Respiratory | PhysioNet eICU-CRD v2.0 | in-hospital death after the first 24 h in ICU |

## Pre-processing (per disease notebook)

Order: duplicates -> split 70/15/15 -> missing data -> outliers -> data noise -> feature scale ->
encoding/domain -> class imbalance -> demographic. For every detected bias all traditional and modern methods
(`docs/pre_methods_metrics.jpeg`) are run and ranked on every metric; the **modern** method with the best mean rank
is applied, the best traditional method is kept as the comparison (`modern_vs_traditional.csv` shows why).
Everything is fitted on training rows; methods are scored on training/validation rows; the test split is only
used for the final end-to-end check.

Feature-scale safeguard (`CollapseGuard` in the feature-scale cell): if the selected scaler turns a varying column
into a constant (Yeo-Johnson did this to the respiratory `temperature_c_max` in the first run),
that one column falls back to plain standardisation fitted on the training rows; the notebook prints which columns
used the fallback. It changes nothing for heart, diabetes and hypertension.

## Latent stage

`latent/latent_alignment.ipynb`: DGCCA, SAINT and MV-DAE learn one shared 16-dimensional space for the four
diseases and the cross-hospital ICU cohort (a per-dataset PCA is the "before" reference). MIMIC rows of the cohort are only
encoded (never trained or scored on) so the in-processing cross-hospital check stays external. The methods come from `latent methods.png`; the best
one is chosen by the metrics in `latent metrics.png`: cross-dataset CKA, Macro-F1, Silhouette, NMI/ARI.

## In-processing stage

`<disease>/<disease>_inprocessing.ipynb` (generated by `code/inprocessing/make_notebooks.py`) trains one small network on
the latent features with every method of `inprocessing.png`: demographic prediction dependence (FairCare, DANN, GroupDRO,
Fair-ERM; CMI, PAD, WGER, EOV), intersectional unfairness (FOMO, intersectional debiasing, fairness-constrained optimization,
GroupDRO-intersectional; HVI, DF-epsilon, MSFV, IWGR), miscalibration (AvUC, CRL, uncertainty-weighted gradient,
calibration-regularized; UCE, AURC, E-AURC, MMCE) and cross-hospital generalization failure (Fishr, V-REx, EQRM, Fish;
IEGVD, ERV, UQER, IDGA). Plain training (ERM) is the before. Per bias the method with the best mean rank over its four
metrics on validation is selected (AUROC within 0.05 of ERM) and reported once on the test split. The cross-hospital bias
uses the cross-hospital ICU cohort with the disease's recorded-condition label: eICU training hospitals for training, eICU
validation hospitals for selection, eICU test hospitals + MIMIC-IV (another hospital system) for the final check.

## Environment

Python 3.12 virtual environment `Heart_correct/.venv` (left in place so the editor's interpreter keeps working);
Jupyter kernel **"Python 3.12 (D:\complete venv)"** (`complete-venv`). Run a notebook from the command line with
`python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=complete-venv <notebook>`.
