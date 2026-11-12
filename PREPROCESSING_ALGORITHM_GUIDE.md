# Preprocessing algorithms and Sample 10 guide

All four preprocessing notebooks now contain per-section algorithm locations, one-line implementation explanations, recorded use and full metric comparisons. All four Sample 10 notebooks contain complete saved before/after field tables for every record and stage, prior-checkpoint evidence for omitted trace fields, raw and final exported rows, removal status, weights and overall comparisons.

| Dataset | Preprocessing notebook | Complete Sample 10 notebook |
|---|---|---|
| heart | [Open](heart/heart_preprocessing.ipynb) | [Open](heart/heart_sample10_comparison.ipynb) |
| diabetes | [Open](diabetes/diabetes_preprocessing.ipynb) | [Open](diabetes/diabetes_sample10_comparison.ipynb) |
| hypertension | [Open](hypertension/hypertension_preprocessing.ipynb) | [Open](hypertension/hypertension_sample10_comparison.ipynb) |
| respiratory | [Open](respiratory/respiratory_preprocessing.ipynb) | [Open](respiratory/respiratory_sample10_comparison.ipynb) |

## Selected algorithms compared with traditional baselines

Mean rank is lower-is-better and comparable only within the same dataset and bias. The project selects from its modern family even where a traditional method ranks ahead. Empty results indicate a skipped stage.

| Dataset | Bias | Selected modern | Modern mean rank | Best traditional | Traditional mean rank | Modern ranks ahead |
|---|---|---|---|---|---|---|
| heart | Duplicate | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| heart | Missing data | HyperImpute | 3.2857142857142856 | GMM | 3.0 | False |
| heart | Outlier | EBM | 2.4285714285714284 | One-Class SVM | 2.857142857142857 | True |
| heart | Data noise | Robust loss (GCE) | 1.1666666666666667 | Denoising autoencoder | 2.8333333333333335 | True |
| heart | Feature scale | PowerTransform (Yeo-Johnson) | 3.0 | Min-Max scaler | 3.3333333333333335 | True |
| heart | Encoding / domain | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| heart | Class imbalance | TabDDPM | 2.7777777777777777 | Borderline-SMOTE | 3.2222222222222223 | True |
| heart | Demographic | Adversarial debiasing | 2.8125 | Optimal Transport | 3.625 | True |
| diabetes | Duplicate | Ditto (transformer matcher) | 2.142857142857143 | GBER (graph-based ER) | 2.357142857142857 | True |
| diabetes | Missing data | MIP-VAE | 4.428571428571429 | MICE | 2.857142857142857 | False |
| diabetes | Outlier | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| diabetes | Data noise | Robust loss (GCE) | 1.4166666666666667 | Denoising autoencoder | 3.3333333333333335 | True |
| diabetes | Feature scale | PowerTransform (Yeo-Johnson) | 2.5 | Min-Max scaler | 3.3333333333333335 | True |
| diabetes | Encoding / domain | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| diabetes | Class imbalance | TabDDPM | 3.111111111111111 | Borderline-SMOTE | 3.0 | False |
| diabetes | Demographic | Adversarial debiasing | 2.9375 | Optimal Transport | 2.875 | False |
| hypertension | Duplicate | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| hypertension | Missing data | HyperImpute | 3.2857142857142856 | KNN | 3.2857142857142856 | False |
| hypertension | Outlier | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| hypertension | Data noise | Confident Learning (Cleanlab) | 1.0 | Denoising autoencoder | 3.0 | True |
| hypertension | Feature scale | PowerTransform (Yeo-Johnson) | 2.0 | Robust scaler | 3.3333333333333335 | True |
| hypertension | Encoding / domain | Domain-generalisation transformer | 2.375 | CORAL | 2.75 | True |
| hypertension | Class imbalance | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| hypertension | Demographic | Counterfactual recourse (DiCE) | 2.5 | Optimal Transport | 3.375 | True |
| respiratory | Duplicate | - (bias not detected: nothing applied) | not available | not available | not available | not available |
| respiratory | Missing data | MIP-VAE | 3.5714285714285716 | Random Forest | 2.857142857142857 | False |
| respiratory | Outlier | ECOD | 3.4285714285714284 | LOF | 1.7142857142857142 | False |
| respiratory | Data noise | Robust loss (GCE) | 1.1666666666666667 | Denoising autoencoder | 4.5 | True |
| respiratory | Feature scale | SNN (SELU) | 2.8333333333333335 | Quantile transform | 2.8333333333333335 | False |
| respiratory | Encoding / domain | Domain-generalisation transformer | 2.875 | CORAL | 3.5 | True |
| respiratory | Class imbalance | TabDDPM | 3.3333333333333335 | SMOTE | 3.7777777777777777 | True |
| respiratory | Demographic | Fair contrastive learning | 2.75 | Reweighting | 2.5 | False |

## Verification and limitations

Documentation was generated from the existing notebook code, comparison CSVs and trace JSON. All existing code cells, execution counts and outputs were preserved exactly; models were not rerun. Only explanatory notebook cells and supporting explanation tables were added. Validation checked ten unique raw IDs, eight trace stages, final record survival, all algorithm explanations and preservation of executable cells. Some historical before-snapshots omit features after a schema change; those values are labelled not captured, with the previous checkpoint shown separately.

Synthetic rows can inherit original record IDs. The saved trace retains only the last snapshot per ID, which can replace an original record with a synthetic descendant after resampling. The Sample 10 appendices distinguish every original and synthetic final row. This documentation identifies the limitation without changing the pipeline.
