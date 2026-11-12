# Recheck report

327 of 327 checks passed.

| Area | Check | Result | Detail |
|---|---|---|---|
| heart | raw data before preprocessing | PASS | 440,111 rows |
| heart | preprocessing notebook executed without errors | PASS | 38 code cells, 0 not executed, 0 with errors |
| heart | 10-sample notebook executed without errors | PASS | 11 code cells, 0 not executed, 0 with errors |
| heart | train: all predictor values finite | PASS | 551,950 rows x 32 predictors |
| heart | train: no predictor collapsed to a constant | PASS | every predictor varies |
| heart | train: binary label | PASS |  |
| heart | train: rows match the run log | PASS | 551,950 vs 551,950 |
| heart | validation: all predictor values finite | PASS | 66,017 rows x 32 predictors |
| heart | validation: binary label | PASS |  |
| heart | validation: rows match the run log | PASS | 66,017 vs 66,017 |
| heart | validation: no synthetic rows | PASS |  |
| heart | test: all predictor values finite | PASS | 66,017 rows x 32 predictors |
| heart | test: binary label | PASS |  |
| heart | test: rows match the run log | PASS | 66,017 vs 66,017 |
| heart | test: no synthetic rows | PASS |  |
| heart | train / validation / test records are disjoint | PASS |  |
| heart | same predictors in every split | PASS | 32 predictors |
| heart | Missing data: detected, comparison table saved | PASS | 8 methods |
| heart | Missing data: applied method is modern and in the table | PASS | selected HyperImpute; best traditional GMM |
| heart | Class imbalance: detected, comparison table saved | PASS | 9 methods |
| heart | Class imbalance: applied method is modern and in the table | PASS | selected TabDDPM; best traditional Borderline-SMOTE |
| heart | Outlier: detected, comparison table saved | PASS | 6 methods |
| heart | Outlier: applied method is modern and in the table | PASS | selected EBM; best traditional One-Class SVM |
| heart | Feature scale: detected, comparison table saved | PASS | 6 methods |
| heart | Feature scale: applied method is modern and in the table | PASS | selected PowerTransform (Yeo-Johnson); best traditional Min-Max scaler |
| heart | Demographic: detected, comparison table saved | PASS | 7 methods |
| heart | Demographic: applied method is modern and in the table | PASS | selected Adversarial debiasing; best traditional Optimal Transport |
| heart | Duplicate: not detected, nothing applied (before = after) | PASS | exact duplicate records (all columns except record ID) |
| heart | Data noise: detected, comparison table saved | PASS | 8 methods |
| heart | Data noise: applied method is modern and in the table | PASS | selected Robust loss (GCE); best traditional Denoising autoencoder |
| heart | Encoding / domain: not detected, nothing applied (before = after) | PASS | one-vs-rest domain-classifier AUC; AUROC drop on held-out domains |
| heart | modern-vs-traditional evidence table saved | PASS |  |
| heart | 10-record trace covers all 8 steps | PASS | 10 records, 8 steps |
| heart | dashboard 01_missing_data_bias_dashboard.html | PASS | 130 KB |
| heart | dashboard 02_class_imbalance_bias_dashboard.html | PASS | 133 KB |
| heart | dashboard 03_outlier_bias_dashboard.html | PASS | 128 KB |
| heart | dashboard 04_feature_scale_bias_dashboard.html | PASS | 132 KB |
| heart | dashboard 05_demographic_bias_dashboard.html | PASS | 155 KB |
| heart | dashboard 06_duplicate_bias_dashboard.html | PASS | 125 KB |
| heart | dashboard 07_data_noise_bias_dashboard.html | PASS | 130 KB |
| heart | dashboard 08_encoding_domain_bias_dashboard.html | PASS | 130 KB |
| heart | dashboard index.html | PASS | 4 KB |
| diabetes | raw data before preprocessing | PASS | 7,876 rows |
| diabetes | preprocessing notebook executed without errors | PASS | 38 code cells, 0 not executed, 0 with errors |
| diabetes | 10-sample notebook executed without errors | PASS | 11 code cells, 0 not executed, 0 with errors |
| diabetes | train: all predictor values finite | PASS | 9,112 rows x 32 predictors |
| diabetes | train: no predictor collapsed to a constant | PASS | every predictor varies |
| diabetes | train: binary label | PASS |  |
| diabetes | train: rows match the run log | PASS | 9,112 vs 9,112 |
| diabetes | validation: all predictor values finite | PASS | 1,179 rows x 32 predictors |
| diabetes | validation: binary label | PASS |  |
| diabetes | validation: rows match the run log | PASS | 1,179 vs 1,179 |
| diabetes | validation: no synthetic rows | PASS |  |
| diabetes | test: all predictor values finite | PASS | 1,179 rows x 32 predictors |
| diabetes | test: binary label | PASS |  |
| diabetes | test: rows match the run log | PASS | 1,179 vs 1,179 |
| diabetes | test: no synthetic rows | PASS |  |
| diabetes | train / validation / test records are disjoint | PASS |  |
| diabetes | same predictors in every split | PASS | 32 predictors |
| diabetes | Missing data: detected, comparison table saved | PASS | 8 methods |
| diabetes | Missing data: applied method is modern and in the table | PASS | selected MIP-VAE; best traditional MICE |
| diabetes | Class imbalance: detected, comparison table saved | PASS | 9 methods |
| diabetes | Class imbalance: applied method is modern and in the table | PASS | selected TabDDPM; best traditional Borderline-SMOTE |
| diabetes | Outlier: not detected, nothing applied (before = after) | PASS | rows beyond skew-adjusted extreme fences |
| diabetes | Feature scale: detected, comparison table saved | PASS | 6 methods |
| diabetes | Feature scale: applied method is modern and in the table | PASS | selected PowerTransform (Yeo-Johnson); best traditional Min-Max scaler |
| diabetes | Demographic: detected, comparison table saved | PASS | 7 methods |
| diabetes | Demographic: applied method is modern and in the table | PASS | selected Adversarial debiasing; best traditional Optimal Transport |
| diabetes | Duplicate: detected, comparison table saved | PASS | 6 methods |
| diabetes | Duplicate: applied method is modern and in the table | PASS | selected Ditto (transformer matcher); best traditional GBER (graph-based ER) |
| diabetes | Data noise: detected, comparison table saved | PASS | 8 methods |
| diabetes | Data noise: applied method is modern and in the table | PASS | selected Robust loss (GCE); best traditional Denoising autoencoder |
| diabetes | Encoding / domain: not detected, nothing applied (before = after) | PASS | no site/domain identifier in this dataset |
| diabetes | modern-vs-traditional evidence table saved | PASS |  |
| diabetes | 10-record trace covers all 8 steps | PASS | 10 records, 8 steps |
| diabetes | dashboard 01_missing_data_bias_dashboard.html | PASS | 75 KB |
| diabetes | dashboard 02_class_imbalance_bias_dashboard.html | PASS | 79 KB |
| diabetes | dashboard 03_outlier_bias_dashboard.html | PASS | 73 KB |
| diabetes | dashboard 04_feature_scale_bias_dashboard.html | PASS | 77 KB |
| diabetes | dashboard 05_demographic_bias_dashboard.html | PASS | 124 KB |
| diabetes | dashboard 06_duplicate_bias_dashboard.html | PASS | 75 KB |
| diabetes | dashboard 07_data_noise_bias_dashboard.html | PASS | 77 KB |
| diabetes | dashboard 08_encoding_domain_bias_dashboard.html | PASS | 74 KB |
| diabetes | dashboard index.html | PASS | 4 KB |
| hypertension | raw data before preprocessing | PASS | 29,471 rows |
| hypertension | preprocessing notebook executed without errors | PASS | 38 code cells, 0 not executed, 0 with errors |
| hypertension | 10-sample notebook executed without errors | PASS | 11 code cells, 0 not executed, 0 with errors |
| hypertension | train: all predictor values finite | PASS | 18,569 rows x 32 predictors |
| hypertension | train: no predictor collapsed to a constant | PASS | every predictor varies |
| hypertension | train: binary label | PASS |  |
| hypertension | train: rows match the run log | PASS | 18,569 vs 18,569 |
| hypertension | validation: all predictor values finite | PASS | 4,421 rows x 32 predictors |
| hypertension | validation: binary label | PASS |  |
| hypertension | validation: rows match the run log | PASS | 4,421 vs 4,421 |
| hypertension | validation: no synthetic rows | PASS |  |
| hypertension | test: all predictor values finite | PASS | 4,421 rows x 32 predictors |
| hypertension | test: binary label | PASS |  |
| hypertension | test: rows match the run log | PASS | 4,421 vs 4,421 |
| hypertension | test: no synthetic rows | PASS |  |
| hypertension | train / validation / test records are disjoint | PASS |  |
| hypertension | same predictors in every split | PASS | 32 predictors |
| hypertension | Missing data: detected, comparison table saved | PASS | 8 methods |
| hypertension | Missing data: applied method is modern and in the table | PASS | selected HyperImpute; best traditional KNN |
| hypertension | Class imbalance: not detected, nothing applied (before = after) | PASS | imbalance ratio (majority/minority) |
| hypertension | Outlier: not detected, nothing applied (before = after) | PASS | rows beyond skew-adjusted extreme fences |
| hypertension | Feature scale: detected, comparison table saved | PASS | 6 methods |
| hypertension | Feature scale: applied method is modern and in the table | PASS | selected PowerTransform (Yeo-Johnson); best traditional Robust scaler |
| hypertension | Demographic: detected, comparison table saved | PASS | 7 methods |
| hypertension | Demographic: applied method is modern and in the table | PASS | selected Counterfactual recourse (DiCE); best traditional Optimal Transport |
| hypertension | Duplicate: not detected, nothing applied (before = after) | PASS | exact duplicate records (all columns except record ID) |
| hypertension | Data noise: detected, comparison table saved | PASS | 8 methods |
| hypertension | Data noise: applied method is modern and in the table | PASS | selected Confident Learning (Cleanlab); best traditional Denoising autoencoder |
| hypertension | Encoding / domain: detected, comparison table saved | PASS | 7 methods |
| hypertension | Encoding / domain: applied method is modern and in the table | PASS | selected Domain-generalisation transformer; best traditional CORAL |
| hypertension | modern-vs-traditional evidence table saved | PASS |  |
| hypertension | 10-record trace covers all 8 steps | PASS | 10 records, 8 steps |
| hypertension | dashboard 01_missing_data_bias_dashboard.html | PASS | 149 KB |
| hypertension | dashboard 02_class_imbalance_bias_dashboard.html | PASS | 169 KB |
| hypertension | dashboard 03_outlier_bias_dashboard.html | PASS | 145 KB |
| hypertension | dashboard 04_feature_scale_bias_dashboard.html | PASS | 152 KB |
| hypertension | dashboard 05_demographic_bias_dashboard.html | PASS | 172 KB |
| hypertension | dashboard 06_duplicate_bias_dashboard.html | PASS | 145 KB |
| hypertension | dashboard 07_data_noise_bias_dashboard.html | PASS | 149 KB |
| hypertension | dashboard 08_encoding_domain_bias_dashboard.html | PASS | 163 KB |
| hypertension | dashboard index.html | PASS | 4 KB |
| respiratory | raw data before preprocessing | PASS | 28,842 rows |
| respiratory | preprocessing notebook executed without errors | PASS | 40 code cells, 0 not executed, 0 with errors |
| respiratory | 10-sample notebook executed without errors | PASS | 11 code cells, 0 not executed, 0 with errors |
| respiratory | train: all predictor values finite | PASS | 33,874 rows x 32 predictors |
| respiratory | train: no predictor collapsed to a constant | PASS | every predictor varies |
| respiratory | train: binary label | PASS |  |
| respiratory | train: rows match the run log | PASS | 33,874 vs 33,874 |
| respiratory | validation: all predictor values finite | PASS | 4,312 rows x 32 predictors |
| respiratory | validation: binary label | PASS |  |
| respiratory | validation: rows match the run log | PASS | 4,312 vs 4,312 |
| respiratory | validation: no synthetic rows | PASS |  |
| respiratory | test: all predictor values finite | PASS | 4,308 rows x 32 predictors |
| respiratory | test: binary label | PASS |  |
| respiratory | test: rows match the run log | PASS | 4,308 vs 4,308 |
| respiratory | test: no synthetic rows | PASS |  |
| respiratory | train / validation / test records are disjoint | PASS |  |
| respiratory | same predictors in every split | PASS | 32 predictors |
| respiratory | Missing data: detected, comparison table saved | PASS | 8 methods |
| respiratory | Missing data: applied method is modern and in the table | PASS | selected MIP-VAE; best traditional Random Forest |
| respiratory | Class imbalance: detected, comparison table saved | PASS | 9 methods |
| respiratory | Class imbalance: applied method is modern and in the table | PASS | selected TabDDPM; best traditional SMOTE |
| respiratory | Outlier: detected, comparison table saved | PASS | 6 methods |
| respiratory | Outlier: applied method is modern and in the table | PASS | selected ECOD; best traditional LOF |
| respiratory | Feature scale: detected, comparison table saved | PASS | 6 methods |
| respiratory | Feature scale: applied method is modern and in the table | PASS | selected SNN (SELU); best traditional Quantile transform |
| respiratory | Demographic: detected, comparison table saved | PASS | 7 methods |
| respiratory | Demographic: applied method is modern and in the table | PASS | selected Fair contrastive learning; best traditional Reweighting |
| respiratory | Duplicate: not detected, nothing applied (before = after) | PASS | exact duplicate records (all columns except record ID) |
| respiratory | Data noise: detected, comparison table saved | PASS | 8 methods |
| respiratory | Data noise: applied method is modern and in the table | PASS | selected Robust loss (GCE); best traditional labels: Denoising autoencoder; signals: — |
| respiratory | Encoding / domain: detected, comparison table saved | PASS | 7 methods |
| respiratory | Encoding / domain: applied method is modern and in the table | PASS | selected Domain-generalisation transformer; best traditional CORAL |
| respiratory | modern-vs-traditional evidence table saved | PASS |  |
| respiratory | 10-record trace covers all 8 steps | PASS | 10 records, 8 steps |
| respiratory | dashboard 01_missing_data_bias_dashboard.html | PASS | 159 KB |
| respiratory | dashboard 02_class_imbalance_bias_dashboard.html | PASS | 172 KB |
| respiratory | dashboard 03_outlier_bias_dashboard.html | PASS | 159 KB |
| respiratory | dashboard 04_feature_scale_bias_dashboard.html | PASS | 165 KB |
| respiratory | dashboard 05_demographic_bias_dashboard.html | PASS | 166 KB |
| respiratory | dashboard 06_duplicate_bias_dashboard.html | PASS | 155 KB |
| respiratory | dashboard 07_data_noise_bias_dashboard.html | PASS | 160 KB |
| respiratory | dashboard 08_encoding_domain_bias_dashboard.html | PASS | 173 KB |
| respiratory | dashboard index.html | PASS | 5 KB |
| respiratory | dashboard 07b_monitor_signal_noise_dashboard.html | PASS | 129 KB |
| hospital | raw cohort tables (eICU + MIMIC, 4 conditions) | PASS | 8 files |
| hospital | train: rows and finite predictors | PASS | 80,606 rows x 25 predictors |
| hospital | train: source | PASS | eicu |
| hospital | train: race labels corrected (eICU mostly White, Asian small) | PASS | White 77.4%, Asian 1.8% |
| hospital | validation: rows and finite predictors | PASS | 20,318 rows x 25 predictors |
| hospital | validation: source | PASS | eicu |
| hospital | validation: race labels corrected (eICU mostly White, Asian small) | PASS | White 72.4%, Asian 1.8% |
| hospital | test: rows and finite predictors | PASS | 8,454 rows x 25 predictors |
| hospital | test: source | PASS | eicu |
| hospital | test: race labels corrected (eICU mostly White, Asian small) | PASS | White 75.6%, Asian 0.8% |
| hospital | external: rows and finite predictors | PASS | 73,990 rows x 25 predictors |
| hospital | external: source | PASS | mimic |
| hospital | training hospitals disjoint from validation/test hospitals | PASS |  |
| hospital | race correction audit saved | PASS |  |
| latent | latent notebook executed without errors | PASS | 5 code cells, 0 not executed, 0 with errors |
| latent | five views: four diseases + cross-hospital ICU cohort | PASS | heart, diabetes, hypertension, respiratory, hospital |
| latent | all latent methods compared on all metrics | PASS |  |
| latent | selected method = best mean rank | PASS | selected SAINT |
| latent | combined train: rows = all preprocessed rows | PASS | 694,111 vs 694,111 |
| latent | combined train: 16 finite latent features | PASS |  |
| latent | heart train: per-dataset latent file | PASS | 551,950 rows |
| latent | diabetes train: per-dataset latent file | PASS | 9,112 rows |
| latent | hypertension train: per-dataset latent file | PASS | 18,569 rows |
| latent | respiratory train: per-dataset latent file | PASS | 33,874 rows |
| latent | hospital train: per-dataset latent file | PASS | 80,606 rows |
| latent | combined validation: rows = all preprocessed rows | PASS | 96,247 vs 96,247 |
| latent | combined validation: 16 finite latent features | PASS |  |
| latent | heart validation: per-dataset latent file | PASS | 66,017 rows |
| latent | diabetes validation: per-dataset latent file | PASS | 1,179 rows |
| latent | hypertension validation: per-dataset latent file | PASS | 4,421 rows |
| latent | respiratory validation: per-dataset latent file | PASS | 4,312 rows |
| latent | hospital validation: per-dataset latent file | PASS | 20,318 rows |
| latent | combined test: rows = all preprocessed rows | PASS | 84,379 vs 84,379 |
| latent | combined test: 16 finite latent features | PASS |  |
| latent | heart test: per-dataset latent file | PASS | 66,017 rows |
| latent | diabetes test: per-dataset latent file | PASS | 1,179 rows |
| latent | hypertension test: per-dataset latent file | PASS | 4,421 rows |
| latent | respiratory test: per-dataset latent file | PASS | 4,308 rows |
| latent | hospital test: per-dataset latent file | PASS | 8,454 rows |
| latent | MIMIC external hospital encoded (not in the combined files) | PASS | 73,990 rows |
| latent | website scatter of the same records in every method | PASS |  |
| latent | dashboard 01_cka_alignment_dashboard.html | PASS |  |
| latent | dashboard 02_macro_f1_preservation_dashboard.html | PASS |  |
| latent | dashboard 03_silhouette_structure_dashboard.html | PASS |  |
| latent | dashboard 04_nmi_ari_domain_invariance_dashboard.html | PASS |  |
| latent | dashboard index.html | PASS |  |
| in:heart | heart_inprocessing.ipynb executed without errors | PASS | 10 code cells, 0 not executed, 0 with errors |
| in:heart | heart_inprocessing_sample10.ipynb executed without errors | PASS | 5 code cells, 0 not executed, 0 with errors |
| in:heart | test split and MIMIC never used for selection | PASS |  |
| in:heart | demographic: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:heart | demographic: selected = best mean rank (validation) | PASS | selected FairCare |
| in:heart | demographic: test metrics saved | PASS |  |
| in:heart | intersectional: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:heart | intersectional: selected = best mean rank (validation) | PASS | selected GroupDRO (intersectional) |
| in:heart | intersectional: test metrics saved | PASS |  |
| in:heart | calibration: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:heart | calibration: selected = best mean rank (validation) | PASS | selected CRL |
| in:heart | calibration: test metrics saved | PASS |  |
| in:heart | environment: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:heart | environment: selected = best mean rank (validation) | PASS | selected Fishr (tied with V-REx at 2.00; first in the method list wins) |
| in:heart | environment: test metrics saved | PASS |  |
| in:heart | cross-hospital predictions validation | PASS | 20,318 rows |
| in:heart | cross-hospital predictions test | PASS | 8,454 rows |
| in:heart | cross-hospital predictions external | PASS | 73,990 rows |
| in:heart | dashboard 01_demographic_prediction_dependence_dashboard.html | PASS | 30 KB |
| in:heart | dashboard 02_intersectional_subgroup_unfairness_dashboard.html | PASS | 49 KB |
| in:heart | dashboard 03_predictive_miscalibration_dashboard.html | PASS | 32 KB |
| in:heart | dashboard 04_cross_environment_generalization_dashboard.html | PASS | 31 KB |
| in:heart | dashboard index.html | PASS | 1 KB |
| in:diabetes | diabetes_inprocessing.ipynb executed without errors | PASS | 10 code cells, 0 not executed, 0 with errors |
| in:diabetes | diabetes_inprocessing_sample10.ipynb executed without errors | PASS | 5 code cells, 0 not executed, 0 with errors |
| in:diabetes | test split and MIMIC never used for selection | PASS |  |
| in:diabetes | demographic: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:diabetes | demographic: selected = best mean rank (validation) | PASS | selected DANN (tied with GroupDRO at 2.00; first in the method list wins) |
| in:diabetes | demographic: test metrics saved | PASS |  |
| in:diabetes | intersectional: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:diabetes | intersectional: selected = best mean rank (validation) | PASS | selected Intersectional debiasing |
| in:diabetes | intersectional: test metrics saved | PASS |  |
| in:diabetes | calibration: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:diabetes | calibration: selected = best mean rank (validation) | PASS | selected Uncertainty-weighted gradient |
| in:diabetes | calibration: test metrics saved | PASS |  |
| in:diabetes | environment: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:diabetes | environment: selected = best mean rank (validation) | PASS | selected V-REx |
| in:diabetes | environment: test metrics saved | PASS |  |
| in:diabetes | cross-hospital predictions validation | PASS | 20,318 rows |
| in:diabetes | cross-hospital predictions test | PASS | 8,454 rows |
| in:diabetes | cross-hospital predictions external | PASS | 73,990 rows |
| in:diabetes | dashboard 01_demographic_prediction_dependence_dashboard.html | PASS | 29 KB |
| in:diabetes | dashboard 02_intersectional_subgroup_unfairness_dashboard.html | PASS | 33 KB |
| in:diabetes | dashboard 03_predictive_miscalibration_dashboard.html | PASS | 30 KB |
| in:diabetes | dashboard 04_cross_environment_generalization_dashboard.html | PASS | 31 KB |
| in:diabetes | dashboard index.html | PASS | 1 KB |
| in:hypertension | hypertension_inprocessing.ipynb executed without errors | PASS | 10 code cells, 0 not executed, 0 with errors |
| in:hypertension | hypertension_inprocessing_sample10.ipynb executed without errors | PASS | 5 code cells, 0 not executed, 0 with errors |
| in:hypertension | test split and MIMIC never used for selection | PASS |  |
| in:hypertension | demographic: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:hypertension | demographic: selected = best mean rank (validation) | PASS | selected GroupDRO |
| in:hypertension | demographic: test metrics saved | PASS |  |
| in:hypertension | intersectional: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:hypertension | intersectional: selected = best mean rank (validation) | PASS | selected GroupDRO (intersectional) |
| in:hypertension | intersectional: test metrics saved | PASS |  |
| in:hypertension | calibration: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:hypertension | calibration: selected = best mean rank (validation) | PASS | selected Calibration-regularized |
| in:hypertension | calibration: test metrics saved | PASS |  |
| in:hypertension | environment: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:hypertension | environment: selected = best mean rank (validation) | PASS | selected V-REx (tied with EQRM at 1.75; first in the method list wins) |
| in:hypertension | environment: test metrics saved | PASS |  |
| in:hypertension | cross-hospital predictions validation | PASS | 20,318 rows |
| in:hypertension | cross-hospital predictions test | PASS | 8,454 rows |
| in:hypertension | cross-hospital predictions external | PASS | 73,990 rows |
| in:hypertension | dashboard 01_demographic_prediction_dependence_dashboard.html | PASS | 30 KB |
| in:hypertension | dashboard 02_intersectional_subgroup_unfairness_dashboard.html | PASS | 42 KB |
| in:hypertension | dashboard 03_predictive_miscalibration_dashboard.html | PASS | 31 KB |
| in:hypertension | dashboard 04_cross_environment_generalization_dashboard.html | PASS | 31 KB |
| in:hypertension | dashboard index.html | PASS | 1 KB |
| in:respiratory | respiratory_inprocessing.ipynb executed without errors | PASS | 10 code cells, 0 not executed, 0 with errors |
| in:respiratory | respiratory_inprocessing_sample10.ipynb executed without errors | PASS | 5 code cells, 0 not executed, 0 with errors |
| in:respiratory | test split and MIMIC never used for selection | PASS |  |
| in:respiratory | demographic: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:respiratory | demographic: selected = best mean rank (validation) | PASS | selected GroupDRO |
| in:respiratory | demographic: test metrics saved | PASS |  |
| in:respiratory | intersectional: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:respiratory | intersectional: selected = best mean rank (validation) | PASS | selected Fairness-constrained optimization |
| in:respiratory | intersectional: test metrics saved | PASS |  |
| in:respiratory | calibration: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:respiratory | calibration: selected = best mean rank (validation) | PASS | selected Uncertainty-weighted gradient |
| in:respiratory | calibration: test metrics saved | PASS |  |
| in:respiratory | environment: all 4 methods + ERM compared on its 4 metrics | PASS | 5 rows |
| in:respiratory | environment: selected = best mean rank (validation) | PASS | selected Fishr (tied with V-REx at 1.75; first in the method list wins) |
| in:respiratory | environment: test metrics saved | PASS |  |
| in:respiratory | cross-hospital predictions validation | PASS | 20,318 rows |
| in:respiratory | cross-hospital predictions test | PASS | 8,454 rows |
| in:respiratory | cross-hospital predictions external | PASS | 73,990 rows |
| in:respiratory | dashboard 01_demographic_prediction_dependence_dashboard.html | PASS | 30 KB |
| in:respiratory | dashboard 02_intersectional_subgroup_unfairness_dashboard.html | PASS | 35 KB |
| in:respiratory | dashboard 03_predictive_miscalibration_dashboard.html | PASS | 31 KB |
| in:respiratory | dashboard 04_cross_environment_generalization_dashboard.html | PASS | 31 KB |
| in:respiratory | dashboard index.html | PASS | 1 KB |
| website | page exists | PASS | 231 KB |
| website | no 'still running' placeholders left | PASS |  |
| website | explorer has the three stages | PASS | Pre-processing, Latent space, In-processing |
| website | pre-processing before/after visual has every dataset and bias | PASS | 4 datasets |
| website | plot explorer links all resolve | PASS | 53 dashboards |
| website | heart dashboards = the project files (website orange theme, nothing else changed) | PASS |  |
| website | diabetes dashboards = the project files (website orange theme, nothing else changed) | PASS |  |
| website | hypertension dashboards = the project files (website orange theme, nothing else changed) | PASS |  |
| website | respiratory dashboards = the project files (website orange theme, nothing else changed) | PASS |  |
| website | latent dashboards published | PASS |  |
| website | latent results shown | PASS |  |
| website | latent merge animation has data | PASS |  |
| website | heart in-processing dashboards published | PASS |  |
| website | heart in-processing results shown | PASS |  |
| website | diabetes in-processing dashboards published | PASS |  |
| website | diabetes in-processing results shown | PASS |  |
| website | hypertension in-processing dashboards published | PASS |  |
| website | hypertension in-processing results shown | PASS |  |
| website | respiratory in-processing dashboards published | PASS |  |
| website | respiratory in-processing results shown | PASS |  |
