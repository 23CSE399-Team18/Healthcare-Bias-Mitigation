# Datasets used in this project

There are **five source datasets** and **six active analysis branches**, producing **12 prediction tasks**. MIMIC/eICU diagnosis tables are research-derived, not four official prepackaged disease datasets. The original respiratory patient-split baseline is retained as a reference; hospital in-processing uses the rebuilt hospital split.

## Source datasets

| Application | Official source/version | Local model-ready rows | Outcome | Local data |
|---|---|---:|---|---|
| Heart | [CDC BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html) | 440,111 | Self-reported heart attack / coronary heart disease / angina | [Model-ready](model_ready/heart_model_ready.csv) / [Raw source](Heart_correct/LLCP2022.XPT) |
| Respiratory | [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) | 28,842 | Subsequent in-hospital mortality after the 24-hour landmark in the respiratory cohort | [Model-ready](model_ready/respiratory_model_ready.csv) / [Raw source](resp_physionet/resp/official_verified) |
| Diabetes | [CDC NHANES August 2021-August 2023](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023) | 7,876 | Self-reported diagnosed diabetes | [Model-ready](model_ready/diabetes_model_ready.csv) / [Raw source](diabetes_correct/raw) |
| Hypertension | [CDC NHIS 2023](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) | 29,471 | Self-reported diagnosed hypertension | [Model-ready](model_ready/hypertension_model_ready.csv) / [Raw source](hypertension_correct/nhis2023) |
| MIMIC | [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) | 73,990 | Four overlapping recorded diagnoses: ischemic heart disease, respiratory-system disease, diabetes, systemic hypertension | [Model-ready](cross_dataset/four_conditions/mimic) / [Raw source](cross_dataset/raw) |

The additional eICU diagnosis comparison cohort contains **109,378 ICU stays, 88,955 patients and 206 hospitals**, from the same eICU source used for respiratory mortality. [Its four target tables](cross_dataset/four_conditions/eicu) and [exact phenotype definitions](cross_dataset/four_conditions/phenotype_definitions.json) are retained. MIMIC contains 73,990 selected stays from 53,931 patients. These are project cohort counts, not the full source database sizes.

## Active preprocessing -> latent -> in-processing branches

| Branch | Targets | Train / validation / test rows | External rows | Preprocessed / latent width |
|---|---|---|---:|---|
| heart | heart | 308,077 / 66,017 / 66,017 | 0 | 36 / 8 |
| diabetes | diabetes | 5,513 / 1,181 / 1,182 | 0 | 14 / 8 |
| hypertension | hypertension | 20,629 / 4,421 / 4,421 | 0 | 69 / 8 |
| respiratory_hospital | respiratory | 21,082 / 4,909 / 2,301 | 0 | 277 / 178 |
| eicu_hospital | heart, respiratory, diabetes, hypertension | 80,606 / 20,318 / 8,454 | 73,990 | 25 / 16 |
| mimic_transfer | heart, respiratory, diabetes, hypertension | 52,229 / 10,829 / 10,932 | 109,378 | 24 / 16 |

Respiratory hospital preprocessing excludes 550 stays from 212 patients who cross partitions, leaving 28,292 stays. The three respiratory hospital partitions contain 137 / 34 / 29 hospitals. The eICU diagnosis partitions contain 141 / 34 / 31 hospitals. Patients are disjoint across each internal partition. MIMIC internal splitting is by patient within one institution.

## Complete files

[DATASET_FILES.csv](DATASET_FILES.csv) lists every complete preprocessing, latent and final prediction CSV. [DATASETS.json](DATASETS.json) provides the machine-readable source/branch catalogue. Individual dashboards link the exact split files. Keep train, validation, test and external files separate.

| Stage | Code | Outputs | Individual dashboards |
|---|---|---|---|
| preprocessing | [preprocessing/code](preprocessing/code/README.md) | [preprocessing/outputs](preprocessing/outputs/README.md) | [preprocessing/plots](preprocessing/plots/index.html) |
| latent | [latent/code](latent/code/README.md) | [latent/outputs](latent/outputs/README.md) | [latent/plots](latent/plots/index.html) |
| inprocessing | [inprocessing/code](inprocessing/code/README.md) | [inprocessing/outputs](inprocessing/outputs/README.md) | [inprocessing/plots](inprocessing/plots/index.html) |

## Meaning and limits

- Survey heart/diabetes/hypertension labels are self-report, not hospital diagnoses. These surveys cannot establish hospital generalization. Survey weights/design variables are preserved, but the current experiments are unweighted.
- Respiratory mortality and respiratory recorded diagnosis are different targets. The ICU diagnosis labels overlap. Zero means no matching recorded diagnosis among documentation-eligible encounters, not confirmed absence of disease. MIMIC discharge ICD coding differs from eICU diagnosis/history documentation.
- Diagnosis experiments are retrospective recorded-label classification, not future disease prediction. Shared ICU baseline features are first-24-hour HR/RR/SpO2 summaries plus age group and sex, with training-fitted imputation/encoding.
- All source and external transforms are fitted only on source training rows. No target, ID, split or `sens_*` metadata is a latent predictor. Age/sex also occur as original predictor features, as documented in the protocols.
- Cross-hospital evidence comes from held-out eICU hospitals. Cross-database institutional non-overlap cannot be proven from anonymous eICU hospital IDs. Results are single-seed, bounded-budget experiments with mixed utility/fairness effects and explicit method adaptations.
