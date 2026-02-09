# Replacement text for the dataset slide (PDF page 18)

Remove the old entries: UCI Heart Disease, CKD, the eICU **demo** link, and NHANES
listed with "demographic fairness validation" as its target. Use the slide below.

## Data Collection: four applications, four official sources

| Application | Dataset (official link) | Records | Predictors | Target (binary) | Positive | Fairness groups | Domain for domain-bias / generalization |
|---|---|---:|---:|---|---:|---|---|
| Heart | [CDC BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html) | 440,111 | 16 | Ever had heart attack or coronary heart disease/angina | 9.0% | Sex, age group, race/ethnicity | US state |
| Hypertension | [CDC NHIS 2023 Sample Adult](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) | 29,471 | 32 | Ever told by a clinician they have hypertension | 37.6% | Sex, age group, race/ethnicity | US census region |
| Diabetes | [CDC NHANES Aug 2021–Aug 2023](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023) | 7,876 | 7 | Ever told by a doctor they have diabetes | 13.6% | Sex, age group, race/ethnicity | none (national exam survey) |
| Respiratory | [PhysioNet eICU-CRD v2.0 (full database)](https://physionet.org/content/eicu-crd/2.0/) | 28,842 ICU stays | 112 | In-hospital death after the first 24 h in ICU | 13.6% | Sex, age group, race/ethnicity | 200 hospitals |

Short slide wording (one box per dataset):

1. **Heart: CDC BRFSS 2022.** Parameters: age, sex, BMI, smoking, alcohol, exercise, sleep, general/physical/mental health, diabetes status, education, income, cost barrier, check-up recency. Target: heart attack or coronary heart disease (binary). Bias studied: sex, age, race/ethnicity.
2. **Hypertension: CDC NHIS 2023.** Parameters: age, sex, BMI category, height, weight, smoking, e-cigarette use, general health, 13 other diagnosed conditions, depression/anxiety screens, income, insurance, food security, usual care, marital status, nativity, employment, urbanicity. Target: diagnosed hypertension (binary). Bias studied: sex, age, race/ethnicity.
3. **Diabetes: CDC NHANES 2021–2023.** Parameters: age, sex, education, income-to-poverty ratio, measured BMI, measured waist, smoking. Target: diagnosed diabetes (binary). Bias studied: sex, age, race/ethnicity.
4. **Respiratory: eICU-CRD v2.0.** Adult respiratory ICU patients still in ICU at 24 h. Parameters (first 24 h): vitals, 26 laboratory tests, P/F ratio, GCS, ventilation, urine output, comorbidities, admission source and diagnosis. Target: in-hospital mortality (binary). Bias studied: sex, age, race/ethnicity; plus held-out hospitals.

## Wording that must be accurate

- Heart, Hypertension and Diabetes classify **self-reported diagnosis history** from
  population surveys. They are not hospital datasets and do not predict future onset.
- Respiratory is an **ICU mortality prognosis** task, not a respiratory-diagnosis task.
- Only Respiratory has hospital identifiers. Cross-hospital methods (Fishr, V-REx,
  EQRM, Fish, DANN, CORAL, IRM) run on eICU across its 200 hospitals. On the surveys
  they can only use state (BRFSS) or region (NHIS) as the domain, which must be called
  **cross-region** generalization, not cross-hospital. The extra MIMIC-IV ↔ eICU
  study in `cross_dataset/` is a separate experiment.

## Latent feature extraction slide (page 21): CKA correction

Linear CKA compares two representations of the **same samples**. The four datasets
contain different people, so "CKA between datasets" is undefined as written. Use one
of these instead and say which:

- CKA between the shared latent and each dataset's own encoder **on the same
  records** (how much the shared space keeps of each dataset's structure), or
- a distribution distance between datasets in the shared latent space
  (MMD or Wasserstein) for cross-dataset alignment.

The other latent metrics (Macro-F1, Silhouette, NMI/ARI against dataset identity) are
fine. A shared space needs shared inputs: age, sex, race/ethnicity and BMI (or BMI
category) are present in all four model-ready files; smoking is present in the three
surveys.

## Objective wording to keep honest

These are retrospective, de-identified secondary datasets. They support bias
auditing and mitigation experiments. They do not by themselves show equitable new
data collection (Objective 1), deployed continuous fairness monitoring (Objective 3)
or a formal privacy guarantee (Objective 4). Describe those as parts to be designed
and simulated, not as properties of the data.
