# Dataset access and publication

| Cohort | Source | Repository contents |
|---|---|---|
| Heart | [BRFSS 2022](https://www.cdc.gov/brfss/annual_data/annual_2022.html) | Public survey files and preprocessing exports |
| Diabetes | [NHANES August 2021–August 2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023) | Public questionnaire, examination and laboratory component files and preprocessing exports |
| Hypertension | [NHIS 2023](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) | Public-use sample adult files and preprocessing exports |
| Respiratory | [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) | Acquisition/extraction code, documentation and aggregate reports; credentialed records remain local |
| Cross-hospital | eICU-CRD plus [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) | Cohort code and aggregate reports; credentialed records and derived models remain local |

Four disease tasks use these sources; the cross-hospital cohort is a fifth representation-learning view. Obtain controlled ICU data directly through PhysioNet credentialing, training and the required data use agreement. Place authorized records in the locations documented in the cohort READMEs, configure source paths and execute the pipeline locally. Do not upload those records to this public repository.

Public survey data retain their original source attribution and conditions; the repository's code license does not supersede dataset conditions. Lossless compressed public files are listed in PUBLICATION_MANIFEST_20261114.json and can be restored with code/publication/restore_public_data.cjs.
