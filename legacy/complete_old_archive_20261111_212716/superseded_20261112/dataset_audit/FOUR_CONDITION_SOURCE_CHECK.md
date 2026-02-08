# Source verification — MIMIC-IV and eICU

**Historical source audit from 7 November 2026.** Downloads, source inspection and
four-condition preprocessing have since completed; see
[current status](../cross_dataset/four_conditions/README.md) and
[updated source review](MIMIC_FOUR_PHENOTYPE_REVIEW.md).

This was a source audit only. No clinical dataset, cohort, model or processing
script was changed for this check.

## What the official resources provide

[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and
[full eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) are official
relational clinical databases. They provide source tables joined by patient,
admission and ICU-stay identifiers. Neither resource is supplied as four
ready-made, matched Heart/Respiratory/Hypertension/Diabetes prediction CSVs.
Constructing a disease-specific research cohort is a separate task whose label,
population, prediction time and outcome must be stated.

## Condition support and required tables

| Condition | Primary-source evidence and scope | MIMIC tables needed | eICU tables relevant |
|---|---|---|---|
| Heart | Official MIT-LCP Charlson code defines myocardial infarction and congestive heart failure from ICD codes. These are specific conditions; neither is automatically equivalent to BRFSS's heart-attack-or-coronary-disease label. | `patients`, `admissions`, `diagnoses_icd`, `d_icd_diagnoses`; `icustays` if restricting to ICU | `patient`, `diagnosis`, `admissionDx`, and potentially `pastHistory` |
| Respiratory | Official Charlson code contains chronic pulmonary disease. eICU diagnosis documentation includes pulmonary/COPD and respiratory-failure concepts. Chronic lung disease, acute respiratory failure, ARDS and an ICU respiratory-mortality cohort are different tasks. | Diagnosis/code tables for a coded phenotype; `chartevents`, `d_items`, possibly `labevents`, `d_labitems`, and ventilation-related tables for a physiologic/support-based definition | `diagnosis`, `admissionDx`; `respiratoryCare`, `respiratoryCharting`, `lab` and vitals as required by the chosen phenotype |
| Diabetes | Official Charlson code distinguishes diabetes with and without chronic complications. eICU `apachePredVar` documents a diabetes indicator. Neither alone establishes future-onset or type-2-only diabetes. | Diagnosis/code tables; `labevents` and `d_labitems` if a separately justified glycemic definition is required | `diagnosis`, `pastHistory`, `apachePredVar`; `lab` if required |
| Hypertension | Diagnosis infrastructure can support coded/documented hypertension, but the actual MIMIC diagnosis tables are not downloaded here and a matched hypertension phenotype has not been verified. Charlson code is not a hypertension phenotype. An elevated ICU BP is not by itself an established chronic-hypertension label. | `diagnoses_icd` plus `d_icd_diagnoses`, with explicit code set and population; demographics/admissions as above | `diagnosis` and `pastHistory`, with explicit clinical-string/code mapping and timing |

Evidence: [official MIT-LCP Charlson definition](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/comorbidity/charlson.sql),
[eICU diagnosis documentation](https://eicu.mit.edu/eicutables/diagnosis/),
[eICU past-history documentation](https://eicu.mit.edu/eicutables/pasthistory/),
and [eICU APACHE predictor documentation](https://eicu.mit.edu/eicutables/apachepredvar/).

## Important differences in diagnosis meaning and timing

MIMIC's [diagnoses_icd documentation](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html)
says these are hospital-billing diagnoses assigned at hospital discharge. The
table has no diagnosis-availability timestamp. Its
[dictionary](https://mimic.mit.edu/docs/iv/modules/hosp/d_icd_diagnoses.html)
joins on both code and ICD version. A discharge-coded condition cannot simply
be assumed known at ICU admission or at a 24-hour prediction landmark.

eICU's [active-diagnosis documentation](https://eicu.mit.edu/eicutables/diagnosis/)
describes diagnoses entered by clinical staff during the ICU stay and includes
the entry offset. The documentation warns that these may differ from billing
diagnoses and do not reliably identify resolution or diagnoses ruled out.
The [past-history documentation](https://eicu.mit.edu/eicutables/pasthistory/)
also warns that detailed history is uncommon. Missing documentation is not
proof that a patient does not have the disease.

**Additional timing limitation found:** the official
[eICU admissionDx documentation](https://eicu.mit.edu/eicutables/admissiondx/)
says APACHE admission diagnoses can be entered during the first **36 hours**.
`admitDxEnteredOffset` records when they were entered. The previously generated
respiratory manifest's `patient.apacheadmissiondx` inclusion route was not
checked against that timestamp. Its APACHE route therefore must not yet be
described as verified available by 24 hours. This audit flags the limitation;
it does not alter the manifest or silently change its definition.

## What is actually downloaded here

The canonical local MIMIC folder currently contains publisher-checksum-verified
`patients`, `admissions`, `icustays`, and `d_items`. The large `chartevents`
download is pending. The following are **not downloaded** in that folder:

- `hosp/diagnoses_icd.csv.gz`
- `hosp/d_icd_diagnoses.csv.gz`
- `hosp/labevents.csv.gz` and `hosp/d_labitems.csv.gz`
- `icu/procedureevents.csv.gz` and other additional treatment/ventilation tables

The local eICU canonical package already includes `diagnosis`, `admissionDx`,
`pastHistory`, `apachePredVar`, clinical labs, respiratory tables and vitals.
Their presence does not validate a matched four-condition label mapping.

The current MIMIC metadata manifest is an **all-adult ICU mortality cohort**.
It is not evidence that four disease-specific MIMIC counterparts have been
downloaded or harmonized. The prepared mortality code should remain separate
from decisions about the four project applications.

## What can be confirmed for the project

The existing CDC sources support separate survey applications using their own
documented diagnosis-history questions. They are population surveys and their
public files do not supply hospital identities for hospital-held-out testing.
Official sources: [BRFSS](https://www.cdc.gov/brfss/annual_data/annual_2022.html),
[NHIS](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html), and
[NHANES](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm).
They must not be presented as hospital datasets or as already matched to the
MIMIC/eICU diagnosis definitions.

MIMIC–eICU can support additional external validation once the **same clinical
task, cohort and outcome** are defined in both sources. There is not yet a basis
to confirm that all four disease-specific counterparts are matched and ready.
