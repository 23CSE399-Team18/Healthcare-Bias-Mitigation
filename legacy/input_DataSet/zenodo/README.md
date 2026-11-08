# Multi-Disease Healthcare Bias Mitigation Dataset

Tabular data for **three diseases: diabetes, heart disease and hypertension**, prepared between 2024 and 2026 as inputs for detecting and mitigating data bias in healthcare machine learning. All three files are processed versions of real, publicly available data; none of the values are synthetic or imputed. Each file keeps the problems a bias-mitigation pipeline has to deal with: missing values, duplicate or repeated records, implausible values, class imbalance and demographic skew. Released as the data accompanying the project's per-disease bias-mitigation pipelines and its multi-disease latent feature alignment.

- **Rows:** 613,987 in total (diabetes 101,766; heart 440,111; hypertension 72,110)
- **Unique patients / respondents:** diabetes 71,518; heart 440,111; hypertension 105
- **Prepared:** 2024-2026
- **Version:** 1.0.0

## Files

| File | What it is |
|---|---|
| `diabetes.csv` | Hospital encounters of patients with diabetes, one row per encounter. The encounters are from 1999-2008. |
| `heart.csv` | Adults surveyed about their health in 2022, one row per respondent, with heart attack or coronary heart disease as the target. All values are self-reported or derived from self-reported answers. |
| `hypertension.csv` | 24-hour Holter ECG recordings of hypertensive patients, one row per 2-minute ECG window, with heart-rate-variability and signal-quality features, the patient's clinical data, and a vascular event in the following 12 months as the target. |
| `SHA256SUMS.txt` | SHA-256 checksums of the three CSV files. |

All files are UTF-8, comma-separated, with a header row. Empty cells are missing values.

## Columns of `diabetes.csv`

| Column | Meaning | Unit |
|---|---|---|
| encounter_id | Encounter identifier (unique per row) | - |
| patient_nbr | Patient identifier; some patients have several encounters | - |
| race | Caucasian, AfricanAmerican, Hispanic, Asian, Other; empty if unknown | - |
| gender | Female, Male, Unknown/Invalid | - |
| age | 10-year age band, e.g. `[70-80)` | years |
| age_mid | Midpoint of the age band (5, 15, ... 95) | years |
| age_group | `<60`, `60-79` or `80+` | - |
| admission_type_id, admission_type | Admission type code and its label (Emergency, Urgent, Elective, ...) | - |
| discharge_disposition_id, discharge_disposition | Discharge disposition code and its label (e.g. Discharged to home) | - |
| admission_source_id, admission_source | Admission source code and its label (e.g. Emergency Room) | - |
| time_in_hospital | Length of stay, 1-14 | days |
| medical_specialty | Specialty of the admitting physician; empty if unknown | - |
| num_lab_procedures, num_procedures, num_medications | Lab tests, procedures and distinct medications during the encounter | count |
| number_outpatient, number_emergency, number_inpatient | Outpatient, emergency and inpatient visits in the year before the encounter | count |
| diag_1, diag_2, diag_3 | Primary, secondary and additional diagnosis | ICD-9 code |
| number_diagnoses | Diagnoses recorded for the encounter | count |
| max_glu_serum | Glucose serum test: `None` (not measured), `Norm`, `>200`, `>300` | mg/dL band |
| A1Cresult | HbA1c test: `None` (not measured), `Norm`, `>7`, `>8` | % band |
| metformin ... metformin-pioglitazone (23 columns) | Each diabetes medication: `No` (not prescribed), `Steady`, `Up`, `Down` | - |
| change | `Ch` if diabetes medication was changed, otherwise `No` | - |
| diabetesMed | `Yes` if any diabetes medication was prescribed | - |
| readmitted | `NO`, `>30` or `<30` days to readmission | - |
| readmitted_30d | 1 if `readmitted` is `<30`, otherwise 0 | - |

In the admission label columns, `NULL`, `Not Available` and `Not Mapped` are labels carried over from the source's ID mapping and mean the value is unknown.

## Columns of `heart.csv`

Original survey variable names are given in brackets.

| Column | Meaning | Unit |
|---|---|---|
| RespondentID | State code + survey sequence number (unique per row) | - |
| State | US state or territory FIPS code (`_STATE`); 54 states and territories | - |
| SurveyMode | `Landline` or `Cellphone` questionnaire | - |
| InterviewLanguage | `English` or `Spanish` | - |
| PartialInterview | 1 = interview only partly completed | - |
| SurveyWeight | Final survey weight (`_LLCPWT`); makes the sample represent the US adult population | - |
| Sex | 1 = male, 0 = female | - |
| AgeCategory | 1 = 18-24, 2 = 25-29, ... 12 = 75-79, 13 = 80+ (`_AGEG5YR`) | 5-year band |
| RaceEthnicity | 1 = White, 2 = Black, 3 = Other race, 4 = Multiracial (all non-Hispanic), 5 = Hispanic | - |
| Education | 1 = no high-school diploma, 2 = high-school graduate, 3 = some college, 4 = college graduate | - |
| Income | 1 = <15k, 2 = 15-25k, 3 = 25-35k, 4 = 35-50k, 5 = 50-100k, 6 = 100-200k, 7 = 200k+ | USD per year |
| GeneralHealth | 1 = excellent ... 5 = poor | - |
| PhysicalActivities, HadStroke, HadAsthma, HadSkinCancer, HadOtherCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, AlcoholDrinkers, HIVTesting, FluVaxLast12, PneumoVaxEver, HighRiskLastYear, HealthInsurance | 1 = yes, 0 = no | - |
| PhysicalHealthDays, MentalHealthDays | Days of poor physical / mental health in the past 30 | days |
| SleepHours | Average sleep per 24 hours | hours |
| LastCheckupTime | 1 = <1 year, 2 = 1-2 years, 3 = 2-5 years, 4 = 5+ years, 5 = never | - |
| RemovedTeeth | 0 = none, 1 = 1-5, 2 = 6 or more but not all, 3 = all | - |
| SmokerStatus | 0 = never, 1 = former, 2 = current some days, 3 = current every day | - |
| ECigaretteUsage | 0 = never, 1 = former, 2 = some days, 3 = every day | - |
| HadDiabetes | 1 = yes, 2 = yes but only during pregnancy, 3 = no, 4 = no, pre-diabetes or borderline | - |
| TetanusLast10Tdap | 1 = yes, Tdap; 2 = yes, tetanus but not Tdap; 3 = yes, type unknown; 4 = no | - |
| CovidPos | 1 = yes, 2 = no, 3 = positive on a home test only | - |
| HeightInMeters | Height | m |
| WeightInKilograms | Weight | kg |
| BMI | WeightInKilograms / HeightInMeters² | kg/m² |
| HeartDisease | **Target.** 1 = ever told they had a heart attack or coronary heart disease (`_MICHD`), 0 = no | - |

## Columns of `hypertension.csv`

Each row is one 2-minute window of one patient's recording. The patient's clinical columns and target are repeated on every window of that patient.

| Column | Meaning | Unit |
|---|---|---|
| record_id | Patient / recording number (5 digits) | - |
| window_id | `record_id` + window number (unique per row) | - |
| window_index | Position of the window in the recording, from 0 | - |
| window_start_s | Start of the window, from the start of the recording | s |
| hour_of_day | Clock hour at the start of the window, 0-23 | h |
| adc_gain | Recorder gain | ADC units per mV |
| gender | `M` or `F` (one patient is coded `m`) | - |
| age, age_group | Age; age band `55-64`, `65-74` or `75+` | years |
| weight_kg, height_cm | Weight, height | kg, cm |
| bsa_m2, bmi | Body surface area, body-mass index | m², kg/m² |
| smoker | `yes` or `no` | - |
| sbp_mmhg, dbp_mmhg | Systolic and diastolic blood pressure | mmHg |
| imt_max_mm | Maximum carotid intima-media thickness | mm |
| lvmi_g_m2 | Left-ventricular mass index | g/m² |
| ef_pct | Ejection fraction | % |
| n_beats | Beats detected in the window | count |
| rr_coverage | Share of the window spanned by the detected beats | 0-1 |
| mean_rr_ms, median_rr_ms, min_rr_ms, max_rr_ms, iqr_rr_ms | Statistics of the intervals between beats (RR intervals) | ms |
| sdnn_ms, rmssd_ms, sdsd_ms | Standard deviation of RR, root mean square and standard deviation of successive RR differences | ms |
| cvnn | sdnn / mean RR | ratio |
| pnn50_pct, pnn20_pct | Share of successive RR differences above 50 ms / 20 ms | % |
| mean_hr_bpm, std_hr_bpm, min_hr_bpm, max_hr_bpm | Heart rate computed from each RR interval | beats/min |
| sd1_ms, sd2_ms, sd1_sd2 | Poincaré plot short- and long-term variability, and their ratio | ms, ratio |
| lf_ms2, hf_ms2, total_power_ms2 | RR power at 0.04-0.15 Hz (LF), 0.15-0.40 Hz (HF) and 0.04-0.40 Hz | ms² |
| lf_hf, lf_nu, hf_nu | LF/HF ratio; LF and HF as a share of LF + HF | ratio, % |
| frac_rr_implausible | Share of RR intervals outside 300-2000 ms | 0-1 |
| frac_rr_jump | Share of successive RR intervals that change by more than 20% | 0-1 |
| sig_std_mv_III, sig_std_mv_V3, sig_std_mv_V5 | Signal standard deviation for each ECG lead | mV |
| flat_frac_III, flat_frac_V3, flat_frac_V5 | Share of 1-second segments in which the lead is flat | 0-1 |
| clip_frac_III, clip_frac_V3, clip_frac_V5 | Share of samples at the recorder's limits (clipping) | 0-1 |
| vascular_event | Event in the 12 months after the recording: `none`, `myocardial infarction`, `stroke` or `syncope` | - |
| event_12m | **Target.** 1 = any vascular event in those 12 months, 0 = none | - |

## How the data were produced

**diabetes.csv**

1. Started from the 101,766 encounters of the *Diabetes 130-US Hospitals for Years 1999-2008* dataset. No rows were added or removed.
2. Dropped `weight` and `payer_code`, which are mostly empty in the source.
3. Replaced `?` placeholders with empty cells.
4. Added the text labels `admission_type`, `discharge_disposition` and `admission_source` from the source's ID mapping.
5. Added `age_mid`, `age_group` and the binary target `readmitted_30d`.

**heart.csv**

1. Started from the 445,132 respondents and 326 variables of the 2022 Behavioral Risk Factor Surveillance System (BRFSS) annual survey file.
2. Kept 48 variables: survey design fields, demographics, health conditions, behaviours, body measures and the heart-disease indicator.
3. Removed the 5,021 respondents with no heart-disease answer, leaving 440,111. Labels were never filled in.
4. Converted survey codes to values: "don't know" and "refused" codes became empty cells; "none" (88) became 0 days; heights and weights coded in feet/inches, pounds, centimetres or kilograms were converted to metres and kilograms. The converted heights and weights agree with the survey's own calculated values on 100% of the rows where both exist.
5. Recoded `LastCheckupTime`, `RemovedTeeth`, `SmokerStatus` and `ECigaretteUsage` so that higher codes mean more (longer ago, more teeth, more smoking), and computed `BMI` from height and weight.

No values were imputed, no outliers were changed and no rows were added: every value is the respondent's own answer or a direct conversion of it.

**hypertension.csv**

1. Started from the SHAREE database: 24-hour Holter ECG recordings (3 leads: III, V3, V5; 128 Hz; 8-bit) of 139 hypertensive patients, the beat positions found by an automatic QRS detector (WQRS, not manually corrected), and each patient's clinical data and vascular events in the following 12 months.
2. Split each recording into non-overlapping 2-minute windows; a final partial window was dropped.
3. For each window, computed heart-rate-variability features from the detector's beat positions **without correcting them**, and signal-quality features from the raw ECG.
4. Left time-domain features empty when a window has fewer than 5 beats, and frequency-domain features empty when it has fewer than 30 beats or spans less than 60 s.
5. Attached each patient's clinical data unchanged (`n/a` became an empty cell) and added `age_group` and `event_12m`.

This file covers 105 of the 139 SHAREE patients: records 01911 to 02299. The remaining 34 recordings (02304 to 02425) are not included. Nothing was cleaned, filtered, imputed or re-encoded.

## Important limitations (please read)

- **`heart.csv` is self-reported survey data, not clinical records.** Heart disease, other conditions, height and weight are what respondents said; none were measured or confirmed by a clinician.
  - 4.66% of cells are missing, and 51.5% of rows have at least one missing value. `Income` is missing for 21.4% of respondents; 90,670 interviews were only partly completed.
  - Some self-reported body measures are impossible: heights from 0.61 m to 7.52 m (9 above 2.4 m, 35 below 1.0 m), weights from 14.5 kg to 352 kg, and BMI from 0.96 to 233 (131 below 12, 116 above 80).
  - 34 respondents have identical answers in every column except `RespondentID` and `SurveyWeight`.
  - The sample is not proportional to the population: for example, Hispanic respondents are under-represented and those aged 65+ over-represented. Use `SurveyWeight` for population-level estimates.
- **`hypertension.csv` has only 105 patients, repeated over many rows.**
  - Each patient contributes 480-720 windows, all with the same clinical data and label. Split training and test sets by `record_id`, never by row, or the same patient will appear in both.
  - Only 11 patients (10.5%; 10.4% of windows) had a vascular event: 8 myocardial infarctions, 2 strokes and 1 syncope.
  - Beat positions come from an automatic detector and were not corrected, so many windows contain detector errors. 31,517 windows (44%) contain at least one RR interval implying more than 250 beats/min, 447 have a mean RR interval above 2 s, and 334 have fewer than 5 detected beats.
  - Clinical values are missing for some patients: blood pressure for 1, IMT for 8, LV mass index and ejection fraction for 9. 15.3% of rows have at least one missing value.
  - `gender` uses `M`, `F` and, for one patient, `m`.
- **The outcomes are imbalanced:**

| File | Target | Distribution |
|---|---|---|
| diabetes.csv | readmitted | NO 54,864; >30 35,545; <30 11,357 (readmitted_30d = 11.2% positive) |
| heart.csv | HeartDisease | 1: 39,751 (9.03%); 0: 400,360 |
| hypertension.csv | event_12m (patients) | 1: 11 (10.5%); 0: 94 |
| hypertension.csv | event_12m (windows) | 1: 7,495 (10.4%); 0: 64,615 |

- `diabetes.csv` reflects US hospital practice in 1999-2008 and the coding conventions of its source (ICD-9 codes, 10-year age bands). 2,273 `race` values and 49,949 `medical_specialty` values are empty.
- These data are for research and teaching. They are not intended for clinical decisions or for drawing medical conclusions.

## Respiratory data (not included)

The project also used a respiratory ICU cohort taken from the eICU Collaborative Research Database v2.0. That database is available only to credentialed users under the PhysioNet Credentialed Health Data Use Agreement, which does not allow sharing it with anyone else, so neither the data nor anything derived from it is included here. Approved users can obtain it at https://doi.org/10.13026/C2WM1R.

## Licence and attribution

The processed files and documentation in this record are released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

The underlying data keep their own terms: `diabetes.csv` is derived from a dataset published under CC BY 4.0; `heart.csv` from publicly available US government survey data; `hypertension.csv` from a PhysioNet database released under the Open Data Commons Attribution License v1.0. Anyone using these files should also cite the original sources.

Please attribute:
- Diabetes encounters: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5230J.
- Heart survey responses: Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System Survey Data. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2022. https://www.cdc.gov/brfss/annual_data/annual_2022.html
- Hypertension recordings: Smart Health for Assessing the Risk of Events via ECG Database (SHAREE), PhysioNet, DOI 10.13026/C2RC7C; and Melillo, P., Izzo, R., Orrico, A., Scala, P., Attanasio, M., Mirra, M., De Luca, N., & Pecchia, L. (2015). Automatic prediction of cardiovascular and cerebrovascular events using heart rate variability analysis. PLOS ONE, 10(3), e0118504. DOI 10.1371/journal.pone.0118504.

## Citation

RatanRaja, E., Datta, K., Srija, U., Sriya, U. (2026). Multi-Disease Healthcare Bias Mitigation Dataset (Version 1.0.0) [Data set]. Zenodo. DOI: [DOI]

## Authors

E RatanRaja, Datta K, Srija U, Sriya U

Department of Computer Science and Engineering, Amrita School of Computing, Coimbatore, Amrita Vishwa Vidyapeetham, India. Supervisors: Dr. Bagavathi Sivakumar P and Dr. Deepika T.
