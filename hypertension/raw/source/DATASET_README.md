# Hypertension: CDC NHIS 2023 sample adult

This is the usable replacement for the BRFSS 2024 file, which lacked a hypertension diagnosis label and was deleted at your request. BRFSS 2023 is also a suitable survey alternative, but CDC's BRFSS download host returned HTTP403 in this session; this package was actually downloaded from CDC's official FTP HTTPS service.

## Source and files

- Official release page: https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html
- Original CSV archive: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip
- Official codebook: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/adult-codebook.pdf
- Survey description: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/srvydesc-508.pdf
- Raw files are under `raw/`; downloaded codebook, summary and survey description are under `documentation/`.
- `download_manifest.json` records the exact source URLs, byte sizes and SHA256 of each downloaded file. ZIP CRC integrity was checked. These hashes record local downloaded content; CDC did not publish an independent hash in this workflow.

## Task and cohort

The label is self-reported **ever clinician-diagnosed hypertension**, from `HYPEV_A`. It is not measured blood pressure, a prospective outcome, or an estimate of undiagnosed hypertension. Code 1 maps to `hypertension=1`; code 2 maps to 0. Codes 7 refused, 8 not ascertained, 9 unknown, and missing are excluded, not converted to negative labels. The codebook tells interviewers to code taking medication to control high blood pressure as yes, so antihypertensive treatment would leak the outcome.

The official sample-adult file contains 29,522 adults. The derived cohort contains 29,471: 11,094 positive and 18,377 negative; 51 refused/unknown labels were excluded. The count and label frequencies match the official June 24, 2024 codebook. All sample-adult records are eligible adults; 59 retained records have an unknown exact age. They remain in the cohort with missing age and cannot be assigned a precise age subgroup.

`derived/hypertension_nhis2023_raw_cohort.csv` contains the labelled, unimputed cohort. `derived/excluded_labels.csv` preserves excluded IDs, source label codes and reasons. `cohort_summary.json`, `feature_schema.json`, and `derived/data_dictionary.csv` record the rules and counts.

## Predictors and metadata

The conservative baseline candidate predictors are age `AGEP_A`, sex `SEX_A`, education `EDUCP_A`, BMI category `BMICAT_A`, and cigarette-smoking category `SMKCIGST_A`. The original 647-variable CSV remains available for a separately justified feature expansion. Do not automatically use every numeric column as a feature.

Age, sex and `HISPALLP_A` support demographic and intersectional auditing. Race/ethnicity is a social classification and the public recode includes CDC-imputed race/ethnicity where unknown. Keep that source imputation visible in fairness reporting. Age is top-coded at 85, meaning 85+. BMI is the published category computed by CDC from its internal height/weight values; do not imply it is a continuous clinical BMI measurement. Smoking category 5 is a smoker with unknown current status and is retained as its own category; code 9 becomes missing.

`HHX` is an identifier. `WTFA_A`, `PSTRAT`, and `PPSU` are survey weight, stratum and PSU metadata. `REGION` and `URBRRL` are domain metadata. These fields are excluded from the baseline predictor list. Use the final annual adult weight and the complex survey design for population estimates and uncertainty, and consider weighted fairness summaries alongside unweighted model metrics. NHIS weights should not be substituted with BRFSS or NHANES weights.

`HYPEV_A` is retained for label traceability and must be excluded from predictors. Hypertension follow-up and treatment fields (`HYPDIF_A`, `HYP12M_A`, `HYPMED_A`) and blood-pressure testing field `BPLAST_A` are not included in the baseline cohort. The full original CSV should not be modelled without an explicit leakage review. The temporal sequence of general predictors and diagnosis is not established by this cross-sectional survey.

Only documented missing/refused/unknown codes were mapped to blank cells. No imputation, synthetic samples, scaling, fairness mitigation, or train/test split was applied. Split before learning any transformation. This dataset supports a hypertension application of the framework; its region codes do not support a cross-hospital evaluation. It is not interchangeable with eICU mortality labels.

## Reproduce

From `D:\complete`, run:

```powershell
python hypertension_correct\nhis2023\download_nhis2023.py
python hypertension_correct\nhis2023\prepare_hypertension.py
```

The preparation script uses pandas, preferring the local reader installed under `D:\complete\_data_reader`. The download script uses only the Python standard library and official CDC HTTPS URLs.
