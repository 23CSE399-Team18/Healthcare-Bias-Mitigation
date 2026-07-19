# Hypertension: replace the BRFSS 2024 task source

`LLCP2024.XPT` is a real CDC BRFSS source with 457,670 respondents and 301
variables in the supplied release. It has no `BPHIGH*`, `BPMEDS*` or hypertension
outcome variable. It cannot supply a valid diagnosed-hypertension label for
this project. No label was inferred from heart disease, diabetes or other proxies.
The unsuitable file was deleted at your request. The cleanup audit records it.

Use **`nhis2023/derived/hypertension_nhis2023_raw_cohort.csv`** instead.
The official CDC National Health Interview Survey 2023 adult source is downloaded
and documented in `nhis2023/DATASET_README.md`.

The corrected binary cohort has 29,471 adults: 11,094 reported clinician-diagnosed
hypertension and 18,377 reported no diagnosis. The target is `HYPEV_A` (1 yes,
2 no); 51 unknown/refused labels are excluded and audited. This is a
cross-sectional diagnosis task, not measured blood pressure or future onset.

Official release: https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html

Official raw CSV ZIP: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip

NHIS provides a distinct survey source alongside BRFSS heart, NHANES diabetes,
and eICU respiratory. Read the dictionary and explicit predictor list: sample
weights, strata and PSUs are metadata and are not hospital identifiers.

## BRFSS alternative, if your guide specifically wants BRFSS

BRFSS2023 has `BPHIGH6` and can also support a hypertension diagnosis task:

- Source: https://www.cdc.gov/brfss/annual_data/annual_2023.html
- Raw ZIP: https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip
- Codebook: https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip

Those www.cdc.gov downloads returned HTTP403 from this environment, so the
BRFSS2023 data is **not downloaded**. The failed attempt is recorded in
`dataset_audit/public_download_manifest.json`; the empty download directory was removed. The
ready-to-use correction is NHIS2023, not a relabelled BRFSS2024 file.
