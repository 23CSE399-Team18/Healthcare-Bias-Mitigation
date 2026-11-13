# Heart: retain CDC BRFSS 2022

Your original `LLCP2022.XPT` is a BRFSS survey file, not UCI Heart Disease.
It contains 445,132 respondents and 328 variables in your archived release.
Its bytes match the XPT member of your stored ZIP, and that member passes ZIP CRC.
The ZIP member's filename has a trailing space; the audit handles that explicitly.

Official source: https://www.cdc.gov/brfss/annual_data/annual_2022.html

Official download: https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip

Official codebook: https://www.cdc.gov/brfss/annual_data/2022/zip/codebook22_llcp-v2-508.zip

The current CDC www download server returned HTTP403 in this environment. Your
existing source was retained and validated locally. We have not claimed a
byte-for-byte comparison against a newly downloaded current release. Current
website variable counts differ from your archived schema; retain the hashes in
`dataset_verified/validation.json` to identify the exact version used.

## Correct task definition

Use **self-reported history of myocardial infarction or coronary heart
disease/angina**, using CDC's `_MICHD` indicator. This is not every heart disease
and does not establish future disease onset. It does not provide the ECG and
measured serum-cholesterol features described for UCI on your PDF's dataset slide.

`dataset_verified/heart_2022_baseline_raw.csv` was rebuilt directly from the
validated XPT, retaining original missingness and survey codes:

- 440,111 respondents with known labels: 39,751 positive and 400,360 negative.
- `_MICHD=1` maps to `HeartDisease=1`; `_MICHD=2` maps to0.
- 5,021 unknown labels are excluded and preserved in `heart_label_exclusions.csv`.
- `_MICHD`, `CVDINFR4` and `CVDCRHD4` are omitted from predictors because they
  define the outcome.

Read `dataset_verified/data_dictionary.csv` before modelling. `_STATE`, `SEQNO`,
`_PSU`, `_STSTR` and `_LLCPWT` are identifiers or sample-design metadata. Retain
age, sex and race/ethnicity for subgroup audits; their use as model inputs is a
separate experimental choice. Decode unknown/refused/skip codes by variable,
rather than treating all numeric codes as continuous measurements. `_BMI5` is
BMI multiplied by100 and `_AGE80=80` means80 or older. Survey PSUs are not hospitals.

The superseded `LLCP2022_raw.csv` and old processed datasets were deleted at your
request. The correct original XPT/ZIP and new baseline remain. Historical figures,
tables and model caches are archived at `D:/complete_old_archive_20261111_212716/complete/dataset_audit/legacy_results_not_for_training/heart`.
Their old method implementations and fairness claims have not been validated.
Use the new baseline for a fresh, documented comparison.

Reproduce from `D:\complete`:

```powershell
python .\dataset_audit\audit_heart.py
```

No imputation, balancing, scaling, learned representation or model training was
performed for the new dataset view.
