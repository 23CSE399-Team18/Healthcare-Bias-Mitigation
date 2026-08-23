# Approved respiratory dataset: full eICU-CRD v2.0

Official source: https://physionet.org/content/eicu-crd/2.0/

This is the full database, not the demo. Its patient table contains 200,859 ICU
stays from 208 hospitals. **Use `official_verified/*.csv.gz` as the raw sources.**
Thirty compressed originals were copied from your existing download, checked
against the supplied SHA256 manifest, and decompressed to validate gzip CRC.
An authenticated comparison with PhysioNet's live manifest remains separate.

## Cleanup completed

At your request, the old top-level uncompressed CSV copies were deleted. Two were
truncated (`vitalPeriodic.csv`, `nurseCharting.csv`); their complete compressed
sources remain. Twenty-eight other CSV copies matched the retained compressed
originals. The unused, unverified `admissionDrug.csv` was also removed.

Old prepared datasets and extraction caches were deleted. Historical notebooks,
figures, tables and fitted-model caches are in
`D:/complete_old_archive_20261111_212716/complete/dataset_audit/legacy_results_not_for_training/respiratory`.
They are reference work, not the approved training datasets.

Drive D uses FAT32. The two large complete CSVs would be 7.94 GB and 11.47 GB,
above its single-file limit. Read compressed files directly; do not expand them.

```python
from eicu_sources import get_eicu_table_path
import pandas as pd

patient = pd.read_csv(get_eicu_table_path("patient"))
for batch in pd.read_csv(get_eicu_table_path("vitalPeriodic"), chunksize=100_000):
    pass  # Filter and aggregate each batch.
```

`eicu_sources.py` checks hashes and refuses incomplete/unapproved sources.
`official_verified/canonical_file_manifest.csv` records source links and hashes.

## Corrected respiratory cohort

This is a research-derived cohort, not an official prepackaged respiratory
dataset. **24-hour timing fix (7 November 2026):** the official
[admissionDx documentation](https://eicu.mit.edu/eicutables/admissiondx/) permits
APACHE admission diagnoses to be entered during the first 36 hours. The APACHE
route now requires `admissionDx.admitdxenteredoffset <= 1440` for that diagnosis;
284 stays that qualified only through a later-entered diagnosis were removed.

Cohort membership: `dataset_verified/respiratory_cohort_24h_landmark.csv`:

- 28,842 ICU stays, 24,091 patients, 200 hospitals.
- 3,931 subsequent hospital deaths and 24,911 survivors.
- Known adult age; patient still in ICU and hospital beyond 24 hours.
- Respiratory APACHE admission diagnosis entered by 24 h, or primary pulmonary
  diagnosis recorded by 24 h.
- Target `hospital_mortality`: subsequent in-hospital death, not ICU mortality.

**Training file:** `D:/complete/model_ready/respiratory_model_ready.csv` (112
first-24 h predictors), built by `extract_respiratory_features.py`. Feature rules
and missing rates: `D:/complete/model_ready/respiratory_feature_audit.json`.

The dictionary and flow JSON are beside the manifest. Reproduce with:

```powershell
python D:\complete\resp_physionet\resp\prepare_respiratory_cohort.py
```

Repeated patients are retained; later splits must group by `uniquepid`. A
held-out-hospital test must separate `hospitalid`. Age `>89` is top-coded; the
value90 in the grouping field is not an exact age. The new cohort fixes the old
four-hour eligibility/24-hour feature mismatch and unbounded diagnosis-time rule.

## Additional MIMIC–eICU study

The current additional study is documented in
[four_conditions/README.md](../../cross_dataset/four_conditions/README.md).
MIMIC downloads and preprocessing are complete for four separate recorded-diagnosis
classification tasks, including respiratory-system disease. The original
respiratory mortality dataset above remains separate. MIMIC discharge coding does
not establish equivalent by-landmark respiratory diagnosis availability, so the
new respiratory classification task must not be substituted for this mortality target.

Harmonised early vitals and source-training-only numerical transformations are
prepared in both directions. Predictive models and four-condition cross-hospital
evaluations have not been run. The older general ICU mortality plan is retained
as historical code; it is not a completed four-condition study.
