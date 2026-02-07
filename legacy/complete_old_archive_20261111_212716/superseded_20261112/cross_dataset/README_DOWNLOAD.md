# MIMIC-IV official download

## Four-condition source tables

MIMIC-IV includes hospital diagnosis records and clinical measurements for heart
disease, diabetes, respiratory diseases and hypertension. It is a relational
database, not four ready-labelled files matching the project's survey datasets.
Each disease cohort and target still needs an explicit definition. Diagnoses are
coded at discharge; they must not be treated as predictors known at ICU admission.

To download the relevant structured source tables with the approved PhysioNet
account, run locally (enter the password only in that terminal):

```powershell
powershell -NoProfile -File D:\complete\cross_dataset\launch_download.ps1 -DiseaseData
```

The `--disease-data` profile downloads ten tables: the original `patients`,
`admissions`, `icustays`, `d_items`, and `chartevents`, plus `diagnoses_icd`,
`d_icd_diagnoses`, `d_labitems`, `omr`, and `labevents` from `hosp/`.
The smaller tables precede the large chart and laboratory event files. Existing
completed sources are reverified and reused. Compressed sources are stored in
1 GB segments under `cross_dataset/raw/mimiciv/3.1/`; no oversized single file is
written to FAT32. This profile is a clinical-data subset of MIMIC-IV, not the
entire database or the separate imaging, notes or waveform projects.

For browser downloads of these ten tables, use the matching importer profile:

```powershell
python cross_dataset\import_browser_downloads.py --disease-data --download-dir C:\Users\ganap\Downloads
```

The existing five-file mortality-study defaults remain available without
`--disease-data`. Downloading this profile does not run a mortality experiment or
change the verified preprocessing matrices.

Official documentation:
[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/),
[hospital tables](https://mimic.mit.edu/docs/iv/modules/hosp/), and
[diagnosis timing](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html).

## Browser and terminal access

The required downloads are already present; see `mimic_download_integrity.json`.
Historical browser-study instructions are in the external archive recorded in
`../dataset_audit/archive_manifest.json`. The instructions below describe terminal
retrieval if needed; it does not inherit a browser login. A terminal username is the
PhysioNet account username shown on the website, not the account email address.

Use the PhysioNet account that is approved for **MIMIC-IV**, even if it uses a
different email from this application. The downloader asks for that account's
PhysioNet username and hidden password in a local terminal. Do not send a password
through chat. Credentials are kept only in the downloader process's memory;
they are not saved in scripts, command arguments, manifests or status files.

Official project: https://physionet.org/content/mimiciv/3.1/

Approval, required training and the project's data-use agreement must already
be complete. The script cannot grant approval or accept agreements. HTTP401/403
means authorization was not established; check the official project while signed
into the approved account. An existing PhysioNet account is not proof of access
to this particular project.

## Run locally

From a local PowerShell terminal:

```powershell
python -u D:\complete\cross_dataset\download_mimic.py
```

The accompanying `launch_download.ps1` wraps that command for a separate local
window. The script downloads the four smaller files first, followed by the large
clinical events file. No further yes/no approval is requested. The data remains
on this computer; no dataset is uploaded. The Python standard library is enough
for downloading and streaming, so no dependency installation is needed.

Files requested, all under the official HTTPS download location:

- https://physionet.org/files/mimiciv/3.1/SHA256SUMS.txt
- https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz
- https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz
- https://physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz
- https://physionet.org/files/mimiciv/3.1/icu/d_items.csv.gz
- https://physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz

Use `--metadata-only` only when intentionally downloading the smaller files first;
the full study still requires `chartevents`. `download_state.json` records progress
without account identifiers or credentials. It is created when you run the script.

## FAT32 storage and integrity

Drive D uses FAT32, which cannot hold one file of4GB or more. Sources are stored
as compressed byte segments of at most1GB, including small files for a consistent
format. For example, the logical path
`raw/mimiciv/3.1/icu/chartevents.csv.gz` is represented by
`chartevents.csv.gz.parts/manifest.json` and numbered `part-00000.bin` files.
No complete uncompressed CSV or oversized joined gzip is written to disk.

Each full compressed source must match the publisher's SHA256 manifest before
it is marked complete. Each segment also has its own checksum. A failed download
can resume from complete1GB segments, after their checksums are rechecked and only
if the server honors the exact HTTP byte-range offset. An incomplete last segment
is discarded and downloaded again. Changed or unregistered segments cause a clear
failure instead of being overwritten. A failed full-file checksum is never marked
as usable. Do not manually edit or concatenate the segment files.

## Read a downloaded source

```python
from pathlib import Path
from streaming_sources import open_csv_text, read_official_csv

logical = Path('raw/mimiciv/3.1/icu/chartevents.csv.gz')
with open_csv_text(logical) as stream:
    # csv.DictReader(stream), or pandas.read_csv(stream, chunksize=...)
    pass

for chunk in read_official_csv(logical, chunksize=200_000):
    # Filter and aggregate each chunk. Do not load every event into RAM.
    pass
```

`read_official_csv` needs pandas; `open_csv_text` does not. Both support regular
`.csv.gz` files and the segmented format. Default reads validate the completed
manifest, publisher-hash agreement and present segment sizes. `verify=True`
additionally rehashes all stored compressed bytes before reading, if an independent
post-download integrity check is desired.

No real authenticated download is performed by creating these tools. A local
interactive sign-in and successful completed manifests are required before a
MIMIC dataset or cross-dataset result can be claimed.
