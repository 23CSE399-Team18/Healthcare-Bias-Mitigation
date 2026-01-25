# Download MIMIC-IV using the signed-in browser

Use the Edge or Chrome session signed into the **PhysioNet account approved for
MIMIC-IV**. This can be a different email from this application. The browser
handles sign-in; no password or browser cookies are read by the import tool.

## Disease-profile downloads in Microsoft Edge

The current four-condition download uses the authenticated Microsoft Edge session.
Along with the five original study tables below, the disease profile includes
these official hospital tables:

- https://physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz
- https://physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz
- https://physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz
- https://physionet.org/files/mimiciv/3.1/hosp/omr.csv.gz
- https://physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz

Keep Edge open and the computer awake until its transfers finish. Downloads go
to `C:\Users\ganap\Downloads`, then the local watcher verifies the completed
sources and imports them into FAT32-safe segments on D. This watcher imports
data only; it does not run the separate mortality experiment.

Status: `disease_browser_import_job.json`; log: `disease_browser_import_job.log`.
The watcher waits up to 48 hours and checks filenames without repeatedly hashing
unfinished files. To restart it locally after checking no other watcher is active:

```powershell
python D:\complete\cross_dataset\watch_disease_browser_downloads.py --download-dir C:\Users\ganap\Downloads
```

Create `cross_dataset/STOP_DISEASE_BROWSER_IMPORT` to stop the watcher before
import starts. This is separate from `STOP_BROWSER_STUDY`, which controls the
older mortality-study runner. Remove the disease stop marker before restarting
the disease watcher. Browser transfers are managed in Edge's Downloads panel.

## Original five-file study subset

1. Open https://physionet.org/content/mimiciv/3.1/ and confirm the heading is
   **MIMIC-IV, version3.1**. Use its **Files** section. This is the full database,
   not the demo and not a separate waveform project.
2. Download `SHA256SUMS.txt` and the five files below. Save them in
   `C:\Users\ganap\Downloads` on the NTFS C drive. That drive supports the large
   `chartevents.csv.gz`; the FAT32 D drive cannot hold that single compressed file
   if its size exceeds4GB. Do not choose D as the browser download destination.
3. Wait for the browser download to finish. A `.crdownload` file is incomplete and is
   never imported. Keep the filenames or normal browser duplicate suffixes such
   as `patients.csv (1).gz`. The current large-file download uses Microsoft Edge.
4. Run the importer. It verifies the downloaded bytes against the publisher
   checksum file before copying. It leaves the original C-drive files untouched.

All six links belong to the official MIMIC-IV v3.1 download location:

| File | Direct official download |
|---|---|
| Publisher checksum manifest | https://physionet.org/files/mimiciv/3.1/SHA256SUMS.txt |
| Patients | https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz |
| Admissions | https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz |
| ICU stays | https://physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz |
| ICU item dictionary | https://physionet.org/files/mimiciv/3.1/icu/d_items.csv.gz |
| ICU charted clinical events | https://physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz |

If a link shows a sign-in or access-denied page, sign in through the official
project page using the approved account. Required training and the project's
data-use agreement must already be complete. A PhysioNet account alone does not
grant access. Do not rename a saved error webpage to a dataset or checksum file.

## Import without a terminal password

From a local PowerShell terminal, or through the agent once the files are ready:

```powershell
python D:\complete\cross_dataset\import_browser_downloads.py --download-dir C:\Users\ganap\Downloads
```

For another download directory, change `--download-dir`. The directory can hold
the browser files flat, or use `hosp/` and `icu/` subfolders. Only the named
datasets and `SHA256SUMS.txt` are considered; unrelated Downloads files and browser
profile data are not read.

The importer can import already completed files while the large clinical file
is still downloading. `browser_import_state.json` then says `waiting_for_files`
and lists missing files. Rerun after Chrome finishes: completed imports are
reverified and reused. `complete` means all five study files are present and
match the checksum manifest. Terminal-download status remains separate in
`download_state.json`; the browser importer does not overwrite it.

## Automatic completion

`finish_browser_study.py` waits for the named browser download to finish, then
verifies/imports the source, extracts the real MIMIC 24-hour features and runs
the two study directions. It reads no browser credentials and makes no network
requests. The agent starts this local runner after code/cohort checks finish.

Progress is recorded in `completion_job.json`; processing output is in
`completion_job.log`. Results are saved under `results/` only after the real
MIMIC and eICU cohorts exist. A waiting status is not a completed study result.
The wait has a 12-hour limit. If needed, rerun locally:

```powershell
python D:\complete\cross_dataset\finish_browser_study.py --download-dir C:\Users\ganap\Downloads
```

To stop a waiting runner before its next processing step, create an empty file
named `STOP_BROWSER_STUDY` inside `cross_dataset`. Remove that marker before
intentionally starting the runner again. Existing source files are not deleted.

## Integrity and FAT32 format

Each completed source is checked against `SHA256SUMS.txt`. Normal duplicate-name
suffixes are recognized, and only a file with the expected full SHA256 is chosen.
Files that change during hashing or copying are rejected. Conflicting checksum
manifests or an existing inconsistent/partial destination are preserved and
reported instead of overwritten.

After verification, the compressed bytes are copied to D in segments of at most
1GB. The full copied bytes are checked again before the destination is marked
complete. For example, the logical path
`raw/mimiciv/3.1/icu/chartevents.csv.gz` is represented by
`chartevents.csv.gz.parts/manifest.json` and numbered `part-00000.bin` files.
The study uses `streaming_sources.py` to read these compressed segments as one
stream. No oversized joined gzip or uncompressed CSV is created on D.

The checksum comparison uses the publisher manifest you downloaded from the
official HTTPS link. It does not infer access approval, authenticate the browser
session, or accept project agreements. No data is uploaded by this importer.
