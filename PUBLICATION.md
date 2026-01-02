# Published project snapshot

Snapshot date: 14 November 2026 (Asia/Calcutta). The import preserves existing upstream history and adds one commit per changed file. New author and committer timestamps are assigned across 1 January–14 November 2026 at the repository owner's request. These are import timestamps, **not evidence of when research or development occurred**. The actual publication occurs on the snapshot date.

## Included

- Source code and notebooks for preprocessing, latent alignment and in-processing.
- Public-use CDC BRFSS 2022, NHANES August 2021–August 2023 and NHIS 2023 survey files, plus their local preprocessing exports and source documentation.
- Aggregate result tables, individual bias dashboards, method comparison figures and project documentation.
- The current Team-18 website in website/, with reviewed dashboard copies.

Files above the regular Git file limit are stored as lossless gzip files. Run 'node code/publication/restore_public_data.cjs' to restore them; this checks SHA-256 and never overwrites an existing file. No Git LFS subscription is required for these compressed files.

## Kept local

Credentialed eICU/MIMIC source records, patient-level derivatives, combined latent rows, trained model binaries and record-level prediction exports are excluded. Source access links and extraction code are included. MIMIC-derived datasets and models require the same controlled access as the source: [MIMIC-IV guidance](https://physionet.org/content/mimiciv/3.1/). The [eICU license](https://physionet.org/content/eicu-crd/2.0/) also restricts sharing access.

Executed notebook outputs and record-level appendix tables are removed from publication copies; executable cells are preserved. Public respiratory dashboards omit patient examples, individual outlier values and point-level Q-Q plots. Public latent dashboards omit individual projections. Aggregate metrics are unchanged. Original local artifacts remain intact.

Python environments, caches, private tool settings and nested Git repositories are excluded. Figure drafts containing restricted examples are excluded. The publication manifest records file-level inclusion and exclusions; this is a reproducible project publication, not an unrestricted backup of the entire drive.

## Validation scope

Publication checks cover hashes, compression round trips, notebook code preservation, dashboard payload removal, JavaScript syntax, local website links, sensitive-file exclusions and Git file sizes. The saved scientific results and RECHECK_REPORT are historical outputs, not a fresh training run. The old local Python environment points to an unavailable base interpreter; recreate an environment before executing the pipeline. Existing scripts may require local path configuration. A successful upload does not establish algorithmic correctness or cross-hospital generalization.

Open the site locally with 'node code/publication/serve_website.cjs', then visit http://127.0.0.1:8080. Publishing website source to main does not by itself configure GitHub Pages hosting.
