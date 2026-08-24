# Canonical eICU gzip sources

Use these `.csv.gz` files directly. They are complete copies of the pre-existing
PhysioNet files in `D:\physionet.org\files\eicu-crd\2.0`; they were not newly
downloaded during this audit. Each copy was checked against the SHA-256 value in
the supplied `SHA256SUMS.txt`. The compressed sources also passed full gzip CRC
validation during the integrity audit. Verification against the supplied local
manifest is not a separate online provenance check.

- Official dataset and access conditions: https://physionet.org/content/eicu-crd/2.0/
- Official files: https://physionet.org/files/eicu-crd/2.0/
- Local per-file sources, sizes, and hashes: `canonical_file_manifest.csv`.
- Full before-repair integrity results: `../../../dataset_audit/eicu_integrity.json`.
- Original license is copied as `LICENSE.txt` when available.

The project contains 30 canonical compressed tables. `admissionDrug.csv.gz` was
not present among the original compressed files. The existing uncompressed
`admissionDrug.csv` is preserved but has not been independently byte verified;
it is not used by the current respiratory notebook.

Two sibling raw CSVs are truncated: `../vitalPeriodic.csv` and
`../nurseCharting.csv`. Their complete decompressed forms exceed FAT32's maximum
single-file size. The compressed originals fit on this drive and can be streamed
without writing the expanded CSVs. Do not use those two raw CSVs as full tables.

These remain credentialed eICU data. Keep files under the applicable PhysioNet
agreement and do not publish patient-level datasets or this folder in a public
repository.
