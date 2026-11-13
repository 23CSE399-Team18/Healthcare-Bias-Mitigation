"""Resolve eICU tables safely without expanding multi-gigabyte CSV files.

Example: pd.read_csv(get_eicu_table_path("patient"))
For large tables use pd.read_csv(path, chunksize=100_000).
"""
from pathlib import Path
import csv
import hashlib

DATA_DIR = Path(__file__).resolve().parents[2] / "respiratory" / "raw" / "source"
GZIP_DIR = DATA_DIR / "official_verified"
TRUNCATED_RAW = {"vitalPeriodic.csv", "nurseCharting.csv"}


def _sha256(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def get_eicu_table_path(table, verify=True):
    """Return a verified canonical gzip path, or an audited intact CSV.

    Accepts names such as "patient", "patient.csv", "patient.csv.gz".
    Never returns the known truncated vitalPeriodic/nurseCharting CSVs.
    verify=False skips the repeated content hash; use only after verification.
    """
    name = str(table)
    if Path(name).name != name or any(c in name for c in "/\\"):
        raise ValueError("Pass an eICU table name, not a path")
    if name.endswith(".gz"):
        name = name[:-3]
    if not name.endswith(".csv"):
        name += ".csv"
    manifest = GZIP_DIR / "canonical_file_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing canonical manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = {row["table"]: row for row in csv.DictReader(fh)}
    record = rows.get(name)
    if record is None:
        raise FileNotFoundError(
            f"No verified canonical source registered for {name}. "
            "Download the official file from PhysioNet using your authorized account; "
            "admissionDrug.csv has not been independently byte verified."
        )
    compressed = GZIP_DIR / record["canonical_file"]
    if compressed.exists():
        if verify and _sha256(compressed) != record["sha256"]:
            raise ValueError(f"Canonical gzip SHA-256 mismatch: {compressed}")
        return compressed
    raw = DATA_DIR / name
    if name in TRUNCATED_RAW:
        raise FileNotFoundError(f"Verified gzip missing; refusing known truncated CSV: {raw}")
    if record["raw_status_before_packaging"] != "verified" or not raw.exists():
        raise FileNotFoundError(f"No audited intact source available for {name}")
    with raw.open("rb") as fh:
        fh.seek(-1, 2)
        if fh.read(1) != b"\n":
            raise ValueError(f"CSV is incomplete or lacks its expected newline: {raw}")
    if verify and _sha256(raw) != record["raw_sha256_before_packaging"]:
        raise ValueError(f"Audited raw CSV SHA-256 mismatch: {raw}")
    return raw
