"""Restore truncated eICU CSVs from locally held, checksum-verified originals.

Existing truncated files are preserved with .truncated_original suffix.
No network access, patient uploads, or preprocessing is performed.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes
import gzip
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "resp_physionet" / "resp"
SOURCES = Path(r"D:\physionet.org\files\eicu-crd\2.0")
EXPECTED = {
    "vitalPeriodic.csv": {
        "gzip_sha256": "329b522f97f7d3d0cb4541e4fc4afc32048eb8fd0f8c9b156907f845a992906c",
        "raw_sha256": "09818ad685112afb5e227ec495a1ecadc2d93aad1ae937875b469ac20f321744",
        "raw_bytes": 7938281236,
    },
    "nurseCharting.csv": {
        "gzip_sha256": "d10444dc6b530dfd198b42f2841de7b76045570fbf10b08e991c133006661c2c",
        "raw_sha256": "a16f0269487fca6123e3b17eb8ee73053f58d630d7b8d9c9682a0ffe305320cb",
        "raw_bytes": 11466038363,
    },
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def restore(item):
    name, expected = item
    source = SOURCES / (name + ".gz")
    destination = DATA / name
    backup = DATA / (name + ".truncated_original")
    tmp = DATA / (name + ".verified_partial")
    if destination.stat().st_size == expected["raw_bytes"] and file_hash(destination) == expected["raw_sha256"]:
        return {"file": name, "status": "already_verified"}
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}; inspect before retrying")
    if file_hash(source) != expected["gzip_sha256"]:
        raise ValueError(f"Compressed source checksum mismatch: {source}")
    digest = hashlib.sha256()
    size = 0
    print("Restoring " + name, flush=True)
    with gzip.open(source, "rb") as inp, tmp.open("wb") as out:
        while block := inp.read(8 * 1024 * 1024):
            out.write(block)
            digest.update(block)
            size += len(block)
    if digest.hexdigest() != expected["raw_sha256"] or size != expected["raw_bytes"]:
        raise ValueError(f"Restored content mismatch: {tmp}")
    # Only replace the original once a complete validated replacement exists.
    destination.rename(backup)
    tmp.rename(destination)
    result = {"file": name, "status": "repaired", "bytes": size,
              "sha256": digest.hexdigest(), "preserved_original": str(backup.relative_to(ROOT)),
              "source": str(source), "source_manifest": "resp_physionet/resp/SHA256SUMS.txt"}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if os.name == "nt":
        filesystem = ctypes.create_unicode_buffer(261)
        result = ctypes.windll.kernel32.GetVolumeInformationW(
            str(DATA.anchor), None, 0, None, None, None, filesystem, 261)
        if result and filesystem.value.upper() == "FAT32":
            raise SystemExit("D: uses FAT32: full CSVs exceed its 4GB file limit. "
                             "Use resp/official_verified/*.csv.gz and eicu_sources.py instead. "
                             "No decompression was started.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(restore, EXPECTED.items()))
    (ROOT / "dataset_audit" / "eicu_repairs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
