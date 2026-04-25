"""Read-only byte integrity audit; only writes its JSON report."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "resp_physionet" / "resp"
GZ = Path(r"D:\physionet.org\files\eicu-crd\2.0")
DEST = ROOT / "dataset_audit" / "eicu_integrity.json"
MANIFEST = RAW / "SHA256SUMS.txt"
SUMS = {name: digest for digest, name in (line.split() for line in MANIFEST.read_text().splitlines() if line.strip())}


def digest_stream(fh):
    h = hashlib.sha256()
    total = 0
    last = b""
    while data := fh.read(8 << 20):
        h.update(data)
        total += len(data)
        last = data[-1:]
    return h.hexdigest(), total, last == b"\n"


def inspect(raw):
    row = {"table": raw.name, "raw_path": str(raw), "raw_bytes": raw.stat().st_size}
    with raw.open("rb") as fh:
        row["raw_sha256"], _, row["raw_ends_newline"] = digest_stream(fh)
    gz = GZ / (raw.name + ".gz")
    row["gzip_path"] = str(gz)
    row["gzip_exists"] = gz.exists()
    if not gz.exists():
        row["status"] = "gzip_not_available_for_byte_comparison"
        return row
    row["gzip_bytes"] = gz.stat().st_size
    with gz.open("rb") as fh:
        row["gzip_sha256"] = hashlib.file_digest(fh, "sha256").hexdigest()
    row["manifest_expected_gzip_sha256"] = SUMS.get(gz.name)
    row["gzip_matches_local_manifest"] = row["gzip_sha256"] == SUMS.get(gz.name)
    try:
        with gzip.open(gz, "rb") as fh:
            expected, size, newline = digest_stream(fh)
        row.update(decompressed_sha256=expected, decompressed_bytes=size,
                   decompressed_ends_newline=newline, gzip_crc_valid=True,
                   raw_matches_decompressed=(row["raw_sha256"] == expected))
        row["status"] = "verified" if row["raw_matches_decompressed"] and row["gzip_matches_local_manifest"] else "repair_raw_from_verified_gzip" if row["gzip_matches_local_manifest"] else "manifest_mismatch"
    except Exception as exc:
        row.update(status="gzip_decompression_error", error=str(exc), gzip_crc_valid=False)
    return row


if __name__ == "__main__":
    started = time.monotonic()
    results = []
    paths = sorted(RAW.glob("*.csv"), key=lambda p: p.stat().st_size)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inspect, p): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result["table"], result["status"], flush=True)
            DEST.write_text(json.dumps({"complete": len(results) == len(paths), "elapsed_seconds": round(time.monotonic()-started, 1), "manifest_path": str(MANIFEST), "manifest_provenance": "Local supplied manifest; official network comparison handled separately", "tables": sorted(results, key=lambda r: r["table"])}, indent=2), encoding="utf-8")
    print("Report:", DEST, "seconds:", round(time.monotonic()-started, 1), flush=True)
