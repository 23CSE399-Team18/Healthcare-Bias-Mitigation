"""Verify/import completed browser MIMIC sources and check full gzip integrity.

Only explicitly named Downloads files are read; originals are never changed.
The checksum manifest must have been acquired from the official v3.1 page.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
from pathlib import Path

from download_mimic import BASE_URL, RAW_ROOT, ROOT, save_json, utc_now
from import_browser_downloads import (choose_publisher_manifest,
    existing_verified_source, import_verified_source, select_verified_source,
    STATE_FILE as IMPORT_STATE_FILE)

PRESENT_RESEARCH_FILES = (
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz",
    "icu/d_items.csv.gz", "hosp/d_icd_diagnoses.csv.gz",
    "hosp/d_labitems.csv.gz", "hosp/diagnoses_icd.csv.gz",
    "icu/chartevents.csv.gz", "hosp/labevents.csv.gz",
)
REPORT_PATH = ROOT / "mimic_download_integrity.json"


def gzip_health(path):
    uncompressed_bytes = 0
    with gzip.open(path, "rb") as stream:
        first = stream.read(8 * 1024 * 1024)
        header = next(csv.reader(io.StringIO(first.split(b"\n", 1)[0].decode("utf-8"))))
        uncompressed_bytes += len(first)
        while block := stream.read(8 * 1024 * 1024):
            uncompressed_bytes += len(block)
    return {"gzip_crc": "passed_full_stream", "uncompressed_bytes": uncompressed_bytes,
            "columns": header}


def verify(download_dir, external_crc=()):
    checksum_path, checksum_bytes, checksums = choose_publisher_manifest(
        download_dir, PRESENT_RESEARCH_FILES)
    report = {"schema_version": 1, "source": "PhysioNet MIMIC-IV v3.1",
              "source_url": "https://physionet.org/content/mimiciv/3.1/",
              "publisher_checksum_path": str(checksum_path),
              "publisher_checksum_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
              "started_utc": utc_now(), "status": "verifying",
              "raw_originals_modified": False, "credentials_or_browser_profile_read": False,
              "files": [], "optional_not_required": ["hosp/omr.csv.gz"],
              "gzip_crc_delegated_to_extractor": list(external_crc)}
    save_json(REPORT_PATH, report)
    canonical_checksum = RAW_ROOT / "SHA256SUMS.txt"
    if not canonical_checksum.exists():
        canonical_checksum.parent.mkdir(parents=True, exist_ok=True)
        canonical_checksum.write_bytes(checksum_bytes)
    sources = {}
    for name in PRESENT_RESEARCH_FILES:
        print("Checking " + name, flush=True)
        source = select_verified_source(download_dir, name, checksums[name])
        if source is None:
            report["files"].append({"relative_name": name, "status": "missing"})
            save_json(REPORT_PATH, report)
            continue
        stored = existing_verified_source(name, checksums[name])
        if stored is None:
            stored = import_verified_source(name, checksums[name], source)
        record = {"relative_name": name, "url": BASE_URL + name,
                  "browser_file": str(source["path"]), "bytes": source["bytes"],
                  "sha256": source["sha256"], "expected_sha256": checksums[name],
                  "sha256_matches_publisher": True,
                  "canonical_logical_path": str(RAW_ROOT / name),
                  "canonical_parts": len(stored["parts"]), "canonical_verified": True,
                  "status": "sha256_verified_and_imported", "gzip_crc": "pending"}
        report["files"].append(record)
        sources[name] = source["path"]
        save_json(REPORT_PATH, report)
        print("Publisher SHA256 verified and canonical source ready: " + name, flush=True)
    report["status"] = "checking_gzip_crc"
    report["all_nine_sha256_verified"] = len(sources) == len(PRESENT_RESEARCH_FILES)
    save_json(REPORT_PATH, report)
    by_name = {entry["relative_name"]: entry for entry in report["files"]}
    # Two independent compressed sources can be checked without changing either.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(gzip_health, path): name for name, path in sources.items()
                   if name not in external_crc}
        for future in as_completed(futures):
            name = futures[future]
            try:
                by_name[name].update(future.result())
                by_name[name]["status"] = "verified"
                print("Full gzip CRC passed: " + name, flush=True)
            except Exception as error:
                by_name[name].update({"gzip_crc": "failed", "status": "failed", "error": str(error)})
                print("Full gzip CRC failed: " + name + ": " + str(error), flush=True)
            save_json(REPORT_PATH, report)
    for name in external_crc:
        if name in by_name:
            by_name[name]["gzip_crc"] = "delegated_to_full_stream_extractor"
    report["status"] = "sha256_passed_crc_delegated" if external_crc else "passed"
    if len(sources) != len(PRESENT_RESEARCH_FILES) or any(
            entry.get("status") == "failed" for entry in report["files"]):
        report["status"] = "incomplete_or_failed"
    report["finished_utc"] = utc_now()
    save_json(REPORT_PATH, report)
    if report["all_nine_sha256_verified"]:
        save_json(IMPORT_STATE_FILE, {
            "schema_version": 1, "source": "MIMIC-IV v3.1",
            "method": "browser download import", "status": "complete",
            "profile": "nine_present_disease_research_sources",
            "download_directory": str(download_dir), "credentials_read": False,
            "browser_profile_read": False, "requested_files": list(PRESENT_RESEARCH_FILES),
            "publisher_checksum_source": str(checksum_path),
            "files": [{"relative_name": entry["relative_name"], "completed": True,
                       "bytes": entry["bytes"], "sha256": entry["sha256"]}
                      for entry in report["files"]],
            "missing_files": [], "optional_not_downloaded": ["hosp/omr.csv.gz"],
            "finished_utc": report["finished_utc"],
            "message": "All nine selected downloaded archives match the publisher SHA256 "
                       "manifest and are imported into FAT32-safe segments. OMR is optional.",
            "integrity_report": str(REPORT_PATH),
        })
    print("Integrity result: " + report["status"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--delegate-crc", action="append", default=[],
                        help="Logical file CRC to be checked by a coordinated full-stream extractor")
    args = parser.parse_args()
    result = verify(args.download_dir.resolve(), args.delegate_crc)
    raise SystemExit(0 if result["status"] in ("passed", "sha256_passed_crc_delegated") else 2)
