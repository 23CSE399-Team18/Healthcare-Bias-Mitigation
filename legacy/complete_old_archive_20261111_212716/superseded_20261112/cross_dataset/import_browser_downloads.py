"""Verify official MIMIC-IV browser downloads and import them safely onto FAT32.

No network connection, browser profile, cookies, credentials, or account state
is accessed. Only explicitly named dataset/checksum files in the download
directory are read. Original browser downloads are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from download_mimic import (BASE_URL, FILES, RAW_ROOT, ROOT, SEGMENT_BYTES, DownloadError,
                           parse_publisher_checksums, save_json, select_files, utc_now)
from streaming_sources import SourceIntegrityError, load_segment_manifest

STATE_FILE = ROOT / "browser_import_state.json"
DEFAULT_DOWNLOAD_DIR = Path.home() / "Downloads"


class BrowserImportError(RuntimeError):
    pass


def browser_name_matches(name, canonical):
    """Match canonical names and normal browser duplicate-name suffixes only."""
    patterns = [re.escape(canonical)]
    for position in [i for i, value in enumerate(canonical) if value == "."] + [len(canonical)]:
        patterns.append(re.escape(canonical[:position]) + r" \([1-9]\d*\)" + re.escape(canonical[position:]))
    return any(re.fullmatch(pattern, name, flags=re.IGNORECASE) for pattern in patterns)


def candidate_files(download_dir, relative_name):
    canonical = Path(relative_name).name
    search_dirs = [download_dir]
    if "/" in relative_name:
        search_dirs.append(download_dir / Path(relative_name).parent)
    candidates = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and browser_name_matches(path.name, canonical):
                candidates.append(path)
    # Prefer the exact name, then browser duplicates. Never read .crdownload files.
    return sorted(set(candidates), key=lambda p: (p.name.lower() != canonical.lower(), p.name.lower(), str(p)))


def fingerprint(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def hash_stable_file(path, *, require_gzip=False):
    """Hash a stable named file, detecting replacement or writes during the read."""
    initial_path = path.stat()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        initial_open = os.fstat(stream.fileno())
        if fingerprint(initial_path) != fingerprint(initial_open):
            raise BrowserImportError(f"Source changed before hashing: {path.name}; wait for the browser download to finish")
        first = stream.read(8 * 1024 * 1024)
        if require_gzip and first[:2] != b"\x1f\x8b":
            raise BrowserImportError(f"Source is not gzip data: {path.name}")
        if first:
            digest.update(first)
            count += len(first)
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            count += len(block)
        final_open = os.fstat(stream.fileno())
    final_path = path.stat()
    if (fingerprint(initial_path) != fingerprint(final_open)
            or fingerprint(initial_path) != fingerprint(final_path)
            or count != initial_path.st_size):
        raise BrowserImportError(f"Source changed while hashing: {path.name}; wait for the browser download to finish")
    return {"path": path, "sha256": digest.hexdigest(), "bytes": count,
            "fingerprint": fingerprint(final_path)}


def choose_publisher_manifest(download_dir, required_files=FILES):
    candidates = candidate_files(download_dir, "SHA256SUMS.txt")
    if not candidates:
        raise BrowserImportError("Missing SHA256SUMS.txt. Download it from the official MIMIC-IV v3.1 Files page first.")
    valid = []
    for path in candidates:
        if path.stat().st_size > 2_000_000:
            continue
        before = fingerprint(path.stat())
        data = path.read_bytes()
        if before != fingerprint(path.stat()):
            raise BrowserImportError("Publisher checksum file changed during reading")
        try:
            checksums = parse_publisher_checksums(data, required_files)
        except Exception:
            continue
        valid.append((path, data, checksums))
    if not valid:
        raise BrowserImportError("No valid MIMIC-IV v3.1 SHA256SUMS.txt found. A saved sign-in/error webpage is not a checksum file.")
    required_maps = {tuple((name, record[2][name]) for name in required_files) for record in valid}
    if len(required_maps) != 1:
        raise BrowserImportError("Several downloaded checksum files describe different source versions. "
                                 "Put the intended MIMIC-IV v3.1 checksum and files in a separate directory and use --download-dir.")
    return valid[0]


def select_verified_source(download_dir, relative_name, expected):
    candidates = candidate_files(download_dir, relative_name)
    if not candidates:
        return None
    rejected = []
    for path in candidates:
        try:
            record = hash_stable_file(path, require_gzip=True)
        except BrowserImportError as error:
            rejected.append(str(error))
            continue
        if record["sha256"] == expected:
            return record
        rejected.append(f"SHA256 mismatch: {path.name}")
    raise BrowserImportError("No completed browser file matches the publisher checksum for "
                             + relative_name + ". " + "; ".join(rejected))


def existing_verified_source(relative_name, expected):
    logical_path = RAW_ROOT / relative_name
    directory = Path(str(logical_path) + ".parts")
    if not directory.exists():
        return None
    if not (directory / "manifest.json").exists():
        # Never overwrite an earlier partial downloader/import result.
        raise BrowserImportError(f"A partial source already exists at {directory}. "
                                 "It was preserved; inspect it before choosing a new import destination.")
    try:
        _, manifest = load_segment_manifest(logical_path, verify=True)
    except SourceIntegrityError as error:
        raise BrowserImportError(f"Existing source failed verification and was preserved: {relative_name}") from error
    if manifest.get("expected_sha256") != expected or manifest.get("url") != BASE_URL + relative_name:
        raise BrowserImportError("Existing source belongs to another publisher version and was preserved: " + relative_name)
    return manifest


def clean_own_staging(directory, logical_path):
    """Remove only numbered files made by this specific failed import."""
    if (directory.resolve().parent != logical_path.resolve().parent
            or not directory.name.startswith(logical_path.name + ".browser_import_")
            or not directory.resolve().is_relative_to(RAW_ROOT.resolve())):
        raise BrowserImportError("Unsafe import staging cleanup target")
    for path in directory.iterdir():
        if path.is_file() and (re.fullmatch(r"part-\d{5}\.bin", path.name)
                               or path.name in ("manifest.json", "manifest.json.tmp")):
            path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()


def import_verified_source(relative_name, expected, source):
    """Copy a previously verified stable browser source into <=1GB segments."""
    existing = existing_verified_source(relative_name, expected)
    if existing is not None:
        return existing
    path = source["path"]
    if source["sha256"] != expected or fingerprint(path.stat()) != source["fingerprint"]:
        raise BrowserImportError("Browser source changed after verification: " + path.name)
    logical_path = RAW_ROOT / relative_name
    logical_path.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(str(logical_path) + ".parts")
    if shutil.disk_usage(logical_path.parent).free < source["bytes"] + 128 * 1024 * 1024:
        raise BrowserImportError(f"Not enough destination space to import {relative_name}")
    staging = Path(tempfile.mkdtemp(prefix=logical_path.name + ".browser_import_", dir=logical_path.parent))
    parts = []
    full_digest = hashlib.sha256()
    copied = 0
    try:
        with path.open("rb") as inp:
            if fingerprint(os.fstat(inp.fileno())) != source["fingerprint"]:
                raise BrowserImportError("Browser source changed before copying: " + path.name)
            while copied < source["bytes"]:
                name = f"part-{len(parts):05d}.bin"
                digest = hashlib.sha256()
                size = 0
                goal = min(SEGMENT_BYTES, source["bytes"] - copied)
                with (staging / name).open("wb") as output:
                    while size < goal:
                        block = inp.read(min(8 * 1024 * 1024, goal - size))
                        if not block:
                            raise BrowserImportError("Browser source ended early during copying: " + path.name)
                        output.write(block)
                        digest.update(block)
                        full_digest.update(block)
                        size += len(block)
                        copied += len(block)
                parts.append({"filename": name, "bytes": size, "sha256": digest.hexdigest()})
                print(f"Imported {relative_name}: {copied / source['bytes']:.1%}", flush=True)
            if inp.read(1) or fingerprint(os.fstat(inp.fileno())) != source["fingerprint"]:
                raise BrowserImportError("Browser source changed during copying: " + path.name)
        if fingerprint(path.stat()) != source["fingerprint"] or full_digest.hexdigest() != expected:
            raise BrowserImportError("Imported bytes failed the publisher checksum; source was not marked complete")
        manifest = {
            "schema_version": 1, "source": "PhysioNet MIMIC-IV v3.1",
            "acquisition": "verified official browser download", "relative_name": relative_name,
            "url": BASE_URL + relative_name, "browser_source": str(path.resolve()),
            "bytes": copied, "expected_sha256": expected, "sha256": full_digest.hexdigest(),
            "parts": parts, "compression": "gzip", "completed": True, "retrieved_utc": utc_now(),
        }
        save_json(staging / "manifest.json", manifest)
        # Directory rename is atomic on this volume and never overwrites an existing source.
        if directory.exists():
            raise BrowserImportError("Another source appeared during import and was preserved: " + relative_name)
        staging.rename(directory)
        print("Publisher SHA256 verified and imported: " + relative_name, flush=True)
        return manifest
    except BaseException:
        if staging.exists():
            clean_own_staging(staging, logical_path)
        raise


def run_import(download_dir, *, disease_data=False):
    selected = select_files(disease_data=disease_data)
    download_dir = Path(download_dir).expanduser().resolve()
    if not download_dir.is_dir():
        raise BrowserImportError("Download directory does not exist: " + str(download_dir))
    state = {"schema_version": 1, "source": "MIMIC-IV v3.1", "method": "browser download import",
             "started_utc": utc_now(), "download_directory": str(download_dir), "status": "checking_downloads",
             "credentials_read": False, "browser_profile_read": False, "files": [], "missing_files": [],
             "requested_files": list(selected), "profile": "disease_data" if disease_data else "icu_mortality"}
    save_json(STATE_FILE, state)
    try:
        checksum_path, checksum_bytes, checksums = choose_publisher_manifest(download_dir, selected)
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        saved_checksum = RAW_ROOT / "SHA256SUMS.txt"
        if saved_checksum.exists():
            try:
                previous = parse_publisher_checksums(saved_checksum.read_bytes(), selected)
            except DownloadError as error:
                raise BrowserImportError("Existing publisher checksum is invalid for the selected files "
                                         "and was preserved: " + str(error)) from error
            if any(previous[name] != checksums[name] for name in selected):
                raise BrowserImportError("Existing publisher checksum describes another version and was preserved")
        else:
            with saved_checksum.open("xb") as output:
                output.write(checksum_bytes)
        state["publisher_checksum_source"] = str(checksum_path)
        state["status"] = "importing"
        save_json(STATE_FILE, state)
        for relative_name in selected:
            expected = checksums[relative_name]
            manifest = existing_verified_source(relative_name, expected)
            if manifest is None:
                source = select_verified_source(download_dir, relative_name, expected)
                if source is None:
                    state["missing_files"].append(relative_name)
                    continue
                manifest = import_verified_source(relative_name, expected, source)
            else:
                print("Already imported and verified: " + relative_name, flush=True)
            state["files"].append({"relative_name": relative_name, "completed": True,
                                   "bytes": manifest["bytes"], "sha256": manifest["sha256"]})
            save_json(STATE_FILE, state)
        state["status"] = "waiting_for_files" if state["missing_files"] else "complete"
        state["finished_utc"] = utc_now()
        if state["missing_files"]:
            state["message"] = "Finish the official browser downloads, then rerun the importer. .crdownload files are never used."
        else:
            state["message"] = f"All {len(selected)} selected files match the publisher checksums and are imported into FAT32-safe segments."
        save_json(STATE_FILE, state)
        return state
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["message"] = str(error) or "Import interrupted; original browser downloads were preserved."
        state["finished_utc"] = utc_now()
        save_json(STATE_FILE, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR,
                        help="Directory containing completed official downloads and SHA256SUMS.txt (default: your Downloads folder)")
    parser.add_argument("--disease-data", action="store_true",
                        help="Include diagnosis, lab, and OMR tables for disease-specific research (ten files)")
    args = parser.parse_args()
    try:
        result = run_import(args.download_dir, disease_data=args.disease_data)
    except (BrowserImportError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Import interrupted. Original browser downloads were not changed.", file=sys.stderr)
        return 130
    print(result["message"], flush=True)
    if result["missing_files"]:
        print("Still needed: " + ", ".join(result["missing_files"]), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
