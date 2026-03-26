"""Download the official MIMIC-IV files needed for the ICU generalization study.

Run in a local interactive console. Credentials are requested with input/getpass,
kept only in process memory, and sent only to the official HTTPS download origin.
No account state, training, access agreement, or project approval is bypassed.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw" / "mimiciv" / "3.1"
BASE_URL = "https://physionet.org/files/mimiciv/3.1/"
STATE_FILE = ROOT / "download_state.json"
SEGMENT_BYTES = 1_000_000_000  # Strictly below FAT32's single-file limit.
FILES = (
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "icu/d_items.csv.gz",
    "icu/chartevents.csv.gz",
)
# Preserve the original mortality-study profile above. Download the smaller
# disease tables before the two large event archives in this optional profile.
DISEASE_FILES = FILES[:-1] + (
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/omr.csv.gz",
    "icu/chartevents.csv.gz",
    "hosp/labevents.csv.gz",
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class DownloadError(RuntimeError):
    pass


def select_files(*, disease_data=False, metadata_only=False):
    if disease_data and metadata_only:
        raise ValueError("--disease-data and --metadata-only cannot be combined")
    if metadata_only:
        return FILES[:-1]
    return DISEASE_FILES if disease_data else FILES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def allowed_url(url):
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname == "physionet.org"
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and "%" not in parsed.path
            and posixpath.normpath(parsed.path).startswith("/files/mimiciv/3.1/"))


class OfficialOnlyRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise DownloadError("PhysioNet redirected outside the approved download location. "
                                "Sign in on the official project page and confirm MIMIC-IV access.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class OfficialSession:
    def __init__(self, username, password):
        credentials = (username + ":" + password).encode("utf-8")
        self._authorization = "Basic " + base64.b64encode(credentials).decode("ascii")
        self._opener = build_opener(OfficialOnlyRedirects())

    def open(self, url, *, offset=0):
        if not allowed_url(url):
            raise DownloadError("Refusing to send credentials outside the official MIMIC-IV HTTPS location")
        headers = {"Authorization": self._authorization,
                   "User-Agent": "MIMIC-IV-Research-Downloader/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        return self._opener.open(Request(url, headers=headers), timeout=120)

    def close(self):
        self._authorization = None
        self._opener = None


def parse_publisher_checksums(data, required_files=FILES):
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DownloadError("Publisher checksum response is not an ASCII checksum manifest") from exc
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or not _HASH.fullmatch(pieces[0].lower()):
            raise DownloadError("Publisher response is not a SHA256SUMS manifest (possibly an access page)")
        name = pieces[1].strip().lstrip("*")
        while name.startswith("./"):
            name = name[2:]
        if name.startswith("/") or ".." in name.split("/"):
            raise DownloadError("Unsafe path in publisher checksum manifest")
        digest = pieces[0].lower()
        if name in result and result[name] != digest:
            raise DownloadError("Publisher manifest has conflicting checksums for one filename")
        result[name] = digest
    missing = [name for name in required_files if name not in result]
    if missing:
        raise DownloadError("Publisher manifest does not contain the required study files: " + ", ".join(missing))
    return result


def fetch_publisher_manifest(session, required_files=FILES):
    with session.open(BASE_URL + "SHA256SUMS.txt") as response:
        if response.status != 200:
            raise DownloadError("Unexpected status obtaining the publisher checksum manifest")
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise DownloadError("Publisher checksum manifest is unexpectedly large")
    checksums = parse_publisher_checksums(data, required_files)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    target = RAW_ROOT / "SHA256SUMS.txt"
    temporary = target.with_suffix(".txt.partial")
    temporary.write_bytes(data)
    temporary.replace(target)
    return checksums


def validated_prefix(directory, url, expected):
    """Rehash completed segments before resuming; incomplete segment is discarded."""
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("url") != url or progress.get("expected_sha256") != expected:
            raise DownloadError("Existing partial source belongs to another URL or publisher version")
        parts = progress.get("parts", [])
    else:
        parts = []
        if list(directory.glob("part-*.bin")):
            raise DownloadError("Unregistered source segments exist; inspect them before restarting")
    full_digest = hashlib.sha256()
    offset = 0
    for index, part in enumerate(parts):
        if part.get("filename") != f"part-{index:05d}.bin" or part.get("bytes") != SEGMENT_BYTES:
            raise DownloadError("Partial-source segment ordering or length is invalid")
        path = directory / part["filename"]
        if not path.exists() or path.stat().st_size != part["bytes"]:
            raise DownloadError("A completed partial-source segment is missing or has changed size")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
                full_digest.update(block)
        if digest.hexdigest() != part.get("sha256"):
            raise DownloadError("A completed partial-source segment failed its saved checksum")
        offset += part["bytes"]
    # Only this downloader's unfinished chunk names can be removed here.
    for partial in directory.glob("part-*.partial"):
        if re.fullmatch(r"part-\d{5}\.partial", partial.name):
            partial.unlink()
    return parts, offset, full_digest


def download_one(session, relative_name, expected):
    from streaming_sources import load_segment_manifest
    logical_path = RAW_ROOT / relative_name
    directory = Path(str(logical_path) + ".parts")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    url = BASE_URL + relative_name
    if manifest_path.exists():
        _, manifest = load_segment_manifest(logical_path, verify=True)
        if manifest.get("expected_sha256") != expected or manifest.get("url") != url:
            raise DownloadError("Existing verified source does not match the current publisher manifest")
        print(f"Already downloaded and verified: {relative_name}", flush=True)
        return manifest
    parts, offset, full_digest = validated_prefix(directory, url, expected)
    with session.open(url, offset=offset) as response:
        if offset:
            content_range = _RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if response.status != 206 or not content_range or int(content_range.group(1)) != offset:
                raise DownloadError("Server did not honor the exact resume offset. Existing completed "
                                    "segments are preserved; no bytes were appended.")
            total = int(content_range.group(3))
        else:
            if response.status != 200:
                raise DownloadError("Unexpected status starting a source download")
            length = response.headers.get("Content-Length")
            if not length or not length.isdigit():
                raise DownloadError("Official download has no reliable content length")
            total = int(length)
        if total <= offset:
            raise DownloadError("Official download has an invalid or changed source length")
        needed = total - offset
        if shutil.disk_usage(directory).free < needed + 128 * 1024 * 1024:
            raise DownloadError(f"Insufficient free disk space for {relative_name}: "
                                f"approximately {needed / 1e9:.2f} GB more is needed")
        print(f"Downloading {relative_name}: {total / 1e9:.3f} GB compressed; "
              f"resume offset {offset / 1e9:.3f} GB. Segments are at most 1 GB.", flush=True)
        received = offset
        while received < total:
            index = len(parts)
            name = f"part-{index:05d}.bin"
            partial = directory / f"part-{index:05d}.partial"
            final = directory / name
            if final.exists():
                raise DownloadError("Unregistered segment would be overwritten; inspect the partial source")
            goal = min(SEGMENT_BYTES, total - received)
            size = 0
            digest = hashlib.sha256()
            with partial.open("wb") as output:
                while size < goal:
                    block = response.read(min(4 * 1024 * 1024, goal - size))
                    if not block:
                        raise DownloadError("Source connection ended early; verified complete segments "
                                            "were preserved for a later resume")
                    output.write(block)
                    digest.update(block)
                    full_digest.update(block)
                    size += len(block)
                    received += len(block)
            partial.replace(final)
            parts.append({"filename": name, "bytes": size, "sha256": digest.hexdigest()})
            save_json(directory / "progress.json", {
                "schema_version": 1, "url": url, "expected_sha256": expected,
                "bytes_expected": total, "parts": parts, "completed": False,
            })
            print(f"  {relative_name}: {received / total:.1%} ({received / 1e9:.3f} GB)", flush=True)
        if response.read(1):
            raise DownloadError("Source length exceeded its published HTTP content length")
    actual = full_digest.hexdigest()
    if actual != expected:
        raise DownloadError("Full compressed source SHA256 does not match the publisher manifest. "
                            "The source is NOT marked complete and must not be used.")
    manifest = {
        "schema_version": 1, "source": "PhysioNet MIMIC-IV v3.1", "relative_name": relative_name,
        "url": url, "bytes": total, "expected_sha256": expected, "sha256": actual,
        "parts": parts, "compression": "gzip", "completed": True, "retrieved_utc": utc_now(),
    }
    save_json(manifest_path, manifest)
    (directory / "progress.json").unlink(missing_ok=True)
    print(f"Publisher SHA256 verified: {relative_name}", flush=True)
    return manifest


def failure_message(error):
    if isinstance(error, HTTPError) and error.code in (401, 403):
        return ("PhysioNet did not authorize this download (HTTP401/403). Use the approved "
                "PhysioNet username and confirm access to MIMIC-IV v3.1, required training, "
                "and its data-use agreement at https://physionet.org/content/mimiciv/3.1/. "
                "An account alone does not establish project approval. No credentials were saved.")
    if isinstance(error, HTTPError):
        return f"Official server returned HTTP{error.code}. Verified completed segments were preserved."
    if isinstance(error, (URLError, TimeoutError, ConnectionError)):
        return "The official source connection failed. Verified completed segments were preserved; rerun to resume."
    return str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--metadata-only", action="store_true", help="Download the four smaller metadata files only")
    profile.add_argument("--disease-data", action="store_true",
                         help="Include diagnosis, lab, and OMR tables for disease-specific research (ten files)")
    args = parser.parse_args()
    selected = select_files(disease_data=args.disease_data, metadata_only=args.metadata_only)
    if not sys.stdin.isatty():
        print("Run this downloader in a local interactive terminal. Do not send credentials through chat or tool input.", file=sys.stderr)
        return 2
    print("MIMIC-IV v3.1 official download: "
          + ("disease-specific research subset (ten files)." if args.disease_data else "MIMIC-eICU ICU study."))
    print("Use the PhysioNet account that has approved MIMIC-IV access.")
    print("The password is hidden and is never stored in files or command arguments.")
    username = input("PhysioNet username for the approved account: ").strip()
    if not username:
        print("No username entered; no download attempted.")
        return 2
    password = getpass.getpass("PhysioNet password (hidden): ")
    if not password:
        print("No password entered; no download attempted.")
        return 2
    session = OfficialSession(username, password)
    username = password = None
    state = {"schema_version": 1, "source": "MIMIC-IV v3.1", "started_utc": utc_now(),
             "status": "checking_project_access", "publisher_manifest_obtained": False,
             "credentials_saved": False, "files": [], "requested_files": list(selected),
             "profile": "disease_data" if args.disease_data else "metadata" if args.metadata_only else "icu_mortality"}
    save_json(STATE_FILE, state)
    try:
        checksums = fetch_publisher_manifest(session, selected)
        state["publisher_manifest_obtained"] = True
        state["status"] = "downloading"
        save_json(STATE_FILE, state)
        for relative_name in selected:
            manifest = None
            for attempt in range(3):
                try:
                    manifest = download_one(session, relative_name, checksums[relative_name])
                    break
                except HTTPError as exc:
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise
                    print("Temporary official-server error; retrying with completed segments preserved.", flush=True)
                    time.sleep(3 * (attempt + 1))
                except (URLError, TimeoutError, ConnectionError):
                    if attempt == 2:
                        raise
                    print("Temporary connection error; retrying with completed segments preserved.", flush=True)
                    time.sleep(3 * (attempt + 1))
            state["files"].append({"relative_name": relative_name, "bytes": manifest["bytes"],
                                   "sha256": manifest["sha256"], "completed": True})
            save_json(STATE_FILE, state)
        state["status"] = "metadata_complete" if args.metadata_only else "complete"
        state["completed_utc"] = utc_now()
        save_json(STATE_FILE, state)
        print("Download finished. Every selected file matches PhysioNet's SHA256 manifest.", flush=True)
        return 0
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["message"] = "User interrupted the download; completed segments remain available for resume."
        save_json(STATE_FILE, state)
        print("Stopped. Verified complete segments were preserved; credentials were not saved.")
        return 130
    except Exception as error:
        message = failure_message(error)
        state["status"] = "access_required" if isinstance(error, HTTPError) and error.code in (401, 403) else "failed"
        state["message"] = message
        state["finished_utc"] = utc_now()
        save_json(STATE_FILE, state)
        print(message, file=sys.stderr)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
