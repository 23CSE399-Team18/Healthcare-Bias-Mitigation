"""Wait for completed disease-profile browser files, then verify/import once.

Only named downloads and small local manifests are inspected while waiting.
No credentials, browser profiles, network requests, extraction, or evaluation.
STOP_DISEASE_BROWSER_IMPORT stops the wait before import starts. Original
downloads remain untouched; the canonical importer validates their SHA256.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from download_mimic import DISEASE_FILES, RAW_ROOT, ROOT, save_json, utc_now
from import_browser_downloads import candidate_files

STATE = ROOT / "disease_browser_import_job.json"
LOG = ROOT / "disease_browser_import_job.log"
LOCK = ROOT / "disease_browser_import_job.lock"
STOP = ROOT / "STOP_DISEASE_BROWSER_IMPORT"


def pending_downloads(download_dir):
    """Report readiness by filenames/manifests; verification happens at import."""
    pending = []
    for name in DISEASE_FILES:
        manifest = Path(str(RAW_ROOT / name) + ".parts") / "manifest.json"
        imported = False
        if manifest.is_file():
            try:
                imported = json.loads(manifest.read_text(encoding="utf-8")).get("completed") is True
            except (ValueError, OSError):
                pass  # The importer will report an invalid destination if present.
        if not imported and not candidate_files(download_dir, name):
            pending.append(name)
    if not candidate_files(download_dir, "SHA256SUMS.txt"):
        pending.append("SHA256SUMS.txt")
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--wait-hours", type=float, default=48, help="Maximum total job hours, up to 48")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--check-only", action="store_true", help="Report readiness without writing or importing")
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 48 or not 0 < args.poll_seconds <= 60:
        parser.error("--wait-hours must be in (0, 48] and --poll-seconds in (0, 60]")
    download_dir = args.download_dir.expanduser().resolve()
    if not download_dir.is_dir():
        parser.error("Download directory does not exist: " + str(download_dir))
    if args.check_only:
        print(json.dumps({"profile": "disease_data", "missing_completed_downloads": pending_downloads(download_dir),
                          "download_directory": str(download_dir), "stop_requested": STOP.exists()}))
        return 0
    try:
        with LOCK.open("x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError:
        print("A disease import watcher lock exists; inspect disease_browser_import_job.json before starting another.")
        return 2
    state = {"schema_version": 1, "pid": os.getpid(), "started_utc": utc_now(), "profile": "disease_data",
             "download_directory": str(download_dir), "requested_files": list(DISEASE_FILES),
             "credentials_read": False, "network_access": False, "browser_profile_read": False,
             "log": str(LOG), "wait_hours": args.wait_hours}

    def update(status, message, **fields):
        state.update(status=status, message=message, updated_utc=utc_now(), **fields)
        save_json(STATE, state)
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"[{utc_now()}] {message}\n")
        print(message, flush=True)

    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        previous = None
        while True:
            if STOP.exists():
                update("stopped", "STOP_DISEASE_BROWSER_IMPORT found; no import started.")
                return 130
            if time.monotonic() >= deadline:
                update("timed_out", "Time limit reached; completed downloads are preserved.")
                return 3
            pending = pending_downloads(download_dir)
            if not pending:
                break
            if pending != previous:
                update("waiting_for_downloads", "Waiting for completed browser files: " + ", ".join(pending),
                       pending_files=pending)
                previous = pending
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        if STOP.exists():
            update("stopped", "STOP_DISEASE_BROWSER_IMPORT found; no import started.")
            return 130
        update("verifying_and_importing", "All named files are ready; verifying/importing the disease profile once.",
               pending_files=[])
        with LOG.open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "import_browser_downloads.py"),
                                     "--disease-data", "--download-dir", str(download_dir)],
                                    cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(0.1, deadline - time.monotonic()))
        if result.returncode:
            update("failed", f"Importer exited with code {result.returncode}; see the job log and browser_import_state.json.")
            return result.returncode
        update("complete", "Disease-profile files verified and imported locally. No extraction or evaluation was run.",
               completed_utc=utc_now())
        return 0
    except subprocess.TimeoutExpired:
        update("timed_out", "Time limit reached during import; original downloads and completed imports are preserved.")
        return 3
    except KeyboardInterrupt:
        update("stopped", "Watcher interrupted; original downloads are preserved.")
        return 130
    except Exception as error:
        update("failed", str(error))
        return 1
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
