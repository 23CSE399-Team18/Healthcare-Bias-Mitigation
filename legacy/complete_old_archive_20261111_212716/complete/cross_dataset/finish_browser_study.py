"""Finish the authorized local study when browser downloads are complete.

Reads only the named MIMIC source files/checksums in Downloads. It performs no
network access, browser control, credential handling, or external uploads.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

from download_mimic import FILES, RAW_ROOT, save_json
from import_browser_downloads import candidate_files

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "completion_job.json"
LOCK = ROOT / "completion_job.lock"
STOP = ROOT / "STOP_BROWSER_STUDY"
LOG = ROOT / "completion_job.log"


def now():
    return datetime.now(timezone.utc).isoformat()


def pending_downloads(download_dir):
    pending = []
    for name in FILES:
        manifest_path = Path(str(RAW_ROOT / name) + ".parts") / "manifest.json"
        imported = False
        if manifest_path.is_file():
            imported = json.loads(manifest_path.read_text()).get("completed") is True
        if not imported and not candidate_files(download_dir, name):
            pending.append(name)
    checksum = RAW_ROOT / "SHA256SUMS.txt"
    if not checksum.is_file() and not candidate_files(download_dir, "SHA256SUMS.txt"):
        pending.append("SHA256SUMS.txt")
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"missing_completed_downloads": pending_downloads(args.download_dir),
                          "download_directory": str(args.download_dir)}))
        return 0
    try:
        with LOCK.open("x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError:
        print("A completion job lock already exists. Inspect completion_job.json before starting another job.")
        return 2
    state = {"schema_version": 1, "pid": os.getpid(), "started_utc": now(),
             "download_directory": str(args.download_dir), "status": "waiting_for_download",
             "credentials_read": False, "network_access": False,
             "log": str(LOG), "results_directory": str(ROOT / "results")}

    def update(status, **fields):
        state.update(status=status, updated_utc=now(), **fields)
        save_json(STATE, state)

    def run_step(label, script, *arguments):
        if STOP.exists():
            raise InterruptedError("Local stop marker found; no further study step was started")
        update(label)
        print(label, flush=True)
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] {label}\n")
            log.flush()
            result = subprocess.run([sys.executable, "-u", str(script), *arguments],
                                    cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} did not complete (exit {result.returncode}); see completion_job.log")

    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        last_pending = None
        while pending := pending_downloads(args.download_dir):
            update("waiting_for_download", pending_files=pending)
            if pending != last_pending:
                print("Waiting for completed browser files: " + ", ".join(pending), flush=True)
                last_pending = pending
            if STOP.exists():
                raise InterruptedError("Local stop marker found while waiting for the browser download")
            if time.monotonic() >= deadline:
                update("download_wait_timed_out", message="Browser file is still incomplete. Rerun this job after the download finishes.")
                return 3
            time.sleep(max(1, min(args.poll_seconds, 60)))
        update("completed_downloads_detected", pending_files=[])
        run_step("verifying_and_importing_mimic", ROOT / "import_browser_downloads.py",
                 "--download-dir", str(args.download_dir))
        run_step("preparing_mimic_24h_features", ROOT / "prepare_cross_dataset.py", "--source", "mimic")
        run_step("evaluating_both_transfer_directions", ROOT / "evaluate_generalization.py")
        run_step("updating_dataset_handoff", ROOT.parent / "dataset_audit/finalize_dataset_report.py")
        update("complete", completed_utc=now(), message="Both real cohorts and cross-source baseline results are ready.")
        print("Study complete. See cross_dataset/results and DATASET_STATUS.md.", flush=True)
        return 0
    except InterruptedError as error:
        update("stopped", message=str(error))
        print(str(error), flush=True)
        return 130
    except Exception as error:
        update("failed", message=str(error))
        print(str(error), flush=True)
        return 1
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
