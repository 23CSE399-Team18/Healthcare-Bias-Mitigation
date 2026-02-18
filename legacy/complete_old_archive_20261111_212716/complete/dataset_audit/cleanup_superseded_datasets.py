"""User-authorized removal of wrong, incomplete and superseded dataset copies.

Only explicitly enumerated files below the project are removed. Correct original
sources, fresh cohort datasets, PDF/images, code and historical figures survive.
"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "dataset_audit"


def checked(path):
    target = path.resolve()
    if not target.is_relative_to(ROOT) or target == ROOT:
        raise ValueError(f"Refusing path outside project: {target}")
    return target


def plan():
    # Verify the approved replacements exist before offering any removal.
    for relative in (
        "Heart_correct/LLCP2022.XPT", "Heart_correct/LLCP2022XPT.zip",
        "Heart_correct/dataset_verified/heart_2022_baseline_raw.csv",
        "hypertension_correct/nhis2023/derived/hypertension_nhis2023_raw_cohort.csv",
        "diabetes_correct/diabetes_baseline_raw.csv",
        "resp_physionet/resp/dataset_verified/respiratory_cohort_24h_landmark.csv",
    ):
        assert checked(ROOT / relative).is_file(), relative
    canonical = ROOT / "resp_physionet/resp/official_verified"
    with (canonical / "canonical_file_manifest.csv").open(newline="", encoding="utf-8") as inp:
        manifest = list(csv.DictReader(inp))
    assert len(manifest) == 30
    for row in manifest:
        target = checked(canonical / row["canonical_file"])
        assert target.is_file() and target.stat().st_size > 0, str(target)
    candidates = {}

    def add(path, reason):
        path = checked(path)
        if path.is_file():
            candidates[path] = reason

    add(ROOT / "hypertension_correct/LLCP2024.XPT", "No hypertension diagnosis outcome in supplied file")
    add(ROOT / "Heart_correct/LLCP2022_raw.csv", "Superseded CSV; verified original XPT and fresh baseline retained")
    # The complete approved eICU originals are now gzip, which fits FAT32.
    for path in (ROOT / "resp_physionet/resp").glob("*.csv"):
        reason = "Superseded uncompressed copy; complete canonical gzip retained"
        if path.name in {"vitalPeriodic.csv", "nurseCharting.csv"}:
            reason = "Truncated/incomplete CSV; complete canonical gzip retained"
        elif path.name == "admissionDrug.csv":
            reason = "Not an approved verified source and not used by project cohort; omitted from retained set"
        add(path, reason)
    # Remove only obsolete dataset/learned-transform files, not code or figures.
    for relative in (
        "Heart_correct/preprocessing_output/data",
        "resp_physionet/resp/preprocessing_output/data",
        "resp_physionet/resp/preprocessing_output/cache/extract",
    ):
        folder = checked(ROOT / relative)
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file():
                    add(path, "Superseded preprocessing cohort or cached extraction; fresh approved cohorts retained")
    rows = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "reason": reason}
            for path, reason in sorted(candidates.items())]
    (AUDIT / "cleanup_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main():
    rows = plan()
    if "--apply" not in sys.argv:
        print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows), "plan": str(AUDIT / "cleanup_plan.json")}))
        return
    removed = []
    for row in rows:
        path = checked(ROOT / row["path"])
        # Stop if something changed between the plan and its execution.
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"File changed before cleanup: {path}")
        path.unlink()
        removed.append(row)
        (AUDIT / "cleanup_completed.json").write_text(json.dumps(removed, indent=2), encoding="utf-8")
    # Move only the remaining historical outputs into a clearly labelled archive.
    archive = checked(AUDIT / "legacy_results_not_for_training")
    archive.mkdir(exist_ok=True)
    for disease, relative in (
        ("heart", "Heart_correct/preprocessing_output"),
        ("respiratory", "resp_physionet/resp/preprocessing_output"),
    ):
        source, destination = checked(ROOT / relative), checked(archive / disease)
        if source.exists():
            if destination.exists():
                raise FileExistsError(f"Archive collision: {destination}")
            source.rename(destination)
    empty = checked(ROOT / "hypertension_correct/official_2023")
    if empty.exists() and not any(empty.iterdir()):
        empty.rmdir()
    (archive / "README.md").write_text(
        "# Historical outputs — not for training\n\n"
        "Old figures, tables and fitted-model caches are retained as historical project work. "
        "Their superseded datasets and extraction caches were removed at the user's request. "
        "Use only the fresh cohorts listed in D:/complete/DATASET_STATUS.md. "
        "Rerunning preprocessing must start from the corrected raw sources and cohort definitions.\n",
        encoding="utf-8")
    print(json.dumps({"removed_files": len(removed), "bytes_removed": sum(r["bytes"] for r in removed),
                      "legacy_outputs": str(archive), "correct_sources_preserved": True}))


if __name__ == "__main__":
    main()
