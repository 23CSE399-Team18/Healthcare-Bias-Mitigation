"""Copy existing eICU gzip originals into the project and verify every copy."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
RESP = ROOT / "resp_physionet" / "resp"
SOURCE = Path(r"D:\physionet.org\files\eicu-crd\2.0")
DEST = RESP / "official_verified"
AUDIT = ROOT / "dataset_audit" / "eicu_integrity.json"


def sha(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    manifest = RESP / "SHA256SUMS.txt"
    sums = {name: digest for digest, name in (line.split() for line in manifest.read_text().splitlines() if line.strip())}
    audit = {row["table"]: row for row in json.loads(AUDIT.read_text())["tables"]}
    rows = []
    for original in sorted(SOURCE.glob("*.csv.gz")):
        destination = DEST / original.name
        expected = sums.get(original.name)
        if not expected:
            raise RuntimeError(f"No manifest checksum for {original.name}")
        if destination.exists():
            actual = sha(destination)
            if actual != expected:
                raise RuntimeError(f"Existing destination preserved: checksum mismatch {destination}")
        else:
            part = DEST / (original.name + ".copying")
            if part.exists():
                raise RuntimeError(f"Existing partial preserved: {part}")
            digest = hashlib.sha256()
            with original.open("rb") as src, part.open("xb") as dst:
                while block := src.read(8 << 20):
                    digest.update(block)
                    dst.write(block)
            actual = digest.hexdigest()
            if actual != expected or sha(part) != expected:
                raise RuntimeError(f"Checksum mismatch; uncommitted partial retained: {part}")
            part.rename(destination)
        table = original.name.removesuffix(".gz")
        prior = audit[table]
        rows.append({"table": table, "canonical_file": original.name, "bytes": destination.stat().st_size,
                     "sha256": actual, "matches_supplied_manifest": True,
                     "source_page": "https://physionet.org/content/eicu-crd/2.0/",
                     "download_url": "https://physionet.org/files/eicu-crd/2.0/" + original.name,
                     "local_original_path": str(original), "raw_status_before_packaging": prior["status"],
                     "raw_sha256_before_packaging": prior["raw_sha256"],
                     "expected_decompressed_bytes": prior.get("decompressed_bytes", ""),
                     "expected_decompressed_sha256": prior.get("decompressed_sha256", "")})
        print(original.name, "copied and SHA-256 verified", flush=True)
    for name in ("SHA256SUMS.txt", "LICENSE.txt"):
        source = manifest if name == "SHA256SUMS.txt" else SOURCE / name
        target = DEST / name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
    out = DEST / "canonical_file_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("DONE", len(rows), "gzip files", sum(r["bytes"] for r in rows), "bytes", flush=True)


if __name__ == "__main__":
    main()
