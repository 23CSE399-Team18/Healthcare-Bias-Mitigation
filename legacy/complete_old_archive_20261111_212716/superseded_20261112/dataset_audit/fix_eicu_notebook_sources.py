"""Patch only dataset paths and integrity failure handling; preserve outputs."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "resp_physionet" / "resp" / "Respiratory_Bias_Preprocessing.ipynb"
backup = path.with_name("Respiratory_Bias_Preprocessing.before_dataset_paths.ipynb.bak")
if not backup.exists():
    shutil.copyfile(path, backup)
book = json.loads(path.read_text(encoding="utf-8"))

old_paths = '''NOTEBOOK_DIR = Path.cwd()
DATA_DIR = NOTEBOOK_DIR if (NOTEBOOK_DIR / "patient.csv").exists() else Path(r"D:\\resp_physionet\\resp")
# checksum-verified PhysioNet originals; used only for CSVs in DATA_DIR that are truncated
GZ_DIR = Path(r"D:\\physionet.org\\files\\eicu-crd\\2.0")'''
new_paths = '''NOTEBOOK_DIR = Path.cwd()
DATA_CANDIDATES = [NOTEBOOK_DIR, NOTEBOOK_DIR / "resp_physionet" / "resp",
                   Path(r"D:\\complete\\resp_physionet\\resp")]
DATA_DIR = next((_d for _d in DATA_CANDIDATES if (_d / "patient.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Open this notebook from D:/complete or its resp_physionet/resp folder")
# Complete gzip sources avoid FAT32's maximum single-file size for expanded CSVs.
GZ_DIR = DATA_DIR / "official_verified"'''

old_checks = '''        trunc = csv_is_truncated(p)
        gz = GZ_DIR / f"{f}.gz"
        src, verified = str(p), ""
        if trunc and gz.exists():
            ok = cached(f"sha256_{f}", lambda: sha256_file(gz)) == sums.get(f"{f}.gz")
            src, verified = str(gz), "SHA-256 matches PhysioNet" if ok else "SHA-256 MISMATCH"
        elif trunc:
            verified = "no original found - truncated file used, data incomplete"'''
new_checks = '''        trunc = csv_is_truncated(p) if p.exists() else True
        gz = GZ_DIR / f"{f}.gz"
        src, verified = str(p), "audited intact raw CSV"
        if gz.exists():
            expected = sums.get(f"{f}.gz")
            if expected is None:
                raise ValueError(f"Missing supplied SHA-256 for {gz.name}")
            # Recheck actual bytes; do not reuse a stale cached checksum.
            if sha256_file(gz) != expected:
                raise ValueError(f"SHA-256 MISMATCH: refusing {gz}")
            src, verified = str(gz), "SHA-256 matches supplied manifest"
        elif trunc:
            raise FileNotFoundError(f"Missing verified original; refusing incomplete CSV: {p}")'''

counts = [0, 0, 0]
for cell in book["cells"]:
    source = "".join(cell.get("source", []))
    if old_paths in source:
        source = source.replace(old_paths, new_paths)
        counts[0] += 1
    if old_checks in source:
        source = source.replace(old_checks, new_checks)
        # Allow future loading when only the canonical gzip has been retained.
        source = source.replace('"size_MB": round(p.stat().st_size / 2 ** 20, 1),',
                                '"size_MB": round((p.stat().st_size if p.exists() else gz.stat().st_size) / 2 ** 20, 1),')
        counts[1] += 1
    # Keep documentation consistent with the source-only loading correction.
    old_doc = '''truncated CSV is found, the notebook streams PhysioNet's original gzip file instead, after checking it against
`SHA256SUMS.txt`. `nurseCharting` is not needed: its vital signs are covered by `vitalPeriodic` and APACHE.'''
    new_doc = '''canonical gzip original is available under `official_verified`, the notebook streams it after checking its
actual bytes against the supplied `SHA256SUMS.txt`. A checksum mismatch or a truncated CSV without its verified
original stops execution. `nurseCharting` is not needed: its vital signs are covered by `vitalPeriodic` and APACHE.'''
    if old_doc in source:
        source = source.replace(old_doc, new_doc)
        counts[2] += 1
    cell["source"] = source.splitlines(keepends=True)
if counts != [1, 1, 1]:
    raise RuntimeError(f"Unexpected patch matches {counts}; notebook was not modified")
path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Updated paths, checksum failure handling, and loading documentation; original backup:", backup)
