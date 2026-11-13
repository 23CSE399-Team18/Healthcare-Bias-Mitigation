"""Fetch the CDC NHIS 2023 sample adult public release and documentation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parents[2] / "hypertension" / "raw" / "source"
FILES = {
    "adult23csv.zip": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
    "adult-codebook.pdf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/adult-codebook.pdf",
    "adult-summary.pdf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/adult-summary.pdf",
    "srvydesc-508.pdf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/srvydesc-508.pdf",
}

def fetch(item):
    name, url = item
    folder = HERE / ("raw" if name.endswith(".zip") else "documentation")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read()
    if name.endswith(".pdf") and not data.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF response: " + name)
    target.write_bytes(data)
    if name.endswith(".zip"):
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP checksum failure")
            for member in archive.infolist():
                if member.filename.lower().endswith(".csv"):
                    out = folder / Path(member.filename).name
                    out.write_bytes(archive.read(member))
    result = {"file": str(target.relative_to(HERE)), "url": url,
              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    print(json.dumps(result), flush=True)
    return result

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, FILES.items()))
    (HERE / "download_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
