"""Download public CDC source material without using mirrors or credentials."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "dataset_audit"
FILES = {
    "brfss2023_data": (
        "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip",
        ROOT / "hypertension_correct" / "official_2023" / "LLCP2023XPT.zip",
    ),
    "brfss2023_codebook": (
        "https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip",
        ROOT / "hypertension_correct" / "official_2023" / "codebook23_llcp-v2-508.zip",
    ),
    "brfss2022_codebook": (
        "https://www.cdc.gov/brfss/annual_data/2022/zip/codebook22_llcp-v2-508.zip",
        ROOT / "Heart_correct" / "official_documentation" / "codebook22_llcp-v2-508.zip",
    ),
    "brfss2023_calculated_variables": (
        "https://www.cdc.gov/brfss/annual_data/2023/pdf/2023-calculated-variables-version4-508.pdf",
        ROOT / "hypertension_correct" / "official_2023" / "2023-calculated-variables.pdf",
    ),
    "brfss2023_source_page": (
        "https://www.cdc.gov/brfss/annual_data/annual_2023.html",
        ROOT / "hypertension_correct" / "official_2023" / "source_page.html",
    ),
    "brfss2022_source_page": (
        "https://www.cdc.gov/brfss/annual_data/annual_2022.html",
        ROOT / "Heart_correct" / "official_documentation" / "source_page.html",
    ),
    "brfss2024_source_page": (
        "https://www.cdc.gov/brfss/annual_data/annual_2024.html",
        AUDIT / "brfss2024_source_page.html",
    ),
    "brfss2023_variable_layout": (
        "https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html",
        ROOT / "hypertension_correct" / "official_2023" / "variable_layout.html",
    ),
}


def download(item):
    key, (url, destination) = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchDatasetVerification/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as output:
                digest = hashlib.sha256()
                size = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                final_url = response.url
            if destination.suffix.lower() == ".zip":
                with zipfile.ZipFile(tmp) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("ZIP integrity failure")
            elif destination.suffix.lower() == ".pdf":
                with tmp.open("rb") as check:
                    if check.read(5) != b"%PDF-":
                        raise ValueError("Response is not a PDF")
            tmp.replace(destination)
            result = {"key": key, "url": url, "resolved_url": final_url,
                      "path": str(destination.relative_to(ROOT)), "bytes": size,
                      "sha256": digest.hexdigest(), "status": "downloaded"}
            print(json.dumps(result), flush=True)
            return result
        except Exception as error:
            if attempt == 2:
                result = {"key": key, "url": url, "status": "failed", "error": str(error)}
                print(json.dumps(result), flush=True)
                return result
            time.sleep(2)


if __name__ == "__main__":
    AUDIT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, FILES.items()))
    (AUDIT / "public_download_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    source = ROOT / "hypertension_correct" / "official_2023" / "LLCP2023XPT.zip"
    if source.exists():
        with zipfile.ZipFile(source) as archive:
            # Use only the XPT basename from the trusted official archive.
            for member in archive.infolist():
                if member.filename.lower().endswith(".xpt"):
                    target = source.parent / Path(member.filename).name
                    if not target.exists():
                        with archive.open(member) as inp, target.open("wb") as out:
                            while block := inp.read(1024 * 1024):
                                out.write(block)
                    print("Extracted " + str(target), flush=True)
