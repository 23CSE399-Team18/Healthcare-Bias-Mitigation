"""Inspect original BRFSS heart data and create an un-imputed labelled task view."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]   # D:/complete
import pandas as pd

DATA = ROOT / "heart" / "raw" / "source"
OUT = DATA / "dataset_verified"
OUT.mkdir(exist_ok=True)
SOURCE_PAGE = "https://www.cdc.gov/brfss/annual_data/annual_2022.html"
SOURCE_ZIP = "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"


def sha256(stream):
    h = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def main():
    with (DATA / "LLCP2022.XPT").open("rb") as inp:
        raw_hash = sha256(inp)
    with zipfile.ZipFile(DATA / "LLCP2022XPT.zip") as archive:
        member = next(x for x in archive.namelist() if x.strip().upper().endswith(".XPT"))
        with archive.open(member) as inp:
            zipped_raw_hash = sha256(inp)
    assert raw_hash == zipped_raw_hash, "Local XPT differs from stored original archive"
    with pd.read_sas(DATA / "LLCP2022.XPT", format="xport", iterator=True, chunksize=1,
                     encoding="latin1") as reader:
        sas_rows = reader.nobs
        sas_columns = list(reader.columns)
    assert sas_rows == 445132
    wanted = ["_STATE", "SEQNO", "_PSU", "_STSTR", "_LLCPWT", "SEXVAR", "_AGE80", "_RACEGR4",
              "_MICHD", "CVDINFR4", "CVDCRHD4", "_BMI5", "_BMI5CAT", "SMOKE100", "SMOKDAY2",
              "_SMOKER3", "EXERANY2", "SLEPTIM1", "GENHLTH", "PHYSHLTH", "MENTHLTH",
              "DIABETE4", "EDUCA", "INCOME3", "MEDCOST1", "CHECKUP1", "ALCDAY4", "AVEDRNK3"]
    # Build directly from the archived-XPT-verified source. Do not trust previously
    # transformed CSVs when reconstructing original labels and missingness.
    parts = []
    with pd.read_sas(DATA / "LLCP2022.XPT", format="xport", chunksize=20000,
                     encoding="latin1") as reader:
        for chunk in reader:
            parts.append(chunk[wanted].copy())
    frame = pd.concat(parts, ignore_index=True)
    frame["SEQNO"] = frame["SEQNO"].astype("string").str.strip()
    assert len(frame) == sas_rows
    assert frame[["_STATE", "SEQNO"]].duplicated().sum() == 0
    eligible = frame["_MICHD"].isin([1, 2])
    task = frame.loc[eligible].drop(columns=["_MICHD", "CVDINFR4", "CVDCRHD4"]).copy()
    task["HeartDisease"] = frame.loc[eligible, "_MICHD"].map({1: 1, 2: 0}).astype(int)
    # Preserve survey sentinel codes; model preprocessing must decode them using the codebook.
    task.to_csv(OUT / "heart_2022_baseline_raw.csv", index=False)
    frame.loc[~eligible, ["_STATE", "SEQNO", "_MICHD", "CVDINFR4", "CVDCRHD4"]].to_csv(
        OUT / "heart_label_exclusions.csv", index=False)
    counts = {str(k): int(v) for k, v in frame["_MICHD"].value_counts(dropna=False).items()}
    with (DATA / "LLCP2022XPT.zip").open("rb") as inp:
        zip_hash = sha256(inp)
    summary = {
        "source_page": SOURCE_PAGE, "original_download_url": SOURCE_ZIP,
        "source_type": "CDC BRFSS 2022; public cross-sectional telephone survey",
        "xpt_rows": int(sas_rows), "xpt_variables": len(sas_columns),
        "derived_csv_rows": len(frame), "unique_respondents": int(len(frame)),
        "local_zip_crc_valid": True, "xpt_matches_archived_member": True,
        "xpt_sha256": raw_hash, "zip_sha256": zip_hash,
        "live_archive_comparison": "Not performed: current CDC www download host returns HTTP 403",
        "target": "HeartDisease = 1 if _MICHD=1; 0 if _MICHD=2; other/missing target excluded",
        "target_meaning": "Ever reported myocardial infarction OR angina/coronary heart disease",
        "raw_target_counts": counts, "included_rows": len(task),
        "excluded_unknown_target_rows": int((~eligible).sum()),
        "positive_rows": int(task.HeartDisease.sum()), "negative_rows": int((task.HeartDisease == 0).sum()),
        "metadata_columns": ["_STATE", "SEQNO", "_PSU", "_STSTR", "_LLCPWT"],
        "sensitive_attributes": ["SEXVAR", "_AGE80", "_RACEGR4"],
        "label_derived_columns_excluded_from_predictors": ["_MICHD", "CVDINFR4", "CVDCRHD4"],
        "limits": ["Not UCI Heart Disease; no measured ECG or serum cholesterol in this task view",
                   "Not all heart diseases; not future-onset prediction", "Raw sentinel codes are intentionally preserved",
                   "Existing preprocessing outputs and their fairness claims were not validated by this source audit"]
    }
    (OUT / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = set(summary["metadata_columns"])
    sensitive = set(summary["sensitive_attributes"])
    with (OUT / "data_dictionary.csv").open("w", newline="", encoding="utf-8") as inp:
        writer = csv.writer(inp)
        writer.writerow(["column", "role", "notes"])
        for col in task.columns:
            role = "target" if col == "HeartDisease" else "survey metadata / not predictor" if col in metadata else "fairness attribute / possible predictor" if col in sensitive else "candidate predictor"
            writer.writerow([col, role, "Raw CDC codes; consult BRFSS 2022 codebook. _BMI5 is BMI x100; _AGE80 top-coded at80." if col != "HeartDisease" else summary["target_meaning"]])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
