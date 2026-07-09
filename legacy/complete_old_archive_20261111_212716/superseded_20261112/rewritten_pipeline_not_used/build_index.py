"""Write D:/complete/index.html and README.md from the saved preprocessing and latent reports."""
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "preprocessing"))
from dashboard import CSS  # noqa: E402

NAMES = {"heart": ("Heart disease", "CDC BRFSS 2022"), "diabetes": ("Diabetes", "CDC NHANES Aug 2021-Aug 2023"),
         "hypertension": ("Hypertension", "CDC NHIS 2023"), "respiratory": ("Respiratory ICU", "PhysioNet eICU-CRD v2.0")}


def main():
    rows_md, cards = [], []
    for name, (title, source) in NAMES.items():
        rep_path = ROOT / name / "preprocessed" / "pipeline_report.json"
        if not rep_path.exists():
            continue
        rep = json.loads(rep_path.read_text())
        raw_rows = sum(1 for _ in open(ROOT / name / "raw" / f"{name}_raw.csv", encoding="utf-8")) - 1
        s = pd.read_csv(ROOT / name / "preprocessed" / "bias_summary.csv")
        detected = ", ".join(s.loc[s.detected.astype(str) == "True", "bias"]) or "none"
        splits = " / ".join(f"{rep['splits'][k]['rows']:,}" for k in ("train", "validation", "test"))
        rows_md.append(f"| {title} | {source} | {raw_rows:,} | {splits} | {detected} | [{name}/plots/index.html]({name}/plots/index.html) |")
        applied = "".join(f"<li><b>{html.escape(r.bias)}</b>: {html.escape(str(r.applied_to_data)) if str(r.applied_to_data) != 'nan' else 'none'}</li>"
                          for r in s.itertuples())
        cards.append(f"<a class='tile' href='{name}/plots/index.html'><div class='card'><h3>{title}</h3><div class='note'>{source} &middot; raw rows {raw_rows:,} "
                     f"&middot; train/val/test {splits}</div><ul class='note'>{applied}</ul></div></a>")
    latent = ROOT / "latent" / "latent_report.json"
    latent_line = ""
    if latent.exists():
        rep = json.loads(latent.read_text())
        r = rep["methods"][rep["selected"]]
        latent_line = (f"Selected **{rep['selected']}**: CKA {r['CKA (alignment)']:.3f}, Macro-F1 {r['Macro-F1 (preservation)']:.3f}, "
                       f"Silhouette {r['Silhouette (structure)']:.3f}, NMI {r['NMI (domain)']:.3f}, ARI {r['ARI (domain)']:.3f}.")
        cards.append(f"<a class='tile' href='latent/plots/index.html'><div class='card'><h3>Combined latent space</h3><div class='note'>"
                     f"{html.escape(latent_line.replace('**', ''))}</div></div></a>")
    readme = [
        "# Healthcare bias mitigation: preprocessing and latent stages", "",
        "Open **[index.html](index.html)** for every dashboard.", "",
        "## Folder layout", "",
        "```text",
        "<dataset>/raw/            data before preprocessing (<dataset>_raw.csv) + raw/source/ official files and cohort extracts",
        "<dataset>/preprocessed/   <dataset>_{train,validation,test}_preprocessed.csv, methods/*_method_comparison.csv,",
        "                          bias_summary.csv, pipeline_report.json, feature_schema.json, run_log.txt",
        "<dataset>/plots/          01..08 one dashboard per bias + index.html",
        "latent/                   data/combined_latent_<split>.csv (all four datasets), data/<dataset>_latent_<split>.csv,",
        "                          latent_method_comparison.csv, latent_report.json, plots/ (one dashboard per latent metric)",
        "code/                     build_raw/ (official source -> raw), preprocessing/, latent/, notebooks/",
        "docs/                     project report, slides and the method/metric tables",
        "```", "",
        "## Datasets", "",
        "| Dataset | Source | Raw rows | Preprocessed train / val / test | Biases detected | Dashboards |",
        "|---|---|---:|---|---|---|", *rows_md, "",
        "## Latent stage", "", latent_line or "Not yet run.", "",
        "## How it was produced", "",
        "1. `python code/build_raw/build_survey_raw.py` (BRFSS, NHIS, NHANES) and the eICU scripts in `code/build_raw/` build `<dataset>/raw/<dataset>_raw.csv` from the official files.",
        "2. `python code/preprocessing/run.py --dataset <name>` runs the 8 biases in order: duplicates, split, missing, outliers, noise, scale, encoding/domain, class imbalance, demographic.",
        "   Every method in the method table is benchmarked with the table's metrics; the best mean rank is selected and applied only when the bias is detected.",
        "3. `python code/latent/latent.py` learns the common latent space (DGCCA, SAINT, MV-DAE vs a PCA baseline).",
        "4. `python code/build_index.py` refreshes this page; `code/notebooks/` holds one notebook per stage.", "",
        "All fitting uses training rows; selection uses validation rows; test rows are never used for selection.",
        "In-processing is not included yet (its metrics have not been defined).", "",
        "Python environment: `Heart_correct/.venv` (left in place so the editor's interpreter keeps working).",
    ]
    (ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    page = (f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
            f"<title>Bias mitigation dashboards</title><style>{CSS} .tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}"
            ".tile{display:block;text-decoration:none;color:var(--text)}.tile:hover .card{outline:2px solid var(--blue-hi)}.tile h3{margin:0 0 6px;color:var(--blue-hi)}"
            ".wrap{padding:16px;max-width:1400px;margin:0 auto}</style></head><body><header class='topbar'><h1>Healthcare bias mitigation \u2014 dashboards</h1>"
            "<div class='sub'>Preprocessing (one dashboard per bias, per dataset) and the combined latent space</div></header>"
            f"<div class='wrap'><div class='tiles'>{''.join(cards)}</div><p class='note mt'>Each dataset folder contains raw/, preprocessed/ and plots/. "
            "See README.md for how the files were produced.</p></div></body></html>")
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    print("wrote index.html and README.md")


if __name__ == "__main__":
    main()
