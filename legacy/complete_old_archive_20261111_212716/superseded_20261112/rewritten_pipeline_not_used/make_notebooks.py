"""Create the project notebooks (code/notebooks/*.ipynb).

Each preprocessing notebook can re-run its dataset's full pipeline (RUN_PIPELINE = True; long)
and always displays the saved results: bias summary, every method-comparison table and links to
the individual dashboards. The latent notebook does the same for the combined latent stage.
Execute with:  python -m jupyter nbconvert --to notebook --execute --inplace code/notebooks/<name>.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "code" / "notebooks"
TITLES = {"heart": "Heart disease | CDC BRFSS 2022", "diabetes": "Diabetes | CDC NHANES 2021-2023",
          "hypertension": "Hypertension | CDC NHIS 2023", "respiratory": "Respiratory ICU mortality | eICU-CRD v2.0"}
ORDER = ["duplicate", "missing", "outlier", "noise", "scale", "domain", "imbalance", "demographic"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


SETUP = """
import sys, json
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML
ROOT = Path(r"D:/complete")
sys.path.insert(0, str(ROOT / "code" / "preprocessing"))
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)
"""


def preprocessing_nb(name):
    cells = [md(f"""
# {TITLES[name]} - 8-bias preprocessing

Raw data before preprocessing: `{name}/raw/{name}_raw.csv`. Outputs: `{name}/preprocessed/` (train / validation / test CSVs,
method tables, report) and `{name}/plots/` (one dashboard per bias).

Order: duplicates -> split (70/15/15) -> missing -> outliers -> noise -> scale -> encoding/domain -> class imbalance -> demographic.
For every bias all traditional and modern methods in the method table are benchmarked with the table's metrics; the best mean
rank is selected and applied only if the bias is detected (otherwise before = after). Everything is fitted on training rows;
the validation split is used for scoring; the test split is never used for selection.
"""), code(SETUP), md("""
## 1. Run the pipeline (optional)
Set `RUN_PIPELINE = True` to recompute everything (this overwrites `preprocessed/` and `plots/`; it takes a long time).
"""), code(f"""
RUN_PIPELINE = False
if RUN_PIPELINE:
    from run import Pipeline
    Pipeline("{name}").run()
"""), md("## 2. Bias summary"), code(f"""
OUT = ROOT / "{name}" / "preprocessed"
report = json.loads((OUT / "pipeline_report.json").read_text())
print("Rows:", {{s: v["rows"] for s, v in report["splits"].items()}}, "| synthetic training rows:", report["splits"]["train"]["synthetic_rows"])
display(pd.read_csv(OUT / "bias_summary.csv"))
"""), md("## 3. Method comparison for each bias (every method x every metric; lower mean rank = better)")]
    for bias in ORDER:
        cells.append(code(f"""
path = OUT / "methods" / "{bias}_method_comparison.csv"
step = report["steps"]["{bias}"]
print("{bias.upper()}: detected =", step["detected"], "| best mean rank =", step["best"], "| applied =", step["applied"])
print("Protocol:", step["protocol"])
display(pd.read_csv(path, index_col=0).round(4) if path.exists() else "not applicable (no domain identifier)")
"""))
    cells += [md("## 4. Individual dashboards"), code(f"""
plots = ROOT / "{name}" / "plots"
display(HTML("<br>".join(f'<a href="{{p.as_uri()}}" target="_blank">{{p.name}}</a>' for p in sorted(plots.glob("*.html")))))
"""), md("## 5. Preprocessed training data (first rows)"), code(f"""
train = pd.read_csv(OUT / "{name}_train_preprocessed.csv", nrows=5)
display(train)
""")]
    return notebook(cells)


def latent_nb():
    return notebook([md("""
# Combined latent feature extraction (heart, diabetes, hypertension, respiratory)

One common 16-dimensional latent space for the four preprocessed datasets. Methods: DGCCA, SAINT, MV-DAE, compared with a
per-dataset PCA baseline (no shared space). Metrics: cross-dataset CKA, Macro-F1, Silhouette, NMI/ARI (validation rows).
Outputs: `latent/data/combined_latent_<split>.csv` (all datasets) and `latent/data/<dataset>_latent_<split>.csv`.
"""), code(SETUP + 'sys.path.insert(0, str(ROOT / "code" / "latent"))'), code("""
RUN_LATENT = False
if RUN_LATENT:
    import latent
    latent.main()
"""), code("""
report = json.loads((ROOT / "latent" / "latent_report.json").read_text())
print("Selected:", report["selected"], "|", report["rule"])
display(pd.read_csv(ROOT / "latent" / "latent_method_comparison.csv", index_col=0).round(4))
display(pd.DataFrame({m: {d: v["macro_f1"] for d, v in r["per_dataset"].items()} for m, r in report["methods"].items()}).assign(uncompressed=pd.Series(report["uncompressed_macro_f1"])).round(4))
"""), code("""
plots = ROOT / "latent" / "plots"
display(HTML("<br>".join(f'<a href="{p.as_uri()}" target="_blank">{p.name}</a>' for p in sorted(plots.glob("*.html")))))
display(pd.read_csv(ROOT / "latent" / "data" / "combined_latent_train.csv", nrows=5))
""")])


def main():
    NB.mkdir(parents=True, exist_ok=True)
    for name in TITLES:
        (NB / f"{name}_preprocessing.ipynb").write_text(json.dumps(preprocessing_nb(name), indent=1), encoding="utf-8")
    (NB / "latent_alignment.ipynb").write_text(json.dumps(latent_nb(), indent=1), encoding="utf-8")
    print("notebooks written to", NB)


if __name__ == "__main__":
    main()
