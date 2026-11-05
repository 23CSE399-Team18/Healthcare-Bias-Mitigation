"""Build the in-processing notebooks for every disease from the shared percent-format source.

    python code/inprocessing/make_notebooks.py

Writes <disease>/<disease>_inprocessing.ipynb (complete code, from inprocessing_cells.py with the disease's settings)
and <disease>/<disease>_inprocessing_sample10.ipynb (10 test records, ERM vs the selected method for every bias).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DISEASES = {   # the cross-hospital bias of every disease uses the cross-hospital ICU cohort (hospital/) with that disease's label
    "heart": {"title": "Heart disease (CDC BRFSS 2022)"},
    "diabetes": {"title": "Diabetes (CDC NHANES 2021-2023)"},
    "hypertension": {"title": "Hypertension (CDC NHIS 2023)"},
    "respiratory": {"title": "Respiratory ICU mortality (eICU-CRD v2.0)"},
}
KERNEL = {"display_name": "Python 3.12 (D:\\complete venv)", "language": "python", "name": "complete-venv"}


def cells_from_percent(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cell = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


SAMPLE_SETUP = '''
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 220)
DATASET = "__DATASET__"
OUT = Path(r"D:/complete") / DATASET / "inprocessed"
sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
sample = pd.read_csv(OUT / "sample10_test_predictions.csv", low_memory=False)
hsample = pd.read_csv(OUT / "sample10_hospital_predictions.csv", low_memory=False)   # 5 eICU test-hospital + 5 MIMIC stays
ID_COL = "meta_patientunitstayid" if "meta_patientunitstayid" in sample else "meta_record_id"
GROUPS = ["sens_sex", "sens_age_group", "sens_race_ethnicity"]


def compare(method, hospital=False):
    frame = hsample if hospital else sample
    th = sel["thresholds_hospital"] if hospital else sel["thresholds"]
    show = (["meta_stay_id", "meta_source", "environment"] if hospital else [ID_COL]) + [c for c in GROUPS if c in frame] + ["target"]
    out = frame[show].copy()
    out["ERM probability"] = frame["p_ERM"].round(4)
    out["ERM decision"] = (frame["p_ERM"] >= th["ERM"]).astype(int)
    out[f"{method} probability"] = frame[f"p_{method}"].round(4)
    out[f"{method} decision"] = (frame[f"p_{method}"] >= th[method]).astype(int)
    out["ERM correct"] = out["ERM decision"] == out["target"]
    out[f"{method} correct"] = out[f"{method} decision"] == out["target"]
    return out.set_index(show[0])
'''


def sample_notebook(name, info):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": (
        f"# {info['title']} - 10 test records through the in-processing methods\n\n"
        "Ten records of the untouched **test** split (fixed random seed). For every in-processing bias the table shows the "
        "ERM reference prediction and the prediction of the **selected** method (its probability, its decision at its own "
        "validation-chosen threshold, and whether that decision matches the recorded label), followed by the full "
        "method comparison used for the selection (validation split). The cross-hospital bias uses the cross-hospital ICU "
        "cohort instead: 5 stays from eICU test hospitals and 5 from MIMIC-IV (the external hospital), label = the recorded "
        f"{name} condition. Values come from the saved run "
        f"(`{name}/inprocessed/`); nothing is retrained here.").splitlines(keepends=True)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": SAMPLE_SETUP.replace("__DATASET__", name).strip("\n").splitlines(keepends=True)}]
    for fam, title in (("demographic", "Demographic prediction dependence"), ("intersectional", "Intersectional / subgroup unfairness"),
                       ("calibration", "Predictive miscalibration"), ("environment", "Cross-hospital generalization failure")):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}"]})
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": (
            f'method = sel["selected"]["{fam}"]\n'
            'display(Markdown(f"**Selected:** {method} (methods compared: {\', \'.join(sel[\'families\'][\'' + fam + '\'][\'methods\'])})"))\n'
            f'display(compare(method, hospital={fam == "environment"}))\n'
            f'display(pd.read_csv(OUT / "comparison_{fam}.csv", index_col=0).round(4))').splitlines(keepends=True)})
    return notebook(cells)


def main():
    source = (HERE / "inprocessing_cells.py").read_text(encoding="utf-8")
    for name, info in DISEASES.items():
        text = source.replace("__DATASET__", name).replace("__TITLE__", info["title"])
        (ROOT / name / f"{name}_inprocessing.ipynb").write_text(json.dumps(notebook(cells_from_percent(text)), indent=1), encoding="utf-8")
        (ROOT / name / f"{name}_inprocessing_sample10.ipynb").write_text(json.dumps(sample_notebook(name, info), indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
