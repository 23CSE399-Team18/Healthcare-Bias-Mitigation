"""Recheck every pre-processing, cross-hospital, latent and in-processing output and the Team-18 website.

    python code/recheck.py          -> prints PASS/FAIL per check and writes RECHECK_REPORT.md

Nothing is recomputed or modified (read-only checks of the saved files).
"""
import hashlib
import json
import re
from pathlib import Path

from build_website import theme_dashboard          # the website publishes the dashboards in its orange + white theme

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = Path("D:/Team-18/Team-18")
DISEASES = {"heart": 440111, "diabetes": 7876, "hypertension": 29471, "respiratory": 28842}
SPLITS = ("train", "validation", "test")
DASHBOARDS = ["01_missing_data_bias_dashboard.html", "02_class_imbalance_bias_dashboard.html", "03_outlier_bias_dashboard.html",
              "04_feature_scale_bias_dashboard.html", "05_demographic_bias_dashboard.html", "06_duplicate_bias_dashboard.html",
              "07_data_noise_bias_dashboard.html", "08_encoding_domain_bias_dashboard.html", "index.html"]
TABLES = {"Missing data": "missing_data", "Class imbalance": "class_imbalance", "Outlier": "outlier", "Feature scale": "feature_scale",
          "Demographic": "demographic", "Duplicate": "duplicate", "Data noise": "data_noise", "Encoding / domain": "encoding_domain"}
TRADITIONAL = {"Duplicate": {"Exact duplicate removal", "Fuzzy matching", "GBER (graph-based ER)"},
               "Missing data": {"GMM", "Random Forest", "GAIN", "MICE", "KNN"},
               "Outlier": {"Isolation Forest", "LOF", "One-Class SVM"},
               "Data noise": {"Denoising autoencoder", "Wavelet denoising", "Median filter", "Savitzky\u2013Golay", "Savitzky-Golay"},
               "Feature scale": {"Quantile transform", "Min-Max scaler", "Robust scaler"},
               "Encoding / domain": {"DANN", "CORAL", "Autoencoder (domain-conditional)"},
               "Class imbalance": {"SMOTE", "ADASYN", "Borderline-SMOTE", "NearMiss"},
               "Demographic": {"Optimal Transport", "FairSMOTE", "Reweighting"}}
results = []


def check(area, name, ok, detail=""):
    results.append({"area": area, "check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"{'PASS' if ok else 'FAIL'}  {area:13s} {name}{(' - ' + str(detail)) if detail else ''}", flush=True)


def published_same(published, project):
    return published.read_text(encoding="utf-8") == theme_dashboard(project.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_ok(path):
    if not path.exists():
        return False, "missing"
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    unexecuted = sum(1 for c in code if c.get("execution_count") is None)
    errors = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
    return unexecuted == 0 and errors == 0, f"{len(code)} code cells, {unexecuted} not executed, {errors} with errors"


def features(df):
    return [c for c in df.columns if c != "target" and not c.startswith(("meta_", "sens_"))]


def check_disease(name, expected_rows):
    base = ROOT / name
    raw = pd.read_csv(base / "raw" / f"{name}_raw.csv", low_memory=False)
    check(name, "raw data before preprocessing", len(raw) == expected_rows, f"{len(raw):,} rows")
    ok, detail = notebook_ok(base / f"{name}_preprocessing.ipynb")
    check(name, "preprocessing notebook executed without errors", ok, detail)
    ok, detail = notebook_ok(base / f"{name}_sample10_comparison.ipynb")
    check(name, "10-sample notebook executed without errors", ok, detail)
    out = base / "preprocessed"
    log = json.loads((out / "pipeline_log.json").read_text(encoding="utf-8"))
    id_col = "meta_patientunitstayid" if name == "respiratory" else "meta_record_id"
    frames = {}
    for split in SPLITS:
        df = pd.read_csv(out / f"{name}_{split}_preprocessed.csv", low_memory=False, dtype={id_col: "string"})
        frames[split] = df
        X = df[features(df)].to_numpy(float)
        check(name, f"{split}: all predictor values finite", np.isfinite(X).all(), f"{df.shape[0]:,} rows x {X.shape[1]} predictors")
        if split == "train":
            constant = [c for c, lo, hi in zip(features(df), X.min(0), X.max(0)) if lo == hi]
            check(name, "train: no predictor collapsed to a constant", not constant, ", ".join(constant) or "every predictor varies")
        check(name, f"{split}: binary label", set(df.target.unique()) <= {0, 1})
        expected = log[f"{split}_rows"]
        check(name, f"{split}: rows match the run log", len(df) == expected, f"{len(df):,} vs {expected:,}")
        if split != "train":
            check(name, f"{split}: no synthetic rows", int(df.get("meta_synthetic", pd.Series([0])).sum()) == 0)
    real_train = frames["train"].loc[frames["train"].get("meta_synthetic", 0) == 0, id_col].astype(str)
    sets = {"train": set(real_train), "validation": set(frames["validation"][id_col].astype(str)), "test": set(frames["test"][id_col].astype(str))}
    check(name, "train / validation / test records are disjoint",
          not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]))
    feats = [features(frames[s]) for s in SPLITS]
    check(name, "same predictors in every split", feats[0] == feats[1] == feats[2], f"{len(feats[0])} predictors")
    for bias, key in TABLES.items():
        rec = log["biases"].get(bias)
        if rec is None:
            check(name, f"{bias}: recorded in the run log", False)
            continue
        table = out / f"comparison_{key}.csv"
        if rec["detected"]:
            tab = pd.read_csv(table, index_col=0) if table.exists() else None
            check(name, f"{bias}: detected, comparison table saved", tab is not None, f"{0 if tab is None else len(tab)} methods")
            sel = rec.get("selected")
            if str(sel).startswith("labels: "):         # respiratory noise: "labels: <method>; signals: <method or —>"
                sel = str(sel)[len("labels: "):].split(";")[0].strip()
            check(name, f"{bias}: applied method is modern and in the table", sel not in TRADITIONAL[bias] and tab is not None and sel in tab.index,
                  f"selected {sel}; best traditional {rec.get('best_traditional')}")
        else:
            check(name, f"{bias}: not detected, nothing applied (before = after)", rec.get("selected") in (None, "\u2014", "-", ""), rec.get("statistic"))
    check(name, "modern-vs-traditional evidence table saved", (out / "modern_vs_traditional.csv").exists())
    trace = json.loads((out / "sample10_trace.json").read_text(encoding="utf-8"))
    check(name, "10-record trace covers all 8 steps", len(trace["records"]) == 10 and len(trace["steps"]) == 8,
          f"{len(trace['records'])} records, {len(trace['steps'])} steps")
    for f in DASHBOARDS + (["07b_monitor_signal_noise_dashboard.html"] if name == "respiratory" else []):
        p = base / "plots" / f
        ok = p.exists() and p.stat().st_size > (1500 if f == "index.html" else 20000)
        if ok and f != "index.html":
            text = p.read_text(encoding="utf-8")
            m = re.search(r"const\s+D\s*=\s*(\{.*?\});\s*\n", text, flags=re.S)
            if m:
                try:
                    json.loads(m.group(1))
                except ValueError:
                    ok = False
        check(name, f"dashboard {f}", ok, f"{p.stat().st_size // 1024} KB" if p.exists() else "missing")
    return {s: len(frames[s]) for s in SPLITS}


HOSPITAL_ROWS = {"train": 80606, "validation": 20318, "test": 8454, "external": 73990}
IN_FILES = ["01_demographic_prediction_dependence_dashboard.html", "02_intersectional_subgroup_unfairness_dashboard.html",
            "03_predictive_miscalibration_dashboard.html", "04_cross_environment_generalization_dashboard.html", "index.html"]
IN_FAMILIES = {"demographic": {"FairCare", "DANN", "GroupDRO", "Fair-ERM"},
               "intersectional": {"FOMO", "Intersectional debiasing", "Fairness-constrained optimization", "GroupDRO (intersectional)"},
               "calibration": {"AvUC", "CRL", "Uncertainty-weighted gradient", "Calibration-regularized"},
               "environment": {"Fishr", "V-REx", "EQRM", "Fish"}}


def check_hospital():
    base = ROOT / "hospital"
    raw = sorted((base / "raw").glob("*/*_model_ready.csv"))
    check("hospital", "raw cohort tables (eICU + MIMIC, 4 conditions)", len(raw) == 8, f"{len(raw)} files")
    rows = {}
    for split, n in HOSPITAL_ROWS.items():
        df = pd.read_csv(base / "preprocessed" / f"hospital_{split}_preprocessed.csv", low_memory=False)
        rows[split] = len(df)
        X = df[[c for c in df.columns if not c.startswith(("meta_", "sens_", "target"))]].to_numpy(float)
        check("hospital", f"{split}: rows and finite predictors", len(df) == n and np.isfinite(X).all(), f"{len(df):,} rows x {X.shape[1]} predictors")
        src = set(df.meta_source.unique())
        check("hospital", f"{split}: source", src == ({"mimic"} if split == "external" else {"eicu"}), ", ".join(sorted(src)))
        if split != "external":
            share = df.sens_race_ethnicity.value_counts(normalize=True)
            check("hospital", f"{split}: race labels corrected (eICU mostly White, Asian small)",
                  share.get("White", 0) > 0.6 and share.get("Asian", 0) < 0.05,
                  f"White {share.get('White', 0):.1%}, Asian {share.get('Asian', 0):.1%}")
    tr = pd.read_csv(base / "preprocessed" / "hospital_train_preprocessed.csv", usecols=["meta_domain_hospitalid"])
    others = set()
    for s in ("validation", "test"):
        others |= set(pd.read_csv(base / "preprocessed" / f"hospital_{s}_preprocessed.csv", usecols=["meta_domain_hospitalid"]).meta_domain_hospitalid)
    check("hospital", "training hospitals disjoint from validation/test hospitals", not (set(tr.meta_domain_hospitalid) & others))
    check("hospital", "race correction audit saved", (base / "preprocessed" / "race_fix_audit.json").exists())
    return rows


def check_latent(rows, hosp_rows):
    base = ROOT / "latent"
    ok, detail = notebook_ok(base / "latent_alignment.ipynb")
    check("latent", "latent notebook executed without errors", ok, detail)
    rep = json.loads((base / "latent_report.json").read_text(encoding="utf-8"))
    check("latent", "five views: four diseases + cross-hospital ICU cohort", list(rep.get("datasets", [])) == list(DISEASES) + ["hospital"],
          ", ".join(rep.get("datasets", [])))
    tab = pd.read_csv(base / "latent_method_comparison.csv", index_col=0)
    check("latent", "all latent methods compared on all metrics",
          {"MV-DAE", "DGCCA", "SAINT", "PCA (no alignment)"} <= set(tab.index) and
          {"CKA (alignment)", "Macro-F1 (preservation)", "Silhouette (structure)", "NMI (domain)", "ARI (domain)"} <= set(tab.columns))
    best = tab.drop(index="PCA (no alignment)")["mean_rank"].astype(float).idxmin()
    check("latent", "selected method = best mean rank", rep["selected"] == best, f"selected {rep['selected']}")
    zc = [f"z{i + 1:02d}" for i in range(16)]
    for split in SPLITS:
        comb = pd.read_csv(base / "data" / f"combined_latent_{split}.csv", low_memory=False)
        expected = sum(rows[d][split] for d in rows) + hosp_rows[split]
        check("latent", f"combined {split}: rows = all preprocessed rows", len(comb) == expected, f"{len(comb):,} vs {expected:,}")
        check("latent", f"combined {split}: 16 finite latent features", np.isfinite(comb[zc].to_numpy(float)).all())
        for d in list(rows) + ["hospital"]:
            per = base / "data" / f"{d}_latent_{split}.csv"
            n = sum(1 for _ in per.open(encoding="utf-8")) - 1 if per.exists() else -1
            want = hosp_rows[split] if d == "hospital" else rows[d][split]
            check("latent", f"{d} {split}: per-dataset latent file", n == want, f"{n:,} rows")
    ext = base / "data" / "hospital_latent_external.csv"
    n = sum(1 for _ in ext.open(encoding="utf-8")) - 1 if ext.exists() else -1
    check("latent", "MIMIC external hospital encoded (not in the combined files)", n == hosp_rows["external"], f"{n:,} rows")
    check("latent", "website scatter of the same records in every method", (base / "latent_scatter.json").exists())
    for f in ["01_cka_alignment_dashboard.html", "02_macro_f1_preservation_dashboard.html", "03_silhouette_structure_dashboard.html",
              "04_nmi_ari_domain_invariance_dashboard.html", "index.html"]:
        p = base / "plots" / f
        check("latent", f"dashboard {f}", p.exists() and p.stat().st_size > 3000)


def payload_ok(path):
    text = path.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, flags=re.S)
    try:
        return m is not None and isinstance(json.loads(m.group(1)), dict)
    except ValueError:
        return False


def check_inprocessing(name):
    base = ROOT / name
    for nb in (f"{name}_inprocessing.ipynb", f"{name}_inprocessing_sample10.ipynb"):
        ok, detail = notebook_ok(base / nb)
        check(f"in:{name}", f"{nb} executed without errors", ok, detail)
    out = base / "inprocessed"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    check(f"in:{name}", "test split and MIMIC never used for selection",
          sel.get("test_used_for_selection") is False and sel.get("external_used_for_selection") is False)
    for fam, methods in IN_FAMILIES.items():
        tab = pd.read_csv(out / f"comparison_{fam}.csv", index_col=0)
        chosen = sel["selected"].get(fam)
        check(f"in:{name}", f"{fam}: all 4 methods + ERM compared on its 4 metrics", methods | {"ERM"} <= set(tab.index), f"{len(tab)} rows")
        cands = tab.loc[sel["families"][fam]["methods"]]          # the bias's method order: ties go to the first listed, as in the notebook
        flag = cands["eligible (AUROC guard)"].astype(str) == "True"
        cands = cands[flag] if flag.any() else cands
        rank = cands["mean rank"].astype(float)
        best = rank.idxmin()
        tied = sorted(m for m in rank.index if abs(rank[m] - rank.min()) < 1e-9 and m != best)
        check(f"in:{name}", f"{fam}: selected = best mean rank (validation)", chosen == best,
              f"selected {chosen}" + (f" (tied with {', '.join(tied)} at {rank.min():.2f}; first in the method list wins)" if tied else ""))
        check(f"in:{name}", f"{fam}: test metrics saved", (out / f"test_metrics_{fam}.csv").exists())
    for split in ("validation", "test", "external"):
        f = out / f"hospital_predictions_{split}.csv"
        k = sum(1 for _ in f.open(encoding="utf-8")) - 1 if f.exists() else -1
        check(f"in:{name}", f"cross-hospital predictions {split}", k == HOSPITAL_ROWS[split], f"{k:,} rows")
    for f in IN_FILES:
        p = base / "plots" / "inprocessing" / f
        ok = p.exists() and (f == "index.html" or payload_ok(p))
        check(f"in:{name}", f"dashboard {f}", ok, f"{p.stat().st_size // 1024} KB" if p.exists() else "missing")


def check_website():
    page = SITE / "index.html"
    text = page.read_text(encoding="utf-8")
    check("website", "page exists", page.exists(), f"{len(text) // 1024} KB")
    check("website", "no 'still running' placeholders left", "notebook running" not in text and "notebooks running" not in text)
    m = re.search(r"const MAP=(\{.*?\});", text, flags=re.S)
    explorer = json.loads(m.group(1)) if m else {}
    links = [v for stage in explorer.values() for d in stage.values() for v in d.values()]
    check("website", "explorer has the three stages", list(explorer) == ["Pre-processing", "Latent space", "In-processing"], ", ".join(explorer))
    pm = re.search(r"const P=(\{.*?\}),tabsEl=document\.getElementById\('preTabs'\)", text, flags=re.S)
    pre = json.loads(pm.group(1)) if pm else {}
    check("website", "pre-processing before/after visual has every dataset and bias",
          len(pre) == len(DISEASES) and all(v and len(v["cards"]) == 8 and v["e2e"] for v in pre.values()), f"{len(pre)} datasets")
    check("website", "plot explorer links all resolve", links and all(v and (SITE / v).exists() for v in links), f"{len(links)} dashboards")
    for name in DISEASES:
        same = all((SITE / "plots" / name / f).exists() and published_same(SITE / "plots" / name / f, ROOT / name / "plots" / f)
                   for f in [p.name for p in (ROOT / name / "plots").glob("*.html")])
        check("website", f"{name} dashboards = the project files (website orange theme, nothing else changed)", same)
    latent_ok = all((SITE / "plots" / "latent" / p.name).exists() for p in (ROOT / "latent" / "plots").glob("*.html"))
    check("website", "latent dashboards published", latent_ok)
    rep = json.loads((ROOT / "latent" / "latent_report.json").read_text(encoding="utf-8"))
    check("website", "latent results shown", rep["selected"] in text)
    check("website", "latent merge animation has data", '"views"' in text)
    for name in DISEASES:
        same = all((SITE / "plots" / name / "inprocessing" / f).exists() and
                   published_same(SITE / "plots" / name / "inprocessing" / f, ROOT / name / "plots" / "inprocessing" / f) for f in IN_FILES)
        check("website", f"{name} in-processing dashboards published", same)
        sel = json.loads((ROOT / name / "inprocessed" / "selection.json").read_text(encoding="utf-8"))
        check("website", f"{name} in-processing results shown", all(v in text for v in sel["selected"].values()))


def main():
    rows = {}
    for name, n in DISEASES.items():
        rows[name] = check_disease(name, n)
    hosp_rows = check_hospital()
    check_latent(rows, hosp_rows)
    for name in DISEASES:
        check_inprocessing(name)
    check_website()
    failed = [r for r in results if not r["ok"]]
    lines = ["# Recheck report", "", f"{len(results) - len(failed)} of {len(results)} checks passed.", "",
             "| Area | Check | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {r['area']} | {r['check']} | {'PASS' if r['ok'] else '**FAIL**'} | {r['detail']} |" for r in results]
    (ROOT / "RECHECK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed; report: {ROOT / 'RECHECK_REPORT.md'}")


if __name__ == "__main__":
    main()
