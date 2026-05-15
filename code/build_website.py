"""Rebuild the Team-18 project website (D:/Team-18/Team-18/index.html) from the current results.

    python code/build_website.py

Writes the team's sections (motivation, research gaps, objectives, problem statement; text of the Review-4 presentation
PPT_changed_in__Copy__2, user request 2026-11-14) and builds datasets, methods, plots and latent results from the real outputs in D:/complete.
Copies every individual bias dashboard to Team-18/plots/<dataset>/ and the latent dashboards to Team-18/plots/latent/.
"""
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = Path("D:/Team-18/Team-18")
OLD_PAGE = SITE / "index.html"
# Website theme: bright, vivid colours on white (user request 2026-11-13). Dataset colours: validated categorical set
# (dataviz validator, light mode, all pairs: lightness, chroma, normal-vision PASS; CVD 6.9 -> scatter also uses shapes).
DATASETS = {
    "heart": {"title": "Heart Disease", "icon": "&#10084;&#65039;", "color": "#e8457c", "source": "CDC BRFSS 2022",
              "target": "Ever told: heart attack or coronary heart disease / angina (self-report)",
              "url": "https://www.cdc.gov/brfss/annual_data/annual_2022.html"},
    "diabetes": {"title": "Diabetes", "icon": "&#129656;", "color": "#e09400", "source": "CDC NHANES Aug 2021 - Aug 2023",
                 "target": "Doctor-diagnosed diabetes (self-report)",
                 "url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023"},
    "hypertension": {"title": "Hypertension", "icon": "&#129658;", "color": "#3d8bfd", "source": "CDC NHIS 2023 Sample Adult",
                     "target": "Clinician-diagnosed hypertension (self-report)",
                     "url": "https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html"},
    "respiratory": {"title": "Respiratory ICU", "icon": "&#129729;", "color": "#0e8f80", "source": "PhysioNet eICU-CRD v2.0",
                    "target": "In-hospital death after the first 24 h in ICU",
                    "url": "https://physionet.org/content/eicu-crd/2.0/"},
}
# Fifth view of the latent space and the data of the in-processing cross-hospital bias (D:/complete/hospital).
# Fifth colour of the validated set.
HOSPITAL_INFO = {"title": "Cross-hospital ICU", "icon": "&#127973;", "color": "#c04ad6", "source": "PhysioNet eICU-CRD v2.0 + MIMIC-IV v3.1",
                 "target": "Recorded heart, diabetes, hypertension and respiratory conditions (four labels)",
                 "url": "https://physionet.org/content/mimiciv/3.1/"}


def hospital_facts():
    base = ROOT / "hospital" / "preprocessed"
    if not base.exists():
        return None
    cols = ["meta_source", "meta_domain_hospitalid", "target_heart", "target_diabetes", "target_hypertension", "target_respiratory"]
    df = pd.concat([pd.read_csv(base / f"hospital_{s}_preprocessed.csv", usecols=cols, low_memory=False).assign(split=s)
                    for s in ("train", "validation", "test", "external")], ignore_index=True)
    eicu = df[df.meta_source == "eicu"]
    feats = json.loads((base / "feature_names.json").read_text(encoding="utf-8"))["output_features"]
    return {"rows": len(df), "eicu": len(eicu), "mimic": int((df.meta_source == "mimic").sum()),
            "hospitals": int(eicu.meta_domain_hospitalid.nunique()) + 1, "predictors": len(feats),
            "positive": {c.replace("target_", ""): float(df[c].mean()) for c in cols if c.startswith("target_")},
            "splits": df.split.value_counts().to_dict()}


BIASES = [  # key in pipeline_log, dashboard file, baseline methods, candidate methods, metrics
    ("Missing data", "01_missing_data_bias_dashboard.html", "GMM, Random Forest, GAIN, MICE, KNN", "HyperImpute, CSDI, MIP-VAE",
     "Wasserstein, correlation preservation, MAE, RMSE, NRMSE, CRPS, NLL"),
    ("Class imbalance", "02_class_imbalance_bias_dashboard.html", "SMOTE, ADASYN, Borderline-SMOTE, NearMiss",
     "TabDDPM, CTGAN, Class-Balanced Focal Loss, DeepSMOTE", "Imbalance ratio, class entropy, balanced F1, macro-F1, balanced accuracy, PR-AUC, G-mean, KS, TVD"),
    ("Outlier", "03_outlier_bias_dashboard.html", "Isolation Forest, LOF, One-Class SVM", "ECOD, Deep SVDD, EBM",
     "Score stability, neighbourhood consistency, AUROC, AUPRC, Precision@K, F1, Brier"),
    ("Feature scale", "04_feature_scale_bias_dashboard.html", "Quantile, Min-Max, Robust scaler", "SNN (SELU), FT-Transformer, Yeo-Johnson",
     "Wasserstein, correlation preservation, skewness reduction, kurtosis, activation variance, gradient stability"),
    ("Demographic", "05_demographic_bias_dashboard.html", "Optimal Transport, FairSMOTE, Reweighting",
     "Adversarial debiasing, DiCE counterfactuals, fair contrastive learning", "SPD, disparate impact, equal opportunity, equalized odds, counterfactual validity, proximity, sparsity"),
    ("Duplicate", "06_duplicate_bias_dashboard.html", "GBER, fuzzy matching, exact removal", "Ditto, GNN entity resolution, LSH",
     "Duplicate rate, redundancy reduction, precision, recall, F1, pair completeness, reduction ratio"),
    ("Data noise", "07_data_noise_bias_dashboard.html", "Denoising AE, wavelet, median, Savitzky-Golay", "Confident Learning, robust (GCE) loss, diffusion denoising",
     "Residual variance, spectral distortion, MAE, RMSE, SNR, label-error F1"),
    ("Encoding / domain", "08_encoding_domain_bias_dashboard.html", "DANN, CORAL, autoencoder", "IRM, OT domain adaptation, DG transformer",
     "MMD, domain-classifier AUC, Wasserstein, worst-group accuracy, DG gap, target AUROC, OOD performance"),
]
# Written sections: text of the Review-4 presentation (user request 2026-11-14; Research Theme removed, Literature Survey removed 2026-11-13).
WRITTEN_SECTIONS = '''<section class="reveal" id="motivation"><div class="container"><span class="s-label">Background &amp; context</span><h2 class="s-title">Motivation</h2>
<div class="grid-2">
<div class="card"><div class="card-h">Biased, demographically skewed data</div><p class="card-p">Healthcare AI systems are frequently developed using datasets that carry sampling bias and are demographically skewed, which restricts fairness and equitable performance across diverse patient populations.</p></div>
<div class="card"><div class="card-h">Accuracy-only optimisation</div><p class="card-p">Conventional machine learning models primarily optimize for overall accuracy, often neglecting performance disparities affecting underrepresented and vulnerable groups.</p></div>
</div></div></section>

<section class="reveal" id="gaps"><div class="container"><span class="s-label">Unresolved challenges</span><h2 class="s-title">Research Gaps</h2>
<div class="grid-2 two">
<div class="card"><div class="card-h">Gap 1 &middot; Stage-isolated bias handling</div><p class="card-p">Most bias mitigation techniques operate only at a single level (data or algorithm or output), and do not analyze how bias propagates across the entire ML pipeline.</p></div>
<div class="card"><div class="card-h">Gap 2 &middot; Lack of clinical risk awareness</div><p class="card-p">Fairness metrics optimize statistical parity but ignore that some errors have medical consequences that are clinically more harmful than others.</p></div>
<div class="card"><div class="card-h">Gap 3 &middot; Absence of continuous fairness monitoring</div><p class="card-p">Existing studies evaluate fairness offline on retrospective datasets and lack an integrated auditing mechanism that monitors demographic performance and detects fairness drift after deployment.</p></div>
<div class="card"><div class="card-h">Gap 4 &middot; Human&ndash;AI interaction neglect</div><p class="card-p">Algorithmic fairness research rarely considers decision-level bias arising from clinician interpretation.</p></div>
</div></div></section>

<section class="reveal" id="objectives"><div class="container"><span class="s-label">Core project goals</span><h2 class="s-title">Objectives</h2>
<div class="glass" style="margin-top:24px"><div class="card-h">Overall problem</div><p class="card-p" style="font-size:1rem">Healthcare AI models trained on real-world clinical data often exhibit uneven predictive reliability across patient groups, resulting in unfair and unsafe clinical decision support. Existing solutions typically address these issues at isolated stages of the learning pipeline, limiting robustness for high-risk patient outcomes.</p></div>
<div class="grid-2 two">
<div class="card"><div class="card-h">Objective 1</div><p class="card-p">Design a healthcare AI framework incorporating equitable data collection and pre-processing techniques to reduce data-related bias and unfair clinical decision outcomes.</p></div>
<div class="card"><div class="card-h">Objective 2</div><p class="card-p">Implement in-processing bias mitigation strategies within the learning pipeline to improve fairness in healthcare AI models.</p></div>
<div class="card"><div class="card-h">Objective 3</div><p class="card-p">Apply post-processing bias mitigation techniques to reduce prediction disparities and improve fairness across patient demographics.</p></div>
<div class="card"><div class="card-h">Objective 4</div><p class="card-p">Incorporate clinical risk-aware learning and privacy-preserving fairness mechanisms to minimize high-risk misclassification errors and enable local (on-device) model operation without exposing patient data.</p></div>
</div></div></section>

<section class="reveal" id="problem"><div class="container"><span class="s-label">Formal problem definition</span><h2 class="s-title">Problem Statement</h2>
<div class="card" style="margin-top:24px;background:rgba(192,74,214,0.03);border-color:rgba(192,74,214,0.2);"><div class="card-h">Core problem</div>
<p class="card-p" style="font-size:1.02rem;color:var(--ink);line-height:1.75;">To develop a clinically risk-aware, multi-stage bias mitigation framework that ensures fairness and consistent predictive performance across diverse patient populations, reducing unfair and unsafe clinical decision support.</p></div>
</div></section>'''
PRE_METHODS = sum(len(t.split(", ")) + len(m.split(", ")) for _, _, t, m, _ in BIASES)          # 53
INPROC = [  # family, title, methods, metrics, plain-language aim, dashboard file   (inprocessing.png)
    ("demographic", "Demographic prediction dependence", "FairCare (adversarial fair representation), DANN, GroupDRO, Fair-ERM",
     "CMI, PAD, WGER, EOV", "Once the true outcome is known, the prediction should carry no information about sex or race.",
     "01_demographic_prediction_dependence_dashboard.html"),
    ("intersectional", "Intersectional / subgroup unfairness", "FOMO, intersectional debiasing, fairness-constrained optimization, GroupDRO (intersectional)",
     "HVI, DF-&epsilon;, MSFV, IWGR", "Small combined groups (sex &times; age &times; race) must not be left behind by a model that is fair only on average.",
     "02_intersectional_subgroup_unfairness_dashboard.html"),
    ("calibration", "Predictive miscalibration", "AvUC, CRL, uncertainty-weighted gradient training, calibration-regularized training",
     "UCE, AURC, E-AURC, MMCE", "A predicted 70% risk should mean that about 70% of such patients have the outcome, and confident predictions should be right more often.",
     "03_predictive_miscalibration_dashboard.html"),
    ("environment", "Cross-hospital generalization failure", "Fishr, V-REx, EQRM, Fish", "IEGVD, ERV, UQER, IDGA",
     "The model should work as well in every hospital, state or region as in the ones it saw most often.",
     "04_cross_environment_generalization_dashboard.html"),
]
IN_METHODS = sum(len(m.split(", ")) for _, _, m, *_ in INPROC)                                 # 16
LATENT_DASH = [("CKA alignment", "01_cka_alignment_dashboard.html"), ("Macro-F1 preservation", "02_macro_f1_preservation_dashboard.html"),
               ("Silhouette structure", "03_silhouette_structure_dashboard.html"), ("NMI / ARI domain invariance", "04_nmi_ari_domain_invariance_dashboard.html")]


def num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def improved(m, tol=0.002):
    """True when the selected method beats ERM on a metric by more than 0.2 % (relative); ties count as not better."""
    if m["erm"] is None or m["sel"] is None:
        return False
    gain = (m["sel"] - m["erm"]) if m["dir"] == "max" else (m["erm"] - m["sel"])
    return gain > tol * max(abs(m["erm"]), abs(m["sel"]), 1e-12)


def inproc_facts(name):
    """Selected in-processing method per bias and its test-split metrics vs ERM; None while the notebook has not run."""
    out = ROOT / name / "inprocessed"
    if not ((out / "selection.json").exists() and (out / "inprocessing_summary.csv").exists()):
        return None
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out / "inprocessing_summary.csv", index_col=0)
    fams = []
    for key, *_, dash in INPROC:
        spec, chosen = sel["families"][key], sel["selected"].get(key)
        if not chosen:
            fams.append({"key": key, "title": spec["title"], "selected": None})
            continue
        row = summary.loc[spec["title"]]
        fams.append({"key": key, "title": spec["title"], "selected": chosen, "methods": spec["methods"],
                     "metrics": [{"name": m, "dir": d, "erm": num(row[f"{m} ERM"]), "sel": num(row[f"{m} selected"])} for m, d in spec["metrics"].items()],
                     "auroc": [num(row["AUROC ERM"]), num(row["AUROC selected"])], "dashboard": f"plots/{name}/inprocessing/{dash}",
                     "extra": [{"name": k.replace(" (external)", ""), "dir": "max", "erm": num(row[f"{k} ERM"]), "sel": num(row[f"{k} selected"]),
                                "note": "external hospital system" if "MIMIC" in k else "unseen eICU hospitals"}
                               for k in ("AUROC eICU test hospitals", "AUROC MIMIC (external)")
                               if key == "environment" and f"{k} ERM" in row.index]})
    return {"families": fams, "erm_validation_auroc": num(sel.get("erm_validation_auroc"))}


def latent_scatter():
    """The same validation records in every latent method's 2-D view; each view is rigidly rotated onto the PCA view
    (orthogonal Procrustes: rotation/reflection + uniform scale, distances between records are unchanged)."""
    path = ROOT / "latent" / "latent_scatter.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    base_name = "PCA (no alignment)"
    pts = {m: np.c_[v["x"], v["y"]].astype(float) for m, v in s["methods"].items()}
    ref = pts[base_name] - pts[base_name].mean(0)
    ref /= np.linalg.norm(ref)
    views = {}
    for m, p in pts.items():
        c = p - p.mean(0)
        c /= np.linalg.norm(c)
        u, _, vt = np.linalg.svd(c.T @ ref)
        v = c @ (u @ vt)
        # each view is uniformly scaled so 98 % of records fill the canvas; the few most extreme records are pinned to the
        # edge (scaling by the single most extreme record shrank all other records into a small blob)
        views[m] = np.clip(v / np.quantile(np.abs(v), 0.98), -1.06, 1.06)
    first = s["methods"][base_name]
    perm = np.random.default_rng(0).permutation(len(first["dataset"]))      # interleaved drawing order: no cohort hides the rest
    info = {**DATASETS, "hospital": HOSPITAL_INFO}
    style = {"heart": ("#e8457c", "circle"), "diabetes": ("#e09400", "square"), "hypertension": ("#3d8bfd", "triangle"),
             "respiratory": ("#0e8f80", "diamond"), "hospital": ("#c04ad6", "cross")}   # dataset colours + a shape (second cue)
    return {"datasets": [info[d]["title"] for d in s["datasets"]], "colors": [style[d][0] for d in s["datasets"]],
            "shapes": [style[d][1] for d in s["datasets"]],
            "dataset": [first["dataset"][i] for i in perm], "label": [first["label"][i] for i in perm], "selected": s["selected"],
            "order": [base_name] + [m for m in s["methods"] if m != base_name],
            "views": {m: v[perm].round(4).tolist() for m, v in views.items()}}


def selection_summary(name):
    """Per detected bias: the selected method, the next-best candidate and the best baseline with their mean ranks
    (written by plots_comparison/make_method_comparison.py from the notebooks' saved comparison tables)."""
    p = ROOT / name / "plots_comparison" / "method_comparison_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def dataset_facts(name):
    raw = pd.read_csv(ROOT / name / "raw" / f"{name}_raw.csv", low_memory=False)
    preds = [c for c in raw.columns if c != "target" and not c.startswith(("meta_", "sens_"))]
    out = ROOT / name / "preprocessed"
    log_path = out / "pipeline_log.json"
    return {"rows": len(raw), "predictors": len(preds), "positive": float(raw.target.mean()),
            "missing": float(raw[preds].isna().to_numpy().mean()),
            "log": json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else None}   # None = notebook still running


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")
PRE_LABELS = {"Duplicate": ("duplicate records", "exact copies"), "Missing data": ("missing cells", "share of predictor cells"),
              "Outlier": ("extreme rows", "share beyond skew-adjusted fences"),
              "Encoding / domain": ("domain-classifier AUC", "0.5 = hospital/site not recognisable"),
              "Class imbalance": ("majority / minority", "1 = balanced classes")}


def numbers(text):
    return [float(x) for x in NUMBER.findall(str(text))]


def pre_rows(key, before, after):
    """The bias statistic of bias_summary.csv as before/after pairs (every one: lower is better)."""
    b, a = numbers(before), numbers(after)
    pick = lambda v, i: v[i] if len(v) > i else None
    if key == "Feature scale":
        rows = [("SD ratio (max / min)", pick(b, 0), pick(a, 0), "1 = predictors on one scale"),
                ("max |skewness|", pick(b, 1), pick(a, 1), "0 = symmetric")]
    elif key == "Data noise":
        rows = [("label-issue rate", pick(b, 0), pick(a, 0), "Confident Learning")]
        if "spikes" in str(before):
            rows.append(("monitor spike rate", pick(b, 1), pick(a, 1), "ICU vital-sign streams"))
    elif key == "Demographic":
        eod = numbers(str(after).split("EOD")[-1]) if "EOD" in str(after) else []
        rows = [("max EOD", pick(b, 1), pick(eod, 0), "equal-opportunity gap over sex, race, age")]
    else:
        name, note = PRE_LABELS[key]
        rows = [(name, pick(b, 0), pick(a, 0), note)]
    return [{"name": n, "dir": "min", "erm": x, "sel": y, "note": note} for n, x, y, note in rows]


def pre_facts(name):
    """Bias statistic before/after the selected method, why it was chosen, and the end-to-end model check."""
    out = ROOT / name / "preprocessed"
    if not (out / "bias_summary.csv").exists():
        return None
    summary = pd.read_csv(out / "bias_summary.csv", index_col=0)
    sel = selection_summary(name)
    sel = {str(r["bias"]).lower(): r for _, r in sel.iterrows()} if sel is not None else {}
    cards = []
    dashboards = {k: f for k, f, *_ in BIASES}
    for key in ("Duplicate", "Missing data", "Outlier", "Data noise", "Feature scale", "Encoding / domain", "Class imbalance", "Demographic"):
        dash = dashboards[key]                                                     # pipeline order, as the notebooks run
        if key not in summary.index:
            continue
        r = summary.loc[key]
        card = {"bias": key, "detected": str(r["detected"]).strip().upper() == "YES", "statistic": str(r["statistic"]),
                "rows": pre_rows(key, r["before"], r["after"]), "selected": str(r["selected method (modern)"]),
                "baseline": str(r["best traditional (comparison only)"]), "compared": int(r["methods compared"]),
                "dashboard": f"plots/{name}/{dash}" if (ROOT / name / "plots" / dash).exists() else ""}
        s = sel.get(key.lower())
        if s is not None and num(s["next_candidate_mean_rank"]) is not None:
            card["rank"] = [num(s["mean_rank"]), num(s["next_candidate_mean_rank"])]
            card.update(next=str(s["next_candidate"]), k=int(s["candidates"]),
                        base=str(s["best_baseline"]) if isinstance(s["best_baseline"], str) else "", base_rank=num(s["best_baseline_mean_rank"]))
        cards.append(card)
    e2e = []
    if (out / "end_to_end_check.csv").exists():
        t = pd.read_csv(out / "end_to_end_check.csv", index_col=0)
        for k, label, d, note in (("AUROC", "AUROC", "max", "discrimination"), ("PR-AUC", "PR-AUC", "max", ""),
                                  ("Balanced accuracy", "Balanced accuracy", "max", ""), ("Brier", "Brier", "min", "probability error"),
                                  ("EOD sex", "EOD sex", "min", "fairness gap"), ("EOD race_ethnicity", "EOD race/ethnicity", "min", "fairness gap"),
                                  ("EOD age_group", "EOD age group", "min", "fairness gap")):
            if k in t.index:
                e2e.append({"name": label, "dir": d, "erm": num(t.loc[k].iloc[0]), "sel": num(t.loc[k].iloc[1]), "note": note})
    return {"cards": cards, "e2e": e2e}

# Bright theme for the published dashboard copies (user request 2026-11-13). The project's plot files in
# D:/complete keep their original colours; only the website copies are recoloured, colour by colour.
DASH_THEME = {
    # before (blue) stays blue but brighter; after (green) -> teal
    "#6FA0DA": "#9DBDFF", "#3E6FA8": "#5B7CFA", "#7BC17E": "#3DD6B0", "#4C9B50": "#0E8F80",
    "#B8BFCC": "#D7D2EE", "#7A8394": "#7A7699",
    # navy header -> violet-to-pink gradient; dark text -> deep indigo
    "#0B1E3F": "#6D3FD9", "#132A52": "#E8457C", "#1B2333": "#1C1745", "#334": "#1C1745", "#556": "#575375",
    "#6B7686": "#575375", "#8791A3": "#7A7699",
    # accent and 'worse'
    "#2563C8": "#6D3FD9", "#D0433F": "#D6336C",
    # dark blue of the six-colour categorical set (the other five are already bright)
    "#1D4FA0": "#6D3FD9",
    # borders and backgrounds -> lavender whites
    "#B7C0D1": "#CFC8EE", "#B9C6E6": "#CFC8EE", "#AEB6C6": "#C3BBE6", "#C9CFDA": "#DDD7F3", "#C8CEDE": "#DDD7F3",
    "#D6DCEA": "#E7E3F7", "#D7DEEC": "#E7E3F7", "#E1E5EE": "#E7E3F7", "#EAF3FF": "#EFEAFF", "#CFE0FF": "#DDD3FF",
    "#EDEFF4": "#F4F2FF", "#EEF1F7": "#F4F2FF", "#EEF0F5": "#F4F2FF", "#F0F2F7": "#F4F2FF", "#F2F4F8": "#F4F2FF",
    "#F3F6FC": "#F8F7FF", "#F4F6FB": "#F8F7FF", "#F6F8FC": "#FAF9FF", "#FBFCFE": "#FDFCFF",
    # reds -> raspberry/coral, greens -> teal, yellows -> sunny
    "#D9534F": "#F0506E", "#B23B37": "#C2185B", "#A8322F": "#C2185B", "#8A1F17": "#8E1340", "#C0533E": "#E8604C",
    "#9B4230": "#B23A48", "#E8836B": "#FF8A70", "#C9624B": "#E86A50", "#F2A49F": "#FFB3C1", "#D9776F": "#F2787F",
    "#FDECEC": "#FFE8EE", "#2E7D32": "#0E8F80", "#1F5E23": "#0A6B60", "#E8F5E9": "#E0F7F3", "#F0B323": "#FFB21F",
    "#C99212": "#E09400",
}
DASH_TEXT = {"rgba(15,23,42,": "rgba(28,23,69,", "rgba(37,99,200,": "rgba(109,63,217,",
             # latent cohorts get exactly the website's five dataset colours
             "const DSC=['#2563C8','#D0433F','#C9A227','#2E9E6B','#7B4FC9'];": "const DSC=['#e8457c','#e09400','#3d8bfd','#0e8f80','#c04ad6'];",
             "Blue = before (per-dataset PCA), green = after (latent method).": "Blue = before (per-dataset PCA), teal = after (latent method)."}
_HEX = re.compile(r"(?<![&\w])#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])")       # not &#128273; emoji codes


# Website wording (user request 2026-11-14): methods that can be applied = candidate methods, comparison-only methods =
# baseline methods; latent methods are all 'latent method' (PCA stays the baseline); no 'Generated <date>' line.
WORDS = [('"values":["Traditional",', '"values":["Latent method",'), ('"values":["Modern",', '"values":["Latent method",'),
         ('"columns":["family",', '"columns":["type",'),
         ("only <b>${esc(d.select_family)}</b> methods can be selected", "only the <b>candidate</b> methods can be selected")]
_WORD = re.compile(r"\b(Traditional|traditional|TRADITIONAL|Modern|modern|MODERN)\b")
_SWAP = {"Traditional": "Baseline", "traditional": "baseline", "TRADITIONAL": "BASELINE",
         "Modern": "Candidate", "modern": "candidate", "MODERN": "CANDIDATE"}
_GENERATED = re.compile(r',\s*\["Generated",\s*"[^"]*"\]|Generated \d{1,2}-[A-Za-z]{3}-\d{4} \u00b7 ')


def theme_dashboard(text):
    for old, new in DASH_TEXT.items():                     # text first: the cohort list is replaced as a whole
        text = text.replace(old, new)
    for old, new in WORDS:
        text = text.replace(old, new)
    text = _GENERATED.sub("", _WORD.sub(lambda m: _SWAP[m.group(0)], text))
    return _HEX.sub(lambda m: DASH_THEME.get(m.group(0).upper(), m.group(0)), text)


def copy_plots():
    """Copy the dashboards that exist now over the published copies, in place. Nothing is deleted: on Windows an editor
    or antivirus can hold a published folder open, and deleting it would stop the build."""
    for name in list(DATASETS) + ["latent"]:
        src, dest = ROOT / name / "plots", SITE / "plots" / name
        dest.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        for f in src.rglob("*"):
            out = dest / f.relative_to(src)
            if f.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif f.suffix == ".html":                                            # dashboards: orange + white theme
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(theme_dashboard(f.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)


def comparison_gallery():
    """Copy the method-comparison plots of <dataset>/plots_comparison (user request 2026-11-13) to the site and describe
    them for the comparison section. The 10-record walkthroughs are no longer shown (user request 2026-11-14). Nothing is deleted."""
    out = {}
    for name, d in DATASETS.items():
        src = ROOT / name / "plots_comparison"
        if not src.exists():
            continue
        dest = SITE / "plots" / name / "plots_comparison"
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.glob("0*.png"):
            shutil.copy2(f, dest / f.name)
        rel = f"plots/{name}/plots_comparison/"
        cmp = []
        if (src / "00_overview_applied_vs_all_methods.png").exists():
            cmp.append({"bias": "Overview", "src": rel + "00_overview_applied_vs_all_methods.png",
                        "note": "The selected method has the lowest mean rank of the candidate methods in every detected bias."})
        summ = src / "method_comparison_summary.csv"
        if summ.exists():
            for _, r in pd.read_csv(summ).iterrows():
                cmp.append({"bias": r["bias"], "src": rel + r["file"],
                            "note": f"{d['title']} · {r['bias']} · selected: {r['applied']} · lowest rank total of the {r['candidates']} candidate methods "
                                    f"({r['rank_total']:g}, mean rank {r['mean_rank']:.2f}; next candidate {r['next_candidate']} {r['next_candidate_rank_total']:g})"
                                    + (f" · tied with {r['tie_with']}, the first in the notebook's list is applied" if isinstance(r["tie_with"], str) and r["tie_with"] else "")
                                    + (f" · all {r['methods']} methods shown; baseline methods are run for comparison only")})
        out[d["title"]] = {"color": d["color"], "cmp": cmp}
    return out


def main():
    facts = {n: dataset_facts(n) for n in DATASETS}
    infacts = {n: inproc_facts(n) for n in DATASETS}
    prefacts = {n: pre_facts(n) for n in DATASETS}
    copy_plots()
    latent = json.loads((ROOT / "latent" / "latent_report.json").read_text()) if (ROOT / "latent" / "latent_report.json").exists() else None
    hosp = hospital_facts()
    total_rows = sum(f["rows"] for f in facts.values()) + (hosp["rows"] if hosp else 0)

    ds_cards = "".join(f"""
<div class="ds-card reveal" style="--accent:{d['color']}"><div class="ds-icon">{d['icon']}</div><h3>{d['title']}</h3>
<p class="ds-target">{html.escape(d['target'])}</p>
<div class="ds-stats"><div><b class="count" data-to="{facts[n]['rows']}">0</b><span>records</span></div><div><b class="count" data-to="{facts[n]['predictors']}">0</b><span>predictors</span></div>
<div><b>{100 * facts[n]['positive']:.1f}%</b><span>positive class</span></div><div><b>{100 * facts[n]['missing']:.1f}%</b><span>missing cells</span></div></div>
<div class="bar-label"><span>positive share of records</span><span>{100 * facts[n]['positive']:.1f}%</span></div>
<div class="bar" title="{d['title']}: {100 * facts[n]['positive']:.1f}% of records are positive"><i style="width:{100 * facts[n]['positive']:.1f}%"></i></div></div>""" for n, d in DATASETS.items())
    if hosp:
        h, pos = HOSPITAL_INFO, hosp["positive"]
        ds_cards += f"""
<div class="ds-card hosp reveal" style="--accent:{h['color']}"><div class="hc-a"><div class="ds-icon">{h['icon']}</div><h3>{h['title']} cohort <small class="tag-mini">latent + in-processing</small></h3>
<p class="ds-target">{html.escape(h['target'])}. It provides the hospitals for the cross-hospital bias: models train on eICU hospitals and are checked on unseen eICU hospitals and on MIMIC-IV, a different hospital system.</p></div>
<div class="hc-b"><div class="ds-stats"><div><b class="count" data-to="{hosp['rows']}">0</b><span>ICU stays</span></div><div><b class="count" data-to="{hosp['hospitals']}">0</b><span>hospitals</span></div>
<div><b class="count" data-to="{hosp['eicu']}">0</b><span>eICU stays (train / val / test hospitals)</span></div><div><b class="count" data-to="{hosp['mimic']}">0</b><span>MIMIC-IV stays (external)</span></div></div></div>
<div class="hc-c">{"".join(f'<div class="bar-label"><span>recorded {k}</span><span>{100 * v:.1f}%</span></div><div class="bar" title="{k}: {100 * v:.1f}% of stays"><i style="width:{100 * v:.1f}%"></i></div>' for k, v in pos.items())}</div></div>"""

    def cell(n, key):
        if facts[n]["log"] is None:
            return '<td><span class="pill run">notebook running&hellip;</span></td>'
        b = facts[n]["log"]["biases"].get(key, {})
        if not b.get("detected"):
            return '<td><span class="pill off">not detected</span></td>'
        return f'<td><span class="pill on">{html.escape(str(b.get("selected", "-")))}</span><small>best baseline: {html.escape(str(b.get("best_traditional", "-")))}</small></td>'
    matrix = "".join(f"<tr><td><b>{k}</b><small>{m}</small></td>" + "".join(cell(n, k) for n in DATASETS) + "</tr>"
                     for k, _, t, m, _ in BIASES)
    bias_cards = "".join(f"""
<div class="bias-card reveal"><div class="bias-num">{i + 1:02d}</div><h3>{k} bias</h3>
<div class="m-row"><span class="tag base">Baseline</span><p>{t}</p></div><div class="m-row"><span class="tag cand">Candidate</span><p>{m}</p></div>
<div class="m-row"><span class="tag met">Metrics</span><p>{mt}</p></div></div>""" for i, (k, _, t, m, mt) in enumerate(BIASES))

    why_rows = []
    for n, d in DATASETS.items():
        w = selection_summary(n)
        if w is None:
            continue
        for _, r in w.iterrows():
            tie = " (tie)" if isinstance(r["tie_with"], str) and r["tie_with"] else ""
            why_rows.append(f"<tr><td>{d['title']}</td><td>{html.escape(str(r['bias']))}</td><td><b>{html.escape(str(r['applied']))}</b></td><td><b>{r['mean_rank']:.2f}</b></td>"
                            f"<td>{html.escape(str(r['next_candidate']))}</td><td>{r['next_candidate_mean_rank']:.2f}{tie}</td>"
                            f"<td>{html.escape(str(r['best_baseline']))}</td><td>{r['best_baseline_mean_rank']:.2f}</td></tr>")
    explorer_map = {"Pre-processing": {}, "Latent space": {}, "In-processing": {}}
    known = {f for _, f, *_ in BIASES} | {"index.html"}
    for n, d in DATASETS.items():
        # "" = dashboard not produced yet (the explorer shows a 'still running' card instead of a broken frame)
        entry = {f"{k} bias": (f"plots/{n}/{f}" if (ROOT / n / "plots" / f).exists() else "") for k, f, *_ in BIASES}
        for extra in sorted((ROOT / n / "plots").glob("*.html")):          # e.g. respiratory monitor-signal noise
            if extra.name not in known:
                label = re.sub(r"^\d+\w?_", "", extra.stem).replace("_dashboard", "").replace("_bias", "").replace("_", " ")
                entry[label[:1].upper() + label[1:]] = f"plots/{n}/{extra.name}"
        explorer_map["Pre-processing"][d["title"]] = entry
        inf = infacts[n]
        explorer_map["In-processing"][d["title"]] = {
            (f["title"]): (f["dashboard"] if f.get("selected") and (SITE / f["dashboard"]).exists() else "")
            for f in (inf["families"] if inf else [{"title": t} for _, t, *_ in INPROC]) if not (inf and f.get("selected") is None)}
    explorer_map["Latent space"]["All five cohorts"] = {k: (f"plots/latent/{f}" if (ROOT / "latent" / "plots" / f).exists() else "")
                                                         for k, f in LATENT_DASH}

    # ---------------------------------------------------------------- in-processing section
    in_cards = "".join(f"""
<div class="bias-card in-card reveal"><div class="bias-num">{i + 1:02d}</div><h3>{t}</h3><p class="aim">{aim}</p>
<div class="m-row"><span class="tag cand">Methods</span><p>{m}</p></div><div class="m-row"><span class="tag met">Metrics</span><p>{mt}</p></div></div>"""
                       for i, (_, t, m, mt, aim, _) in enumerate(INPROC))

    def in_cell(n, key):
        inf = infacts[n]
        if inf is None:
            return '<td><span class="pill run">notebook running&hellip;</span></td>'
        f = next(x for x in inf["families"] if x["key"] == key)
        if not f.get("selected"):
            return '<td><span class="pill off">not applicable</span><small>no hospital / region identifier</small></td>'
        better = sum(1 for m in f["metrics"] if improved(m))
        return f'<td><span class="pill vio">{html.escape(f["selected"])}</span><small>{better} of {len(f["metrics"])} metrics better than ERM (test)</small></td>'
    in_matrix = "".join(f"<tr><td><b>{t}</b><small>{mt}</small></td>" + "".join(in_cell(n, k) for n in DATASETS) + "</tr>"
                        for k, t, _, mt, *_ in INPROC)
    in_json = {DATASETS[n]["title"]: (dict(infacts[n], color=DATASETS[n]["color"]) if infacts[n] else None) for n in DATASETS}
    pre_json = {DATASETS[n]["title"]: (dict(prefacts[n], color=DATASETS[n]["color"]) if prefacts[n] else None) for n in DATASETS}
    scatter = latent_scatter()
    if latent and scatter:
        for m in scatter["order"]:
            r = latent["methods"].get(m, {})
            scatter.setdefault("metrics", {})[m] = {k: num(r.get(k)) for k in ("CKA (alignment)", "Macro-F1 (preservation)", "NMI (domain)", "ARI (domain)")}

    latent_html = ('<div class="glass reveal"><h3>Measured results</h3><p class="s-desc">The latent stage runs as soon as the four '
                   'preprocessing notebooks have finished; its measured CKA, Macro-F1, Silhouette and NMI/ARI values and the four '
                   'metric dashboards appear here automatically.</p></div>')
    if latent:
        sel = latent["selected"]
        rows = "".join(f"<tr class='{'sel' if m == sel else ''}'><td>{m}{' &#9733;' if m == sel else ''}</td>"
                       + "".join(f"<td>{r[k]:.3f}</td>" for k in ("CKA (alignment)", "Macro-F1 (preservation)", "Silhouette (structure)", "NMI (domain)", "ARI (domain)"))
                       + "</tr>" for m, r in latent["methods"].items())
        bars = "".join(f"<div class='lbar' title='{m}: mean cross-dataset CKA {r['CKA (alignment)']:.3f}'><span>{m}</span>"
                       f"<div><i style='width:{100 * max(r['CKA (alignment)'], 0):.0f}%'></i></div><b>{r['CKA (alignment)']:.3f}</b></div>"
                       for m, r in latent["methods"].items())
        latent_html = f"""<div class="glass reveal"><h3>Measured results (validation) &mdash; selected: <span class="grad">{sel}</span></h3>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>CKA &uarr;</th><th>Macro-F1 &uarr;</th><th>Silhouette &uarr;</th><th>NMI &darr;</th><th>ARI &darr;</th></tr></thead><tbody>{rows}</tbody></table></div>
<h4>Cross-dataset alignment (CKA)</h4>{bars}<div class="plot-links">
<a href="plots/latent/01_cka_alignment_dashboard.html" target="_blank">CKA dashboard</a><a href="plots/latent/02_macro_f1_preservation_dashboard.html" target="_blank">Macro-F1 dashboard</a>
<a href="plots/latent/03_silhouette_structure_dashboard.html" target="_blank">Silhouette dashboard</a><a href="plots/latent/04_nmi_ari_domain_invariance_dashboard.html" target="_blank">NMI / ARI dashboard</a></div></div>"""

    # records flowing through the pipe: rose (still biased) -> cyan (repaired by pre-processing) -> violet (in-processing model)
    stream_dots = "".join(
        f'<circle r="{5 + (i % 3)}" fill="#e8457c" fill-opacity=".95" stroke="#ffffff" stroke-width="1.5"><animateMotion dur="8s" begin="-{i * 0.57:.2f}s" repeatCount="indefinite">'
        f'<mpath href="#flowPath"/></animateMotion><animate attributeName="fill" values="#e8457c;#e8457c;#0e8f80;#0e8f80;#6d3fd9;#6d3fd9" '
        f'keyTimes="0;0.18;0.36;0.55;0.75;1" dur="8s" begin="-{i * 0.57:.2f}s" repeatCount="indefinite"/></circle>' for i in range(14))
    if not why_rows:
        why_rows = ['<tr><td colspan="8"><span class="pill run">notebooks running&hellip;</span> the comparison appears when each dataset finishes</td></tr>']
    page = TEMPLATE
    for key, value in {"__TOTAL__": f"{total_rows:,}", "__DS_CARDS__": ds_cards, "__MATRIX__": matrix, "__BIAS_CARDS__": bias_cards,
                       "__WHY__": "".join(why_rows), "__EXPLORER__": json.dumps(explorer_map), "__LATENT__": latent_html,
                       "__KEEP__": WRITTEN_SECTIONS, "__TOTAL_RAW__": str(total_rows),
                       "__STREAM_DOTS__": stream_dots, "__IN_CARDS__": in_cards, "__IN_MATRIX__": in_matrix,
                       "__CMP_JSON__": json.dumps(comparison_gallery(), allow_nan=False).replace("</", "<\\/"),
                       "__IN_JSON__": json.dumps(in_json, allow_nan=False).replace("</", "<\\/"),
                       "__PRE_JSON__": json.dumps(pre_json, allow_nan=False).replace("</", "<\\/"),
                       "__SCATTER_JSON__": json.dumps(scatter, allow_nan=False).replace("</", "<\\/"),
                       "__PRE_N__": str(PRE_METHODS), "__IN_N__": str(IN_METHODS), "__ALL_N__": str(PRE_METHODS + 3 + IN_METHODS)}.items():
        page = page.replace(key, value)
    OLD_PAGE.write_text(page, encoding="utf-8")
    print("website written:", OLD_PAGE)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare Bias Mitigation | Group 18</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#ffffff;--ink:#1c1745;--ink2:#575375;--line:#e7e3f7;--tint:#f7f5ff;
--pink:#e8457c;--amber:#e09400;--blue:#3d8bfd;--teal:#0e8f80;--orchid:#c04ad6;--violet:#6d3fd9;--coral:#ff6a3d;--sun:#ffc21a;--mint:#2ed3a5;
--txt:var(--ink);--txt2:var(--ink2);--border:var(--line);--card:#fff;--bg2:var(--tint);--cyan:var(--teal);--green:var(--teal);--sec:var(--violet);
--grad:linear-gradient(100deg,#ff4f8b,#ff8a3d 22%,#ffc21a 40%,#2ed3a5 60%,#3d8bfd 80%,#8e44ec);
--grad-text:linear-gradient(100deg,#e8457c,#e0560f 28%,#8e2fd0 62%,#2f6ff0);
--grad-btn:linear-gradient(110deg,#e8457c,#a83be0 55%,#5b5cf0);
--shadow:0 12px 32px rgba(28,23,69,.07)}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--ink);line-height:1.6;overflow-x:hidden}
body:before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;background:
 radial-gradient(42vw 42vw at 0% 0%,rgba(255,79,139,.09),transparent 62%),radial-gradient(40vw 40vw at 100% 18%,rgba(61,139,253,.09),transparent 62%),
 radial-gradient(46vw 46vw at 30% 100%,rgba(46,211,165,.09),transparent 62%),radial-gradient(36vw 36vw at 100% 100%,rgba(255,194,26,.10),transparent 62%)}
a{color:#2f5fd8}h1,h2,h3,h4{font-family:Outfit,Inter,sans-serif;margin:.2em 0 .5em;color:var(--ink)}
.container{max-width:1240px;margin:0 auto;padding:0 22px}
nav{position:fixed;top:0;left:0;right:0;z-index:50;background:rgba(255,255,255,.88);backdrop-filter:blur(16px)}
nav:after{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:var(--grad)}
nav .container{display:flex;align-items:center;justify-content:center;height:62px}
.logo{font-family:Outfit;font-weight:800;font-size:1.18rem;background:var(--grad-text);-webkit-background-clip:text;color:transparent}
.nav-links{display:flex;gap:4px;flex-wrap:wrap}.nav-links a{color:var(--ink2);text-decoration:none;font-size:.82rem;font-weight:600;padding:6px 11px;border-radius:999px;transition:.25s}
.nav-links a:hover,.nav-links a.active{color:#fff;background:var(--grad-btn);box-shadow:0 6px 16px rgba(168,59,224,.25)}
@media(max-width:980px){nav .container{justify-content:flex-start}.nav-links{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;max-width:100%}}
#hero{position:relative;min-height:100vh;display:flex;align-items:center;overflow:hidden;padding-top:62px;background:#fff}
#hero:before,#hero:after{content:"";position:absolute;width:520px;height:520px;border-radius:50%;filter:blur(90px);opacity:.30;animation:float 16s ease-in-out infinite}
#hero:before{background:#ffc21a;top:30%;left:38%;animation-delay:-4s}#hero:after{background:#2ed3a5;bottom:-180px;left:-120px;animation-delay:-10s}
#net{position:absolute;inset:0;width:100%;height:100%;z-index:1}
.hero-glow{position:absolute;width:640px;height:640px;border-radius:50%;filter:blur(90px);opacity:.32;animation:float 14s ease-in-out infinite}
.g1{background:#ff4f8b;top:-180px;left:-140px}.g2{background:#3d8bfd;bottom:-200px;right:-160px;animation-delay:-7s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(60px,40px) scale(1.12)}}
.hero-inner{position:relative;z-index:2;max-width:900px}
.badge{display:inline-flex;gap:8px;align-items:center;padding:6px 14px;border-radius:999px;border:1px solid var(--line);background:#fff;font-size:.8rem;color:var(--ink2);font-weight:600;box-shadow:var(--shadow)}
.badge i{width:8px;height:8px;border-radius:50%;background:var(--pink);box-shadow:0 0 12px var(--pink);animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.35}}
.hero-title{font-size:clamp(2.1rem,5vw,3.9rem);line-height:1.08;font-weight:800;margin:18px 0}
.grad{background:var(--grad-text);background-size:200% 200%;-webkit-background-clip:text;color:transparent;animation:shift 6s ease infinite}
@keyframes shift{50%{background-position:100% 50%}}
.hero-sub{color:var(--ink2);font-size:1.08rem;max-width:760px}
.ecg{width:100%;max-width:760px;height:70px;margin:12px 0}.ecg path{fill:none;stroke:url(#ecgGrad);stroke-width:2.8;stroke-linecap:round;stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 3.2s linear infinite}
@keyframes draw{to{stroke-dashoffset:0}}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:26px}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:inset 0 4px 0 var(--sec),var(--shadow);transition:.3s}
.stat:hover{transform:translateY(-4px)}
.stat:nth-child(1){--sec:var(--pink)}.stat:nth-child(2){--sec:var(--blue)}.stat:nth-child(3){--sec:var(--teal)}.stat:nth-child(4){--sec:var(--orchid)}
.stat b{font-family:Outfit;font-size:1.9rem;display:block;color:color-mix(in srgb,var(--sec) 82%,#000)}.stat span{color:var(--ink2);font-size:.82rem}
.btns{display:flex;gap:12px;margin-top:26px;flex-wrap:wrap}.btn{padding:12px 20px;border-radius:999px;text-decoration:none;font-weight:700;font-size:.9rem;transition:.25s}
.btn.p{background:var(--grad-btn);color:#fff;box-shadow:0 12px 28px rgba(168,59,224,.32)}
.btn.s{border:2px solid var(--line);color:var(--ink);background:#fff}.btn.s:hover{border-color:var(--violet);color:var(--violet)}.btn:hover{transform:translateY(-3px)}
section{padding:96px 0;position:relative;background:transparent}section:nth-of-type(even){background:rgba(247,245,255,.75)}
#motivation{--sec:var(--pink)}#gaps{--sec:var(--teal)}#objectives{--sec:var(--blue)}#problem{--sec:var(--orchid)}
#datasets{--sec:var(--blue)}#pipeline{--sec:var(--violet)}#preprocessing{--sec:var(--pink)}#why-selected{--sec:var(--amber)}
#latent{--sec:var(--orchid)}#inprocessing{--sec:var(--teal)}#explorer{--sec:var(--blue)}
.s-label{display:inline-flex;align-items:center;gap:8px;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800;padding:5px 12px;border-radius:999px;
 background:color-mix(in srgb,var(--sec) 13%,#fff);color:color-mix(in srgb,var(--sec) 72%,#000)}
.s-label:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--sec)}
.s-title{font-size:clamp(1.7rem,3vw,2.5rem);font-weight:800}
.s-title:after{content:"";display:block;width:70px;height:5px;border-radius:5px;margin-top:12px;background:linear-gradient(90deg,var(--sec),color-mix(in srgb,var(--sec) 35%,#fff))}
.s-desc{color:var(--ink2);max-width:820px}
.card,.glass{position:relative;background:#fff;border:1px solid var(--line);border-radius:20px;padding:22px;transition:.3s;box-shadow:inset 0 4px 0 var(--sec),var(--shadow)}
.grid-2>.card:nth-child(4n+1),.grid-3>.card:nth-child(4n+1),.grid-4>.card:nth-child(4n+1){--sec:var(--pink)}
.grid-2>.card:nth-child(4n+2),.grid-3>.card:nth-child(4n+2),.grid-4>.card:nth-child(4n+2){--sec:var(--blue)}
.grid-2>.card:nth-child(4n+3),.grid-3>.card:nth-child(4n+3),.grid-4>.card:nth-child(4n+3){--sec:var(--teal)}
.grid-2>.card:nth-child(4n+4),.grid-3>.card:nth-child(4n+4),.grid-4>.card:nth-child(4n+4){--sec:var(--orchid)}
.card:hover{transform:translateY(-5px);box-shadow:inset 0 4px 0 var(--sec),0 22px 44px color-mix(in srgb,var(--sec) 22%,transparent)}
.card-h{font-family:Outfit;font-weight:700;font-size:1.05rem;margin-bottom:6px;color:color-mix(in srgb,var(--sec) 60%,#1c1745)}.card-p{color:var(--ink2);font-size:.92rem}
.grid-2,.grid-3,.grid-4{display:grid;gap:18px;margin-top:24px}.grid-2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.grid-3{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}.grid-4{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.table-wrap{overflow-x:auto;margin-top:18px;border-radius:16px;border:1px solid var(--line);background:#fff}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:linear-gradient(90deg,#6d3fd9,#3b6cf0);font-family:Outfit;font-weight:700;color:#fff;white-space:nowrap}tr:hover td{background:#faf9ff}
td small{display:block;color:var(--ink2);font-size:.72rem;margin-top:3px}tr.sel td{background:#fff4d6;font-weight:600}
.badge-check,.badge-tech{display:inline-block;padding:3px 10px;border-radius:999px;background:#e9e3ff;color:#4c2bb0;font-size:.75rem;font-weight:700}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.76rem;font-weight:700}.pill.on{background:#dcf7ee;color:#076b5e}.pill.off{background:#f1f0f7;color:var(--ink2)}
.ds-card{position:relative;background:#fff;border:1px solid var(--line);border-radius:22px;padding:24px;overflow:hidden;transition:.35s;box-shadow:inset 0 5px 0 var(--accent),var(--shadow)}
.ds-card:before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 12% 0%,var(--accent),transparent 58%);opacity:.13;pointer-events:none}
.ds-card:hover{transform:translateY(-6px) rotate(-.3deg);box-shadow:inset 0 5px 0 var(--accent),0 24px 48px color-mix(in srgb,var(--accent) 25%,transparent)}
.ds-icon{font-size:2.1rem;animation:beat 1.8s ease-in-out infinite;display:inline-block}
@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.18)}30%{transform:scale(1)}}
.ds-card h3{color:color-mix(in srgb,var(--accent) 70%,#1c1745)}
.ds-src{font-size:.82rem;font-weight:600}.ds-target{color:var(--ink2);font-size:.86rem;min-height:44px}
.ds-stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px 14px;margin:14px 0}.ds-stats b{font-family:Outfit;font-size:1.3rem;display:block;line-height:1.2;color:var(--ink)}.ds-stats span{font-size:.74rem;color:var(--ink2)}
.bar-label{display:flex;justify-content:space-between;font-size:.74rem;color:var(--ink2);margin:4px 0 6px}
.bar{height:7px;border-radius:9px;background:#f1effa;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);border-radius:9px;animation:grow 2s ease}
@keyframes grow{from{width:0}}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:center;margin-top:30px}
.node{padding:11px 16px;border-radius:999px;background:#fff;border:2px solid var(--line);font-weight:700;font-size:.84rem;position:relative;animation:glow 4s ease-in-out infinite;color:var(--ink)}
.node.k{border-color:var(--teal);background:#e9faf6}.node.l{border-color:var(--orchid);background:#fbefff}.node.f{opacity:.55;border-style:dashed}
@keyframes glow{50%{box-shadow:0 0 22px rgba(168,59,224,.25)}}
.arrow{width:34px;height:3px;border-radius:3px;background:linear-gradient(90deg,#ffd0df,#c9b8ff);position:relative;overflow:hidden}
.arrow:after{content:"";position:absolute;top:-1px;width:10px;height:5px;border-radius:4px;background:var(--pink);animation:move 1.4s linear infinite}
@keyframes move{from{left:-10px}to{left:34px}}
.bias-card{position:relative;background:#fff;border:1px solid var(--line);border-radius:20px;padding:20px;transition:.3s;box-shadow:inset 0 4px 0 var(--sec),var(--shadow)}
.grid-4>.bias-card:nth-child(4n+1){--sec:var(--pink)}.grid-4>.bias-card:nth-child(4n+2){--sec:var(--amber)}.grid-4>.bias-card:nth-child(4n+3){--sec:var(--blue)}.grid-4>.bias-card:nth-child(4n+4){--sec:var(--teal)}
.bias-card:hover{transform:translateY(-5px);box-shadow:inset 0 4px 0 var(--sec),0 22px 44px color-mix(in srgb,var(--sec) 22%,transparent)}
.bias-num{position:absolute;right:16px;top:12px;font-family:Outfit;font-size:2rem;font-weight:800;opacity:.35;color:var(--sec)}
.m-row{display:flex;gap:10px;align-items:flex-start;margin:8px 0}.m-row p{margin:0;font-size:.84rem;color:var(--ink2)}
.tag{flex:none;font-size:.66rem;font-weight:800;padding:3px 8px;border-radius:7px;text-transform:uppercase;letter-spacing:.06em}
.tag.base{background:#eef0ff;color:#3b46b5}.tag.cand{background:var(--grad-btn);color:#fff}.tag.met{background:#e0f7f3;color:#076b5e}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 12px}.tab{padding:9px 17px;border-radius:999px;border:2px solid var(--line);background:#fff;color:var(--ink);cursor:pointer;font-weight:700;font-size:.85rem;transition:.2s}
.tab.on{background:var(--grad-btn);border-color:transparent;color:#fff;box-shadow:0 8px 20px rgba(168,59,224,.28)}.tab:hover{transform:translateY(-2px);border-color:#c9b8ff}
.biasbtns{display:flex;gap:8px;flex-wrap:wrap}.bb{padding:7px 12px;border-radius:999px;border:2px solid var(--line);background:#fff;color:var(--ink2);cursor:pointer;font-size:.8rem;font-weight:600}
.bb.on{color:#1f4fc0;border-color:var(--blue);background:#eaf2ff}
.frame{margin-top:16px;border-radius:20px;overflow:hidden;border:3px solid transparent;background:linear-gradient(#fff,#fff) padding-box,var(--grad) border-box;height:820px;box-shadow:0 30px 60px rgba(28,23,69,.12)}
.grid-2.two{grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr))}
.cmp-h{margin:26px 0 10px;font-size:1.1rem}.cmp-note{color:var(--ink2);font-size:.88rem;margin:12px 0 0}
.cmp-img{margin-top:12px;border-radius:20px;overflow:hidden;border:3px solid transparent;background:linear-gradient(#fff,#fff) padding-box,var(--grad) border-box;box-shadow:0 30px 60px rgba(28,23,69,.12)}
.cmp-img img{width:100%;display:block}
.frame iframe{width:100%;height:100%;border:0}.frame-bar{display:flex;justify-content:space-between;align-items:center;margin-top:14px;color:var(--ink2);font-size:.85rem}
.lbar{display:grid;grid-template-columns:160px 1fr 60px;gap:10px;align-items:center;margin:8px 0;font-size:.85rem}.lbar div{height:12px;border-radius:9px;background:#f1effa}
.lbar i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,var(--pink),var(--orchid),var(--blue));animation:grow 2.2s ease}
.pill.run{background:#fff1d6;color:#7a4f00}.pill.run:before{content:"";display:inline-block;width:7px;height:7px;margin-right:6px;border-radius:50%;background:var(--amber);animation:pulse 1.2s infinite}
.stream{width:100%;height:120px;margin-top:26px}.stream text{fill:var(--ink2);font-size:12px;font-family:Inter}
.legend-row{display:flex;gap:18px;justify-content:center;flex-wrap:wrap;color:var(--ink2);font-size:.8rem;margin-top:6px}
.legend-row i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.wait{display:flex;align-items:center;justify-content:center;height:100%;background:var(--tint);color:var(--ink2);font-size:1rem;text-align:center;padding:30px}
.plot-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.plot-links a{padding:9px 15px;border-radius:999px;background:var(--grad-btn);text-decoration:none;color:#fff;font-size:.84rem;font-weight:700;box-shadow:0 8px 18px rgba(168,59,224,.22)}
.reveal{opacity:0;transform:translateY(28px);transition:opacity .8s ease,transform .8s ease}.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion: reduce){*,*:before,*:after{animation:none!important;transition:none!important}.reveal{opacity:1;transform:none}.ecg path{stroke-dashoffset:0}}
.ds-card.hosp{grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr);gap:8px 30px;align-items:center}@media(max-width:900px){.ds-card.hosp{grid-template-columns:1fr}}.ds-card.hosp .ds-target{min-height:0}
.tag-mini{font-size:.6rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:#f6e4fb;color:#86259c;vertical-align:middle}
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin-top:28px}
.stage{position:relative;padding:20px;border-radius:20px;border:1px solid var(--line);background:#fff;overflow:hidden;transition:.3s;box-shadow:inset 0 4px 0 var(--sec),var(--shadow)}
.stage:nth-child(1){--sec:var(--pink)}.stage:nth-child(2){--sec:var(--orchid)}.stage:nth-child(3){--sec:var(--teal)}
.stage:hover{transform:translateY(-4px)}
.stage .num{font-family:Outfit;font-size:2.6rem;font-weight:800;opacity:.4;position:absolute;right:14px;top:4px;color:var(--sec)}
.stage h3{margin:.1em 0 .4em;color:color-mix(in srgb,var(--sec) 65%,#1c1745)}.stage p{color:var(--ink2);font-size:.88rem;margin:0}.stage ul{margin:.6em 0 0;padding-left:1.1em;color:var(--ink2);font-size:.82rem}
.stage .run{height:5px;border-radius:5px;margin-top:14px;background:var(--grad);background-size:300% 100%;animation:flowbar 5s linear infinite}
@keyframes flowbar{to{background-position:300% 0}}
.morph-wrap{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);gap:22px;align-items:start;margin-top:22px}@media(max-width:900px){.morph-wrap{grid-template-columns:1fr}}
.morph{position:relative;border-radius:20px;border:1px solid var(--line);background:radial-gradient(circle at 50% 42%,rgba(168,59,224,.06),transparent 70%),#fff;aspect-ratio:1.3;box-shadow:var(--shadow)}
.morph canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.morph .cap{position:absolute;left:14px;top:12px;font-size:.78rem;color:var(--ink2);font-weight:600;pointer-events:none}
.mbtns{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 12px}
.mbtn{padding:8px 14px;border-radius:999px;border:2px solid var(--line);background:#fff;color:var(--ink2);cursor:pointer;font-size:.82rem;font-weight:700;transition:.2s}
.mbtn.on{color:#fff;border-color:transparent;background:var(--grad-btn);box-shadow:0 8px 18px rgba(168,59,224,.25)}.mbtn:hover{border-color:#c9b8ff;color:var(--ink)}.mbtn.on:hover{color:#fff}
.mstat{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px}.mstat div{padding:10px 12px;border-radius:14px;border:1px solid var(--line);background:#fff;box-shadow:inset 3px 0 0 var(--s,var(--violet))}
.mstat div:nth-child(1){--s:var(--pink)}.mstat div:nth-child(2){--s:var(--blue)}.mstat div:nth-child(3){--s:var(--teal)}.mstat div:nth-child(4){--s:var(--orchid)}
.mstat b{font-family:Outfit;font-size:1.3rem;display:block;color:color-mix(in srgb,var(--s,var(--violet)) 80%,#000)}.mstat span{font-size:.74rem;color:var(--ink2)}
.ds-legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.8rem;color:var(--ink2);margin-top:12px}.ds-legend svg{margin-right:6px;vertical-align:-2px}
.in-card .aim{color:var(--ink);font-size:.88rem;margin:.2em 0 .8em;opacity:.92}
.pill.vio{background:#f6e4fb;color:#86259c}
.ba{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;margin-top:14px}
.ba-card{border:1px solid var(--line);border-radius:18px;padding:16px 16px 10px;background:#fff;box-shadow:inset 0 4px 0 var(--accent,var(--violet)),var(--shadow)}
.ba-card h4{margin:0 0 2px;font-size:1rem}.ba-card .selm{font-size:.8rem;color:var(--ink2);margin-bottom:8px}.ba-card .selm b{color:var(--ink)}
.ba-row{display:grid;grid-template-columns:120px 1fr;gap:10px;align-items:center;margin:10px 0;font-size:.8rem}
.ba-row .nm{color:var(--ink)}.ba-row .nm small{display:block;color:var(--ink2);font-size:.68rem}
.ba-bars{display:grid;gap:4px}.ba-bar{display:grid;grid-template-columns:minmax(0,1fr) 58px;gap:8px;align-items:center}
.ba-bar .trk{position:relative;height:14px;border-radius:0 5px 5px 0;background:#f6f4fd}
.ba-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:0 5px 5px 0;width:0;transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
.in-card h3{padding-right:46px}
.ba-bar.erm i{border:1.5px solid #8c9eff;background:repeating-linear-gradient(135deg,rgba(91,124,250,.30) 0 3px,transparent 3px 6px)}
.ba-bar.sel i{background:var(--accent)}
.ba-bar span{font-size:.7rem;color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap}
.verdict{display:inline-block;font-size:.66rem;font-weight:800;padding:1px 8px;border-radius:999px;margin-left:4px}
.verdict.up{background:#dcf7ee;color:#076b5e}.verdict.dn{background:#ffe3ea;color:#b01e4f}.verdict.eq{background:#f1f0f7;color:var(--ink2)}
.ba-key{display:flex;gap:16px;flex-wrap:wrap;font-size:.78rem;color:var(--ink2);margin-top:6px}.ba-key i{display:inline-block;width:22px;height:10px;border-radius:0 3px 3px 0;margin-right:6px;vertical-align:-1px}
.stage-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.stage-tab{padding:9px 17px;border-radius:999px;border:2px solid var(--line);background:#fff;color:var(--ink2);cursor:pointer;font-weight:800;font-size:.82rem;transition:.2s}
.stage-tab.on{color:#fff;background:var(--grad-btn);border-color:transparent;box-shadow:0 8px 20px rgba(168,59,224,.28)}
.ba-card.off{opacity:.85;align-self:start;background:var(--tint);box-shadow:none}.ba-card.off .selm{margin:8px 0 6px}
.why-line{font-size:.74rem;color:var(--ink2);margin:10px 0 4px;line-height:1.55}.why-line b{color:var(--ink)}
.verdict.amb{background:#fff1d6;color:#7a4f00}
footer{padding:46px 0;color:#fff;font-size:.88rem;text-align:center;background:var(--grad-btn);font-weight:600}
.wizard,.wiz-steps{display:none}
</style></head><body>
<nav><div class="container"><div class="nav-links">
<a href="#hero">Home</a><a href="#motivation">Motivation</a><a href="#objectives">Objectives</a><a href="#datasets">Datasets</a>
<a href="#pipeline">Pipeline</a><a href="#preprocessing">Pre-processing</a><a href="#why-selected">Method selection</a><a href="#latent">Latent</a><a href="#inprocessing">In-processing</a><a href="#explorer">Plots</a></div></div></nav>

<section id="hero"><canvas id="net"></canvas><div class="hero-glow g1"></div><div class="hero-glow g2"></div>
<div class="container hero-inner">
<h1 class="hero-title">Healthcare Bias Mitigation:<br><span class="grad">a multi-method approach across diverse clinical applications</span></h1>
<svg class="ecg" viewBox="0 0 760 70"><defs><linearGradient id="ecgGrad"><stop offset="0" stop-color="#e8457c"/><stop offset=".35" stop-color="#e09400"/><stop offset=".65" stop-color="#0e8f80"/><stop offset="1" stop-color="#3d8bfd"/></linearGradient></defs>
<path d="M0 40 H120 L135 40 L145 18 L155 62 L165 6 L178 40 H300 L315 40 L325 18 L335 62 L345 6 L358 40 H480 L495 40 L505 18 L515 62 L525 6 L538 40 H760"/></svg>
<p class="hero-sub">Three stages against bias: <b>pre-processing</b> repairs the data of four disease cohorts (eight biases, every
method benchmarked on every metric, the best-ranked candidate method applied), the <b>latent stage</b> maps the four diseases and a cross-hospital ICU cohort into one shared
feature space, and <b>in-processing</b> trains the model itself to be fair, calibrated and robust in hospitals it has never seen.</p>
<div class="stats"><div class="stat"><b class="count" data-to="5">0</b><span>cohorts (4 diseases + cross-hospital ICU)</span></div><div class="stat"><b class="count" data-to="__TOTAL_RAW__">0</b><span>records analysed</span></div>
<div class="stat"><b class="count" data-to="12">0</b><span>bias types (8 data + 4 model)</span></div><div class="stat"><b class="count" data-to="__ALL_N__">0</b><span>methods compared</span></div></div>
<div class="btns"><a class="btn p" href="#explorer">Explore the dashboards</a><a class="btn s" href="#preprocessing">Pre-processing</a><a class="btn s" href="#latent">Latent space</a><a class="btn s" href="#inprocessing">In-processing</a></div></div></section>

__KEEP__

<section id="datasets" class="reveal"><div class="container"><span class="s-label">Clinical data foundations</span><h2 class="s-title">Four disease datasets + a cross-hospital ICU cohort</h2>
<p class="s-desc">Real, publicly documented sources (__TOTAL__ records in total). Each disease folder holds the raw data before preprocessing, the preprocessing notebook, the preprocessed CSVs and one dashboard per bias. The cross-hospital ICU cohort (eICU hospitals + MIMIC-IV) joins at the latent stage and provides the hospitals for the in-processing cross-hospital bias.</p>
<div class="grid-4">__DS_CARDS__</div></div></section>

<section id="pipeline" class="reveal"><div class="container"><span class="s-label">End-to-end workflow</span><h2 class="s-title">Bias-aware pipeline</h2>
<p class="s-desc">Every step is fitted on training rows; methods are scored on training and validation rows; the test split is only used for the final check.</p>
<div class="stages">
<div class="stage reveal"><div class="num">1</div><h3>Pre-processing</h3><p>Repair the <b>data</b> of each disease.</p><ul><li>8 data biases, __PRE_N__ methods</li><li>best-ranked <b>candidate</b> method applied, baselines kept as comparison</li><li>one dashboard per bias</li></ul><div class="run"></div></div>
<div class="stage reveal"><div class="num">2</div><h3>Latent space</h3><p>Put all cohorts into <b>one shared 16-D feature space</b>.</p><ul><li>DGCCA, SAINT, MV-DAE vs PCA</li><li>chosen by CKA, Macro-F1, Silhouette, NMI/ARI</li><li>4 diseases + cross-hospital ICU cohort</li></ul><div class="run"></div></div>
<div class="stage reveal"><div class="num">3</div><h3>In-processing</h3><p>Train the <b>model</b> to be fair, calibrated and robust.</p><ul><li>4 model biases, __IN_N__ methods, ERM = before</li><li>best mean rank over each bias's 4 metrics</li><li>final check on unseen hospitals + MIMIC</li></ul><div class="run"></div></div>
</div>
<div class="flow"><div class="node k">Raw data</div><div class="arrow"></div><div class="node">Duplicates</div><div class="arrow"></div><div class="node">Split 70/15/15</div><div class="arrow"></div>
<div class="node">Missing data</div><div class="arrow"></div><div class="node">Outliers</div><div class="arrow"></div><div class="node">Noise</div><div class="arrow"></div><div class="node">Feature scale</div><div class="arrow"></div>
<div class="node">Encoding / domain</div><div class="arrow"></div><div class="node">Class imbalance</div><div class="arrow"></div><div class="node">Demographic</div><div class="arrow"></div>
<div class="node l">Common latent space</div><div class="arrow"></div><div class="node k">In-processing</div><div class="arrow"></div><div class="node">Fair, calibrated predictions</div></div>
<svg class="stream" viewBox="0 0 1000 120" role="img" aria-label="Animation: records enter with biases (rose) and leave the eight bias steps balanced (teal)">
<defs><linearGradient id="pipe" x1="0" x2="1"><stop offset="0" stop-color="#e8457c" stop-opacity=".28"/><stop offset=".5" stop-color="#0e8f80" stop-opacity=".22"/><stop offset="1" stop-color="#6d3fd9" stop-opacity=".28"/></linearGradient></defs>
<path id="flowPath" d="M20 60 C 180 10, 320 110, 500 60 S 820 10, 980 60" fill="none" stroke="url(#pipe)" stroke-width="16" stroke-linecap="round"/>
__STREAM_DOTS__
<text x="20" y="112">raw records (missing, skewed, imbalanced, duplicated)</text><text x="500" y="112" text-anchor="middle">common latent space</text><text x="980" y="112" text-anchor="end">fair, calibrated prediction</text></svg>
<div class="legend-row"><span><i style="background:#e8457c"></i>record still carrying a data bias</span><span><i style="background:#0e8f80"></i>repaired by pre-processing</span><span><i style="background:#6d3fd9"></i>prediction from the in-processing model</span></div></div></section>

<section id="preprocessing"><div class="container"><span class="s-label reveal">Pre-processing</span><h2 class="s-title reveal">Eight biases &middot; candidate and baseline methods</h2>
<p class="s-desc reveal">All listed methods run on every detected bias and are ranked on every metric; the <b>candidate</b> method with the lowest mean rank is applied and the <b>baseline</b> methods are kept as the comparison.</p>
<div class="grid-4">__BIAS_CARDS__</div>
<div class="glass reveal" style="margin-top:28px"><h3>What was applied to each dataset</h3><p class="s-desc">Green = selected method (below: the best baseline method it was compared with).</p>
<div class="table-wrap"><table><thead><tr><th>Bias</th><th>Heart</th><th>Diabetes</th><th>Hypertension</th><th>Respiratory</th></tr></thead><tbody>__MATRIX__</tbody></table></div></div>
<div class="glass reveal" style="margin-top:28px"><h3>Before (raw data) &rarr; after (selected method)</h3>
<p class="s-desc">Each detected bias, measured before its step and after the selected method; every bias statistic is better when lower. Under the bars: why that method was chosen. The last card trains the same model on the raw and on the fully pre-processed data and checks it on the untouched test split.</p>
<div class="tabs" id="preTabs"></div><div class="ba-key"><span><i style="border:1.5px solid #8c9eff;background:repeating-linear-gradient(135deg,rgba(91,124,250,.30) 0 3px,transparent 3px 6px)"></i>before (raw data)</span><span><i id="preKey" style="background:#e8457c"></i>after (selected method), in the dataset's colour</span><span>&darr; lower is better &middot; &uarr; higher is better</span></div>
<div class="ba" id="prePanels"></div></div></div></section>

<section id="why-selected" class="reveal"><div class="container"><span class="s-label">Evidence</span><h2 class="s-title">Why each method was selected</h2>
<p class="s-desc">Mean rank over the bias metrics (lower = better). The selected method has the lowest mean rank of the candidate methods; the next-best candidate and the best baseline method (run for comparison only) are shown with theirs.</p>
<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Bias</th><th>Selected method</th><th>Mean rank</th><th>Next-best candidate</th><th>Mean rank</th><th>Best baseline (comparison only)</th><th>Mean rank</th></tr></thead><tbody>__WHY__</tbody></table></div></div></section>

<section id="comparison" class="reveal"><div class="container"><span class="s-label">Method comparison plots</span><h2 class="s-title">Every method compared on every metric</h2>
<p class="s-desc">For each detected bias, every method that was run (candidate methods on top, baseline methods below the dashed line): A the rank total used for the selection, B the rank on each metric, C the raw value of every metric (blue = selected method, &#9733; = best value of that metric), and why the method was chosen. Click an image to open it full size.</p>
<div class="tabs" id="cmpTabs"></div>
<h3 class="cmp-h">All methods per bias</h3><div class="biasbtns" id="cmpBias"></div><p class="cmp-note" id="cmpNote"></p>
<div class="cmp-img"><a id="cmpOpen" target="_blank"><img id="cmpImg" alt="method comparison plot" loading="lazy"></a></div></div></section>

<section id="latent"><div class="container"><span class="s-label reveal">Latent feature extraction</span><h2 class="s-title reveal">One common latent space for four diseases and the cross-hospital ICU cohort</h2>
<p class="s-desc reveal">The diseases come from different surveys and hospitals with different variables, so each gets its own encoder; all encoders write into the <b>same 16 coordinates</b>. A good shared space keeps each outcome predictable (Macro-F1, Silhouette), makes matching patient groups line up across datasets (CKA) and stops the coordinates from simply revealing which dataset a patient came from (NMI/ARI). MIMIC-IV rows are only encoded, never trained on, so the in-processing cross-hospital check stays external.</p>
<div class="morph-wrap reveal"><div><div class="mbtns" id="mBtns"></div><div class="morph"><canvas id="morph" aria-label="Validation records of the five cohorts moving from the per-dataset PCA layout into the selected latent space"></canvas><div class="cap" id="mCap"></div></div>
<div class="ds-legend" id="mLegend"></div></div><div class="glass"><h3 style="margin-top:0">Watch the cohorts merge</h3><p class="s-desc" style="font-size:.9rem">Each dot is one real validation record (300 per cohort). <b>Before</b>: every dataset is compressed on its own (PCA), so there is no shared meaning. Press a method to move the same records into its shared space; each view is only rotated and uniformly scaled to fit, so distances between records keep their proportions (the few most extreme records sit on the edge).</p>
<div class="mstat" id="mStat"></div><p class="s-desc" style="font-size:.78rem;margin-top:10px">CKA &uarr; and Macro-F1 &uarr; are better when higher, NMI &darr; and ARI &darr; when lower (validation split).</p></div></div>
<div class="grid-3">
<div class="card reveal"><div class="card-h">Deep Generalized CCA (DGCCA)</div><p class="card-p">Dataset-specific networks maximise the MAXVAR correlation of stratum-matched views (age &times; sex &times; BMI; age &times; sex for the ICU cohort, which has no BMI), followed by a global GCCA projection.</p></div>
<div class="card reveal"><div class="card-h">SAINT</div><p class="card-p">Feature tokens with column attention and inter-sample attention to a fixed bank of training rows; contrastive and denoising pre-training.</p></div>
<div class="card reveal"><div class="card-h">Multi-View Denoising Autoencoder</div><p class="card-p">Dataset-specific encoders/decoders with masking noise, a shared 16-D code and an MMD alignment term.</p></div></div>
<div class="grid-4" style="margin-top:18px">
<div class="card reveal"><div class="card-h">CKA &uarr;</div><p class="card-p">Cross-dataset alignment of the latent features in the common space.</p></div>
<div class="card reveal"><div class="card-h">Macro-F1 &uarr;</div><p class="card-p">Clinical information preserved for each disease's outcome.</p></div>
<div class="card reveal"><div class="card-h">Silhouette &uarr;</div><p class="card-p">How clearly clinical classes separate in the latent space.</p></div>
<div class="card reveal"><div class="card-h">NMI / ARI &darr;</div><p class="card-p">Dataset-specific information left in the common space (lower = more domain-invariant).</p></div></div>
<div style="margin-top:22px">__LATENT__</div></div></section>

<section id="inprocessing"><div class="container"><span class="s-label reveal">In-processing</span><h2 class="s-title reveal">Training the model itself to be fair, calibrated and robust</h2>
<p class="s-desc reveal">Pre-processing repairs the <b>data</b>; in-processing changes <b>how the model learns</b>. Every method trains the same small network on the latent features with a different training objective. Plain training (<b>ERM</b>) is the &ldquo;before&rdquo;. For each bias the method with the best mean rank over that bias's four metrics on the validation split is selected (it must stay within 0.05 AUROC of ERM) and checked once on the untouched test split. The cross-hospital bias uses the cross-hospital ICU cohort with each disease's recorded label: trained on 141 eICU hospitals, selected on 34 others, checked on 31 unseen eICU hospitals and on MIMIC-IV, a different hospital system.</p>
<div class="grid-4">__IN_CARDS__</div>
<div class="glass reveal" style="margin-top:28px"><h3>Selected in-processing method per disease</h3><p class="s-desc">Violet = selected method; below: how many of the bias's four metrics beat ERM on the test split.</p>
<div class="table-wrap"><table><thead><tr><th>Bias</th><th>Heart</th><th>Diabetes</th><th>Hypertension</th><th>Respiratory</th></tr></thead><tbody>__IN_MATRIX__</tbody></table></div></div>
<div class="glass reveal" style="margin-top:28px"><h3>Before (ERM) &rarr; after (selected method), test split</h3>
<div class="tabs" id="inTabs"></div><div class="ba-key"><span><i style="border:1.5px solid #8c9eff;background:repeating-linear-gradient(135deg,rgba(91,124,250,.30) 0 3px,transparent 3px 6px)"></i>ERM (before)</span><span><i id="inKey" style="background:#e8457c"></i>selected method (after), in the disease's colour</span><span>&darr; lower is better &middot; &uarr; higher is better</span></div>
<div class="ba" id="inPanels"></div></div></div></section>

<section id="explorer"><div class="container"><span class="s-label reveal">Interactive dashboards</span><h2 class="s-title reveal">Plot explorer: every stage, every bias</h2>
<p class="s-desc reveal">Pre-processing: one dashboard per bias and dataset (before = blue, after = teal). Latent: one dashboard per latent metric. In-processing: one dashboard per model bias (before = ERM, after = selected method).</p>
<div class="stage-tabs" id="stTabs"></div><div class="tabs" id="dsTabs"></div><div class="biasbtns" id="biasBtns"></div>
<div class="frame-bar"><span id="frameTitle"></span><a id="frameOpen" target="_blank">open in a new tab &#8599;</a></div>
<div class="frame"><iframe id="frame" title="bias dashboard" loading="lazy"></iframe></div></div></section>



<script>
// animated particle network in the hero
(function(){const c=document.getElementById('net'),x=c.getContext('2d');let w,h,p=[];function rs(){w=c.width=c.offsetWidth;h=c.height=c.offsetHeight;
p=Array.from({length:Math.min(90,Math.floor(w/14))},()=>({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.45,vy:(Math.random()-.5)*.45,r:Math.random()*2.2+.9,c:['#e8457c','#e09400','#3d8bfd','#0e8f80','#c04ad6','#ffc21a'][Math.floor(Math.random()*6)]}))}
rs();addEventListener('resize',rs);(function f(){x.clearRect(0,0,w,h);for(const a of p){a.x+=a.vx;a.y+=a.vy;if(a.x<0||a.x>w)a.vx*=-1;if(a.y<0||a.y>h)a.vy*=-1;
x.beginPath();x.arc(a.x,a.y,a.r,0,7);x.fillStyle=a.c;x.globalAlpha=.75;x.fill()}for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++){const d=Math.hypot(p[i].x-p[j].x,p[i].y-p[j].y);
if(d<130){x.globalAlpha=1;x.strokeStyle=`rgba(109,63,217,${.2*(1-d/130)})`;x.beginPath();x.moveTo(p[i].x,p[i].y);x.lineTo(p[j].x,p[j].y);x.stroke()}}requestAnimationFrame(f)})()})();
// reveal on scroll + counters
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');e.target.querySelectorAll('.count').forEach(count);io.unobserve(e.target)}}),{threshold:.12});
document.querySelectorAll('.reveal,.stats').forEach(el=>io.observe(el));
function count(el){if(el.dataset.done)return;el.dataset.done=1;const to=+el.dataset.to;if(matchMedia('(prefers-reduced-motion: reduce)').matches){el.textContent=to.toLocaleString();return}const t0=performance.now();(function s(t){const k=Math.min(1,(t-t0)/1600);el.textContent=Math.round(to*(1-Math.pow(1-k,3))).toLocaleString();if(k<1)requestAnimationFrame(s)})(t0)}
document.querySelectorAll('#hero .count').forEach(count);
if(matchMedia('(prefers-reduced-motion: reduce)').matches){document.querySelectorAll('svg').forEach(s=>s.pauseAnimations&&s.pauseAnimations());
 document.documentElement.style.setProperty('scroll-behavior','auto');document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in'))}
// active nav link
const links=[...document.querySelectorAll('.nav-links a')];addEventListener('scroll',()=>{let cur='';document.querySelectorAll('section[id]').forEach(s=>{if(scrollY>=s.offsetTop-120)cur=s.id});links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+cur))});
// plot explorer: stage -> dataset -> dashboard
const MAP=__EXPLORER__;let st=Object.keys(MAP)[0],ds=Object.keys(MAP[st])[0],bias=Object.keys(MAP[st][ds])[0];
const stt=document.getElementById('stTabs'),tabs=document.getElementById('dsTabs'),bb=document.getElementById('biasBtns');
function mk(cls,txt,on,fn){const b=document.createElement('button');b.className=cls+(on?' on':'');b.textContent=txt;b.onclick=fn;return b}
function render(){if(!MAP[st][ds])ds=Object.keys(MAP[st])[0];if(!(bias in MAP[st][ds]))bias=Object.keys(MAP[st][ds])[0];
stt.innerHTML='';Object.keys(MAP).forEach(s=>stt.appendChild(mk('stage-tab',s,s===st,()=>{st=s;render()})));
tabs.innerHTML='';Object.keys(MAP[st]).forEach(d=>tabs.appendChild(mk('tab',d,d===ds,()=>{ds=d;render()})));
bb.innerHTML='';Object.keys(MAP[st][ds]).forEach(k=>bb.appendChild(mk('bb',k,k===bias,()=>{bias=k;render()})));
const src=MAP[st][ds][bias],fr=document.getElementById('frame'),op=document.getElementById('frameOpen');document.getElementById('frameTitle').textContent=st+' \u00b7 '+ds+' \u00b7 '+bias;
if(src){fr.removeAttribute('srcdoc');fr.src=src;op.href=src;op.style.visibility='visible'}
else{fr.removeAttribute('src');op.style.visibility='hidden';fr.srcdoc="<body style='margin:0;font-family:Inter,sans-serif;background:#f7f5ff;color:#575375;display:flex;align-items:center;justify-content:center;height:100vh;text-align:center'>"
+"<div><div style='font-size:42px'>&#9203;</div><h3 style='color:#1c1745'>"+st+" for "+ds+" is still running</h3><p>This dashboard appears here as soon as that run finishes.</p></div></body>"}}
render();
// method comparison plots: dataset -> bias -> image
(function(){const C=__CMP_JSON__;const keys=Object.keys(C);if(!keys.length){document.getElementById('comparison').style.display='none';return}
const tb=document.getElementById('cmpTabs'),cb=document.getElementById('cmpBias'),img=document.getElementById('cmpImg'),op=document.getElementById('cmpOpen'),nt=document.getElementById('cmpNote');
let d=keys[0],b=0;
function draw(){tb.innerHTML='';keys.forEach(k=>tb.appendChild(mk('tab',k,k===d,()=>{d=k;b=0;draw()})));const x=C[d];
cb.innerHTML='';x.cmp.forEach((c,i)=>cb.appendChild(mk('bb',c.bias,i===b,()=>{b=i;draw()})));const c=x.cmp[b];
if(c){img.src=c.src;img.alt=d+' · '+c.bias;op.href=c.src;nt.textContent=c.note}}
draw()})();
// latent: the same validation records move from the per-dataset PCA layout into each method's shared space
(function(){const S=__SCATTER_JSON__;const c=document.getElementById('morph');if(!c)return;const box=c.parentElement;
if(!S){box.innerHTML='<div class="wait">The latent run starts when the pre-processing notebooks finish; this animation appears here automatically.</div>';return}
const ctx=c.getContext('2d'),N=S.dataset.length,base=S.order[0];let cur=base,pos=S.views[base].map(p=>p.slice()),anim=0,hover=-1;
const tip=document.createElement('div');tip.style.cssText='position:absolute;pointer-events:none;display:none;background:#fff;border:1.5px solid #6d3fd9;color:#1c1745;box-shadow:0 6px 18px rgba(28,23,69,.15);font-size:12px;padding:6px 9px;border-radius:8px;white-space:nowrap';box.appendChild(tip);
const SH={circle:'<circle cx="7" cy="7" r="5"/>',square:'<rect x="2.5" y="2.5" width="9" height="9"/>',triangle:'<path d="M7 1.5 13 12H1z"/>',diamond:'<path d="M7 1 13 7 7 13 1 7z"/>',cross:'<path d="M5 1.5h4v3.5h3.5v4H9v3.5H5V9H1.5V5H5z"/>'};
document.getElementById('mLegend').innerHTML=S.datasets.map((d,i)=>`<span><svg width="14" height="14" viewBox="0 0 14 14" fill="${S.colors[i]}" aria-hidden="true">${SH[S.shapes[i]]}</svg>${d}</span>`).join('');
function W(){return box.getBoundingClientRect()}
function xy(p){const r=W(),m=24;return [r.width/2+p[0]*(r.width/2-m),r.height/2-p[1]*(r.height/2-m)]}
function mark(x,y,r,s){ctx.beginPath();if(s==='square')ctx.rect(x-r*.9,y-r*.9,r*1.8,r*1.8);
 else if(s==='triangle'){ctx.moveTo(x,y-r*1.15);ctx.lineTo(x+r*1.05,y+r*.75);ctx.lineTo(x-r*1.05,y+r*.75);ctx.closePath()}
 else if(s==='diamond'){ctx.moveTo(x,y-r*1.2);ctx.lineTo(x+r*1.2,y);ctx.lineTo(x,y+r*1.2);ctx.lineTo(x-r*1.2,y);ctx.closePath()}
 else if(s==='cross'){const a=r*1.1,b=r*.38;ctx.rect(x-a,y-b,2*a,2*b);ctx.rect(x-b,y-a,2*b,2*a)}else ctx.arc(x,y,r,0,7)}
function draw(){const r=W();ctx.clearRect(0,0,r.width,r.height);for(let i=0;i<N;i++){const q=xy(pos[i]),k=S.dataset[i];mark(q[0],q[1],i===hover?5.5:3.2,S.shapes[k]);
 ctx.fillStyle=S.colors[k];ctx.globalAlpha=i===hover?1:.85;ctx.fill();if(i===hover){ctx.lineWidth=2;ctx.strokeStyle='#1c1745';ctx.stroke()}}ctx.globalAlpha=1}
function size(){const r=W(),d=window.devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function stats(){const M=(S.metrics||{})[cur]||{},f=v=>v===null||v===undefined?'n/a':v.toFixed(3);
 document.getElementById('mCap').textContent=cur===base?'Before: each dataset compressed separately (no shared space)':cur+(cur===S.selected?' \u2605 selected':'')+': one shared latent space';
 document.getElementById('mStat').innerHTML=[['CKA \u2191 (alignment)',M['CKA (alignment)']],['Macro-F1 \u2191 (information kept)',M['Macro-F1 (preservation)']],['NMI \u2193 (dataset identity)',M['NMI (domain)']],['ARI \u2193 (dataset identity)',M['ARI (domain)']]]
  .map(r=>`<div><b>${f(r[1])}</b><span>${r[0]}</span></div>`).join('')}
function buttons(){const b=document.getElementById('mBtns');b.innerHTML='';S.order.forEach(m=>b.appendChild(mk('mbtn',(m===base?'Before: ':'')+m+(m===S.selected?' \u2605':''),m===cur,()=>go(m))))}
function go(m){const from=pos.map(p=>p.slice()),to=S.views[m];cur=m;stats();buttons();cancelAnimationFrame(anim);
 if(matchMedia('(prefers-reduced-motion: reduce)').matches){pos=to.map(p=>p.slice());draw();return}
 const t0=performance.now();(function f(t){let k=Math.min(1,(t-t0)/1700);k=k<.5?4*k*k*k:1-Math.pow(-2*k+2,3)/2;
  for(let i=0;i<N;i++){pos[i][0]=from[i][0]+(to[i][0]-from[i][0])*k;pos[i][1]=from[i][1]+(to[i][1]-from[i][1])*k}draw();if(k<1)anim=requestAnimationFrame(f)})(t0)}
c.addEventListener('mousemove',e=>{const r=W();let best=-1,bd=100;for(let i=0;i<N;i++){const q=xy(pos[i]),d=(q[0]-e.clientX+r.left)**2+(q[1]-e.clientY+r.top)**2;if(d<bd){bd=d;best=i}}
 if(best!==hover){hover=best;draw()}if(best<0){tip.style.display='none';return}const q=xy(pos[best]);tip.style.display='block';tip.style.left=Math.min(q[0]+10,r.width-170)+'px';tip.style.top=(q[1]-34)+'px';
 tip.textContent=S.datasets[S.dataset[best]]+' \u00b7 validation record \u00b7 outcome '+(S.label[best]?'positive':'negative')});
c.addEventListener('mouseleave',()=>{hover=-1;tip.style.display='none';draw()});
addEventListener('resize',size);size();buttons();stats();
new IntersectionObserver((es,o)=>es.forEach(e=>{if(e.isIntersecting){o.disconnect();setTimeout(()=>go(S.selected),600)}}),{threshold:.35}).observe(box)})();
// in-processing: ERM (before) -> selected method (after) on the test split, bars grow when shown
(function(){const I=__IN_JSON__,tabsEl=document.getElementById('inTabs'),panels=document.getElementById('inPanels');if(!tabsEl)return;
const names=Object.keys(I);let cur=names.find(n=>I[n])||names[0];
function fmt(v){return v===null||v===undefined?'n/a':(Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=1?v.toFixed(3):v.toFixed(4))}
function gain(m){return m.erm===null||m.sel===null?null:(m.dir==='max'?m.sel-m.erm:m.erm-m.sel)}
function tie(m){return Math.abs(gain(m))<=0.002*Math.max(Math.abs(m.erm),Math.abs(m.sel),1e-12)}
function row(m,sel){const mx=Math.max(Math.abs(m.erm||0),Math.abs(m.sel||0))||1,ok=gain(m)!==null;const same=ok&&tie(m);
 const better=ok&&!same&&gain(m)>0;const v=!ok?'':same?'<span class="verdict eq">same</span>':better?'<span class="verdict up">better</span>':'<span class="verdict dn">worse</span>';
 return `<div class="ba-row"><div class="nm">${m.name} ${m.dir==='max'?'\u2191':'\u2193'}${v}<small>${m.note||''}</small></div><div class="ba-bars">
 <div class="ba-bar erm" title="ERM (before): ${fmt(m.erm)}"><div class="trk"><i data-w="${100*Math.abs(m.erm||0)/mx}"></i></div><span>${fmt(m.erm)}</span></div>
 <div class="ba-bar sel" title="${sel} (after): ${fmt(m.sel)}"><div class="trk"><i data-w="${100*Math.abs(m.sel||0)/mx}"></i></div><span>${fmt(m.sel)}</span></div></div></div>`}
function render(){tabsEl.innerHTML='';names.forEach(n=>tabsEl.appendChild(mk('tab',n,n===cur,()=>{cur=n;render()})));const D=I[cur];
 document.getElementById('inKey').style.background=D?D.color:'#6d3fd9';
 if(!D){panels.innerHTML='<div class="ba-card"><span class="pill run">still running&hellip;</span> The in-processing results for '+cur+' appear here when its notebook finishes.</div>';return}
 panels.innerHTML=D.families.map(f=>{if(!f.selected)return `<div class="ba-card"><h4>${f.title}</h4><p class="selm">not applicable</p></div>`;
  const won=f.metrics.filter(m=>gain(m)!==null&&!tie(m)&&gain(m)>0).length;
  const extra=[{name:'AUROC',dir:'max',erm:f.auroc[0],sel:f.auroc[1],note:'accuracy guard'}].concat(f.extra||[]);
  return `<div class="ba-card" style="--accent:${D.color}"><h4>${f.title}</h4><p class="selm">selected <b>${f.selected}</b> &middot; ${won} of ${f.metrics.length} bias metrics better than ERM`
   +(f.dashboard?` &middot; <a href="${f.dashboard}" target="_blank" rel="noopener">dashboard &#8599;</a>`:'')+`</p>${f.metrics.concat(extra).map(m=>row(m,f.selected)).join('')}</div>`}).join('');
 const grow=()=>panels.querySelectorAll('.ba-bar i').forEach(i=>i.style.width=Math.max(.8,+i.dataset.w)+'%');requestAnimationFrame(()=>requestAnimationFrame(grow))}
render()})();
// pre-processing: each bias statistic before (raw data) -> after (selected method); bars grow when shown
(function(){const P=__PRE_JSON__,tabsEl=document.getElementById('preTabs'),panels=document.getElementById('prePanels');if(!tabsEl)return;
const names=Object.keys(P);let cur=names.find(n=>P[n])||names[0];
const fmt=v=>v===null||v===undefined?'n/a':Number.isInteger(v)?String(v):(Math.abs(v)>=100?v.toFixed(1):Math.abs(v)>=1?v.toFixed(3):v.toFixed(4));
const gain=m=>m.erm===null||m.sel===null?null:(m.dir==='max'?m.sel-m.erm:m.erm-m.sel);
const tie=m=>Math.abs(gain(m))<=0.002*Math.max(Math.abs(m.erm),Math.abs(m.sel),1e-12);
function row(m,a,b,w){const mx=Math.max(Math.abs(m.erm||0),Math.abs(m.sel||0))||1,g=gain(m);
 const v=g===null?'':tie(m)?'<span class="verdict eq">same</span>':g>0?`<span class="verdict up">${w[0]}</span>`:`<span class="verdict dn">${w[1]}</span>`;
 return `<div class="ba-row"><div class="nm">${m.name} ${m.dir==='max'?'\u2191':'\u2193'}${v}<small>${m.note||''}</small></div><div class="ba-bars">
 <div class="ba-bar erm" title="${a}: ${fmt(m.erm)}"><div class="trk"><i data-w="${100*Math.abs(m.erm||0)/mx}"></i></div><span>${fmt(m.erm)}</span></div>
 <div class="ba-bar sel" title="${b}: ${fmt(m.sel)}"><div class="trk"><i data-w="${100*Math.abs(m.sel||0)/mx}"></i></div><span>${fmt(m.sel)}</span></div></div></div>`}
function why(c){if(!c.rank)return '';const s=c.rank[0],nx=c.rank[1],cls=s<nx-1e-9?'up':'eq';
 const txt=cls==='up'?'best of the '+c.k+' candidate methods':'tied with the next candidate (first in the list applied)';
 return `<p class="why-line"><span class="verdict ${cls}">${txt}</span> mean rank ${s.toFixed(2)} vs ${nx.toFixed(2)} for <b>${c.next}</b> (next candidate, lower = better)`
  +(c.base&&c.base_rank!==null?` &middot; best baseline (comparison only): ${c.base} ${c.base_rank.toFixed(2)}`:'')+'</p>'}
function render(){tabsEl.innerHTML='';names.forEach(n=>tabsEl.appendChild(mk('tab',n,n===cur,()=>{cur=n;render()})));const D=P[cur];
 document.getElementById('preKey').style.background=D?D.color:'#6d3fd9';
 if(!D){panels.innerHTML='<div class="ba-card"><span class="pill run">still running&hellip;</span> The pre-processing results for '+cur+' appear here when its notebook finishes.</div>';return}
 panels.innerHTML=D.cards.map(c=>{const head=`<h4>${c.bias} bias</h4>`;
  if(!c.detected)return `<div class="ba-card off">${head}<p class="selm"><span class="verdict eq">not detected</span> ${c.statistic}: <b>${c.rows.map(r=>fmt(r.erm)).join(' / ')}</b> &middot; nothing applied</p></div>`;
  return `<div class="ba-card" style="--accent:${D.color}">${head}<p class="selm">applied <b>${c.selected}</b> &middot; ${c.compared} methods compared`
   +(c.dashboard?` &middot; <a href="${c.dashboard}" target="_blank" rel="noopener">dashboard &#8599;</a>`:'')+'</p>'
   +c.rows.map(m=>row(m,'before (raw data)','after ('+c.selected+')',['reduced','increased'])).join('')+why(c)+'</div>'}).join('')
  +(D.e2e.length?`<div class="ba-card" style="--accent:${D.color}"><h4>Whole pre-processing pipeline</h4><p class="selm">the same model trained on the raw and on the pre-processed data, checked on the untouched test split</p>`
   +D.e2e.map(m=>row(m,'model on raw data','model on pre-processed data',['better','worse'])).join('')+'</div>':'');
 requestAnimationFrame(()=>requestAnimationFrame(()=>panels.querySelectorAll('.ba-bar i').forEach(i=>i.style.width=Math.max(.8,+i.dataset.w)+'%')))}
render()})();
</script></body></html>"""


if __name__ == "__main__":
    main()
