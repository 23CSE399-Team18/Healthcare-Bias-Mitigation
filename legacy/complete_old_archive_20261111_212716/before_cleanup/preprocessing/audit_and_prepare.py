"""Audit saved research outputs and build a fixed, train-only preprocessing baseline.

Run with Python 3.12 + numpy/pandas/scikit-learn/matplotlib. No clinical source
files or earlier research results are overwritten. No prediction model is fitted.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preprocessing/verified_baseline"
AUDIT = ROOT / "preprocessing/audit_current"
DATASETS = ("heart", "respiratory", "diabetes", "hypertension")
SEED = 20261111
NUMERIC_MIMIC = [f"{v}_{s}" for v in ("heart_rate", "resp_rate", "spo2") for s in ("mean", "min", "max")]
CAT_MIMIC = ["age_group", "sex"]
META_MIMIC = ["meta_source", "meta_patient_id", "meta_stay_id", "meta_domain_hospitalid", "meta_split"]
SENS_MIMIC = ["sens_age_group", "sens_sex", "sens_race_ethnicity"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def predictors(df):
    return [c for c in df if c != "target" and not c.startswith(("meta_", "sens_"))]


def read_source(name):
    path = ROOT / f"model_ready/{name}_model_ready.csv"
    cols = pd.read_csv(path, nrows=0).columns
    ids = [c for c in cols if c in ("meta_record_id", "meta_patientunitstayid", "meta_split_group_uniquepid")]
    return pd.read_csv(path, dtype={c: "string" for c in ids}), path


def audit_original(name, raw):
    base = ROOT / f"preprocessing/outputs/{name}"
    log = json.loads((base / "pipeline_log.json").read_text())
    idcol = "meta_patientunitstayid" if name == "respiratory" else "meta_record_id"
    idcols = ["meta_domain_state", idcol] if name == "heart" else [idcol]
    source = raw.set_index(idcols)
    assert source.index.is_unique, (name, "source IDs not unique")
    reports, keys, groups = {}, {}, {}
    for split in ("train", "validation", "test"):
        path = base / f"{name}_{split}_preprocessed.csv"
        d = pd.read_csv(path, dtype={idcol: "string", "meta_split_group_uniquepid": "string"})
        feats = predictors(d)
        real = d.loc[d.get("meta_synthetic", pd.Series(0, index=d.index)).fillna(0).eq(0)]
        ids = pd.MultiIndex.from_frame(real[idcols]) if len(idcols) > 1 else pd.Index(real[idcol])
        known = ids.isin(source.index)
        aligned = source.reindex(ids)
        labels_match = bool(np.array_equal(real.target.to_numpy(), aligned.target.to_numpy()))
        weights = pd.to_numeric(d.meta_sample_weight, errors="coerce")
        x = d[feats].to_numpy(float)
        checks = {
            "rows_match_log": len(d) == log[f"{split}_rows"],
            "features_match_log": feats == log["final_predictors"],
            "predictors_all_finite": bool(np.isfinite(x).all()),
            "binary_labels": bool(d.target.isin([0, 1]).all()),
            "real_ids_in_current_source": bool(known.all()),
            "real_ids_unique": bool(ids.is_unique),
            "real_labels_match_current_source": labels_match,
            "weights_finite_positive": bool(np.isfinite(weights).all() and weights.gt(0).all()),
            "no_synthetic_holdout": split == "train" or len(real) == len(d),
        }
        keys[split] = set(ids)
        if name == "respiratory":
            groups[split] = set(real.meta_split_group_uniquepid)
        reports[split] = {"rows": len(d), "real_rows": len(real), "synthetic_rows": len(d)-len(real),
            "predictors": len(feats), "positive_rows": int(d.target.sum()), "checks": checks,
            "sha256": sha(path)}
    overlap = {f"{a}_{b}": len(keys[a] & keys[b]) for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test"))}
    group_overlap = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in
                     (("train", "validation"), ("train", "test"), ("validation", "test"))} if groups else {}
    notebook = json.loads((ROOT / f"preprocessing/{name}_preprocessing.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in notebook["cells"] if c["cell_type"] == "code")
    errors = [o.get("ename") for c in notebook["cells"] for o in c.get("outputs", []) if o["output_type"] == "error"]
    table = pd.read_csv(base / "end_to_end_check.csv", index_col=0)
    return {"saved_preprocessing_exists": True, "current_source_rows": len(raw), "partitions": reports,
        "record_key": idcols, "record_overlap": overlap, "patient_overlap": group_overlap, "notebook_saved_errors": errors,
        "file_integrity_passed": all(all(v["checks"].values()) for v in reports.values()) and not any(overlap.values()) and not any(group_overlap.values()),
        "test_used_for_method_selection": 'Xte_i, yte_i = part(TE, "test")' in code and 'imbalance_eval(name, Xb, yb, Xte_i, yte_i)' in code,
        "selection_restricted_to_modern": 'SELECT_FAMILY = "Modern"' in code,
        "reported_exploratory_metrics": table.loc[["AUROC", "PR-AUC", "Brier"]].to_dict(orient="index"),
        "status": "historical experimental outputs; test scores reused during selection; not independent test evidence"}


def audit_legacy_dashboards(name):
    folder = Path("D:/plots") / f"{name}_preprocessing"
    result = []
    for path in sorted(folder.glob("*dashboard.html")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"const D\s*=\s*(\{[^\n]+\});", text)
        if not match:
            result.append({"file": str(path), "payload_valid": False, "reason": "payload not found"})
            continue
        d = json.loads(match.group(1))
        comparisons_match = None
        table_name = {"01": "missing_data", "02": "class_imbalance", "03": "outlier", "04": "feature_scale",
                      "05": "demographic", "06": "duplicate", "07": "data_noise", "08": "encoding_domain"}.get(path.name[:2])
        table_path = ROOT / f"preprocessing/outputs/{name}/comparison_{table_name}.csv"
        if d.get("comparison") and table_path.exists():
            tab = pd.read_csv(table_path, index_col=0)
            comparisons_match = True
            for row in d["comparison"]["rows"]:
                for metric, value in zip(d["comparison"]["metrics"], row["values"]):
                    expected = tab.loc[row["method"], metric]
                    comparisons_match &= bool(pd.isna(expected) if value is None else np.isclose(float(value), float(expected), rtol=1e-9, atol=1e-10))
        result.append({"file": str(path), "payload_valid": True, "variables": len(d["vars"]),
            "comparison_matches_saved_csv": comparisons_match,
            "histogram_counts_nonnegative": all(all(v >= 0 for v in x[k]) for x in d["dist"].values() for k in ("b", "a") if k in x),
            "limitations": ["Density selector draws a frequency polygon, not a density estimate",
                "Continuous histogram tails are clipped into outer bins; statistics can use capped samples",
                "Undefined correlations were exported as zero",
                "A modern-only winner is not necessarily the overall best method",
                "Held-out test scores influenced method selection"]})
    return result


def feature_frame(df, nums, cats):
    d = df[nums+cats].copy()
    for c in nums:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(float)
    for c in cats:
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    return d


def make_plots(name, raw_parts, matrices, nums, names, base, imputed_train):
    """All rates use full partitions; distributions share actual, untrimmed bins."""
    base.mkdir(parents=True, exist_ok=True)
    train = raw_parts["train"]
    feats = predictors(train) if "target" in train else NUMERIC_MIMIC + CAT_MIMIC
    missing = train[feats].isna().mean().sort_values(ascending=False).head(12)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    labels = list(raw_parts)
    counts = [len(raw_parts[s]) for s in labels]
    axes[0, 0].bar(labels, counts, color="#3569a5")
    for i, n in enumerate(counts):
        axes[0, 0].text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_ylim(0, max(counts)*1.18)
    axes[0, 0].set(title="Rows retained in each partition", ylabel="Rows")
    axes[0, 1].barh(missing.index[::-1], (100*missing)[::-1], color="#ba6655")
    axes[0, 1].set(title="Missing predictors before imputation (train)", xlabel="Missing cells (%)")
    ax = axes[0, 2]
    rates = [float(100*(~np.isfinite(matrices[s])).mean()) for s in labels]
    ax.bar(labels, rates, color="#318b71")
    for i, rate in enumerate(rates):
        ax.text(i, rate+0.015, f"{rate:g}%", ha="center")
    ax.set(ylim=(0, max(1, max(rates)*1.2)), title="Invalid values after preprocessing", ylabel="Nonfinite cells (%)")
    sd = np.std(matrices["train"][:, :len(nums)], axis=0)
    order = np.argsort(sd)[-12:]
    axes[1, 0].barh([nums[i] for i in order], sd[order], color="#318b71")
    axes[1, 0].axvline(1, color="#777", ls="--", lw=1)
    axes[1, 0].set(title="Scaled numeric features (train)", xlabel="Standard deviation; constants stay at 0")
    if "target" in train:
        rates = [100*raw_parts[s].target.mean() for s in labels]
        axes[1, 1].bar(labels, rates, color="#8071ab")
        for i, rate in enumerate(rates):
            axes[1, 1].text(i, rate+1, f"{rate:.2f}%", ha="center")
        axes[1, 1].set(ylim=(0, 100), title="Recorded target prevalence (no resampling)", ylabel="Target = 1 (%)")
    else:
        cs = ["target_"+c for c in DATASETS]
        rates = [100*train[c].mean() for c in cs]
        axes[1, 1].bar([c.replace("target_", "") for c in cs], rates, color="#8071ab")
        axes[1, 1].set(ylim=(0, 100), title="MIMIC training diagnosis-label prevalence", ylabel="Documented diagnosis (%)")
        axes[1, 1].tick_params(axis="x", rotation=20)
    nonconstant = [c for c in nums if train[c].nunique() > 12]
    v = max(nonconstant or nums, key=lambda c: train[c].isna().mean())
    before = train[v].dropna().to_numpy(float)
    after = imputed_train[:, nums.index(v)]
    edges = np.histogram_bin_edges(np.r_[before, after], bins=30)
    ax = axes[1, 2]
    ax.hist(before, bins=edges, weights=np.ones(len(before))/max(1,len(before))*100,
            histtype="step", linewidth=1.7, label=f"Observed ({len(before):,})")
    ax.hist(after, bins=edges, weights=np.ones(len(after))/len(after)*100,
            histtype="step", linewidth=1.7, label=f"Imputed ({len(after):,})")
    ax.set(title=f"{v}: imputation in original units", xlabel=v, ylabel="Share of each series (%)")
    ax.legend(fontsize=8)
    fig.suptitle(f"{name.title()} | verified fixed preprocessing baseline", fontsize=17)
    fig.savefig(base / "preprocessing_checks.png", dpi=150)
    plt.close(fig)
    save_json(base / "plot_data.json", {"partition_rows": dict(zip(labels, counts)),
        "missing_train_percent": (100*missing).to_dict(),
        "numeric_train_std": dict(zip(nums, map(float, sd))), "histogram_feature": v,
        "histogram_edges": edges.tolist(), "observed_counts": np.histogram(before, edges)[0].tolist(),
        "imputed_counts": np.histogram(after, edges)[0].tolist(),
        "observed_denominator": len(before), "imputed_denominator": len(after),
        "histogram_count_reconciliation": int(np.histogram(before, edges)[0].sum()) == len(before) and int(np.histogram(after, edges)[0].sum()) == len(after),
        "feature_count": len(names), "plot_population": "full partitions, no tail clipping or sampling"})


def build_baseline(name, raw, source_path):
    base = OUT / name
    base.mkdir(parents=True, exist_ok=True)
    feats = predictors(raw)
    nums = [c for c in feats if pd.api.types.is_numeric_dtype(raw[c])]
    cats = [c for c in feats if c not in nums]
    all_idx = np.arange(len(raw))
    if name == "respiratory":
        group = raw.meta_split_group_uniquepid
        assert group.notna().all()
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=SEED).split(raw, groups=group))
        bv, bt = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED+1).split(raw.iloc[b], groups=group.iloc[b]))
        split_idx = {"train": a, "validation": b[bv], "test": b[bt]}
    else:
        a, b = train_test_split(all_idx, test_size=.30, random_state=SEED, stratify=raw.target)
        bv, bt = train_test_split(b, test_size=.5, random_state=SEED+1, stratify=raw.iloc[b].target)
        split_idx = {"train": a, "validation": bv, "test": bt}
    parts = {s: raw.iloc[ix].reset_index(drop=True) for s, ix in split_idx.items()}
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                              ("scale", StandardScaler())]), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats)],
        sparse_threshold=0, remainder="drop")
    transformer.fit(feature_frame(parts["train"], nums, cats))
    names = transformer.get_feature_names_out().tolist()
    matrices = {s: transformer.transform(feature_frame(d, nums, cats)) for s, d in parts.items()}
    imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    medians = parts["train"][nums].median().fillna(0).to_numpy()
    np.testing.assert_allclose(imputer.statistics_, medians)
    imputed = imputer.transform(feature_frame(parts["train"], nums, cats)[nums])
    np.testing.assert_allclose(scaler.mean_, imputed.mean(axis=0), atol=1e-10)
    assert int(np.asarray(scaler.n_samples_seen_).max()) == len(parts["train"])
    report = {"status": "verified_complete", "seed": SEED, "source_rows": len(raw), "source_sha256": sha(source_path),
        "fit_partition": "train", "fit_rows": len(parts["train"]), "all_source_rows_retained": True,
        "numeric": nums, "categorical": cats, "output_predictors": names, "partitions": {},
        "methods": "Training median + missing indicators + StandardScaler; missing category + training one-hot levels, unknowns ignored",
        "prediction_models_trained": 0, "method_selection": "fixed in advance; no label or holdout-driven selection",
        "synthetic_rows": 0, "test_performance_claimed": False,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    metadata = [c for c in raw if c.startswith(("meta_", "sens_"))]
    for split, d in parts.items():
        x = matrices[split]
        assert np.isfinite(x).all() and d.target.isin([0, 1]).all()
        assert x.shape == (len(d), len(names))
        result = pd.DataFrame(x, columns=names)
        result["target"] = d.target.to_numpy()
        result = pd.concat([result, d[metadata]], axis=1)
        result["meta_split"] = split
        result["meta_synthetic"] = 0
        path = base / f"{name}_{split}_preprocessed.csv"
        result.to_csv(path, index=False)
        np.savez_compressed(base / f"X_{split}.npz", X=x)
        np.savez_compressed(base / f"y_{split}.npz", y=d.target.to_numpy(np.int8))
        # Verify the export, rather than merely checking the in-memory frame.
        saved = pd.read_csv(path, usecols=names+["target"])
        np.testing.assert_allclose(saved[names], x, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(saved.target, d.target)
        report["partitions"][split] = {"rows": len(d), "positive_rows": int(d.target.sum()), "features": x.shape[1],
            "all_finite": True, "csv_roundtrip_passed": True, "sha256": sha(path)}
    assert sum(len(d) for d in parts.values()) == len(raw)
    assert len(set(np.concatenate(list(split_idx.values())))) == len(raw)
    report["record_partitions_disjoint"] = True
    if name == "respiratory":
        gs = {s: set(d.meta_split_group_uniquepid) for s, d in parts.items()}
        assert not (gs["train"] & gs["test"] or gs["train"] & gs["validation"] or gs["validation"] & gs["test"])
        report["patient_partitions_disjoint"] = True
        report["evaluation_scope"] = "patient-disjoint; not a hospital-held-out split"
    joblib.dump(transformer, base / "preprocessor.joblib")
    save_json(base / "feature_names.json", {"features": names})
    save_json(base / "validation.json", report)
    make_plots(name, parts, matrices, nums, names, base, imputed)
    return report


def audit_mimic():
    sources = {}
    hashes = {}
    root = ROOT / "cross_dataset/four_conditions"
    for source in ("mimic", "eicu"):
        common = None
        for condition in DATASETS:
            p = root / source / f"{condition}_model_ready.csv"
            d = pd.read_csv(p, dtype={c: "string" for c in META_MIMIC})
            assert d.meta_stay_id.is_unique and d.target.isin([0, 1]).all()
            cols = META_MIMIC + SENS_MIMIC + NUMERIC_MIMIC + CAT_MIMIC
            if common is None:
                common = d[cols].copy()
            else:
                pd.testing.assert_frame_equal(common[cols], d[cols])
            common["target_"+condition] = d.target.to_numpy(np.int8)
            hashes[f"{source}/{p.name}"] = sha(p)
        assert common.groupby("meta_patient_id").meta_split.nunique().max() == 1
        if source == "eicu":
            assert common.groupby("meta_domain_hospitalid").meta_split.nunique().max() == 1
        sources[source] = common
    directions = {}
    for source, external in (("mimic", "eicu"), ("eicu", "mimic")):
        base = root / "baseline_preprocessed" / f"{source}_to_{external}"
        record = json.loads((base / "transformation_validation.json").read_text())
        for p, expected in record["input_csv_sha256"].items():
            key = "/".join(p.replace("\\", "/").split("/")[-2:])
            assert hashes[key] == expected, key
        stats = record["fit_statistics"]
        train = sources[source].loc[sources[source].meta_split.eq("train")]
        raw = train[NUMERIC_MIMIC].to_numpy(float)
        medians = np.nanmedian(raw, axis=0)
        np.testing.assert_allclose(medians, stats["source_training_numeric_imputation_values"], rtol=1e-12, atol=1e-12)
        indicators = np.flatnonzero(np.isnan(raw).any(axis=0))
        np.testing.assert_array_equal(indicators, stats["numeric_missing_indicator_indices"])
        def impute(d):
            x = d[NUMERIC_MIMIC].to_numpy(float)
            return np.column_stack([np.where(np.isnan(x), medians, x), np.isnan(x[:, indicators]).astype(float)])
        imp_train = impute(train)
        mean, scale = imp_train.mean(axis=0), imp_train.std(axis=0)
        scale[scale == 0] = 1
        np.testing.assert_allclose(mean, stats["numeric_scaler_mean"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scale, stats["numeric_scaler_scale"], rtol=1e-10, atol=1e-10)
        modes, levels = [], []
        for i, c in enumerate(CAT_MIMIC):
            mode = train[c].dropna().mode().sort_values().iloc[0]
            cats = sorted(train[c].fillna(mode).unique())
            assert mode == stats["source_training_categorical_imputation_values"][i]
            assert cats == stats["source_training_categories"][i]
            modes.append(mode)
            levels.append(cats)
        results, matrices, parts = {}, {}, {}
        names = json.loads((base / "feature_names.json").read_text())["output_features"]
        for split in ("train", "validation", "test", "external"):
            d = (sources[external] if split == "external" else sources[source].loc[sources[source].meta_split.eq(split)]).reset_index(drop=True)
            cat = [np.column_stack([(d[c].fillna(modes[i]) == v).to_numpy(float) for v in levels[i]]) for i, c in enumerate(CAT_MIMIC)]
            independent = np.column_stack([(impute(d)-mean)/scale]+cat)
            with np.load(base / f"X_{split}.npz", allow_pickle=False) as archive:
                x = archive["X"]
            np.testing.assert_allclose(x, independent, rtol=1e-10, atol=1e-10)
            assert np.isfinite(x).all() and x.shape == (len(d), len(names))
            rows = pd.read_csv(base / f"rows_{split}.csv", dtype={c: "string" for c in META_MIMIC})
            pd.testing.assert_frame_equal(rows[META_MIMIC+SENS_MIMIC], d[META_MIMIC+SENS_MIMIC], check_dtype=False)
            with np.load(base / f"y_{split}.npz", allow_pickle=False) as ys:
                for condition in DATASETS:
                    np.testing.assert_array_equal(ys[condition], d["target_"+condition])
                    np.testing.assert_array_equal(rows["target_"+condition], d["target_"+condition])
            for fn, expected in record["partitions"][split]["sha256"].items():
                assert sha(base / fn) == expected, fn
            results[split] = {"rows": len(d), "features": x.shape[1], "all_finite": True,
                "matches_independent_train_only_transformation": True, "ids_and_labels_match": True, "saved_hashes_match": True}
            matrices[split] = x
            parts[split] = d
        directions[f"{source}_to_{external}"] = {"passed": True, "fit_rows": len(train), "partitions": results}
        if source == "mimic":
            make_plots("mimic", {k: v for k, v in parts.items() if k != "external"}, matrices,
                       NUMERIC_MIMIC, names, OUT / "mimic", imp_train)
    clinical = json.loads((root / "mimic/clinical_preprocessing_validation.json").read_text())
    for name, expected in clinical["output_sha256"].items():
        assert sha(root / "mimic" / name) == expected, name
    result = {"passed": True, "rows": len(sources["mimic"]), "patients": sources["mimic"].meta_patient_id.nunique(),
        "directions": directions, "clinical_outputs_match_previous_full_stream_audit": True,
        "raw_archives_restreamed_this_run": False, "patient_partitions_disjoint": True,
        "status": "existing baseline preprocessing verified; no rebuild needed",
        "eight_bias_mitigation_experiment": "not completed; development fragments and fixture previews are not patient results"}
    save_json(AUDIT / "mimic_validation.json", result)
    return result


def write_report(state):
    lines = ["# Dataset and plot verification", "", "Audit date: 11 November 2026 (Asia/Calcutta).", "",
        "All four original datasets have saved preprocessing outputs. They are historical experiments: test-set scores were reused when choosing preprocessing methods, so their end-to-end metrics are exploratory rather than independent test results.", "",
        "A new fixed baseline is available in `../verified_baseline/`. It fits median imputation, missing indicators, scaling and category encoding on training rows only. It retains all real source records, labels and survey metadata, adds no synthetic rows and fits no prediction models. All exports were read back and checked. This is baseline preprocessing, not a completed eight-bias mitigation comparison.", "",
        "| Dataset | Source rows | Existing output integrity | New train / validation / test |", "|---|---:|---|---|"]
    for n in DATASETS:
        a, b = state["original"][n], state["baseline"][n]
        counts = " / ".join(f"{b['partitions'][s]['rows']:,}" for s in ("train", "validation", "test"))
        lines.append(f"| {n.title()} | {b['source_rows']:,} | {'PASS' if a['file_integrity_passed'] else 'FAIL: see audit.json'} | {counts} |")
    lines += ["", "The file-integrity checks cover finite predictors, row counts, source label/ID alignment, positive finite weights, synthetic-row flags and disjoint partitions. The historical learned representations were not refitted.", "",
        "### Historical performance values", "", "These are the saved exploratory values, not newly measured independent test results. AUROC and average precision declined in all four saved experiments; lower Brier is better.", "",
        "| Dataset | AUROC before / after | Average precision before / after | Brier before / after |", "|---|---|---|---|"]
    for n in DATASETS:
        m = state["original"][n]["reported_exploratory_metrics"]
        cells = [f"{m[k]['raw data']:.4f} / {m[k]['pre-processed data']:.4f}" for k in ("AUROC", "PR-AUC", "Brier")]
        lines.append(f"| {n.title()} | " + " | ".join(cells) + " |")
    lines += ["", "MIMIC: **73,990 stays / 53,931 patients**. Existing MIMIC-to-eICU and eICU-to-MIMIC matrices were independently recomputed from source-training statistics and checked against all saved matrices, labels, IDs and hashes. Its baseline preprocessing is complete. Raw archives were not restreamed; derived clinical files match the earlier full-stream audit hashes.", "",
        "## Plots", "", "Open `../verified_baseline/index.html` for regenerated plots for all five datasets. Each chart has `plot_data.json` with counts and denominators. Histograms use shared original-unit bins and full data; after-scaling standard deviations appear separately. No tail clipping, fake density curves or undefined-correlation-as-zero displays are used.", "",
        "The older dashboards' embedded comparison numbers were checked against saved CSVs; see `audit.json`. Those plots cannot be certified as fully correct: the density view is a count-frequency polygon, tails are clipped into edge bins, and undefined correlations are displayed as zero. Separate before/after scales and sampled/oversampled rows also require care. The new plots replace these for baseline verification; old dashboards remain historical research artifacts.", "",
        "## Interpretation", "", "- Modern-only selection is an explicit restriction in the old experiment; it does not identify the best method overall.",
        "- Original heart, respiratory, diabetes and hypertension test metrics are not a clean evaluation after selection on those same test labels. Fresh preprocessing does not retrospectively repair those metrics.",
        "- Original respiratory data predict mortality in a respiratory ICU cohort; the MIMIC respiratory label is a recorded respiratory diagnosis. These are different targets.",
        "- MIMIC's eight-bias mitigation experiment has not been completed. Its earlier development plot preview is a test fixture, not a patient-data result.",
        "- New survey splits are stratified by target; respiratory splits keep patients together. They are not hospital-held-out splits. Original survey weights are retained as metadata; no population-weighted model or fairness claim is made.",
        "- Brier scores on the same evaluation cases remain comparable after resampling; worsening calibration must not be dismissed as 'not comparable'.", "",
        "## Reproduce", "", "`python preprocessing/audit_and_prepare.py`", "",
        "Machine-readable evidence: `audit.json`, `mimic_validation.json`, and each baseline folder's `validation.json` and `plot_data.json`.", ""]
    (AUDIT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for name in (*DATASETS, "mimic"):
        cards.append(f'<section><h2>{name.title()}</h2><a href="{name}/preprocessing_checks.png"><img src="{name}/preprocessing_checks.png" alt="{name} preprocessing checks"></a><p><a href="{name}/plot_data.json">Exact plot data</a></p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Verified preprocessing</title><style>body{font:16px system-ui;margin:24px auto;max-width:1500px;padding:0 20px;background:#f4f6fa;color:#172b44}section{background:white;padding:18px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}a{color:#1d5a93}</style><h1>Verified preprocessing — 11 November 2026</h1><p>Fixed train-only baseline for heart, respiratory, diabetes and hypertension; independently verified existing MIMIC baseline. These checks do not claim predictive or fairness improvement.</p><p><a href="../audit_current/REPORT.md">Audit findings and limitations</a></p>'+"".join(cards)+'</html>', encoding="utf-8")


def refresh_plots():
    """Render from the audited counts, without repeating preprocessing or CSV exports."""
    for name in (*DATASETS, "mimic"):
        base = OUT / name
        d = json.loads((base / "plot_data.json").read_text())
        assert d["histogram_count_reconciliation"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
        labels, counts = list(d["partition_rows"]), list(d["partition_rows"].values())
        axes[0, 0].bar(labels, counts, color="#3569a5")
        for i, n in enumerate(counts):
            axes[0, 0].text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
        axes[0, 0].set(ylim=(0, max(counts)*1.18), title="Rows retained in each partition", ylabel="Rows")
        missing = d["missing_train_percent"]
        axes[0, 1].barh(list(missing)[::-1], list(missing.values())[::-1], color="#ba6655")
        axes[0, 1].set(title="Missing predictors before imputation\nTraining partition", xlabel="Missing cells (%)")
        if name == "mimic":
            checked = json.loads((AUDIT / "mimic_validation.json").read_text())["directions"]["mimic_to_eicu"]["partitions"]
        else:
            checked = json.loads((base / "validation.json").read_text())["partitions"]
        assert all(checked[s]["all_finite"] and checked[s]["rows"] == n for s, n in zip(labels, counts))
        ax = axes[0, 2]
        ax.bar(labels, [0]*len(labels), color="#318b71")
        for i in range(len(labels)):
            ax.text(i, .015, "0%", ha="center")
        ax.set(ylim=(0, 1), title="Invalid values after preprocessing", ylabel="Nonfinite cells (%)")
        sds = d["numeric_train_std"]
        vs = sorted(sds, key=sds.get)[-12:]
        axes[1, 0].barh(vs, [sds[v] for v in vs], color="#318b71")
        axes[1, 0].axvline(1, color="#777", ls="--", lw=1)
        axes[1, 0].set(title="Scaled numeric features (train)", xlabel="Standard deviation; constants stay at 0")
        if name == "mimic":
            with np.load(ROOT / "cross_dataset/four_conditions/baseline_preprocessed/mimic_to_eicu/y_train.npz") as ys:
                rate_labels = list(DATASETS)
                rates = [100*float(ys[c].mean()) for c in rate_labels]
            title = "MIMIC training diagnosis labels\nNo resampling"
            axes[1, 1].tick_params(axis="x", rotation=20)
        else:
            rate_labels = labels
            rates = [100*checked[s]["positive_rows"]/checked[s]["rows"] for s in labels]
            title = "Recorded target prevalence\nNo resampling"
        axes[1, 1].bar(rate_labels, rates, color="#8071ab")
        for i, rate in enumerate(rates):
            axes[1, 1].text(i, rate+1, f"{rate:.2f}%", ha="center", fontsize=9)
        axes[1, 1].set(ylim=(0, 100), title=title, ylabel="Target = 1 (%)")
        ax = axes[1, 2]
        edges = np.asarray(d["histogram_edges"])
        for key, label in (("observed", "Observed"), ("imputed", "Imputed")):
            values = np.asarray(d[f"{key}_counts"])
            denominator = d[f"{key}_denominator"]
            assert values.sum() == denominator
            ax.stairs(100*values/denominator, edges, linewidth=1.7, label=f"{label} ({denominator:,})")
        ax.set(title=f"{d['histogram_feature']}\nImputation in original units; common bins",
               xlabel=d["histogram_feature"], ylabel="Share of each series (%)")
        ax.legend(fontsize=8)
        fig.suptitle(f"{name.title()} | verified fixed preprocessing baseline", fontsize=17)
        fig.savefig(base / "preprocessing_checks.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        d["target_prevalence_percent"] = dict(zip(rate_labels, rates))
        d["nonfinite_after_percent"] = dict.fromkeys(labels, 0)
        save_json(base / "plot_data.json", d)
    print("Refreshed all five figures from reconciled plot data", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    state = {"checked_utc": datetime.now(timezone.utc).isoformat(), "original": {}, "dashboards": {}, "baseline": {}}
    for name in DATASETS:
        print(f"Auditing {name}", flush=True)
        raw, path = read_source(name)
        state["original"][name] = audit_original(name, raw)
        state["dashboards"][name] = audit_legacy_dashboards(name)
        print(f"Building fixed baseline: {name}", flush=True)
        state["baseline"][name] = build_baseline(name, raw, path)
        save_json(AUDIT / "audit.json", state)
    print("Independently verifying existing MIMIC preprocessing", flush=True)
    state["mimic"] = audit_mimic()
    save_json(AUDIT / "audit.json", state)
    write_report(state)
    refresh_plots()
    print("Completed: " + str(AUDIT / "REPORT.md"), flush=True)


if __name__ == "__main__":
    if "--finalize" in sys.argv:
        write_report(json.loads((AUDIT / "audit.json").read_text()))
        refresh_plots()
    elif "--plots-only" in sys.argv:
        refresh_plots()
    else:
        main()
