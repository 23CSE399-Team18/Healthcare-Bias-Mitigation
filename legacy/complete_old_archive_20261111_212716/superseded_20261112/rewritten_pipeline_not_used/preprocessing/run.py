"""Run the 8-bias preprocessing for one dataset.

    python code/preprocessing/run.py --dataset heart|diabetes|hypertension|respiratory

Outputs
    <dataset>/preprocessed/<dataset>_{train,validation,test}_preprocessed.csv
    <dataset>/preprocessed/methods/<bias>_method_comparison.csv, bias_summary.csv,
        pipeline_report.json, feature_schema.json, run_log.txt, dashboard_data/<bias>.json
    <dataset>/plots/01..08_<bias>_dashboard.html + index.html

Processing order (each step receives the output of the previous one):
    duplicates -> split -> missing -> outliers -> noise -> scale -> encoding/domain
    -> class imbalance -> demographic.
Every method in the method table is benchmarked for every bias; the best mean rank over the
table's metrics is selected; it is applied only when the bias is detected (otherwise before =
after). For encoding/domain and demographic bias, methods that only change model training
(DANN, IRM, transformer; adversarial debiasing, contrastive learning) cannot change the
dataset, so the best data-level method is the one applied (both are reported).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_demographic  # noqa: E402
import bias_domain  # noqa: E402
import bias_duplicate  # noqa: E402
import bias_imbalance  # noqa: E402
import bias_missing  # noqa: E402
import bias_noise  # noqa: E402
import bias_outlier  # noqa: E402
import bias_scale  # noqa: E402
import dashboard  # noqa: E402
from common import (DATASETS, ROOT, SEED, Frame, Logger, build_schema, encode, load_raw, meta_frame,  # noqa: E402
                    predictor_columns, set_seed, write_json)

MODULES = {"missing": bias_missing, "outlier": bias_outlier, "noise": bias_noise, "scale": bias_scale,
           "domain": bias_domain, "imbalance": bias_imbalance, "demographic": bias_demographic,
           "duplicate": bias_duplicate}
ORDER = ["duplicate", "missing", "outlier", "noise", "scale", "domain", "imbalance", "demographic"]


def label(c):
    return c.replace("=", ": ").replace("_", " ")


def split(raw, name):
    y = raw.target.to_numpy()
    if DATASETS[name]["group"]:
        groups = raw[DATASETS[name]["group"]].to_numpy()
        folds = np.zeros(len(raw), int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=SEED).split(raw, y, groups)):
            folds[te] = k
        assign = np.where(folds < 14, "train", np.where(folds < 17, "validation", "test"))
    else:
        idx = np.arange(len(raw))
        a, b = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
        v, t = train_test_split(b, test_size=0.50, random_state=SEED + 1, stratify=y[b])
        assign = np.empty(len(raw), object)
        assign[a], assign[v], assign[t] = "train", "validation", "test"
    return {s: raw.loc[assign == s].reset_index(drop=True) for s in ("train", "validation", "test")}


def anchor_bmi(raw):
    """Shared BMI category (1 <18.5, 2 18.5-25, 3 25-30, 4 >=30) kept as metadata for the latent stage."""
    if "bmi_category" in raw:
        return raw.bmi_category
    col = "admission_bmi" if "admission_bmi" in raw else "bmi"
    return pd.cut(raw[col], [0, 18.5, 25, 30, 1000], right=False, labels=[1, 2, 3, 4]).astype(float)


class Pipeline:
    def __init__(self, name):
        self.name = name
        self.info = DATASETS[name]
        self.out = ROOT / name / "preprocessed"
        self.plots = ROOT / name / "plots"
        for folder in (self.out, self.plots):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
        self.log = Logger(self.out / "run_log.txt")
        self.records = {}
        self.display_tf = None          # inverse of the applied scale transform, for original-unit display
        set_seed()

    # ----------------------------------------------------------------- helpers
    def display(self, X, cols):
        D = X[cols].copy()
        if self.display_tf is not None:
            tf, scal = self.display_tf
            inv = tf.inverse_transform(X[scal].to_numpy(float))
            for j, c in enumerate(scal):
                if c in D:
                    D[c] = inv[:, j]
        return D

    def snapshot(self, frames):
        return {s: f.copy() for s, f in frames.items()}

    def pick_applied(self, bias, table, best, data_level):
        if not data_level:
            return best
        ok = table.index.isin(data_level)
        if "eligible (AUROC guard)" in table:
            ok &= table["eligible (AUROC guard)"].astype(bool).to_numpy()
        cands = table.loc[ok].sort_values("mean_rank")
        return cands.index[0] if len(cands) else None

    def record(self, bias, detection, table, best, applied, protocol, notes, before, after, stat_after, extra=None,
               directions=None, info_rows=None):
        detected = detection.get("detected") if detection.get("applicable", True) else None
        rec = {"bias": bias, "detection": {k: v for k, v in detection.items() if k not in ("issues", "duplicate_mask")},
               "detected": detected, "best": best, "applied": applied if detected else None,
               "best_family": None if table is None or best is None else table.loc[best, "family"],
               "protocol": protocol, "notes": notes, "stat_after": stat_after, "extra": extra or {}}
        if table is not None:
            table.to_csv(self.out / "methods" / f"{bias}_method_comparison.csv", index_label="method")
        self.records[bias] = rec
        self.render(bias, rec, table, before, after, directions or MODULES[bias].DIRECTIONS, info_rows or [])
        self.log(f"{bias}: detected={detected} best={best} applied={rec['applied']}")

    # ----------------------------------------------------------------- dashboards
    def render(self, bias, rec, table, before, after, directions, info_rows):
        file, title = next((f, t) for k, f, t in dashboard.BIAS_FILES if k == bias)
        spec, core = rec["extra"].pop("spec"), rec["extra"].pop("core")
        det = rec["detection"]
        fmtv = lambda v: "n/a" if v is None else (f"{v:,}" if isinstance(v, (int, np.integer)) else
                                                  (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)))
        model_level_best = rec["best"] is not None and rec["applied"] is not None and rec["best"] != rec["applied"]
        if rec["detected"] is None:
            decision = f"<b>Not applicable:</b> {det.get('statistic')}. No method was applied; before = after."
        elif rec["detected"]:
            decision = (f"<b>Detected</b> ({det.get('rule')}). All {0 if table is None else len(table)} methods were run; "
                        f"<b>{rec['best']}</b> ({rec['best_family']}) has the best mean rank over the metrics. ")
            decision += (f"It changes model training rather than the rows/values, so the best data-level method "
                         f"<b>{rec['applied']}</b> was applied to the data." if model_level_best else
                         "It was applied to the data.")
        else:
            decision = (f"<b>Not detected</b> ({det.get('rule')}). All methods were still benchmarked (best mean rank: "
                        f"<b>{rec['best']}</b>), but nothing was applied: before = after.")
        payload = {
            "title": f"{self.info['title']} | {title}", "bias_title": title,
            "subtitle": (f"{self.info['title']} · {self.info['source']} · step {ORDER.index(bias) + 1} of 8 in the pipeline "
                         f"· target: {self.info['target_meaning']}"),
            "legend_before": rec["extra"].get("legend_before", f"Before {title.lower()} step"),
            "legend_after": rec["extra"].get("legend_after", f"After ({rec['applied'] or 'no change'})"),
            "units_note": rec["extra"].get("units_note", ""), "detected": rec["detected"],
            "stat_label": rec["extra"].get("stat_label", "Statistic"), "stat_before": fmtv(det.get("before")),
            "stat_after": fmtv(rec["stat_after"]), "applied": rec["applied"], "applied_method": rec["applied"],
            "best": rec["best"], "decision": decision, "protocol": rec["protocol"] or "",
            "method_notes": [[k, v] for k, v in rec["notes"].items()], "dist_note": rec["extra"].get("dist_note", ""),
            "box_note": rec["extra"].get("box_note", "Box = quartiles; whiskers = most extreme values within 1.5 IQR; "
                                         "circles = up to 40 of the most extreme points beyond them."),
            "info": info_rows, "core": core, "spec": spec,
            "table": None if table is None else dashboard.table_payload(table), "directions": directions,
            "footer": (f"Generated by code/preprocessing/run.py · rule: {det.get('rule')} · statistic: {det.get('statistic')} "
                       "· Benchmarks use training sub-samples and the validation split only; the test split is never used "
                       "for selection.")}
        write_json(self.out / "dashboard_data" / f"{bias}.json", payload)
        dashboard.render(payload, self.plots / file)

    def info_rows(self, before, after, extra=()):
        return [["Dataset", self.info["source"]], ["Partition shown", "training"],
                ["Rows before", f"{len(before):,}"], ["Rows after", f"{len(after):,}"],
                ["Variables shown", f"{len(self.schema.scalable)}"], *extra]

    def core(self, before, after, separate=False, priority=None, display_after=True):
        cols = self.schema.scalable
        Xb = self.display(before.X, cols)
        Xa = self.display(after.X, cols) if display_after else after.X[cols]
        return dashboard.core_payload(Xb, Xa, cols, {c: label(c) for c in cols}, separate, priority=priority)

    # ----------------------------------------------------------------- run
    def run(self):
        t0 = time.time()
        (self.out / "methods").mkdir(exist_ok=True)
        raw, path = load_raw(self.name)
        raw["meta_anchor_bmi_cat"] = anchor_bmi(raw)
        self.log(f"Loaded {path} rows={len(raw):,} predictors={len(predictor_columns(raw))}")
        raw = self.step_duplicate(raw)
        parts = split(raw, self.name)
        self.schema = build_schema(self.name, parts["train"])
        frames = {s: Frame(encode(df, self.schema), meta_frame(df)) for s, df in parts.items()}
        self.log("split " + ", ".join(f"{s}={len(f):,}" for s, f in frames.items()))
        if DATASETS[self.name]["group"]:
            g = {s: set(f.meta[DATASETS[self.name]["group"]]) for s, f in frames.items()}
            assert not (g["train"] & g["validation"] or g["train"] & g["test"] or g["validation"] & g["test"])
        frames = self.step_missing(frames)
        frames = self.step_outlier(frames)
        frames = self.step_noise(frames)
        frames = self.step_scale(frames)
        frames = self.step_domain(frames)
        frames = self.step_imbalance(frames)
        frames = self.step_demographic(frames)
        self.save(frames, time.time() - t0)
        return self.records

    # ----------------------------------------------------------------- steps
    def step_duplicate(self, raw):
        det = bias_duplicate.detect(raw)
        self.log(f"duplicate: {det['before']:,} exact duplicate records")
        table, best, protocol, ctx = bias_duplicate.benchmark(raw, self.log)
        after = bias_duplicate.apply(raw, best, ctx, self.log)[0] if det["detected"] else raw
        stat_after = int(bias_duplicate.detect(after)["before"])
        cols = [c for c in predictor_columns(raw) if pd.api.types.is_numeric_dtype(raw[c]) and raw[c].nunique() > 2]
        core = dashboard.core_payload(raw[cols], after[cols], cols, {c: label(c) for c in cols})
        keys = [c for c in raw.columns if c not in bias_duplicate.ID_COLS]
        dup_b, dup_a = raw.duplicated(subset=keys), after.duplicated(subset=keys)
        spec = {"title": "Duplicate records (all rows, before splitting)", "kind": "bars",
                "note": "Exact duplicate = identical in every column except the record ID. Lower chart: benchmark on injected duplicates.",
                "bars": [{"title": "Records and exact duplicates", "cats": ["Records", "Exact duplicates"],
                          "before": [len(raw), int(dup_b.sum())], "after": [len(after), int(dup_a.sum())]},
                         {"title": "Benchmark: precision / recall / F1 (injected duplicates)", "cats": list(table.index),
                          "series": [{"n": m, "v": table[m].round(4).tolist()} for m in ("Precision", "Recall", "F1-score")],
                          "opt": {"w": 560}}]}
        if DATASETS[self.name]["group"]:
            gcol = DATASETS[self.name]["group"]
            rep = int(raw[gcol].duplicated().sum())
            spec["note"] += f" {rep:,} stays are repeat ICU stays of the same patient: kept (not duplicates) and grouped in the split."
        fb, fa = Frame(raw[cols], raw[["target"]]), Frame(after[cols], after[["target"]])
        self.schema = type("S", (), {"scalable": cols})()           # temporary for info rows
        self.record("duplicate", det, table, best, best, protocol, bias_duplicate.NOTES, fb, fa, stat_after,
                    extra={"core": core, "spec": spec, "stat_label": "Exact duplicates",
                           "legend_before": "Before (all records)", "legend_after": f"After ({best if det['detected'] else 'no change'})",
                           "dist_note": "All records before splitting, original units. Bars show the share of each series' rows."},
                    info_rows=self.info_rows(fb, fa, [["Exact duplicates before", f"{int(dup_b.sum()):,}"]]))
        return after

    def step_missing(self, frames):
        schema = self.schema
        tr = frames["train"]
        too_sparse = [c for c in schema.numeric if tr.X[c].isna().mean() > 0.8]
        too_sparse_cat = [s for s, cols in schema.onehot.items() if tr.X[cols[0]].isna().mean() > 0.8]
        if too_sparse or too_sparse_cat:
            for s in too_sparse_cat:
                for f in frames.values():
                    f.X = f.X.drop(columns=schema.onehot[s])
                schema.onehot.pop(s)
                schema.dropped[s] = "more than 80% missing in training"
            for f in frames.values():
                f.X = f.X.drop(columns=too_sparse)
            schema.drop(too_sparse, "more than 80% missing in training")
            self.log(f"missing: dropped {len(too_sparse) + len(too_sparse_cat)} predictors with >80% missing: {too_sparse + too_sparse_cat}")
        det = bias_missing.detect(frames["train"], schema)
        det["detected"] = bool(det["before"] > 0) or det["detected"]
        det["rule"] = "present if any predictor cell is missing (downstream methods need complete data)"
        table, best, protocol, _ = bias_missing.benchmark(frames["train"], schema, self.log)
        before = self.snapshot(frames)
        if det["detected"]:
            new_X, _ = bias_missing.apply(frames, schema, best, self.log)
            for s in frames:
                frames[s].X = new_X[s]
        after_stat = bias_missing.detect(frames["train"], schema)["before"]
        allcols = schema.columns()
        mb, ma = before["train"].X[allcols].isna(), frames["train"].X[allcols].isna()
        top = sorted([c for c in schema.numeric] + list(schema.onehot), key=lambda c: -det["per_feature"].get(c, 0))[:35]
        rows = [c if c in schema.numeric else schema.onehot[c][0] for c in top]
        spec = {"title": "Missing-value matrix (share of missing cells per variable and record group)", "kind": "heat",
                "rows": rows, "before": dashboard.bin_matrix(mb, rows), "after": dashboard.bin_matrix(ma, rows),
                "label": "missing", "note": (f"Training split. {len(top)} predictors with the most missing values (a categorical "
                                             "predictor is shown by its first one-hot column). " +
                                             (f"Dropped before imputation (>80% missing): {', '.join(schema.dropped)}." if schema.dropped else "")),
                "bars": [{"title": "Missing share per predictor", "cats": [label(c) for c in top[:15]],
                          "before": [det["per_feature"].get(c, 0) for c in top[:15]], "after": [0.0] * min(15, len(top)),
                          "opt": {"pct": True, "w": 560}}]}
        core = self.core(before["train"], frames["train"])
        self.record("missing", det, table, best, best, protocol, bias_missing.NOTES, before["train"], frames["train"],
                    after_stat, extra={"core": core, "spec": spec, "stat_label": "Missing cells",
                                       "legend_before": "Before (observed values only)", "legend_after": f"After ({best} imputation)",
                                       "dist_note": "Training split, original units. Before = observed values only; after = observed + imputed values."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Missing cells before", f"{det['before']:.2%}"], ["Imputer fitted on", "training rows only"]]))
        return frames

    def step_outlier(self, frames):
        det = bias_outlier.detect(frames["train"], self.schema)
        table, best, protocol = bias_outlier.benchmark(frames, self.schema, det, self.log)
        before = self.snapshot(frames)
        info = {}
        if det["detected"]:
            frames, info = bias_outlier.apply(frames, self.schema, best, det, self.log)
        fences = {c: tuple(v) for c, v in det["fences"].items()}
        fb, fa = bias_outlier.flagged_cells(before["train"].X, fences), bias_outlier.flagged_cells(frames["train"].X, fences)
        after_stat = float(fa.to_numpy().mean()) if fa.size else 0.0
        rows = sorted(fences, key=lambda c: -det["per_feature"][c])[:30]
        spec = {"title": "Outlier heatmap (share of values beyond skew-adjusted fences)", "kind": "heat", "rows": rows,
                "before": dashboard.bin_matrix(fb, rows), "after": dashboard.bin_matrix(fa, rows), "label": "outlying values",
                "note": ("Training split; fences from the training data (adjusted boxplot). Flagged statistical tails are not proven "
                         "errors. The selected detector removes only training rows; validation/test rows are never removed."),
                "bars": [{"title": "Outlying-value share per variable", "cats": [label(c) for c in rows[:15]],
                          "before": [float(fb[c].mean()) for c in rows[:15]], "after": [float(fa[c].mean()) for c in rows[:15]],
                          "opt": {"pct": True, "w": 560}}]}
        core = self.core(before["train"], frames["train"], priority=rows[:6])
        self.record("outlier", det, table, best, best, protocol, bias_outlier.NOTES, before["train"], frames["train"], after_stat,
                    extra={"core": core, "spec": spec, "stat_label": "Outlying cells",
                           "dist_note": "Training split, original units (after imputation)."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Rows removed", f"{info.get('removed_training_rows', 0):,}"],
                                              ["Contamination used", f"{det['contamination']:.3f}"]]))
        return frames

    def step_noise(self, frames):
        det = bias_noise.detect(frames["train"], self.schema)
        self.log(f"noise: CL label-issue share {det['before']:.4f}")
        table, best, protocol = bias_noise.benchmark(frames["train"], self.schema, self.log)
        before = self.snapshot(frames)
        info = {}
        if det["detected"]:
            frames, info = bias_noise.apply(frames, self.schema, best, det, self.log)
            after = bias_noise.detect(frames["train"], self.schema)
            after_stat = after["before"]
            issues_after = after["issues"]
        else:
            after_stat, issues_after = det["before"], det["issues"]
        yb, ya = before["train"].meta.target.to_numpy(), frames["train"].meta.target.to_numpy()
        spec = {"title": "Label-noise screen (Confident Learning) and feature change", "kind": "bars",
                "note": ("Share of training rows whose label Confident Learning flags as a likely error, by recorded class. "
                         "These are model-disagreement flags, not verified errors. Right: mean absolute change of each "
                         "continuous predictor (training rows present in both, SD units)."),
                "bars": [{"title": "Flagged label-issue share by class", "cats": ["All", "Class 0", "Class 1"],
                          "before": [float(det["issues"].mean()), float(det["issues"][yb == 0].mean()), float(det["issues"][yb == 1].mean())],
                          "after": [float(issues_after.mean()), float(issues_after[ya == 0].mean()), float(issues_after[ya == 1].mean())],
                          "opt": {"pct": True}}]}
        common_idx = before["train"].X.index.intersection(frames["train"].X.index)
        cont = self.schema.continuous[:15]
        if len(cont):
            sd = before["train"].X[cont].std().replace(0, 1)
            change = ((frames["train"].X.loc[common_idx, cont] - before["train"].X.loc[common_idx, cont]).abs() / sd).mean()
            spec["bars"].append({"title": "Mean |after - before| per continuous predictor (SD units)", "cats": [label(c) for c in cont],
                                 "series": [{"n": "Change", "v": change.round(4).tolist()}], "opt": {"w": 560}})
        core = self.core(before["train"], frames["train"])
        self.record("noise", det, table, best, best, protocol, bias_noise.NOTES, before["train"], frames["train"], after_stat,
                    extra={"core": core, "spec": spec, "stat_label": "Label-issue share",
                           "dist_note": "Training split, original units."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Change applied", info.get("changed", "none")]]))
        return frames

    def step_scale(self, frames):
        det = bias_scale.detect(frames["train"], self.schema)
        table, best, protocol = bias_scale.benchmark(frames, self.schema, self.log)
        before = self.snapshot(frames)
        info = {}
        if det["detected"]:
            frames, info = bias_scale.apply(frames, self.schema, best, self.log)
            self.display_tf = (info["transform"], list(self.schema.scalable))
        after = bias_scale.detect(frames["train"], self.schema)
        cols = self.schema.scalable
        sdb, sda = before["train"].X[cols].std(), frames["train"].X[cols].std()
        skb = np.abs(skew(before["train"].X[cols].to_numpy(float), axis=0))
        ska = np.abs(skew(frames["train"].X[cols].to_numpy(float), axis=0))
        top = list(sdb.sort_values(ascending=False).index[:20])
        spec = {"title": "Feature scales before and after", "kind": "bars",
                "note": "Standard deviation (log axis) and |skewness| of each continuous/ordinal predictor, training split.",
                "bars": [{"title": "Standard deviation (log scale)", "cats": [label(c) for c in top],
                          "before": [float(sdb[c]) for c in top], "after": [float(sda[c]) for c in top], "opt": {"log": True, "w": 560}},
                         {"title": "|Skewness|", "cats": [label(c) for c in top],
                          "before": [float(skb[cols.index(c)]) for c in top], "after": [float(ska[cols.index(c)]) for c in top],
                          "opt": {"w": 560}}]}
        # before in original units, after in transformed units: each series on its own axis
        Xb, Xa = before["train"].X[cols], frames["train"].X[cols]
        core = dashboard.core_payload(Xb, Xa, cols, {c: label(c) for c in cols}, separate_axes=bool(det["detected"]))
        self.record("scale", det, table, best, best, protocol, bias_scale.NOTES, before["train"], frames["train"],
                    f"{after['before']:.2f}; skew {after['max_abs_skew']:.2f}",
                    extra={"core": core, "spec": spec, "stat_label": "SD ratio (max/min)",
                           "units_note": "Before = original units; after = transformed units." if det["detected"] else "",
                           "dist_note": ("Training split. Before in original units, after in transformed units, each on its own "
                                         "horizontal range." if det["detected"] else "Training split, original units.")},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Max |skewness| before", f"{det['max_abs_skew']:.2f}"],
                                              ["Data-side transform", info.get("data_side", "none")]]))
        self.records["scale"]["stat_after_value"] = after["before"]
        return frames

    def step_domain(self, frames):
        det = bias_domain.detect(frames, self.schema)
        before = self.snapshot(frames)
        if not det["applicable"]:
            table, best, protocol, applied = None, None, "Not applicable: no domain identifier.", None
            stat_after = None
        else:
            table, best, protocol, _ = bias_domain.benchmark(frames, self.schema, self.log)
            applied = self.pick_applied("domain", table, best, bias_domain.DATA_LEVEL)
            if det["detected"] and applied:
                frames, _ = bias_domain.apply(frames, self.schema, applied, self.log)
                self.schema.snap_ordinal = False
            stat_after = bias_domain.detect(frames, self.schema)["before"]
        spec = {"title": "Encoding / domain view", "kind": "bars", "bars": [],
                "note": ("Unseen category levels (validation/test values absent from training become all-zero one-hot blocks): " +
                         ", ".join(f"{s}: {sum(v.values())}" for s, v in det["unseen_levels"].items()) + ".")}
        if det["applicable"]:
            cols = self.schema.scalable
            db, da = bias_domain.domains_of(before["train"]), bias_domain.domains_of(frames["train"])
            regions = [r for r in sorted(set(db)) if r != "Unknown"]
            def smd(frame, d):
                X = self.display(frame.X, cols)
                mu, sd = X.mean(), X.std().replace(0, 1)
                return {r: ((X[d == r].mean() - mu) / sd).abs() for r in regions}
            sb, sa = smd(before["train"], db), smd(frames["train"], da)
            top = sorted(cols, key=lambda c: -max(sb[r][c] for r in regions))[:25]
            spec.update({"kind": "heat", "title": "Regional shift: |standardised mean difference| of each region vs all training rows",
                         "rows": top, "cols": regions, "axis": "regions",
                         "before": [[float(sb[r][c]) for r in regions] for c in top],
                         "after": [[float(sa[r][c]) for r in regions] for c in top], "label": "|SMD|"})
            spec["bars"].append({"title": "Domain-classifier AUROC (0.5 = regions indistinguishable)", "cats": ["AUROC"],
                                 "before": [det["before"]], "after": [stat_after]})
        core = self.core(before["train"], frames["train"])
        self.record("domain", det, table, best, applied, protocol, bias_domain.NOTES, before["train"], frames["train"], stat_after,
                    extra={"core": core, "spec": spec, "stat_label": "Domain-classifier AUROC",
                           "dist_note": "Training split, shown in original-unit equivalents (inverse of the scale step)."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Domains", ", ".join(f"{k} ({v:,})" for k, v in det.get("domains", {}).items()) or "none"]]))
        return frames

    def step_imbalance(self, frames):
        det = bias_imbalance.detect(frames["train"])
        table, best, protocol = bias_imbalance.benchmark(frames, self.schema, self.log)
        before = self.snapshot(frames)
        info = {}
        if det["detected"]:
            frames, info = bias_imbalance.apply(frames, self.schema, best, self.log)
        after = bias_imbalance.detect(frames["train"])
        def counts(f):
            y, w = f.meta.target.to_numpy(), f.meta.sample_weight.to_numpy()
            return [float(w[y == 0].sum()), float(w[y == 1].sum())]
        def by_group(f, a):
            g, y, w = f.meta[a].fillna("Unknown").astype(str).to_numpy(), f.meta.target.to_numpy(), f.meta.sample_weight.to_numpy()
            return {k: float((w * y)[g == k].sum() / w[g == k].sum()) for k in sorted(set(g)) if k != "Unknown"}
        spec = {"title": "Class distribution", "kind": "bars",
                "note": "Training split (weighted counts when sample weights are used). Validation/test keep their natural prevalence.",
                "bars": [{"title": "Class counts", "cats": ["Class 0 (negative)", "Class 1 (positive)"],
                          "before": counts(before["train"]), "after": counts(frames["train"])}]}
        for a in ("sens_sex", "sens_age_group", "sens_race_ethnicity"):
            gb, ga = by_group(before["train"], a), by_group(frames["train"], a)
            spec["bars"].append({"title": f"Positive share by {a.replace('sens_', '').replace('_', ' ')}", "cats": list(gb),
                                 "before": list(gb.values()), "after": [ga.get(k) for k in gb], "opt": {"pct": True}})
        core = self.core(before["train"], frames["train"])
        self.record("imbalance", det, table, best, best, protocol, bias_imbalance.NOTES, before["train"], frames["train"],
                    after["before"], extra={"core": core, "spec": spec, "stat_label": "Imbalance ratio",
                                            "dist_note": "Training split, original-unit equivalents; shares are per series (row counts differ)."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Minority share before", f"{det['minority_share']:.2%}"],
                                              ["Synthetic rows added", f"{info.get('synthetic_rows', 0):,}"]]))
        return frames

    def step_demographic(self, frames):
        det = bias_demographic.detect(frames, self.schema)
        self.log(f"demographic: attribute {det['attribute']} equalized-odds {det['before']:.3f}")
        table, best, protocol, eligible = bias_demographic.benchmark(frames, self.schema, det, self.log)
        applied = self.pick_applied("demographic", table, best, bias_demographic.DATA_LEVEL)
        before = self.snapshot(frames)
        if det["detected"] and applied:
            frames, _ = bias_demographic.apply(frames, self.schema, applied, det, self.log)
        after = bias_demographic.detect(frames, self.schema)
        attr = det["attribute"]
        def rates(f, a):
            g, y, w = f.meta[a].fillna("Unknown").astype(str).to_numpy(), f.meta.target.to_numpy(), f.meta.sample_weight.to_numpy()
            return {k: float((w * y)[g == k].sum() / w[g == k].sum()) for k in sorted(set(g)) if k != "Unknown" and (g == k).sum() >= 30}
        rb, ra = rates(before["train"], attr), rates(frames["train"], attr)
        rows = []
        for a, m in det["metrics"].items():
            m2 = after["metrics"][a]
            rows.append(f"<tr><td>{a.replace('sens_', '')}</td>" + "".join(
                f"<td>{m[k]:.3f}</td><td>{m2[k]:.3f}</td>" for k in ("spd", "di", "eod", "eodds")) + "</tr>")
        tbl = ("<table class='cmp'><thead><tr><th>Attribute</th><th>SPD before</th><th>SPD after</th><th>DI before</th><th>DI after</th>"
               "<th>EOD before</th><th>EOD after</th><th>Eq. odds before</th><th>Eq. odds after</th></tr></thead><tbody>"
               + "".join(rows) + "</tbody></table>")
        spec = {"title": f"Demographic view (mitigated attribute: {attr.replace('sens_', '')})", "kind": "bars", "table": tbl,
                "note": ("Left: weighted outcome rate by group in the training data. Right: equalized-odds difference of the reference MLP "
                         "trained on the data before vs after this step, scored on the validation split (table: all audited attributes; "
                         "age is audited only)."),
                "bars": [{"title": f"Outcome rate by {attr.replace('sens_', '')} (training)", "cats": list(rb),
                          "before": list(rb.values()), "after": [ra.get(k) for k in rb], "opt": {"pct": True}},
                         {"title": "Equalized-odds difference (validation)", "cats": [a.replace("sens_", "") for a in det["metrics"]],
                          "before": [det["metrics"][a]["eodds"] for a in det["metrics"]],
                          "after": [after["metrics"][a]["eodds"] for a in det["metrics"]]}]}
        core = self.core(before["train"], frames["train"])
        self.record("demographic", det, table, best, applied, protocol, bias_demographic.NOTES, before["train"], frames["train"],
                    after["metrics"][attr]["eodds"],
                    extra={"core": core, "spec": spec, "stat_label": f"Equalized-odds diff. ({attr.replace('sens_', '')})",
                           "dist_note": "Training split, original-unit equivalents."},
                    info_rows=self.info_rows(before["train"], frames["train"],
                                             [["Mitigated attribute", attr], ["Reference AUROC", f"{det['reference_auroc']:.3f}"],
                                              ["AUROC after", f"{after['reference_auroc']:.3f}"]]))
        return frames

    # ----------------------------------------------------------------- save
    def save(self, frames, seconds):
        schema = self.schema
        files = {}
        for split_name, f in frames.items():
            out = pd.concat([f.X.reset_index(drop=True), f.meta.reset_index(drop=True)], axis=1)
            out["meta_split"] = split_name
            assert np.isfinite(f.X.to_numpy(float)).all(), f"{split_name}: non-finite features"
            path = self.out / f"{self.name}_{split_name}_preprocessed.csv"
            out.to_csv(path, index=False)
            files[split_name] = {"rows": len(out), "synthetic_rows": int(f.meta.meta_synthetic.sum()), "file": path.name}
        write_json(self.out / "feature_schema.json", {
            "features": schema.columns(), "continuous": schema.continuous, "ordinal": schema.ordinal, "binary": schema.binary,
            "onehot": schema.onehot, "dropped": schema.dropped, "target": "target", "weights": "sample_weight",
            "metadata": "meta_* and sens_* columns are never model inputs; meta_synthetic = 1 marks generated rows"})
        summary = []
        for bias in ORDER:
            r = self.records[bias]
            summary.append({"step": ORDER.index(bias) + 1, "bias": bias, "detected": r["detected"],
                            "statistic": r["detection"].get("statistic"), "before": r["detection"].get("before"),
                            "after": r["stat_after"], "best_method_mean_rank": r["best"], "best_family": r["best_family"],
                            "applied_to_data": r["applied"]})
        pd.DataFrame(summary).to_csv(self.out / "bias_summary.csv", index=False)
        write_json(self.out / "pipeline_report.json", {"dataset": self.name, "source": self.info["source"], "seed": SEED,
                   "seconds": round(seconds), "splits": files, "steps": self.records})
        tiles = []
        for key, file, title in dashboard.BIAS_FILES:
            r = self.records[key]
            status = ('<span class="badge na">not applicable</span>' if r["detected"] is None else
                      '<span class="badge yes">detected</span>' if r["detected"] else '<span class="badge no">not detected</span>')
            line = f"Best mean rank: {r['best'] or 'n/a'} | applied: {r['applied'] or 'none (before = after)'}"
            tiles.append({"file": file, "title": title, "status": status, "line": line})
        rows = "".join(f"<tr><td>{s['step']}</td><td>{s['bias']}</td><td>{'n/a' if s['detected'] is None else ('yes' if s['detected'] else 'no')}</td>"
                       f"<td class='txt'>{s['best_method_mean_rank'] or 'n/a'}</td><td class='txt'>{s['applied_to_data'] or 'none'}</td></tr>" for s in summary)
        counts = " / ".join(f"{files[s]['rows']:,}" for s in ("train", "validation", "test"))
        dashboard.render_index(self.plots / "index.html", f"{self.info['title']} bias dashboards",
                               f"{self.info['title']} — individual bias dashboards",
                               f"{self.info['source']} · preprocessed train / validation / test rows: {counts}",
                               "<div class='tbl-wrap'><table class='cmp'><thead><tr><th>Step</th><th>Bias</th><th>Detected</th>"
                               "<th>Best mean rank</th><th>Applied to data</th></tr></thead><tbody>" + rows + "</tbody></table></div>",
                               tiles, "Each tile opens one bias. Blue = before that step, green = after it.")
        self.log(f"saved: {counts} rows; {seconds/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    Pipeline(args.dataset).run()


if __name__ == "__main__":
    main()
