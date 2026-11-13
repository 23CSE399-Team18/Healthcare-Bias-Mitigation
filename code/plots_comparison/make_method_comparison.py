"""Method-comparison plots for every detected bias of the four datasets.

Reads the comparison table that each pre-processing notebook already saved inside its dashboard
(<dataset>/plots/0X_*_dashboard.html -> const D.comparison: metrics, directions, every method's values,
mean rank, selected method). Nothing is re-run. One PNG per bias goes to <dataset>/plots_comparison/.

Each figure shows, for every method that was run:
  A  mean rank over all metrics (lower = better)          C  every metric as its own panel (raw values)
  B  rank of every method on every metric (1 = best)      D  head-to-head: metrics the applied method wins / ties / loses
and a text panel stating, from the numbers, why the applied method was chosen.
Method family labels are deliberately not shown (user request 2026-11-13).
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"heart": "Heart disease · CDC BRFSS 2022", "diabetes": "Diabetes · CDC NHANES 2021-23",
            "hypertension": "Hypertension · CDC NHIS 2023", "respiratory": "Respiratory · eICU-CRD v2.0"}

# colours: one accent for the applied method, neutral for every other method (identity is also written in text)
SEL, OTHER, REF = "#2a78d6", "#c3c2bb", "#8a8984"
INK, INK2, MUTED, GRID, SURF = "#1d1d1b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
WIN, LOSS, TIE = "#2a78d6", "#e34948", "#d6d5cf"
BLUES = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#e6f0fc"]

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def load_comparison(html: Path):
    s = html.read_text(encoding="utf-8")
    if "const D = " not in s:
        return None, None
    D, _ = json.JSONDecoder().raw_decode(s[s.index("const D = ") + 10:])
    c = D.get("comparison")
    if not c or not c.get("rows"):
        return None, D
    return c, D


def rank_table(c):
    """Per-metric ranks exactly as the notebooks rank them (average ranks for ties; reference rows not ranked)."""
    rows = [r for r in c["rows"] if r["family"] != "reference"]
    vals = pd.DataFrame([r["values"] for r in rows], index=[r["method"] for r in rows], columns=c["metrics"], dtype=float)
    ranks = pd.DataFrame(index=vals.index)
    for m, d in zip(c["metrics"], c["dirs"]):
        v = vals[m]
        if isinstance(d, (list, tuple)):
            key = v.abs() if d[0] == "absmin" else (v - float(d[1])).abs()
            ranks[m] = key.rank(method="average", ascending=True)
        else:
            ranks[m] = v.rank(method="average", ascending=(d == "min"))
    mean_rank = ranks.mean(1)
    saved = pd.Series({r["method"]: r["mean_rank"] for r in rows}, dtype=float)
    assert np.allclose(mean_rank.loc[saved.index], saved, atol=1e-9), "rank recomputation differs from the notebook"
    order = list(saved.index)          # the notebook's order: sorted by mean rank, ties kept as the notebook broke them
    ref = [r for r in c["rows"] if r["family"] == "reference"]
    return vals.loc[order], ranks.loc[order], saved.loc[order], ref


def arrow(d):
    if isinstance(d, (list, tuple)):
        return "closest to 0" if d[0] == "absmin" else f"closest to {d[1]}"
    return "lower is better" if d == "min" else "higher is better"


def fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


SHORT = {"Domain-generalisation transformer": "DG transformer", "Autoencoder (domain-conditional)": "Domain-cond. autoencoder",
         "Counterfactual recourse (DiCE)": "DiCE counterfactuals", "Class-Balanced Focal Loss": "Class-balanced focal loss",
         "PowerTransform (Yeo-Johnson)": "Yeo-Johnson power", "Confident Learning (Cleanlab)": "Confident Learning",
         "Ditto (transformer matcher)": "Ditto matcher", "GBER (graph-based ER)": "GBER (graph ER)",
         "GNN entity resolution": "GNN entity resolution", "OT domain adaptation": "OT domain adaptation",
         "No adaptation (raw predictors)": "No adaptation", "No mitigation (reference)": "No mitigation",
         "Source-domain AUROC (accuracy guard)": "Source-domain AUROC", "Correlation preservation |Δr|": "Correlation |Δr|",
         "Activation variance |log σ²|": "Activation var |log σ²|", "Label-error detection F1": "Label-error F1"}


def short(name, n=26):
    name = SHORT.get(name, name)
    return name if len(name) <= n else name[: n - 1] + "…"


def why_text(c, vals, ranks, mean_rank, best):
    n = len(mean_rank)
    pos = list(mean_rank.index).index(best) + 1
    firsts = [m for m in c["metrics"] if ranks.loc[best, m] == ranks[m].min()]
    lines = [f"Applied method: {best}  ·  mean rank {mean_rank[best]:.2f}  ·  position {pos} of {n} methods"]
    if firsts:
        lines.append(f"Best (or joint best) of all {n} methods on {len(firsts)} of {len(c['metrics'])} metrics: " + ", ".join(firsts) + ".")
    else:
        lines.append(f"Not first on any single metric; it is chosen for its balance across all {len(c['metrics'])} metrics.")
    h2h = []
    for o in mean_rank.index:
        if o == best:
            continue
        w = int((ranks.loc[best] < ranks.loc[o]).sum()); l = int((ranks.loc[best] > ranks.loc[o]).sum())
        h2h.append((o, w, l))
    beats = [o for o, w, l in h2h if w > l]
    lines.append(f"Head-to-head it wins more metrics than it loses against {len(beats)} of the {n - 1} other methods.")
    if pos == 1:
        lines.append("It has the lowest mean rank of every method run, so it is the best overall on this bias.")
    else:
        ahead = list(mean_rank.index[: pos - 1])
        lines.append("Honest note: " + ", ".join(f"{a} ({mean_rank[a]:.2f})" for a in ahead)
                     + (" has" if len(ahead) == 1 else " have") + " a lower mean rank. Those methods are benchmarks only: "
                     "the project's method table (pre_methods_metrics) allows only its listed newer methods to be applied, "
                     f"and {best} is the best-ranked of those.")
    return lines, h2h


def plot_bias(ds, html, out_dir):
    c, D = load_comparison(html)
    if c is None:
        return None
    vals, ranks, mean_rank, ref = rank_table(c)
    best = c["best"]
    methods = list(mean_rank.index)
    n, k = len(methods), len(c["metrics"])
    bias = D["title"].replace(" Bias", "")
    lines, h2h = why_text(c, vals, ranks, mean_rank, best)

    ncol = 4
    nrow_m = int(np.ceil(k / ncol))
    fig = plt.figure(figsize=(17, 8.6 + 2.55 * nrow_m))
    hr = [0.42, 3.3, 0.55] + [2.35] * nrow_m + [1.55]
    gs = fig.add_gridspec(len(hr), 12, height_ratios=hr, hspace=0.62, wspace=1.6, left=0.11, right=0.985, top=0.965, bottom=0.03)

    # ---------- header
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0, 0.95, f"{bias} bias — comparison of all {n} methods", fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(0, 0.15, f"{DATASETS[ds]}  ·  {k} metrics  ·  applied: {best}  ·  every method run on the same data split  ·  "
                     "source: the notebook's saved comparison table (no re-run)", fontsize=10.5, color=INK2, va="top")

    # ---------- A: mean rank
    ax = fig.add_subplot(gs[1, 0:4])
    y = np.arange(n)
    colors = [SEL if m == best else OTHER for m in methods]
    ax.barh(y, mean_rank.values, color=colors, height=0.62, zorder=2)
    for i, (m, v) in enumerate(mean_rank.items()):
        ax.text(v + 0.06, i, f"{v:.2f}" + ("  ← applied" if m == best else ""), va="center", fontsize=9.5,
                color=INK if m == best else INK2, fontweight="bold" if m == best else "normal")
    ax.set_yticks(y, [short(m) for m in methods]); ax.invert_yaxis()
    for t, m in zip(ax.get_yticklabels(), methods):
        if m == best:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.set_xlim(0, max(mean_rank.max() * 1.32, n * 0.75)); ax.set_xlabel("mean rank across all metrics (lower = better)")
    ax.grid(axis="x", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A  Overall: mean rank", loc="left", fontsize=12, fontweight="bold")

    # ---------- B: rank heatmap
    ax = fig.add_subplot(gs[1, 4:8])
    R = ranks.values
    for i in range(n):
        for j in range(k):
            r = R[i, j]
            step = min(int(round(r)) - 1, len(BLUES) - 1)
            col = BLUES[max(step, 0)]
            ax.add_patch(Rectangle((j, i), 0.96, 0.92, color=col, zorder=1))
            ax.text(j + 0.48, i + 0.46, f"{r:g}", ha="center", va="center", fontsize=9,
                    color="white" if step <= 3 else INK, fontweight="bold" if r == 1 else "normal")
    bi = methods.index(best)
    ax.add_patch(Rectangle((-0.04, bi - 0.04), k + 0.04, 1.0, fill=False, ec=INK, lw=1.8, zorder=3))
    ax.set_xlim(0, k); ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(k) + 0.48, [short(m, 24) for m in c["metrics"]], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(n) + 0.46, [short(m, 24) for m in methods], fontsize=8.5)
    for t, m in zip(ax.get_yticklabels(), methods):
        if m == best:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("B  Rank on every metric (1 = best; darker = better)", loc="left", fontsize=12, fontweight="bold")

    # ---------- D: head-to-head
    ax = fig.add_subplot(gs[1, 9:12])
    opp = [o for o, _, _ in h2h]
    w = np.array([x[1] for x in h2h]); l = np.array([x[2] for x in h2h]); t_ = k - w - l
    yy = np.arange(len(opp))
    ax.barh(yy, w, color=WIN, height=0.6, zorder=2, label="metrics won")
    ax.barh(yy, -l, color=LOSS, height=0.6, zorder=2, label="metrics lost")
    for i in range(len(opp)):
        ax.text(w[i] + 0.15, i, f"{w[i]}", va="center", fontsize=9, color=INK2)
        ax.text(-l[i] - 0.15, i, f"{l[i]}", va="center", ha="right", fontsize=9, color=INK2)
        if t_[i]:
            ax.text(k + 0.9, i, f"tie {t_[i]}", va="center", fontsize=8, color=MUTED)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(yy, [short(o, 20) for o in opp], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(-k - 1.2, k + 2.4); ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title(f"D  {short(best, 24)} vs each method", loc="left", fontsize=12, fontweight="bold")

    # ---------- C: one panel per metric (same method order everywhere)
    refvals = ref[0]["values"] if ref else None
    for j, (m, d) in enumerate(zip(c["metrics"], c["dirs"])):
        r_, c_ = 3 + j // ncol, (j % ncol) * 3
        ax = fig.add_subplot(gs[r_, c_:c_ + 3])
        v = vals[m].values
        ax.barh(y, np.nan_to_num(v), color=colors, height=0.62, zorder=2)
        best_i = int(np.nanargmin(ranks[m].values))
        span = np.nanmax(np.abs(np.r_[v, refvals[j] if refvals and refvals[j] is not None else np.nan]))
        lo = min(0, np.nanmin(v)); hi = max(0, np.nanmax(v))
        if refvals and refvals[j] is not None:
            lo, hi = min(lo, refvals[j]), max(hi, refvals[j])
            ax.axvline(refvals[j], color=REF, ls="--", lw=1.2, zorder=3)
        pad = (hi - lo) * 0.42 or (span or 1) * 0.4
        ax.set_xlim(lo - (pad * 0.15 if lo < 0 else 0), hi + pad)
        for i, vv in enumerate(v):
            lab = fmt(vv) + ("  ★ best" if ranks[m].values[i] == ranks[m].min() else "")
            # labels always start right of the bar end (or right of zero for negative bars) so they never hit the axis labels
            ax.text(max(vv if np.isfinite(vv) else 0, 0) + (hi - lo) * 0.02, i, lab, va="center", ha="left", fontsize=8,
                    color=INK if methods[i] == best else INK2, fontweight="bold" if methods[i] == best else "normal")
        ax.set_yticks(y, [short(mm, 20) for mm in methods] if c_ == 0 else [""] * n, fontsize=8.2)
        ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{short(m, 30)}  ({arrow(d)})", loc="left", fontsize=10, fontweight="bold")
    fig.text(0.11, 0.965 - 0.935 * (sum(hr[:2]) + 0.42) / sum(hr),
             "C  Every metric, raw values (blue = applied method; ★ = best value"
             + ("; dashed line = no-mitigation reference" if ref else "") + ")",
             fontsize=12, fontweight="bold", color=INK)

    # ---------- why panel
    ax = fig.add_subplot(gs[-1, :]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02", transform=ax.transAxes,
                                fc="#eef4fc", ec="#9ec5f4", lw=1.2))
    ax.text(0.012, 0.86, "Why this method", fontsize=12.5, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
    body = "\n".join(textwrap.fill("•  " + s, 175, subsequent_indent="    ") for s in lines)
    ax.text(0.012, 0.66, body, fontsize=10, color=INK, transform=ax.transAxes, va="top", linespacing=1.5)

    out = out_dir / f"{html.stem.replace('_dashboard', '')}_method_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    pos = methods.index(best) + 1
    return {"dataset": ds, "bias": bias, "applied": best, "mean_rank": round(float(mean_rank[best]), 3),
            "position": f"{pos} of {n}", "lowest_mean_rank": methods[0], "file": out.name}


def overview(ds, rows, out_dir):
    """One summary figure per dataset: position of the applied method among all methods, per bias."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.62 * len(rows)))
    for i, r in enumerate(rows):
        c, _ = load_comparison(ROOT / ds / "plots" / (r["file"].replace("_method_comparison.png", "_dashboard.html")))
        vals, ranks, mean_rank, _ = rank_table(c)
        for m, v in mean_rank.items():
            ax.scatter(v, i, s=140 if m == r["applied"] else 60, color=SEL if m == r["applied"] else OTHER,
                       zorder=3 if m == r["applied"] else 2, edgecolor=SURF, linewidth=2)
        ax.text(mean_rank[r["applied"]], i - 0.33, r["applied"], ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r["bias"] for r in rows]); ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xlabel("mean rank across the bias's metrics (lower = better); each dot = one method")
    ax.grid(axis="x", color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{DATASETS[ds]} — where the applied method sits among all methods, per detected bias",
                 loc="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "00_overview_applied_vs_all_methods.png", dpi=130)
    plt.close(fig)


def main():
    summary = []
    for ds in DATASETS:
        out_dir = ROOT / ds / "plots_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for html in sorted((ROOT / ds / "plots").glob("0*_dashboard.html")):
            r = plot_bias(ds, html, out_dir)
            if r:
                rows.append(r); print(f"  {ds:12s} {r['bias']:18s} -> {r['file']}")
            else:
                print(f"  {ds:12s} {html.stem:40s} no methods run (bias not detected)")
        overview(ds, rows, out_dir)
        pd.DataFrame(rows).to_csv(out_dir / "method_comparison_summary.csv", index=False)
        summary += rows
    print(pd.DataFrame(summary)[["dataset", "bias", "applied", "position", "lowest_mean_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
