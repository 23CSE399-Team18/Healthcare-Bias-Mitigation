"""Method-comparison plots for every detected bias of the four datasets, drawn the way the notebooks select.

Selection rule in the notebook code (rank_methods): every method is ranked on every metric (average ranks for ties),
the ranks are averaged (mean rank, lower = better) and the applied method is the best-ranked SELECTION CANDIDATE
(the methods the project's method table allows to be applied); ties go to the first candidate in the notebook's list.
The other methods are BASELINE BENCHMARKS: run and scored on the same data, never applied.

Reads the comparison table each notebook saved inside its dashboard (<dataset>/plots/0X_*_dashboard.html ->
const D.comparison). Nothing is re-run and no value is changed. One PNG per bias -> <dataset>/plots_comparison/.

Each figure shows every method that was run (candidate methods first, baseline methods below a dashed line):
  A  rank total of every method (the selection score, segment = rank on one metric)
  B  rank of every method on every metric                C  raw value of every metric for every method
  plus a "why this method" box written from the numbers.
"""
from __future__ import annotations

import json
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

SEL, OTHER, BASE, BASE_HI = "#2a78d6", "#c3c2bb", "#dedcd5", "#eb6834"
INK, INK2, MUTED, GRID, SURF = "#1d1d1b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
WIN, LOSS = "#2a78d6", "#e34948"
BLUES = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#e6f0fc", "#f3f7fd"]

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
    fam = {r["method"]: r["family"] for r in rows}
    ref = [r for r in c["rows"] if r["family"] == "reference"]
    return vals.loc[order], ranks.loc[order], saved.loc[order], fam, ref


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


def analyse(c):
    vals, ranks, mean_rank, fam, ref = rank_table(c)
    best = c["best"]
    cand = [m for m in mean_rank.index if fam[m] == c["select_family"]]
    base = [m for m in mean_rank.index if fam[m] != c["select_family"]]
    assert cand[0] == best, f"applied {best} is not the first candidate {cand[0]}"
    return vals, ranks, mean_rank, cand, base, ref


def totals(ranks):
    """Rank total of every method = sum of its ranks over the metrics (= mean rank x number of metrics)."""
    out = {}
    for m in ranks.index:
        t = float(ranks.loc[m].sum())
        out[m] = int(round(t)) if abs(t - round(t)) < 1e-9 else round(t, 1)
    return out


def why_lines(c, ranks, mean_rank, cand, base, best, tot):
    k, M, n = len(cand), len(c["metrics"]), len(mean_rank)
    r0 = mean_rank[best]
    lines = [f"Selection rule in the notebook code: all {n} methods are ranked on each of the {M} metrics and the ranks are added up "
             f"(rank total, lower = better). The {k} candidate methods ({', '.join(short(m) for m in cand)}) can be applied; "
             f"the {len(base)} baseline methods ({', '.join(short(m) for m in base)}) are run on the same data for comparison only."]
    nxt = cand[1] if k > 1 else None
    tie = nxt is not None and np.isclose(mean_rank[nxt], r0)
    lines.append("Rank totals of the candidate methods: " + ", ".join(f"{m} {tot[m]}" for m in cand)
                 + f"  →  {best} has the lowest total and is applied"
                 + (f" (tied with {nxt}; the tie goes to the first candidate in the notebook's list)." if tie else "."))
    firsts = [m for m in c["metrics"] if ranks.loc[best, m] == ranks.loc[cand, m].min()]
    worst = {m: float(ranks.loc[m].max()) for m in cand}
    lines.append(f"{best} is the best candidate on {len(firsts)} of {M} metrics" + (f" ({', '.join(firsts)})" if firsts else "")
                 + (f"; its worst rank on any metric is {worst[best]:g} of {n}" + "".join(f", {o}'s is {worst[o]:g}" for o in cand[1:])
                    + ": the total rewards being good on every metric at once." if k > 1 and worst[best] <= min(worst[o] for o in cand[1:]) else "."))
    better = [b for b in base if mean_rank[b] < r0 - 1e-9]
    if not better:
        lines.append(f"Every baseline method has a higher rank total than {best} ({tot[best]}), so no method beats it on the rank total.")
    else:
        lines.append(", ".join(f"{b} (total {tot[b]})" for b in better) + (" has" if len(better) == 1 else " have")
                     + f" a lower rank total than {best} ({tot[best]}); " + ("it is a baseline method" if len(better) == 1 else "they are baseline methods")
                     + ", run for comparison only and not applied.")
    return lines


def group_rows(k, nb):
    """y positions: candidate methods first, a gap, then the baseline methods."""
    return np.array(list(range(k)) + [k + 0.95 + j for j in range(nb)], dtype=float)


def group_marks(ax, k, nb, x_text, fs=8.5):
    ax.text(x_text, -0.88, "CANDIDATE METHODS (can be applied)", fontsize=fs, fontweight="bold", color=SEL, va="center")
    if nb:
        ax.axhline(k - 0.05, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(x_text, k + 0.27, "BASELINE METHODS (comparison only)", fontsize=fs, fontweight="bold", color=MUTED, va="center")


def colour(m, best, cand):
    return SEL if m == best else (OTHER if m in cand else BASE)


def plot_bias(ds, html, out_dir):
    c, D = load_comparison(html)
    if c is None:
        return None
    vals, ranks, mean_rank, cand, base, ref = analyse(c)
    best = c["best"]
    k, M, n = len(cand), len(c["metrics"]), len(mean_rank)
    nb = len(base)
    order = cand + base
    ypos = group_rows(k, nb)
    ylim = (ypos[-1] + 0.6, -1.25)
    bias = D["title"].replace(" Bias", "")
    tot = totals(ranks)
    lines = why_lines(c, ranks, mean_rank, cand, base, best, tot)

    ncol = 4
    nrow_m = int(np.ceil(M / ncol))
    hr = [0.5, 0.36 * (n + 1.9) + 0.5] + [0.25 * (n + 1.7) + 0.55] * nrow_m + [2.2]
    fig = plt.figure(figsize=(17, 1.6 + sum(hr) * 1.25))
    gs = fig.add_gridspec(len(hr), 12, height_ratios=hr, hspace=0.5, wspace=1.6, left=0.11, right=0.985, top=0.975, bottom=0.02)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0, 1.0, f"{DATASETS[ds].split(' · ')[0].upper()}  ·  {bias} bias — why {best} was selected", fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(0, 0.05, f"{DATASETS[ds]}  ·  all {n} methods ({k} candidate + {nb} baseline) run on the same split, "
                     f"{M} metrics  ·  values = the notebook's saved comparison table", fontsize=10.5, color=INK2, va="top")

    # ---------- A: the selection score exactly as computed: sum of the ranks over the metrics, every method
    ax = fig.add_subplot(gs[1, 0:6])
    for y, m in zip(ypos, order):
        left = 0.0
        for j, met in enumerate(c["metrics"]):
            r = float(ranks.loc[m, met])
            shade = ("#2a78d6", "#5598e7") if m == best else (("#b9b8b1", "#d3d2cb") if m in cand else ("#d9d8d2", "#e8e7e2"))
            ax.barh(y, r, left=left, height=0.62, color=shade[j % 2], edgecolor=SURF, linewidth=2, zorder=2)
            if r >= 0.9:
                ax.text(left + r / 2, y, f"{r:g}", ha="center", va="center", fontsize=8, color="white" if m == best else INK2)
            left += r
        ax.text(left + 0.4, y, f"{tot[m]}  (mean {mean_rank[m]:.2f})" + ("   ✓ SELECTED" if m == best else ""), va="center",
                fontsize=10.5 if m == best else 9.5, color=INK if m == best else INK2, fontweight="bold" if m == best else "normal")
    ax.set_yticks(ypos, [short(m) for m in order]); ax.set_ylim(*ylim)
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(INK if m in cand else MUTED)
        if m == best:
            t.set_fontweight("bold")
    ax.set_xlim(0, max(tot.values()) * 1.6)
    group_marks(ax, k, nb, 0.3)
    ax.set_xlabel(f"rank total — one segment per metric = rank among all {n} methods (lower = better)")
    ax.grid(axis="x", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A  Selection score of every method (as computed in the code)", loc="left", fontsize=12, fontweight="bold")

    # ---------- B: the ranks that make up the score (rank among all methods)
    ax = fig.add_subplot(gs[1, 7:12])
    for y, m in zip(ypos, order):
        for j, met in enumerate(c["metrics"]):
            r = float(ranks.loc[m, met])
            step = min(int(np.ceil(r * len(BLUES) / n)) - 1, len(BLUES) - 1)
            ax.add_patch(Rectangle((j, y - 0.46), 0.96, 0.92, color=BLUES[max(step, 0)], zorder=1))
            ax.text(j + 0.48, y, f"{r:g}", ha="center", va="center", fontsize=9,
                    color="white" if step <= 1 else INK, fontweight="bold" if r == ranks[met].min() else "normal")
    yb = ypos[order.index(best)]
    ax.add_patch(Rectangle((-0.04, yb - 0.5), M + 0.04, 1.0, fill=False, ec=INK, lw=2, zorder=3))
    ax.set_xlim(0, M); ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(M) + 0.48, [short(m, 22) for m in c["metrics"]], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(ypos, [short(m, 24) for m in order], fontsize=9)
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(INK if m in cand else MUTED)
        if m == best:
            t.set_fontweight("bold")
    group_marks(ax, k, nb, 0.05, fs=8)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"B  Rank on each metric among all {n} methods (1 = best)", loc="left", fontsize=12, fontweight="bold")

    # ---------- C: every metric, every method (raw values)
    for j, (met, d) in enumerate(zip(c["metrics"], c["dirs"])):
        r_, c_ = 2 + j // ncol, (j % ncol) * 3
        ax = fig.add_subplot(gs[r_, c_:c_ + 3])
        if j == 0:
            ax.annotate("C  Raw value of every metric for every method  (blue = selected · grey = other candidate methods · "
                        "light = baseline methods · ★ = best value of that metric)", xy=(0, 1), xycoords="axes fraction",
                        xytext=(0, 22), textcoords="offset points", fontsize=12, fontweight="bold", color=INK, va="bottom")
        v = vals.loc[order, met].values.astype(float)
        ax.barh(ypos, np.nan_to_num(v), color=[colour(m, best, cand) for m in order], height=0.62, zorder=2)
        lo, hi = min(0, np.nanmin(v)), max(0, np.nanmax(v))
        ax.set_xlim(lo - (hi - lo) * 0.05 * (lo < 0), hi + (hi - lo) * 0.6 or 1)
        top = (ranks.loc[order, met] == ranks[met].min()).values
        for y, m, vv, tp in zip(ypos, order, v, top):
            ax.text(max(vv, 0) + (hi - lo) * 0.03, y, fmt(vv) + ("  ★" if tp else ""), va="center", ha="left", fontsize=8,
                    color=INK if m == best else INK2, fontweight="bold" if m == best else "normal")
        ax.set_yticks(ypos, [short(m, 20) for m in order] if c_ == 0 else [""] * n, fontsize=8)
        if c_ == 0:
            for t, m in zip(ax.get_yticklabels(), order):
                t.set_color(INK if m in cand else MUTED)
                if m == best:
                    t.set_fontweight("bold")
        if nb:
            ax.axhline(k - 0.05, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(ypos[-1] + 0.6, -0.6); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color=GRID, zorder=0); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{short(met, 30)}  ({arrow(d)})", loc="left", fontsize=10, fontweight="bold")

    # ---------- why box
    ax = fig.add_subplot(gs[-1, :]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02", transform=ax.transAxes,
                                fc="#eef4fc", ec="#9ec5f4", lw=1.2))
    ax.text(0.012, 0.93, "Why this method", fontsize=12.5, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
    body = "\n".join(textwrap.fill("•  " + s, 200, subsequent_indent="    ") for s in lines)
    ax.text(0.012, 0.76, body, fontsize=10, color=INK, transform=ax.transAxes, va="top", linespacing=1.5)

    out = out_dir / f"{html.stem.replace('_dashboard', '')}_method_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    tie = k > 1 and np.isclose(mean_rank[cand[1]], mean_rank[best])
    nlow = sum(mean_rank[b] < mean_rank[best] - 1e-9 for b in base)
    return {"dataset": ds, "bias": bias, "applied": best, "mean_rank": round(float(mean_rank[best]), 3), "rank_total": tot[best],
            "methods": n, "candidates": k, "candidate_position": 1, "tie_with": cand[1] if tie else "",
            "next_candidate": cand[1] if k > 1 else "", "next_candidate_mean_rank": round(float(mean_rank[cand[1]]), 3) if k > 1 else None,
            "next_candidate_rank_total": tot[cand[1]] if k > 1 else None,
            "best_baseline": base[0] if base else "", "best_baseline_mean_rank": round(float(mean_rank[base[0]]), 3) if base else None,
            "best_baseline_rank_total": tot[base[0]] if base else None,
            "baselines_with_lower_mean_rank": nlow, "file": out.name}


def overview(ds, rows, out_dir):
    """Per dataset: every method of every detected bias on one mean-rank axis; the selected one is marked."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.7 * len(rows)))
    for i, r in enumerate(rows):
        c, _ = load_comparison(ROOT / ds / "plots" / (r["file"].replace("_method_comparison.png", "_dashboard.html")))
        vals, ranks, mean_rank, cand, base, _ = analyse(c)
        for m in base:
            ax.scatter(mean_rank[m], i, s=60, facecolor=SURF, edgecolor=MUTED, linewidth=1.4, zorder=2)
        for m in cand:
            ax.scatter(mean_rank[m], i, s=150 if m == r["applied"] else 70, color=SEL if m == r["applied"] else OTHER,
                       zorder=3 if m == r["applied"] else 2, edgecolor=SURF, linewidth=2)
        ax.text(mean_rank[r["applied"]], i - 0.3, r["applied"] + " ✓", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r["bias"] for r in rows]); ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xlabel("mean rank over the bias's metrics (lower = better)  ·  grey = other candidate methods  ·  hollow = baseline methods (comparison only)")
    ax.grid(axis="x", color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{DATASETS[ds]} — the selected method has the lowest mean rank of the candidate methods in every bias",
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
        overview(ds, rows, out_dir)
        pd.DataFrame(rows).to_csv(out_dir / "method_comparison_summary.csv", index=False)
        summary += rows
    s = pd.DataFrame(summary)
    print(s[["dataset", "bias", "applied", "mean_rank", "next_candidate", "next_candidate_mean_rank", "tie_with",
             "best_baseline", "best_baseline_mean_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
