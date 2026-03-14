"""Self-contained Matplotlib report fragment for measured notebook comparisons."""
from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch

BEFORE_COLOR = "#138b87"
AFTER_COLOR = "#8054b3"
NEUTRAL_COLOR = "#8996a6"
TEXT_COLOR = "#243246"


def _report_number(value):
    return "N/A" if not np.isfinite(value) else f"{value:.3g}"


def _report_column(frame, aliases):
    names = {str(name).lower().replace(" ", "_"): name for name in frame.columns}
    return next((names[name] for name in aliases if name in names), None)


def _report_family_color(value):
    value = str(value).casefold()
    if any(word in value for word in ["modern", "neural", "generative"]):
        return AFTER_COLOR
    if any(word in value for word in ["traditional", "classic", "conventional"]):
        return BEFORE_COLOR
    return NEUTRAL_COLOR


def _report_numeric(values):
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    result[~np.isfinite(result)] = np.nan
    return result


def _report_empty(axis, message):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes,
              color="#66778c", fontsize=10, wrap=True)


def plot_bias_comparison(condition, bias, before_metrics, after_metrics,
                         comparison, selected, status, note="", output_dir=None):
    """Show supplied measurements without inventing unmeasured after values.

    Metric bars and the heatmap use one shared |maximum| per Before/After
    metric pair; annotations retain the original values. Pairwise scaling is
    visual only and is not a method score. Candidate bars use the original
    common metric. Missing values stay N/A. Candidate families are read from
    the comparison table, never inferred from method names.
    """
    before = pd.Series(before_metrics).copy(deep=True)
    after = pd.Series(after_metrics).copy(deep=True)
    if not before.index.is_unique or not after.index.is_unique:
        raise ValueError("Metric names must be unique")
    status_key = str(status).strip().casefold().replace(" ", "_").replace("-", "_")
    if status_key in {"not_detected", "not_assessable"}:
        assert before.index.equals(after.index), "No-change reports require identical metric names/order"
        assert np.array_equal(_report_numeric(before), _report_numeric(after), equal_nan=True), \
            "No-change reports require identical Before/After measurements, including missing values"
    candidates = pd.DataFrame(comparison).copy(deep=True)
    if not candidates.columns.is_unique:
        raise ValueError("Candidate comparison columns must be unique")
    names = before.index.append(after.index[~after.index.isin(before.index)])
    original = np.column_stack([_report_numeric(before.reindex(names)), _report_numeric(after.reindex(names))]) if len(names) else np.empty((0, 2))
    finite = np.isfinite(original)
    pair_scale = np.max(np.where(finite, np.abs(original), 0), axis=1) if len(names) else np.array([])
    pair_scale = np.where(pair_scale > 0, pair_scale, 1)
    normalized = original / pair_scale[:, None]
    shown = np.arange(len(names))
    display_note = "All supplied metrics shown."
    if len(names) > 12:
        change = np.where(np.isfinite(normalized).all(axis=1), np.abs(normalized[:, 1] - normalized[:, 0]), -1.)
        shown = np.argsort(-change, kind="stable")[:12]
        display_note = f"Showing 12 of {len(names)} metrics, ordered by relative pair change."
    shown_names, raw, scaled = names[shown], original[shown], normalized[shown]
    method_col = _report_column(candidates, ["method", "candidate", "name"])
    family_col = _report_column(candidates, ["family", "method_family", "category"])
    ba_col = _report_column(candidates, ["balanced_accuracy", "balanced_acc", "balancedaccuracy"])
    eo_col = _report_column(candidates, ["max_equalized_odds_gap", "equalized_odds_gap", "eo_gap", "equalized_odds_difference"])
    rank_col = _report_column(candidates, ["mean_rank", "rank_mean", "rank"])
    if method_col is None:
        candidates["_report_method"] = candidates.index.astype(str) if not isinstance(candidates.index, pd.RangeIndex) else [f"Candidate {i + 1}" for i in range(len(candidates))]
        method_col = "_report_method"
    candidate_names = candidates[method_col].astype(str).to_numpy()
    candidate_colors = [_report_family_color(value) for value in candidates[family_col]] if family_col is not None else [NEUTRAL_COLOR] * len(candidates)
    selected_mask = np.array([name.casefold() == str(selected).casefold() for name in candidate_names])
    family_legend = [Patch(facecolor=BEFORE_COLOR, label="Traditional"), Patch(facecolor=AFTER_COLOR, label="Modern"),
                     Patch(facecolor=NEUTRAL_COLOR, label="Baseline / other")] if family_col is not None else [Patch(facecolor=NEUTRAL_COLOR, label="Candidate")]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.labelcolor": TEXT_COLOR, "text.color": TEXT_COLOR,
                         "xtick.color": "#586a7f", "ytick.color": "#586a7f",
                         "axes.edgecolor": "#dce3eb", "svg.fonttype": "path"}):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#f5f7fb")
        fig.subplots_adjust(left=.095, right=.975, bottom=.08, top=.855, wspace=.43, hspace=.56)
        for axis in axes.flat:
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)
        fig.text(.095, .955, str(condition).replace("_", " ").title() + "  |  " + str(bias).replace("_", " ").title(),
                 fontsize=20, fontweight="bold", va="top")
        fig.text(.095, .911, f"Status: {str(status).replace('_', ' ')}     Selected: {selected}", fontsize=11, color="#52667e")

        # Identical values create equal-height bars; NaN never creates a zero bar.
        axis = axes[0, 0]
        axis.set_title("Before / after: paired measurements", loc="left", pad=13)
        if len(shown_names):
            x = np.arange(len(shown_names))
            for column, color, label, offset in [(0, BEFORE_COLOR, "Before", -.19), (1, AFTER_COLOR, "After", .19)]:
                available = np.isfinite(scaled[:, column])
                axis.bar(x[available] + offset, scaled[available, column], width=.35, color=color, label=label, zorder=3)
                for index in range(len(x)):
                    if available[index]:
                        value = scaled[index, column]
                        axis.annotate(_report_number(raw[index, column]), (x[index] + offset, value),
                            xytext=(0, 3 if value >= 0 else -3), textcoords="offset points", ha="center",
                            va="bottom" if value >= 0 else "top", fontsize=7, rotation=90 if len(x) > 7 else 0)
                    else:
                        axis.text(x[index] + offset, .04, "N/A", ha="center", va="bottom", rotation=90,
                                  fontsize=7, color="#758497", transform=axis.get_xaxis_transform())
            axis.set_xticks(x, [textwrap.fill(str(name).replace("_", " "), 13) for name in shown_names],
                            rotation=35, ha="right", fontsize=7)
            axis.set_xlim(-.6, len(x) - .4)
            axis.set_ylim((-1.25, 1.25) if np.any(scaled[np.isfinite(scaled)] < 0) else (0, 1.25))
            axis.set_ylabel("Value / shared pair |maximum|")
            axis.axhline(0, color="#b7c2d0", lw=.7)
            axis.grid(axis="y", color="#e8edf4", lw=.6, zorder=0)
            axis.legend(handles=[Patch(facecolor=BEFORE_COLOR, label="Before"), Patch(facecolor=AFTER_COLOR, label="After")],
                        loc="upper right", frameon=False, fontsize=8)
        else:
            _report_empty(axis, "No supplied metric measurements")

        # The same color normalization applies to both heatmap columns.
        axis = axes[0, 1]
        axis.set_title("Before / after: shared-normalization heatmap", loc="left", pad=13)
        if len(shown_names):
            cmap = plt.get_cmap("viridis").with_extremes(bad="#e6ebf2")
            has_negative = bool(np.any(scaled[np.isfinite(scaled)] < 0))
            norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) if has_negative else colors.Normalize(vmin=0, vmax=1)
            axis.imshow(np.ma.masked_invalid(scaled), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            axis.set_xticks([0, 1], ["Before", "After"])
            axis.get_xticklabels()[0].set_color(BEFORE_COLOR)
            axis.get_xticklabels()[1].set_color(AFTER_COLOR)
            axis.set_yticks(np.arange(len(shown_names)), [textwrap.fill(str(name).replace("_", " "), 24) for name in shown_names], fontsize=7)
            for row in range(len(shown_names)):
                for column in [0, 1]:
                    value = scaled[row, column]
                    color = "#42516a" if not np.isfinite(value) or norm(value) > .65 else "white"
                    axis.text(column, row, _report_number(raw[row, column]), ha="center", va="center", color=color, fontsize=8)
            axis.tick_params(length=0)
            axis.text(.5, -.13, "Cell labels = original values; gray = N/A", ha="center", transform=axis.transAxes, fontsize=8, color="#66778c")
        else:
            _report_empty(axis, "No supplied metric measurements")

        axis = axes[0, 2]
        axis.set_title("Candidate trade-off: utility and parity", loc="left", pad=13)
        if ba_col is not None and eo_col is not None and len(candidates):
            accuracy, gap = _report_numeric(candidates[ba_col]), _report_numeric(candidates[eo_col])
            valid = np.isfinite(accuracy) & np.isfinite(gap)
            for index in np.flatnonzero(valid):
                axis.scatter(gap[index], accuracy[index], c=[candidate_colors[index]], s=115 if selected_mask[index] else 52,
                             marker="*" if selected_mask[index] else "o", edgecolor=TEXT_COLOR if selected_mask[index] else "white",
                             linewidth=1.1 if selected_mask[index] else .7, zorder=4)
                axis.annotate(textwrap.shorten(candidate_names[index], width=27, placeholder="…"), (gap[index], accuracy[index]),
                              xytext=(5, 4), textcoords="offset points", fontsize=6.5)
            if not valid.any():
                _report_empty(axis, "No candidates have both eligible EO and accuracy measurements")
            else:
                axis.set_xlabel("Eligible equalized-odds gap ↓")
                axis.set_ylabel("Balanced accuracy ↑")
                axis.grid(color="#e8edf4", lw=.6)
                axis.margins(x=.2, y=.25)
                axis.text(0, -.22, f"{int((~valid).sum())} candidate(s) lack paired measurements.", transform=axis.transAxes,
                          fontsize=7.5, color="#66778c")
        else:
            _report_empty(axis, "EO / balanced-accuracy comparison unavailable\nNo parity values have been invented.")

        # Every supplied candidate remains represented; missing bars are N/A.
        axis = axes[1, 0]
        preferred_metrics = [ba_col]
        preferred_metrics.extend(_report_column(candidates, [name]) for name in
            ["macro_f1", "auroc", "auprc", "rmse", "nrmse", "mae", "wasserstein"])
        preferred_metrics.extend(name for name in before.index if name in candidates and pd.api.types.is_numeric_dtype(candidates[name]))
        preferred_metrics = list(dict.fromkeys(name for name in preferred_metrics if name is not None))
        metric_col = next((name for name in preferred_metrics if np.isfinite(_report_numeric(candidates[name])).any()),
                          preferred_metrics[0] if preferred_metrics else None)
        axis.set_title("All candidates against the supplied baseline", loc="left", pad=13)
        if metric_col is not None and len(candidates):
            values = _report_numeric(candidates[metric_col])
            y_positions = np.arange(len(candidates))
            for index, value in enumerate(values):
                if np.isfinite(value):
                    axis.barh(index, value, color=candidate_colors[index], edgecolor=TEXT_COLOR if selected_mask[index] else "white",
                              linewidth=1.7 if selected_mask[index] else .4, height=.66)
                    axis.annotate(_report_number(value), (value, index), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                                  ha="left" if value >= 0 else "right", va="center", fontsize=7)
                else:
                    axis.text(.02, index, "N/A", transform=axis.get_yaxis_transform(), va="center", fontsize=7, color="#758497")
            axis.set_yticks(y_positions, [textwrap.shorten(name, width=29, placeholder="…") for name in candidate_names],
                            fontsize=7 if len(candidates) <= 18 else 5.5)
            axis.invert_yaxis()
            axis.set_xlabel(str(metric_col).replace("_", " "))
            axis.grid(axis="x", color="#e8edf4", lw=.6)
            axis.set_axisbelow(True)
            baseline = None
            for index, name in enumerate(candidate_names):
                if any(token in name.casefold() for token in ["baseline", "no change", "no mitigation", "identity", "unmodified"]):
                    if np.isfinite(values[index]):
                        baseline = values[index]
                        break
            if baseline is None and metric_col in before.index:
                candidate_baseline = _report_numeric(before.loc[[metric_col]])[0]
                if np.isfinite(candidate_baseline):
                    baseline = candidate_baseline
            if baseline is not None:
                axis.axvline(baseline, color="#3d4b61", linestyle="--", lw=1, label="Baseline")
            if np.isfinite(values).any():
                axis.margins(x=.23)
            else:
                axis.set_xlim(0, 1)
            axis.legend(handles=family_legend, loc="lower right", frameon=False, fontsize=6.5)
        else:
            _report_empty(axis, "No common measured candidate metric\nAll unavailable values remain N/A.")

        axis = axes[1, 1]
        axis.set_title("Candidate ranking / measured comparison", loc="left", pad=13)
        axis.axis("off")
        if len(candidates):
            ordered = candidates.sort_values(rank_col, kind="stable", na_position="last") if rank_col is not None else candidates
            displayed = ordered.head(12)
            columns = [method_col]
            if family_col is not None:
                columns.append(family_col)
            columns.extend(column for column in [rank_col, metric_col, ba_col, eo_col] if column is not None and column not in columns
                           and (column == rank_col or column == metric_col or np.isfinite(_report_numeric(candidates[column])).any()))
            headers = {method_col: "Method", family_col: "Family", rank_col: "Mean rank" if rank_col and "mean" in str(rank_col).lower() else "Rank",
                       ba_col: "Bal. acc.", eo_col: "EO gap"}
            cells = []
            for _, row in displayed.iterrows():
                cells.append([textwrap.fill(str(row[column]), 22 if column == method_col else 13)
                    if column in [method_col, family_col] else _report_number(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])
                    for column in columns])
            widths = [2.5 if column == method_col else 1.2 if column == family_col else 1. for column in columns]
            widths = np.asarray(widths) / sum(widths)
            table = axis.table(cellText=cells, colLabels=[headers.get(column, str(column).replace("_", " ")) for column in columns],
                               colWidths=widths, cellLoc="left", colLoc="left", loc="upper left", bbox=[0, .05, 1, .92])
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            for (row_index, column_index), cell in table.get_celld().items():
                cell.set_edgecolor("#e3e9f1")
                cell.set_linewidth(.4)
                if row_index == 0:
                    cell.set_facecolor("#e9eef6")
                    cell.get_text().set_fontweight("bold")
                elif str(displayed.iloc[row_index - 1][method_col]).casefold() == str(selected).casefold():
                    cell.set_facecolor("#eee7f6")
                else:
                    cell.set_facecolor("white" if row_index % 2 else "#f7f9fc")
            axis.text(0, -.07, (f"Table: first 12 of {len(candidates)}; every candidate appears in the bar panel."
                      if len(candidates) > 12 else f"All {len(candidates)} supplied candidates shown.")
                      + (" Ordered by supplied rank." if rank_col is not None else " Provided order; no new ranking computed."),
                      transform=axis.transAxes, fontsize=7, color="#66778c", wrap=True)
        else:
            _report_empty(axis, "No measured candidate comparison supplied")

        axis = axes[1, 2]
        axis.set_title("Scope and interpretation", loc="left", pad=13)
        axis.axis("off")
        identity_note = "No change: supplied Before and After values are identical." if status_key in {"not_detected", "not_assessable"} else "After values are the supplied measurements for the selected candidate."
        provenance = before_metrics.attrs.get("source", "MIMIC-IV v3.1") if isinstance(before_metrics, pd.Series) else "MIMIC-IV v3.1"
        cohort = before_metrics.attrs.get("cohort", "73,990 eligible ICU stays; single center (BIDMC)") if isinstance(before_metrics, pd.Series) else "73,990 eligible ICU stays; single center (BIDMC)"
        notes = [f"SOURCE  {provenance}", f"COHORT  {cohort}", identity_note, str(note).strip(), display_note,
                 "Bars / heatmap: each metric shares one Before/After scale; printed numbers retain original units. Heights across different metrics are not comparable scores.",
                 "N/A means unavailable or ineligible; it is never replaced with zero. Stars / dark outlines identify the selected method."]
        wrapped = "\n\n".join(textwrap.fill(message, width=58) for message in notes if message)
        axis.text(0, .98, wrapped, va="top", ha="left", fontsize=8, linespacing=1.3, transform=axis.transAxes)
        fig.text(.095, .022, "Measured before / after comparison  •  Teal: Before / traditional   Purple: After / modern   Gray: unavailable / baseline", fontsize=8, color="#66778c")
        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            slug = lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_") or "plot"
            basename = slug(condition) + "__" + slug(bias)
            saved_paths = []
            for extension in ["png", "svg"]:
                path = destination / (basename + "." + extension)
                fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
                saved_paths.append(str(path))
            fig.saved_paths_ = saved_paths
        return fig
