"""Verify no-change reports, missing measurements and portable artifacts."""
from pathlib import Path
import os
import tempfile
import unittest

WORK = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "mpl_test_config"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plots_fragment import plot_bias_comparison


class PlotFragmentTests(unittest.TestCase):
    def test_no_change_preserves_inputs_and_writes_png_svg(self):
        before = pd.Series({"missing_fraction": .03, "balanced_accuracy": .61, "unavailable": np.nan})
        after = before.copy()
        comparison = pd.DataFrame({"method": ["Baseline", "Method B", "Failed candidate"],
            "family": ["baseline", "modern", "traditional"], "mean_rank": [2., 1., np.nan],
            "balanced_accuracy": [.61, .64, np.nan], "max_equalized_odds_gap": [.12, .09, np.nan]})
        originals = [before.copy(deep=True), after.copy(deep=True), comparison.copy(deep=True)]
        parent = WORK / "_plot_test_artifacts"
        parent.mkdir(exist_ok=True)
        assert parent.resolve().is_relative_to(WORK.resolve())
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            figure = plot_bias_comparison("heart", "missing", before, after, comparison,
                selected="Baseline", status="not_detected", note="Measured validation comparison.", output_dir=Path(directory))
            self.assertEqual(tuple(figure.get_size_inches()), (16., 9.))
            png = Path(directory) / "heart__missing.png"
            svg = Path(directory) / "heart__missing.svg"
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("<svg", svg.read_text(encoding="utf-8"))
            bar_heights = [patch.get_height() for patch in figure.axes[0].patches]
            self.assertEqual(bar_heights[:2], bar_heights[2:])
            plt.close(figure)
        pd.testing.assert_series_equal(before, originals[0])
        pd.testing.assert_series_equal(after, originals[1])
        pd.testing.assert_frame_equal(comparison, originals[2])

    def test_all_missing_metrics_remain_masked_without_zero_bars(self):
        before = pd.Series([np.nan, np.nan], index=["metric_a", "metric_b"])
        figure = plot_bias_comparison("diabetes", "domain", before, before.copy(), pd.DataFrame(),
                                     selected="No change", status="not_assessable")
        self.assertEqual(len(figure.axes[0].patches), 0)
        self.assertTrue(np.ma.getmaskarray(figure.axes[1].images[0].get_array()).all())
        plt.close(figure)

    def test_no_change_status_rejects_different_values(self):
        with self.assertRaisesRegex(AssertionError, "identical Before/After"):
            plot_bias_comparison("heart", "duplicate", pd.Series({"rate": .1}), pd.Series({"rate": .0}),
                                 pd.DataFrame(), selected="No change", status="not_detected")


if __name__ == "__main__":
    unittest.main()
