"""Small correctness checks; no real-data model or transformation is fitted."""
import unittest
import numpy as np
import pandas as pd
from fairness_metrics_fragment import (classification_metrics, subgroup_metrics,
    reweighing_weights, CorrelationProjection)


class FairnessFragmentTests(unittest.TestCase):
    def test_metric_definitions_and_single_group_insufficiency(self):
        result = classification_metrics([0, 0, 1, 1], [.1, .8, .4, .9])
        for name in ["balanced_accuracy", "macro_f1", "balanced_f1", "f1_positive", "f1_negative",
                     "precision", "recall", "specificity", "gmean"]:
            self.assertAlmostEqual(result[name], .5)
        self.assertAlmostEqual(result["brier"], .255)
        y = np.tile([0, 1], 200)
        result = classification_metrics(y, .1 + .8 * y, pd.DataFrame({"sens_sex": ["one"] * len(y)}))
        for name in ["max_tpr_gap", "max_fpr_gap", "max_equalized_odds_gap", "max_selection_gap", "min_selection_ratio"]:
            self.assertTrue(np.isnan(result[name]))
        self.assertFalse(result["fairness_complete"])

    def test_calibration_diagnostics_identify_separation(self):
        result = classification_metrics([0, 0, 1, 1], [.1, .2, .8, .9], calibration=True)
        self.assertTrue(np.isnan(result["calibration_slope"]))
        rng = np.random.default_rng(41)
        probability = rng.uniform(.1, .9, size=2000)
        labels = rng.binomial(1, probability)
        measured = classification_metrics(labels, probability, calibration=True)
        self.assertTrue(np.isfinite(measured["calibration_slope"]))
        self.assertTrue(np.isfinite(measured["calibration_intercept"]))

    def test_protected_comparisons_and_age_audit_are_separate(self):
        y = np.tile([0, 1], 200)
        p = .1 + .8 * y
        groups = pd.DataFrame({"sens_sex": ["Female"] * 200 + ["Male"] * 200,
                               "sens_race_ethnicity": ["GroupA"] * 200 + ["GroupB"] * 200,
                               "sens_age_group": ["Young"] * 200 + ["Old"] * 200})
        result = classification_metrics(y, p, groups)
        self.assertEqual(result["max_equalized_odds_gap"], 0.)
        self.assertEqual(result["min_selection_ratio"], 1.)
        self.assertEqual(result["worst_group_balanced_accuracy"], 1.)
        self.assertTrue(result["fairness_complete"])
        table = subgroup_metrics(y, p, groups)
        self.assertTrue(table.loc[table.attribute.eq("age"), "audit_role"].eq("audit_only").all())
        # A class-insufficient group cannot support an equalized-odds comparison.
        rare_y = np.array([1] * 3 + [0] * 197 + [0, 1] * 100)
        result = classification_metrics(rare_y, .1 + .8 * rare_y, groups[["sens_sex"]])
        self.assertTrue(np.isnan(result["max_equalized_odds_gap"]))
        self.assertTrue(np.isnan(result["max_tpr_gap"]))
        self.assertFalse(result["fairness_complete"])

    def test_reweighing_independence_and_bounded_mean_one(self):
        y = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
        group = np.array(["A"] * 100 + ["B"] * 100)
        weights = reweighing_weights(y, group, smoothing=0, clip=None)
        self.assertAlmostEqual(weights.mean(), 1.)
        for g in ["A", "B"]:
            for label in [0, 1]:
                self.assertAlmostEqual(weights[(group == g) & (y == label)].sum(), 50.)
        bounded = reweighing_weights(y, group)
        self.assertAlmostEqual(bounded.mean(), 1.)
        self.assertGreaterEqual(bounded.min(), .25)
        self.assertLessEqual(bounded.max(), 4.)

    def test_projection_is_linear_and_heldout_transform_never_refits(self):
        rng = np.random.default_rng(42)
        group = np.array([0] * 100 + [1] * 100)
        sensitive = pd.DataFrame({"Female": 1 - group, "Male": group})
        X = np.column_stack([5 * group + rng.normal(size=200), group, rng.normal(size=200)])
        projection = CorrelationProjection(alpha=1.)
        residual = projection.fit_transform(X, sensitive)
        covariance = (sensitive.to_numpy() - sensitive.to_numpy().mean(axis=0)).T @ residual / len(X)
        np.testing.assert_allclose(covariance, 0, atol=1e-12)
        np.testing.assert_allclose(residual.mean(axis=0), X.mean(axis=0), atol=1e-12)
        original_coef = projection.coef_.copy()
        original_mean = projection.sensitive_mean_.copy()
        heldout = pd.DataFrame({"Male": [0, 1], "Female": [1, 0], "Unseen": [0, 1]})
        result = projection.transform(np.full((2, 3), 1e6), heldout)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(projection.coef_, original_coef)
        np.testing.assert_array_equal(projection.sensitive_mean_, original_mean)
        np.testing.assert_array_equal(CorrelationProjection(alpha=0.).fit_transform(X, sensitive), X)


if __name__ == "__main__":
    unittest.main()
