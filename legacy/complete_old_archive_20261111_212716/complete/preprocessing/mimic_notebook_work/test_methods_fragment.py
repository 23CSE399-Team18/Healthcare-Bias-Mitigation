"""Small scientific checks for the inline notebook methods."""
import unittest
import numpy as np
import methods_fragment as m


class NotebookMethodIntegrityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(240, 4))
        self.X[:, 1] *= 100
        self.X[::12, 2] = np.nan
        self.X[:, 3] = np.nan
        self.finite = np.nan_to_num(self.X)
        self.y = np.r_[np.zeros(180, int), np.ones(60, int)]

    def test_imputers_preserve_observed_values_and_do_not_refit_on_holdout(self):
        for name in ("median", "knn", "iterative_bayes", "iterative_extra_trees"):
            with self.subTest(method=name):
                imputer = m.make_imputer(name, n_estimators=8, max_iter=2).fit(self.X)
                before_center = imputer.center_.copy()
                original = imputer.transform(self.X)
                changed = self.X[:15].copy()
                changed[:, 0] = 1e8
                imputer.transform(changed)
                np.testing.assert_array_equal(before_center, imputer.center_)
                np.testing.assert_array_equal(original[~np.isnan(self.X)], self.X[~np.isnan(self.X)])
                self.assertTrue(np.isfinite(original).all())
                self.assertEqual(original.shape, self.X.shape)
                self.assertEqual(imputer.diagnostics_["validation_or_test_rows_used_for_fit"], 0)

    def test_scalers_use_training_statistics_after_extreme_holdout(self):
        for name in ("standard", "robust", "minmax", "quantile_normal", "yeo_johnson"):
            with self.subTest(method=name):
                model = m.make_scaler(name).fit(self.finite)
                original = model.transform(self.finite)
                model.transform(self.finite + 1e6)
                np.testing.assert_array_equal(original, model.transform(self.finite))

    def test_detectors_freeze_training_thresholds_and_guard_pyod_novelty(self):
        for name in ("iqr", "mad", "isolation_forest", "lof", "one_class_svm", "ecod", "copod"):
            with self.subTest(method=name):
                detector = m.fit_outlier_detector(name, self.finite, contamination=.05)
                threshold = detector.threshold_
                self.assertTrue(np.isfinite(detector.score_samples(self.finite)).all())
                self.assertLessEqual(int(detector.predict_outliers(self.finite).sum()), 12)
                if name in ("ecod", "copod"):
                    with self.assertRaises(m.MethodNotApplicable):
                        detector.score_samples(self.finite + 1)
                else:
                    detector.score_samples(self.finite + 100)
                self.assertEqual(detector.threshold_, threshold)

    def test_real_row_resampling_retains_exact_metadata_indices(self):
        for name in ("random_oversampling", "nearmiss"):
            with self.subTest(method=name):
                Xr, yr, info = m.resample_training(name, self.finite, self.y, sampling_strategy=.5)
                indices = info["original_row_indices"]
                np.testing.assert_array_equal(Xr, self.finite[indices])
                np.testing.assert_array_equal(yr, self.y[indices])
                self.assertEqual(info["synthetic_rows"], 0)
                self.assertEqual(info["holdout_rows_used"], 0)

    def test_continuous_smote_methods_refuse_categoricals_and_smotenc_preserves_levels(self):
        X = self.finite.copy()
        X[:, 3] = np.arange(len(X)) % 2
        for name in ("smote", "adasyn", "borderline_smote", "kmeans_smote"):
            with self.subTest(method=name):
                with self.assertRaises(m.MethodNotApplicable):
                    m.resample_training(name, X, self.y, categorical_indices=[3])
        Xr, yr, info = m.resample_training("smotenc", X, self.y, categorical_indices=[3])
        self.assertTrue(set(Xr[:, 3]).issubset({0, 1}))
        self.assertFalse(info["synthetic_sensitive_groups_fabricated"])
        self.assertIsNone(info["original_row_indices"])

    def test_reconstruction_metrics_never_use_naturally_missing_truth(self):
        truth = np.array([[1., np.nan], [5., 7.]])
        filled = np.array([[2., 1e9], [6., 7.]])
        mask = np.array([[True, True], [False, False]])
        result = m.reconstruction_metrics(truth, filled, mask, [2, 2])
        self.assertEqual(result["masked_observed_cells"], 1)
        self.assertEqual(result["rmse"], 1)
        self.assertEqual(result["normalized_rmse"], .5)

    def test_adaptive_selection_uses_internal_training_and_transform_models_are_frozen(self):
        from joblib import hash as model_hash
        from unittest.mock import patch
        rng = np.random.default_rng(13)
        X = rng.normal(size=(170, 4))
        X[:, 1] = X[:, 0] ** 2 + rng.normal(0, .1, len(X))
        X[::9, 1] = np.nan
        X[::11, 0] = np.nan
        X[:, 3] = np.nan
        adaptive = m.make_imputer("adaptive_inductive", fit_cap=120, n_passes=2).fit(X)
        self.assertEqual(len(adaptive.models_), 2)
        self.assertEqual(adaptive.diagnostics_["training_rows_used_for_fit"], 120)
        self.assertEqual(adaptive.diagnostics_["validation_or_test_rows_used_for_fit"], 0)
        self.assertTrue(all(d.get("inner_medians_fitted_on_inner_training_only", True)
                            for d in adaptive.selection_diagnostics_))
        scored = [d for d in adaptive.selection_diagnostics_ if "normalized_mse_by_candidate" in d]
        self.assertTrue(scored)
        self.assertTrue(all(len(d["normalized_mse_by_candidate"]) == 3 for d in scored))
        before_hash = model_hash(adaptive.models_)
        observed_before = adaptive.transform(X)
        external = X[:23].copy()
        external[:, 2] = 1e8
        # Calling fit from transform would fail, rather than silently passing a
        # frozen-parameter check after an accidental identical refit.
        frozen_models = [record["model"] for chain in adaptive.models_ for record in chain if record["model"] is not None]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for model in frozen_models:
                stack.enter_context(patch.object(model, "fit", side_effect=AssertionError("Transform attempted a refit")))
            result = adaptive.transform(external)
        self.assertEqual(before_hash, model_hash(adaptive.models_))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[~np.isnan(external)], external[~np.isnan(external)])
        np.testing.assert_array_equal(observed_before[~np.isnan(X)], X[~np.isnan(X)])
        self.assertTrue(np.all(observed_before[:, 3] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
