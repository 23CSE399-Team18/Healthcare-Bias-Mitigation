"""Assembly fragment: all definitions are copied INLINE into the MIMIC notebook.

This file is temporary assembly material, not a runtime notebook dependency.
Factories never receive validation/test rows or labels. Diagnostics state when
an expensive method fits a deterministic subset of TRAINING rows.
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis, spearmanr
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer


class MethodNotApplicable(ValueError):
    """Record a method as inapplicable; never silently substitute another method."""


def _numeric_matrix(X, allow_missing=False):
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or not len(array) or not array.shape[1]:
        raise ValueError("Expected a nonempty two-dimensional predictor matrix")
    if np.isinf(array).any() or (not allow_missing and np.isnan(array).any()):
        raise ValueError("Unexpected nonfinite predictors")
    return array


def _method_key(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


METHOD_METADATA = {
    "median": {"family": "established", "label": "Training median"},
    "knn": {"family": "established", "label": "Distance-weighted KNN imputation"},
    "iterative_bayes": {"family": "established", "label": "Chained Bayesian-ridge imputation; one deterministic completed dataset"},
    "iterative_extra_trees": {"family": "ensemble_baseline", "label": "ExtraTrees chained imputation, MissForest-style; not the original MissForest algorithm"},
    "adaptive_inductive": {"family": "modern_research_adaptation", "label": "Adaptive inductive chained imputation inspired by HyperImpute (2022); new research adaptation, not its plugin or paper reproduction"},
    "standard": {"family": "established", "label": "Training mean/SD scaling"},
    "robust": {"family": "established", "label": "Training median/IQR scaling"},
    "minmax": {"family": "established", "label": "Training min/max scaling"},
    "quantile_normal": {"family": "established", "label": "Training empirical quantile-to-normal transformation"},
    "yeo_johnson": {"family": "established_2000", "label": "Yeo-Johnson power transformation"},
    "iqr": {"family": "established", "label": "IQR-distance anomaly score"},
    "mad": {"family": "established", "label": "Median absolute deviation anomaly score"},
    "isolation_forest": {"family": "established_2008", "label": "Isolation Forest"},
    "lof": {"family": "established_2000", "label": "Local Outlier Factor"},
    "one_class_svm": {"family": "established_2001", "label": "One-Class SVM"},
    "ecod": {"family": "modern_2022", "label": "Official PyOD ECOD, training-row screening only"},
    "copod": {"family": "modern_2020", "label": "Official PyOD COPOD, training-row screening only"},
    "random_oversampling": {"family": "established", "label": "Random oversampling; exact real-row duplicates"},
    "nearmiss": {"family": "established", "label": "NearMiss-1 undersampling of real rows"},
    "smote": {"family": "established_2002", "label": "SMOTE; continuous-only arm"},
    "adasyn": {"family": "established_2008", "label": "ADASYN; continuous-only arm"},
    "borderline_smote": {"family": "established_2005", "label": "Borderline-SMOTE-1; continuous-only arm"},
    "kmeans_smote": {"family": "established_2018", "label": "KMeans-SMOTE; continuous-only arm, not a current SOTA claim"},
    "smotenc": {"family": "categorical_aware_established", "label": "Official SMOTENC for explicitly declared categorical columns"},
}


class TrainOnlyImputer:
    """Inductive numeric imputer preserving observed entries and feature count.

    KNN distances use training-only mean/SD normalization. All-empty training
    columns are filled with zero, explicitly recorded as unidentified columns.
    Bayesian/ExtraTrees iterative versions are single-imputation baselines, not
    multiple imputation with Rubin pooling or claimed uncertainty intervals.
    """
    def __init__(self, name, seed=42, fit_cap=10000, n_neighbors=5, max_iter=5, n_estimators=40):
        self.name = _method_key(name)
        self.seed, self.fit_cap = int(seed), fit_cap
        self.n_neighbors, self.max_iter, self.n_estimators = n_neighbors, max_iter, n_estimators

    def fit(self, X, y=None):
        X = _numeric_matrix(X, allow_missing=True)
        started = time.monotonic()
        self.n_features_in_ = X.shape[1]
        self.empty_columns_ = np.flatnonzero(np.isnan(X).all(axis=0))
        self.center_ = np.zeros(X.shape[1])
        self.scale_ = np.ones(X.shape[1])
        if self.name == "knn":
            for j in range(X.shape[1]):
                observed = X[np.isfinite(X[:, j]), j]
                if len(observed):
                    self.center_[j] = float(observed.mean())
                    self.scale_[j] = float(observed.std()) or 1.0
        normalized = (X - self.center_) / self.scale_
        expensive = self.name in {"knn", "iterative_extra_trees"}
        fit_rows = np.arange(len(X))
        if expensive and self.fit_cap is not None and len(X) > self.fit_cap:
            fit_rows = np.sort(np.random.default_rng(self.seed).choice(len(X), int(self.fit_cap), replace=False))
        if self.name == "median":
            self.model_ = SimpleImputer(strategy="median", keep_empty_features=True)
        elif self.name == "knn":
            self.model_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance", keep_empty_features=True)
        elif self.name in {"iterative_bayes", "iterative_extra_trees"}:
            estimator = (BayesianRidge() if self.name == "iterative_bayes" else
                         ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=12,
                                             min_samples_leaf=3, max_features=0.8,
                                             n_jobs=1, random_state=self.seed))
            self.model_ = IterativeImputer(estimator=estimator, max_iter=self.max_iter,
                initial_strategy="median", sample_posterior=False, random_state=self.seed,
                skip_complete=False, keep_empty_features=True)
        else:
            raise MethodNotApplicable("Unknown imputation method: " + self.name)
        self.model_.fit(normalized[fit_rows])
        self.diagnostics_ = {"method": self.name, "train_rows_received": len(X),
            "training_rows_used_for_fit": len(fit_rows), "fit_cap": self.fit_cap if expensive else None,
            "all_empty_training_columns": self.empty_columns_.tolist(),
            "all_empty_column_policy": "zero; unidentifiable from training data",
            "distance_normalization": "training-only mean/SD" if self.name == "knn" else "none",
            "fit_seconds": time.monotonic() - started, "validation_or_test_rows_used_for_fit": 0}
        return self

    def transform(self, X):
        X = _numeric_matrix(X, allow_missing=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Imputation feature schema changed")
        # Bound pairwise KNN memory while retaining the same fixed training model.
        blocks = []
        for start in range(0, len(X), 2000):
            block = X[start:start + 2000]
            filled = self.model_.transform((block - self.center_) / self.scale_)
            if filled.shape != block.shape:
                raise ValueError("Imputer dropped a predictor")
            filled = filled * self.scale_ + self.center_
            result = block.copy()
            missing = np.isnan(block)
            result[missing] = filled[missing]
            blocks.append(result)
        output = np.vstack(blocks)
        if not np.isfinite(output).all():
            raise ValueError("Imputation left nonfinite values")
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class AdaptiveInductiveImputer:
    """Explicit research adaptation inspired by adaptive imputation (2022).

    It is NOT the HyperImpute plugin or a reproduction/SOTA claim. On a fixed
    training subset, each numeric column selects BayesianRidge, ExtraTrees or
    histogram gradient boosting using a deterministic internal observed-row
    holdout. Inner-training medians fill predictors during selection. This
    conservative selection matrix avoids indirect internal-holdout reuse via
    previously fitted chained columns. The selected family is then refitted in
    two chained passes on observed target rows of that training subset. Frozen
    models are applied to new missing cells; transform never fits anything.
    """
    def __init__(self, seed=42, fit_cap=5000, n_passes=2, validation_fraction=0.2):
        self.seed, self.fit_cap = int(seed), int(fit_cap)
        self.n_passes, self.validation_fraction = int(n_passes), float(validation_fraction)
        if self.fit_cap < 10 or self.n_passes < 1 or not 0 < self.validation_fraction < 0.5:
            raise ValueError("Adaptive training cap/passes/internal validation fraction is invalid")

    @staticmethod
    def _column_medians(X):
        medians = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            observed = X[np.isfinite(X[:, j]), j]
            if len(observed):
                medians[j] = np.median(observed)
        return medians

    def _candidate(self, name, seed):
        if name == "bayesian_ridge":
            return BayesianRidge()
        if name == "extra_trees":
            return ExtraTreesRegressor(n_estimators=20, max_depth=10, min_samples_leaf=3,
                                       n_jobs=1, random_state=seed)
        if name == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15,
                early_stopping=False, random_state=seed)
        raise ValueError("Unknown adaptive candidate family")

    def fit(self, X, y=None):
        X = _numeric_matrix(X, allow_missing=True)
        started = time.monotonic()
        self.n_features_in_ = X.shape[1]
        rows = np.arange(len(X))
        if len(X) > self.fit_cap:
            rows = np.sort(np.random.default_rng(self.seed).choice(len(X), self.fit_cap, replace=False))
        train = X[rows].copy()
        self.fit_row_indices_ = rows
        self.medians_ = self._column_medians(train)
        self.empty_columns_ = np.flatnonzero(np.isnan(train).all(axis=0))
        self.selections_, self.selection_diagnostics_ = {}, []
        families = ("bayesian_ridge", "extra_trees", "hist_gradient_boosting")
        for column in range(train.shape[1]):
            observed_rows = np.flatnonzero(np.isfinite(train[:, column]))
            others = np.delete(np.arange(train.shape[1]), column)
            if not len(observed_rows):
                self.selections_[column] = "constant"
                self.selection_diagnostics_.append({"column": column, "selected": "constant_zero",
                    "reason": "Target column entirely unobserved in fitted training subset", "internal_validation_rows": 0})
                continue
            target = train[observed_rows, column]
            if not len(others) or np.ptp(target) < 1e-12 or len(observed_rows) < 12:
                self.selections_[column] = "constant"
                self.selection_diagnostics_.append({"column": column, "selected": "constant_training_median",
                    "reason": "No predictors, constant target, or fewer than12 observed target rows", "internal_validation_rows": 0})
                continue
            generator = np.random.default_rng(self.seed + 1009 * (column + 1))
            shuffled = generator.permutation(observed_rows)
            n_validation = max(2, int(np.ceil(len(shuffled) * self.validation_fraction)))
            inner_validation, inner_training = shuffled[:n_validation], shuffled[n_validation:]
            # These medians see inner-training rows only, and the target column
            # itself is absent from predictors. No model for another column has
            # seen the held-out target value when candidate errors are measured.
            inner_medians = self._column_medians(train[inner_training][:, others])
            inner_x = train[inner_training][:, others]
            validation_x = train[inner_validation][:, others]
            inner_x = np.where(np.isnan(inner_x), inner_medians, inner_x)
            validation_x = np.where(np.isnan(validation_x), inner_medians, validation_x)
            target_scale = float(np.std(train[inner_training, column])) or 1.0
            scores, failures = {}, {}
            for position, family in enumerate(families):
                candidate = self._candidate(family, self.seed + column * 31 + position)
                try:
                    candidate.fit(inner_x, train[inner_training, column])
                    prediction = candidate.predict(validation_x)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Candidate returned nonfinite internal-holdout predictions")
                    scores[family] = float(np.mean(((prediction - train[inner_validation, column]) / target_scale) ** 2))
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    failures[family] = type(error).__name__ + ": " + str(error)[:180]
            if not scores:
                raise ValueError(f"No adaptive candidate fit column {column}; explicit failure, not a substituted method")
            selected = min(scores, key=lambda family: (scores[family], families.index(family)))
            self.selections_[column] = selected
            self.selection_diagnostics_.append({"column": column, "selected": selected,
                "internal_training_rows": len(inner_training), "internal_validation_rows": len(inner_validation),
                "normalized_mse_by_candidate": scores, "failed_candidates": failures,
                "inner_medians_fitted_on_inner_training_only": True})
        self.models_ = []
        work = np.where(np.isnan(train), self.medians_, train)
        for iteration in range(self.n_passes):
            pass_models = []
            for column in range(train.shape[1]):
                others = np.delete(np.arange(train.shape[1]), column)
                observed = np.isfinite(train[:, column])
                missing = ~observed
                family = self.selections_[column]
                if family == "constant":
                    model = None
                    if missing.any():
                        work[missing, column] = self.medians_[column]
                else:
                    model = self._candidate(family, self.seed + 10007 * iteration + column)
                    model.fit(work[observed][:, others], train[observed, column])
                    if missing.any():
                        work[missing, column] = model.predict(work[missing][:, others])
                pass_models.append({"column": column, "predictors": others, "family": family,
                                    "model": model, "constant": float(self.medians_[column])})
            self.models_.append(pass_models)
        if not np.isfinite(work).all():
            raise ValueError("Adaptive chained fit produced nonfinite values")
        self.diagnostics_ = {"method": "adaptive_inductive", "implementation": "Explicit research adaptation inspired by HyperImpute2022; not the original plugin/paper algorithm",
            "train_rows_received": len(X), "training_rows_used_for_fit": len(train), "fit_cap": self.fit_cap,
            "chained_passes": self.n_passes, "candidate_selection_count_per_column": 1,
            "selection_scope": "Deterministic internal observed-target-row holdout on median-filled predictors; inner-training-only medians",
            "candidate_families": list(families), "column_selections": self.selection_diagnostics_,
            "all_empty_training_columns": self.empty_columns_.tolist(), "all_empty_column_policy": "zero",
            "validation_or_test_rows_used_for_fit": 0, "transform_refits_models": False,
            "fit_seconds": time.monotonic() - started}
        return self

    def transform(self, X):
        X = _numeric_matrix(X, allow_missing=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Adaptive imputation predictor schema changed")
        missing = np.isnan(X)
        work = np.where(missing, self.medians_, X)
        for pass_models in self.models_:
            for record in pass_models:
                column = record["column"]
                rows = missing[:, column]
                if not rows.any():
                    continue
                if record["model"] is None:
                    work[rows, column] = record["constant"]
                else:
                    work[rows, column] = record["model"].predict(work[rows][:, record["predictors"]])
        if not np.isfinite(work).all():
            raise ValueError("Frozen adaptive imputation produced nonfinite values")
        # Observed clinical measurements are returned bit-for-bit unchanged.
        result = X.copy()
        result[missing] = work[missing]
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_imputer(name, seed=42, **kwargs):
    aliases = {"mice": "iterative_bayes", "bayesian_ridge": "iterative_bayes",
               "missforest_style": "iterative_extra_trees", "extra_trees": "iterative_extra_trees"}
    key = _method_key(name)
    if key == "adaptive_inductive":
        return AdaptiveInductiveImputer(seed=seed, **kwargs)
    return TrainOnlyImputer(aliases.get(key, key), seed=seed, **kwargs)


def make_scaler(name, seed=42, **kwargs):
    key = _method_key(name)
    if key in {"standard", "zscore", "z_score"}:
        return StandardScaler(**kwargs)
    if key == "robust":
        return RobustScaler(**kwargs)
    if key in {"minmax", "min_max"}:
        return MinMaxScaler(**kwargs)
    if key in {"quantile", "quantile_normal"}:
        return QuantileTransformer(n_quantiles=kwargs.pop("n_quantiles", 200),
                                   output_distribution="normal", subsample=None,
                                   random_state=seed, **kwargs)
    if key in {"yeo_johnson", "power"}:
        return PowerTransformer(method="yeo-johnson", standardize=True, **kwargs)
    raise MethodNotApplicable("Unknown scaler: " + str(name))


class TrainingOutlierDetector:
    """Fixed training threshold; anomalous scores are always larger.

    ECOD/COPOD's official PyOD novelty function concatenates training and test
    batches before recomputing ECDF/skewness. To preserve frozen training-only
    statistics, those two methods screen training rows only. They do not score
    validation/test rows here. Downstream classifiers still use untouched holdouts.
    """
    def __init__(self, name, contamination=0.02, seed=42, fit_cap=15000):
        self.name, self.contamination, self.seed, self.fit_cap = _method_key(name), float(contamination), seed, fit_cap
        if not 0 < self.contamination < 0.5:
            raise ValueError("Contamination is a declared screening budget in (0,.5), not observed error prevalence")

    def fit(self, X):
        X = _numeric_matrix(X)
        started = time.monotonic()
        self.training_data_ = X.copy()
        self.active_ = np.flatnonzero(np.ptp(X, axis=0) > 0)
        self.center_ = np.median(X, axis=0)
        q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)
        self.q1_, self.q3_ = q1, q3
        self.iqr_ = np.where(q3 > q1, q3 - q1, np.std(X, axis=0))
        self.iqr_ = np.where(self.iqr_ > 1e-12, self.iqr_, 1.0)
        self.mad_ = 1.4826 * np.median(np.abs(X - self.center_), axis=0)
        self.mad_ = np.where(self.mad_ > 1e-12, self.mad_, self.iqr_)
        self.normalizer_ = RobustScaler().fit(X)
        normalized = self.normalizer_.transform(X)
        fit_rows = np.arange(len(X))
        if self.name in {"lof", "one_class_svm"} and self.fit_cap and len(X) > self.fit_cap:
            fit_rows = np.sort(np.random.default_rng(self.seed).choice(len(X), self.fit_cap, replace=False))
        self.model_ = None
        if self.name in {"ecod", "copod"}:
            if not len(self.active_):
                self.training_scores_ = np.zeros(len(X))
            else:
                try:
                    if self.name == "ecod":
                        from pyod.models.ecod import ECOD
                        self.model_ = ECOD(contamination=self.contamination, n_jobs=1)
                    else:
                        from pyod.models.copod import COPOD
                        self.model_ = COPOD(contamination=self.contamination, n_jobs=1)
                except ImportError as error:
                    raise MethodNotApplicable("Official PyOD dependency unavailable: " + self.name) from error
                self.model_.fit(X[:, self.active_])
                self.training_scores_ = np.asarray(self.model_.decision_scores_, dtype=float)
        elif self.name == "isolation_forest":
            self.model_ = IsolationForest(n_estimators=150, max_samples=min(2048, len(X)),
                random_state=self.seed, contamination="auto", n_jobs=1).fit(normalized)
            self.training_scores_ = -self.model_.score_samples(normalized)
        elif self.name == "lof":
            if len(fit_rows) < 3:
                raise MethodNotApplicable("LOF needs at least3 training rows")
            self.model_ = LocalOutlierFactor(n_neighbors=min(25, len(fit_rows) - 1), novelty=True,
                                             contamination="auto", n_jobs=1).fit(normalized[fit_rows])
            self.training_scores_ = -self.model_.score_samples(normalized)
        elif self.name == "one_class_svm":
            self.model_ = OneClassSVM(nu=self.contamination, gamma="scale").fit(normalized[fit_rows])
            self.training_scores_ = -self.model_.score_samples(normalized)
        elif self.name in {"iqr", "mad"}:
            self.training_scores_ = self._analytic_score(X)
        else:
            raise MethodNotApplicable("Unknown outlier detector: " + self.name)
        if not np.isfinite(self.training_scores_).all():
            raise ValueError("Outlier detector returned nonfinite scores")
        self.threshold_ = float(np.quantile(self.training_scores_, 1 - self.contamination))
        self.diagnostics_ = {"method": self.name, "train_rows_received": len(X),
            "training_rows_used_for_fit": len(X) if self.name in {"ecod", "copod"} else len(fit_rows),
            "training_threshold_quantile": 1 - self.contamination, "threshold": self.threshold_,
            "flagged_training_rows": int((self.training_scores_ > self.threshold_).sum()),
            "budget_is_not_verified_error_prevalence": True,
            "training_screening_only": self.name in {"ecod", "copod"},
            "validation_or_test_rows_used_for_fit": 0,
            "fit_seconds": time.monotonic() - started}
        return self

    def _analytic_score(self, X):
        if self.name == "mad":
            return np.max(np.abs(X - self.center_) / self.mad_, axis=1)
        return np.max(np.maximum(self.q1_ - X, X - self.q3_) / self.iqr_, axis=1)

    def score_samples(self, X):
        X = _numeric_matrix(X)
        if X.shape == self.training_data_.shape and np.array_equal(X, self.training_data_):
            return self.training_scores_.copy()
        if self.name in {"ecod", "copod"}:
            raise MethodNotApplicable("Official " + self.name.upper() + " is used for training-row screening only; its PyOD novelty API recomputes statistics with the test batch")
        if self.name in {"iqr", "mad"}:
            return self._analytic_score(X)
        return -self.model_.score_samples(self.normalizer_.transform(X))

    def predict_outliers(self, X):
        return self.score_samples(X) > self.threshold_

    def score(self, X):
        return self.score_samples(X)


def fit_outlier_detector(name, X_train, contamination=0.02, seed=42, **kwargs):
    aliases = {"iforest": "isolation_forest", "ocsvm": "one_class_svm"}
    key = _method_key(name)
    return TrainingOutlierDetector(aliases.get(key, key), contamination=contamination,
                                  seed=seed, **kwargs).fit(X_train)


def class_balance_metrics(y):
    labels, count = np.unique(np.asarray(y), return_counts=True)
    proportions = count / count.sum()
    return {"n": int(count.sum()), "class_counts": {str(k): int(v) for k, v in zip(labels, count)},
            "imbalance_ratio_majority_to_minority": float(count.max() / count.min()) if len(count) > 1 else None,
            "class_entropy_bits": float(-(proportions * np.log2(proportions)).sum()),
            "minority_fraction": float(proportions.min()) if len(count) > 1 else 0.0}


def resample_training(name, X, y, seed=42, sampling_strategy=0.5, categorical_indices=None):
    """Resample provided TRAINING data only; no holdout arguments exist.

    Synthetic methods never fabricate patient IDs or sensitive-group metadata.
    ROS/NearMiss return original_row_indices so real metadata can be retained.
    SMOTE variants refuse categorical columns; use their continuous-only arm.
    SMOTENC requires original categorical columns, NOT one-hot dummy groups.
    """
    key = _method_key(name)
    key = {"ros": "random_oversampling", "randomoversampler": "random_oversampling",
           "near_miss": "nearmiss", "borderlinesmote": "borderline_smote",
           "kmeanssmote": "kmeans_smote"}.get(key, key)
    X = _numeric_matrix(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("Training predictor/label lengths differ")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise MethodNotApplicable("Binary resampling comparison requires two observed training classes")
    cats = [] if categorical_indices is None else list(categorical_indices)
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= X.shape[1] for i in cats):
        raise ValueError("Invalid categorical column index")
    if cats and key in {"smote", "adasyn", "borderline_smote", "kmeans_smote"}:
        raise MethodNotApplicable("Ordinary " + key + " interpolates categorical values; use continuous-only arm or SMOTENC")
    if key == "smotenc" and (not cats or len(cats) == X.shape[1]):
        raise MethodNotApplicable("SMOTENC requires declared original categorical columns AND continuous predictors")
    baseline = class_balance_metrics(y)
    current_ratio = counts.min() / counts.max()
    info = {"method": key, "status": "completed", "before": baseline,
            "holdout_rows_used": 0, "synthetic_patient_ids_created": False,
            "synthetic_sensitive_groups_fabricated": False,
            "categorical_indices": cats, "categoricals_must_not_be_one_hot_groups": key == "smotenc"}
    if not isinstance(sampling_strategy, (float, int)) or not 0 < float(sampling_strategy) <= 1:
        raise ValueError("Use a prespecified binary minority/majority target ratio in (0,1]")
    if current_ratio >= float(sampling_strategy):
        info.update(status="not_needed", reason="Training minority fraction already meets the prespecified ratio",
                    after=baseline, original_row_indices=np.arange(len(X)), synthetic_rows=0)
        return X.copy(), y.copy(), info
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTENC
        from imblearn.under_sampling import NearMiss
    except ImportError as error:
        raise MethodNotApplicable("Official imbalanced-learn is unavailable") from error
    neighbors = min(5, int(counts.min()) - 1)
    if key not in {"random_oversampling", "nearmiss"} and neighbors < 1:
        raise MethodNotApplicable("At least two minority training rows are needed for synthetic neighbors")
    options = {"random_state": seed, "sampling_strategy": float(sampling_strategy)}
    factories = {
        "random_oversampling": lambda: RandomOverSampler(**options),
        "nearmiss": lambda: NearMiss(version=1, sampling_strategy=float(sampling_strategy), n_neighbors=min(3, int(counts.min()))),
        "smote": lambda: SMOTE(k_neighbors=neighbors, **options),
        "adasyn": lambda: ADASYN(n_neighbors=neighbors, **options),
        "borderline_smote": lambda: BorderlineSMOTE(k_neighbors=neighbors, m_neighbors=min(10, len(X) - 1), kind="borderline-1", **options),
        "kmeans_smote": lambda: KMeansSMOTE(k_neighbors=min(2, neighbors), cluster_balance_threshold="auto", n_jobs=1, **options),
        "smotenc": lambda: SMOTENC(categorical_features=cats, k_neighbors=neighbors, **options),
    }
    if key not in factories:
        raise MethodNotApplicable("Unknown or unavailable resampling method: " + key)
    sampler = factories[key]()
    try:
        resampled, labels_out = sampler.fit_resample(X, y)
    except (ValueError, RuntimeError) as error:
        raise MethodNotApplicable("Method could not satisfy its neighborhood/cluster assumptions: " + str(error)) from error
    indices = getattr(sampler, "sample_indices_", None)
    generated = max(0, len(resampled) - len(X)) if indices is None else 0
    info.update(after=class_balance_metrics(labels_out), synthetic_rows=generated,
                exact_real_row_resampling=indices is not None,
                original_row_indices=None if indices is None else np.asarray(indices, dtype=int),
                synthetic_group_audits="Not performed; generated rows have no validated patient/protected-group identity" if generated else "Real-row metadata may follow original_row_indices")
    return np.asarray(resampled), np.asarray(labels_out), info


def reconstruction_metrics(truth, filled, mask, train_scale=None):
    truth, filled = np.asarray(truth, float), np.asarray(filled, float)
    mask = np.asarray(mask, bool)
    if truth.shape != filled.shape or truth.shape != mask.shape:
        raise ValueError("Reconstruction arrays and prespecified masking positions differ")
    eligible = mask & np.isfinite(truth)
    if not eligible.any():
        return {"masked_observed_cells": 0, "rmse": None, "mae": None, "normalized_rmse": None, "normalized_mae": None}
    if not np.isfinite(filled[eligible]).all():
        raise ValueError("Imputation failed at a masked evaluation cell")
    difference = filled - truth
    scale = np.ones(truth.shape[1]) if train_scale is None else np.asarray(train_scale, float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    normalized = difference / scale
    error, norm = difference[eligible], normalized[eligible]
    return {"masked_observed_cells": int(eligible.sum()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "normalized_rmse": float(np.sqrt(np.mean(norm ** 2))),
            "normalized_mae": float(np.mean(np.abs(norm))),
            "truth_scope": "Artificially hidden observed cells only; natural missing values have no reconstruction truth",
            "uncertainty_crps_or_coverage": "Not reported for deterministic single-imputation methods"}


def distribution_shift_metrics(before, after):
    """Descriptive augmentation/selection drift; never evidence of clinical truth."""
    before, after = _numeric_matrix(before), _numeric_matrix(after)
    if before.shape[1] != after.shape[1]:
        raise ValueError("Distribution comparison predictor schemas differ")
    normalized_wasserstein, ks_statistics, histogram_tvd = [], [], []
    for j in range(before.shape[1]):
        real, changed = before[:, j], after[:, j]
        scale = float(np.std(real)) or 1.0
        normalized_wasserstein.append(wasserstein_distance(real, changed) / scale)
        ks_statistics.append(ks_2samp(real, changed, method="asymp").statistic)
        internal = np.unique(np.quantile(real, np.arange(0.1, 1, 0.1)))
        edges = np.r_[-np.inf, internal, np.inf]
        p, q = np.histogram(real, edges)[0] / len(real), np.histogram(changed, edges)[0] / len(changed)
        histogram_tvd.append(0.5 * np.abs(p - q).sum())
    return {"mean_wasserstein_normalized_by_before_sd": float(np.mean(normalized_wasserstein)),
            "mean_ks_statistic": float(np.mean(ks_statistics)),
            "mean_quantile_histogram_tvd": float(np.mean(histogram_tvd)),
            "rows_before": len(before), "rows_after": len(after),
            "interpretation": "Descriptive predictor drift; bins/scales fitted to before rows, not clinical fidelity or privacy proof"}


def scaling_diagnostics(before, after):
    before, after = _numeric_matrix(before), _numeric_matrix(after)
    if before.shape != after.shape:
        raise ValueError("Scaling diagnostic requires paired rows/features")
    active = (np.std(before, axis=0) > 1e-12) & (np.std(after, axis=0) > 1e-12)
    correlation_change = None
    if active.sum() > 1:
        old, new = np.corrcoef(before[:, active], rowvar=False), np.corrcoef(after[:, active], rowvar=False)
        upper = np.triu_indices(active.sum(), k=1)
        correlation_change = float(np.mean(np.abs(old[upper] - new[upper])))
    sd = np.std(after, axis=0)
    positive_sd = sd[sd > 1e-12]
    return {"feature_sd_ratio_after": float(positive_sd.max() / positive_sd.min()) if len(positive_sd) else None,
            "mean_absolute_pearson_correlation_change": correlation_change,
            "mean_absolute_skewness_after": float(np.nanmean(np.abs(skew(after[:, active], axis=0)))) if active.any() else None,
            "mean_absolute_excess_kurtosis_after": float(np.nanmean(np.abs(kurtosis(after[:, active], axis=0)))) if active.any() else None,
            "interpretation": "Numerical shape diagnostics; normality is not a fairness or clinical-quality guarantee"}
