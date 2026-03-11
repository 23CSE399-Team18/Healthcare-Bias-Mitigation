"""Inline notebook fragment: measured classification/fairness and preprocessing.

Copy this whole fragment into the notebook; it has no local-helper dependency.
These diagnostics do not establish clinical validity or causal fairness.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, log_loss, roc_auc_score)


def _validated_predictions(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels/probabilities must be nonempty, aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Require binary labels and finite probabilities in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    return y.astype(int), p, (p >= threshold).astype(int)


def _base_classification_metrics(y, p, threshold=.5, calibration=False):
    y, p, prediction = _validated_predictions(y, p, threshold)
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    positive, negative = tp + fn, tn + fp
    recall = tp / positive if positive else np.nan
    specificity = tn / negative if negative else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    both_classes = positive > 0 and negative > 0
    # Fixed equal-width bins. Probability exactly 1 belongs to the last bin.
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float((bins == b).mean()) * abs(float(p[bins == b].mean()) - float(y[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    result = {
        "n": len(y), "n_positive": positive, "n_negative": negative,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "auprc": float(average_precision_score(y, p)) if positive else np.nan,
        "balanced_accuracy": (recall + specificity) / 2 if both_classes else np.nan,
        "macro_f1": (f1_positive + f1_negative) / 2,
        "f1_positive": f1_positive, "f1_negative": f1_negative,
        "balanced_f1": (f1_positive + f1_negative) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
        "tpr": recall, "fpr": fp / negative if negative else np.nan,
        "gmean": float(np.sqrt(recall * specificity)) if both_classes else np.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_bins": float(ece), "selection_rate": float(prediction.mean()),
        "calibration_intercept": np.nan, "calibration_slope": np.nan,
    }
    if calibration and both_classes:
        # Optional evaluation diagnostic, not a fitted probability transform.
        # Perfect separation or constant probabilities cannot identify a finite
        # intercept/slope; return NaN rather than a misleading boundary estimate.
        from scipy.optimize import minimize
        from scipy.special import expit
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        separated = (z[y == 1].min() >= z[y == 0].max() or z[y == 1].max() <= z[y == 0].min())
        if np.std(z) > 1e-12 and not separated:
            design = np.column_stack([np.ones(len(z)), z])
            def objective(beta):
                eta = design @ beta
                return float(np.mean(np.logaddexp(0, eta) - y * eta))
            def gradient(beta):
                return design.T @ (expit(design @ beta) - y) / len(y)
            fit = minimize(objective, np.array([0., 1.]), jac=gradient, method="BFGS")
            if fit.success and np.isfinite(fit.x).all() and np.max(np.abs(fit.x)) < 30:
                information = (design.T * (expit(design @ fit.x) * (1 - expit(design @ fit.x)))) @ design
                if np.linalg.eigvalsh(information).min() > 1e-6:
                    result["calibration_intercept"], result["calibration_slope"] = map(float, fit.x)
    return result


def _audit_group_definitions(groups_df):
    frame = pd.DataFrame(groups_df).reset_index(drop=True)
    aliases = {
        "sex": ["sens_sex", "sex"],
        "race": ["sens_race_ethnicity", "race_group", "race_ethnicity", "race"],
        "age": ["sens_age_group", "age_group", "age"],
    }
    groups = {}
    for attribute, names in aliases.items():
        column = next((name for name in names if name in frame), None)
        if column is not None:
            groups[attribute] = frame[column].astype("string").fillna("Unknown")
    result = [(name, values, "audit_only" if name == "age" else "protected") for name, values in groups.items()]
    if "sex" in groups and "race" in groups:
        intersection = groups["sex"].str.cat(groups["race"], sep=" | ")
        result.append(("sex_x_race", intersection, "intersection"))
    return frame, result


def subgroup_metrics(y, p, groups_df, threshold=.5, min_group_n=100, min_class_n=20):
    """Detailed sex/race/age and sex-by-race audit with explicit denominators.

    Age is audit-only. TPR/FPR each require their actual-class denominator;
    balanced accuracy and AUROC require both. Small groups retain their counts
    but receive NaN rates. No performance or parity is inferred from one group.
    """
    y, p, prediction = _validated_predictions(y, p, threshold)
    frame, definitions = _audit_group_definitions(groups_df)
    if len(frame) != len(y):
        raise ValueError("Group metadata must align positionally with labels")
    if min_group_n < 1 or min_class_n < 1:
        raise ValueError("Group and class minimum denominators must be positive")
    rows = []
    for attribute, values, role in definitions:
        for group in sorted(values.unique()):
            mask = values.eq(group).to_numpy(dtype=bool)
            metrics = _base_classification_metrics(y[mask], p[mask], threshold)
            enough = metrics["n"] >= min_group_n
            tpr_eligible = enough and metrics["n_positive"] >= min_class_n
            fpr_eligible = enough and metrics["n_negative"] >= min_class_n
            joint_eligible = tpr_eligible and fpr_eligible
            if not enough:
                for key in ["brier", "log_loss", "ece_10_bins", "selection_rate"]:
                    metrics[key] = np.nan
            if not tpr_eligible:
                for key in ["tpr", "recall", "f1_positive", "precision"]:
                    metrics[key] = np.nan
            if not fpr_eligible:
                for key in ["fpr", "specificity", "f1_negative"]:
                    metrics[key] = np.nan
            if not joint_eligible:
                for key in ["balanced_accuracy", "auroc", "auprc", "macro_f1", "balanced_f1", "gmean"]:
                    metrics[key] = np.nan
            rows.append({"attribute": attribute, "group": str(group), "audit_role": role,
                "minimum_group_n": min_group_n, "minimum_class_n": min_class_n,
                "selection_eligible": enough, "tpr_eligible": tpr_eligible,
                "fpr_eligible": fpr_eligible, "joint_eligible": joint_eligible, **metrics})
    return pd.DataFrame(rows)


def classification_metrics(y, p, groups_df=None, threshold=.5, calibration=False):
    """Classification plus protected-group gaps; balanced_f1 equals macro_f1.

    AUPRC is average precision. Fairness summaries cover sex, race and their
    intersection; age remains in the detailed audit. Every group gap needs at
    least two eligible groups from the same attribute. Equalized-odds gaps use
    groups eligible for both TPR and FPR. Missing evidence stays NaN.
    """
    result = _base_classification_metrics(y, p, threshold, calibration)
    for key in ["worst_group_balanced_accuracy", "max_tpr_gap", "max_fpr_gap",
                "max_selection_gap", "max_equalized_odds_gap", "min_selection_ratio",
                "worst_audited_age_balanced_accuracy"]:
        result[key] = np.nan
    result["fairness_comparison_available"] = False
    result["fairness_complete"] = False
    if groups_df is None:
        return result
    table = subgroup_metrics(y, p, groups_df, threshold)
    if table.empty:
        return result
    protected = table.loc[table.audit_role.ne("audit_only")]
    age = table.loc[table.attribute.eq("age") & table.joint_eligible]
    if len(age):
        result["worst_audited_age_balanced_accuracy"] = float(age.balanced_accuracy.min())
    gaps = {name: [] for name in ["tpr", "fpr", "selection", "equalized_odds", "selection_ratio"]}
    attributes_complete = []
    worst = []
    for _, groups in protected.groupby("attribute", sort=True):
        joint = groups.loc[groups.joint_eligible]
        attributes_complete.append(len(joint) >= 2 and len(joint) == len(groups))
        if len(joint) >= 2:
            gaps["equalized_odds"].append(max(float(joint.tpr.max() - joint.tpr.min()), float(joint.fpr.max() - joint.fpr.min())))
            worst.extend(joint.balanced_accuracy.tolist())
        for name, column, eligibility in [("tpr", "tpr", "tpr_eligible"),
                ("fpr", "fpr", "fpr_eligible"), ("selection", "selection_rate", "selection_eligible")]:
            selected = groups.loc[groups[eligibility], column]
            if len(selected) >= 2:
                gaps[name].append(float(selected.max() - selected.min()))
                if name == "selection" and selected.max() > 0:
                    gaps["selection_ratio"].append(float(selected.min() / selected.max()))
    if worst:
        result["worst_group_balanced_accuracy"] = float(min(worst))
    for name, key in [("tpr", "max_tpr_gap"), ("fpr", "max_fpr_gap"),
                      ("selection", "max_selection_gap"), ("equalized_odds", "max_equalized_odds_gap")]:
        if gaps[name]:
            result[key] = float(max(gaps[name]))
    if gaps["selection_ratio"]:
        result["min_selection_ratio"] = float(min(gaps["selection_ratio"]))
    result["fairness_comparison_available"] = bool(gaps["equalized_odds"])
    result["fairness_complete"] = bool(attributes_complete and all(attributes_complete))
    return result


def reweighing_weights(y, group, smoothing=1., clip=(.25, 4.)):
    """Training-only P(y)P(g)/P(y,g), with optional joint-count smoothing.

    This function receives training rows only; it has no held-out fitting step.
    Bounded normalization preserves both the requested bounds and mean one.
    It addresses representation in a training objective, not causal fairness.
    """
    y = np.asarray(y)
    group = np.asarray(group)
    if y.ndim != 1 or group.ndim != 1 or len(y) != len(group) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Require aligned one-dimensional training labels and groups")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Smoothing must be nonnegative and finite")
    if clip is not None:
        lower, upper = map(float, clip)
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower <= 1 <= upper):
            raise ValueError("Weight bounds must be finite with 0 <= lower <= 1 <= upper")
    if len(np.unique(y)) < 2:
        return np.ones(len(y), dtype=float)
    groups = pd.Series(group).astype("string").fillna("Unknown")
    categories = pd.Index(sorted(groups.unique()))
    counts = pd.crosstab(groups, y).reindex(index=categories, columns=[0, 1], fill_value=0).to_numpy(dtype=float)
    joint = counts + smoothing
    joint /= joint.sum()
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    weight_grid = np.divide(expected, joint, out=np.zeros_like(joint), where=joint > 0)
    weights = weight_grid[categories.get_indexer(groups), y.astype(int)]
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid training reweighing weights")
    if clip is None:
        return weights / weights.mean()
    if lower == upper:
        return np.ones(len(weights), dtype=float)
    low_scale, high_scale = 0., 1.
    while np.clip(high_scale * weights, lower, upper).mean() < 1:
        high_scale *= 2
    for _ in range(100):
        scale = (low_scale + high_scale) / 2
        if np.clip(scale * weights, lower, upper).mean() < 1:
            low_scale = scale
        else:
            high_scale = scale
    return np.clip(((low_scale + high_scale) / 2) * weights, lower, upper)


class CorrelationProjection:
    """Training-fitted linear correlation removal, following Fairlearn's method.

    Mathematical reference (per-column sensitive means):
    https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.CorrelationRemover.html
    This is an inline implementation of that documented projection, not an
    instance of the Fairlearn estimator or a new research method.

    S contains explicit, frozen one-hot sensitive indicators from source train.
    Fit solves least squares against centered S; transform subtracts alpha times
    that projection using the training mean and coefficients. DataFrame S columns
    are frozen at fit; new columns are ignored and absent columns become zero.
    X remains an explicit predictor matrix; this class never selects or removes
    protected predictor columns automatically and never receives outcome labels.
    Linear decorrelation does not guarantee independence, equalized odds or
    preservation of clinical information. Alpha is selected on validation only.
    """
    def __init__(self, alpha=1., rcond=None):
        self.alpha = alpha
        self.rcond = rcond

    @staticmethod
    def _finite_matrix(values, name):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError(name + " must be a nonempty finite two-dimensional numeric matrix")
        return matrix

    def fit(self, X, S):
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("Alpha must be in [0, 1]")
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.sensitive_feature_names_in_ = list(S.columns) if isinstance(S, pd.DataFrame) else None
        if self.sensitive_feature_names_in_ is not None and len(set(self.sensitive_feature_names_in_)) != len(self.sensitive_feature_names_in_):
            raise ValueError("Sensitive indicator column names must be unique")
        x = self._finite_matrix(X, "X")
        sensitive = self._finite_matrix(S, "S")
        if len(x) != len(sensitive):
            raise ValueError("Sensitive indicators and X must have aligned rows")
        self.n_features_in_, self.n_sensitive_features_in_ = x.shape[1], sensitive.shape[1]
        self.n_samples_seen_ = len(x)
        self.sensitive_mean_ = sensitive.mean(axis=0)
        self.x_mean_ = x.mean(axis=0)
        self.coef_ = np.linalg.lstsq(sensitive - self.sensitive_mean_, x - self.x_mean_, rcond=self.rcond)[0]
        return self

    def transform(self, X, S):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit CorrelationProjection on source training rows first")
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            if set(X.columns) != set(self.feature_names_in_):
                raise ValueError("Predictor columns changed after fitting")
            X = X.loc[:, self.feature_names_in_]
        if isinstance(S, pd.DataFrame) and self.sensitive_feature_names_in_ is not None:
            if S.columns.duplicated().any():
                raise ValueError("Sensitive indicator column names must be unique")
            S = S.reindex(columns=self.sensitive_feature_names_in_, fill_value=0)
        x, sensitive = self._finite_matrix(X, "X"), self._finite_matrix(S, "S")
        if x.shape[1] != self.n_features_in_ or sensitive.shape != (len(x), self.n_sensitive_features_in_):
            raise ValueError("Feature/sensitive matrix dimensions changed after fitting")
        return x - self.alpha * (sensitive - self.sensitive_mean_) @ self.coef_

    def fit_transform(self, X, S):
        return self.fit(X, S).transform(X, S)
