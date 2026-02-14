"""Run a fixed baseline in both transfer directions with source-only training.

Requires the two real, prepared datasets. Does not generate placeholder results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
for reader in [ROOT.parent / "_data_reader", ROOT.parent / "_model_reader"]:
    if reader.exists():
        sys.path.insert(0, str(reader))
import numpy as np
import pandas as pd
from prepare_cross_dataset import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, refresh_status


def patient_is_heldout(patient_id, seed=42, fraction=0.2):
    value = int.from_bytes(hashlib.sha256(f"{seed}|{patient_id}".encode()).digest()[:8], "big") / 2**64
    return value < fraction


def patient_split(frame, seed=42):
    by_patient = {patient: patient_is_heldout(patient, seed) for patient in frame.patient_id.unique()}
    heldout = frame.patient_id.map(by_patient).astype(bool)
    train, test = frame.loc[~heldout].copy(), frame.loc[heldout].copy()
    assert not set(train.patient_id).intersection(test.patient_id)
    if train.empty or test.empty:
        raise ValueError("Patient-held-out split is empty; provide a full prepared cohort")
    return train, test


def metric_row(y, probabilities, threshold=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                                 f1_score, brier_score_loss, confusion_matrix)
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    result = {"n": int(len(y)), "positive": int(y.sum()), "negative": int((y == 0).sum()),
              "auroc": float(roc_auc_score(y, p)) if both_classes else None,
              "auprc": float(average_precision_score(y, p)) if both_classes else None,
              "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both_classes else None,
              "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
              "brier": float(brier_score_loss(y, p)),
              "tpr": float(tp / (tp + fn)) if tp + fn else None,
              "fpr": float(fp / (fp + tn)) if fp + tn else None,
              "predicted_positive_rate": float(prediction.mean()),
              "false_negatives": int(fn), "false_positives": int(fp), "threshold": threshold}
    # Equal-width expected calibration error: descriptive, fixed bins, no fitting.
    bins = np.minimum((p * 10).astype(int), 9)
    result["ece_10_bins"] = float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
                                    for b in range(10) if (bins == b).any()))
    return result


def subgroup_rows(frame, probabilities, direction, evaluation, min_group=100, min_class=20):
    rows = []
    for attribute in ["age_group", "sex", "race_group"]:
        for group in sorted(frame[attribute].fillna("Unknown").unique()):
            mask = frame[attribute].fillna("Unknown").eq(group).to_numpy()
            group_y = frame.loc[mask, TARGET].to_numpy()
            metrics = metric_row(group_y, np.asarray(probabilities)[mask])
            sufficient = metrics["n"] >= min_group
            if not sufficient or metrics["positive"] < min_class:
                metrics["tpr"] = None
            if not sufficient or metrics["negative"] < min_class:
                metrics["fpr"] = None
            if not sufficient or min(metrics["positive"], metrics["negative"]) < min_class:
                metrics["auroc"] = metrics["auprc"] = None
            rows.append({"direction": direction, "evaluation": evaluation,
                         "attribute": attribute, "group": group,
                         "minimum_group_n_met": sufficient, **metrics})
    return rows


def create_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                                     ("categorical", categorical, CATEGORICAL_FEATURES)], remainder="drop")
    return Pipeline([("prepare", transformer),
                     ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])


def evaluate_direction(source, target, source_name, target_name, seed, output_dir, save_predictions=False):
    train, heldout = patient_split(source, seed)
    if train[TARGET].nunique() != 2:
        raise ValueError("Training data must have both outcome classes")
    direction = f"{source_name}_to_{target_name}"
    model = create_model()
    # Every learned operation is inside this pipeline and sees source training rows only.
    model.fit(train[FEATURES], train[TARGET])
    scores, groups = [], []
    for name, evaluation in [("source_patient_heldout", heldout), ("external_source", target)]:
        probabilities = model.predict_proba(evaluation[FEATURES])[:, 1]
        scores.append({"direction": direction, "evaluation": name, "train_stays": len(train),
                       "train_patients": int(train.patient_id.nunique()),
                       **metric_row(evaluation[TARGET], probabilities)})
        groups.extend(subgroup_rows(evaluation, probabilities, direction, name))
        if save_predictions:
            out = evaluation[["source", "patient_id", "stay_id", TARGET, "age_group", "sex", "race_group"]].copy()
            out["predicted_hospital_death_probability"] = probabilities
            out.to_csv(output_dir / f"{direction}_{name}_predictions.csv", index=False)
    split = {"direction": direction, "seed": seed,
             "source_train_stays": len(train), "source_train_patients": int(train.patient_id.nunique()),
             "source_test_stays": len(heldout), "source_test_patients": int(heldout.patient_id.nunique()),
             "source_patient_overlap": len(set(train.patient_id).intersection(heldout.patient_id)),
             "external_stays": len(target), "external_patients": int(target.patient_id.nunique()),
             "target_rows_used_for_fit_or_tuning": 0,
             "threshold_selected_from_target": False}
    return scores, groups, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "prepared")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    paths = {source: args.prepared_dir / f"{source}_24h_features.csv" for source in ["eicu", "mimic"]}
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise SystemExit("Cannot run cross-dataset evaluation until both REAL cohorts exist. Missing: " + "; ".join(missing))
    try:
        import sklearn
    except ImportError:
        raise SystemExit("scikit-learn is required for evaluation; install it in the project model environment.")
    frames = {source: pd.read_csv(path) for source, path in paths.items()}
    for source, frame in frames.items():
        if not frame.stay_id.is_unique or not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"Invalid IDs or target in {source}")
        if frame.source.nunique() != 1 or frame.source.iloc[0] != source:
            raise ValueError(f"Wrong source in {source} feature file")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, groups, splits = [], [], []
    for source, target in [("eicu", "mimic"), ("mimic", "eicu")]:
        values = evaluate_direction(frames[source], frames[target], source, target,
                                    args.seed, args.output_dir, args.save_predictions)
        metrics.extend(values[0]); groups.extend(values[1]); splits.append(values[2])
    pd.DataFrame(metrics).to_csv(args.output_dir / "generalization_metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(args.output_dir / "demographic_metrics.csv", index=False)
    details = {"method": "Fixed logistic regression baseline; C1.0, max_iter2000, no class reweighting",
               "preprocessing": "Source-training-only median imputation, missing indicators, scaling and one-hot encoding",
               "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
               "target": TARGET, "threshold": 0.5, "splits": splits,
               "minimum_demographic_group_n": 100, "minimum_class_denominator": 20,
               "uncertainty": "Point estimates; no confidence intervals or statistical-significance claims",
               "versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
               "input_sha256": {name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for name, path in paths.items()},
               "not_clinically_validated": True}
    (args.output_dir / "evaluation_protocol.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    refresh_status(args.prepared_dir, results_dir=args.output_dir)
    print(json.dumps({"completed": True, "output_dir": str(args.output_dir), "directions": 2}), flush=True)


if __name__ == "__main__":
    main()
