r"""
=============================================================================
 BASELINE DEEP LEARNING — MLP (Multi-Layer Perceptron)
 Datasets: UCI Heart Disease | Breast Cancer Coimbra | NHANES 2021-2023
=============================================================================
 WHAT IT DOES:
   Trains a PyTorch MLP classifier on each dataset using preprocessed data.
   Reports Accuracy, F1-Score, AUROC overall AND per-demographic-group.
   Saves results to baseline_dl_results.txt

 ARCHITECTURE:
   Input → 128 (ReLU, BN, Dropout) → 64 (ReLU, BN, Dropout) → 32 (ReLU) → 1 (Sigmoid)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(BASE, "preprocessed_output")
LOG_FILE = os.path.join(BASE, "baseline_dl_results.txt")

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ═══════════════════════════════════════════════════════════════════
#  MLP MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    Architecture: Input → 128 → 64 → 32 → 1
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name,
                       epochs=50, lr=0.001, batch_size=64):
    """Train MLP and return predictions + metrics."""
    log(f"\n  Training MLP: {X_train.shape[1]} features, "
        f"{len(X_train)} train / {len(X_test)} test samples")
    log(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            log(f"    Epoch {epoch+1:3d}/{epochs} — Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_prob = model(X_te).numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auroc = float("nan")

    log(f"\n  ── {dataset_name} — OVERALL RESULTS ──")
    log(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    log(f"  F1-Score : {f1:.4f}")
    log(f"  AUROC    : {auroc:.4f}")
    log(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    log(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    log(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    log(f"\n  Classification Report:")
    log(classification_report(y_test, y_pred, target_names=["Negative", "Positive"],
                              zero_division=0))

    return y_pred, y_prob, model


def demographic_audit(df_test, y_test, y_pred, y_prob, group_col, group_labels, dataset_name):
    """Compute per-group metrics for fairness analysis."""
    log(f"\n  ── {dataset_name} — FAIRNESS AUDIT by {group_col} ──")
    log(f"  {'Group':<25} {'N':>5} {'Acc':>7} {'F1':>7} {'AUROC':>7} {'TPR':>7} {'FPR':>7}")
    log(f"  {'-'*70}")

    for val, label in group_labels.items():
        mask = df_test[group_col] == val
        n = mask.sum()
        if n < 5:
            log(f"  {label:<25} {n:>5}  (too few samples)")
            continue

        y_t = y_test[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        acc = accuracy_score(y_t, y_p)
        f1 = f1_score(y_t, y_p, zero_division=0)
        try:
            auroc = roc_auc_score(y_t, y_pr)
        except ValueError:
            auroc = float("nan")

        # TPR (sensitivity) and FPR
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tp = cm[1][1] if cm.shape[0] > 1 else 0
        fn = cm[1][0] if cm.shape[0] > 1 else 0
        fp = cm[0][1]
        tn = cm[0][0]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        log(f"  {label:<25} {n:>5} {acc:>7.3f} {f1:>7.3f} {auroc:>7.3f} {tpr:>7.3f} {fpr:>7.3f}")


# ═══════════════════════════════════════════════════════════════════
#  DATASET 1: UCI HEART DISEASE
# ═══════════════════════════════════════════════════════════════════

def run_heart_disease():
    log("\n" + "=" * 70)
    log(" DATASET 1: UCI HEART DISEASE ")
    log(" Deep Learning Baseline — MLP (PyTorch)")
    log("=" * 70)

    # Load preprocessed data
    path = os.path.join(PRE, "heart_disease_preprocessed.csv")
    df = pd.read_csv(path)

    # Remove counterfactual rows
    if "Is_Counterfactual" in df.columns:
        df = df[df["Is_Counterfactual"] == 0].copy()
        df = df.drop(columns=["Is_Counterfactual"])

    log(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Target and sensitive
    target_col = "Heart_Disease"
    sensitive_col = "sex"

    if target_col not in df.columns:
        # Try alternative name
        for c in df.columns:
            if "heart" in c.lower() or "target" in c.lower() or "num" in c.lower():
                target_col = c
                break

    log(f"  Target: {target_col}")
    log(f"  Positive: {df[target_col].sum()}/{len(df)} ({df[target_col].mean()*100:.1f}%)")

    # Features: drop target, sensitive, weights
    drop_cols = [target_col, "Sample_Weight"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Ensure all numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=42, stratify=y)

    # Train
    y_pred, y_prob, model = train_and_evaluate(
        X_train, X_test, y_train, y_test, "Heart Disease", epochs=60)

    # Fairness audit by sex
    df_test = df.loc[idx_test].reset_index(drop=True)
    if sensitive_col in df_test.columns:
        demographic_audit(df_test, y_test, y_pred, y_prob,
                          sensitive_col,
                          {1: "Male", 0: "Female"},
                          "Heart Disease")

    return model


# ═══════════════════════════════════════════════════════════════════
#  DATASET 2: BREAST CANCER COIMBRA
# ═══════════════════════════════════════════════════════════════════

def run_breast_cancer():
    log("\n" + "=" * 70)
    log(" DATASET 2: BREAST CANCER COIMBRA")
    log(" Deep Learning Baseline — MLP (PyTorch)")
    log("=" * 70)

    # Load preprocessed data
    path = os.path.join(PRE, "breast_cancer_preprocessed.csv")
    df = pd.read_csv(path)

    # Remove counterfactual rows
    if "Is_Counterfactual" in df.columns:
        df = df[df["Is_Counterfactual"] == 0].copy()
        df = df.drop(columns=["Is_Counterfactual"])

    log(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    target_col = "Cancer_Risk"
    sensitive_col = "Age_Group"

    log(f"  Target: {target_col}")
    log(f"  Positive: {df[target_col].sum()}/{len(df)} ({df[target_col].mean()*100:.1f}%)")

    # Features
    drop_cols = [target_col, "Sample_Weight"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=42, stratify=y)

    # Train (more epochs for small dataset)
    y_pred, y_prob, model = train_and_evaluate(
        X_train, X_test, y_train, y_test, "Breast Cancer", epochs=80, lr=0.0005)

    # Fairness audit by Age_Group
    df_test = df.loc[idx_test].reset_index(drop=True)
    if sensitive_col in df_test.columns:
        demographic_audit(df_test, y_test, y_pred, y_prob,
                          sensitive_col,
                          {0: "Younger (<median)", 1: "Older (>=median)"},
                          "Breast Cancer")

    return model


# ═══════════════════════════════════════════════════════════════════
#  DATASET 3: NHANES 2021-2023
# ═══════════════════════════════════════════════════════════════════

def run_nhanes():
    log("\n" + "=" * 70)
    log(" DATASET 3: NHANES 2021-2023")
    log(" Deep Learning Baseline — MLP (PyTorch)")
    log("=" * 70)

    # Load preprocessed data
    path = os.path.join(PRE, "NHANES_preprocessed.csv")
    df = pd.read_csv(path)

    # Remove counterfactual rows
    if "Is_Counterfactual" in df.columns:
        df = df[df["Is_Counterfactual"] == 0].copy()
        df = df.drop(columns=["Is_Counterfactual"])

    log(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Use Diabetes_Risk as primary target
    target_col = "Diabetes_Risk"
    sensitive_col = "Gender"

    if target_col not in df.columns:
        log(f"  [!] {target_col} not found. Available: {list(df.columns)[:20]}")
        return None

    log(f"  Target: {target_col}")
    log(f"  Positive: {df[target_col].sum()}/{len(df)} ({df[target_col].mean()*100:.1f}%)")

    # Features: drop targets, sensitive, IDs, weights, labels
    drop_cols = ["Diabetes_Risk", "Hypertension_Risk", "CKD_Risk", "Obesity",
                 "Sample_Weight", "Patient_ID",
                 "Gender_Label", "Ethnicity_Label",
                 "Marital_Status_Label", "Education_Label"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric features
    numeric_features = []
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() > len(df) * 0.1:  # at least 10% non-null
            numeric_features.append(c)

    feature_cols = numeric_features
    log(f"  Using {len(feature_cols)} numeric features")

    df = df.dropna(subset=[target_col])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # Sample if too large (for speed)
    if len(df) > 15000:
        log(f"  Sampling 15,000 from {len(df)} for training speed...")
        df = df.sample(n=15000, random_state=42).reset_index(drop=True)

    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=42, stratify=y)

    # Train
    y_pred, y_prob, model = train_and_evaluate(
        X_train, X_test, y_train, y_test, "NHANES (Diabetes)", epochs=50)

    # Fairness audit by Gender
    df_test = df.loc[idx_test].reset_index(drop=True)
    if sensitive_col in df_test.columns:
        demographic_audit(df_test, y_test, y_pred, y_prob,
                          sensitive_col,
                          {1.0: "Male", 2.0: "Female"},
                          "NHANES (Diabetes)")

    # Also audit by Ethnicity if available
    eth_col = "Ethnicity_With_Asian"
    if eth_col in df_test.columns:
        demographic_audit(df_test, y_test, y_pred, y_prob,
                          eth_col,
                          {3.0: "Non-Hispanic White",
                           4.0: "Non-Hispanic Black",
                           1.0: "Mexican American",
                           6.0: "Non-Hispanic Asian"},
                          "NHANES (Diabetes)")

    return model


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log("=" * 70)
    log(" BASELINE DEEP LEARNING — MLP (Multi-Layer Perceptron)")
    log(" Framework: PyTorch | Architecture: 128→64→32→1")
    log(" Datasets: Heart Disease | Breast Cancer | NHANES")
    log("=" * 70)

    results = {}

    # Dataset 1
    m1 = run_heart_disease()
    results["heart"] = m1

    # Dataset 2
    m2 = run_breast_cancer()
    results["cancer"] = m2

    # Dataset 3
    m3 = run_nhanes()
    results["nhanes"] = m3

    # Summary
    log("\n" + "=" * 70)
    log(" BASELINE DL COMPLETE — ALL 3 DATASETS")
    log("=" * 70)
    log("  Model: MLP (128→64→32→1) with BatchNorm + Dropout")
    log("  Purpose: Establish baseline DL performance before applying")
    log("           fairness-constrained training (Obj 2: AD, CFReg)")
    log("  Next:    Compare these baselines with XGBoost and")
    log("           fairness-mitigated DL models")
    log("=" * 70)

    # Save results
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\n  Results saved: {LOG_FILE}")
