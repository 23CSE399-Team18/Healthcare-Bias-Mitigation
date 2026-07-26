"""
================================================================================
 CKD Dataset — Multi-Bias Mitigation Pipeline
================================================================================
Addresses, in order:

  1. MISSING DATA BIAS   -> Random Forest iterative imputation + Gaussian
                            Mixture Model (EM-style) imputation, blended.
  2. OUTLIER BIAS        -> Isolation Forest / One-Class SVM / Elliptic
                            Envelope / LOF (selectable, contamination or
                            n_neighbors configurable). Detected outliers are
                            converted to NaN, then IMMEDIATELY re-imputed by
                            re-invoking the Missing-Data component (step 1).
  3. DATA NOISE BIAS     -> Autoencoder (MLP bottleneck reconstruction) used
                            to denoise the numeric feature space.
  4. FEATURE SCALE BIAS  -> StandardScaler applied to put every numeric
                            feature on a common scale before modeling.
  5. CLASS IMBALANCE BIAS-> Custom SMOTE (k-NN interpolation in feature
                            space) to balance ckd / notckd classes.

No internet access is available in this environment, so this uses only
numpy / pandas / scikit-learn (imbalanced-learn and tensorflow/torch are not
installed here) — SMOTE and the autoencoder are implemented from scratch so
the script is self-contained and portable.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# ============================================================================
# 0. LOAD + PARSE (ARFF-style CSV with '?' missing markers)
# ============================================================================
def load_arff_style_csv(path):
    """Parses the UCI CKD ARFF-formatted file into a clean DataFrame."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    attr_names, data_lines, in_data = [], [], False
    for line in lines:
        line = line.strip("\r\n")
        if line.strip().lower().startswith("@attribute"):
            name = line.split()[1].strip("'\"")
            attr_names.append(name)
        elif line.strip().lower().startswith("@data"):
            in_data = True
            continue
        elif in_data and line.strip():
            data_lines.append(line.strip())

    n_expected = len(attr_names)
    rows = []
    for r in data_lines:
        parts = r.split(",")
        if len(parts) > n_expected:
            # A handful of rows in this file have a stray extra comma
            # (empty token from a formatting slip, not a real missing
            # value marker like '?') -- drop empty tokens first so the
            # remaining fields still line up with the schema.
            non_empty = [p for p in parts if p.strip() != ""]
            if len(non_empty) == n_expected:
                parts = non_empty
            else:
                parts = parts[:n_expected]
        rows.append(parts)
    df = pd.DataFrame(rows, columns=attr_names)
    df = df.replace(["?", "", " ", "\t?"], np.nan)
    for col in df.columns:
        df[col] = df[col].str.strip() if df[col].dtype == object else df[col]
    return df


NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
                 "pot", "hemo", "pcv", "wbcc", "rbcc"]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet",
                     "pe", "ane"]
TARGET_COL = "class"


def coerce_types(df):
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df.loc[df[col] == "nan", col] = np.nan
    # fix known dirty category labels in this UCI file (e.g. "\tno", "ckd\t")
    df["dm"] = df["dm"].replace({"\tno": "no", "\tyes": "yes", " yes": "yes"})
    df["cad"] = df["cad"].replace({"\tno": "no"})
    df["class"] = df["class"].replace({"ckd\t": "ckd"})
    return df


def encode_categoricals(df, cat_cols, mappings=None):
    """Integer-encodes categoricals, preserving NaN. Returns df + mappings."""
    df = df.copy()
    mappings = mappings or {}
    for col in cat_cols:
        if col not in mappings:
            cats = sorted(df[col].dropna().unique())
            mappings[col] = {c: i for i, c in enumerate(cats)}
        df[col] = df[col].map(mappings[col])
    return df, mappings


def decode_categoricals(df, cat_cols, mappings):
    df = df.copy()
    for col in cat_cols:
        inv = {v: k for k, v in mappings[col].items()}
        df[col] = df[col].round().clip(0, len(inv) - 1).map(inv)
    return df


# ============================================================================
# 1. MISSING DATA BIAS — Random Forest (iterative) + GMM (EM-style) blend
# ============================================================================
def impute_missing_random_forest(X, n_estimators=100, max_iter=10):
    """Iterative imputation using Random Forest as the per-column estimator
    (MICE-style: each feature is predicted from all others, round-robin)."""
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators,
                                         random_state=RANDOM_STATE, n_jobs=-1),
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        sample_posterior=False,
    )
    return imputer.fit_transform(X), imputer


def impute_missing_gmm(X, n_components=3, n_iter=15):
    """EM-style imputation with a Gaussian Mixture Model:
       1. warm-start missing cells with column means
       2. fit a GMM on the current complete matrix
       3. for every row, replace its missing cells with the responsibility-
          weighted average of the component means for those cells
       4. repeat until convergence (fixed iterations here)
    """
    X = X.copy()
    mask = np.isnan(X)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[mask[:, j], j] = col_means[j]

    for _ in range(n_iter):
        gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE,
                               reg_covar=1e-3, max_iter=100)
        gmm.fit(X)
        resp = gmm.predict_proba(X)          # (n_samples, n_components)
        component_means = gmm.means_          # (n_components, n_features)
        weighted_means = resp @ component_means  # (n_samples, n_features)
        X[mask] = weighted_means[mask]
    return X, gmm


def missing_data_bias_component(X, feature_names, cat_cols):
    """Blends RF-iterative imputation with GMM-EM imputation.
       Numeric features -> average of both estimates.
       Categorical (label-encoded) features -> RF estimate, rounded to the
       nearest valid category code (GMM is less reliable for discrete codes).
    """
    cat_idx = [feature_names.index(c) for c in cat_cols]
    num_idx = [i for i in range(len(feature_names)) if i not in cat_idx]

    X_rf, _ = impute_missing_random_forest(X)
    X_gmm, _ = impute_missing_gmm(X)

    X_final = X_rf.copy()
    for i in num_idx:
        X_final[:, i] = 0.5 * X_rf[:, i] + 0.5 * X_gmm[:, i]
    for i in cat_idx:
        X_final[:, i] = np.round(X_rf[:, i])
    return X_final


# ============================================================================
# 2. OUTLIER BIAS — detect, convert to NaN, re-impute via component (1)
# ============================================================================
def detect_outliers(X_numeric_scaled, method="isolation_forest",
                     contamination=0.05, n_neighbors=20):
    """Returns a boolean mask, True = outlier row."""
    if method == "isolation_forest":
        model = IsolationForest(contamination=contamination,
                                 random_state=RANDOM_STATE, n_jobs=-1)
        pred = model.fit_predict(X_numeric_scaled)
    elif method == "one_class_svm":
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        pred = model.fit_predict(X_numeric_scaled)
    elif method == "elliptic_envelope":
        model = EllipticEnvelope(contamination=contamination,
                                  random_state=RANDOM_STATE)
        pred = model.fit_predict(X_numeric_scaled)
    elif method == "lof":
        model = LocalOutlierFactor(n_neighbors=n_neighbors,
                                    contamination=contamination)
        pred = model.fit_predict(X_numeric_scaled)
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    return pred == -1  # -1 => outlier, 1 => inlier


def iqr_bounds(X_col, k=1.5):
    """Standard Tukey IQR fence, computed from the WHOLE column (all rows) —
       i.e. against what's normal dataset-wide, not just within flagged rows."""
    q1, q3 = np.nanpercentile(X_col, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_bias_component(X, feature_names, cat_cols, num_idx,
                            method="isolation_forest", contamination=0.05,
                            n_neighbors=20, iqr_k=1.5):
    """Two-stage outlier handling so a single extreme column can no longer
       wipe out perfectly normal values sitting in the same row:

       1. Isolation Forest flags SUSPICIOUS ROWS (multivariate — catches
          rows where the combination of numeric values looks off).
       2. Within those flagged rows only, an IQR test (per numeric column,
          bounds computed from the FULL dataset of 400 rows, not just the
          flagged subset) checks each cell individually.
       3. Only the cells that actually fail the IQR test become NaN.
          Cells in a flagged row that are individually normal (e.g. a
          totally ordinary age sitting next to an extreme creatinine
          value) are left untouched.
       4. Only those specific NaNs are re-imputed via the missing-data
          component (step 1) — everything else keeps its real value.
    """
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[:, num_idx])

    row_mask = detect_outliers(X_num_scaled, method=method,
                                contamination=contamination,
                                n_neighbors=n_neighbors)

    # dataset-wide IQR bounds per numeric column (original units)
    bounds = {i: iqr_bounds(X[:, i], k=iqr_k) for i in num_idx}

    cell_mask = np.zeros_like(X, dtype=bool)
    flagged_rows = np.where(row_mask)[0]
    for i in num_idx:
        lo, hi = bounds[i]
        col_vals = X[flagged_rows, i]
        is_abnormal = (col_vals < lo) | (col_vals > hi)
        cell_mask[flagged_rows[is_abnormal], i] = True

    X_marked = X.copy()
    X_marked[cell_mask] = np.nan

    X_clean = missing_data_bias_component(X_marked, feature_names, cat_cols)
    return X_clean, row_mask, cell_mask


# ============================================================================
# 3. DATA NOISE BIAS — Autoencoder (MLP bottleneck), SELECTIVE correction only
# ============================================================================
def autoencoder_denoise(X_original, num_idx, bottleneck=8, hidden=16,
                         max_iter=2000, error_threshold_std=2.0, iqr_k=1.5):
    """Trains an MLP autoencoder on a TEMPORARY scaled copy of the numeric
       features to learn normal clinical patterns, then:
         1. computes the per-cell reconstruction error
         2. flags cells whose error exceeds
            mean_error + error_threshold_std * std_error (per feature)
         3. of those, replaces a cell ONLY if the ORIGINAL value is also
            statistically abnormal per a dataset-wide IQR test for that
            feature (high reconstruction error alone is no longer enough —
            a clinically valid value like age=53 stays untouched even if
            the network reconstructs it poorly)
         4. leaves every other value untouched — genuine clinical values
            are preserved, the temporary scaled copy is discarded.
       Returns the corrected matrix (original units) + a boolean mask of
       which cells were corrected, for transparency/auditing.
    """
    # 1. temporary scaled copy (discarded at the end of this function)
    temp_scaler = StandardScaler()
    X_num_scaled = temp_scaler.fit_transform(X_original[:, num_idx])

    # 2. train autoencoder to learn normal patterns
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden, bottleneck, hidden),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=20,
    )
    ae.fit(X_num_scaled, X_num_scaled)
    reconstruction_scaled = ae.predict(X_num_scaled)

    # 3. per-cell reconstruction error, thresholded per feature (column)
    cell_error = np.abs(X_num_scaled - reconstruction_scaled)
    col_mean_err = cell_error.mean(axis=0)
    col_std_err = cell_error.std(axis=0)
    threshold = col_mean_err + error_threshold_std * col_std_err
    error_mask = cell_error > threshold  # True = high reconstruction error

    # 3b. IQR gate (dataset-wide bounds, original units) — a high-error cell
    #     is only genuinely "noisy" if its ORIGINAL value is also abnormal.
    #     Clinically valid values (e.g. age=53, age=60) are never replaced,
    #     no matter how poorly the network reconstructs them.
    abnormal_mask = np.zeros_like(error_mask, dtype=bool)
    for col_pos, feat_idx in enumerate(num_idx):
        lo, hi = iqr_bounds(X_original[:, feat_idx], k=iqr_k)
        col_vals = X_original[:, feat_idx]
        abnormal_mask[:, col_pos] = (col_vals < lo) | (col_vals > hi)

    noisy_mask = error_mask & abnormal_mask  # BOTH conditions required

    # 4. replace ONLY flagged cells, in original units; everything else
    #    keeps its real, untouched original value
    reconstruction_original = temp_scaler.inverse_transform(reconstruction_scaled)
    X_corrected = X_original.copy()
    num_block = X_corrected[:, num_idx]
    num_block[noisy_mask] = reconstruction_original[noisy_mask]
    X_corrected[:, num_idx] = num_block

    n_flagged = noisy_mask.sum()
    return X_corrected, ae, noisy_mask, n_flagged


# ============================================================================
# 4. FEATURE SCALE BIAS — standardization
# ============================================================================
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ============================================================================
# 5. CLASS IMBALANCE BIAS — train/test split FIRST, then SMOTE (train only)
# ============================================================================
def train_test_split_manual(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified split so class ratio is preserved in both sets, done
       BEFORE any SMOTE oversampling so the test set never sees synthetic
       or leaked information."""
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def smote(X, y, minority_label, k_neighbors=5, target_count=None,
          random_state=RANDOM_STATE):
    """Classic SMOTE: for each minority sample, interpolate toward a random
       one of its k nearest minority neighbors to synthesize new samples."""
    rng = np.random.RandomState(random_state)
    X_min = X[y == minority_label]
    n_min = X_min.shape[0]
    n_maj = (y != minority_label).sum()
    target_count = target_count or n_maj
    n_to_generate = max(0, target_count - n_min)
    if n_to_generate == 0:
        return X, y

    k = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, neighbor_idx = nn.kneighbors(X_min)

    synthetic = np.zeros((n_to_generate, X.shape[1]))
    for s in range(n_to_generate):
        i = rng.randint(0, n_min)
        neighbor = neighbor_idx[i, rng.randint(1, k + 1)]  # skip self at 0
        gap = rng.uniform(0, 1)
        synthetic[s] = X_min[i] + gap * (X_min[neighbor] - X_min[i])

    X_bal = np.vstack([X, synthetic])
    y_bal = np.concatenate([y, np.full(n_to_generate, minority_label)])
    return X_bal, y_bal


def round_categoricals(X, cat_idx, valid_ranges):
    """After SMOTE interpolation, snap categorical columns back to valid
       integer category codes."""
    X = X.copy()
    for i, max_code in zip(cat_idx, valid_ranges):
        X[:, i] = np.clip(np.round(X[:, i]), 0, max_code)
    return X


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline(path, outlier_method="isolation_forest", contamination=0.05,
                  n_neighbors=20):
    print("=" * 70)
    print("STEP 0: Load + parse ARFF-style CSV")
    print("=" * 70)
    df_raw = load_arff_style_csv(path)
    df = coerce_types(df_raw)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    df_enc, cat_mappings = encode_categoricals(df, CATEGORICAL_COLS)
    y_series = df_enc[TARGET_COL]
    y_map = {c: i for i, c in enumerate(sorted(y_series.dropna().unique()))}
    y = y_series.map(y_map).values.astype(float)

    # drop rows with missing target (can't impute the label itself)
    keep = ~np.isnan(y)
    df_enc = df_enc.loc[keep].reset_index(drop=True)
    y = y[keep]

    X = df_enc[feature_cols].values.astype(float)
    num_idx = [feature_cols.index(c) for c in NUMERIC_COLS]
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_COLS]

    print("\n" + "=" * 70)
    print("STEP 1: MISSING DATA BIAS -> Random Forest + GMM imputation")
    print("=" * 70)
    X_imputed = missing_data_bias_component(X, feature_cols, CATEGORICAL_COLS)
    print("Remaining NaNs after imputation:", np.isnan(X_imputed).sum())

    print("\n" + "=" * 70)
    print(f"STEP 2: OUTLIER BIAS -> {outlier_method} flags rows -> IQR "
          "(whole-dataset bounds) checks cells within those rows -> "
          "only abnormal cells -> NaN -> re-impute (step 1)")
    print("=" * 70)
    X_clean, row_mask, cell_mask = outlier_bias_component(
        X_imputed, feature_cols, CATEGORICAL_COLS, num_idx,
        method=outlier_method, contamination=contamination,
        n_neighbors=n_neighbors)
    print(f"Rows flagged as suspicious by Isolation Forest: "
          f"{row_mask.sum()} / {len(row_mask)}")
    print(f"Of those, individual cells confirmed abnormal by IQR "
          f"& re-imputed: {cell_mask.sum()} cell(s) "
          f"(out of {row_mask.sum() * len(num_idx)} numeric cells checked)")
    outlier_mask = row_mask  # kept for downstream reporting/back-compat

    print("\n" + "=" * 70)
    print("STEP 3: DATA NOISE BIAS -> Autoencoder, SELECTIVE correction only")
    print("=" * 70)
    X_denoised, ae_model, noisy_mask, n_flagged = autoencoder_denoise(
        X_clean, num_idx, error_threshold_std=2.0)
    print(f"Noisy cells flagged & corrected: {n_flagged} "
          f"out of {X_clean[:, num_idx].size} numeric cells "
          f"({100 * n_flagged / X_clean[:, num_idx].size:.2f}%)")
    print("All other values left exactly as-is (genuine clinical values preserved).")

    print("\n" + "=" * 70)
    print("STEP 4: FEATURE SCALE BIAS -> StandardScaler (training copy only)")
    print("=" * 70)
    # The ORIGINAL (unscaled, human-readable) dataset is kept as-is for
    # interpretation. A SEPARATE scaled copy is produced only for model
    # training; the two are never mixed.
    X_original_final = X_denoised.copy()          # human-readable, for reports
    X_scaled, final_scaler = scale_features(X_denoised)  # for ML training only
    print("Feature means ~0, std ~1 after scaling:",
          np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6),
          np.allclose(X_scaled.std(axis=0), 1, atol=1e-6))
    print("Original-unit dataset kept unchanged and returned alongside it.")

    print("\n" + "=" * 70)
    print("STEP 5: CLASS IMBALANCE BIAS -> split first, then SMOTE (train only)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X_scaled, y, test_size=0.2)
    unique, counts = np.unique(y_train, return_counts=True)
    print("Training-set class counts before SMOTE:", dict(zip(unique, counts)))
    minority_label = unique[np.argmin(counts)]
    X_train_bal, y_train_bal = smote(X_train, y_train,
                                      minority_label=minority_label, k_neighbors=5)
    unique2, counts2 = np.unique(y_train_bal, return_counts=True)
    print("Training-set class counts after SMOTE:", dict(zip(unique2, counts2)))
    print(f"Test set untouched by SMOTE: {X_test.shape[0]} rows, "
          f"class counts {dict(zip(*np.unique(y_test, return_counts=True)))}")

    print("\nPipeline complete.")
    print(f"Final training matrix (balanced, scaled): {X_train_bal.shape}")
    print(f"Final test matrix (scaled, untouched by SMOTE): {X_test.shape}")

    return {
        "X_train": X_train_bal,
        "y_train": y_train_bal,
        "X_test": X_test,
        "y_test": y_test,
        "X_original_readable": X_original_final,  # full dataset, original units
        "y_original": y,
        "feature_names": feature_cols,
        "cat_mappings": cat_mappings,
        "y_map": y_map,
        "outlier_mask": outlier_mask,       # rows flagged by Isolation Forest
        "outlier_cell_mask": cell_mask,     # cells actually confirmed abnormal by IQR & re-imputed
        "noisy_mask": noisy_mask,
        "final_scaler": final_scaler,
        "autoencoder": ae_model,
    }


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "chronic_kidney_disease_full.csv"
    results = run_pipeline(input_path)
    inv_y_map = {v: k for k, v in results["y_map"].items()}
    feat_names = results["feature_names"]

    # Real clinical precision per numeric column (matches the original
    # source data) -- avoids ugly long float tails from RF/GMM/autoencoder
    # blending, e.g. 140.4730368157243 -> 140, without destroying columns
    # that genuinely need decimals (e.g. sg needs 3 d.p., hemo needs 1).
    ROUND_DECIMALS = {
        "age": 0, "bp": 0, "sg": 3, "al": 0, "su": 0, "bgr": 0, "bu": 1,
        "sc": 2, "sod": 1, "pot": 1, "hemo": 1, "pcv": 0, "wbcc": 0, "rbcc": 1,
    }

    # 1. Human-readable dataset: original units, cleaned (imputed, outlier-
    #    corrected, selectively denoised) but NOT scaled — for interpretation.
    readable_df = pd.DataFrame(results["X_original_readable"], columns=feat_names)
    readable_df["class"] = pd.Series(results["y_original"]).map(inv_y_map)
    # decode integer-coded categoricals back to their real text labels
    readable_df = decode_categoricals(readable_df, CATEGORICAL_COLS, results["cat_mappings"])
    # round numeric columns to real clinical precision (int columns as int)
    for col, dec in ROUND_DECIMALS.items():
        readable_df[col] = readable_df[col].round(dec)
        if dec == 0:
            readable_df[col] = readable_df[col].astype(int)
    readable_df.to_csv("ckd_cleaned_readable.csv", index=False)

    # 2. Training set: scaled + SMOTE-balanced -- for model training only.
    train_df = pd.DataFrame(results["X_train"], columns=feat_names)
    train_df["class"] = pd.Series(results["y_train"]).map(inv_y_map)
    train_df.to_csv("ckd_train_scaled_balanced.csv", index=False)

    # 3. Test set: scaled, untouched by SMOTE -- for honest evaluation.
    test_df = pd.DataFrame(results["X_test"], columns=feat_names)
    test_df["class"] = pd.Series(results["y_test"]).map(inv_y_map)
    test_df.to_csv("ckd_test_scaled.csv", index=False)

    print("\nSaved -> ckd_cleaned_readable.csv (original units, for interpretation)")
    print("Saved -> ckd_train_scaled_balanced.csv (scaled + SMOTE, for training)")
    print("Saved -> ckd_test_scaled.csv (scaled, untouched by SMOTE, for evaluation)")
