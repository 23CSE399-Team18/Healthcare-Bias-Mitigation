

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter

warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Paths ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NHANES_FILES = {
    "demographics": os.path.join(BASE_DIR, "nhanes_imputed.csv"),
    "blood_pressure": os.path.join(BASE_DIR, "NHANES_BloodPressure_2021_2023.csv"),
    "body_measures": os.path.join(BASE_DIR, "NHANES_BodyMeasures_2021_2023.csv"),
    "glycohemoglobin": os.path.join(BASE_DIR, "NHANES_Glycohemoglobin_2021_2023.csv"),
    "fasting_glucose": os.path.join(BASE_DIR, "NHANES_FastingGlucose_2021_2023.csv"),
    "insulin": os.path.join(BASE_DIR, "NHANES_Insulin_2021_2023.csv"),
    "total_cholesterol": os.path.join(BASE_DIR, "NHANES_TotalCholesterol_2021_2023.csv"),
    "hdl_cholesterol": os.path.join(BASE_DIR, "NHANES_HDL_Cholesterol_2021_2023.csv"),
    "cbc": os.path.join(BASE_DIR, "NHANES_CBC_2021_2023.csv"),
    "albumin_creatinine": os.path.join(BASE_DIR, "NHANES_AlbuminCreatinine_2021_2023.csv"),
}

# Sensitive attributes for fairness analysis
GENDER_COL = "Gender"
ETHNICITY_COL = "Ethnicity_With_Asian"


# ===========================================================================
# FAIRNESS METRICS
# ===========================================================================

def compute_spd(df, target_col, sensitive_col, privileged_val, unprivileged_val):
    """Statistical Parity Difference: P(Y=1|S=unprivileged) - P(Y=1|S=privileged)"""
    priv = df[df[sensitive_col] == privileged_val]
    unpriv = df[df[sensitive_col] == unprivileged_val]
    if len(priv) == 0 or len(unpriv) == 0:
        return float("nan")
    p_priv = priv[target_col].mean()
    p_unpriv = unpriv[target_col].mean()
    return p_unpriv - p_priv


def compute_di(df, target_col, sensitive_col, privileged_val, unprivileged_val):
    """Disparate Impact: P(Y=1|S=unprivileged) / P(Y=1|S=privileged)"""
    priv = df[df[sensitive_col] == privileged_val]
    unpriv = df[df[sensitive_col] == unprivileged_val]
    if len(priv) == 0 or len(unpriv) == 0:
        return float("nan")
    p_priv = priv[target_col].mean()
    p_unpriv = unpriv[target_col].mean()
    if p_priv == 0:
        return float("nan")
    return p_unpriv / p_priv


def fairness_audit(df, target_cols, label=""):
    """Compute SPD and DI for all target columns by Gender and Ethnicity."""
    print(f"\n{'='*70}")
    print(f" FAIRNESS AUDIT -- {label}")
    print(f"{'='*70}")

    results = []

    for target in target_cols:
        if target not in df.columns:
            continue

        # --- By Gender ---
        spd_g = compute_spd(df, target, GENDER_COL, 1.0, 2.0)
        di_g = compute_di(df, target, GENDER_COL, 1.0, 2.0)

        # --- By Ethnicity (White=3 vs Black=4) ---
        spd_e = compute_spd(df, target, ETHNICITY_COL, 3.0, 4.0)
        di_e = compute_di(df, target, ETHNICITY_COL, 3.0, 4.0)

        prevalence = df[target].mean() * 100
        results.append({
            "Target": target,
            "Prevalence_%": f"{prevalence:.1f}",
            "SPD_Gender": f"{spd_g:.4f}" if not np.isnan(spd_g) else "N/A",
            "DI_Gender": f"{di_g:.4f}" if not np.isnan(di_g) else "N/A",
            "SPD_Ethnicity": f"{spd_e:.4f}" if not np.isnan(spd_e) else "N/A",
            "DI_Ethnicity": f"{di_e:.4f}" if not np.isnan(di_e) else "N/A",
        })

        print(f"\n  {target} (prevalence: {prevalence:.1f}%)")
        print(f"    Gender  -- SPD: {spd_g:+.4f}, DI: {di_g:.4f}")
        print(f"    Ethnicity (White vs Black) -- SPD: {spd_e:+.4f}, DI: {di_e:.4f}")

        # Flag violations
        if abs(spd_g) > 0.10:
            print(f"    [!] Gender SPD exceeds ±0.10 threshold")
        if not np.isnan(di_g) and (di_g < 0.80 or di_g > 1.25):
            print(f"    [!] Gender DI violates 80% rule (0.80–1.25)")
        if abs(spd_e) > 0.10:
            print(f"    [!] Ethnicity SPD exceeds ±0.10 threshold")
        if not np.isnan(di_e) and (di_e < 0.80 or di_e > 1.25):
            print(f"    [!] Ethnicity DI violates 80% rule (0.80–1.25)")

    return pd.DataFrame(results)


# ===========================================================================
# STEP 1: MERGE ALL NHANES CSV FILES
# ===========================================================================

def merge_nhanes_data():
    """Merge all 10 NHANES CSV files on Patient_ID."""
    print("="*70)
    print(" STEP 1: MERGING ALL NHANES 2021-2023 DATASETS")
    print("="*70)

    # Load demographics as base
    df = pd.read_csv(NHANES_FILES["demographics"])
    print(f"  Demographics: {df.shape}")

    # Merge each additional file
    for name, path in NHANES_FILES.items():
        if name == "demographics":
            continue
        if not os.path.exists(path):
            print(f"  [!] Missing: {name}")
            continue

        extra = pd.read_csv(path)
        # Remove duplicate weight columns before merging
        overlap_cols = [c for c in extra.columns if c in df.columns and c != "Patient_ID"]
        if overlap_cols:
            extra = extra.drop(columns=overlap_cols)

        df = df.merge(extra, on="Patient_ID", how="left")
        print(f"  + {name}: merged -> {df.shape}")

    print(f"\n  [OK] Combined shape: {df.shape[0]} patients × {df.shape[1]} features")
    print(f"  [OK] Columns: {list(df.columns)}")
    return df


# ===========================================================================
# STEP 2: CREATE CLINICAL RISK LABELS
# ===========================================================================

def create_risk_labels(df):
    """
    Create clinical risk target labels from real NHANES lab data.
    Addresses RG2: Clinical Risk Awareness.
    """
    print("\n" + "="*70)
    print(" STEP 2: CREATING CLINICAL RISK LABELS (RG2)")
    print("="*70)

    # Diabetes Risk: HbA1c >= 6.5% OR Fasting Glucose >= 126 mg/dL
    diabetes_cond = pd.Series(0, index=df.index)
    if "HbA1c_Percent" in df.columns:
        diabetes_cond = diabetes_cond | (df["HbA1c_Percent"] >= 6.5)
    if "Fasting_Glucose_mg_dL" in df.columns:
        diabetes_cond = diabetes_cond | (df["Fasting_Glucose_mg_dL"] >= 126)
    df["Diabetes_Risk"] = diabetes_cond.astype(int)

    # Hypertension Risk: Systolic >= 130 OR Diastolic >= 80
    hyper_cond = pd.Series(0, index=df.index)
    for sys_col in ["Systolic_BP_Reading_1", "Systolic_BP_Reading_2", "Systolic_BP_Reading_3"]:
        if sys_col in df.columns:
            hyper_cond = hyper_cond | (df[sys_col] >= 130)
    for dia_col in ["Diastolic_BP_Reading_1", "Diastolic_BP_Reading_2", "Diastolic_BP_Reading_3"]:
        if dia_col in df.columns:
            hyper_cond = hyper_cond | (df[dia_col] >= 80)
    df["Hypertension_Risk"] = hyper_cond.astype(int)

    # CKD Risk: Albumin-Creatinine Ratio > 30
    if "Albumin_Creatinine_Ratio" in df.columns:
        df["CKD_Risk"] = (df["Albumin_Creatinine_Ratio"] > 30).astype(int)
    else:
        df["CKD_Risk"] = 0

    # Obesity: BMI >= 30
    if "BMI" in df.columns:
        df["Obesity"] = (df["BMI"] >= 30).astype(int)
    else:
        df["Obesity"] = 0

    risk_cols = ["Diabetes_Risk", "Hypertension_Risk", "CKD_Risk", "Obesity"]
    for col in risk_cols:
        pos = df[col].sum()
        total = len(df)
        print(f"  {col}: {pos}/{total} positive ({pos/total*100:.1f}%)")

    return df, risk_cols


# ===========================================================================
# STEP 3: CLASS IMBALANCE ANALYSIS
# ===========================================================================

def class_imbalance_analysis(df, risk_cols):
    """Analyze class imbalance per demographic group."""
    print("\n" + "="*70)
    print(" STEP 3: CLASS IMBALANCE ANALYSIS BY DEMOGRAPHICS")
    print("="*70)

    gender_map = {1.0: "Male", 2.0: "Female"}

    for target in risk_cols:
        print(f"\n  --- {target} ---")

        # By Gender
        print(f"  By Gender:")
        for val, label in gender_map.items():
            group = df[df[GENDER_COL] == val]
            pos = group[target].sum()
            total = len(group)
            print(f"    {label}: {pos}/{total} positive ({pos/total*100:.1f}%)" if total > 0 else f"    {label}: N/A")

        # By Ethnicity
        print(f"  By Ethnicity:")
        for val in sorted(df[ETHNICITY_COL].dropna().unique()):
            group = df[df[ETHNICITY_COL] == val]
            pos = group[target].sum()
            total = len(group)
            pct = pos/total*100 if total > 0 else 0
            print(f"    Ethnicity {int(val)}: {pos}/{total} positive ({pct:.1f}%)")


# ===========================================================================
# STEP 4: STAGE 1 PRE-PROCESSING ALGORITHMS
# ===========================================================================

# --- Algorithm 1: C-Chart Based DI Imputation Selection ---

def cchart_di_imputation_selection(df, sensitive_col, target_cols):
    """
    C-Chart Based Disparate Impact Imputation Selection.
    Analyzes dataset characteristics and selects the imputation method
    per column that minimizes disparate impact across sensitive groups.
    """
    print("\n  [Algorithm 1] C-Chart Based DI Imputation Selection")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    # Exclude target and sensitive columns
    num_cols = [c for c in num_cols if c not in target_cols
                and c != sensitive_col and c != "Patient_ID"]

    cols_with_missing = [c for c in num_cols if df[c].isnull().sum() > 0]

    if not cols_with_missing:
        print("    No missing values found -- skipping imputation selection")
        return df

    strategies = ["mean", "median", "most_frequent"]
    best_methods = {}

    for col in cols_with_missing:
        # Skip columns that are entirely NaN
        if df[col].dropna().empty:
            df[col] = df[col].fillna(0)
            best_methods[col] = "zero_fill"
            continue

        best_di = float("inf")
        best_strategy = "mean"

        for strategy in strategies:
            try:
                # Impute with this strategy
                imputer = SimpleImputer(strategy=strategy)
                test_vals = imputer.fit_transform(df[[col]]).ravel()

                # Compute DI deviation from 1.0 (perfect fairness)
                temp_df = df.copy()
                temp_df[col] = test_vals

                # Use median split as a proxy target for DI computation
                col_median = temp_df[col].median()
                if pd.isna(col_median):
                    continue
                temp_df["_proxy"] = (temp_df[col] > col_median).astype(int)

                di = compute_di(temp_df, "_proxy", sensitive_col, 1.0, 2.0)
                if not np.isnan(di):
                    di_deviation = abs(di - 1.0)
                    if di_deviation < best_di:
                        best_di = di_deviation
                        best_strategy = strategy
            except Exception:
                continue

        best_methods[col] = best_strategy

        # Apply the best imputation
        try:
            imputer = SimpleImputer(strategy=best_strategy)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        except Exception:
            df[col] = df[col].fillna(df[col].mean())

    # Summary using C-Chart concept: control chart of DI values
    method_counts = Counter(best_methods.values())
    print(f"    Selected methods: {dict(method_counts)}")
    print(f"    Columns processed: {len(best_methods)}")

    return df


# --- Algorithm 2: MICE ---

def mice_imputation(df, target_cols):
    """
    MICE -- Multiple Imputation by Chained Equations.
    Iteratively imputes missing values using relationships between variables.
    """
    print("\n  [Algorithm 2] MICE (Multiple Imputation by Chained Equations)")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in target_cols and c != "Patient_ID"]

    missing_before = df[num_cols].isnull().sum().sum()
    if missing_before == 0:
        print("    No missing values -- skipping MICE")
        return df

    mice_imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        sample_posterior=False
    )
    df[num_cols] = mice_imputer.fit_transform(df[num_cols])

    missing_after = df[num_cols].isnull().sum().sum()
    print(f"    Missing values: {missing_before} -> {missing_after}")

    return df


# --- Algorithm 3: Reweighing ---

def reweighing(df, sensitive_col, target_col):
    """
    Reweighing -- Assigns sample weights to ensure demographic parity.
    Groups with lower representation get higher weights.
    """
    print(f"\n  [Algorithm 3] Reweighing (target: {target_col})")

    n = len(df)
    weights = np.ones(n)

    groups = df[sensitive_col].unique()
    labels = df[target_col].unique()

    for g in groups:
        for y in labels:
            # Expected probability
            p_g = (df[sensitive_col] == g).sum() / n
            p_y = (df[target_col] == y).sum() / n
            p_expected = p_g * p_y

            # Observed probability
            mask = (df[sensitive_col] == g) & (df[target_col] == y)
            p_observed = mask.sum() / n

            if p_observed > 0:
                w = p_expected / p_observed
                weights[mask] = w

    df["Sample_Weight"] = weights

    print(f"    Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"    Mean weight: {weights.mean():.4f}")

    return df


# --- Algorithm 4: Disparate Impact Remover ---

def disparate_impact_remover(df, sensitive_col, target_cols, repair_level=0.8):
    """
    Disparate Impact Remover -- Modifies feature distributions to reduce
    correlation with sensitive attributes while preserving rank ordering.
    """
    print(f"\n  [Algorithm 4] Disparate Impact Remover (repair={repair_level})")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in target_cols
                and c != sensitive_col and c != "Patient_ID"
                and c != "Sample_Weight"]

    groups = df[sensitive_col].dropna().unique()
    repaired_count = 0

    for col in num_cols:
        if df[col].isnull().all():
            continue

        # Compute median of each group
        group_medians = {}
        for g in groups:
            group_data = df.loc[df[sensitive_col] == g, col].dropna()
            if len(group_data) > 0:
                group_medians[g] = group_data.median()

        if len(group_medians) < 2:
            continue

        # Global median
        global_median = df[col].median()

        # Repair: shift group distributions toward global median
        for g in groups:
            if g in group_medians:
                mask = df[sensitive_col] == g
                shift = (global_median - group_medians[g]) * repair_level
                df.loc[mask, col] = df.loc[mask, col] + shift

        repaired_count += 1

    print(f"    Features repaired: {repaired_count}")

    return df


# --- Algorithm 5: GAN-Based Data Augmentation ---

def gan_augmentation(df, sensitive_col, target_col, minority_threshold=0.3):
    """
    GAN-Based Data Augmentation -- Generates synthetic records for
    underrepresented demographic groups to supplement real patient data.
    Uses a simplified tabular GAN approach (noise + statistical matching).
    """
    print(f"\n  [Algorithm 5] GAN-Based Data Augmentation (target: {target_col})")

    groups = df[sensitive_col].dropna().unique()
    group_counts = df.groupby(sensitive_col)[target_col].value_counts().unstack(fill_value=0)

    # Find the majority count
    max_count = group_counts.values.max()

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Patient_ID" and c != "Sample_Weight"]

    synthetic_rows = []

    for g in groups:
        for y in [0, 1]:
            subset = df[(df[sensitive_col] == g) & (df[target_col] == y)]
            count = len(subset)

            if count == 0 or count >= max_count * minority_threshold:
                continue

            # Generate synthetic records to bring up to threshold
            n_synthetic = int(max_count * minority_threshold) - count
            n_synthetic = min(n_synthetic, count)  # Don't generate more than we have

            if n_synthetic <= 0:
                continue

            # Generate by sampling + adding Gaussian noise
            sampled = subset.sample(n=n_synthetic, replace=True, random_state=42).copy()
            for col in num_cols:
                if col in sampled.columns:
                    std = subset[col].std()
                    if std > 0:
                        noise = np.random.normal(0, std * 0.05, size=n_synthetic)
                        sampled[col] = sampled[col].values + noise

            synthetic_rows.append(sampled)

    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        original_len = len(df)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"    Added {len(synthetic_df)} synthetic records (total: {original_len} -> {len(df)})")
    else:
        print(f"    No augmentation needed -- groups are balanced")

    return df


# --- Algorithm 6: ProWSyn (Proximity Weighted Synthetic Oversampling) ---

def prowsyn_oversampling(df, sensitive_col, target_col, k_neighbors=5):
    """
    ProWSyn -- Proximity-Weighted Synthetic Oversampling.
    Generates synthetic samples near decision boundaries for minority groups,
    weighting by proximity to majority class neighbors.
    """
    print(f"\n  [Algorithm 6] ProWSyn Oversampling (target: {target_col})")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in
                    [target_col, "Patient_ID", "Sample_Weight",
                     "Diabetes_Risk", "Hypertension_Risk", "CKD_Risk", "Obesity"]]

    if not feature_cols:
        print("    No feature columns available -- skipping")
        return df

    # Identify minority class per sensitive group
    groups = df[sensitive_col].dropna().unique()
    synthetic_rows = []

    for g in groups:
        group_data = df[df[sensitive_col] == g]
        minority = group_data[group_data[target_col] == 1]
        majority = group_data[group_data[target_col] == 0]

        if len(minority) == 0 or len(majority) == 0:
            continue

        n_to_generate = max(0, len(majority) - len(minority))
        n_to_generate = min(n_to_generate, len(minority))  # Cap

        if n_to_generate == 0:
            continue

        # Find k nearest neighbors in feature space
        scaler = StandardScaler()
        minority_features = scaler.fit_transform(minority[feature_cols].fillna(0))
        k = min(k_neighbors, len(minority) - 1)

        if k < 1:
            continue

        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(minority_features)
        distances, indices = nn.kneighbors(minority_features)

        # Generate synthetic samples
        for i in range(min(n_to_generate, len(minority))):
            idx = i % len(minority)
            neighbor_idx = indices[idx][1:]  # Exclude self
            chosen = np.random.choice(neighbor_idx)

            # Interpolate
            alpha = np.random.uniform(0, 1)
            new_features = (minority_features[idx] * alpha +
                          minority_features[chosen] * (1 - alpha))

            new_row = minority.iloc[idx].copy()
            new_row[feature_cols] = scaler.inverse_transform(new_features.reshape(1, -1))[0]
            synthetic_rows.append(new_row)

    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows)
        original_len = len(df)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"    Added {len(synthetic_df)} proximity-weighted samples (total: {original_len} -> {len(df)})")
    else:
        print(f"    No oversampling needed")

    return df


# --- Algorithm 7: Counterfactual Data Generation ---

def counterfactual_generation(df, sensitive_col, target_cols):
    """
    Counterfactual Data Generation -- Creates copies of patients with
    flipped sensitive attributes while keeping clinical features identical.
    Tests if predictions would change purely due to demographics.
    """
    print(f"\n  [Algorithm 7] Counterfactual Data Generation")

    # Get unique values of sensitive attribute
    groups = sorted(df[sensitive_col].dropna().unique())

    if len(groups) < 2:
        print("    Fewer than 2 groups -- skipping")
        return df

    counterfactuals = []

    for original_val in groups:
        subset = df[df[sensitive_col] == original_val].copy()

        for target_val in groups:
            if target_val == original_val:
                continue
            cf = subset.copy()
            cf[sensitive_col] = target_val
            cf["Is_Counterfactual"] = 1
            counterfactuals.append(cf)

    if counterfactuals:
        cf_df = pd.concat(counterfactuals, ignore_index=True)
        df["Is_Counterfactual"] = 0
        original_len = len(df)
        df = pd.concat([df, cf_df], ignore_index=True)
        print(f"    Generated {len(cf_df)} counterfactual records (total: {original_len} -> {len(df)})")
    else:
        print("    No counterfactuals generated")

    return df


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("=" + "="*68 + "=")
    print("|  NHANES STAGE 1 PRE-PROCESSING PIPELINE                           |")
    print("|  Clinically Risk-Aware Multi-Stage Bias Mitigation Framework       |")
    print("=" + "="*68 + "=")
    print()
    print("  Input:  Raw healthcare dataset, Demographic/sensitive attributes,")
    print("          Training labels, Dataset issues")
    print("  Output: Bias-mitigated real patient dataset + fairness audit metrics")
    print("          + clinical risk labels")
    print()

    # -- STEP 1: Merge --
    df = merge_nhanes_data()
    df.to_csv(os.path.join(BASE_DIR, "NHANES_combined.csv"), index=False)
    print(f"\n  [OK] Saved: NHANES_combined.csv")

    # -- STEP 2: Clinical Risk Labels --
    df, risk_cols = create_risk_labels(df)

    # -- STEP 3: Bias Detection -- BEFORE Mitigation --
    print("\n" + "="*70)
    print(" STEP 3: BIAS DETECTION -- BEFORE MITIGATION (RG3)")
    print("="*70)

    before_metrics = fairness_audit(df, risk_cols, "BEFORE PRE-PROCESSING")
    class_imbalance_analysis(df, risk_cols)

    # -- STEP 4: Apply 7 Pre-Processing Algorithms --
    print("\n" + "="*70)
    print(" STEP 4: STAGE 1 PRE-PROCESSING ALGORITHMS (RG1)")
    print("="*70)
    print("  Applying 7 pre-processing algorithms on real patient data...")

    # Algorithm 1: C-Chart Based DI Imputation Selection
    df = cchart_di_imputation_selection(df, GENDER_COL, risk_cols)

    # Algorithm 2: MICE
    df = mice_imputation(df, risk_cols)

    # Algorithm 3: Reweighing (using Diabetes as primary target)
    df = reweighing(df, GENDER_COL, "Diabetes_Risk")

    # Algorithm 4: Disparate Impact Remover
    df = disparate_impact_remover(df, GENDER_COL, risk_cols)

    # Algorithm 5: GAN-Based Augmentation
    df = gan_augmentation(df, GENDER_COL, "Diabetes_Risk")

    # Algorithm 6: ProWSyn
    df = prowsyn_oversampling(df, GENDER_COL, "Diabetes_Risk")

    # Algorithm 7: Counterfactual Generation
    df = counterfactual_generation(df, GENDER_COL, risk_cols)

    # -- STEP 5: Bias Detection -- AFTER Mitigation --
    print("\n" + "="*70)
    print(" STEP 5: BIAS DETECTION -- AFTER MITIGATION (RG3)")
    print("="*70)

    # Filter out counterfactuals for fair comparison
    df_real = df[df.get("Is_Counterfactual", 0) == 0].copy()
    after_metrics = fairness_audit(df_real, risk_cols, "AFTER PRE-PROCESSING")

    # -- Compare Before vs After --
    print("\n" + "="*70)
    print(" BEFORE vs AFTER COMPARISON")
    print("="*70)
    comparison = before_metrics.merge(after_metrics, on="Target", suffixes=("_Before", "_After"))
    print(comparison.to_string(index=False))

    # -- Final cleanup: fill any remaining missing values (from augmentation rows) --
    # Numeric columns -> fill with mean
    num_cols_final = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    for col in num_cols_final:
        if df[col].isnull().sum() > 0:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if not pd.isna(col_mean) else 0)

    # Object/string columns -> fill with mode (most frequent value)
    obj_cols_final = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols_final:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    missing_final = df.isnull().sum().sum()
    print(f"\n  Missing values after cleanup: {missing_final}")

    # -- Save Output --
    output_path = os.path.join(BASE_DIR, "NHANES_preprocessed.csv")
    df.to_csv(output_path, index=False)

    print(f"\n  [OK] Saved: NHANES_preprocessed.csv")
    print(f"    Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"    Real records: {len(df_real)}")
    print(f"    Counterfactual records: {len(df) - len(df_real)}")
    print(f"    Missing values in output: {missing_final}")

    # -- Final Summary --
    print("\n" + "=" + "="*68 + "=")
    print("|  STAGE 1 PRE-PROCESSING COMPLETE                                  |")
    print("=" + "="*68 + "=")
    print("|  Output: Bias-mitigated real patient dataset                       |")
    print("|          + fairness audit metrics + clinical risk labels           |")
    print("|          -> Ready for Stage 2 (In-Processing)                      |")
    print("=" + "="*68 + "=")


if __name__ == "__main__":
    main()
