import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def process_diabetes_final():
    print("=" * 80, flush=True)
    print("STARTING DIABETES HEALTHCARE BIAS MITIGATION & SYNTHETIC BALANCING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    
    start_time = time.time()
    input_path = r"d:\output\diabetes.csv"
    output_path = r"d:\output\diabetes_final.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.", flush=True)
        sys.exit(1)
        
    print(f"[STEP 1] Loading raw dataset from: {input_path}", flush=True)
    t0 = time.time()
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Raw dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns (Loaded in {time.time()-t0:.2f}s)", flush=True)
    print(f"  Total missing values: {df.isnull().sum().sum():,}", flush=True)
    
    # ---------------------------------------------------------
    # STEP 1: DEDUPLICATION & IDENTIFIER STANDARDIZATION
    # ---------------------------------------------------------
    print("\n[STEP 2] Deduplicating records & standardizing identifiers...", flush=True)
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(df)
    print(f"  Deduplicated rows removed: {initial_rows - dedup_rows}", flush=True)
    
    # ---------------------------------------------------------
    # STEP 2: CATEGORICAL HARMONIZATION & ENCODING BIAS FIX
    # ---------------------------------------------------------
    print("\n[STEP 3] Harmonizing categorical attributes & fixing domain/encoding bias...", flush=True)
    
    # Standardize Ethnicity
    if 'Ethnicity' in df.columns:
        df['Ethnicity'] = df['Ethnicity'].replace({
            'Hisp': 'Hispanic/Latino',
            'H/L': 'Hispanic/Latino',
            'Non-H/L': 'Non-Hispanic/Latino'
        })
        
    # Standardize MEDS
    if 'MEDS' in df.columns:
        df['MEDS'] = df['MEDS'].fillna('No_Medication').replace({'None': 'No_Medication'})

    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'string']
    for exclude_col in ['SUBJECT NUMBER', 'Group', 'group2']:
        if exclude_col in cat_cols:
            cat_cols.remove(exclude_col)

    print(f"  Harmonizing {len(cat_cols)} categorical feature columns...", flush=True)
    
    # Fast Mode Imputation
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
                
    # Ensure group2 matches Group if present
    if 'group2' in df.columns:
        df['group2'] = df['Group']

    # ---------------------------------------------------------
    # STEP 3: NUMERIC MISSING VALUE IMPUTATION (100% FILLED)
    # ---------------------------------------------------------
    print("\n[STEP 4] Imputing missing numeric values (Fast Vectorized Median)...", flush=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric columns count: {len(num_cols)}", flush=True)
    
    initial_num_nulls = df[num_cols].isnull().sum().sum()
    print(f"  Initial missing numeric values: {initial_num_nulls:,}", flush=True)
    
    # Vectorized column median imputation
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    df[num_cols] = df[num_cols].fillna(0)
    
    remaining_nulls = df.isnull().sum().sum()
    print(f"  Remaining missing values after full imputation: {remaining_nulls} (Target: 0)", flush=True)
    assert remaining_nulls == 0, f"Error: {remaining_nulls} null values remain!"

    # ---------------------------------------------------------
    # STEP 4: OUTLIER BIAS & DATA NOISE MITIGATION
    # ---------------------------------------------------------
    print("\n[STEP 5] Mitigating outlier bias & signal noise (Vectorized Quantile Capping)...", flush=True)
    q01 = df[num_cols].quantile(0.01)
    q99 = df[num_cols].quantile(0.99)
    df[num_cols] = df[num_cols].clip(lower=q01, upper=q99, axis=1)
    print("  Outliers capped within robust [P01, P99] bounds across all continuous features.", flush=True)

    # ---------------------------------------------------------
    # STEP 5: DEMOGRAPHIC & SYNTHETIC CLASS BALANCING (SMOTE)
    # ---------------------------------------------------------
    print("\n[STEP 6] Evaluating class imbalance & generating synthetic balanced cohort...", flush=True)
    target_counts = df['Group'].value_counts()
    print("  Current class distribution:", flush=True)
    for cls, count in target_counts.items():
        print(f"    - {cls}: {count:,} ({count/len(df)*100:.2f}%)", flush=True)
        
    majority_class = 'Control'
    minority_class = 'Diabetes_Type2'
    
    n_majority = target_counts[majority_class]
    n_minority = target_counts[minority_class]
    n_synth_needed = n_majority - n_minority
    
    print(f"  Target balance: {n_majority:,} per class", flush=True)
    print(f"  Synthetic samples to generate for '{minority_class}': {n_synth_needed:,}", flush=True)
    
    # Extract minority class dataframe
    df_minority = df[df['Group'] == minority_class].reset_index(drop=True)
    df_majority = df[df['Group'] == majority_class].reset_index(drop=True)
    
    # Prepare numeric matrix for SMOTE interpolation
    X_minority_num = df_minority[num_cols].values
    
    print("  Generating synthetic minority feature vectors via Pairwise SMOTE interpolation...", flush=True)
    np.random.seed(42)
    
    # Randomly select base indices and neighbor indices from minority pool
    base_idx = np.random.choice(n_minority, size=n_synth_needed, replace=True)
    neighbor_idx = np.random.choice(n_minority, size=n_synth_needed, replace=True)
    
    # Random interpolation factor lambda ~ U(0.05, 0.95)
    lambdas = np.random.uniform(0.05, 0.95, size=(n_synth_needed, 1)).astype(np.float32)
    
    # Interpolate numeric features
    X_base = X_minority_num[base_idx]
    X_neighbor = X_minority_num[neighbor_idx]
    X_synth_num = X_base + lambdas * (X_neighbor - X_base)
    
    # Create Synthetic Dataframe
    df_synth = pd.DataFrame(X_synth_num, columns=num_cols)
    
    # Inherit categorical attributes & demographic features from base samples
    for c in cat_cols:
        df_synth[c] = df_minority[c].iloc[base_idx].values
        
    df_synth['Group'] = minority_class
    if 'group2' in df.columns:
        df_synth['group2'] = minority_class
        
    # Generate unique Subject Numbers for synthetic cohort
    start_sub_id = len(df) + 1
    synth_sub_ids = [f"S{i:06d}" for i in range(start_sub_id, start_sub_id + n_synth_needed)]
    df_synth['SUBJECT NUMBER'] = synth_sub_ids
    
    # Reorder columns to match original dataframe exactly
    df_synth = df_synth[df.columns]
    
    # Combine majority, minority real, and synthetic minority
    df_final = pd.concat([df_majority, df_minority, df_synth], axis=0, ignore_index=True)
    
    # Shuffle final dataset
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Round float columns to 4 decimal places for clean storage
    float_cols = df_final.select_dtypes(include=['float64', 'float32']).columns
    df_final[float_cols] = df_final[float_cols].round(4)

    # ---------------------------------------------------------
    # STEP 6: VERIFICATION & EXPORT
    # ---------------------------------------------------------
    print("\n[STEP 7] Verifying final dataset properties...", flush=True)
    final_shape = df_final.shape
    final_nulls = df_final.isnull().sum().sum()
    final_counts = df_final['Group'].value_counts()
    
    print(f"  Final shape: {final_shape[0]:,} rows x {final_shape[1]} columns", flush=True)
    print(f"  Final total nulls: {final_nulls} (100% Filled)", flush=True)
    print("  Final class distribution:", flush=True)
    for cls, count in final_counts.items():
        print(f"    - {cls}: {count:,} ({count/len(df_final)*100:.2f}%)", flush=True)
        
    print(f"\n[STEP 8] Exporting to: {output_path}...", flush=True)
    t_exp = time.time()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    df_final.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - start_time
    
    print("=" * 80, flush=True)
    print(f"DIABETES PIPELINE SUCCESSFUL!", flush=True)
    print(f"Saved file: {output_path} ({file_size_mb:.2f} MB in {time.time()-t_exp:.2f}s export time)", flush=True)
    print(f"Total processing time: {elapsed:.2f} seconds", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    process_diabetes_final()
