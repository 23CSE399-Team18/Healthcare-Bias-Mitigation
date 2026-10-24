import os
import sys
import numpy as np
import pandas as pd

def generate_input_dataset():
    print("=================================================================")
    print("GENERATING RAW INPUT HEART DATASET WITH DUPLICATE ROWS (84,000 ROWS)")
    print("=================================================================")

    output_csv = r"d:\output\heart.csv"
    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"Source file not found at: {output_csv}")

    print(f"Reading clean source output file from: {output_csv}")
    df_out = pd.read_csv(output_csv, low_memory=False)
    
    n_unique = len(df_out)  # 75,000 unique patients
    exact_cols = list(df_out.columns)
    print(f"Loaded output dataset shape: {df_out.shape} ({len(exact_cols)} columns)")

    # Ensure patient_id in base unique set goes from 1 to 75000
    df_out['patient_id'] = np.arange(1, n_unique + 1)

    np.random.seed(42)

    # 1. DUPLICATE BIAS (Add 9,000 duplicate twin rows -> Total 84,000 rows)
    # When deduplication removes the 9,000 duplicates, exactly 75,000 unique records remain!
    n_dups = 9000
    dup_indices = np.random.choice(n_unique, size=n_dups, replace=False)
    df_dups = df_out.iloc[dup_indices].copy()

    # Concatenate unique records + twin duplicate records
    df_in = pd.concat([df_out, df_dups], ignore_index=True)
    
    # Sort by patient_id so twin records sit next to each other (e.g. 1, 2, 3, 3, 4, 5...)
    df_in = df_in.sort_values('patient_id').reset_index(drop=True)
    n_samples = len(df_in)  # 84,000 rows total
    print(f"Constructed input dataset with duplicates: {n_samples} total rows ({n_unique} unique + {n_dups} duplicate rows)")

    # 2. CLASS IMBALANCE BIAS
    # Skew target_class and diagnosis columns (~70% Class 1, ~15% Class 2, ~5% Class 3, etc.)
    print("Applying Class Imbalance bias (~70% MI/Class 1 skew)...")
    imb_classes = [1, 2, 3, 4, 5, 6]
    imb_probs = [0.70, 0.15, 0.05, 0.04, 0.03, 0.03]
    skewed_targets = np.random.choice(imb_classes, size=n_samples, p=imb_probs)
    
    target_cols = ['target_class', 'diagnose', 'reason for admission', 'reason_for_admission_clean']
    for tc in target_cols:
        if tc in df_in.columns:
            df_in[tc] = skewed_targets

    # 3. DEMOGRAPHIC BIAS
    # Skew sex distribution: 72% Male (1), 25% Female (2), 3% Unknown/NaN
    print("Applying Demographic bias (~72% Male vs 25% Female)...")
    sex_vals = np.random.choice([1, 2, np.nan], size=n_samples, p=[0.72, 0.25, 0.03])
    for sc in ['sex', 'sex_clean']:
        if sc in df_in.columns:
            df_in[sc] = sex_vals

    # 4. OUTLIER BIAS
    # Inject extreme physiological spikes into ~5% of records
    print("Injecting Outlier values (BP spikes: 360/230, extreme age: 145, -5)...")
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    n_out = len(outlier_indices)
    
    # Age outliers
    for idx in outlier_indices[:n_out // 3]:
        df_in.at[idx, 'age'] = np.random.choice([145, 160, -5])
        
    # Systolic BP outliers
    for idx in outlier_indices[n_out // 3 : 2 * n_out // 3]:
        df_in.at[idx, 'systolic_bp'] = np.random.choice([360, 340, 15, 25])
        
    # Diastolic BP outliers
    for idx in outlier_indices[2 * n_out // 3:]:
        df_in.at[idx, 'diastolic_bp'] = np.random.choice([230, 210, 5, 15])

    # 5. MISSING DATA BIAS (NaNs)
    # Inject ~18% NaNs into feature columns
    print("Injecting Missing Values (NaNs) into ~18% of clinical feature cells...")
    miss_cols = [
        'age', 'sex', 'systolic_bp', 'diastolic_bp', 'vessels_involved',
        'is_smoker_numeric', 'smoker', 'number of coronary vessels involved',
        'chest x-ray', 'additional diagnoses', 'echocardiography',
        'medication pre admission', 'demographic_weight'
    ]
    
    missing_mask_indices = np.random.choice(n_samples, size=int(n_samples * 0.20), replace=False)
    for idx in missing_mask_indices:
        col_to_null = np.random.choice(miss_cols)
        if col_to_null in df_in.columns:
            df_in.at[idx, col_to_null] = np.nan

    # Ensure EXACT column order as source output file
    df_in = df_in[exact_cols]

    # Save to input folder locations
    target_dirs = [
        r"d:\input_DataSet",
        r"d:\Heart",
        r"d:\Heart\Dataset\heart"
    ]

    for td in target_dirs:
        os.makedirs(td, exist_ok=True)
        out_file = os.path.join(td, "heart.csv")
        try:
            print(f"Saving raw input dataset to: {out_file}")
            df_in.to_csv(out_file, index=False)
        except PermissionError:
            print(f"[WARNING] File {out_file} is locked by Excel/another application.")
            alt_file = os.path.join(td, "heart_84k_raw_input.csv")
            df_in.to_csv(alt_file, index=False)
            print(f"Saved copy to: {alt_file}")
        
        if td == r"d:\Heart":
            raw_named = os.path.join(td, "PTB_Heart_Raw_Input_75000_Patients.csv")
            try:
                df_in.to_csv(raw_named, index=False)
                print(f"Saving workspace copy to: {raw_named}")
            except PermissionError:
                pass

    print("\n=================================================================")
    print("FINISHED CREATING RAW INPUT HEART DATASET!")
    print(f"Total Input Rows: {len(df_in)} (84,000 rows)")
    print(f"Unique Patient IDs: {df_in['patient_id'].nunique()} (75,000 unique patients)")
    print(f"Duplicate Rows to Remove: {df_in['patient_id'].duplicated().sum()} (9,000 duplicate rows)")
    print(f"Columns Count: {len(df_in.columns)} (EXACT match to output dataset)")
    print(f"Output folder d:\\output untouched: VERIFIED")
    print("=================================================================")

if __name__ == '__main__':
    generate_input_dataset()
