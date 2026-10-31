import os
import sys
import numpy as np
import pandas as pd

def build_raw_input_dataset():
    print("=================================================================")
    print("BUILDING RAW INPUT DATASET (75,000 ROWS, EXACT OUTPUT FEATURE COLUMNS)")
    print("=================================================================")

    n_samples = 75000
    np.random.seed(42)

    # EXACT 11 FEATURE COLUMNS MATCHING OUTPUT DATASET
    # record_id, patient_id, age, sex, systolic_bp, diastolic_bp, vessels_involved, smoker, duration_sec, reason_for_admission, target_class

    record_ids = [f"rec_{i:05d}" for i in range(1, n_samples + 1)]

    # DUPLICATE BIAS: 56,250 unique patient IDs repeated across 75,000 records
    unique_pids = [f"patient{i:05d}" for i in range(1, 56251)]
    patient_pool = unique_pids + list(np.random.choice(unique_pids, size=n_samples - len(unique_pids)))
    np.random.shuffle(patient_pool)
    patient_ids = patient_pool[:n_samples]

    # DEMOGRAPHIC BIAS: 74% Male, 23% Female, 3% ?
    sex_vals = np.random.choice(['Male', 'Female', '?'], size=n_samples, p=[0.74, 0.23, 0.03])

    # Base Age
    age_vals = np.random.normal(61.5, 11.2, size=n_samples).round(1)

    # Base Systolic & Diastolic BP
    sys_bp = np.random.normal(134, 18, size=n_samples).round(1)
    dia_bp = np.random.normal(84, 11, size=n_samples).round(1)

    # Base Vessels Involved
    vessels = np.random.choice(['0', '1', '2', '3'], size=n_samples, p=[0.35, 0.40, 0.18, 0.07])

    # Base Smoker Status
    smoker = np.random.choice(['No', 'Yes', '?'], size=n_samples, p=[0.58, 0.32, 0.10])

    # Base Duration Sec
    duration = np.random.choice([115, 120, 180, 240, 300], size=n_samples, p=[0.20, 0.55, 0.10, 0.10, 0.05])

    # CLASS IMBALANCE BIAS: 68% Myocardial infarction, 16% Healthy control, 6% Cardiomyopathy, 5% Bundle branch block, 3% Dysrhythmia, 2% Valvular heart disease
    diagnoses = np.random.choice(
        ['Myocardial infarction', 'Healthy control', 'Cardiomyopathy', 'Bundle branch block', 'Dysrhythmia', 'Valvular heart disease'],
        size=n_samples,
        p=[0.68, 0.16, 0.06, 0.05, 0.03, 0.02]
    )

    df_raw = pd.DataFrame({
        'record_id': record_ids,
        'patient_id': patient_ids,
        'age': age_vals.astype(object),
        'sex': sex_vals.astype(object),
        'systolic_bp': sys_bp.astype(object),
        'diastolic_bp': dia_bp.astype(object),
        'vessels_involved': vessels.astype(object),
        'smoker': smoker.astype(object),
        'duration_sec': duration,
        'reason_for_admission': diagnoses,
        'target_class': diagnoses
    })

    # 1. INJECT MISSING DATA BIAS DIRECTLY INTO FEATURE CELLS (?, NaN, nan, '')
    print("Injecting Missing Data Bias ('?', 'NaN', '') directly into feature cells...")
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.20), replace=False)
    missing_tokens = ['?', 'NaN', 'nan', '']
    target_miss_cols = ['age', 'sex', 'systolic_bp', 'diastolic_bp', 'vessels_involved', 'smoker']
    for idx in missing_indices:
        tok = np.random.choice(missing_tokens)
        c = np.random.choice(target_miss_cols)
        df_raw.at[idx, c] = tok

    # 2. INJECT OUTLIER BIAS DIRECTLY INTO NUMERIC/BP CELLS (360, 340, 25, 145, -5)
    print("Injecting Outlier Bias (extreme physiological spikes) directly into feature cells...")
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    half_out = len(outlier_indices) // 2
    for idx in outlier_indices[:half_out]:
        df_raw.at[idx, 'systolic_bp'] = np.random.choice(['360', '340', '25', '15'])
        df_raw.at[idx, 'diastolic_bp'] = np.random.choice(['230', '210', '15', '5'])
    for idx in outlier_indices[half_out:]:
        df_raw.at[idx, 'age'] = np.random.choice(['145', '160', '-5'])

    # 3. INJECT DUPLICATE BIAS (Exact & Fuzzy repeat patient acquisitions)
    print("Injecting Duplicate Bias directly into patient_id rows...")
    dup_indices = np.random.choice(n_samples, size=int(n_samples * 0.12), replace=False)
    for idx in dup_indices:
        donor_idx = np.random.randint(0, n_samples)
        df_raw.at[idx, 'patient_id'] = df_raw.at[donor_idx, 'patient_id']
        df_raw.at[idx, 'age'] = df_raw.at[donor_idx, 'age']
        df_raw.at[idx, 'sex'] = df_raw.at[donor_idx, 'sex']
        df_raw.at[idx, 'reason_for_admission'] = df_raw.at[donor_idx, 'reason_for_admission']
        df_raw.at[idx, 'target_class'] = df_raw.at[donor_idx, 'target_class']

    print("Checking final shape and columns of raw input dataset:")
    print("Shape:", df_raw.shape)
    print("Columns:", list(df_raw.columns))

    # SAVE TO CSV AND EXCEL LOCATIONS
    out_dirs = [r"d:\Heart", r"d:\Heart\Dataset\heart"]
    for od in out_dirs:
        os.makedirs(od, exist_ok=True)
        csv_path = os.path.join(od, "PTB_Heart_Raw_Input_75000_Patients.csv")
        df_raw.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        excel_path = os.path.join(od, "PTB_Heart_Raw_Input_75000_Patients.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_raw.to_excel(writer, sheet_name='heart', index=False)
        print(f"Saved Excel (sheet 'heart'): {excel_path}")

    print("=================================================================")
    print("FINISHED BUILDING RAW INPUT DATASET WITH EXACT OUTPUT COLUMNS!")
    print("=================================================================")

if __name__ == '__main__':
    build_raw_input_dataset()
