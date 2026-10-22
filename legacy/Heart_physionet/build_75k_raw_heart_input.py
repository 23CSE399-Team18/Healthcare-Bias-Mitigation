import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def create_single_folder(item):
    dataset_dir, i = item
    p_dir = os.path.join(dataset_dir, f"patient{i:05d}")
    if not os.path.exists(p_dir):
        try:
            os.mkdir(p_dir)
        except FileExistsError:
            pass
    if i <= 1000:
        hea_file = os.path.join(p_dir, f"patient{i:05d}.hea")
        if not os.path.exists(hea_file):
            with open(hea_file, "w") as f:
                f.write(f"patient{i:05d} 15 1000 120000 15:00:00 24/09/2025\n# age: 58\n# sex: male\n# reason for admission: Myocardial infarction\n")

def create_75k_heart_input():
    print("=================================================================")
    print("BUILDING 75,000 PATIENT FOLDERS & RAW INPUT DATASET (EXACT PTB LAYOUT)")
    print("=================================================================")

    base_dir = r"d:\Heart"
    dataset_dir = os.path.join(base_dir, "Dataset", "heart")
    os.makedirs(dataset_dir, exist_ok=True)

    n_samples = 75000

    print(f"Creating 75,000 patient subfolders concurrently in: {dataset_dir}")
    items = [(dataset_dir, i) for i in range(1, n_samples + 1)]
    with ThreadPoolExecutor(max_workers=64) as executor:
        list(executor.map(create_single_folder, items))

    print(f"[OK] Successfully created 75,000 patient subfolders (patient00001 to patient75000)!")

    ptb_csv_path = os.path.join(base_dir, 'PTB_Debiased_Records.csv')
    if os.path.exists(ptb_csv_path):
        df_tpl = pd.read_csv(ptb_csv_path)
    else:
        df_tpl = pd.DataFrame()

    ptb_cols = [
        'patient_id', 'num_signals', 'sampling_freq_hz', 'num_samples', 'duration_sec',
        'age', 'sex', 'diagnose', 'reason for admission', 'acute infarction (localization)',
        'former infarction (localization)', 'additional diagnoses', 'smoker',
        'number of coronary vessels involved', 'hemodynamics', 'ventriculography', 'chest x-ray',
        'peripheral blood pressure (syst/diast)', 'pulmonary artery pressure (at rest) (syst/diast)',
        'pulmonary artery pressure (at rest) (mean)', 'pulmonary capillary wedge pressure (at rest)',
        'cardiac output (at rest)', 'cardiac index (at rest)', 'stroke volume index (at rest)',
        'pulmonary artery pressure (laod) (syst/diast)', 'pulmonary artery pressure (laod) (mean)',
        'pulmonary capillary wedge pressure (load)', 'cardiac output (load)', 'cardiac index (load)',
        'stroke volume index (load)', 'aorta (at rest) (syst/diast)', 'aorta (at rest) mean',
        'left ventricular enddiastolic pressure', 'left coronary artery stenoses (riva)',
        'left coronary artery stenoses (rcx)', 'right coronary artery stenoses (rca)',
        'echocardiography', 'therapy', 'medication pre admission', 'start lysis therapy (hh.mm)',
        'lytic agent', 'dosage (lytic agent)', 'additional medication', 'in hospital medication',
        'medication after discharge'
    ]

    print("Generating 75,000 raw input patient records with embedded bias indicators (?, NaN, outliers, repeat duplicates)...")
    np.random.seed(42)

    unique_pids = [f"patient{i:05d}" for i in range(1, 56251)]
    patient_pool = unique_pids + list(np.random.choice(unique_pids, size=n_samples - len(unique_pids)))
    np.random.shuffle(patient_pool)
    patient_ids = patient_pool[:n_samples]

    data_dict = {'patient_id': patient_ids}

    for col in ptb_cols[1:]:
        if col in df_tpl.columns and not df_tpl[col].dropna().empty:
            vals = df_tpl[col].dropna().values
            data_dict[col] = np.random.choice(vals, size=n_samples, replace=True)
        else:
            if 'num_signals' in col:
                data_dict[col] = np.random.choice([12, 15], size=n_samples)
            elif 'sampling' in col:
                data_dict[col] = 1000
            elif 'num_samples' in col:
                data_dict[col] = np.random.choice([38400, 120000], size=n_samples)
            elif 'duration' in col:
                data_dict[col] = np.random.choice([115, 120, 180, 240], size=n_samples)
            elif 'age' in col:
                data_dict[col] = np.random.normal(61.5, 11.2, size=n_samples).round(1)
            elif 'sex' in col:
                data_dict[col] = np.random.choice(['male', 'female', '?'], size=n_samples, p=[0.72, 0.25, 0.03])
            elif 'reason for admission' in col or 'diagnose' in col:
                data_dict[col] = np.random.choice(['Myocardial infarction', 'Healthy control', 'Cardiomyopathy', 'Bundle branch block', 'Dysrhythmia', 'Other Pathology'], size=n_samples, p=[0.68, 0.15, 0.05, 0.05, 0.04, 0.03])
            elif 'smoker' in col:
                data_dict[col] = np.random.choice(['yes', 'no', '?'], size=n_samples, p=[0.32, 0.58, 0.10])
            elif 'vessels' in col:
                data_dict[col] = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.40, 0.18, 0.07])
            elif 'pressure' in col or 'bp' in col:
                sys_val = np.random.normal(134, 18, size=n_samples).round().astype(int)
                dia_val = np.random.normal(84, 11, size=n_samples).round().astype(int)
                data_dict[col] = [f"{s}/{d}" for s, d in zip(sys_val, dia_val)]
            else:
                data_dict[col] = np.random.randint(1, 100, size=n_samples)

    df_raw_75k = pd.DataFrame(data_dict)

    for col in ['age', 'sex', 'smoker', 'number of coronary vessels involved', 'peripheral blood pressure (syst/diast)', 'chest x-ray', 'additional diagnoses']:
        if col in df_raw_75k.columns:
            df_raw_75k[col] = df_raw_75k[col].astype(object)

    print("Injecting Missing Data Bias ('?', 'NaN', '') into clinical feature cells...")
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.20), replace=False)
    missing_tokens = ['?', 'NaN', 'nan', '']
    target_miss_cols = ['age', 'sex', 'smoker', 'number of coronary vessels involved', 'peripheral blood pressure (syst/diast)', 'chest x-ray']
    for idx in missing_indices:
        tok = np.random.choice(missing_tokens)
        c = np.random.choice(target_miss_cols)
        df_raw_75k.at[idx, c] = tok

    print("Injecting Outlier Bias (extreme unphysiological readings) into clinical cells...")
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for idx in outlier_indices[:len(outlier_indices)//2]:
        df_raw_75k.at[idx, 'peripheral blood pressure (syst/diast)'] = np.random.choice(['360/230', '340/210', '25/15', '15/5'])
    for idx in outlier_indices[len(outlier_indices)//2:]:
        df_raw_75k.at[idx, 'age'] = np.random.choice(['145', '160', '-5'])

    print("Injecting Duplicate Bias (Exact & Fuzzy repeat patient acquisitions)...")
    dup_indices = np.random.choice(n_samples, size=int(n_samples * 0.12), replace=False)
    for idx in dup_indices:
        donor_idx = np.random.randint(0, n_samples)
        df_raw_75k.at[idx, 'patient_id'] = df_raw_75k.at[donor_idx, 'patient_id']
        df_raw_75k.at[idx, 'age'] = df_raw_75k.at[donor_idx, 'age']
        df_raw_75k.at[idx, 'sex'] = df_raw_75k.at[donor_idx, 'sex']
        df_raw_75k.at[idx, 'reason for admission'] = df_raw_75k.at[donor_idx, 'reason for admission']
        if np.random.rand() > 0.5:
            df_raw_75k.at[idx, 'peripheral blood pressure (syst/diast)'] = df_raw_75k.at[donor_idx, 'peripheral blood pressure (syst/diast)']

    csv_path = os.path.join(dataset_dir, "PTB_Heart_Raw_Input_75000_Patients.csv")
    csv_root = os.path.join(base_dir, "PTB_Heart_Raw_Input_75000_Patients.csv")
    excel_path = os.path.join(dataset_dir, "PTB_Heart_Raw_Input_75000_Patients.xlsx")
    excel_root = os.path.join(base_dir, "PTB_Heart_Raw_Input_75000_Patients.xlsx")

    print(f"Saving Raw 75k CSV dataset to: {csv_path}")
    df_raw_75k.to_csv(csv_path, index=False)
    df_raw_75k.to_csv(csv_root, index=False)

    print(f"Saving Raw 75k Excel dataset (sheet name 'heart') to: {excel_path}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_raw_75k.head(75000).to_excel(writer, sheet_name='heart', index=False)

    import shutil
    shutil.copyfile(excel_path, excel_root)

    print("\n=================================================================")
    print("75,000 PATIENT RAW INPUT DATASET BUILD COMPLETE!")
    print(f"Total Patient Folders: 75,000 in {dataset_dir}")
    print(f"Excel File: {excel_path} (Sheet Name: 'heart', Shape: {df_raw_75k.shape})")
    print("=================================================================")

if __name__ == '__main__':
    create_75k_heart_input()
