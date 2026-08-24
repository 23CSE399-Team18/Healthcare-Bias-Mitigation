# -*- coding: utf-8 -*-
import pandas as pd
import os

od = r'c:\Users\U SRIYA\Documents\Data_Set\preprocessed_output'

mapping = {
    'breast_cancer_preprocessed.csv': 'CHRONIC',
    'cdc_diabetes_preprocessed.csv': 'CHRONIC',
    'ckd_preprocessed.csv': 'CHRONIC',
    'eicu_preprocessed.csv': 'ACUTE+CHRONIC',
    'heart_disease_preprocessed.csv': 'CHRONIC',
    'NHANES_preprocessed.csv': 'CHRONIC',
}

for fname, label in mapping.items():
    path = os.path.join(od, fname)
    if not os.path.exists(path):
        print(f'[SKIP] {fname} not found')
        continue
    df = pd.read_csv(path)
    df['Chronic_Disease'] = label
    # Write to temp file first, then rename
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        print(f'[OK] {fname} -> Chronic_Disease={label} ({df.shape[0]}x{df.shape[1]})')
    except PermissionError:
        # If original is locked, save as _v2
        v2 = path.replace('.csv', '_v2.csv')
        os.rename(tmp, v2)
        print(f'[OK] Saved as {os.path.basename(v2)} (original was locked)')

print('\nDone!')
