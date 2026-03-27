import pandas as pd
import os

# === CDC DIABETES ===
print('='*70)
print('  CDC DIABETES - RAW DATA')
print('='*70)
cdc = pd.read_csv(r'c:\Users\U SRIYA\Documents\Data_Set\CDC diabetes\diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
print(f'  Shape: {cdc.shape[0]} rows x {cdc.shape[1]} cols')
print(f'  Columns ({len(cdc.columns)}):')
for c in cdc.columns:
    print(f'    - {c}: unique={cdc[c].nunique()}, missing={cdc[c].isnull().sum()}, dtype={cdc[c].dtype}')
print(f'  Target: Diabetes_binary -> 0={int((cdc.Diabetes_binary==0).sum())}, 1={int((cdc.Diabetes_binary==1).sum())}')
print(f'  Sex: 0(Female)={int((cdc.Sex==0).sum())}, 1(Male)={int((cdc.Sex==1).sum())}')

print()
print('  CDC Preprocessed:')
cdc_p = pd.read_csv(r'c:\Users\U SRIYA\Documents\Data_Set\preprocessed_output\cdc_diabetes_preprocessed.csv')
print(f'  Shape: {cdc_p.shape[0]} rows x {cdc_p.shape[1]} cols')
print(f'  Missing: {cdc_p.isnull().sum().sum()}')
print(f'  Columns: {list(cdc_p.columns)}')

# === eICU ===
print()
print('='*70)
print('  eICU - RAW DATA')
print('='*70)
eicu_dir = r'c:\Users\U SRIYA\Documents\Data_Set\eICU'
eicu_files = [f for f in os.listdir(eicu_dir) if f.endswith('.csv')]
print(f'  Files in eICU folder: {eicu_files}')

for f in eicu_files[:5]:
    df = pd.read_csv(os.path.join(eicu_dir, f), nrows=3)
    print(f'\n  {f}:')
    print(f'    Columns: {list(df.columns)}')

# Load the main patient file
patient_file = None
for f in eicu_files:
    if 'patient' in f.lower():
        patient_file = f
        break

if patient_file:
    pat = pd.read_csv(os.path.join(eicu_dir, patient_file))
    print(f'\n  Main patient file: {patient_file}')
    print(f'  Shape: {pat.shape[0]} rows x {pat.shape[1]} cols')
    print(f'  Columns:')
    for c in pat.columns:
        print(f'    - {c}: unique={pat[c].nunique()}, missing={pat[c].isnull().sum()}')

print()
print('='*70)
print('  eICU - PREPROCESSED')
print('='*70)
eicu_p = pd.read_csv(r'c:\Users\U SRIYA\Documents\Data_Set\preprocessed_output\eicu_preprocessed.csv')
print(f'  Shape: {eicu_p.shape[0]} rows x {eicu_p.shape[1]} cols')
print(f'  Missing: {eicu_p.isnull().sum().sum()}')
print(f'  Columns: {list(eicu_p.columns)}')
print(f'  Target (Mortality): 0={int((eicu_p.Mortality==0).sum())}, 1={int((eicu_p.Mortality==1).sum())}')
print(f'  Gender_Encoded: {eicu_p.Gender_Encoded.value_counts().to_dict()}')
if 'Is_Counterfactual' in eicu_p.columns:
    print(f'  Real rows: {int((eicu_p.Is_Counterfactual==0).sum())}')
    print(f'  Counterfactuals: {int((eicu_p.Is_Counterfactual==1).sum())}')
print()
print('  First 5 rows:')
print(eicu_p.head().to_string())
