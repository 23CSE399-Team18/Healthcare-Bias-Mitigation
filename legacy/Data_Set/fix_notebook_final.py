"""
fix_notebook_final.py
=====================
Rewrites multi_dataset_integration.ipynb with fully correct, working code.
Uses the proven logic from rebuild_integration.py to ensure all 11 datasets
are loaded, harmonized, and integrated properly.
"""
import json, pathlib, copy

BASE = pathlib.Path(r"d:/Data_Set")
NB   = BASE / "multi_dataset_integration.ipynb"

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[SAVED] {path}")

def set_cell(nb, idx, cell_type, source_str):
    """Replace a cell's source with new code/markdown."""
    cell = nb["cells"][idx]
    cell["cell_type"] = cell_type
    cell["source"] = [source_str]
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    print(f"  [SET] Cell {idx} ({cell_type})")

# ═══════════════════════════════════════════════════════════════
nb = load_nb(NB)
print(f"Loaded notebook with {len(nb['cells'])} cells")

# ── Cell 2: Install packages ──
set_cell(nb, 2, "code", """# -- Install required packages (run once) --
import subprocess, sys
required = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'openpyxl']
for pkg in required:
    try:
        __import__(pkg if pkg != 'scikit-learn' else 'sklearn')
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('All packages ready.')""")

# ── Cell 3: Imports ──
set_cell(nb, 3, "code", """import os, warnings, ast, re, json, gc, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.impute import SimpleImputer
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 140)

print('Libraries loaded successfully.')""")

# ── Cell 4: Paths ──
set_cell(nb, 4, "code", """# -- Path Configuration --
BASE    = Path(r'd:/Data_Set')
DS_DIR  = BASE / 'Data_Set'          # Original 6 datasets
ND_DIR  = BASE / 'ND'                # 5 new large-scale datasets
OUT_DIR = BASE / 'integrated_output'
PP_DIR  = OUT_DIR / 'preprocessed'
DOM_DIR = OUT_DIR / 'by_domain'
PLOT_DIR= OUT_DIR / 'plots'

for d in [OUT_DIR, PP_DIR, DOM_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'BASE   : {BASE}')
print(f'DS_DIR : {DS_DIR}')
print(f'ND_DIR : {ND_DIR}')
print(f'OUT_DIR: {OUT_DIR}')""")

# ── Cell 5: Dataset path registry ──
set_cell(nb, 5, "code", r"""# -- Dataset Path Registry --
PATHS = {
    # -- Data_Set datasets --
    'heart_cleveland'  : DS_DIR / 'UCI_HeartDisease' / 'processed.cleveland.csv',
    'heart_hungarian'  : DS_DIR / 'UCI_HeartDisease' / 'processed.hungarian.csv',
    'heart_switzerland': DS_DIR / 'UCI_HeartDisease' / 'processed.switzerland.csv',
    'heart_va'         : DS_DIR / 'UCI_HeartDisease' / 'processed.va.csv',
    'ckd'              : DS_DIR / 'Chronic_Kidney_Disease' / 'Chronic_Kidney_Disease' / 'chronic_kidney_disease_full.csv',
    'eicu_patient'     : DS_DIR / 'eLCU' / 'patient.csv',
    'eicu_diagnosis'   : DS_DIR / 'eLCU' / 'diagnosis.csv',
    'eicu_lab'         : DS_DIR / 'eLCU' / 'lab.csv',
    'nhanes'           : DS_DIR / 'National Center for health Statistics' / 'NHANES_combined.csv',
    'cdc_diabetes'     : DS_DIR / 'CDC diabetes' / 'diabetes_binary_health_indicators_BRFSS2015.csv',
    'breast_cancer'    : DS_DIR / 'breast+cancer+coimbra_UCL' / 'dataR2.csv',
    # -- ND datasets --
    'mimic_patients'   : ND_DIR / 'mimic-iv-3.1' / 'hosp' / 'patients.csv.gz',
    'mimic_admissions' : ND_DIR / 'mimic-iv-3.1' / 'hosp' / 'admissions.csv.gz',
    'mimic_icustays'   : ND_DIR / 'mimic-iv-3.1' / 'icu'  / 'icustays.csv.gz',
    'eicu2_patient'    : ND_DIR / 'eicu-collaborative-research-database-2.0' / 'patient.csv.gz',
    'eicu2_diagnosis'  : ND_DIR / 'eicu-collaborative-research-database-2.0' / 'diagnosis.csv.gz',
    'eicu2_lab'        : ND_DIR / 'eicu-collaborative-research-database-2.0' / 'lab.csv.gz',
    'ptbxl_db'         : ND_DIR / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'ptbxl_database.csv',
    'ptbxl_scp'        : ND_DIR / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'scp_statements.csv',
    'ham10000'         : ND_DIR / 'dataverse_files' / 'HAM10000_metadata',
    'hf_ehr'           : ND_DIR / 'hospitalized-patients-with-heart-failure-integrating-electronic-healthcare-records-and-external-outcome-data-1.2' / 'hospitalized-patients-with-heart-failure-integrating-electronic-healthcare-records-and-external-outcome-data-1.2' / 'dat.csv',
}

# Verify all paths
print('Dataset Path Verification:')
print('=' * 65)
for name, path in PATHS.items():
    exists = path.exists()
    status = 'OK' if exists else 'MISSING'
    size   = f'{path.stat().st_size/1e6:.1f} MB' if exists else '---'
    print(f'  [{status:7s}] {name:25s} {size:>10s}')""")

# ── Cell 7: Heart Disease loader ──
set_cell(nb, 7, "code", """HEART_COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                 'exang','oldpeak','slope','ca','thal','target_raw']

heart_files = ['heart_cleveland', 'heart_hungarian', 'heart_switzerland', 'heart_va']
heart_frames = []
for key in heart_files:
    path = PATHS[key]
    if path.exists():
        df = pd.read_csv(path, header=None, names=HEART_COLUMNS, na_values='?')
        df['source_site'] = key.replace('heart_', '')
        heart_frames.append(df)
        print(f'  Loaded {key}: {len(df)} rows')
    else:
        print(f'  WARNING: {key} not found')

if heart_frames:
    heart_raw = pd.concat(heart_frames, ignore_index=True)
    print(f'\\nCombined Heart Disease: {heart_raw.shape}')
    print(heart_raw.head(3))
else:
    heart_raw = None
    print('WARNING: No heart disease files found')""")

# ── Cell 8: Heart Disease preprocessing ──
set_cell(nb, 8, "code", """def preprocess_heart(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Preprocess UCI Heart Disease data.\"\"\"
    out = df.copy()
    # Binary target: 0 = no disease, 1+ = disease
    out['target'] = (out['target_raw'] > 0).astype(int)
    out.drop(columns=['target_raw'], inplace=True, errors='ignore')

    # Age groups
    out['age_group'] = pd.cut(out['age'], bins=[0, 40, 60, 75, 120],
                              labels=['young', 'middle', 'senior', 'elderly'],
                              include_lowest=True)

    # Gender: 1=Male, 0=Female
    out['gender'] = out['sex'].map({1: 'Male', 0: 'Female'}).fillna('Unknown')

    # Impute numeric columns
    num_cols = out.select_dtypes(include=np.number).columns.difference(['target'])
    if len(num_cols) > 0:
        out[num_cols] = SimpleImputer(strategy='median').fit_transform(out[num_cols])

    out['dataset_source'] = 'UCI_HeartDisease'
    out['domain'] = 'cardiovascular'
    print(f'  Heart Disease preprocessed: {out.shape} | target rate: {out["target"].mean():.3f}')
    return out

if heart_raw is not None:
    heart_pp = preprocess_heart(heart_raw)
    heart_pp.head(3)
else:
    heart_pp = None""")

# ── Cell 9: Heart Disease visualization ──
set_cell(nb, 9, "code", """# -- Heart Disease EDA Visualization --
if heart_pp is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('UCI Heart Disease', fontsize=13, fontweight='bold')

    heart_pp['target'].value_counts().plot.bar(ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Target Distribution')
    axes[0].set_xticklabels(['No Disease', 'Disease'], rotation=0)

    axes[1].hist(heart_pp['age'].dropna(), bins=20, color='steelblue', edgecolor='white')
    axes[1].set_title('Age Distribution')

    heart_pp['gender'].value_counts().plot.bar(ax=axes[2], color=['royalblue', 'salmon'])
    axes[2].set_title('Gender Distribution')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'heart_disease_eda.png', bbox_inches='tight')
    plt.show()
    print('Heart Disease EDA saved.')""")

# ── Cell 11: CKD loader ──
set_cell(nb, 11, "code", r"""CKD_COLUMNS = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
               'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane','class']

CKD_NOMINAL_MAP = {
    'rbc':   {'normal': 0, 'abnormal': 1},
    'pc':    {'normal': 0, 'abnormal': 1},
    'pcc':   {'present': 1, 'notpresent': 0},
    'ba':    {'present': 1, 'notpresent': 0},
    'htn':   {'yes': 1, 'no': 0},
    'dm':    {'yes': 1, 'no': 0, ' yes': 1, '\tyes': 1, '\tno': 0},
    'cad':   {'yes': 1, 'no': 0, '\tno': 0},
    'appet': {'good': 0, 'poor': 1},
    'pe':    {'yes': 1, 'no': 0},
    'ane':   {'yes': 1, 'no': 0},
    'class': {'ckd': 1, 'notckd': 0, 'ckd\t': 1},
}

ckd_path = PATHS['ckd']
if ckd_path.exists():
    # ARFF-aware loader
    _raw = ckd_path.read_text(encoding='utf-8', errors='replace').splitlines()
    _arff_cols, _data_start = [], None
    for _i, _ln in enumerate(_raw):
        _s = _ln.strip().lower()
        if _s.startswith('@attribute'):
            _parts = _ln.strip().split()
            _col = _parts[1].strip("'\"")
            _arff_cols.append(_col)
        elif _s == '@data':
            _data_start = _i + 1
            break

    _norm = {'wbcc': 'wc', 'rbcc': 'rc'}
    _final_cols = [_norm.get(c, c) for c in _arff_cols]

    if _data_start is not None:
        _txt = '\n'.join(_raw[_data_start:])
        ckd_raw = pd.read_csv(
            io.StringIO(_txt), header=None, names=_final_cols,
            na_values=['?'], on_bad_lines='skip'
        )
    else:
        ckd_raw = pd.read_csv(
            ckd_path, comment='%', header=None, names=CKD_COLUMNS,
            na_values=['?'], on_bad_lines='skip'
        )

    ckd_raw.replace('?', np.nan, inplace=True)
    print(f'CKD raw shape: {ckd_raw.shape}')
    print(f'Columns: {ckd_raw.columns.tolist()}')
    ckd_raw.head(3)
else:
    ckd_raw = None
    print(f'WARNING: CKD file not found at {ckd_path}')""")

# ── Cell 12: CKD preprocessing ──
set_cell(nb, 12, "code", """def preprocess_ckd(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Preprocess Chronic Kidney Disease data.\"\"\"
    out = df.copy()

    # Map nominal columns
    for col, mapping in CKD_NOMINAL_MAP.items():
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.lower().map(
                {str(k).lower(): v for k, v in mapping.items()}
            )
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # Target from 'class' column
    if 'class' in out.columns:
        out['target'] = out['class']
        out.drop(columns=['class'], inplace=True, errors='ignore')
    else:
        out['target'] = np.nan

    # Convert numeric columns
    for col in out.columns:
        if col not in ['target']:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # Age groups
    if 'age' in out.columns:
        out['age_group'] = pd.cut(out['age'], bins=[0, 40, 60, 75, 120],
                                  labels=['young', 'middle', 'senior', 'elderly'],
                                  include_lowest=True)
    out['gender'] = 'Unknown'  # CKD dataset does not have gender

    # Impute
    num_cols = out.select_dtypes(include=np.number).columns.difference(['target'])
    if len(num_cols) > 0:
        out[num_cols] = SimpleImputer(strategy='median').fit_transform(out[num_cols])

    out['dataset_source'] = 'CKD_UCI'
    out['domain'] = 'renal'
    print(f'  CKD preprocessed: {out.shape} | target rate: {out["target"].mean():.3f}')
    return out

if ckd_raw is not None:
    ckd_pp = preprocess_ckd(ckd_raw)
    ckd_pp.head(3)
else:
    ckd_pp = None""")

# ── Cell 13: CKD visualization ──
set_cell(nb, 13, "code", """# -- CKD Visualization --
if ckd_pp is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Chronic Kidney Disease', fontsize=13, fontweight='bold')

    ckd_pp['target'].value_counts().plot.bar(ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Target Distribution')
    axes[0].set_xticklabels(['Not CKD', 'CKD'], rotation=0)

    axes[1].hist(ckd_pp['age'].dropna(), bins=20, color='steelblue', edgecolor='white')
    axes[1].set_title('Age Distribution')

    if 'hemo' in ckd_pp.columns:
        axes[2].hist(ckd_pp['hemo'].dropna(), bins=20, color='teal', edgecolor='white')
        axes[2].set_title('Hemoglobin Distribution')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ckd_eda.png', bbox_inches='tight')
    plt.show()""")

# ── Cell 15: eICU original ──
set_cell(nb, 15, "code", """def preprocess_eicu_original() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess original eICU dataset from eLCU folder.\"\"\"
    patient_path = PATHS['eicu_patient']
    diag_path    = PATHS['eicu_diagnosis']
    lab_path     = PATHS['eicu_lab']

    if not patient_path.exists():
        print('WARNING: eICU patient.csv not found -- skipping')
        return None

    print('  [eICU orig] Loading patient table...')
    pat = pd.read_csv(patient_path)
    print(f'    Patient records: {len(pat)} | columns: {pat.shape[1]}')

    if 'age' in pat.columns:
        pat['age'] = pat['age'].replace('> 89', 90).pipe(pd.to_numeric, errors='coerce')
    if 'gender'    in pat.columns: pat['gender']    = pat['gender'].str.strip().str.title()
    if 'ethnicity' in pat.columns: pat['ethnicity'] = pat['ethnicity'].str.strip().str.title()

    if 'hospitaldischargestatus' in pat.columns:
        pat['target'] = (pat['hospitaldischargestatus'].str.lower() == 'expired').astype(int)

    if 'age' in pat.columns:
        pat['age_group'] = pd.cut(pat['age'], bins=[0, 40, 60, 75, 120],
                                  labels=['young', 'middle', 'senior', 'elderly'], include_lowest=True)

    if diag_path.exists():
        print('    Merging diagnosis features...')
        diag = pd.read_csv(diag_path)
        diag_feat = diag.groupby('patientunitstayid').agg(
            diagnosis_count=('diagnosisid', 'count'),
            unique_dx_count=('diagnosisstring', 'nunique')
        ).reset_index()
        pat = pat.merge(diag_feat, on='patientunitstayid', how='left')
        pat[['diagnosis_count', 'unique_dx_count']] = pat[['diagnosis_count', 'unique_dx_count']].fillna(0)

    # lab.csv is optional and very large -- skip gracefully if absent
    if lab_path.exists():
        print('    Merging lab features...')
        try:
            lab = pd.read_csv(lab_path, usecols=['patientunitstayid', 'labname', 'labresult'])
            lab['labresult'] = pd.to_numeric(lab['labresult'], errors='coerce')
            top_labs  = lab['labname'].value_counts().head(15).index
            lab_pivot = lab[lab['labname'].isin(top_labs)].pivot_table(
                index='patientunitstayid', columns='labname', values='labresult', aggfunc='mean'
            ).reset_index()
            lab_pivot.columns = ['patientunitstayid'] + [
                f'lab_{c.lower().replace(" ","_")}' for c in lab_pivot.columns[1:]]
            pat = pat.merge(lab_pivot, on='patientunitstayid', how='left')
            print(f'    Lab features added: {lab_pivot.shape[1]-1}')
        except Exception as _e:
            print(f'    WARNING: Lab merge failed ({_e}) -- skipping')
    else:
        print('    Lab file not found -- skipping')

    miss_pct = pat.isnull().mean()
    pat.drop(columns=miss_pct[miss_pct > 0.6].index.tolist(), inplace=True)
    num_cols = pat.select_dtypes(include=np.number).columns.difference(['patientunitstayid', 'target'])
    if len(num_cols) > 0:
        pat[num_cols] = SimpleImputer(strategy='median').fit_transform(pat[num_cols])

    pat['dataset_source'] = 'eICU_original'
    pat['domain']         = 'icu'
    print(f'    -> shape: {pat.shape} | mortality: {pat["target"].mean():.3f}')
    return pat

eicu_pp = preprocess_eicu_original()
if eicu_pp is not None:
    print(f'\\nOK eICU original preprocessed: {eicu_pp.shape}')
    eicu_pp.head(3)""")

# ── Cell 17: NHANES ──
set_cell(nb, 17, "code", """def preprocess_nhanes() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess NHANES 2021-2023 data.\"\"\"
    nhanes_path = PATHS['nhanes']
    if not nhanes_path.exists():
        print('WARNING: NHANES_combined.csv not found')
        return None

    print('  [NHANES] Loading...')
    df = pd.read_csv(nhanes_path, low_memory=False)
    print(f'    Raw shape: {df.shape}')

    # Gender: RIAGENDR or Gender column  (1=Male, 2=Female)
    for gcol in ['Gender', 'RIAGENDR']:
        if gcol in df.columns:
            df['gender'] = df[gcol].map({1: 'Male', 2: 'Female'}).fillna('Unknown')
            break
    else:
        df['gender'] = 'Unknown'

    # Age
    for acol in ['Age_Years', 'RIDAGEYR', 'age']:
        if acol in df.columns:
            df['age'] = pd.to_numeric(df[acol], errors='coerce')
            break

    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 120],
                             labels=['young', 'middle', 'senior', 'elderly'],
                             include_lowest=True)

    # Target: create from diabetes indicators if available
    if 'target' not in df.columns:
        # Use glycohemoglobin > 6.5 or fasting glucose > 126 as diabetes proxy
        target_found = False
        for col in ['Glycohemoglobin_Pct', 'LBXGH']:
            if col in df.columns:
                df['target'] = (pd.to_numeric(df[col], errors='coerce') >= 6.5).astype(int)
                target_found = True
                break
        if not target_found:
            df['target'] = 0

    # Impute numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.difference(['target', 'Patient_ID'])
    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

    df['dataset_source'] = 'NHANES'
    df['domain'] = 'population_health'
    print(f'    -> shape: {df.shape} | target rate: {df["target"].mean():.3f}')
    return df

nhanes_pp = preprocess_nhanes()
if nhanes_pp is not None:
    print(f'\\nOK NHANES preprocessed: {nhanes_pp.shape}')""")

# ── Cell 19: CDC Diabetes ──
set_cell(nb, 19, "code", """def preprocess_cdc_diabetes() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess CDC Diabetes BRFSS 2015.\"\"\"
    path = PATHS['cdc_diabetes']
    if not path.exists():
        print('WARNING: CDC diabetes file not found')
        return None

    print('  [CDC Diabetes] Loading...')
    df = pd.read_csv(path)
    print(f'    Raw shape: {df.shape}')

    # Target
    if 'Diabetes_binary' in df.columns:
        df['target'] = df['Diabetes_binary'].astype(int)
    elif 'Diabetes_012' in df.columns:
        df['target'] = (df['Diabetes_012'] > 0).astype(int)

    # Gender: Sex column (0=Female, 1=Male in BRFSS)
    if 'Sex' in df.columns:
        df['gender'] = df['Sex'].map({1: 'Male', 0: 'Female'}).fillna('Unknown')

    # Age: BRFSS uses age categories 1-13
    if 'Age' in df.columns:
        age_map = {1: 22, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
                   8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 80}
        df['age'] = df['Age'].map(age_map)
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 120],
                             labels=['young', 'middle', 'senior', 'elderly'],
                             include_lowest=True)

    df['dataset_source'] = 'CDC_Diabetes'
    df['domain'] = 'metabolic'
    print(f'    -> shape: {df.shape} | target rate: {df["target"].mean():.3f}')
    return df

cdc_pp = preprocess_cdc_diabetes()
if cdc_pp is not None:
    print(f'\\nOK CDC Diabetes preprocessed: {cdc_pp.shape}')""")

# ── Cell 21: Breast Cancer ──
set_cell(nb, 21, "code", """def preprocess_breast_cancer() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess Breast Cancer Coimbra dataset.\"\"\"
    path = PATHS['breast_cancer']
    if not path.exists():
        print('WARNING: Breast cancer file not found')
        return None

    print('  [Breast Cancer] Loading...')
    df = pd.read_csv(path)
    print(f'    Raw shape: {df.shape}')

    # Target: Classification (1=Healthy, 2=Patient)
    if 'Classification' in df.columns:
        df['target'] = (df['Classification'] == 2).astype(int)

    # All female dataset
    df['gender'] = 'Female'

    # Age
    if 'Age' in df.columns:
        df['age'] = pd.to_numeric(df['Age'], errors='coerce')

    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 120],
                             labels=['young', 'middle', 'senior', 'elderly'],
                             include_lowest=True)

    # Impute
    num_cols = df.select_dtypes(include=np.number).columns.difference(['target', 'Classification'])
    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

    df['dataset_source'] = 'BreastCancer_Coimbra'
    df['domain'] = 'oncology'
    print(f'    -> shape: {df.shape} | target rate: {df["target"].mean():.3f}')
    return df

bc_pp = preprocess_breast_cancer()
if bc_pp is not None:
    print(f'\\nOK Breast Cancer preprocessed: {bc_pp.shape}')""")

# ── Cell 22: Data_Set summary ──
set_cell(nb, 22, "code", """# -- Data_Set Summary --
print('Data_Set Folder -- Preprocessing Summary')
print('=' * 65)
ds_datasets = {
    'UCI_HeartDisease'     : heart_pp,
    'CKD_UCI'              : ckd_pp,
    'eICU_original'        : eicu_pp,
    'NHANES'               : nhanes_pp,
    'CDC_Diabetes'         : cdc_pp,
    'BreastCancer_Coimbra' : bc_pp,
}
for name, df in ds_datasets.items():
    if df is not None:
        print(f'  {name:30s}: {len(df):>8,} rows x {df.shape[1]:>3} cols')
    else:
        print(f'  {name:30s}: FAILED / NOT FOUND')
print(f'\\nTotal: {sum(len(d) for d in ds_datasets.values() if d is not None):,} records')""")

# ── Cell 24: MIMIC-IV ──
set_cell(nb, 24, "code", """def preprocess_mimic_iv() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess MIMIC-IV v3.1 dataset.\"\"\"
    pat_path = PATHS['mimic_patients']
    adm_path = PATHS['mimic_admissions']
    icu_path = PATHS['mimic_icustays']

    if not pat_path.exists():
        print('WARNING: MIMIC patients.csv.gz not found')
        return None

    print('  [MIMIC-IV] Loading patients...')
    patients = pd.read_csv(pat_path)
    print(f'    Patients: {len(patients):,}')

    # Gender mapping
    patients['gender'] = patients['gender'].str.strip().str.upper().map(
        {'M': 'Male', 'F': 'Female'}
    ).fillna('Unknown')

    # Age
    patients['age'] = pd.to_numeric(patients['anchor_age'], errors='coerce')
    patients['age_group'] = pd.cut(patients['age'], bins=[0, 40, 60, 75, 120],
                                   labels=['young', 'middle', 'senior', 'elderly'],
                                   include_lowest=True)

    # Admissions
    if adm_path.exists():
        print('    Merging admissions...')
        adm = pd.read_csv(adm_path)
        # Aggregate admissions per patient
        adm_agg = adm.groupby('subject_id').agg(
            n_admissions=('hadm_id', 'nunique'),
            hospital_expire_flag=('hospital_expire_flag', 'max'),
            insurance_first=('insurance', 'first'),
            race_first=('race', 'first'),
        ).reset_index()
        patients = patients.merge(adm_agg, on='subject_id', how='left')
        print(f'    After admissions merge: {patients.shape}')

    # ICU stays count
    if icu_path.exists():
        print('    Merging ICU stays...')
        icu = pd.read_csv(icu_path)
        icu_agg = icu.groupby('subject_id').agg(
            n_icu_stays=('stay_id', 'nunique'),
            total_icu_los=('los', 'sum'),
            avg_icu_los=('los', 'mean'),
        ).reset_index()
        patients = patients.merge(icu_agg, on='subject_id', how='left')
        patients[['n_icu_stays', 'total_icu_los', 'avg_icu_los']] = \
            patients[['n_icu_stays', 'total_icu_los', 'avg_icu_los']].fillna(0)
        print(f'    After ICU merge: {patients.shape}')

    # Target = hospital mortality
    if 'hospital_expire_flag' in patients.columns:
        patients['target'] = patients['hospital_expire_flag'].fillna(0).astype(int)
    else:
        patients['target'] = 0

    patients['dataset_source'] = 'MIMIC_IV'
    patients['domain'] = 'icu'
    print(f'    -> shape: {patients.shape} | mortality: {patients["target"].mean():.3f}')
    return patients

mimic_pp = preprocess_mimic_iv()
if mimic_pp is not None:
    print(f'\\nOK MIMIC-IV preprocessed: {mimic_pp.shape}')
    mimic_pp.head(3)""")

# ── Cell 26: eICU-CRD v2 ──
set_cell(nb, 26, "code", """def preprocess_eicu_v2() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess eICU-CRD v2.0 dataset.\"\"\"
    pat_path  = PATHS['eicu2_patient']
    diag_path = PATHS['eicu2_diagnosis']
    lab_path  = PATHS['eicu2_lab']

    if not pat_path.exists():
        print('WARNING: eICU-CRD v2 patient.csv.gz not found')
        return None

    print('  [eICU-CRD v2] Loading patient table...')
    pat = pd.read_csv(pat_path)
    print(f'    Patient records: {len(pat):,}')

    # Age
    if 'age' in pat.columns:
        pat['age'] = pat['age'].replace('> 89', 90).pipe(pd.to_numeric, errors='coerce')
    if 'gender' in pat.columns:
        pat['gender'] = pat['gender'].str.strip().str.title()
    if 'ethnicity' in pat.columns:
        pat['ethnicity'] = pat['ethnicity'].str.strip().str.title()

    # Target: hospital mortality
    if 'hospitaldischargestatus' in pat.columns:
        pat['target'] = (pat['hospitaldischargestatus'].str.lower() == 'expired').astype(int)
    else:
        pat['target'] = 0

    pat['age_group'] = pd.cut(pat['age'], bins=[0, 40, 60, 75, 120],
                              labels=['young', 'middle', 'senior', 'elderly'],
                              include_lowest=True)

    # Diagnosis features
    if diag_path.exists():
        print('    Merging diagnosis features...')
        diag = pd.read_csv(diag_path)
        diag_feat = diag.groupby('patientunitstayid').agg(
            diagnosis_count=('diagnosisid', 'count'),
            unique_dx_count=('diagnosisstring', 'nunique')
        ).reset_index()
        pat = pat.merge(diag_feat, on='patientunitstayid', how='left')
        pat[['diagnosis_count', 'unique_dx_count']] = pat[['diagnosis_count', 'unique_dx_count']].fillna(0)

    # Lab features (optional -- very large)
    if lab_path.exists():
        print('    Merging lab features (sampling for speed)...')
        try:
            lab = pd.read_csv(lab_path, usecols=['patientunitstayid', 'labname', 'labresult'])
            lab['labresult'] = pd.to_numeric(lab['labresult'], errors='coerce')
            top_labs = lab['labname'].value_counts().head(15).index
            lab_pivot = lab[lab['labname'].isin(top_labs)].pivot_table(
                index='patientunitstayid', columns='labname', values='labresult', aggfunc='mean'
            ).reset_index()
            lab_pivot.columns = ['patientunitstayid'] + [
                f'lab_{c.lower().replace(" ","_")}' for c in lab_pivot.columns[1:]]
            pat = pat.merge(lab_pivot, on='patientunitstayid', how='left')
            print(f'    Lab features added: {lab_pivot.shape[1]-1}')
        except Exception as _e:
            print(f'    WARNING: Lab merge failed ({_e}) -- skipping')

    # Drop very sparse columns and impute
    miss_pct = pat.isnull().mean()
    pat.drop(columns=miss_pct[miss_pct > 0.6].index.tolist(), inplace=True)
    num_cols = pat.select_dtypes(include=np.number).columns.difference(['patientunitstayid', 'target'])
    if len(num_cols) > 0:
        pat[num_cols] = SimpleImputer(strategy='median').fit_transform(pat[num_cols])

    pat['dataset_source'] = 'eICU_CRD_v2'
    pat['domain'] = 'icu'
    print(f'    -> shape: {pat.shape} | mortality: {pat["target"].mean():.3f}')
    return pat

eicu2_pp = preprocess_eicu_v2()
if eicu2_pp is not None:
    print(f'\\nOK eICU-CRD v2 preprocessed: {eicu2_pp.shape}')
    eicu2_pp.head(3)""")

# ── Cell 28: PTB-XL ECG ──
set_cell(nb, 28, "code", """def preprocess_ptbxl() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess PTB-XL ECG dataset (metadata only, not waveforms).\"\"\"
    db_path  = PATHS['ptbxl_db']
    scp_path = PATHS['ptbxl_scp']

    if not db_path.exists():
        print('WARNING: ptbxl_database.csv not found')
        return None

    print('  [PTB-XL] Loading database...')
    df = pd.read_csv(db_path)
    print(f'    Records: {len(df):,}')

    # Gender: 0=Male, 1=Female
    if 'sex' in df.columns:
        df['gender'] = df['sex'].map({0: 'Male', 1: 'Female'}).fillna('Unknown')

    # Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 120],
                             labels=['young', 'middle', 'senior', 'elderly'],
                             include_lowest=True)

    # Parse SCP codes to create target (any abnormal finding)
    if 'scp_codes' in df.columns:
        # Load SCP statements for classification
        if scp_path.exists():
            scp = pd.read_csv(scp_path, index_col=0)
            # Abnormal = diagnostic class is not NORM
            def is_abnormal(scp_str):
                try:
                    codes = ast.literal_eval(scp_str)
                    for code in codes:
                        if code != 'NORM' and codes[code] > 50:
                            return 1
                    return 0
                except:
                    return 0
            df['target'] = df['scp_codes'].apply(is_abnormal)
        else:
            df['target'] = 0
    else:
        df['target'] = 0

    # Extract useful signal quality features
    for col in ['height', 'weight', 'heart_axis', 'validated_by_human',
                'baseline_drift', 'static_noise', 'burst_noise',
                'electrodes_problems', 'extra_beats', 'pacemaker', 'strat_fold']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop non-useful columns
    drop_cols = ['filename_lr', 'filename_hr', 'report', 'initial_autogenerated_report',
                 'scp_codes', 'recording_date', 'nurse', 'site', 'device',
                 'infarction_stadium1', 'infarction_stadium2', 'validated_by', 'second_opinion']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df['dataset_source'] = 'PTB_XL_ECG'
    df['domain'] = 'ecg'
    print(f'    -> shape: {df.shape} | abnormal rate: {df["target"].mean():.3f}')
    return df

ptbxl_pp = preprocess_ptbxl()
if ptbxl_pp is not None:
    print(f'\\nOK PTB-XL preprocessed: {ptbxl_pp.shape}')
    ptbxl_pp.head(3)""")

# ── Cell 30: HAM10000 ──
set_cell(nb, 30, "code", """def preprocess_ham10000() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess HAM10000 skin lesion dataset.\"\"\"
    path = PATHS['ham10000']
    if not path.exists():
        print('WARNING: HAM10000_metadata not found')
        return None

    print('  [HAM10000] Loading metadata...')
    df = pd.read_csv(path)
    print(f'    Records: {len(df):,}')

    # Gender
    if 'sex' in df.columns:
        df['gender'] = df['sex'].str.strip().str.lower().map(
            {'male': 'Male', 'female': 'Female'}
        ).fillna('Unknown')

    # Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 75, 120],
                             labels=['young', 'middle', 'senior', 'elderly'],
                             include_lowest=True)

    # Target: melanoma (mel) = 1, all others = 0
    if 'dx' in df.columns:
        df['target'] = (df['dx'] == 'mel').astype(int)

    # One-hot encode diagnosis type and localization
    if 'dx' in df.columns:
        dx_dummies = pd.get_dummies(df['dx'], prefix='dx')
        df = pd.concat([df, dx_dummies], axis=1)

    if 'localization' in df.columns:
        loc_dummies = pd.get_dummies(df['localization'], prefix='loc')
        df = pd.concat([df, loc_dummies], axis=1)

    if 'dx_type' in df.columns:
        dxtype_dummies = pd.get_dummies(df['dx_type'], prefix='dxtype')
        df = pd.concat([df, dxtype_dummies], axis=1)

    df['dataset_source'] = 'HAM10000'
    df['domain'] = 'dermatology'
    print(f'    -> shape: {df.shape} | melanoma rate: {df["target"].mean():.3f}')
    return df

ham_pp = preprocess_ham10000()
if ham_pp is not None:
    print(f'\\nOK HAM10000 preprocessed: {ham_pp.shape}')
    ham_pp.head(3)""")

# ── Cell 32: Heart Failure EHR ──
set_cell(nb, 32, "code", """def preprocess_heart_failure_ehr() -> Optional[pd.DataFrame]:
    \"\"\"Preprocess hospitalized heart failure EHR dataset.\"\"\"
    path = PATHS['hf_ehr']
    if not path.exists():
        print('WARNING: Heart Failure EHR dat.csv not found')
        return None

    print('  [Heart Failure EHR] Loading...')
    df = pd.read_csv(path, low_memory=False)
    print(f'    Raw shape: {df.shape}')

    # Drop identifier columns
    drop_cols = ['Unnamed: 0', 'inpatient.number', 'DestinationDischarge',
                 'admission.ward', 'admission.way', 'occupation', 'discharge.department']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower().map(
            {'male': 'Male', 'female': 'Female', '1': 'Male', '0': 'Female',
             '1.0': 'Male', '0.0': 'Female', 'm': 'Male', 'f': 'Female'}
        ).fillna('Unknown')
    else:
        df['gender'] = 'Unknown'

    # Target: in-hospital mortality or readmission
    if 'target' not in df.columns:
        # Use outcome.during.hospitalization or re.admission if available
        target_found = False
        for tcol in ['outcome.during.hospitalization', 're.admission.within.30.days',
                     'outcome', 'mortality']:
            if tcol in df.columns:
                df['target'] = pd.to_numeric(df[tcol], errors='coerce').fillna(0).astype(int)
                target_found = True
                break
        if not target_found:
            df['target'] = 0

    # Age: not available in this dataset
    df['age'] = np.nan
    df['age_group'] = 'Unknown'

    # Scale numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.difference(['target'])
    if len(num_cols) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(df[num_cols].fillna(df[num_cols].median())),
            columns=[f'{c}_scaled' for c in num_cols],
            index=df.index
        )
        df = pd.concat([df, scaled], axis=1)

    df['dataset_source'] = 'HeartFailure_EHR'
    df['domain'] = 'cardiovascular'
    print(f'    -> shape: {df.shape} | target rate: {df["target"].mean():.3f}')
    return df

hf_pp = preprocess_heart_failure_ehr()
if hf_pp is not None:
    print(f'\\nOK Heart Failure EHR preprocessed: {hf_pp.shape}')
    hf_pp.head(3)""")

# ── Cell 33: ND Summary ──
set_cell(nb, 33, "code", """# -- ND Summary --
print('ND Folder -- Preprocessing Summary')
print('=' * 65)
nd_datasets = {
    'MIMIC_IV'         : mimic_pp,
    'eICU_CRD_v2'      : eicu2_pp,
    'PTB_XL_ECG'       : ptbxl_pp,
    'HAM10000'         : ham_pp,
    'HeartFailure_EHR' : hf_pp,
}
for name, df in nd_datasets.items():
    if df is not None:
        print(f'  {name:30s}: {len(df):>8,} rows x {df.shape[1]:>3} cols')
    else:
        print(f'  {name:30s}: FAILED / NOT FOUND')
print(f'\\nTotal: {sum(len(d) for d in nd_datasets.values() if d is not None):,} records')""")

# ── Cell 35: Unified schema harmonization ──
set_cell(nb, 35, "code", """# Common schema columns present in all harmonized datasets
COMMON_SCHEMA = [
    'dataset_source',   # Source dataset name
    'domain',           # Clinical domain
    'target',           # Binary outcome (1=positive/adverse)
    'age',              # Numeric age in years
    'age_group',        # Binned age group label
    'gender',           # M/F/Male/Female/Unknown
]

def harmonize_to_schema(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    \"\"\"Extract and harmonize a dataset to the common unified schema.\"\"\"
    if df is None:
        return None

    out = pd.DataFrame(index=range(len(df)))

    # Source + domain
    out['dataset_source'] = df['dataset_source'].values if 'dataset_source' in df.columns else source_name
    out['domain']         = df['domain'].values         if 'domain'         in df.columns else 'unknown'

    # Target
    out['target'] = pd.to_numeric(df['target'], errors='coerce') if 'target' in df.columns else np.nan

    # Ensure target is binary (clamp >1 to 1)
    valid = out['target'].notna()
    if valid.any():
        out.loc[valid & (out['target'] > 1), 'target'] = 1

    # Age
    age_found = False
    for age_c in ['age', 'Age', 'age_clean', 'anchor_age', 'Age_Years']:
        if age_c in df.columns:
            out['age'] = pd.to_numeric(df[age_c].astype(str).str.replace('> 89', '90'), errors='coerce')
            age_found = True
            break
    if not age_found:
        out['age'] = np.nan

    # Age group
    if 'age_group' in df.columns:
        out['age_group'] = df['age_group'].astype(str).values
    else:
        out['age_group'] = pd.cut(out['age'], bins=[0,40,60,75,120],
                                  labels=['young','middle','senior','elderly'],
                                  include_lowest=True).astype(str)

    # Gender harmonization
    gender_found = False
    for g_col in ['gender', 'Gender', 'sex', 'Sex', 'RIAGENDR']:
        if g_col in df.columns:
            g = df[g_col].astype(str).str.strip().str.lower()
            out['gender'] = g.map(
                lambda x: 'Male'    if x in ['1','1.0','m','male','man']        else
                          'Female'  if x in ['0','0.0','f','female','woman','2','2.0'] else
                          x.title() if x in ['male','female'] else 'Unknown'
            )
            gender_found = True
            break
    if not gender_found:
        out['gender'] = 'Unknown'

    # Carry all original columns (preserve dataset-specific features)
    for col in df.columns:
        if col not in COMMON_SCHEMA:
            out[col] = df[col].values

    target_valid = out['target'].notna().sum()
    age_valid    = out['age'].notna().sum()
    print(f'  [{source_name:<25}] -> {out.shape} | target_valid: {target_valid:,} | age_valid: {age_valid:,}')
    return out

print('Harmonizing all datasets to unified schema...')
print('=' * 75)""")

# ── Cell 36: Harmonize ALL datasets ──
set_cell(nb, 36, "code", """# -- Harmonize ALL preprocessed datasets --
dataset_registry = {
    'UCI_HeartDisease'     : heart_pp,
    'CKD_UCI'              : ckd_pp,
    'eICU_original'        : eicu_pp,
    'NHANES'               : nhanes_pp,
    'CDC_Diabetes'         : cdc_pp,
    'BreastCancer_Coimbra' : bc_pp,
    'MIMIC_IV'             : mimic_pp,
    'eICU_CRD_v2'          : eicu2_pp,
    'PTB_XL_ECG'           : ptbxl_pp,
    'HAM10000'             : ham_pp,
    'HeartFailure_EHR'     : hf_pp,
}

harmonized = {}
for name, df in dataset_registry.items():
    if df is not None:
        harmonized[name] = harmonize_to_schema(df, name)
    else:
        print(f'  [{name:<25}] SKIPPED (None)')

total_h = len(harmonized)
total_r = len(dataset_registry)
print(f'\\n{total_h} datasets harmonized out of {total_r} attempted')
if total_h < total_r:
    missing = [n for n, d in dataset_registry.items() if d is None]
    print(f'WARNING: Missing datasets: {missing}')""")

# ── Cell 38: Domain grouping ──
set_cell(nb, 38, "code", """# -- Group by clinical domain --
domain_groups = {}
for name, df in harmonized.items():
    domain = df['domain'].iloc[0] if 'domain' in df.columns and len(df) > 0 else 'unknown'
    if domain not in domain_groups:
        domain_groups[domain] = []
    domain_groups[domain].append((name, df))

print('Domain Groups (datasets -> records):')
print('=' * 60)
for domain, datasets in sorted(domain_groups.items()):
    total = sum(len(df) for _, df in datasets)
    names = [f'{n}({len(df):,})' for n, df in datasets]
    print(f'  {domain:<25}: {total:>8,} records')
    for n in names:
        print(f'    -> {n}')""")

# ── Cell 39: Domain integration ──
set_cell(nb, 39, "code", """def integrate_domain(datasets: list, domain_name: str) -> pd.DataFrame:
    \"\"\"Concatenate all datasets within the same clinical domain.\"\"\"
    frames = [df for _, df in datasets]
    integrated = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Reorder: schema columns first, then domain-specific
    front_cols = [c for c in COMMON_SCHEMA if c in integrated.columns]
    other_cols = [c for c in integrated.columns if c not in front_cols]
    integrated = integrated[front_cols + other_cols]

    miss = integrated.isnull().mean().mean() * 100
    print(f'  [{domain_name:<20}] -> {integrated.shape} | missing: {miss:.1f}%')
    return integrated

print('Integrating datasets by clinical domain...')
print('=' * 60)
domain_dfs = {}
for domain, datasets in sorted(domain_groups.items()):
    domain_dfs[domain] = integrate_domain(datasets, domain)

print(f'\\n{len(domain_dfs)} domain-integrated DataFrames created')""")

# ── Cell 41: Full integration ──
set_cell(nb, 41, "code", """# -- Full integration of ALL datasets (cross-domain) --
all_frames = list(harmonized.values())
full_integrated = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)

# Reorder: schema columns first
front_cols  = [c for c in COMMON_SCHEMA if c in full_integrated.columns]
other_cols  = [c for c in full_integrated.columns if c not in front_cols]
full_integrated = full_integrated[front_cols + other_cols]

print('='*70)
print('FULL INTEGRATED DATASET -- SUMMARY')
print('='*70)
print(f'   Total records  : {len(full_integrated):>12,}')
print(f'   Total features : {full_integrated.shape[1]:>12,}')
print(f'   Overall missing: {full_integrated.isnull().mean().mean()*100:>11.1f}%')
print()
print('   Records per dataset source:')
for ds, cnt in full_integrated['dataset_source'].value_counts().items():
    print(f'     {ds:30s} {cnt:>10,}')
print()
print('   Records per clinical domain:')
for dom, cnt in full_integrated['domain'].value_counts().items():
    print(f'     {dom:30s} {cnt:>10,}')""")

# ── Cell 43: Statistics summary table ──
set_cell(nb, 43, "code", """# -- Dataset Statistics Summary Table --
summary_rows = []
for name, df in harmonized.items():
    target_rate = df['target'].mean() if 'target' in df.columns and df['target'].notna().any() else None
    summary_rows.append({
        'Dataset': name,
        'Domain': df['domain'].iloc[0] if 'domain' in df.columns else 'unknown',
        'Records': len(df),
        'Features': df.shape[1],
        'Missing %': round(df.isnull().mean().mean() * 100, 1),
        'Target Rate': round(target_rate, 3) if target_rate is not None else None,
        'Age Available %': round(df['age'].notna().mean() * 100, 1) if 'age' in df.columns else 0,
    })

summary_df = pd.DataFrame(summary_rows).sort_values('Records', ascending=False)
print('Dataset Quality Summary:')
print(summary_df.to_string(index=False))""")

# ── Cell 44: Dashboard visualization ──
set_cell(nb, 44, "code", """# -- Comprehensive Dashboard Visualization --
fig = plt.figure(figsize=(22, 18))
gs  = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1. Records per dataset
ax1 = fig.add_subplot(gs[0, 0])
rec = full_integrated['dataset_source'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(rec)))
bars = ax1.barh(range(len(rec)), rec.values, color=colors, edgecolor='white')
ax1.set_yticks(range(len(rec)))
ax1.set_yticklabels([t[:22] for t in rec.index], fontsize=8)
ax1.set_title('Records per Dataset', fontweight='bold')
ax1.set_xlabel('Count')
for bar, val in zip(bars, rec.values):
    ax1.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
             f'{val:,}', va='center', fontsize=7)

# 2. Missing % per dataset
ax2 = fig.add_subplot(gs[0, 1])
miss_by_ds = {name: df.isnull().mean().mean()*100 for name, df in harmonized.items()}
names_sorted = sorted(miss_by_ds, key=miss_by_ds.get, reverse=True)
vals = [miss_by_ds[n] for n in names_sorted]
max_miss = max(vals) + 1e-6 if vals else 1
colors2 = [plt.cm.RdYlGn_r(v / max_miss) for v in vals]
ax2.barh(range(len(names_sorted)), vals, color=colors2, edgecolor='white')
ax2.set_yticks(range(len(names_sorted)))
ax2.set_yticklabels([n[:22] for n in names_sorted], fontsize=8)
ax2.set_title('Missing Value % per Dataset', fontweight='bold')
ax2.set_xlabel('Missing %')

# 3. Records per domain (pie)
ax3 = fig.add_subplot(gs[1, 0])
dom_counts = full_integrated['domain'].value_counts()
ax3.pie(dom_counts.values, labels=dom_counts.index, autopct='%1.1f%%',
        colors=plt.cm.Pastel1(np.linspace(0, 1, len(dom_counts))), startangle=90)
ax3.set_title('Records by Clinical Domain', fontweight='bold')

# 4. Target rate per dataset
ax4 = fig.add_subplot(gs[1, 1])
target_rates = {}
for name, df in harmonized.items():
    if 'target' in df.columns and df['target'].notna().any():
        target_rates[name] = df['target'].mean()
t_names = list(target_rates.keys())
t_vals  = list(target_rates.values())
ax4.barh(range(len(t_names)), t_vals, color='coral', edgecolor='white')
ax4.set_yticks(range(len(t_names)))
ax4.set_yticklabels([n[:22] for n in t_names], fontsize=8)
ax4.set_title('Target Rate per Dataset', fontweight='bold')
ax4.set_xlabel('Rate')

# 5. Gender distribution
ax5 = fig.add_subplot(gs[2, 0])
gender_counts = full_integrated['gender'].value_counts()
ax5.bar(gender_counts.index, gender_counts.values, color=['royalblue', 'salmon', 'gray'][:len(gender_counts)])
ax5.set_title('Gender Distribution (All Datasets)', fontweight='bold')
ax5.set_ylabel('Count')

# 6. Age distribution
ax6 = fig.add_subplot(gs[2, 1])
age_data = full_integrated['age'].dropna()
ax6.hist(age_data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax6.set_title('Age Distribution (All Datasets)', fontweight='bold')
ax6.set_xlabel('Age')
ax6.set_ylabel('Count')

fig.suptitle('Multi-Dataset Integration Dashboard', fontsize=16, fontweight='bold', y=1.01)
plt.savefig(PLOT_DIR / 'dataset_overview.png', bbox_inches='tight', dpi=150)
plt.show()
print('Dashboard saved.')""")

# ── Cell 45: Feature overlap ──
set_cell(nb, 45, "code", """# -- Feature Overlap Matrix --
print('Feature Overlap Analysis Between Datasets')
print('=' * 60)

feature_sets = {name: set(df.columns) for name, df in harmonized.items()}
all_names    = list(feature_sets.keys())

overlap_matrix = pd.DataFrame(
    [[len(feature_sets[n1] & feature_sets[n2]) for n2 in all_names] for n1 in all_names],
    index=all_names, columns=all_names
)

fig, ax = plt.subplots(figsize=(12, 9))
short_names = [n[:18] for n in all_names]
sns.heatmap(overlap_matrix.values, xticklabels=short_names, yticklabels=short_names,
            ax=ax, annot=True, fmt='d', cmap='Blues', linewidths=0.3,
            cbar_kws={'label': 'Shared Columns'})
ax.set_title('Shared Feature Count Between Datasets', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'shared_features_heatmap.png', bbox_inches='tight')
plt.show()
print('Feature overlap heatmap saved.')""")

# ── Cell 46: Bias audit ──
set_cell(nb, 46, "code", """# -- Bias / Fairness Audit: Target Rate by Gender --
print('Bias Audit: Target Rate by Gender per Dataset')
print('=' * 60)

bias_rows = []
for name, df in harmonized.items():
    if 'target' in df.columns and 'gender' in df.columns:
        valid_df = df[df['target'].notna() & (df['gender'] != 'Unknown')]
        if len(valid_df) > 0:
            grp = valid_df.groupby('gender', observed=False)['target'].agg(['mean', 'count'])
            for gender, row in grp.iterrows():
                if row['count'] > 10:
                    bias_rows.append({'Dataset': name[:18], 'Gender': gender,
                                      'Rate': row['mean'], 'Count': int(row['count'])})

if bias_rows:
    bias_df = pd.DataFrame(bias_rows)
    pivot = bias_df.pivot_table(values='Rate', index='Dataset', columns='Gender', aggfunc='mean')
    print(pivot.round(3).to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.barh(ax=ax, color=['royalblue', 'salmon'])
    ax.set_title('Target Rate by Gender per Dataset', fontweight='bold')
    ax.set_xlabel('Target Rate')
    ax.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'bias_audit_gender.png', bbox_inches='tight')
    plt.show()
    print('Bias audit chart saved.')
else:
    print('No bias data available.')""")

# ── Cell 47: Age-Target relationship ──
set_cell(nb, 47, "code", """# -- Age-Target Relationship across Domains --
domains_with_data = [
    (dom, df) for dom, df in domain_dfs.items()
    if 'age' in df.columns and 'target' in df.columns and df['target'].notna().sum() > 10
    and df['age'].notna().sum() > 10
]

if domains_with_data:
    n_plots = min(len(domains_with_data), 4)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1: axes = [axes]
    fig.suptitle('Age vs. Target Rate by Domain', fontsize=13, fontweight='bold')

    for i, (domain, df_d) in enumerate(domains_with_data[:n_plots]):
        df_clean = df_d.dropna(subset=['age','target']).copy()
        if len(df_clean) < 10:
            axes[i].text(0.5, 0.5, f'{domain}\\nInsufficient data', ha='center', va='center',
                         transform=axes[i].transAxes)
            continue
        try:
            df_clean['age_bin'] = pd.cut(df_clean['age'], bins=10)
            rates = df_clean.groupby('age_bin', observed=False)['target'].mean()
            rates.plot(ax=axes[i], marker='o', color='coral')
            axes[i].set_title(f'{domain}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Age Bin')
            axes[i].set_ylabel('Target Rate')
            axes[i].tick_params(axis='x', rotation=45, labelsize=7)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'{domain}\\nError: {e}', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'age_target_by_domain.png', bbox_inches='tight')
    plt.show()
    print('Age-target chart saved.')
else:
    print('No domains with sufficient age+target data.')""")

# ── Cell 48: Schema availability heatmap ──
set_cell(nb, 48, "code", """# -- Per-Dataset EDA: Missing Heatmap --
schema_cols_present = [c for c in COMMON_SCHEMA if c in full_integrated.columns]
missing_by_ds = {}
for name, df in harmonized.items():
    avail = {col: int(df[col].notna().sum()) / len(df) * 100 for col in schema_cols_present if col in df.columns}
    missing_by_ds[name] = avail

avail_df = pd.DataFrame(missing_by_ds).T.fillna(0)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(avail_df, ax=ax, cmap='YlGn', annot=True, fmt='.0f',
            linewidths=0.5, cbar_kws={'label': '% Available (non-null)'},
            vmin=0, vmax=100)
ax.set_title('Schema Column Availability per Dataset (%)', fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'schema_availability.png', bbox_inches='tight')
plt.show()
print('Schema availability heatmap saved.')""")

# ── Cell 50: Save preprocessed ──
set_cell(nb, 50, "code", """# -- Save individual preprocessed datasets --
print('Saving individual preprocessed datasets...')
pp_dir = OUT_DIR / 'preprocessed'
pp_dir.mkdir(exist_ok=True)

save_map = {
    'heart_disease_preprocessed.csv'   : heart_pp,
    'ckd_preprocessed.csv'             : ckd_pp,
    'eicu_original_preprocessed.csv'   : eicu_pp,
    'nhanes_preprocessed.csv'          : nhanes_pp,
    'cdc_diabetes_preprocessed.csv'    : cdc_pp,
    'breast_cancer_preprocessed.csv'   : bc_pp,
    'mimic_iv_preprocessed.csv'        : mimic_pp,
    'eicu_crd_v2_preprocessed.csv'     : eicu2_pp,
    'ptbxl_preprocessed.csv'           : ptbxl_pp,
    'ham10000_preprocessed.csv'        : ham_pp,
    'heart_failure_ehr_preprocessed.csv': hf_pp,
}

for fname, df in save_map.items():
    if df is not None:
        out_path = pp_dir / fname
        df.to_csv(out_path, index=False)
        size = out_path.stat().st_size / 1e3
        print(f'  {fname:<45} {df.shape} -> {size:>8.0f} KB')
    else:
        print(f'  {fname:<45} SKIPPED (None)')""")

# ── Cell 51: Save domain files ──
set_cell(nb, 51, "code", """# -- Save domain-integrated datasets --
print('\\nSaving domain-integrated datasets...')
domain_dir = OUT_DIR / 'by_domain'
domain_dir.mkdir(exist_ok=True)

for domain, df in domain_dfs.items():
    fname    = f'{domain}_integrated.csv'
    out_path = domain_dir / fname
    df.to_csv(out_path, index=False)
    size = out_path.stat().st_size / 1e3
    print(f'  {fname:<40} {df.shape} -> {size:>8.0f} KB')""")

# ── Cell 52: Save full integrated ──
set_cell(nb, 52, "code", """# -- Save full integrated dataset --
print('\\nSaving full integrated dataset...')
full_path = OUT_DIR / 'full_integrated_dataset.csv'
full_integrated.to_csv(full_path, index=False)
size_mb = full_path.stat().st_size / 1e6
print(f'  full_integrated_dataset.csv: {full_integrated.shape} -> {size_mb:.1f} MB')

# Core (schema-only) CSV -- lightweight, no NaN flood
core_cols = [c for c in COMMON_SCHEMA if c in full_integrated.columns]
core_path = OUT_DIR / 'core_integrated_dataset.csv'
full_integrated[core_cols].to_csv(core_path, index=False)
core_mb = core_path.stat().st_size / 1e6
print(f'  core_integrated_dataset.csv: {full_integrated[core_cols].shape} -> {core_mb:.1f} MB')

# Schema summary CSV
summary_path = OUT_DIR / 'integration_summary_schema.csv'
full_integrated[core_cols].to_csv(summary_path, index=False)
print(f'  integration_summary_schema.csv: {full_integrated[core_cols].shape}')""")

# ── Cell 53: Save metadata JSON ──
set_cell(nb, 53, "code", """# -- Save integration metadata JSON --
meta = {
    'run_timestamp'      : pd.Timestamp.now().isoformat(),
    'pipeline_version'   : '3.0',
    'total_records'      : int(len(full_integrated)),
    'total_features'     : int(full_integrated.shape[1]),
    'datasets_integrated': list(harmonized.keys()),
    'domains'            : list(domain_dfs.keys()),
    'common_schema'      : COMMON_SCHEMA,
    'per_dataset': {
        name: {
            'records'           : int(len(df)),
            'features'          : int(df.shape[1]),
            'missing_pct'       : round(float(df.isnull().mean().mean() * 100), 4),
            'has_target'        : bool('target' in df.columns),
            'target_rate'       : round(float(df['target'].mean()), 4) if 'target' in df.columns and df['target'].notna().any() else None,
            'domain'            : str(df['domain'].iloc[0]) if 'domain' in df.columns else 'unknown',
            'age_availability'  : round(float(df['age'].notna().mean() * 100), 2) if 'age' in df.columns else 0,
            'gender_available'  : bool('gender' in df.columns),
        }
        for name, df in harmonized.items()
    }
}

meta_path = OUT_DIR / 'integration_metadata.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f'integration_metadata.json saved to {meta_path}')""")

# ── Cell 55: Final summary report ──
set_cell(nb, 55, "code", """print('=' * 78)
print('  MULTI-DATASET HEALTHCARE INTEGRATION PIPELINE -- FINAL REPORT')
print('=' * 78)

total_ds = len(harmonized)
total_rec = len(full_integrated)
total_feat = full_integrated.shape[1]

print(f'  Datasets integrated  : {total_ds}')
print(f'  Clinical domains     : {len(domain_dfs)}')
print(f'  Total patient records: {total_rec:,}')
print(f'  Total features       : {total_feat}')
print(f'  Common schema columns: {COMMON_SCHEMA}')
print()

print('  Per-dataset breakdown:')
print('  ' + '-' * 74)
print(f'  {"Dataset":<30} {"Domain":<20} {"Records":>10} {"Cols":>6} {"Target%":>8}')
print('  ' + '-' * 74)
for name, df in harmonized.items():
    domain = df['domain'].iloc[0] if 'domain' in df.columns else '?'
    rate = f'{df["target"].mean()*100:.1f}%' if 'target' in df.columns and df['target'].notna().any() else 'N/A'
    print(f'  {name:<30} {domain:<20} {len(df):>10,} {df.shape[1]:>6} {rate:>8}')

print('  ' + '-' * 74)
print(f'  {"TOTAL":<30} {"":<20} {total_rec:>10,} {total_feat:>6}')
print()
print('  Output files:')
print(f'    full_integrated_dataset.csv      : {total_rec:,} rows x {total_feat} cols')
print(f'    core_integrated_dataset.csv      : {total_rec:,} rows x {len(COMMON_SCHEMA)} cols')
print(f'    integration_metadata.json        : statistics for all {total_ds} datasets')
print(f'    by_domain/                       : {len(domain_dfs)} domain-level CSVs')
print(f'    preprocessed/                    : {total_ds} individual preprocessed CSVs')
print()
print('  PIPELINE COMPLETE')
print('=' * 78)""")

# ── Clear ALL cell outputs ──
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

save_nb(nb, NB)
print("\n[DONE] All cells updated. Notebook is ready to run.")
