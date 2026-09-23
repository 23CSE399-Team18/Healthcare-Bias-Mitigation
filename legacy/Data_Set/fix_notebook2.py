"""
fix_notebook2_v2.py  —  clean rewrite that avoids quote escaping issues.
"""
import json, pathlib

BASE = pathlib.Path(r"d:/Data_Set")
NB   = BASE / "multi_dataset_integration.ipynb"

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[SAVED] {path}")

def replace_cell_source(nb, fragment, new_lines, label):
    """
    Find first code cell containing `fragment` and replace its source
    with `new_lines` (a plain Python list of strings, one per line).
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if fragment in src:
            # nbformat stores source as a list of lines ending with \n
            # (last line has no \n)
            joined = "\n".join(new_lines)
            cell["source"] = [joined]
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"  [PATCHED] {label}")
            return True
    print(f"  [NOT FOUND] {label}")
    return False

# ─────────────────────────────────────────────────────────────────
nb = load_nb(NB)

# ── BUG 1: CKD — ARFF loader ────────────────────────────────────
ckd_new = [
    "CKD_COLUMNS = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',",
    "                'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane','class']",
    "",
    "CKD_NOMINAL_MAP = {",
    "    'rbc':   {'normal': 0, 'abnormal': 1},",
    "    'pc':    {'normal': 0, 'abnormal': 1},",
    "    'pcc':   {'present': 1, 'notpresent': 0},",
    "    'ba':    {'present': 1, 'notpresent': 0},",
    "    'htn':   {'yes': 1, 'no': 0},",
    r"    'dm':    {'yes': 1, 'no': 0, ' yes': 1, '\tyes': 1, '\tno': 0},",
    r"    'cad':   {'yes': 1, 'no': 0, '\tno': 0},",
    "    'appet': {'good': 0, 'poor': 1},",
    "    'pe':    {'yes': 1, 'no': 0},",
    "    'ane':   {'yes': 1, 'no': 0},",
    "    'class': {'ckd': 1, 'notckd': 0},",
    "}",
    "",
    "ckd_path = PATHS['ckd']",
    "if ckd_path.exists():",
    "    # ARFF-aware loader: file uses Weka ARFF format with % comments",
    "    # and @attribute headers; data begins after '@data'",
    "    import io",
    "    _raw = ckd_path.read_text(encoding='utf-8', errors='replace').splitlines()",
    "    _arff_cols, _data_start = [], None",
    "    for _i, _ln in enumerate(_raw):",
    "        _s = _ln.strip().lower()",
    "        if _s.startswith('@attribute'):",
    "            _parts = _ln.strip().split()",
    "            _col = _parts[1].strip(\"'\\\"\")",
    "            _arff_cols.append(_col)",
    "        elif _s == '@data':",
    "            _data_start = _i + 1",
    "            break",
    "    _norm = {'wbcc': 'wc', 'rbcc': 'rc'}",
    "    _final_cols = [_norm.get(c, c) for c in _arff_cols]",
    "    if _data_start is not None:",
    "        _txt = '\\n'.join(_raw[_data_start:])",
    "        ckd_raw = pd.read_csv(",
    "            io.StringIO(_txt), header=None, names=_final_cols,",
    "            na_values=['?'], on_bad_lines='skip'",
    "        )",
    "    else:",
    "        ckd_raw = pd.read_csv(",
    "            ckd_path, comment='%', header=None, names=CKD_COLUMNS,",
    "            na_values=['?'], on_bad_lines='skip'",
    "        )",
    "    ckd_raw.replace('?', np.nan, inplace=True)",
    "    print(f'CKD raw shape: {ckd_raw.shape}')",
    "    print(f'Columns: {ckd_raw.columns.tolist()}')",
    "    print(ckd_raw.dtypes)",
    "    ckd_raw.head(3)",
    "else:",
    "    ckd_raw = None",
    "    print(f'WARNING: CKD file not found at {ckd_path}')",
]

replace_cell_source(
    nb,
    "ckd_raw = pd.read_csv(ckd_path, na_values=['?'])",
    ckd_new,
    "CKD ARFF loader (v2)"
)

# ── BUG 2: eICU lab safe-guard ──────────────────────────────────
eicu_new = [
    "def preprocess_eicu_original() -> Optional[pd.DataFrame]:",
    "    \"\"\"Preprocess original eICU dataset from eLCU folder.\"\"\"",
    "    patient_path = PATHS['eicu_patient']",
    "    diag_path    = PATHS['eicu_diagnosis']",
    "    lab_path     = PATHS['eicu_lab']",
    "",
    "    if not patient_path.exists():",
    "        print('WARNING: eICU patient.csv not found -- skipping')",
    "        return None",
    "",
    "    print('  [eICU orig] Loading patient table...')",
    "    pat = pd.read_csv(patient_path)",
    "    print(f'    Patient records: {len(pat)} | columns: {pat.shape[1]}')",
    "",
    "    if 'age' in pat.columns:",
    "        pat['age'] = pat['age'].replace('> 89', 90).pipe(pd.to_numeric, errors='coerce')",
    "    if 'gender'    in pat.columns: pat['gender']    = pat['gender'].str.strip().str.title()",
    "    if 'ethnicity' in pat.columns: pat['ethnicity'] = pat['ethnicity'].str.strip().str.title()",
    "",
    "    if 'hospitaldischargestatus' in pat.columns:",
    "        pat['target'] = (pat['hospitaldischargestatus'].str.lower() == 'expired').astype(int)",
    "",
    "    if 'age' in pat.columns:",
    "        pat['age_group'] = pd.cut(pat['age'], bins=[0, 40, 60, 75, 120],",
    "                                  labels=['young', 'middle', 'senior', 'elderly'], include_lowest=True)",
    "",
    "    if diag_path.exists():",
    "        print('    Merging diagnosis features...')",
    "        diag = pd.read_csv(diag_path)",
    "        diag_feat = diag.groupby('patientunitstayid').agg(",
    "            diagnosis_count=('diagnosisid', 'count'),",
    "            unique_dx_count=('diagnosisstring', 'nunique')",
    "        ).reset_index()",
    "        if 'diagnosispriority' in diag.columns:",
    "            diag_feat['has_major_dx'] = diag.groupby('patientunitstayid')['diagnosispriority'] \\",
    "                .apply(lambda x: int('Major' in x.values)).values[:len(diag_feat)]",
    "        pat = pat.merge(diag_feat, on='patientunitstayid', how='left')",
    "        pat[['diagnosis_count', 'unique_dx_count']] = pat[['diagnosis_count', 'unique_dx_count']].fillna(0)",
    "",
    "    # lab.csv is optional and very large -- skip gracefully if absent",
    "    if lab_path.exists():",
    "        print('    Merging lab features...')",
    "        try:",
    "            lab = pd.read_csv(lab_path, usecols=['patientunitstayid', 'labname', 'labresult'])",
    "            lab['labresult'] = pd.to_numeric(lab['labresult'], errors='coerce')",
    "            top_labs  = lab['labname'].value_counts().head(15).index",
    "            lab_pivot = lab[lab['labname'].isin(top_labs)].pivot_table(",
    "                index='patientunitstayid', columns='labname', values='labresult', aggfunc='mean'",
    "            ).reset_index()",
    "            lab_pivot.columns = ['patientunitstayid'] + [",
    "                f'lab_{c.lower().replace(\" \",\"_\")}' for c in lab_pivot.columns[1:]]",
    "            pat = pat.merge(lab_pivot, on='patientunitstayid', how='left')",
    "            print(f'    Lab features added: {lab_pivot.shape[1]-1}')",
    "        except Exception as _e:",
    "            print(f'    WARNING: Lab merge failed ({_e}) -- skipping')",
    "    else:",
    "        print('    Lab file not found -- skipping')",
    "",
    "    miss_pct = pat.isnull().mean()",
    "    pat.drop(columns=miss_pct[miss_pct > 0.6].index.tolist(), inplace=True)",
    "    num_cols = pat.select_dtypes(include=np.number).columns.difference(['patientunitstayid', 'target'])",
    "    pat[num_cols] = SimpleImputer(strategy='median').fit_transform(pat[num_cols])",
    "",
    "    pat['dataset_source'] = 'eICU_original'",
    "    pat['domain']         = 'icu'",
    "    print(f'    -> shape: {pat.shape} | mortality: {pat[\"target\"].mean():.3f}')",
    "    return pat",
    "",
    "eicu_pp = preprocess_eicu_original()",
    "if eicu_pp is not None:",
    "    print(f'\\nOK eICU original preprocessed: {eicu_pp.shape}')",
    "    eicu_pp.head(3)",
]

replace_cell_source(
    nb,
    "lab = pd.read_csv(lab_path, usecols=['patientunitstayid', 'labname', 'labresult'])",
    eicu_new,
    "eICU lab safe-guard (v2)"
)

# ── Clear all code cell outputs ──────────────────────────────────
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

save_nb(nb, NB)
print("[DONE]")
