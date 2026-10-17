import io, pandas as pd, numpy as np

ckd_path = r'd:/Data_Set/Data_Set/Chronic_Kidney_Disease/Chronic_Kidney_Disease/chronic_kidney_disease_full.csv'
raw_lines = open(ckd_path, encoding='utf-8', errors='replace').read().splitlines()

arff_cols, data_start = [], None
for idx, ln in enumerate(raw_lines):
    s = ln.strip().lower()
    if s.startswith('@attribute'):
        col = ln.strip().split()[1].strip("'\"")
        arff_cols.append(col)
    elif s == '@data':
        data_start = idx + 1
        break

data_text = '\n'.join(raw_lines[data_start:])
norm_map = {'wbcc': 'wc', 'rbcc': 'rc'}
final_cols = [norm_map.get(c, c) for c in arff_cols]

df = pd.read_csv(io.StringIO(data_text), header=None, names=final_cols,
                 na_values=['?'], on_bad_lines='skip')
print('Shape:', df.shape)
print('Cols:', df.columns.tolist())
print('Missing pct:', round(df.isnull().mean().mean(), 3))
print(df.head(3))
