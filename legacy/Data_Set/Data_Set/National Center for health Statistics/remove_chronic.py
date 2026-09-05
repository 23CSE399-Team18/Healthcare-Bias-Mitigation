# Remove Chronic_Disease column from all preprocessed CSVs
import pandas as pd
import os

od = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessed_output')
print(f'Dir: {od}')

for fname in os.listdir(od):
    if not fname.endswith('.csv'):
        continue
    path = os.path.join(od, fname)
    df = pd.read_csv(path)
    if 'Chronic_Disease' in df.columns:
        df = df.drop(columns=['Chronic_Disease'])
        df.to_csv(path, index=False)
        print(f'[REMOVED] {fname} -> now {df.shape[0]}x{df.shape[1]}')
    else:
        print(f'[OK]      {fname} -> no Chronic_Disease column ({df.shape[0]}x{df.shape[1]})')
print('Done!')
