"""One complete input file for the diabetes bias pipeline from UCI Diabetes 130-US Hospitals.

Keeps every encounter and every original column with its original values, except weight and
payer_code, which are removed completely. Changes:
'?' -> empty cell (standard missing); adds the text meaning of the three ID columns from
IDS_mapping.csv; adds age_mid (middle of the 10-year age band, an approximate age) and
age_group (<60, 60-79, 80+) after age; adds readmitted_30d (1 = readmitted within 30 days).
Nothing is cleaned, so all the biases stay in for the preprocessing stage.

Usage: python build_diabetes_input.py <folder with diabetic_data.csv and IDS_mapping.csv> <output.csv>
"""
import csv
import sys

import pandas as pd

DROP = ["weight", "payer_code"]

src, out = sys.argv[1], sys.argv[2]
raw = pd.read_csv(f"{src}/diabetic_data.csv", dtype=str, keep_default_na=False)

maps, cur = {}, None
with open(f"{src}/IDS_mapping.csv", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if not row or not any(c.strip() for c in row):
            cur = None
        elif row[1].strip() == "description":
            cur = row[0].strip()
            maps[cur] = {}
        else:
            maps[cur][row[0].strip()] = row[1].strip()

df = raw.replace("?", "").drop(columns=DROP)
for id_col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    unmapped = sorted(set(df[id_col]) - set(maps[id_col]))
    print(f"{id_col}: unmapped values in data: {unmapped or 'none'}")
    df.insert(df.columns.get_loc(id_col) + 1, id_col[:-3], df[id_col].map(maps[id_col]))

band = df["age"].str.extract(r"^\[(\d+)-(\d+)\)$").astype(int)   # fails loudly on an unexpected value
df.insert(df.columns.get_loc("age") + 1, "age_mid", (band[0] + band[1]) // 2)
df.insert(df.columns.get_loc("age_mid") + 1, "age_group",
          pd.cut(band[0], [-1, 59, 79, 200], labels=["<60", "60-79", "80+"]).astype(str))
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
df.to_csv(out, index=False)

# checks: same rows, dropped columns gone, kept original columns unchanged apart from '?' -> empty
chk = pd.read_csv(out, dtype=str, keep_default_na=False)
assert len(chk) == len(raw) == 101766, len(chk)
assert not set(DROP) & set(chk.columns), "dropped column still present"
kept = raw.columns.drop(DROP)
for c in kept:
    assert (raw[c].replace("?", "") == chk[c]).all(), c
print(f"\nwrote {out}: {len(chk)} rows x {chk.shape[1]} columns, {chk.patient_nbr.nunique()} unique patients")
print(f"removed: {DROP}; the other {len(kept)} original columns are identical to the UCI file except '?' -> empty")
print("\nage -> age_mid -> age_group (visits):")
print(chk.groupby(["age", "age_mid", "age_group"]).size().to_string())
first = chk.drop_duplicates("patient_nbr")
print("\nage_group, patients:", first.age_group.value_counts().reindex(["<60", "60-79", "80+"]).to_dict())
print("age_group, visits:  ", chk.age_group.value_counts().reindex(["<60", "60-79", "80+"]).to_dict())
print("readmitted_30d:", chk.readmitted_30d.value_counts().to_dict())
print("columns:", list(chk.columns))
