# ==========================================
# NHANES CLEANING WITH MISSING VALUE HANDLING
# ==========================================

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("NHANES_DEMO_2021_2023.csv")

# -------------------------------
# STEP 2: REPLACE NHANES MISSING CODES WITH NaN
# (7/9/77/99 = Refused / Don't Know in categorical cols)
# -------------------------------
code_cols = [
    "RIAGENDR", "RIDRETH1", "RIDRETH3", "RIDEXMON",
    "DMQMILIZ", "DMDBORN4", "DMDEDUC2", "DMDMARTZ",
    "RIDEXPRG", "DMDHRGND", "DMDHRAGZ", "DMDHREDZ",
    "DMDHRMAZ", "DMDHSEDZ"
]
for col in code_cols:
    if col in df.columns:
        df[col] = df[col].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})

# -------------------------------
# STEP 3: RENAME TO HUMAN-READABLE COLUMNS
# -------------------------------
df = df.rename(columns={
    "SEQN":      "Patient_ID",
    "SDDSRVYR":  "Survey_Cycle",
    "RIDSTATR":  "Interview_Exam_Status",
    "RIAGENDR":  "Gender",
    "RIDAGEYR":  "Age_Years",
    "RIDAGEMN":  "Age_Months",
    "RIDRETH1":  "Ethnicity",
    "RIDRETH3":  "Ethnicity_With_Asian",
    "RIDEXMON":  "Exam_Month_Category",
    "RIDEXAGM":  "Exam_Age_Months",
    "DMQMILIZ":  "Military_Service",
    "DMDBORN4":  "Country_Of_Birth",
    "DMDYRUSR":  "Years_In_US",
    "DMDEDUC2":  "Education_Level",
    "DMDMARTZ":  "Marital_Status",
    "RIDEXPRG":  "Pregnancy_Status",
    "DMDHHSIZ":  "Household_Size",
    "DMDHRGND":  "HH_Ref_Person_Gender",
    "DMDHRAGZ":  "HH_Ref_Person_Age_Group",
    "DMDHREDZ":  "HH_Ref_Person_Education",
    "DMDHRMAZ":  "HH_Ref_Person_Marital_Status",
    "DMDHSEDZ":  "HH_Ref_Person_Spouse_Education",
    "WTINT2YR":  "Interview_Weight",
    "WTMEC2YR":  "Exam_Weight",
    "SDMVSTRA":  "Variance_Stratum",
    "SDMVPSU":   "Variance_PSU",
    "INDFMPIR":  "Family_Income_Poverty_Ratio"
})

# -------------------------------
# STEP 4: HANDLE MISSING VALUES (ALL COLUMNS)
# -------------------------------
# All columns are numeric (float64), so use mean imputation
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if num_cols:
    imputer = SimpleImputer(strategy="mean")
    df[num_cols] = imputer.fit_transform(df[num_cols])

# -------------------------------
# STEP 5: SAVE (all 27 columns preserved)
# -------------------------------
df.to_csv("nhanes_imputed.csv", index=False)

# -------------------------------
# STEP 6: CHECK
# -------------------------------
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nMissing values:\n", df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")