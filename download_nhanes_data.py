# ================================================================
# DOWNLOAD & CONVERT NHANES 2021-2023 LAB + EXAM DATA
# ================================================================
# Downloads XPT files from CDC and converts to CSV
# Focused on datasets relevant to Healthcare AI Fairness analysis

import pandas as pd
import os
import urllib.request

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# NHANES 2021-2023 datasets to download (name, XPT URL, output CSV name)
DATASETS = {
    # --- EXAMINATION DATA ---
    "Blood Pressure": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.xpt",
        "NHANES_BloodPressure_2021_2023.csv"
    ),
    "Body Measures": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt",
        "NHANES_BodyMeasures_2021_2023.csv"
    ),

    # --- LABORATORY DATA ---
    "Glycohemoglobin (HbA1c)": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt",
        "NHANES_Glycohemoglobin_2021_2023.csv"
    ),
    "Total Cholesterol": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt",
        "NHANES_TotalCholesterol_2021_2023.csv"
    ),
    "HDL Cholesterol": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.xpt",
        "NHANES_HDL_Cholesterol_2021_2023.csv"
    ),
    "Complete Blood Count (CBC)": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt",
        "NHANES_CBC_2021_2023.csv"
    ),
    "Albumin & Creatinine - Urine": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ALB_CR_L.xpt",
        "NHANES_AlbuminCreatinine_2021_2023.csv"
    ),
    "Insulin": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INS_L.xpt",
        "NHANES_Insulin_2021_2023.csv"
    ),
    "Plasma Fasting Glucose": (
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.xpt",
        "NHANES_FastingGlucose_2021_2023.csv"
    ),
}

# ----------------------------
# Download and convert each
# ----------------------------
for name, (url, csv_name) in DATASETS.items():
    xpt_path = os.path.join(OUT_DIR, csv_name.replace(".csv", ".xpt"))
    csv_path = os.path.join(OUT_DIR, csv_name)

    # Download
    print(f"\n{'='*50}")
    print(f"Downloading: {name}")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, xpt_path)
        print(f"  Saved XPT: {xpt_path}")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        continue

    # Convert XPT → CSV
    try:
        df = pd.read_sas(xpt_path)
        df.to_csv(csv_path, index=False)
        print(f"  Converted to CSV: {csv_name}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Clean up XPT file
        os.remove(xpt_path)
    except Exception as e:
        print(f"  ERROR converting: {e}")

print(f"\n{'='*50}")
print("DONE! All datasets downloaded and converted.")
