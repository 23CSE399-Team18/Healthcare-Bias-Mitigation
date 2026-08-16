"""
Team-18 — Auto Dataset Downloader
Healthcare Bias Mitigation Research Project
Downloads all FREE-ACCESS datasets (no registration/DUA required)
Run: python download_all_datasets.py
"""

import os
import urllib.request
import zipfile
import subprocess
import sys

BASE = r"D:\Data_Set\Data_Set"

def make_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"[DIR] {path}")

def download_file(url, dest_path, label=""):
    if os.path.exists(dest_path):
        print(f"[SKIP] Already exists: {dest_path}")
        return
    print(f"[DOWNLOADING] {label} ...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [OK] Saved to: {dest_path}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        print(f"  Manual download: {url}")

def extract_zip(zip_path, extract_to):
    print(f"[EXTRACTING] {zip_path} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"  [OK] Extracted to: {extract_to}")
    except Exception as e:
        print(f"  [FAIL] {e}")

def wget_physionet(url, dest_dir):
    """Use wget to download PhysioNet files (handles recursive download)"""
    make_dir(dest_dir)
    cmd = f'wget -r -N -c -np --directory-prefix="{dest_dir}" {url}'
    print(f"[WGET] {url}")
    print(f"  Command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("  [OK] Download complete")
    except subprocess.CalledProcessError:
        print("  [FAIL] wget not found or download failed")
        print("  Install wget: choco install wget  OR  use manual download link")
        print(f"  Manual: {url}")

print("=" * 60)
print("TEAM-18 — DATASET DOWNLOADER")
print("Healthcare Bias Mitigation Research")
print("=" * 60)
print()

# ============================================================
# C1 — PTB-XL ECG Dataset (FREE — CC BY 4.0)
# Portal: PhysioNet / Nature Scientific Data
# Paper: Wagner et al., Scientific Data 2020
# ============================================================
print("=" * 60)
print("C1: PTB-XL ECG Dataset (Nature Scientific Data)")
print("    physionet.org/content/ptb-xl/1.0.3/")
print("=" * 60)
ecg_dir = os.path.join(BASE, "ECG", "PTB_XL")
make_dir(ecg_dir)
# Download just the metadata CSV first (small — instant)
download_file(
    "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
    os.path.join(ecg_dir, "ptbxl_database.csv"),
    "PTB-XL Metadata CSV"
)
download_file(
    "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv",
    os.path.join(ecg_dir, "scp_statements.csv"),
    "PTB-XL SCP Statements"
)
print("  [NOTE] Full ECG signals (2.2 GB) — use wget:")
print(f"  wget -r -N -c -np --directory-prefix={ecg_dir} https://physionet.org/files/ptb-xl/1.0.3/")
print()

# ============================================================
# C2 — MIT-BIH Arrhythmia Database (FREE — ODC-By)
# Portal: PhysioNet / MIT
# ============================================================
print("=" * 60)
print("C2: MIT-BIH Arrhythmia Database")
print("    physionet.org/content/mitdb/1.0.0/")
print("=" * 60)
mitbih_dir = os.path.join(BASE, "ECG", "MIT_BIH")
make_dir(mitbih_dir)
print("  [NOTE] Use wget to download full database:")
print(f"  wget -r -N -c -np --directory-prefix={mitbih_dir} https://physionet.org/files/mitdb/1.0.0/")
print()

# ============================================================
# C3 — Mendeley ECG Heartbeat CSV Dataset (FREE — CC BY 4.0)
# Portal: Mendeley Data (Elsevier)
# Paper: Kachuee et al., IEEE EMBC 2018
# ============================================================
print("=" * 60)
print("C3: Mendeley ECG Heartbeat Categorization (Elsevier)")
print("    data.mendeley.com/datasets/7dybx7wyfn/3")
print("=" * 60)
mendeley_ecg_dir = os.path.join(BASE, "ECG", "Mendeley_ECG")
make_dir(mendeley_ecg_dir)
# Mendeley direct file download links
download_file(
    "https://data.mendeley.com/public-files/datasets/7dybx7wyfn/files/8b327869-74b5-4c65-97c2-89fcce32a36a/file_downloaded",
    os.path.join(mendeley_ecg_dir, "mitbih_train.csv"),
    "Mendeley ECG Train CSV"
)
download_file(
    "https://data.mendeley.com/public-files/datasets/7dybx7wyfn/files/d07dde85-c012-41a9-98bd-f37b2aef4790/file_downloaded",
    os.path.join(mendeley_ecg_dir, "mitbih_test.csv"),
    "Mendeley ECG Test CSV"
)
print()

# ============================================================
# C4 — PhysioNet Sepsis Challenge 2019 (FREE — CC BY 4.0)
# Portal: PhysioNet
# Paper: Reyna et al., Critical Care Medicine 2020
# ============================================================
print("=" * 60)
print("C4: PhysioNet Sepsis Early Prediction 2019")
print("    physionet.org/content/challenge-2019/1.0.0/")
print("=" * 60)
sepsis_dir = os.path.join(BASE, "Sepsis_Challenge_2019")
make_dir(sepsis_dir)
print("  [NOTE] Use wget for full dataset:")
print(f"  wget -r -N -c -np --directory-prefix={sepsis_dir} https://physionet.org/files/challenge-2019/1.0.0/")
print()

# ============================================================
# B1 — HAM10000 Skin Lesion Metadata (FREE — Harvard Dataverse)
# Portal: Harvard Dataverse
# Paper: Tschandl et al., Nature Scientific Data 2018
# ============================================================
print("=" * 60)
print("B1: HAM10000 Skin Lesion Dataset (Harvard Dataverse)")
print("    dataverse.harvard.edu doi:10.7910/DVN/DBW86T")
print("=" * 60)
ham_dir = os.path.join(BASE, "HAM10000")
make_dir(ham_dir)
download_file(
    "https://dataverse.harvard.edu/api/access/datafile/3172585",
    os.path.join(ham_dir, "HAM10000_metadata.csv"),
    "HAM10000 Metadata CSV"
)
print("  [NOTE] Image downloads (11 GB) — manual from Harvard Dataverse:")
print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
print()

# ============================================================
# A12 — CDC BRFSS 2023 (FREE — Public Domain)
# Portal: CDC (Centers for Disease Control)
# ============================================================
print("=" * 60)
print("A12: CDC BRFSS 2023 Survey Data")
print("    cdc.gov/brfss/annual_data/annual_2023.html")
print("=" * 60)
brfss_dir = os.path.join(BASE, "BRFSS_2023")
make_dir(brfss_dir)
download_file(
    "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip",
    os.path.join(brfss_dir, "LLCP2023XPT.zip"),
    "BRFSS 2023 XPT Data"
)
if os.path.exists(os.path.join(brfss_dir, "LLCP2023XPT.zip")):
    extract_zip(os.path.join(brfss_dir, "LLCP2023XPT.zip"), brfss_dir)
    print("  [CONVERT] Run this to convert XPT to CSV:")
    print("  pip install pyreadstat")
    print("  import pyreadstat")
    print("  df, meta = pyreadstat.read_xport('LLCP2023.XPT')")
    print("  df.to_csv('brfss_2023.csv', index=False)")
print()

# ============================================================
# A13 — WHO Global Health Observatory (FREE — Open API)
# Portal: World Health Organization
# ============================================================
print("=" * 60)
print("A13: WHO Global Health Observatory Data")
print("    who.int/data/gho  |  API: ghoapi.azureedge.net")
print("=" * 60)
who_dir = os.path.join(BASE, "WHO_GHO")
make_dir(who_dir)
# Download key WHO indicators via API
who_indicators = {
    "WHOSIS_000001": "life_expectancy.csv",            # Life expectancy at birth
    "WHOSIS_000015": "healthy_life_expectancy.csv",    # Healthy life expectancy
    "DIABETES_PREV": "diabetes_prevalence.csv",        # Diabetes prevalence
    "NCD_HYP_PREVALENCE_A": "hypertension_prevalence.csv", # Hypertension
}
for code, fname in who_indicators.items():
    url = f"https://ghoapi.azureedge.net/api/{code}?$format=csv"
    download_file(url, os.path.join(who_dir, fname), f"WHO {fname}")
print()

# ============================================================
# A14 — World Bank Health Data (FREE)
# Portal: World Bank Open Data
# ============================================================
print("=" * 60)
print("A14: World Bank Health, Nutrition & Population Data")
print("    data.worldbank.org/topic/health")
print("=" * 60)
wb_dir = os.path.join(BASE, "WorldBank_Health")
make_dir(wb_dir)
download_file(
    "https://api.worldbank.org/v2/en/topic/8?downloadformat=csv",
    os.path.join(wb_dir, "worldbank_health.zip"),
    "World Bank Health Topic Data"
)
if os.path.exists(os.path.join(wb_dir, "worldbank_health.zip")):
    extract_zip(os.path.join(wb_dir, "worldbank_health.zip"), wb_dir)
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("DOWNLOAD COMPLETE — SUMMARY")
print("=" * 60)
print()
print("DOWNLOADED / ATTEMPTED (Free Access):")
print("  C1 PTB-XL ECG metadata     → ECG/PTB_XL/")
print("  C2 MIT-BIH ECG (use wget)  → ECG/MIT_BIH/")
print("  C3 Mendeley ECG CSV        → ECG/Mendeley_ECG/")
print("  C4 Sepsis 2019 (use wget)  → Sepsis_Challenge_2019/")
print("  B1 HAM10000 metadata       → HAM10000/")
print("  A12 BRFSS 2023             → BRFSS_2023/")
print("  A13 WHO GHO indicators     → WHO_GHO/")
print("  A14 World Bank Health      → WorldBank_Health/")
print()
print("NEED REGISTRATION (Free — Quick Sign Up):")
print("  B2  CheXpert Chest X-Rays  → aimi.stanford.edu/datasets/chexpert")
print("  B4  ISIC 2019 Skin Images  → isic-archive.com")
print("  B6  OASIS-2 Brain MRI      → oasis-brains.org")
print("  B9  DRIVE Retinal Images   → drive.grand-challenge.org")
print("  A10 SEER Cancer Data       → seer.cancer.gov/data")
print()
print("NEED DUA (PhysioNet Login — Instant/Few Days):")
print("  A7  MIMIC-IV Demo          → physionet.org/content/mimic-iv-demo/2.2/")
print("  B3  MIMIC-CXR              → physionet.org/content/mimic-cxr/2.0.0/")
print("  A9  Amsterdam UMCdb        → physionet.org/content/amsterdamumcdb/1.0.2/")
print()
print("WGET COMMANDS FOR PHYSIONET (Paste in terminal):")
ecg_dir_ptb = os.path.join(BASE, "ECG", "PTB_XL")
ecg_dir_mit = os.path.join(BASE, "ECG", "MIT_BIH")
sepsis_dir2 = os.path.join(BASE, "Sepsis_Challenge_2019")
print(f'  wget -r -N -c -np --directory-prefix="{ecg_dir_ptb}" https://physionet.org/files/ptb-xl/1.0.3/')
print(f'  wget -r -N -c -np --directory-prefix="{ecg_dir_mit}" https://physionet.org/files/mitdb/1.0.0/')
print(f'  wget -r -N -c -np --directory-prefix="{sepsis_dir2}" https://physionet.org/files/challenge-2019/1.0.0/')
print()
print("Done! Check your D:\\Data_Set\\Data_Set\\ folder.")
