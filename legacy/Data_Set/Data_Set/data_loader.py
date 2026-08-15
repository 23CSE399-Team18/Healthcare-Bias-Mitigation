"""
=============================================================================
 Data Collection & Loading Module
=============================================================================
 Unified data loaders for all 4 clinical datasets used in the Multi-Stage
 Bias Mitigation Framework:
   1. UCI Heart Disease  (4 sites — Cleveland, Hungarian, Switzerland, VA)
   2. Chronic Kidney Disease (ARFF-format, 400 patients)
   3. eICU Collaborative Research Database (ICU patients)
   4. NHANES Demographics (National Health Survey)

 Each loader returns a clean pandas DataFrame with:
   - Standardised column names
   - Mapped sensitive attributes
   - Binarised target variable
   - Missing values handled (NaN)
=============================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union

from config import (
    DATASET_PATHS, HEART_COLUMNS, CKD_COLUMNS, CKD_NOMINAL_MAP,
    NHANES_GENDER_MAP, NHANES_RACE_MAP, EICU_PATIENT_COLS,
    SENSITIVE_ATTRIBUTES, MODEL_CONFIG
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# 1. UCI HEART DISEASE
# ═══════════════════════════════════════════════════════════════════════════

def load_heart_disease(
    sites: Optional[List[str]] = None,
    binarise_target: bool = True,
    add_site_column: bool = True,
) -> pd.DataFrame:
    """
    Load UCI Heart Disease data from one or more clinical sites.

    Parameters
    ----------
    sites : list of str, optional
        Subset of ['cleveland', 'hungarian', 'switzerland', 'va'].
        Defaults to all four sites.
    binarise_target : bool
        If True, converts target (num) from 0-4 to binary {0, 1}.
        0 = no disease, 1 = disease present (values 1-4).
    add_site_column : bool
        If True, adds a 'site' column indicating data origin.

    Returns
    -------
    pd.DataFrame
        Cleaned, merged heart disease dataset.
    """
    available_sites = ["cleveland", "hungarian", "switzerland", "va"]
    if sites is None:
        sites = available_sites
    else:
        for s in sites:
            if s not in available_sites:
                raise ValueError(f"Unknown site '{s}'. Choose from {available_sites}")

    paths = DATASET_PATHS["heart_disease"]
    frames = []

    for site in sites:
        key = f"processed.{site}" if site != "va" else "processed.va"
        # Use the processed files
        path_key = site
        filepath = paths.get(site, None)
        if filepath is None or not os.path.exists(filepath):
            # Try processed version
            for k, v in paths.items():
                if site in k and os.path.exists(v):
                    filepath = v
                    break

        if filepath is None or not os.path.exists(filepath):
            warnings.warn(f"Data file for site '{site}' not found. Skipping.")
            continue

        # Load CSV — no header in processed files
        df = pd.read_csv(filepath, header=None, na_values=["?", "-9.0", "-9"])
        
        # Assign column names (processed files have 14 columns)
        if df.shape[1] == 14:
            df.columns = HEART_COLUMNS
        elif df.shape[1] == 76:
            # Full dataset — extract the 14 key columns
            idx = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
            df = df.iloc[:, idx]
            df.columns = HEART_COLUMNS
        else:
            warnings.warn(
                f"Site '{site}' has {df.shape[1]} columns (expected 14 or 76). "
                f"Attempting to use first 14."
            )
            df = df.iloc[:, :14]
            df.columns = HEART_COLUMNS

        if add_site_column:
            df["site"] = site

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No heart disease data files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # ── Clean numeric columns ──
    for col in HEART_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # ── Binarise target ──
    if binarise_target:
        combined["target"] = (combined["num"] > 0).astype(int)
    else:
        combined["target"] = combined["num"].astype(float)

    # ── Map sensitive attribute labels ──
    combined["sex_label"] = combined["sex"].map({0.0: "Female", 1.0: "Male", 0: "Female", 1: "Male"})

    # ── Summary statistics ──
    print(f"[Heart Disease] Loaded {len(combined)} patients from {len(sites)} site(s)")
    print(f"  Sites: {sites}")
    print(f"  Target distribution: {combined['target'].value_counts().to_dict()}")
    print(f"  Sex distribution: {combined['sex_label'].value_counts().to_dict()}")
    print(f"  Missing values: {combined.isnull().sum().sum()} total across all columns")
    print()

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHRONIC KIDNEY DISEASE (CKD)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_arff_data(filepath: str) -> pd.DataFrame:
    """
    Custom parser for the CKD ARFF-like CSV file.
    Skips comment lines (%) and @-directives, reads @data section.
    """
    data_lines = []
    in_data = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if in_data and line and not line.startswith("%"):
                data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data found in {filepath}")

    # Parse CSV lines
    rows = []
    for line in data_lines:
        # Split by comma, strip whitespace and tabs
        values = [v.strip().strip("\t") for v in line.split(",")]
        rows.append(values)

    df = pd.DataFrame(rows)

    # Trim to expected number of columns
    if df.shape[1] >= len(CKD_COLUMNS):
        df = df.iloc[:, :len(CKD_COLUMNS)]
        df.columns = CKD_COLUMNS
    else:
        raise ValueError(
            f"Expected {len(CKD_COLUMNS)} columns, got {df.shape[1]}."
        )

    return df


def load_ckd(
    create_age_groups: bool = True,
    age_bins: Optional[List[int]] = None,
    age_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load Chronic Kidney Disease dataset from ARFF-like format.

    Parameters
    ----------
    create_age_groups : bool
        If True, creates 'age_group' protected attribute from 'age'.
    age_bins : list of int
        Bin edges for age groups. Default: [0, 40, 60, 120].
    age_labels : list of str
        Labels for age groups. Default: ['young', 'middle', 'senior'].

    Returns
    -------
    pd.DataFrame
        Cleaned CKD dataset with encoded nominal features.
    """
    filepath = DATASET_PATHS["ckd"]["data"]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CKD data file not found: {filepath}")

    df = _parse_arff_data(filepath)

    # ── Replace '?' with NaN ──
    df.replace("?", np.nan, inplace=True)

    # ── Convert numeric columns ──
    numeric_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Map nominal columns ──
    for col, mapping in CKD_NOMINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # Handle any unmapped values
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Map sg, al, su as numeric ──
    for col in ["sg", "al", "su"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Create binary target ──
    df["target"] = df["class"].astype(int) if "class" in df.columns else np.nan

    # ── Create age groups (protected attribute) ──
    if create_age_groups:
        sa_config = SENSITIVE_ATTRIBUTES["ckd"]
        bins = age_bins or sa_config["bins"]
        labels = age_labels or sa_config["bin_labels"]
        df["age_group"] = pd.cut(
            df["age"], bins=bins, labels=labels, include_lowest=True
        )

    # ── Summary ──
    print(f"[CKD] Loaded {len(df)} patients")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    if create_age_groups:
        print(f"  Age group distribution: {df['age_group'].value_counts().to_dict()}")
    print(f"  Missing values: {df.isnull().sum().sum()} total across all columns")
    print()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. eICU COLLABORATIVE RESEARCH DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def load_eicu(
    merge_diagnosis: bool = True,
    merge_lab: bool = True,
    lab_pivot: bool = True,
    lab_agg: str = "mean",
) -> pd.DataFrame:
    """
    Load eICU data by joining patient, diagnosis, and lab tables.

    Parameters
    ----------
    merge_diagnosis : bool
        If True, merges diagnosis data (creates diagnosis count features).
    merge_lab : bool
        If True, merges lab results (pivoted to one column per lab type).
    lab_pivot : bool
        If True, pivots lab results into columns (one per lab name).
    lab_agg : str
        Aggregation for lab values ('mean', 'median', 'last').

    Returns
    -------
    pd.DataFrame
        Merged eICU dataset with demographics and clinical features.
    """
    paths = DATASET_PATHS["eicu"]

    # ── Load patient table ──
    if not os.path.exists(paths["patient"]):
        raise FileNotFoundError(f"eICU patient file not found: {paths['patient']}")

    patient_df = pd.read_csv(paths["patient"])
    print(f"[eICU] Loaded {len(patient_df)} patient records")

    # ── Clean demographics ──
    # Age handling — eICU uses "> 89" for elderly patients
    patient_df["age_clean"] = patient_df["age"].replace("> 89", 90)
    patient_df["age_clean"] = pd.to_numeric(patient_df["age_clean"], errors="coerce")

    # Gender standardisation
    patient_df["gender"] = patient_df["gender"].str.strip().str.title()

    # Ethnicity standardisation
    patient_df["ethnicity"] = patient_df["ethnicity"].str.strip().str.title()

    # ── Create mortality target ──
    patient_df["mortality"] = (
        patient_df["hospitaldischargestatus"].str.lower() == "expired"
    ).astype(int)

    # ── Age groups ──
    patient_df["age_group"] = pd.cut(
        patient_df["age_clean"],
        bins=[0, 40, 60, 75, 120],
        labels=["young", "middle", "senior", "elderly"],
        include_lowest=True,
    )

    # ── Merge diagnosis data ──
    if merge_diagnosis and os.path.exists(paths["diagnosis"]):
        diag_df = pd.read_csv(paths["diagnosis"])
        print(f"[eICU] Loaded {len(diag_df)} diagnosis records")

        # Create diagnosis features
        diag_features = diag_df.groupby("patientunitstayid").agg(
            diagnosis_count=("diagnosisid", "count"),
            has_major_dx=("diagnosispriority", lambda x: int("Major" in x.values)),
            unique_dx_count=("diagnosisstring", "nunique"),
        ).reset_index()

        # Extract top-level diagnosis categories
        diag_df["dx_category"] = diag_df["diagnosisstring"].apply(
            lambda x: x.split("|")[0].strip() if isinstance(x, str) else "unknown"
        )

        # One-hot encode top diagnosis categories
        top_categories = diag_df["dx_category"].value_counts().head(10).index.tolist()
        for cat in top_categories:
            col_name = f"dx_{cat.lower().replace(' ', '_').replace('/', '_')[:20]}"
            cat_patients = diag_df[diag_df["dx_category"] == cat]["patientunitstayid"].unique()
            diag_features[col_name] = diag_features["patientunitstayid"].isin(cat_patients).astype(int)

        patient_df = patient_df.merge(diag_features, on="patientunitstayid", how="left")
        # Fill NaN for patients with no diagnosis records
        fill_cols = [c for c in diag_features.columns if c != "patientunitstayid"]
        patient_df[fill_cols] = patient_df[fill_cols].fillna(0)

    # ── Merge lab data ──
    if merge_lab and os.path.exists(paths["lab"]):
        print(f"[eICU] Loading lab data (this may take a moment)...")

        # Read in chunks for memory efficiency
        lab_chunks = []
        for chunk in pd.read_csv(paths["lab"], chunksize=50000):
            # Keep only key lab types
            lab_chunks.append(chunk[["patientunitstayid", "labname", "labresult"]])
        lab_df = pd.concat(lab_chunks, ignore_index=True)
        print(f"[eICU] Loaded {len(lab_df)} lab records")

        # Clean lab results
        lab_df["labresult"] = pd.to_numeric(lab_df["labresult"], errors="coerce")

        if lab_pivot:
            # Get top N most common lab tests
            top_labs = lab_df["labname"].value_counts().head(20).index.tolist()
            lab_subset = lab_df[lab_df["labname"].isin(top_labs)]

            # Pivot: one column per lab test with aggregated value
            lab_pivot_df = lab_subset.pivot_table(
                index="patientunitstayid",
                columns="labname",
                values="labresult",
                aggfunc=lab_agg,
            ).reset_index()

            # Clean column names
            lab_pivot_df.columns = [
                f"lab_{c.lower().replace(' ', '_')}" if c != "patientunitstayid" else c
                for c in lab_pivot_df.columns
            ]

            patient_df = patient_df.merge(lab_pivot_df, on="patientunitstayid", how="left")
        else:
            # Just add lab count
            lab_count = lab_df.groupby("patientunitstayid").size().reset_index(name="lab_count")
            patient_df = patient_df.merge(lab_count, on="patientunitstayid", how="left")
            patient_df["lab_count"] = patient_df["lab_count"].fillna(0)

    # ── Summary ──
    print(f"[eICU] Final dataset: {patient_df.shape[0]} patients, {patient_df.shape[1]} features")
    print(f"  Mortality rate: {patient_df['mortality'].mean():.3f}")
    print(f"  Gender distribution: {patient_df['gender'].value_counts().to_dict()}")
    print(f"  Top ethnicities: {patient_df['ethnicity'].value_counts().head(5).to_dict()}")
    print(f"  Missing values: {patient_df.isnull().sum().sum()} total")
    print()

    return patient_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. NHANES DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════

def load_nhanes(
    map_codes: bool = True,
    create_age_groups: bool = True,
) -> pd.DataFrame:
    """
    Load NHANES 2021-2023 demographic data.

    Parameters
    ----------
    map_codes : bool
        If True, maps numeric codes to human-readable labels.
    create_age_groups : bool
        If True, creates age group bins.

    Returns
    -------
    pd.DataFrame
        NHANES demographics with mapped labels.
    """
    filepath = DATASET_PATHS["nhanes"]["demographics"]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NHANES demographics not found: {filepath}")

    df = pd.read_csv(filepath)

    print(f"[NHANES] Loaded {len(df)} participant records")

    # ── Rename key columns for readability ──
    rename_map = {
        "SEQN": "participant_id",
        "RIAGENDR": "gender_code",
        "RIDAGEYR": "age",
        "RIDRETH1": "race_ethnicity_code",
        "RIDRETH3": "race_ethnicity_detailed_code",
        "DMDBORN4": "country_of_birth",
        "DMDEDUC2": "education_level",
        "DMDMARTZ": "marital_status",
        "DMDHHSIZ": "household_size",
        "INDFMPIR": "poverty_income_ratio",
        "RIDEXPRG": "pregnancy_status",
        "RIDSTATR": "interview_exam_status",
        "WTINT2YR": "interview_weight_2yr",
        "WTMEC2YR": "exam_weight_2yr",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── Map coded variables ──
    if map_codes:
        if "gender_code" in df.columns:
            df["gender"] = df["gender_code"].map(NHANES_GENDER_MAP)

        if "race_ethnicity_code" in df.columns:
            df["race_ethnicity"] = df["race_ethnicity_code"].map(NHANES_RACE_MAP)

        if "race_ethnicity_detailed_code" in df.columns:
            df["race_ethnicity_detailed"] = df["race_ethnicity_detailed_code"].map(NHANES_RACE_MAP)

    # ── Create age groups ──
    if create_age_groups and "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 17, 40, 60, 80, 120],
            labels=["child", "young_adult", "middle_aged", "senior", "elderly"],
            include_lowest=True,
        )

    # ── Summary ──
    if "gender" in df.columns:
        print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
    if "race_ethnicity" in df.columns:
        print(f"  Race/Ethnicity distribution:")
        for k, v in df["race_ethnicity"].value_counts().items():
            print(f"    {k}: {v} ({v/len(df)*100:.1f}%)")
    if "age_group" in df.columns:
        print(f"  Age group distribution: {df['age_group'].value_counts().to_dict()}")
    print(f"  Missing values: {df.isnull().sum().sum()} total")
    print()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. UNIFIED MULTI-DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(
    name: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Unified dataset loader — dispatches to dataset-specific loaders.

    Parameters
    ----------
    name : str
        One of 'heart_disease', 'ckd', 'eicu', 'nhanes'.
    **kwargs
        Passed to the dataset-specific loader.

    Returns
    -------
    pd.DataFrame
    """
    loaders = {
        "heart_disease": load_heart_disease,
        "heart": load_heart_disease,
        "ckd": load_ckd,
        "eicu": load_eicu,
        "nhanes": load_nhanes,
    }

    name_lower = name.lower().strip()
    if name_lower not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}")

    print(f"{'='*70}")
    print(f" Loading Dataset: {name.upper()}")
    print(f"{'='*70}")

    return loaders[name_lower](**kwargs)


def get_sensitive_info(dataset_name: str) -> Dict:
    """
    Return the sensitive attribute configuration for a dataset.
    """
    key = dataset_name.lower().strip()
    if key in ("heart", "heart_disease"):
        key = "heart_disease"

    if key not in SENSITIVE_ATTRIBUTES:
        raise ValueError(f"No sensitive attribute config for '{dataset_name}'")

    return SENSITIVE_ATTRIBUTES[key]


def get_feature_target_split(
    df: pd.DataFrame,
    dataset_name: str,
    drop_sensitive: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a loaded DataFrame into features (X), target (y), and
    sensitive attribute (s).

    Parameters
    ----------
    df : pd.DataFrame
        Output of one of the load_* functions.
    dataset_name : str
        Name of the dataset.
    drop_sensitive : bool
        If True, removes the sensitive attribute from X.

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : pd.Series   — binary target
    s : pd.Series   — sensitive attribute
    """
    sa_info = get_sensitive_info(dataset_name)
    sa_col = sa_info["primary"]
    target_col = "target" if "target" in df.columns else "mortality"

    # Define columns to drop
    drop_cols = [target_col, "num", "class"]  # raw target columns
    # Add metadata columns
    drop_cols += ["site", "sex_label", "age_group"]
    if dataset_name.lower() in ("eicu",):
        drop_cols += [
            "patientunitstayid", "patienthealthsystemstayid", "uniquepid",
            "hospitaladmittime24", "hospitaldischargetime24",
            "unitadmittime24", "unitdischargetime24",
            "hospitaldischargestatus", "unitdischargestatus",
            "hospitaldischargelocation", "unitdischargelocation",
            "hospitaladmitsource", "unitadmitsource",
            "apacheadmissiondx", "wardid", "hospitalid",
            "hospitaldischargeyear",
        ]
    if dataset_name.lower() in ("nhanes",):
        drop_cols += [
            "participant_id", "gender_code", "race_ethnicity_code",
            "race_ethnicity_detailed_code", "gender", "race_ethnicity",
            "race_ethnicity_detailed",
        ]

    if drop_sensitive and sa_col in df.columns:
        drop_cols.append(sa_col)

    # Build feature matrix
    existing_drops = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drops, errors="ignore")

    # Target
    y = df[target_col].copy() if target_col in df.columns else pd.Series(dtype=float)

    # Sensitive attribute
    s = df[sa_col].copy() if sa_col in df.columns else pd.Series(dtype=object)

    return X, y, s


# ═══════════════════════════════════════════════════════════════════════════
# 6. DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def data_quality_report(df: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
    """
    Generate a comprehensive data quality report for a loaded dataset.

    Returns a DataFrame with per-column statistics:
      dtype, non_null_count, null_count, null_pct, unique_count,
      mean/std (numeric), top_values (categorical)
    """
    report = []

    for col in df.columns:
        info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": df[col].notna().sum(),
            "null_count": df[col].isna().sum(),
            "null_pct": f"{df[col].isna().mean()*100:.1f}%",
            "unique": df[col].nunique(),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            info["mean"] = f"{df[col].mean():.2f}" if df[col].notna().any() else "N/A"
            info["std"] = f"{df[col].std():.2f}" if df[col].notna().any() else "N/A"
            info["min"] = df[col].min() if df[col].notna().any() else "N/A"
            info["max"] = df[col].max() if df[col].notna().any() else "N/A"
            info["top_values"] = ""
        else:
            info["mean"] = ""
            info["std"] = ""
            info["min"] = ""
            info["max"] = ""
            top = df[col].value_counts().head(3)
            info["top_values"] = "; ".join(
                [f"{k}({v})" for k, v in top.items()]
            )

        report.append(info)

    report_df = pd.DataFrame(report)

    if dataset_name:
        print(f"\n{'='*70}")
        print(f" Data Quality Report — {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f" Total rows: {len(df)}")
        print(f" Total columns: {df.shape[1]}")
        print(f" Total missing cells: {df.isna().sum().sum()} "
              f"({df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)")
        print()
        print(report_df.to_string(index=False))
        print()

    return report_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Demo / Quick Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" MULTI-STAGE BIAS MITIGATION FRAMEWORK — DATA COLLECTION MODULE")
    print("=" * 70)
    print()

    # ── 1. Load UCI Heart Disease ──
    try:
        heart_df = load_heart_disease()
        data_quality_report(heart_df, "Heart Disease")
        X_h, y_h, s_h = get_feature_target_split(heart_df, "heart_disease")
        print(f"  Feature shape: {X_h.shape}, Target shape: {y_h.shape}, Sensitive shape: {s_h.shape}")
        print()
    except Exception as e:
        print(f"  [ERROR] Heart Disease: {e}\n")

    # ── 2. Load CKD ──
    try:
        ckd_df = load_ckd()
        data_quality_report(ckd_df, "CKD")
        X_c, y_c, s_c = get_feature_target_split(ckd_df, "ckd")
        print(f"  Feature shape: {X_c.shape}, Target shape: {y_c.shape}, Sensitive shape: {s_c.shape}")
        print()
    except Exception as e:
        print(f"  [ERROR] CKD: {e}\n")

    # ── 3. Load eICU ──
    try:
        eicu_df = load_eicu(merge_lab=False)  # skip lab for quick test
        data_quality_report(eicu_df, "eICU")
        X_e, y_e, s_e = get_feature_target_split(eicu_df, "eicu")
        print(f"  Feature shape: {X_e.shape}, Target shape: {y_e.shape}, Sensitive shape: {s_e.shape}")
        print()
    except Exception as e:
        print(f"  [ERROR] eICU: {e}\n")

    # ── 4. Load NHANES ──
    try:
        nhanes_df = load_nhanes()
        data_quality_report(nhanes_df, "NHANES")
        print()
    except Exception as e:
        print(f"  [ERROR] NHANES: {e}\n")

    print("=" * 70)
    print(" DATA COLLECTION COMPLETE")
    print("=" * 70)
