# ================================================================
# RENAME ALL NHANES 2021-2023 CSV COLUMNS TO HUMAN-READABLE ENGLISH
# ================================================================

import pandas as pd

RENAME_MAPS = {
    "NHANES_BloodPressure_2021_2023.csv": {
        "SEQN": "Patient_ID", "BPAOARM": "BP_Arm_Used", "BPAOCSZ": "BP_Cuff_Size",
        "BPXOSY1": "Systolic_BP_Reading_1", "BPXODI1": "Diastolic_BP_Reading_1",
        "BPXOSY2": "Systolic_BP_Reading_2", "BPXODI2": "Diastolic_BP_Reading_2",
        "BPXOSY3": "Systolic_BP_Reading_3", "BPXODI3": "Diastolic_BP_Reading_3",
        "BPXOPLS1": "Pulse_Reading_1", "BPXOPLS2": "Pulse_Reading_2", "BPXOPLS3": "Pulse_Reading_3",
    },
    "NHANES_BodyMeasures_2021_2023.csv": {
        "SEQN": "Patient_ID", "BMDSTATS": "Body_Measures_Status",
        "BMXWT": "Weight_kg", "BMIWT": "Weight_Comment",
        "BMXRECUM": "Recumbent_Length_cm", "BMIRECUM": "Recumbent_Length_Comment",
        "BMXHEAD": "Head_Circumference_cm", "BMIHEAD": "Head_Circumference_Comment",
        "BMXHT": "Height_cm", "BMIHT": "Height_Comment",
        "BMXBMI": "BMI", "BMDBMIC": "BMI_Category_Children",
        "BMXLEG": "Upper_Leg_Length_cm", "BMILEG": "Upper_Leg_Length_Comment",
        "BMXARML": "Upper_Arm_Length_cm", "BMIARML": "Upper_Arm_Length_Comment",
        "BMXARMC": "Arm_Circumference_cm", "BMIARMC": "Arm_Circumference_Comment",
        "BMXWAIST": "Waist_Circumference_cm", "BMIWAIST": "Waist_Circumference_Comment",
        "BMXHIP": "Hip_Circumference_cm", "BMIHIP": "Hip_Circumference_Comment",
    },
    "NHANES_Glycohemoglobin_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTPH2YR": "Phlebotomy_Weight", "LBXGH": "HbA1c_Percent",
    },
    "NHANES_TotalCholesterol_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTPH2YR": "Phlebotomy_Weight",
        "LBXTC": "Total_Cholesterol_mg_dL", "LBDTCSI": "Total_Cholesterol_mmol_L",
    },
    "NHANES_HDL_Cholesterol_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTPH2YR": "Phlebotomy_Weight",
        "LBDHDD": "HDL_Cholesterol_mg_dL", "LBDHDDSI": "HDL_Cholesterol_mmol_L",
    },
    "NHANES_CBC_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTPH2YR": "Phlebotomy_Weight",
        "LBXWBCSI": "White_Blood_Cell_Count", "LBXLYPCT": "Lymphocyte_Percent",
        "LBXMOPCT": "Monocyte_Percent", "LBXNEPCT": "Neutrophil_Percent",
        "LBXEOPCT": "Eosinophil_Percent", "LBXBAPCT": "Basophil_Percent",
        "LBDLYMNO": "Lymphocyte_Count", "LBDMONO": "Monocyte_Count",
        "LBDNENO": "Neutrophil_Count", "LBDEONO": "Eosinophil_Count",
        "LBDBANO": "Basophil_Count", "LBXRBCSI": "Red_Blood_Cell_Count",
        "LBXHGB": "Hemoglobin_g_dL", "LBXHCT": "Hematocrit_Percent",
        "LBXMCVSI": "Mean_Cell_Volume_fL", "LBXMC": "Mean_Cell_Hgb_Concentration",
        "LBXMCHSI": "Mean_Cell_Hemoglobin_pg", "LBXRDW": "Red_Cell_Distribution_Width",
        "LBXPLTSI": "Platelet_Count", "LBXMPSI": "Mean_Platelet_Volume_fL",
        "LBXNRBC": "Nucleated_Red_Blood_Cells",
    },
    "NHANES_AlbuminCreatinine_2021_2023.csv": {
        "SEQN": "Patient_ID", "URXUMA": "Urinary_Albumin_ug_mL",
        "URXUMS": "Urinary_Albumin_mg_L", "URDUMALC": "Urinary_Albumin_Comment",
        "URXUCR": "Urinary_Creatinine_mg_dL", "URXCRS": "Urinary_Creatinine_umol_L",
        "URDUCRLC": "Urinary_Creatinine_Comment", "URDACT": "Albumin_Creatinine_Ratio",
    },
    "NHANES_Insulin_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTSAF2YR": "Fasting_Sample_Weight",
        "LBXIN": "Insulin_uU_mL", "LBDINSI": "Insulin_pmol_L", "LBDINLC": "Insulin_Comment",
    },
    "NHANES_FastingGlucose_2021_2023.csv": {
        "SEQN": "Patient_ID", "WTSAF2YR": "Fasting_Sample_Weight",
        "LBXGLU": "Fasting_Glucose_mg_dL", "LBDGLUSI": "Fasting_Glucose_mmol_L",
    },
}

for filename, col_map in RENAME_MAPS.items():
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")

    df = pd.read_csv(filename)
    df = df.rename(columns=col_map)

    # Fill numeric NaNs with column mean
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].mean())

    df.to_csv(filename, index=False)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape:   {df.shape}")
    print(f"  Missing: {df.isnull().sum().sum()}")

print(f"\n{'='*50}")
print("DONE! All files renamed and cleaned.")
