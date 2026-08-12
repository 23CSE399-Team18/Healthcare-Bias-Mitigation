"""
=============================================================================
 Configuration — Clinically Risk-Aware Multi-Stage Bias Mitigation Framework
=============================================================================
Central configuration for dataset paths, sensitive attributes, clinical
severity weights, model hyper-parameters, and fairness thresholds.
"""

import os

# ─── Base Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATHS = {
    "heart_disease": {
        "cleveland": os.path.join(BASE_DIR, "UCI_HeartDisease", "processed.cleveland.csv"),
        "hungarian": os.path.join(BASE_DIR, "UCI_HeartDisease", "processed.hungarian.csv"),
        "switzerland": os.path.join(BASE_DIR, "UCI_HeartDisease", "processed.switzerland.csv"),
        "va": os.path.join(BASE_DIR, "UCI_HeartDisease", "processed.va.csv"),
        "full_cleveland": os.path.join(BASE_DIR, "UCI_HeartDisease", "cleveland.csv"),
        "full_hungarian": os.path.join(BASE_DIR, "UCI_HeartDisease", "hungarian.csv"),
        "full_switzerland": os.path.join(BASE_DIR, "UCI_HeartDisease", "switzerland.csv"),
        "full_va": os.path.join(BASE_DIR, "UCI_HeartDisease", "long-beach-va.csv"),
    },
    "ckd": {
        "data": os.path.join(BASE_DIR, "Chronic_Kidney_Disease", "Chronic_Kidney_Disease", "chronic_kidney_disease_full.csv"),
        "info": os.path.join(BASE_DIR, "Chronic_Kidney_Disease", "Chronic_Kidney_Disease", "chronic_kidney_disease.info.txt"),
    },
    "eicu": {
        "patient": os.path.join(BASE_DIR, "eLCU", "patient.csv"),
        "diagnosis": os.path.join(BASE_DIR, "eLCU", "diagnosis.csv"),
        "lab": os.path.join(BASE_DIR, "eLCU", "lab.csv"),
    },
    "nhanes": {
        "demographics": os.path.join(BASE_DIR, "National Center for health Statistics", "NHANES_DEMO_2021_2023.csv"),
    },
}

# ─── Column Definitions per Dataset ────────────────────────────────────────

# UCI Heart Disease — 14 processed attributes
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# CKD — 25 attributes (24 features + class)
CKD_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "class"
]

CKD_NOMINAL_MAP = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0, " yes": 1, "\tyes": 1, "\tno": 0},
    "cad": {"yes": 1, "no": 0, "\tno": 0},
    "appet": {"good": 0, "poor": 1},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
    "class": {"ckd": 1, "notckd": 0},
}

# NHANES demographic variable mappings
NHANES_GENDER_MAP = {1: "Male", 2: "Female"}
NHANES_RACE_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other/Multi-Racial",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-Racial",
}

# eICU key columns
EICU_PATIENT_COLS = [
    "patientunitstayid", "gender", "age", "ethnicity",
    "hospitaldischargestatus", "unittype", "apacheadmissiondx",
    "admissionweight", "admissionheight"
]

# ─── Sensitive Attribute Definitions ───────────────────────────────────────

SENSITIVE_ATTRIBUTES = {
    "heart_disease": {
        "primary": "sex",
        "privileged_value": 1,         # male
        "unprivileged_value": 0,       # female
        "label": {"0": "Female", "1": "Male"},
    },
    "ckd": {
        "primary": "age_group",        # derived: young/middle/senior
        "privileged_value": "middle",
        "unprivileged_value": "senior",
        "label": {"young": "≤40", "middle": "41-60", "senior": ">60"},
        "bins": [0, 40, 60, 120],
        "bin_labels": ["young", "middle", "senior"],
    },
    "eicu": {
        "primary": "gender",
        "secondary": "ethnicity",
        "privileged_value": "Male",
        "unprivileged_value": "Female",
        "label": {"Male": "Male", "Female": "Female"},
    },
    "nhanes": {
        "primary": "RIAGENDR",
        "secondary": "RIDRETH3",
        "privileged_value": 1,         # male
        "unprivileged_value": 2,       # female
    },
}

# ─── Clinical Risk-Aware Loss Weights ──────────────────────────────────────
# False Negative is more harmful in critical diagnoses → higher weight
CLINICAL_SEVERITY = {
    "heart_disease": {
        "fn_weight": 3.0,   # missing a heart disease case is 3× worse
        "fp_weight": 1.0,
        "severity_label": "Critical — Cardiovascular",
    },
    "ckd": {
        "fn_weight": 2.5,   # missing CKD leads to irreversible damage
        "fp_weight": 1.0,
        "severity_label": "High — Renal",
    },
    "eicu": {
        "fn_weight": 4.0,   # ICU mortality prediction — highest stakes
        "fp_weight": 1.0,
        "severity_label": "Critical — ICU Mortality",
    },
}

# ─── Fairness Thresholds ───────────────────────────────────────────────────
FAIRNESS_CONFIG = {
    "disparate_impact_range": (0.8, 1.25),    # 80% rule
    "max_eod": 0.10,                           # max equalized odds difference
    "max_spd": 0.10,                           # max statistical parity difference
    "drift_window_size": 100,                  # sliding window for monitoring
    "drift_alpha": 0.05,                       # significance level for drift test
}

# ─── Model Hyper-parameters ────────────────────────────────────────────────
MODEL_CONFIG = {
    "adversarial": {
        "hidden_dims": [128, 64],
        "adversary_dims": [32],
        "lambda_adv": 1.0,        # adversary loss weight
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 64,
    },
    "baseline": {
        "logistic_regression": {"max_iter": 1000, "C": 1.0},
        "xgboost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 500},
    },
    "gan": {
        "epochs": 300,
        "batch_size": 500,
        "generator_dim": (256, 256),
        "discriminator_dim": (256, 256),
    },
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
}
