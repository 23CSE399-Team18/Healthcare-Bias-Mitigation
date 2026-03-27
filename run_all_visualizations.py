r"""
=============================================================================
 VISUALIZATION SUITE — 5 PLOT TYPES x BEFORE & AFTER PREPROCESSING
 Datasets: UCI Heart Disease | Breast Cancer Coimbra | NHANES 2021-2023
=============================================================================
 HOW TO RUN:
   cd "c:\Users\U SRIYA\Documents\Data_Set"
   python run_all_visualizations.py

 WHAT IT DOES:
   For EACH of the 5 plot types, generates TWO separate images:
     one using RAW data (BEFORE preprocessing)
     one using PREPROCESSED data (AFTER preprocessing)

   Plot Types:
     1. Swarm Plot
     2. Violin Plot
     3. Interactive Plot  (Plotly — opens in browser)
     4. Strip Plot
     5. Pair Plot

   Output (saved to plots/ folder):
     1_swarm_BEFORE.png          1_swarm_AFTER.png
     2_violin_BEFORE.png         2_violin_AFTER.png
     3_interactive_BEFORE.html   3_interactive_AFTER.html
     4_strip_BEFORE.png          4_strip_AFTER.png
     5a_pair_heart_BEFORE.png    5a_pair_heart_AFTER.png
     5b_pair_cancer_BEFORE.png   5b_pair_cancer_AFTER.png
     5c_pair_nhanes_BEFORE.png   5c_pair_nhanes_AFTER.png
=============================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # explicit backend — avoids double-init issues
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Safe show — suppresses the known Tkinter/Python 3.12 TclError ────────
def safe_show():
    """Show the current figure, then close all figures to free memory.
    Silently ignores the known Tkinter 'can't delete Tcl command' bug."""
    try:
        safe_show()
    except Exception:
        pass
    try:
        plt.close("all")
    except Exception:
        pass


# ─── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PRE  = os.path.join(BASE, "preprocessed_output")
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
PAL_HEART  = {"Male": "#4C72B0", "Female": "#DD8452"}
PAL_ETH    = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]


# ═══════════════════════════════════════════════════════════════════════════
#  LOADERS — RAW DATA (BEFORE PREPROCESSING)
# ═══════════════════════════════════════════════════════════════════════════

def load_heart_raw():
    cols = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
    files = {
        "Cleveland":   os.path.join(BASE, "UCI_HeartDisease", "processed.cleveland.csv"),
        "Hungarian":   os.path.join(BASE, "UCI_HeartDisease", "processed.hungarian.csv"),
        "Switzerland": os.path.join(BASE, "UCI_HeartDisease", "processed.switzerland.csv"),
        "VA":          os.path.join(BASE, "UCI_HeartDisease", "processed.va.csv"),
    }
    frames = []
    for site, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, na_values=["?","-9.0","-9"])
            df = df.iloc[:, :14]; df.columns = cols
            df["site"] = site; frames.append(df)
    h = pd.concat(frames, ignore_index=True).apply(pd.to_numeric, errors="coerce")
    h["target"]    = (h["num"] > 0).astype(int)
    h["sex_label"] = h["sex"].map({0:"Female", 1:"Male"})
    h["diagnosis"] = h["target"].map({0:"No Disease", 1:"Heart Disease"})
    h["cp_label"]  = h["cp"].map({1:"Typical Angina",2:"Atypical Angina",
                                  3:"Non-Anginal",4:"Asymptomatic"})
    print(f"  [+] Heart Raw      -- {len(h)} patients")
    return h

def load_cancer_raw():
    df = pd.read_csv(os.path.join(BASE, "breast+cancer+coimbra_UCL", "dataR2.csv"))
    df["label"] = df["Classification"].map({1:"Healthy", 2:"Cancer"})
    print(f"  [+] Cancer Raw     -- {len(df)} patients")
    return df

def load_nhanes_raw():
    demo = os.path.join(BASE, "National Center for health Statistics", "nhanes_imputed.csv")
    df = pd.read_csv(demo)
    ren = {"SEQN":"Patient_ID","RIAGENDR":"Gender","RIDAGEYR":"Age_Years",
           "RIDRETH3":"Ethnicity_With_Asian","DMDEDUC2":"Education_Level",
           "DMDHHSIZ":"Household_Size","INDFMPIR":"Family_Income_Poverty_Ratio"}
    df.rename(columns={k:v for k,v in ren.items() if k in df.columns}, inplace=True)
    if "Gender" in df.columns:
        df["Gender_Label"] = df["Gender"].map({1.0:"Male", 2.0:"Female"})
    eth = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hisp White",
           4:"Non-Hisp Black",5:"Other/Multi",6:"Non-Hisp Asian",7:"Other/Multi"}
    if "Ethnicity_With_Asian" in df.columns:
        df["Ethnicity_Label"] = df["Ethnicity_With_Asian"].map(eth)
    if "Age_Years" in df.columns:
        df["Age_Group"] = pd.cut(df["Age_Years"], bins=[0,17,40,60,80,120],
                                  labels=["Child","Young Adult","Middle-Aged","Senior","Elderly"],
                                  include_lowest=True)
    bm = os.path.join(BASE, "National Center for health Statistics", "NHANES_BodyMeasures_2021_2023.csv")
    if os.path.exists(bm):
        bdf = pd.read_csv(bm).rename(columns={"SEQN":"Patient_ID"})
        bmi_c = next((c for c in bdf.columns if "BMI" in c.upper()), None)
        if bmi_c and "Patient_ID" in bdf.columns:
            df = df.merge(bdf[["Patient_ID", bmi_c]], on="Patient_ID", how="left")
            df.rename(columns={bmi_c:"BMI"}, inplace=True)
    ch = os.path.join(BASE, "National Center for health Statistics", "NHANES_TotalCholesterol_2021_2023.csv")
    if os.path.exists(ch):
        cdf = pd.read_csv(ch).rename(columns={"SEQN":"Patient_ID"})
        chol_c = next((c for c in cdf.columns if "LBXTC" in c or "Cholesterol" in c), None)
        if chol_c and "Patient_ID" in cdf.columns:
            df = df.merge(cdf[["Patient_ID", chol_c]], on="Patient_ID", how="left")
            df.rename(columns={chol_c:"Total_Cholesterol"}, inplace=True)
    print(f"  [+] NHANES Raw     -- {len(df)} participants, {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  LOADERS — PREPROCESSED DATA (AFTER PREPROCESSING)
# ═══════════════════════════════════════════════════════════════════════════

def load_heart_pre():
    df = pd.read_csv(os.path.join(PRE, "heart_disease_preprocessed.csv"))
    df["sex_label"] = df["sex"].map({0:"Female", 1:"Male"})
    df["target"]    = df["Heart_Disease"] if "Heart_Disease" in df.columns else (df["num"]>0).astype(int)
    df["diagnosis"] = df["target"].map({0:"No Disease", 1:"Heart Disease"})
    df["cp_label"]  = df["cp"].map({1:"Typical Angina",2:"Atypical Angina",
                                    3:"Non-Anginal",4:"Asymptomatic"})
    df["site"] = "Preprocessed"
    print(f"  [+] Heart Pre      -- {len(df)} patients")
    return df

def load_cancer_pre():
    df = pd.read_csv(os.path.join(PRE, "breast_cancer_preprocessed.csv"))
    if "Cancer_Risk" in df.columns:
        df["Classification"] = df["Cancer_Risk"].map({0:1, 1:2}).fillna(1).astype(int)
    if "Classification" not in df.columns:
        df["Classification"] = 1
    df["label"] = df["Classification"].map({1:"Healthy", 2:"Cancer"})
    print(f"  [+] Cancer Pre     -- {len(df)} patients")
    return df

def load_nhanes_pre():
    df = pd.read_csv(os.path.join(PRE, "NHANES_preprocessed.csv"))
    if "Gender" in df.columns:
        df["Gender_Label"] = df["Gender"].map({1:"Male", 2:"Female", 1.0:"Male", 2.0:"Female"})
    eth = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hisp White",
           4:"Non-Hisp Black",5:"Other/Multi",6:"Non-Hisp Asian",7:"Other/Multi"}
    if "Ethnicity_With_Asian" in df.columns:
        df["Ethnicity_Label"] = df["Ethnicity_With_Asian"].map(eth)
    if "Age_Years" in df.columns:
        df["Age_Group"] = pd.cut(df["Age_Years"], bins=[0,17,40,60,80,120],
                                  labels=["Child","Young Adult","Middle-Aged","Senior","Elderly"],
                                  include_lowest=True)
    print(f"  [+] NHANES Pre     -- {len(df)} participants, {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  1. SWARM PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_swarm(heart, cancer, nhanes, tag="BEFORE"):
    print(f"\n-- [1/5] Swarm Plot ({tag}) --")
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(f"SWARM PLOT  ({tag} PREPROCESSING)\n"
                 f"UCI Heart Disease  |  Breast Cancer Coimbra  |  NHANES",
                 fontsize=15, fontweight="bold", y=0.98)

    # Heart: cholesterol by sex and diagnosis
    ax = axes[0]
    hd = heart.dropna(subset=["chol","sex_label","diagnosis"])
    hd = hd[hd["chol"] < 500]
    sns.swarmplot(data=hd, x="diagnosis", y="chol", hue="sex_label",
                  palette=PAL_HEART, dodge=True, size=3.5, alpha=0.85, ax=ax)
    ax.set_title("Heart Disease\nCholesterol by Diagnosis & Sex")
    ax.set_xlabel("Diagnosis"); ax.set_ylabel("Serum Cholesterol (mg/dl)")
    ax.legend(title="Sex", loc="upper right")

    # Cancer: BMI and Age by classification
    ax = axes[1]
    avail = [c for c in ["BMI","Age"] if c in cancer.columns]
    if avail and "label" in cancer.columns:
        cm = cancer.melt(id_vars="label", value_vars=avail,
                         var_name="Measure", value_name="Value")
        sns.swarmplot(data=cm, x="Measure", y="Value", hue="label",
                      palette=["#2CA02C","#D62728"], dodge=True, size=4, alpha=0.85, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nBMI & Age by Classification")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value")
    ax.legend(title="Group")

    # NHANES: Age by Gender
    ax = axes[2]
    if "Gender_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        nh = nhanes[["Age_Years","Gender_Label"]].dropna()
        sample = nh.sample(min(600, len(nh)), random_state=42)
        sns.swarmplot(data=sample, x="Gender_Label", y="Age_Years",
                      palette=PAL_HEART, size=2.5, alpha=0.6, ax=ax)
    ax.set_title("NHANES 2021-2023\nAge Distribution by Gender")
    ax.set_xlabel("Gender"); ax.set_ylabel("Age (years)")

    plt.subplots_adjust(top=0.85, bottom=0.08, wspace=0.3)
    path = os.path.join(OUT, f"1_swarm_{tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"   Saved -> {path}")
    safe_show()


# ═══════════════════════════════════════════════════════════════════════════
#  2. VIOLIN PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_violin(heart, cancer, nhanes, tag="BEFORE"):
    print(f"\n-- [2/5] Violin Plot ({tag}) --")
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(f"VIOLIN PLOT  ({tag} PREPROCESSING)\n"
                 f"UCI Heart Disease  |  Breast Cancer Coimbra  |  NHANES",
                 fontsize=15, fontweight="bold", y=0.98)

    # Heart: max heart rate by sex x diagnosis
    ax = axes[0]
    hd = heart.dropna(subset=["thalach","sex_label","diagnosis"])
    sns.violinplot(data=hd, x="diagnosis", y="thalach", hue="sex_label",
                   palette=PAL_HEART, split=False, inner="quartile",
                   linewidth=1.2, ax=ax)
    ax.set_title("Heart Disease\nMax Heart Rate by Diagnosis & Sex")
    ax.set_xlabel("Diagnosis"); ax.set_ylabel("Max Heart Rate (bpm)")
    ax.legend(title="Sex")

    # Cancer: Glucose & Resistin by group
    ax = axes[1]
    avail = [c for c in ["Glucose","Resistin"] if c in cancer.columns]
    if avail and "label" in cancer.columns:
        cm = cancer.melt(id_vars="label", value_vars=avail,
                         var_name="Biomarker", value_name="Value")
        sns.violinplot(data=cm, x="Biomarker", y="Value", hue="label",
                       palette=["#2CA02C","#D62728"], split=False,
                       inner="box", linewidth=1.2, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nGlucose & Resistin by Group")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value")
    ax.legend(title="Group")

    # NHANES: Age by Ethnicity
    ax = axes[2]
    if "Ethnicity_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        top = nhanes["Ethnicity_Label"].value_counts().head(5).index.tolist()
        nh = nhanes[nhanes["Ethnicity_Label"].isin(top)].dropna(
            subset=["Age_Years","Ethnicity_Label"])
        sns.violinplot(data=nh, x="Ethnicity_Label", y="Age_Years",
                       palette=PAL_ETH, inner="quartile", linewidth=1.1, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_title("NHANES 2021-2023\nAge Distribution by Ethnicity")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Age (years)")

    plt.subplots_adjust(top=0.85, bottom=0.08, wspace=0.3)
    path = os.path.join(OUT, f"2_violin_{tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"   Saved -> {path}")
    safe_show()


# ═══════════════════════════════════════════════════════════════════════════
#  3. INTERACTIVE PLOT (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_interactive(heart, cancer, nhanes, tag="BEFORE"):
    print(f"\n-- [3/5] Interactive Plot ({tag}) --")

    # 3 panels — one per dataset, each with its own isolated legend group
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Heart Disease — Age vs Max Heart Rate",
            "Breast Cancer — BMI vs Glucose",
            "NHANES — Age by Ethnicity",
        ),
        horizontal_spacing=0.09,
    )

    # ── Panel 1: Heart Disease — Age vs Max HR coloured by Sex + Diagnosis ──
    HEART_COL = {
        "Female / Heart Disease": "#D62728",
        "Female / No Disease":    "#FF9896",
        "Male / Heart Disease":   "#1F77B4",
        "Male / No Disease":      "#AEC7E8",
    }
    hd = heart.dropna(subset=["age","thalach","sex_label","diagnosis"])
    first_h = True
    for sex, grp_s in hd.groupby("sex_label"):
        for dx, sub in grp_s.groupby("diagnosis"):
            key = f"{sex} / {dx}"
            fig.add_trace(
                go.Scatter(
                    x=sub["age"], y=sub["thalach"], mode="markers",
                    name=key,
                    legendgroup="heart",
                    legendgrouptitle_text="Heart Disease" if first_h else "",
                    marker=dict(size=6, opacity=0.70, color=HEART_COL.get(key, "#888")),
                    hovertemplate=f"<b>{key}</b><br>Age: %{{x}} yrs<br>Max HR: %{{y}} bpm<extra></extra>",
                ), row=1, col=1)
            first_h = False
    fig.update_xaxes(title_text="Age (years)", row=1, col=1)
    fig.update_yaxes(title_text="Max Heart Rate (bpm)", row=1, col=1)

    # ── Panel 2: Breast Cancer — BMI vs Glucose coloured by Healthy/Cancer ──
    CANCER_COL = {"Healthy": "#2CA02C", "Cancer": "#D62728"}
    if all(c in cancer.columns for c in ["BMI","Glucose","label"]):
        first_c = True
        for lbl, sub in cancer.groupby("label"):
            fig.add_trace(
                go.Scatter(
                    x=sub["BMI"], y=sub["Glucose"], mode="markers",
                    name=lbl,
                    legendgroup="cancer",
                    legendgrouptitle_text="Breast Cancer" if first_c else "",
                    marker=dict(size=9, opacity=0.80, color=CANCER_COL.get(lbl,"#888"),
                                line=dict(width=0.5, color="white")),
                    hovertemplate=f"<b>{lbl}</b><br>BMI: %{{x:.1f}}<br>Glucose: %{{y:.0f}} mg/dL<extra></extra>",
                ), row=1, col=2)
            first_c = False
    fig.update_xaxes(title_text="BMI (kg/m²)", row=1, col=2)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=2)

    # ── Panel 3: NHANES — Box plot of Age by Ethnicity (or Gender if no ethnicity) ──
    ETH_COL = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD"]
    if "Ethnicity_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        nh = nhanes.dropna(subset=["Age_Years","Ethnicity_Label"])
        top5 = nh["Ethnicity_Label"].value_counts().head(5).index.tolist()
        first_n = True
        for i, eth in enumerate(top5):
            sub = nh[nh["Ethnicity_Label"] == eth]
            fig.add_trace(
                go.Box(
                    y=sub["Age_Years"], name=eth, boxmean=True,
                    legendgroup="nhanes",
                    legendgrouptitle_text="NHANES" if first_n else "",
                    marker_color=ETH_COL[i % len(ETH_COL)],
                    hovertemplate=f"<b>{eth}</b><br>Age: %{{y:.0f}} yrs<extra></extra>",
                ), row=1, col=3)
            first_n = False
    elif "Gender_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        nh = nhanes[["Age_Years","Gender_Label"]].dropna()
        GEN_COL = {"Male":"#4C72B0","Female":"#DD8452"}
        first_n = True
        for g, sub in nh.groupby("Gender_Label"):
            fig.add_trace(
                go.Box(
                    y=sub["Age_Years"], name=g, boxmean=True,
                    legendgroup="nhanes",
                    legendgrouptitle_text="NHANES" if first_n else "",
                    marker_color=GEN_COL.get(g,"#888"),
                    hovertemplate=f"<b>{g}</b><br>Age: %{{y:.0f}} yrs<extra></extra>",
                ), row=1, col=3)
            first_n = False
    fig.update_yaxes(title_text="Age (years)", row=1, col=3)

    fig.update_layout(
        title=dict(
            text=(f"<b>Interactive Dashboard — {tag} PREPROCESSING</b><br>"
                  "<span style='font-size:13px'>"
                  "UCI Heart Disease &nbsp;|&nbsp; "
                  "Breast Cancer Coimbra &nbsp;|&nbsp; "
                  "NHANES 2021-2023</span>"),
            font=dict(size=17), x=0.5,
        ),
        height=560,
        template="plotly_white",
        legend=dict(
            groupclick="toggleitem",
            x=1.01, y=1,
            font=dict(size=11),
            tracegroupgap=20,
        ),
    )

    path = os.path.join(OUT, f"3_interactive_{tag}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"   Saved -> {path}")
    webbrowser.open(f"file:///{path}")
    print("   Opened in browser.")


# ═══════════════════════════════════════════════════════════════════════════
#  4. STRIP PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_strip(heart, cancer, nhanes, tag="BEFORE"):
    print(f"\n-- [4/5] Strip Plot ({tag}) --")
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(f"STRIP PLOT  ({tag} PREPROCESSING)\n"
                 f"UCI Heart Disease  |  Breast Cancer Coimbra  |  NHANES",
                 fontsize=15, fontweight="bold", y=0.98)

    # Heart: oldpeak by chest pain type x diagnosis
    ax = axes[0]
    hd = heart.dropna(subset=["oldpeak","cp_label","diagnosis"])
    sns.stripplot(data=hd, x="cp_label", y="oldpeak", hue="diagnosis",
                  palette={"No Disease":"#4C72B0","Heart Disease":"#D62728"},
                  dodge=True, jitter=0.18, size=4, alpha=0.75, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right", fontsize=9)
    ax.set_title("Heart Disease\nST Depression by Chest Pain Type & Diagnosis")
    ax.set_xlabel("Chest Pain Type"); ax.set_ylabel("ST Depression (oldpeak)")
    ax.legend(title="Diagnosis")

    # Cancer: Leptin & Adiponectin by group
    ax = axes[1]
    avail = [c for c in ["Leptin","Adiponectin"] if c in cancer.columns]
    if avail and "label" in cancer.columns:
        cm = cancer.melt(id_vars="label", value_vars=avail,
                         var_name="Biomarker", value_name="Value")
        sns.stripplot(data=cm, x="Biomarker", y="Value", hue="label",
                      palette=["#2CA02C","#D62728"], dodge=True,
                      jitter=0.2, size=5, alpha=0.8, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nLeptin & Adiponectin by Group")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value (ng/mL)")
    ax.legend(title="Group")

    # NHANES: Household size by ethnicity
    ax = axes[2]
    if "Ethnicity_Label" in nhanes.columns and "Household_Size" in nhanes.columns:
        top = nhanes["Ethnicity_Label"].value_counts().head(5).index.tolist()
        nh = nhanes[nhanes["Ethnicity_Label"].isin(top)].dropna(
            subset=["Household_Size","Ethnicity_Label"])
        nh_s = nh.sample(min(800, len(nh)), random_state=42)
        sns.stripplot(data=nh_s, x="Ethnicity_Label", y="Household_Size",
                      palette=PAL_ETH, jitter=0.25, size=3, alpha=0.55, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_title("NHANES 2021-2023\nHousehold Size by Ethnicity")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Household Size")

    plt.subplots_adjust(top=0.85, bottom=0.08, wspace=0.3)
    path = os.path.join(OUT, f"4_strip_{tag}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"   Saved -> {path}")
    safe_show()


# ═══════════════════════════════════════════════════════════════════════════
#  5. PAIR PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pair(heart, cancer, nhanes, tag="BEFORE"):
    print(f"\n-- [5/5] Pair Plots ({tag}) --")

    # 5a -- Heart Disease
    print("   Drawing Heart Disease pair plot...")
    hd_cols = ["age","trestbps","chol","thalach","oldpeak"]
    hd_avail = [c for c in hd_cols if c in heart.columns]
    hd_pp = heart[hd_avail + ["diagnosis"]].dropna()
    if "chol" in hd_pp.columns:
        hd_pp = hd_pp[hd_pp["chol"] < 500]
    g = sns.pairplot(hd_pp, vars=hd_avail, hue="diagnosis",
                     palette={"No Disease":"#4C72B0","Heart Disease":"#D62728"},
                     plot_kws={"alpha":0.55,"s":25}, diag_kind="kde", corner=False)
    g.figure.suptitle(f"Pair Plot -- Heart Disease ({tag} PREPROCESSING)",
                       y=1.02, fontsize=14, fontweight="bold")
    path = os.path.join(OUT, f"5a_pair_heart_{tag}.png")
    g.savefig(path, dpi=110, bbox_inches="tight")
    print(f"   Saved -> {path}")
    safe_show()

    # 5b -- Breast Cancer
    print("   Drawing Breast Cancer pair plot...")
    c_cols = ["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin"]
    c_avail = [c for c in c_cols if c in cancer.columns]
    if c_avail and "label" in cancer.columns:
        c_pp = cancer[c_avail + ["label"]].dropna()
        g2 = sns.pairplot(c_pp, vars=c_avail, hue="label",
                          palette={"Healthy":"#2CA02C","Cancer":"#D62728"},
                          plot_kws={"alpha":0.65,"s":30}, diag_kind="kde", corner=False)
        g2.figure.suptitle(f"Pair Plot -- Breast Cancer ({tag} PREPROCESSING)",
                            y=1.02, fontsize=14, fontweight="bold")
        path = os.path.join(OUT, f"5b_pair_cancer_{tag}.png")
        g2.savefig(path, dpi=110, bbox_inches="tight")
        print(f"   Saved -> {path}")
        safe_show()

    # 5c -- NHANES
    print("   Drawing NHANES pair plot...")
    nh_cols = ["Age_Years","Household_Size","Family_Income_Poverty_Ratio"]
    if "BMI" in nhanes.columns:
        nh_cols.append("BMI")
    if "Total_Cholesterol" in nhanes.columns:
        nh_cols.append("Total_Cholesterol")
    nh_avail = [c for c in nh_cols if c in nhanes.columns]
    if nh_avail and "Gender_Label" in nhanes.columns:
        nh_pp = nhanes[nh_avail + ["Gender_Label"]].dropna()
        nh_pp = nh_pp.sample(min(1500, len(nh_pp)), random_state=42)
        g3 = sns.pairplot(nh_pp, vars=nh_avail, hue="Gender_Label",
                          palette=PAL_HEART, plot_kws={"alpha":0.35,"s":12},
                          diag_kind="kde", corner=False)
        g3.figure.suptitle(f"Pair Plot -- NHANES ({tag} PREPROCESSING)",
                            y=1.02, fontsize=14, fontweight="bold")
        path = os.path.join(OUT, f"5c_pair_nhanes_{tag}.png")
        g3.savefig(path, dpi=110, bbox_inches="tight")
        print(f"   Saved -> {path}")
        safe_show()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" VISUALIZATION SUITE")
    print(" 5 Plot Types x BEFORE & AFTER Preprocessing")
    print(" Datasets: Heart Disease | Breast Cancer | NHANES")
    print("=" * 70)

    # ── Load RAW data (BEFORE preprocessing) ──
    print("\n--- Loading RAW Datasets (BEFORE) ---")
    heart_raw   = load_heart_raw()
    cancer_raw  = load_cancer_raw()
    nhanes_raw  = load_nhanes_raw()

    # ── Load PREPROCESSED data (AFTER preprocessing) ──
    print("\n--- Loading PREPROCESSED Datasets (AFTER) ---")
    heart_pre   = load_heart_pre()
    cancer_pre  = load_cancer_pre()
    nhanes_pre  = load_nhanes_pre()

    # ════════════════════════════════════════════════════════════════════════
    #  BEFORE PREPROCESSING — all 5 plot types using RAW data
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" BEFORE PREPROCESSING (Raw Data) -- 5 Plots")
    print(" Close each plot window to proceed to the next one.")
    print("=" * 70)

    plot_swarm(heart_raw, cancer_raw, nhanes_raw, tag="BEFORE")
    plot_violin(heart_raw, cancer_raw, nhanes_raw, tag="BEFORE")
    plot_interactive(heart_raw, cancer_raw, nhanes_raw, tag="BEFORE")
    plot_strip(heart_raw, cancer_raw, nhanes_raw, tag="BEFORE")
    plot_pair(heart_raw, cancer_raw, nhanes_raw, tag="BEFORE")

    # ════════════════════════════════════════════════════════════════════════
    #  AFTER PREPROCESSING — all 5 plot types using PREPROCESSED data
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" AFTER PREPROCESSING (Preprocessed Data) -- 5 Plots")
    print(" Close each plot window to proceed to the next one.")
    print("=" * 70)

    plot_swarm(heart_pre, cancer_pre, nhanes_pre, tag="AFTER")
    plot_violin(heart_pre, cancer_pre, nhanes_pre, tag="AFTER")
    plot_interactive(heart_pre, cancer_pre, nhanes_pre, tag="AFTER")
    plot_strip(heart_pre, cancer_pre, nhanes_pre, tag="AFTER")
    plot_pair(heart_pre, cancer_pre, nhanes_pre, tag="AFTER")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" ALL DONE!")
    print(f" Plots saved to: {OUT}")
    print("=" * 70)
    print("\n BEFORE preprocessing:")
    print("   1_swarm_BEFORE.png")
    print("   2_violin_BEFORE.png")
    print("   3_interactive_BEFORE.html  <- opens in browser")
    print("   4_strip_BEFORE.png")
    print("   5a_pair_heart_BEFORE.png")
    print("   5b_pair_cancer_BEFORE.png")
    print("   5c_pair_nhanes_BEFORE.png")
    print("\n AFTER preprocessing:")
    print("   1_swarm_AFTER.png")
    print("   2_violin_AFTER.png")
    print("   3_interactive_AFTER.html   <- opens in browser")
    print("   4_strip_AFTER.png")
    print("   5a_pair_heart_AFTER.png")
    print("   5b_pair_cancer_AFTER.png")
    print("   5c_pair_nhanes_AFTER.png")
    print("=" * 70)
