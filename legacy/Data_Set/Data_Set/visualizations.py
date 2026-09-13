"""
=============================================================================
 HIGH-LEVEL VISUALIZATION SUITE
 Datasets: UCI Heart Disease | Breast Cancer Coimbra | NHANES 2021-2023
=============================================================================
 Plots:
   1. Swarm Plot
   2. Violin Plot
   3. Interactive Plot  (Plotly)
   4. Strip Plot
   5. Pair Plot
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────
BASE = r"c:\Users\U SRIYA\Documents\Data_Set"
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

# ─── Global Style ─────────────────────────────────────────────────────────
PALETTE_HEART   = {"Male": "#4C72B0", "Female": "#DD8452"}
PALETTE_CANCER  = {1: "#2CA02C", 2: "#D62728"}   # 1=Healthy, 2=Cancer
PALETTE_NHANES  = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
})


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════════

def load_heart():
    """UCI Heart Disease — merge all 4 sites, binarise target, label sex."""
    cols = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
    files = {
        "Cleveland":  os.path.join(BASE, "UCI_HeartDisease", "processed.cleveland.csv"),
        "Hungarian":  os.path.join(BASE, "UCI_HeartDisease", "processed.hungarian.csv"),
        "Switzerland":os.path.join(BASE, "UCI_HeartDisease", "processed.switzerland.csv"),
        "VA":         os.path.join(BASE, "UCI_HeartDisease", "processed.va.csv"),
    }
    frames = []
    for site, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, na_values=["?","-9.0","-9"])
            if df.shape[1] >= 14:
                df = df.iloc[:, :14]
            df.columns = cols[:df.shape[1]]
            df["site"] = site
            frames.append(df)

    hd = pd.concat(frames, ignore_index=True)
    hd = hd.apply(pd.to_numeric, errors="coerce")
    hd["target"]    = (hd["num"] > 0).astype(int)
    hd["sex_label"] = hd["sex"].map({0: "Female", 1: "Male"})
    hd["diagnosis"] = hd["target"].map({0: "No Disease", 1: "Heart Disease"})
    hd["cp_label"]  = hd["cp"].map({1:"Typical Angina",2:"Atypical Angina",
                                     3:"Non-Anginal",4:"Asymptomatic"})
    print(f"[Heart]   {hd.shape[0]} patients loaded")
    return hd


def load_cancer():
    """Breast Cancer Coimbra — 116 patients, biomarkers."""
    path = os.path.join(BASE, "breast+cancer+coimbra_UCL", "dataR2.csv")
    df = pd.read_csv(path)
    df["label"] = df["Classification"].map({1: "Healthy Control", 2: "Breast Cancer"})
    print(f"[Cancer]  {df.shape[0]} patients loaded")
    return df


def load_nhanes():
    """NHANES 2021-2023 — demographics + key clinical measures."""
    # Start from the imputed demographics file
    demo_path = os.path.join(BASE, "National Center for health Statistics", "nhanes_imputed.csv")
    df = pd.read_csv(demo_path)

    # Rename to readable if raw codes still present
    rename = {
        "SEQN": "Patient_ID", "RIAGENDR": "Gender", "RIDAGEYR": "Age_Years",
        "RIDRETH1": "Ethnicity", "RIDRETH3": "Ethnicity_With_Asian",
        "DMDEDUC2": "Education_Level", "DMDMARTZ": "Marital_Status",
        "DMDHHSIZ": "Household_Size", "INDFMPIR": "Family_Income_Poverty_Ratio",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Merge BMI from body measures if available
    bm_path = os.path.join(BASE, "National Center for health Statistics",
                           "NHANES_BodyMeasures_2021_2023.csv")
    if os.path.exists(bm_path):
        bm = pd.read_csv(bm_path)
        id_col = "Patient_ID" if "Patient_ID" in bm.columns else "SEQN"
        bm = bm.rename(columns={"SEQN": "Patient_ID"})
        bmi_col = next((c for c in bm.columns if "BMI" in c.upper()), None)
        if bmi_col and "Patient_ID" in bm.columns:
            df = df.merge(bm[["Patient_ID", bmi_col]], on="Patient_ID", how="left")
            df.rename(columns={bmi_col: "BMI"}, inplace=True)

    # Merge cholesterol if available
    chol_path = os.path.join(BASE, "National Center for health Statistics",
                             "NHANES_TotalCholesterol_2021_2023.csv")
    if os.path.exists(chol_path):
        ch = pd.read_csv(chol_path)
        ch = ch.rename(columns={"SEQN": "Patient_ID"})
        chol_col = next((c for c in ch.columns if "LBXTC" in c or "Cholesterol" in c), None)
        if chol_col and "Patient_ID" in ch.columns:
            df = df.merge(ch[["Patient_ID", chol_col]], on="Patient_ID", how="left")
            df.rename(columns={chol_col: "Total_Cholesterol"}, inplace=True)

    # Map gender
    if "Gender" in df.columns:
        df["Gender_Label"] = df["Gender"].map({1.0:"Male", 2.0:"Female"})

    # Age group
    if "Age_Years" in df.columns:
        df["Age_Group"] = pd.cut(df["Age_Years"],
                                 bins=[0,17,40,60,80,120],
                                 labels=["Child","Young Adult","Middle-Aged","Senior","Elderly"],
                                 include_lowest=True)
    # Ethnicity map
    eth_map = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hisp White",
               4:"Non-Hisp Black",5:"Other/Multi",6:"Non-Hisp Asian",7:"Other/Multi"}
    if "Ethnicity_With_Asian" in df.columns:
        df["Ethnicity_Label"] = df["Ethnicity_With_Asian"].map(eth_map)

    print(f"[NHANES]  {df.shape[0]} participants loaded, {df.shape[1]} features")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. SWARM PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_swarm(heart, cancer, nhanes):
    print("\n[1/5] Drawing Swarm Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Swarm Plot  |  UCI Heart Disease · Breast Cancer Coimbra · NHANES",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Heart: cholesterol by sex and diagnosis
    ax = axes[0]
    hd_clean = heart.dropna(subset=["chol", "sex_label", "diagnosis"])
    hd_clean = hd_clean[hd_clean["chol"] < 500]   # remove outliers
    sns.swarmplot(data=hd_clean, x="diagnosis", y="chol",
                  hue="sex_label", palette=PALETTE_HEART,
                  dodge=True, size=3.5, alpha=0.85, ax=ax)
    ax.set_title("Heart Disease\nCholesterol by Diagnosis & Sex")
    ax.set_xlabel("Diagnosis"); ax.set_ylabel("Serum Cholesterol (mg/dl)")
    ax.legend(title="Sex", loc="upper right")

    # ── Cancer: BMI and Age by classification
    ax = axes[1]
    cancer_m = cancer.melt(id_vars="label", value_vars=["BMI","Age"],
                            var_name="Measure", value_name="Value")
    sns.swarmplot(data=cancer_m, x="Measure", y="Value",
                  hue="label", palette=["#2CA02C","#D62728"],
                  dodge=True, size=4, alpha=0.85, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nBMI & Age by Classification")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value")
    ax.legend(title="Group")

    # ── NHANES: Age distribution by Gender
    ax = axes[2]
    nh_clean = nhanes.dropna(subset=["Age_Years","Gender_Label"]) if "Gender_Label" in nhanes else nhanes
    if "Gender_Label" in nh_clean.columns:
        sample = nh_clean[["Age_Years","Gender_Label"]].dropna().sample(
            min(600, len(nh_clean)), random_state=42)
        sns.swarmplot(data=sample, x="Gender_Label", y="Age_Years",
                      palette={"Male":"#4C72B0","Female":"#DD8452"},
                      size=2.5, alpha=0.6, ax=ax)
    ax.set_title("NHANES 2021–2023\nAge Distribution by Gender")
    ax.set_xlabel("Gender"); ax.set_ylabel("Age (years)")

    plt.tight_layout()
    path = os.path.join(OUT, "1_swarm_plot.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIOLIN PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_violin(heart, cancer, nhanes):
    print("\n[2/5] Drawing Violin Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Violin Plot  |  UCI Heart Disease · Breast Cancer Coimbra · NHANES",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Heart: max heart rate by sex × diagnosis
    ax = axes[0]
    hd_clean = heart.dropna(subset=["thalach","sex_label","diagnosis"])
    sns.violinplot(data=hd_clean, x="diagnosis", y="thalach",
                   hue="sex_label", palette=PALETTE_HEART,
                   split=False, inner="quartile", linewidth=1.2, ax=ax)
    ax.set_title("Heart Disease\nMax Heart Rate by Diagnosis & Sex")
    ax.set_xlabel("Diagnosis"); ax.set_ylabel("Max Heart Rate (bpm)")
    ax.legend(title="Sex")

    # ── Cancer: Glucose & Resistin by group
    ax = axes[1]
    c_m = cancer.melt(id_vars="label", value_vars=["Glucose","Resistin"],
                      var_name="Biomarker", value_name="Value")
    sns.violinplot(data=c_m, x="Biomarker", y="Value",
                   hue="label", palette=["#2CA02C","#D62728"],
                   split=False, inner="box", linewidth=1.2, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nGlucose & Resistin by Group")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value")
    ax.legend(title="Group")

    # ── NHANES: Age by Ethnicity group
    ax = axes[2]
    if "Ethnicity_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        top_eth = nhanes["Ethnicity_Label"].value_counts().head(5).index.tolist()
        nh_eth = nhanes[nhanes["Ethnicity_Label"].isin(top_eth)].dropna(
            subset=["Age_Years","Ethnicity_Label"])
        sns.violinplot(data=nh_eth, x="Ethnicity_Label", y="Age_Years",
                       palette=PALETTE_NHANES, inner="quartile",
                       linewidth=1.1, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_title("NHANES 2021–2023\nAge Distribution by Ethnicity")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Age (years)")

    plt.tight_layout()
    path = os.path.join(OUT, "2_violin_plot.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. INTERACTIVE PLOT (Plotly — saves as HTML)
# ═══════════════════════════════════════════════════════════════════════════

def plot_interactive(heart, cancer, nhanes):
    print("\n[3/5] Drawing Interactive Plots...")

    # Subplot grid: 2 rows × 2 cols
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Heart Disease — Age vs Max Heart Rate (by Sex & Diagnosis)",
            "Breast Cancer — BMI vs Glucose (by Group)",
            "NHANES — Age Distribution by Gender (Box)",
            "Heart Disease — Cholesterol by Site",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.18,
    )

    # ── Panel 1: Heart scatter
    hd_clean = heart.dropna(subset=["age","thalach","sex_label","diagnosis"])
    for sex, grp in hd_clean.groupby("sex_label"):
        for dx, sub in grp.groupby("diagnosis"):
            fig.add_trace(
                go.Scatter(
                    x=sub["age"], y=sub["thalach"],
                    mode="markers",
                    name=f"{sex} / {dx}",
                    marker=dict(size=6, opacity=0.65),
                    text="Age: "+sub["age"].astype(str)+" | HR: "+sub["thalach"].astype(str),
                    hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                ),
                row=1, col=1
            )
    fig.update_xaxes(title_text="Age (years)", row=1, col=1)
    fig.update_yaxes(title_text="Max Heart Rate (bpm)", row=1, col=1)

    # ── Panel 2: Cancer scatter
    fig.add_trace(
        go.Scatter(
            x=cancer["BMI"], y=cancer["Glucose"],
            mode="markers",
            marker=dict(
                size=9,
                color=cancer["Classification"],
                colorscale=[[0,"#2CA02C"],[1,"#D62728"]],
                showscale=False,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text="Label: "+cancer["label"]+"<br>HOMA: "+cancer["HOMA"].round(2).astype(str),
            hovertemplate="%{text}<extra></extra>",
            name="Cancer (Red) / Healthy (Green)",
            showlegend=True,
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="BMI", row=1, col=2)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=2)

    # ── Panel 3: NHANES box by gender
    if "Gender_Label" in nhanes.columns and "Age_Years" in nhanes.columns:
        nh_samp = nhanes[["Age_Years","Gender_Label"]].dropna().sample(
            min(3000, len(nhanes)), random_state=42)
        for g, grp in nh_samp.groupby("Gender_Label"):
            fig.add_trace(
                go.Box(y=grp["Age_Years"], name=g,
                       boxmean="sd", marker_color="#4C72B0" if g=="Male" else "#DD8452"),
                row=2, col=1
            )
    fig.update_yaxes(title_text="Age (years)", row=2, col=1)

    # ── Panel 4: Heart violin by site
    hd_site = heart.dropna(subset=["chol","site"])
    hd_site = hd_site[hd_site["chol"] < 500]
    for site, grp in hd_site.groupby("site"):
        fig.add_trace(
            go.Violin(y=grp["chol"], name=site, box_visible=True,
                      meanline_visible=True, opacity=0.7),
            row=2, col=2
        )
    fig.update_yaxes(title_text="Cholesterol (mg/dl)", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Interactive Dashboard — UCI Heart Disease · Breast Cancer · NHANES</b>",
            font=dict(size=16), x=0.5
        ),
        height=820,
        template="plotly_white",
        legend=dict(x=1.02, y=1, font=dict(size=10)),
    )

    path = os.path.join(OUT, "3_interactive_plot.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"   Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. STRIP PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_strip(heart, cancer, nhanes):
    print("\n[4/5] Drawing Strip Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Strip Plot  |  UCI Heart Disease · Breast Cancer Coimbra · NHANES",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Heart: oldpeak (ST depression) by chest pain type × diagnosis
    ax = axes[0]
    hd_clean = heart.dropna(subset=["oldpeak","cp_label","diagnosis"])
    sns.stripplot(data=hd_clean, x="cp_label", y="oldpeak",
                  hue="diagnosis", palette={"No Disease":"#4C72B0","Heart Disease":"#D62728"},
                  dodge=True, jitter=0.18, size=4, alpha=0.75, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right", fontsize=9)
    ax.set_title("Heart Disease\nST Depression by Chest Pain Type & Diagnosis")
    ax.set_xlabel("Chest Pain Type"); ax.set_ylabel("ST Depression (oldpeak)")
    ax.legend(title="Diagnosis")

    # ── Cancer: Leptin & Adiponectin by group
    ax = axes[1]
    c_m = cancer.melt(id_vars="label", value_vars=["Leptin","Adiponectin"],
                      var_name="Biomarker", value_name="Value")
    sns.stripplot(data=c_m, x="Biomarker", y="Value",
                  hue="label", palette=["#2CA02C","#D62728"],
                  dodge=True, jitter=0.2, size=5, alpha=0.8, ax=ax)
    ax.set_title("Breast Cancer Coimbra\nLeptin & Adiponectin by Group")
    ax.set_xlabel("Biomarker"); ax.set_ylabel("Value (ng/mL)")
    ax.legend(title="Group")

    # ── NHANES: Household size by ethnicity
    ax = axes[2]
    if "Ethnicity_Label" in nhanes.columns and "Household_Size" in nhanes.columns:
        top_eth = nhanes["Ethnicity_Label"].value_counts().head(5).index.tolist()
        nh_eth = nhanes[nhanes["Ethnicity_Label"].isin(top_eth)].dropna(
            subset=["Household_Size","Ethnicity_Label"])
        nh_samp = nh_eth.sample(min(800, len(nh_eth)), random_state=42)
        sns.stripplot(data=nh_samp, x="Ethnicity_Label", y="Household_Size",
                      palette=PALETTE_NHANES, jitter=0.25, size=3, alpha=0.55, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_title("NHANES 2021–2023\nHousehold Size by Ethnicity")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Household Size")

    plt.tight_layout()
    path = os.path.join(OUT, "4_strip_plot.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PAIR PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pair(heart, cancer, nhanes):
    print("\n[5/5] Drawing Pair Plots...")

    # ── 5a. Heart Disease pair plot
    hd_cols = ["age","trestbps","chol","thalach","oldpeak"]
    hd_pp = heart[hd_cols + ["sex_label","diagnosis"]].dropna()
    hd_pp = hd_pp[hd_pp["chol"] < 500]
    g = sns.pairplot(hd_pp, vars=hd_cols, hue="diagnosis",
                     palette={"No Disease":"#4C72B0","Heart Disease":"#D62728"},
                     plot_kws={"alpha":0.55,"s":25},
                     diag_kind="kde", corner=False)
    g.figure.suptitle("Pair Plot — UCI Heart Disease\n(colour = Diagnosis)",
                       y=1.02, fontsize=14, fontweight="bold")
    path_a = os.path.join(OUT, "5a_pairplot_heart.png")
    g.savefig(path_a, dpi=110, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved → {path_a}")

    # ── 5b. Breast Cancer pair plot
    c_cols = ["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin"]
    c_pp = cancer[c_cols + ["label"]].dropna()
    g2 = sns.pairplot(c_pp, vars=c_cols, hue="label",
                      palette={"Healthy Control":"#2CA02C","Breast Cancer":"#D62728"},
                      plot_kws={"alpha":0.65,"s":30},
                      diag_kind="kde", corner=False)
    g2.figure.suptitle("Pair Plot — Breast Cancer Coimbra\n(colour = Group)",
                        y=1.02, fontsize=14, fontweight="bold")
    path_b = os.path.join(OUT, "5b_pairplot_cancer.png")
    g2.savefig(path_b, dpi=110, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved → {path_b}")

    # ── 5c. NHANES pair plot (subset of numeric features)
    nh_numeric = ["Age_Years","Household_Size","Family_Income_Poverty_Ratio"]
    if "BMI" in nhanes.columns:
        nh_numeric.append("BMI")
    if "Total_Cholesterol" in nhanes.columns:
        nh_numeric.append("Total_Cholesterol")

    nh_pp = nhanes[nh_numeric + ["Gender_Label"]].dropna()
    nh_pp = nh_pp.sample(min(1500, len(nh_pp)), random_state=42)
    g3 = sns.pairplot(nh_pp, vars=nh_numeric, hue="Gender_Label",
                      palette={"Male":"#4C72B0","Female":"#DD8452"},
                      plot_kws={"alpha":0.35,"s":12},
                      diag_kind="kde", corner=False)
    g3.figure.suptitle("Pair Plot — NHANES 2021–2023\n(colour = Gender)",
                        y=1.02, fontsize=14, fontweight="bold")
    path_c = os.path.join(OUT, "5c_pairplot_nhanes.png")
    g3.savefig(path_c, dpi=110, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved → {path_c}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" HEALTHCARE VISUALIZATION SUITE — 5 Plot Types")
    print("=" * 70)

    # Load all datasets
    heart  = load_heart()
    cancer = load_cancer()
    nhanes = load_nhanes()

    print(f"\n Output folder: {OUT}")
    print("=" * 70)

    # Generate all 5 plot types
    plot_swarm(heart, cancer, nhanes)
    plot_violin(heart, cancer, nhanes)
    plot_interactive(heart, cancer, nhanes)
    plot_strip(heart, cancer, nhanes)
    plot_pair(heart, cancer, nhanes)

    print("\n" + "=" * 70)
    print(" ALL PLOTS SAVED TO:", OUT)
    print("=" * 70)
    print("\n Files generated:")
    print("   1_swarm_plot.png")
    print("   2_violin_plot.png")
    print("   3_interactive_plot.html   ← open in browser")
    print("   4_strip_plot.png")
    print("   5a_pairplot_heart.png")
    print("   5b_pairplot_cancer.png")
    print("   5c_pairplot_nhanes.png")
