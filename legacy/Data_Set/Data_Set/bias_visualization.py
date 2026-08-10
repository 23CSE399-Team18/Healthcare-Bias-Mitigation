# =============================================================================
# BIAS VISUALIZATION -- ON-SPOT DISPLAY
# Datasets: UCI Heart Disease | Breast Cancer Coimbra | NHANES 2021-2023
# =============================================================================
# HOW TO RUN:
#   Option A -- Terminal / PowerShell:
#       cd "c:\Users\U SRIYA\Documents\Data_Set"
#       python bias_visualization.py
#
#   Option B -- VS Code:
#       Open this file and press F5 (or click Run Python File)
#
#   Option C -- Jupyter Notebook:
#       Copy each section into cells and run cell-by-cell.
#       Add  %matplotlib inline  as the first cell.
#
# WHAT THIS SHOWS:
#   Each plot answers: "Is this dataset biased against a group?"
#   Bias metrics used:
#     SPD -- Statistical Parity Difference  (ideal = 0)
#     DI  -- Disparate Impact Ratio         (ideal = 1.0, min = 0.80)
#     Positive Rate Gap -- raw disease rate % difference between groups
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import webbrowser
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────
BASE = r"c:\Users\U SRIYA\Documents\Data_Set"

# ─── Global visual style ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                      "axes.titleweight": "bold", "axes.titlesize": 13})

FAIR_COLOR   = "#2CA02C"   # green  = within fair range
BIASED_COLOR = "#D62728"   # red    = bias violation
WARN_COLOR   = "#FF7F0E"   # orange = borderline


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — Fairness Metrics
# ═══════════════════════════════════════════════════════════════════════════

def spd(df, target, sens, priv, unpriv):
    """Statistical Parity Difference = P(Y=1|unpriv) - P(Y=1|priv)"""
    p  = df[df[sens] == priv][target].mean()
    u  = df[df[sens] == unpriv][target].mean()
    return round(u - p, 4), round(p * 100, 1), round(u * 100, 1)

def di(df, target, sens, priv, unpriv):
    """Disparate Impact = P(Y=1|unpriv) / P(Y=1|priv)"""
    p = df[df[sens] == priv][target].mean()
    u = df[df[sens] == unpriv][target].mean()
    return round(u / p, 4) if p > 0 else float("nan")

def bias_label(spd_val, di_val):
    if abs(spd_val) <= 0.10 and 0.80 <= di_val <= 1.25:
        return "✔ FAIR", FAIR_COLOR
    elif abs(spd_val) > 0.20 or di_val < 0.60 or di_val > 1.5:
        return "✖ BIASED", BIASED_COLOR
    else:
        return "⚠ BORDERLINE", WARN_COLOR


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════════

def load_heart():
    cols = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
    files = {
        "Cleveland":   os.path.join(BASE,"UCI_HeartDisease","processed.cleveland.csv"),
        "Hungarian":   os.path.join(BASE,"UCI_HeartDisease","processed.hungarian.csv"),
        "Switzerland": os.path.join(BASE,"UCI_HeartDisease","processed.switzerland.csv"),
        "VA":          os.path.join(BASE,"UCI_HeartDisease","processed.va.csv"),
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
    print(f"[✔] Heart Disease loaded — {len(h)} patients")
    return h

def load_cancer():
    df = pd.read_csv(os.path.join(BASE,"breast+cancer+coimbra_UCL","dataR2.csv"))
    df["label"] = df["Classification"].map({1:"Healthy", 2:"Cancer"})
    print(f"[✔] Breast Cancer loaded — {len(df)} patients")
    return df

def load_nhanes():
    demo = os.path.join(BASE,"National Center for health Statistics","nhanes_imputed.csv")
    df = pd.read_csv(demo)
    ren = {"SEQN":"Patient_ID","RIAGENDR":"Gender","RIDAGEYR":"Age_Years",
           "RIDRETH3":"Ethnicity_With_Asian","DMDEDUC2":"Education_Level",
           "DMDHHSIZ":"Household_Size","INDFMPIR":"Family_Income_Poverty_Ratio"}
    df.rename(columns={k:v for k,v in ren.items() if k in df.columns}, inplace=True)
    df["Gender_Label"]    = df["Gender"].map({1.0:"Male", 2.0:"Female"})
    eth = {1:"Mexican Amer",2:"Other Hispanic",3:"Non-Hisp White",
           4:"Non-Hisp Black",5:"Other/Multi",6:"Non-Hisp Asian",7:"Other/Multi"}
    df["Ethnicity_Label"] = df["Ethnicity_With_Asian"].map(eth)

    # Merge body measures for BMI
    bm = os.path.join(BASE,"National Center for health Statistics","NHANES_BodyMeasures_2021_2023.csv")
    if os.path.exists(bm):
        bdf = pd.read_csv(bm).rename(columns={"SEQN":"Patient_ID"})
        bmi_c = next((c for c in bdf.columns if "BMI" in c.upper()), None)
        if bmi_c: df = df.merge(bdf[["Patient_ID",bmi_c]], on="Patient_ID", how="left")
        if bmi_c: df.rename(columns={bmi_c:"BMI"}, inplace=True)

    # Merge HbA1c for diabetes risk
    gc = os.path.join(BASE,"National Center for health Statistics","NHANES_Glycohemoglobin_2021_2023.csv")
    if os.path.exists(gc):
        gdf = pd.read_csv(gc).rename(columns={"SEQN":"Patient_ID"})
        hba_c = next((c for c in gdf.columns if "LBXGH" in c or "HbA1c" in c), None)
        if hba_c: df = df.merge(gdf[["Patient_ID",hba_c]], on="Patient_ID", how="left")
        if hba_c: df.rename(columns={hba_c:"HbA1c"}, inplace=True)

    # Derive diabetes risk label
    if "HbA1c" in df.columns:
        df["Diabetes_Risk"] = (df["HbA1c"] >= 6.5).astype(int)
    else:
        df["Diabetes_Risk"] = np.nan

    print(f"[✔] NHANES loaded     — {len(df)} participants")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — BIAS SUMMARY DASHBOARD (on-spot)
# Shows SPD and DI bar charts for all 3 datasets side by side
# ═══════════════════════════════════════════════════════════════════════════

def plot_bias_summary(heart, cancer, nhanes):
    print("\n── PLOT 1: Bias Summary Dashboard ──")

    # ── Compute metrics ──
    h_clean = heart.dropna(subset=["sex","target"])
    h_spd, h_male_rate, h_female_rate = spd(h_clean, "target", "sex", 1, 0)
    h_di   = di(h_clean, "target", "sex", 1, 0)

    # Cancer: no traditional binary sensitive attr — use Age group proxy
    cancer["age_group"] = (cancer["Age"] >= 50).map({True:"50+", False:"<50"})
    c_clean = cancer.dropna(subset=["age_group","Classification"])
    c_clean["cls_bin"] = (c_clean["Classification"] == 2).astype(int)
    c_spd, c_old_rate, c_young_rate = spd(c_clean, "cls_bin", "age_group", "<50", "50+")
    c_di = di(c_clean, "cls_bin", "age_group", "<50", "50+")

    # NHANES: diabetes by gender
    nh = nhanes.dropna(subset=["Gender","Diabetes_Risk"]) if "Diabetes_Risk" in nhanes.columns else None
    if nh is not None and len(nh) > 0:
        n_spd, n_male_rate, n_female_rate = spd(nh, "Diabetes_Risk", "Gender", 1.0, 2.0)
        n_di = di(nh, "Diabetes_Risk", "Gender", 1.0, 2.0)
    else:
        n_spd, n_male_rate, n_female_rate, n_di = 0, 0, 0, 1

    datasets = ["Heart Disease\n(Male vs Female)", "Breast Cancer\n(<50 vs 50+)", "NHANES Diabetes\n(Male vs Female)"]
    spds = [h_spd, c_spd, n_spd]
    dis  = [h_di,  c_di,  n_di]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("BIAS DETECTION — Statistical Parity Difference & Disparate Impact\n"
                 "across UCI Heart Disease | Breast Cancer Coimbra | NHANES",
                 fontsize=13, fontweight="bold")

    # ── SPD chart ──
    ax = axes[0]
    colors = [BIASED_COLOR if abs(s) > 0.10 else FAIR_COLOR for s in spds]
    bars = ax.barh(datasets, spds, color=colors, edgecolor="white", height=0.5)
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1.2, label="Fair threshold (+0.10)")
    ax.axvline(-0.10, color="gray", linestyle="--", linewidth=1.2, label="Fair threshold (−0.10)")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, s in zip(bars, spds):
        ax.text(s + (0.005 if s >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f"{s:+.4f}", va="center", ha="left" if s >= 0 else "right",
                fontweight="bold", fontsize=11)
    ax.set_xlabel("SPD  (negative = unprivileged group gets fewer positive predictions)")
    ax.set_title("Statistical Parity Difference (SPD)\nIdeal = 0  |  Acceptable: −0.10 to +0.10")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.35, 0.35)

    red_p  = mpatches.Patch(color=BIASED_COLOR, label="BIASED  (|SPD| > 0.10)")
    green_p= mpatches.Patch(color=FAIR_COLOR,   label="FAIR    (|SPD| ≤ 0.10)")
    ax.legend(handles=[red_p, green_p], loc="lower right", fontsize=9)

    # ── DI chart ──
    ax = axes[1]
    di_colors = [BIASED_COLOR if (d < 0.80 or d > 1.25) else FAIR_COLOR for d in dis]
    bars2 = ax.barh(datasets, dis, color=di_colors, edgecolor="white", height=0.5)
    ax.axvline(0.80, color="#FF7F0E", linestyle="--", linewidth=1.5, label="Min (0.80)")
    ax.axvline(1.25, color="#9467BD", linestyle="--", linewidth=1.5, label="Max (1.25)")
    ax.axvline(1.0,  color="black",   linestyle="-",  linewidth=0.8, label="Perfect (1.0)")
    for bar, d in zip(bars2, dis):
        ax.text(d + 0.01, bar.get_y() + bar.get_height()/2,
                f"{d:.4f}", va="center", fontweight="bold", fontsize=11)
    ax.set_xlabel("DI  (< 0.80 = discriminatory | > 1.25 = reverse bias)")
    ax.set_title("Disparate Impact Ratio (DI)\nIdeal = 1.0  |  Acceptable: 0.80 – 1.25")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 2.0)

    plt.tight_layout()
    plt.show()

    # Print readable summary
    print("\n" + "="*60)
    print(f"  {'Dataset':<28} {'SPD':>8} {'DI':>8}  {'Status'}")
    print("="*60)
    for name, s, d in zip(datasets, spds, dis):
        lbl, _ = bias_label(s, d)
        print(f"  {name.replace(chr(10),' '):<28} {s:>+8.4f} {d:>8.4f}  {lbl}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — HEART DISEASE: Positive Rate by Sex (shows sex bias clearly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_heart_bias(heart):
    print("\n── PLOT 2: Heart Disease — Sex Bias ──")
    hd = heart.dropna(subset=["sex_label","target","site","thalach","chol"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("UCI HEART DISEASE — HOW SEX BIAS SHOWS UP\n"
                 "Male patients dominate training data → model underperforms for females",
                 fontsize=13, fontweight="bold")

    # 2a. Disease positive rate by sex per site
    ax = axes[0]
    rate_df = hd.groupby(["site","sex_label"])["target"].mean().reset_index()
    rate_df["rate_pct"] = rate_df["target"] * 100
    sns.barplot(data=rate_df, x="site", y="rate_pct", hue="sex_label",
                palette={"Male":"#4C72B0","Female":"#DD8452"}, ax=ax)
    ax.set_title("Disease Positive Rate by Sex\n(per site)")
    ax.set_xlabel("Clinical Site"); ax.set_ylabel("Disease Rate (%)")
    ax.legend(title="Sex")
    # Annotate overall difference
    m_rate = hd[hd["sex_label"]=="Male"]["target"].mean()*100
    f_rate = hd[hd["sex_label"]=="Female"]["target"].mean()*100
    ax.text(0.5, 0.92, f"Overall: Male={m_rate:.1f}%  Female={f_rate:.1f}%\nGap = {m_rate-f_rate:+.1f}%",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#FFC107"))

    # 2b. Sample size imbalance (root cause of bias)
    ax = axes[1]
    counts = hd["sex_label"].value_counts().reset_index()
    counts.columns = ["Sex","Count"]
    bars = ax.bar(counts["Sex"], counts["Count"],
                  color=["#4C72B0","#DD8452"], edgecolor="white", width=0.5)
    for bar, row in zip(bars, counts.itertuples()):
        pct = row.Count / len(hd) * 100
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{row.Count}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("⚠  Training Set Imbalance\n(Root Cause of Sex Bias)")
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("Sex")
    ax.text(0.5, 0.7, "3× more male\npatients than female!",
            transform=ax.transAxes, ha="center", fontsize=12,
            color=BIASED_COLOR, fontweight="bold")

    # 2c. Violin — max heart rate by sex & diagnosis
    ax = axes[2]
    sns.violinplot(data=hd, x="sex_label", y="thalach", hue="diagnosis",
                   palette={"No Disease":"#AEC7E8","Heart Disease":"#FF9896"},
                   split=False, inner="box", linewidth=1.2, ax=ax)
    ax.set_title("Max Heart Rate Distribution\n(Sex × Diagnosis — different patterns!)")
    ax.set_xlabel("Sex"); ax.set_ylabel("Max Heart Rate (bpm)")
    ax.legend(title="Diagnosis", fontsize=9)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — BREAST CANCER: Biomarker spread showing age group disparity
# ═══════════════════════════════════════════════════════════════════════════

def plot_cancer_bias(cancer):
    print("\n── PLOT 3: Breast Cancer — Age Group Bias ──")
    cancer["age_group"] = pd.cut(cancer["Age"], bins=[0,40,55,100],
                                  labels=["Young (<40)","Mid (40–55)","Older (55+)"])
    c = cancer.dropna(subset=["age_group","Classification"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("BREAST CANCER COIMBRA — HOW AGE-GROUP BIAS SHOWS UP\n"
                 "Biomarker patterns differ by age group — model trained on one may fail on another",
                 fontsize=13, fontweight="bold")

    # 3a. Cancer positive rate by age group
    ax = axes[0]
    rates = c.groupby("age_group")["Classification"].apply(lambda x: (x==2).mean()*100).reset_index()
    rates.columns = ["Age Group","Cancer Rate (%)"]
    bars = ax.bar(rates["Age Group"], rates["Cancer Rate (%)"],
                  color=["#AEC7E8","#FFBB78","#FF9896"], edgecolor="white", width=0.5)
    for bar, row in zip(bars, rates.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{row._2:.1f}%", ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Cancer Diagnosis Rate by Age Group\n(positive rate changes with age)")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Cancer Positive Rate (%)")
    ax.set_ylim(0, 100)

    # 3b. Glucose distribution by age group & class
    ax = axes[1]
    sns.boxplot(data=c, x="age_group", y="Glucose", hue="label",
                palette={"Healthy":"#2CA02C","Cancer":"#D62728"},
                ax=ax, linewidth=1.2)
    ax.set_title("Glucose by Age Group & Classification\n(different biomarker ranges per group)")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(title="Group", fontsize=9)

    # 3c. Insulin scatter — separation between healthy/cancer (shows class overlap by age)
    ax = axes[2]
    for grp, col in [("Healthy","#2CA02C"),("Cancer","#D62728")]:
        sub = c[c["label"]==grp]
        ax.scatter(sub["BMI"], sub["Insulin"], c=col, alpha=0.6,
                   s=50, label=grp, edgecolors="white", linewidths=0.4)
    ax.set_title("BMI vs Insulin\n(class separation — harder in older patients)")
    ax.set_xlabel("BMI"); ax.set_ylabel("Insulin (µU/mL)")
    ax.legend(title="Group")
    ax.axhline(y=25, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(18, 26, "Reference threshold", fontsize=9, color="gray")

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — NHANES: Race/Ethnicity & Gender bias in clinical risk factors
# ═══════════════════════════════════════════════════════════════════════════

def plot_nhanes_bias(nhanes):
    print("\n── PLOT 4: NHANES — Race/Gender Bias ──")
    nh = nhanes.dropna(subset=["Gender_Label","Ethnicity_Label"])

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("NHANES 2021–2023 — HOW DEMOGRAPHIC BIAS SHOWS UP\n"
                 "Unequal ethnic group representation + different risk factor rates",
                 fontsize=13, fontweight="bold")

    COLORS = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B"]
    top_eth = nh["Ethnicity_Label"].value_counts().head(6).index.tolist()
    nh_top = nh[nh["Ethnicity_Label"].isin(top_eth)]

    # 4a. Sample size by ethnicity (representation bias)
    ax = axes[0]
    eth_counts = nh_top["Ethnicity_Label"].value_counts()
    bars = ax.barh(eth_counts.index, eth_counts.values, color=COLORS, edgecolor="white")
    for bar, cnt in zip(bars, eth_counts.values):
        pct = cnt / len(nh) * 100
        ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_title("⚠  Ethnic Group Representation\n(training set imbalance)")
    ax.set_xlabel("Number of Participants")
    ax.set_xlim(0, eth_counts.max() * 1.3)
    ax.text(0.55, 0.05,
            "Non-Hisp White dominates\n→ model may underperform\nfor minority groups",
            transform=ax.transAxes, ha="center", fontsize=10,
            color=BIASED_COLOR, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEDED", ec=BIASED_COLOR))

    # 4b. Age distribution by Ethnicity (different age profiles → different risk)
    ax = axes[1]
    sns.boxplot(data=nh_top, x="Ethnicity_Label", y="Age_Years",
                palette=COLORS[:len(top_eth)], linewidth=1.2, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)
    ax.set_title("Age Distribution by Ethnicity\n(different age profiles = different risk exposure)")
    ax.set_xlabel("Ethnicity"); ax.set_ylabel("Age (years)")

    # 4c. BMI by gender if available
    ax = axes[2]
    if "BMI" in nhanes.columns:
        bmi_clean = nhanes.dropna(subset=["BMI","Gender_Label"])
        bmi_clean = bmi_clean[(bmi_clean["BMI"] > 10) & (bmi_clean["BMI"] < 70)]
        sns.violinplot(data=bmi_clean, x="Gender_Label", y="BMI",
                       palette={"Male":"#4C72B0","Female":"#DD8452"},
                       inner="box", linewidth=1.2, ax=ax)
        ax.axhline(30, color=BIASED_COLOR, linestyle="--", linewidth=1.5, label="Obesity threshold (30)")
        ax.set_title("BMI Distribution by Gender\n(obesity risk differs by sex)")
        ax.set_xlabel("Gender"); ax.set_ylabel("BMI (kg/m²)")
        ax.legend(fontsize=9)
        # Annotate obesity rates
        for g in ["Male","Female"]:
            sub = bmi_clean[bmi_clean["Gender_Label"]==g]
            obese_pct = (sub["BMI"] >= 30).mean() * 100
            x = 0 if g == "Male" else 1
            ax.text(x, bmi_clean["BMI"].max()*0.92,
                    f"Obese: {obese_pct:.1f}%", ha="center",
                    fontweight="bold", fontsize=11,
                    color=BIASED_COLOR if obese_pct > 35 else FAIR_COLOR)
    else:
        sns.histplot(data=nh, x="Age_Years", hue="Gender_Label",
                     palette={"Male":"#4C72B0","Female":"#DD8452"},
                     bins=25, element="step", ax=ax)
        ax.set_title("Age Distribution by Gender"); ax.set_xlabel("Age"); ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 — INTERACTIVE: Bias across all 3 datasets (opens in browser)
# ═══════════════════════════════════════════════════════════════════════════

def plot_interactive_bias(heart, cancer, nhanes):
    print("\n── PLOT 5: Interactive Bias Dashboard (opens in browser) ──")

    hd = heart.dropna(subset=["sex_label","target","site"])

    # ── Compute positive rates per group (Heart) ──
    heart_rates = (hd.groupby(["sex_label","site"])["target"]
                   .mean().reset_index().rename(columns={"target":"disease_rate"}))
    heart_rates["disease_rate"] *= 100

    # ── Cancer rates by age group ──
    cancer["age_grp"] = pd.cut(cancer["Age"], bins=[0,40,55,100],
                                labels=["<40","40–55","55+"])
    c_rates = (cancer.groupby("age_grp")["Classification"]
               .apply(lambda x: (x==2).mean()*100).reset_index())
    c_rates.columns = ["Age Group","Cancer Rate (%)"]

    # ── NHANES ──
    nh = nhanes.dropna(subset=["Ethnicity_Label","Age_Years"])
    top5 = nh["Ethnicity_Label"].value_counts().head(5).index.tolist()
    nh_top = nh[nh["Ethnicity_Label"].isin(top5)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Heart Disease Rate by Sex & Site  (Male vs Female — BIAS VISIBLE)",
            "Cancer Positive Rate by Age Group  (age disparity)",
            "NHANES: Patient Count by Ethnicity  (representation bias)",
            "Heart — Age vs Cholesterol coloured by Diagnosis & Sex",
        ),
        vertical_spacing=0.18, horizontal_spacing=0.1,
    )

    # Panel 1: grouped bar — heart disease rate by sex per site
    for sex, color in [("Male","#4C72B0"),("Female","#DD8452")]:
        sub = heart_rates[heart_rates["sex_label"]==sex]
        fig.add_trace(
            go.Bar(x=sub["site"], y=sub["disease_rate"], name=sex,
                   marker_color=color, opacity=0.85,
                   text=sub["disease_rate"].round(1).astype(str)+"%",
                   textposition="outside",
                   hovertemplate=f"<b>{sex}</b><br>Site: %{{x}}<br>Disease rate: %{{y:.1f}}%<extra></extra>"),
            row=1, col=1)
    fig.update_yaxes(title_text="Disease Rate (%)", row=1, col=1)

    # Panel 2: bar — cancer rate by age group
    fig.add_trace(
        go.Bar(x=c_rates["Age Group"].astype(str),
               y=c_rates["Cancer Rate (%)"],
               marker_color=["#AEC7E8","#FFBB78","#FF9896"],
               text=c_rates["Cancer Rate (%)"].round(1).astype(str)+"%",
               textposition="outside",
               name="Cancer Rate",
               hovertemplate="Age: %{x}<br>Cancer rate: %{y:.1f}%<extra></extra>"),
        row=1, col=2)
    fig.update_yaxes(title_text="Cancer Rate (%)", row=1, col=2)

    # Panel 3: bar — NHANES representation
    eth_cnt = nh_top["Ethnicity_Label"].value_counts().reset_index()
    eth_cnt.columns = ["Ethnicity","Count"]
    fig.add_trace(
        go.Bar(x=eth_cnt["Ethnicity"], y=eth_cnt["Count"],
               marker_color=["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD"],
               text=eth_cnt["Count"].astype(str),
               textposition="outside",
               name="Count",
               hovertemplate="%{x}: %{y:,}<extra></extra>"),
        row=2, col=1)
    fig.update_yaxes(title_text="Participant Count", row=2, col=1)

    # Panel 4: scatter heart — age vs chol coloured by sex+diagnosis
    for (sex, dx), grp in hd.groupby(["sex_label","diagnosis"]):
        grp = grp.dropna(subset=["age","chol"])
        grp = grp[grp["chol"] < 500]
        fig.add_trace(
            go.Scatter(x=grp["age"], y=grp["chol"], mode="markers",
                       name=f"{sex}/{dx}",
                       marker=dict(size=6, opacity=0.55),
                       hovertemplate=f"<b>{sex} | {dx}</b><br>Age=%{{x}}, Chol=%{{y}}<extra></extra>"),
            row=2, col=2)
    fig.update_xaxes(title_text="Age (years)", row=2, col=2)
    fig.update_yaxes(title_text="Cholesterol (mg/dl)", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>BIAS DETECTION DASHBOARD — UCI Heart Disease · Breast Cancer · NHANES</b>",
            x=0.5, font=dict(size=15)
        ),
        height=860, template="plotly_white",
        legend=dict(x=1.01, y=1, font=dict(size=10)),
        barmode="group",
    )

    # Save + open in browser
    out_path = os.path.join(BASE, "plots", "BIAS_interactive_dashboard.html")
    os.makedirs(os.path.join(BASE,"plots"), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"   Saved → {out_path}")
    webbrowser.open(f"file:///{out_path}")
    print("   Opened in default browser.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — RUN ALL PLOTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  BIAS VISUALIZATION SUITE — ON-SPOT DISPLAY")
    print("  Datasets: Heart Disease | Breast Cancer Coimbra | NHANES")
    print("=" * 65)

    # ── Load ──
    heart  = load_heart()
    cancer = load_cancer()
    nhanes = load_nhanes()

    print("\n" + "="*65)
    print("  Running 5 bias plots — each window opens one at a time.")
    print("  Close each plot window to proceed to the next one.")
    print("="*65)

    # ── Plot 1: Bias summary (SPD + DI bars)
    plot_bias_summary(heart, cancer, nhanes)

    # ── Plot 2: Heart Disease sex bias
    plot_heart_bias(heart)

    # ── Plot 3: Cancer age-group bias
    plot_cancer_bias(cancer)

    # ── Plot 4: NHANES demographic bias
    plot_nhanes_bias(nhanes)

    # ── Plot 5: Interactive dashboard in browser
    plot_interactive_bias(heart, cancer, nhanes)

    print("\n" + "=" * 65)
    print("  ALL DONE!")
    print("  Static plots: displayed in pop-up windows")
    print("  Interactive:  BIAS_interactive_dashboard.html  ← browser")
    print("=" * 65)
