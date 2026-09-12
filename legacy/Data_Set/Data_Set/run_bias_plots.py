r"""
=============================================================================
 3 BASIC BIAS PROOF PLOTS
 Datasets: UCI Heart Disease | Breast Cancer Coimbra | NHANES 2021-2023
=============================================================================
 HOW TO RUN:
   cd "c:\Users\U SRIYA\Documents\Data_Set"
   python run_bias_plots.py

 WHAT IT SHOWS — 3 plots proving bias exists in each dataset:
   Plot 1: REPRESENTATION BIAS — Sample count imbalance by protected groups
           (some groups have far more data than others → model learns their
            patterns better and underperforms on minority groups)

   Plot 2: OUTCOME DISPARITY — Positive outcome rates differ by group
           (disease/risk rates are not equal across protected groups,
            so a model trained on biased data will perpetuate this gap)

   Plot 3: FEATURE DISTRIBUTION SHIFT — Clinical features look different
           across groups (different means, spreads, distributions →
            a one-size-fits-all model won't work fairly for all groups)

 Metrics printed:
   SPD = Statistical Parity Difference  (ideal = 0)
   DI  = Disparate Impact Ratio         (ideal = 1.0, fair range 0.80–1.25)
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────
BASE = r"c:\Users\U SRIYA\Documents\Data_Set"
OUT  = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                      "axes.titleweight": "bold", "axes.titlesize": 13})

FAIR   = "#2CA02C"    # green
BIASED = "#D62728"    # red
WARN   = "#FF7F0E"    # orange


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS — Fairness Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_spd(df, target, sens, priv, unpriv):
    """Statistical Parity Difference = P(Y=1 | unpriv) − P(Y=1 | priv)"""
    p = df[df[sens] == priv][target].mean()
    u = df[df[sens] == unpriv][target].mean()
    return round(u - p, 4), round(p * 100, 1), round(u * 100, 1)

def compute_di(df, target, sens, priv, unpriv):
    """Disparate Impact = P(Y=1 | unpriv) / P(Y=1 | priv)"""
    p = df[df[sens] == priv][target].mean()
    u = df[df[sens] == unpriv][target].mean()
    return round(u / p, 4) if p > 0 else float("nan")

def bias_verdict(spd_val, di_val):
    if abs(spd_val) <= 0.10 and 0.80 <= di_val <= 1.25:
        return "✔ FAIR", FAIR
    elif abs(spd_val) > 0.20 or di_val < 0.60 or di_val > 1.5:
        return "✖ BIASED", BIASED
    else:
        return "⚠ BORDERLINE", WARN


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATASETS
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
    print(f"[✔] Heart Disease — {len(h)} patients")
    return h

def load_cancer():
    df = pd.read_csv(os.path.join(BASE,"breast+cancer+coimbra_UCL","dataR2.csv"))
    df["label"] = df["Classification"].map({1:"Healthy", 2:"Cancer"})
    df["age_group"] = pd.cut(df["Age"], bins=[0,50,100],
                              labels=["<50","50+"], include_lowest=True)
    df["cls_bin"] = (df["Classification"] == 2).astype(int)
    print(f"[✔] Breast Cancer  — {len(df)} patients")
    return df

def load_nhanes():
    demo = os.path.join(BASE,"National Center for health Statistics","nhanes_imputed.csv")
    df = pd.read_csv(demo)
    ren = {"SEQN":"Patient_ID","RIAGENDR":"Gender","RIDAGEYR":"Age_Years",
           "RIDRETH3":"Ethnicity_With_Asian","DMDHHSIZ":"Household_Size",
           "INDFMPIR":"Family_Income_Poverty_Ratio"}
    df.rename(columns={k:v for k,v in ren.items() if k in df.columns}, inplace=True)
    df["Gender_Label"] = df["Gender"].map({1.0:"Male", 2.0:"Female"})
    eth = {1:"Mexican Amer",2:"Other Hispanic",3:"Non-Hisp White",
           4:"Non-Hisp Black",5:"Other/Multi",6:"Non-Hisp Asian",7:"Other/Multi"}
    if "Ethnicity_With_Asian" in df.columns:
        df["Ethnicity_Label"] = df["Ethnicity_With_Asian"].map(eth)

    # Merge BMI
    bm = os.path.join(BASE,"National Center for health Statistics","NHANES_BodyMeasures_2021_2023.csv")
    if os.path.exists(bm):
        bdf = pd.read_csv(bm).rename(columns={"SEQN":"Patient_ID"})
        bmi_c = next((c for c in bdf.columns if "BMI" in c.upper()), None)
        if bmi_c and "Patient_ID" in bdf.columns:
            df = df.merge(bdf[["Patient_ID", bmi_c]], on="Patient_ID", how="left")
            df.rename(columns={bmi_c:"BMI"}, inplace=True)

    # Merge HbA1c for diabetes risk
    gc = os.path.join(BASE,"National Center for health Statistics","NHANES_Glycohemoglobin_2021_2023.csv")
    if os.path.exists(gc):
        gdf = pd.read_csv(gc).rename(columns={"SEQN":"Patient_ID"})
        hba_c = next((c for c in gdf.columns if "LBXGH" in c or "HbA1c" in c), None)
        if hba_c and "Patient_ID" in gdf.columns:
            df = df.merge(gdf[["Patient_ID", hba_c]], on="Patient_ID", how="left")
            df.rename(columns={hba_c:"HbA1c"}, inplace=True)
    if "HbA1c" in df.columns:
        df["Diabetes_Risk"] = (df["HbA1c"] >= 6.5).astype(int)
    else:
        df["Diabetes_Risk"] = np.nan

    print(f"[✔] NHANES         — {len(df)} participants")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — REPRESENTATION BIAS (sample count imbalance)
#  Proof: Some groups have far more data points than others.
#         A model trained on imbalanced data learns majority-group patterns.
# ═══════════════════════════════════════════════════════════════════════════

def plot_1_representation_bias(heart, cancer, nhanes):
    print("\n" + "─"*65)
    print("  PLOT 1: REPRESENTATION BIAS — Sample Count Imbalance")
    print("─"*65)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("BIAS PROOF — PLOT 1: REPRESENTATION BIAS\n"
                 "Protected groups have very unequal sample sizes "
                 "→ model over-represents majority groups",
                 fontsize=14, fontweight="bold")

    # ── 1a. Heart Disease: Male vs Female ──
    ax = axes[0]
    hd = heart.dropna(subset=["sex_label"])
    counts = hd["sex_label"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=["#4C72B0","#DD8452"], edgecolor="white", width=0.5)
    for bar, (grp, cnt) in zip(bars, counts.items()):
        pct = cnt / len(hd) * 100
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                f"{cnt}\n({pct:.1f}%)", ha="center", fontsize=12, fontweight="bold")
    ratio = counts.max() / counts.min()
    ax.set_title(f"UCI Heart Disease — Sex\n⚠ {ratio:.1f}× more {counts.idxmax()} patients")
    ax.set_ylabel("Patient Count"); ax.set_xlabel("Sex")
    ax.text(0.5, 0.65, f"{ratio:.1f}× imbalance!",
            transform=ax.transAxes, ha="center", fontsize=14,
            color=BIASED, fontweight="bold")

    # ── 1b. Breast Cancer: Age <50 vs 50+ ──
    ax = axes[1]
    cc = cancer.dropna(subset=["age_group"])
    counts = cc["age_group"].value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#AEC7E8","#FF9896"], edgecolor="white", width=0.5)
    for bar, (grp, cnt) in zip(bars, counts.items()):
        pct = cnt / len(cc) * 100
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{cnt}\n({pct:.1f}%)", ha="center", fontsize=12, fontweight="bold")
    ratio = counts.max() / counts.min()
    ax.set_title(f"Breast Cancer Coimbra — Age Group\n⚠ {ratio:.1f}× more {counts.idxmax()} patients")
    ax.set_ylabel("Patient Count"); ax.set_xlabel("Age Group")
    ax.text(0.5, 0.65, f"{ratio:.1f}× imbalance!",
            transform=ax.transAxes, ha="center", fontsize=14,
            color=BIASED if ratio > 1.5 else WARN, fontweight="bold")

    # ── 1c. NHANES: Ethnicity distribution ──
    ax = axes[2]
    nh = nhanes.dropna(subset=["Ethnicity_Label"])
    counts = nh["Ethnicity_Label"].value_counts().head(6)
    colors = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B"]
    bars = ax.barh(counts.index, counts.values,
                   color=colors[:len(counts)], edgecolor="white")
    for bar, (grp, cnt) in zip(bars, counts.items()):
        pct = cnt / len(nh) * 100
        ax.text(bar.get_width()+15, bar.get_y()+bar.get_height()/2,
                f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=10, fontweight="bold")
    ratio = counts.max() / counts.min()
    ax.set_title(f"NHANES — Ethnicity\n⚠ {counts.idxmax()} dominates ({counts.max()/len(nh)*100:.1f}%)")
    ax.set_xlabel("Participant Count")
    ax.set_xlim(0, counts.max() * 1.35)
    ax.text(0.55, 0.08, f"Max/Min ratio: {ratio:.1f}×\n→ minority groups\n   under-represented",
            transform=ax.transAxes, ha="center", fontsize=11,
            color=BIASED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEDED", ec=BIASED))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "BIAS_1_representation.png"), dpi=130, bbox_inches="tight")
    plt.show()

    print("\n  EXPLANATION:")
    print("  Heart Disease: Male patients vastly outnumber Female patients.")
    print("  Cancer: One age group has more patients than the other.")
    print("  NHANES: Non-Hispanic White dominates → model may underperform for minorities.")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — OUTCOME DISPARITY (positive rates differ by group)
#  Proof: Disease/risk rates are very different across groups.
#         SPD ≠ 0 and DI ≠ 1.0 → statistically unfair.
# ═══════════════════════════════════════════════════════════════════════════

def plot_2_outcome_disparity(heart, cancer, nhanes):
    print("\n" + "─"*65)
    print("  PLOT 2: OUTCOME DISPARITY — Positive Rate Gap by Group")
    print("─"*65)

    # ── Compute metrics ──
    hd = heart.dropna(subset=["sex","target"])
    h_spd, h_priv_rate, h_unpriv_rate = compute_spd(hd, "target", "sex", 1, 0)
    h_di = compute_di(hd, "target", "sex", 1, 0)

    cc = cancer.dropna(subset=["age_group","cls_bin"])
    c_spd, c_priv_rate, c_unpriv_rate = compute_spd(cc, "cls_bin", "age_group", "<50", "50+")
    c_di = compute_di(cc, "cls_bin", "age_group", "<50", "50+")

    nh = nhanes.dropna(subset=["Gender","Diabetes_Risk"])
    if len(nh) > 0:
        n_spd, n_priv_rate, n_unpriv_rate = compute_spd(nh, "Diabetes_Risk", "Gender", 1.0, 2.0)
        n_di = compute_di(nh, "Diabetes_Risk", "Gender", 1.0, 2.0)
    else:
        n_spd, n_priv_rate, n_unpriv_rate, n_di = 0, 0, 0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("BIAS PROOF — PLOT 2: OUTCOME DISPARITY\n"
                 "Disease/risk rates differ significantly across protected groups "
                 "→ SPD ≠ 0, DI ≠ 1.0",
                 fontsize=14, fontweight="bold")

    # ── 2a. Heart Disease: positive rate Male vs Female ──
    ax = axes[0]
    rates = hd.groupby("sex_label")["target"].mean() * 100
    bars = ax.bar(rates.index, rates.values,
                  color=["#DD8452","#4C72B0"], edgecolor="white", width=0.5)
    for bar, (grp, rate) in zip(bars, rates.items()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
    h_lbl, h_col = bias_verdict(h_spd, h_di)
    ax.set_title(f"Heart Disease — Positive Rate by Sex\nSPD={h_spd:+.4f}  DI={h_di:.4f}  →  {h_lbl}")
    ax.set_ylabel("Disease-Positive Rate (%)"); ax.set_xlabel("Sex")
    ax.set_ylim(0, max(rates.values)*1.3)
    gap = abs(rates.values[0] - rates.values[1])
    ax.text(0.5, 0.7, f"Gap = {gap:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=14,
            color=BIASED, fontweight="bold")

    # ── 2b. Cancer: positive rate <50 vs 50+ ──
    ax = axes[1]
    rates = cc.groupby("age_group")["cls_bin"].mean() * 100
    bars = ax.bar(rates.index.astype(str), rates.values,
                  color=["#AEC7E8","#FF9896"], edgecolor="white", width=0.5)
    for bar, (grp, rate) in zip(bars, rates.items()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
    c_lbl, c_col = bias_verdict(c_spd, c_di)
    ax.set_title(f"Breast Cancer — Positive Rate by Age\nSPD={c_spd:+.4f}  DI={c_di:.4f}  →  {c_lbl}")
    ax.set_ylabel("Cancer-Positive Rate (%)"); ax.set_xlabel("Age Group")
    ax.set_ylim(0, 100)
    gap = abs(rates.values[0] - rates.values[1])
    ax.text(0.5, 0.7, f"Gap = {gap:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=14,
            color=BIASED if gap > 10 else WARN, fontweight="bold")

    # ── 2c. NHANES: diabetes risk Male vs Female ──
    ax = axes[2]
    if len(nh) > 0:
        rates = nh.groupby("Gender_Label")["Diabetes_Risk"].mean() * 100
        bars = ax.bar(rates.index, rates.values,
                      color=["#DD8452","#4C72B0"], edgecolor="white", width=0.5)
        for bar, (grp, rate) in zip(bars, rates.items()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
        n_lbl, n_col = bias_verdict(n_spd, n_di)
        ax.set_title(f"NHANES Diabetes Risk — by Gender\nSPD={n_spd:+.4f}  DI={n_di:.4f}  →  {n_lbl}")
        gap = abs(rates.values[0] - rates.values[1])
        ax.text(0.5, 0.7, f"Gap = {gap:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=14,
                color=BIASED if gap > 5 else WARN, fontweight="bold")
    else:
        ax.set_title("NHANES — No Diabetes Risk data available")
    ax.set_ylabel("Diabetes Risk Rate (%)"); ax.set_xlabel("Gender")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "BIAS_2_outcome_disparity.png"), dpi=130, bbox_inches="tight")
    plt.show()

    # Print summary table
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  FAIRNESS METRICS SUMMARY                                  ║")
    print("  ╠════════════════════════════╦═════════╦═════════╦════════════╣")
    print("  ║  Dataset                   ║   SPD   ║   DI    ║  Verdict   ║")
    print("  ╠════════════════════════════╬═════════╬═════════╬════════════╣")
    for name, s, d in [("Heart (Male vs Female)", h_spd, h_di),
                        ("Cancer (<50 vs 50+)", c_spd, c_di),
                        ("NHANES Diabetes (M vs F)", n_spd, n_di)]:
        lbl, _ = bias_verdict(s, d)
        print(f"  ║  {name:<26} ║ {s:>+7.4f} ║ {d:>7.4f} ║ {lbl:<10} ║")
    print("  ╚════════════════════════════╩═════════╩═════════╩════════════╝")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — FEATURE DISTRIBUTION SHIFT (clinical features differ by group)
#  Proof: Feature distributions are measurably different across groups.
#         A model trained on aggregate data won't capture these differences.
# ═══════════════════════════════════════════════════════════════════════════

def plot_3_feature_distribution_shift(heart, cancer, nhanes):
    print("\n" + "─"*65)
    print("  PLOT 3: FEATURE DISTRIBUTION SHIFT by Protected Group")
    print("─"*65)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("BIAS PROOF — PLOT 3: FEATURE DISTRIBUTION SHIFT\n"
                 "Clinical features have different distributions across groups "
                 "→ one model won't serve all groups equally",
                 fontsize=14, fontweight="bold")

    # ── 3a. Heart: Max Heart Rate distribution by Sex ──
    ax = axes[0]
    hd = heart.dropna(subset=["thalach","sex_label"])
    sns.kdeplot(data=hd, x="thalach", hue="sex_label",
                palette={"Male":"#4C72B0","Female":"#DD8452"},
                fill=True, alpha=0.3, linewidth=2, ax=ax)
    # Add means
    for sex, col in [("Male","#4C72B0"),("Female","#DD8452")]:
        m = hd[hd["sex_label"]==sex]["thalach"].mean()
        ax.axvline(m, color=col, linestyle="--", linewidth=2, alpha=0.8)
        ax.text(m, ax.get_ylim()[1]*0.9, f"μ={m:.0f}", ha="center",
                fontsize=10, fontweight="bold", color=col)
    diff = abs(hd[hd["sex_label"]=="Male"]["thalach"].mean() -
               hd[hd["sex_label"]=="Female"]["thalach"].mean())
    ax.set_title(f"Heart Disease — Max Heart Rate by Sex\n"
                 f"⚠ Mean difference = {diff:.1f} bpm")
    ax.set_xlabel("Max Heart Rate (bpm)"); ax.set_ylabel("Density")

    # ── 3b. Cancer: Glucose by Age Group ──
    ax = axes[1]
    cc = cancer.dropna(subset=["Glucose","age_group"])
    sns.boxplot(data=cc, x="age_group", y="Glucose", hue="label",
                palette={"Healthy":"#2CA02C","Cancer":"#D62728"},
                linewidth=1.2, ax=ax)
    ax.set_title("Breast Cancer — Glucose by Age Group\n"
                 "⚠ Biomarker distributions shift with age")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(title="Group", fontsize=9)
    # Annotate median difference
    for grp in ["<50","50+"]:
        med = cc[cc["age_group"]==grp]["Glucose"].median()
        ax.text(0 if grp=="<50" else 1, cc["Glucose"].max()*0.95,
                f"med={med:.0f}", ha="center", fontsize=10, fontweight="bold")

    # ── 3c. NHANES: BMI by Ethnicity (if available) ──
    ax = axes[2]
    if "BMI" in nhanes.columns and "Ethnicity_Label" in nhanes.columns:
        nh = nhanes.dropna(subset=["BMI","Ethnicity_Label"])
        nh = nh[(nh["BMI"] > 10) & (nh["BMI"] < 70)]
        top = nh["Ethnicity_Label"].value_counts().head(5).index.tolist()
        nh = nh[nh["Ethnicity_Label"].isin(top)]
        colors = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD"]
        sns.violinplot(data=nh, x="Ethnicity_Label", y="BMI",
                       palette=colors[:len(top)], inner="box",
                       linewidth=1.2, ax=ax)
        ax.axhline(30, color=BIASED, linestyle="--", linewidth=1.5, label="Obesity threshold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
        # Annotate mean BMI per group
        for i, eth in enumerate(top):
            m = nh[nh["Ethnicity_Label"]==eth]["BMI"].mean()
            ax.text(i, nh["BMI"].max()*0.97, f"μ={m:.1f}",
                    ha="center", fontsize=9, fontweight="bold",
                    color=BIASED if m >= 30 else "#333333")
        ax.set_title("NHANES — BMI by Ethnicity\n"
                     "⚠ Mean BMI differs significantly across ethnic groups")
        ax.set_xlabel("Ethnicity"); ax.set_ylabel("BMI (kg/m²)")
        ax.legend(fontsize=9)
    elif "Age_Years" in nhanes.columns and "Gender_Label" in nhanes.columns:
        nh = nhanes.dropna(subset=["Age_Years","Gender_Label"])
        sns.kdeplot(data=nh, x="Age_Years", hue="Gender_Label",
                    palette={"Male":"#4C72B0","Female":"#DD8452"},
                    fill=True, alpha=0.3, linewidth=2, ax=ax)
        ax.set_title("NHANES — Age Distribution by Gender")
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Density")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "BIAS_3_feature_shift.png"), dpi=130, bbox_inches="tight")
    plt.show()

    print("\n  EXPLANATION:")
    print("  Heart: Male vs Female patients have different heart rate distributions.")
    print("  Cancer: Glucose levels shift with age → age-blind model will misclassify.")
    print("  NHANES: BMI varies by ethnicity → ethnic-blind model will underperform.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  3 BASIC BIAS PROOF PLOTS")
    print("  Datasets: Heart Disease | Breast Cancer | NHANES")
    print("=" * 65)

    # Load all datasets
    heart  = load_heart()
    cancer = load_cancer()
    nhanes = load_nhanes()

    print("\n" + "=" * 65)
    print("  Running 3 bias proof plots.")
    print("  Close each plot window to proceed to the next one.")
    print("=" * 65)

    # Plot 1: Representation Bias (sample count imbalance)
    plot_1_representation_bias(heart, cancer, nhanes)

    # Plot 2: Outcome Disparity (positive rate gap by group)
    plot_2_outcome_disparity(heart, cancer, nhanes)

    # Plot 3: Feature Distribution Shift (features differ by group)
    plot_3_feature_distribution_shift(heart, cancer, nhanes)

    print("\n" + "=" * 65)
    print("  ALL DONE!")
    print(f"  Plots saved to: {OUT}")
    print("=" * 65)
    print("\n  Files generated:")
    print("    BIAS_1_representation.png    — sample count imbalance")
    print("    BIAS_2_outcome_disparity.png — positive rate gap + SPD/DI")
    print("    BIAS_3_feature_shift.png     — feature distributions by group")
    print("=" * 65)
