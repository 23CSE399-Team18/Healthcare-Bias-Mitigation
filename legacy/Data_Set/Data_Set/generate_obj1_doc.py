"""
Generate Objective 1 — Complete Explanation as a Word Document (.docx)
"""
import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
OUT = os.path.join(BASE, "Objective1_Complete_Explanation.docx")

doc = Document()

# ── Styles ──
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)
    return h

def para(text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def bullet(text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    # Rows
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()  # spacing

def add_image_if_exists(path, caption, width=6.0):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph()
        run = p.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        para(f"[Image not found: {os.path.basename(path)}]", italic=True)


# ═══════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ═══════════════════════════════════════════════════════════════════

doc.add_paragraph()
doc.add_paragraph()
title = doc.add_heading('Objective 1 — Complete Explanation', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

subtitle = doc.add_heading(
    'Healthcare Bias Mitigation: A Multi-Method Approach '
    'to Addressing Bias Across Diverse Clinical Applications', level=2)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run("GROUP 18 — Project Review 1\n").bold = True
info.add_run("Students: E. RatanRaja · Datta K · Srija U · Sriya U\n")
info.add_run("Guides: Dr. Bagavathi Sivakumar P, Dr. Deepika T\n")

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: RESEARCH GAPS
# ═══════════════════════════════════════════════════════════════════

heading("1. Research Gaps — What Problems Does This Research Solve?")

para("Healthcare AI has four fundamental problems that no existing work has fully solved. "
     "These are the 'research gaps' — the holes in current research that our project fills.")

# RG1
heading("Research Gap 1 (RG1): Stage-Isolated Bias Handling", level=2)

para("The Problem:", bold=True)
para("Suppose you have a biased dataset (e.g., 80% male, 20% female heart patients). "
     "Existing solutions fix this at only one stage — for example, they might rebalance "
     "the data (pre-processing), OR add fairness constraints to the model (in-processing), "
     "OR adjust the predictions afterwards (post-processing). But they never do all three together.")

para("Why this is dangerous:", bold=True)
para("If you fix the data but your model still learns biased patterns, the output predictions "
     "are still unfair. Fixing one stage alone is like patching one hole in a sinking boat "
     "while ignoring the other two.")

para("Our solution:", bold=True)
para("We build a three-stage pipeline — Stage 1 (pre-processing: fix the data), "
     "Stage 2 (in-processing: train the model to be fair), "
     "Stage 3 (post-processing: calibrate the final predictions). "
     "All three stages work together so bias cannot escape.")

# RG2
heading("Research Gap 2 (RG2): Lack of Clinical Risk Awareness", level=2)

para("The Problem:", bold=True)
para("Current fairness metrics treat all errors equally. But in healthcare, a false negative "
     "for cancer (telling a cancer patient they're healthy) is catastrophically worse than a "
     "false positive (telling a healthy person to get more tests). Existing AI models don't "
     "understand this difference.")

para("Why this is dangerous:", bold=True)
para("A model could pass every statistical fairness test while simultaneously underdiagnosing "
     "cancer in women because the fairness metric doesn't weigh medical consequences.")

para("Our solution:", bold=True)
para("We add clinical risk-aware loss functions — the model is penalised more heavily for "
     "dangerous errors (missing a disease) than for safe errors (recommending extra screening).")

# RG3
heading("Research Gap 3 (RG3): Absence of Continuous Fairness Monitoring", level=2)

para("The Problem:", bold=True)
para("Existing studies test fairness once on old data and then declare success. But real-world "
     "data changes over time — new patient populations arrive, disease patterns shift, "
     "demographics evolve. A model that was fair in 2024 could become biased in 2025.")

para("Why this is dangerous:", bold=True)
para("Without monitoring, a deployed model could silently become unfair and no one would "
     "know until patients are harmed.")

para("Our solution:", bold=True)
para("We build an automated auditing module that continuously checks fairness metrics "
     "(SPD, DI, HAITI Trust Index) on live predictions and triggers alerts when fairness degrades.")

# RG4
heading("Research Gap 4 (RG4): Human–AI Interaction Neglect", level=2)

para("The Problem:", bold=True)
para("Even if an AI model is perfectly fair, the clinician using it might interpret its output "
     "with their own biases. For example, a doctor might trust a high-confidence AI prediction "
     "for male patients but second-guess it for female patients.")

para("Why this is dangerous:", bold=True)
para("The final clinical decision combines AI output + human judgement. If only the AI side "
     "is fair, the overall system can still produce biased outcomes.")

para("Our solution:", bold=True)
para("We include AI literacy training modules for clinicians, a dashboard with SHAP/LIME "
     "explanations, and patient-facing interfaces with fairness-aware disclaimers.")

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: OBJECTIVE 1
# ═══════════════════════════════════════════════════════════════════

heading("2. Objective 1 — Consistent Prediction Performance")

para("What Objective 1 Says:", bold=True, size=12)
para('"Design a healthcare AI framework that maintains consistent predictive performance '
     'across diverse patient populations to reduce unfair clinical decision outcomes."',
     italic=True)

para("What This Means in Simple Terms:", bold=True, size=12)
para("When we say 'consistent predictive performance', we mean:")
bullet("If the model correctly diagnoses heart disease 85% of the time for male patients, "
       "it should also correctly diagnose it approximately 85% of the time for female patients.")
bullet("If it predicts diabetes risk correctly for White patients, it should predict equally "
       "well for Black patients, Hispanic patients, Asian patients, etc.")
bullet("The model's accuracy should not depend on which demographic group the patient belongs to.")

para("Why It Matters:", bold=True, size=12)
para("Imagine a model that is 95% accurate overall — sounds great! But what if:")
bullet("It's 99% accurate for males and only 75% accurate for females")
bullet("It's 98% accurate for White patients and 60% for Black patients")
para("That overall 95% hides a dangerous disparity. Objective 1 ensures this cannot happen.")

para("Which Research Gaps It Addresses:", bold=True, size=12)
bullet("RG1 — ", "We ensure consistency by using a multi-stage pipeline (not just one fix)")
bullet("RG4 — ", "We make the consistency metrics visible to clinicians through dashboards")

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════

heading("3. Architecture Diagram — Objective 1")

add_image_if_exists(os.path.join(BASE, "obj1.png"),
                    "Figure 1: Architecture Diagram for Objective 1 — Consistent Prediction Performance",
                    width=6.5)

heading("Reading the Diagram — Step by Step", level=2)

heading("Left Side — Data Entry", level=3)
add_table(
    ["Component", "What it does"],
    [
        ["PATIENT DEMOGRAPHICS\n(pink box, top-left)",
         "Patient attributes: Age, Sex, Ethnicity, Socioeconomic Status.\n"
         "These are the sensitive attributes that the model must not be biased against."],
        ["EHR SYSTEMS\n(blue box, bottom-left)",
         "Electronic Health Records from hospitals.\nIncludes eICU and NHANES data sources."],
        ["Tabular Records",
         "Structured numeric data — lab values, vitals, diagnoses"],
        ["Clinical Notes",
         "Unstructured text data — doctor notes, discharge summaries"],
        ["▶ Start (blue oval)",
         "The entry point — data flows right from here"],
    ]
)

heading("Middle — Feature Normalization (Yellow Box)", level=3)
add_table(
    ["Component", "What it does"],
    [
        ["FEATURE NORMALIZATION",
         "Standardises all features so they're on the same scale (e.g., age 0–100 and "
         "cholesterol 100–400 both become 0–1). Critical because different hospitals may "
         "record values in different units or ranges."],
        ["Bias Source Decoder",
         "Identifies which features carry demographic bias. For example, if cholesterol "
         "values are systematically different for males vs females, this module detects and flags it."],
        ["Clinical Variable Standardization",
         "Maps different hospital coding systems (ICD-9 vs ICD-10) into a unified format."],
    ]
)

heading("Middle — Evaluation (Blue Box)", level=3)
add_table(
    ["Component", "What it does"],
    [
        ["Evaluate AUROC\nAcross Subgroups",
         "Computes the model's diagnosis accuracy (AUROC) separately for each demographic "
         "group — male vs female, each ethnicity, each age group.\nIf any group has "
         "significantly lower AUROC, the model is biased."],
        ["Evaluate TPR Gaps\n(green box, top)",
         "TPR = True Positive Rate = 'What percentage of sick patients does the model "
         "correctly identify?'\nIf TPR is 90% for males but only 60% for females, "
         "that's a TPR gap — the model is underdiagnosing women."],
    ]
)

heading("Middle-Bottom — Core AI Inference (Green Box)", level=3)
add_table(
    ["Component", "What it does"],
    [
        ["CORE AI INFERENCE",
         "This is where the actual disease prediction happens — the model takes patient "
         "data and outputs a clinical prediction."],
        ["Predict Clinical Outcome",
         "The model predicts: Does this patient have heart disease? Diabetes? Cancer?"],
        ["Generate Risk Probabilities",
         "Instead of just yes/no, the model outputs a probability (e.g., '78% chance of "
         "diabetes'). More useful clinically than a binary label."],
    ]
)

heading("Right Side — Decision & Feedback", level=3)
add_table(
    ["Component", "What it does"],
    [
        ["Fairness Thresholds\nBreached? (red diamond)",
         "A decision point: Are the fairness metrics (SPD, DI) within acceptable limits?"],
        ["→ NO path",
         "If fairness thresholds are met → proceed to Clinical Decision Output "
         "and Update Clinician Dashboard"],
        ["→ YES path",
         "If thresholds ARE breached → Trigger Disparity Alert (red box) → "
         "loop back to re-process"],
        ["Performance Consistency\nCheck (yellow diamond)",
         "Verifies that the model performs equally well across all subgroups before deployment."],
        ["Clinical Decision Output",
         "The final, fair prediction that goes to the clinician."],
        ["Generate Fair Clinical\nRecommendation (far right)",
         "Converts the model output into actionable medical advice."],
        ["Update Clinician\nDashboard (far right)",
         "Updates the real-time dashboard with new predictions and fairness metrics."],
    ]
)

para("Feedback Loop (dashed line):", bold=True)
para("The dashed line from 'Fairness Thresholds Breached' going back to 'Feature Normalization' "
     "is the feedback loop — if bias is detected, the system re-adjusts the feature processing "
     "and re-trains.")

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: INPUT / OUTPUT
# ═══════════════════════════════════════════════════════════════════

heading("4. Input and Output for Objective 1")

heading("What Are 'Input' and 'Output' in Research?", level=2)
para("Input = What you feed INTO the system — the raw materials it works with.\n"
     "Output = What the system produces — the results.\n\n"
     "Think of it like cooking: Input = ingredients, Output = the dish.")

heading("Objective 1 — Specific I/O", level=2)

add_table(
    ["", "Details"],
    [
        ["INPUT:\nPatient medical data",
         "Lab results (cholesterol, blood sugar, hemoglobin), vitals (blood pressure, heart rate), "
         "medical history (previous diagnoses), imaging results"],
        ["INPUT:\nDemographic attributes",
         "Age, Sex (male/female), Ethnicity (White, Black, Hispanic, Asian, etc.), "
         "Socioeconomic Status"],
        ["OUTPUT:\nFair disease prediction model",
         "A trained AI model that predicts diseases (heart disease, cancer, diabetes, CKD) "
         "with equal accuracy for every demographic group"],
        ["OUTPUT:\nBias evaluation results",
         "Quantified fairness metrics (SPD, DI, AUROC gaps) showing the model IS fair, with proof"],
        ["OUTPUT:\nPrivacy-preserving AI system",
         "The model can be deployed on local hospital servers without patient data leaving the hospital"],
    ]
)

para("Why This Specific Input-Output Matters:", bold=True)
para("The input includes demographic attributes NOT because we want the model to use them for "
     "prediction, but because we need to check whether the model is fair across these groups. "
     "The model should predict disease based on clinical features (cholesterol, blood pressure), "
     "not demographics (race, sex). But we need the demographics to audit whether predictions "
     "are equitable.")

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: ALGORITHMS
# ═══════════════════════════════════════════════════════════════════

heading("5. Algorithms Mapped to Objective 1")

para("Two main algorithms power Objective 1:")

heading("Algorithm 1: CXR-Foundation-Bias Algorithm", level=2)

para("What it is:", bold=True)
para("An ensemble learning algorithm where each new model in the sequence corrects the "
     "errors of the previous one.")

para("How it works — step by step:", bold=True)
bullet("Train Model A on the full dataset → it makes some errors")
bullet("Identify which patients Model A got WRONG — especially cases where it was wrong "
       "for a specific demographic group")
bullet("Train Model B specifically to fix those errors")
bullet("Combine Model A + Model B → fewer errors")
bullet("Repeat: Train Model C to fix the combined model's errors")
bullet("Final ensemble = A + B + C + ... → maximally accurate AND fair")

para("Why it helps Objective 1:", bold=True)
para("Each iteration specifically targets demographic groups where performance is weakest. "
     "If Model A underdiagnoses women, Model B will be trained to fix that. The ensemble "
     "naturally converges toward equal performance across all groups.")

para("Key insight: The algorithm learns disease-focused features (what does the disease look "
     "like clinically?) that are independent of sensitive attributes (the features work "
     "regardless of patient sex/race).", italic=True)

para("Reference: [3] Iqbal et al. (2025) — FairBias framework", italic=True, size=9)

heading("Algorithm 2: XGBoost (Extreme Gradient Boosting)", level=2)

para("What it is:", bold=True)
para("A powerful machine learning algorithm that combines hundreds of small decision "
     "trees into one strong predictor.")

para("How it works — step by step:", bold=True)
bullet("Build a simple decision tree: 'If cholesterol > 240 AND age > 50 → high risk'")
bullet("Check where this tree makes errors")
bullet("Build a second tree that focuses on those errors")
bullet("Combine both trees (weighted voting)")
bullet("Repeat for 100–500 trees")
bullet("Final prediction = weighted vote of all trees")

para("Why it helps Objective 1:", bold=True)
para("XGBoost is the state-of-the-art for tabular clinical data. It handles missing values "
     "(common in medical records), mixed feature types (numeric + categorical), class "
     "imbalance (few cancer patients vs many healthy), and feature interactions (cholesterol "
     "alone may be okay, but cholesterol + high blood pressure together = dangerous).")

para("Reference: [14] Shaheen et al. (2026) — Deep Residual Network with XGBoost", italic=True, size=9)

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════

heading("6. Methodology Mapped to Objective 1")

heading("Methodology 1: Domain-Specific Batch Normalization (DSBN)", level=2)

para("What is Batch Normalization?", bold=True)
para("When training a neural network, the values flowing through it can drift to extreme "
     "ranges (very large or very small numbers). Batch Normalization rescales them to a "
     "standard range (mean=0, std=1) after each layer.")

para("What makes DSBN 'Domain-Specific'?", bold=True)
para("In healthcare, data comes from multiple sources (Cleveland Heart clinic, Hungarian "
     "clinic, Swiss clinic, NHANES surveys, eICU hospitals). Each source has slightly "
     "different data characteristics — different equipment calibrations, different units, "
     "different patient populations.")

para("Standard BN computes one set of normalization statistics for ALL data mixed together. "
     "This hurts because it forces Cleveland and NHANES data into the same distribution.")

para("DSBN learns separate normalization parameters for each data source — Cleveland gets "
     "its own mean and std, NHANES gets its own, eICU gets its own.")

para("How this helps Objective 1:", bold=True)
para("Patient demographics often correlate with data source. By normalizing per source, DSBN "
     "prevents the model from learning shortcuts based on which hospital the data came from — "
     "shortcuts that would create demographic bias.")

para("Reference: [3] Iqbal et al. (2025)", italic=True, size=9)

heading("Methodology 2: Domain-Invariant Feature Extraction", level=2)

para("What it is:", bold=True)
para("Forces the model to learn medical features that are the same regardless of patient demographics.")

para("How it works:", bold=True)
para("Imagine the model learns that 'high cholesterol = heart disease risk'. In a biased "
     "dataset, it might also learn 'male = heart disease risk' because males are "
     "over-represented in heart disease data. Domain-Invariant Feature Extraction ensures:")
bullet("✅ The model CAN learn: 'high cholesterol → risk' (clinical feature)")
bullet("❌ The model CANNOT learn: 'male → risk' (demographic shortcut)")

para("Technically: During training, a second 'adversary' network tries to predict the "
     "patient's sex/race from the model's internal features. The main model is trained to "
     "make the adversary fail — meaning the internal features carry NO demographic "
     "information, only clinical information.")

para("Reference: [3] Iqbal et al. (2025)", italic=True, size=9)

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

heading("7. Preprocessing — What We Actually Did")

heading("7A. NHANES Preprocessing (nhanes_preprocessing.py)", level=2)

para("Input: 10 raw NHANES CSV files (demographics, blood pressure, body measures, "
     "glycohemoglobin, fasting glucose, insulin, total cholesterol, HDL cholesterol, "
     "CBC, albumin-creatinine)", bold=True)

add_table(
    ["Step", "What happens", "Why"],
    [
        ["Step 1:\nMerge",
         "Combines all 10 CSV files into one big table using Patient_ID as the link.\n"
         "Result: ~8,700 patients × 80+ features",
         "Raw NHANES data is split across many files;\nwe need everything in one place"],
        ["Step 2:\nRisk Labels",
         "Creates 4 clinical risk labels from real lab data:\n"
         "• Diabetes_Risk (HbA1c ≥ 6.5% OR Glucose ≥ 126)\n"
         "• Hypertension_Risk (BP ≥ 130/80)\n"
         "• CKD_Risk (Albumin-Creatinine > 30)\n"
         "• Obesity (BMI ≥ 30)",
         "These are the target variables the model\nwill predict — uses real medical\n"
         "thresholds from clinical guidelines"],
        ["Step 3:\nBias Detection\nBEFORE",
         "Computes SPD and DI by Gender (Male vs Female)\n"
         "and Ethnicity (White vs Black) for all 4 risk labels",
         "This is the baseline — how biased is\nthe raw data before we touch it?"],
        ["Step 4:\n7 Algorithms",
         "Applies all 7 pre-processing algorithms\n(detailed in table below)",
         "Fix the bias in the data"],
        ["Step 5:\nBias Detection\nAFTER",
         "Recomputes SPD and DI after mitigation",
         "Proves that our algorithms\nactually reduced bias"],
        ["Save",
         "Outputs NHANES_preprocessed.csv",
         "This clean, debiased dataset\ngoes to Stage 2"],
    ]
)

para("The 7 Algorithms applied (in order):", bold=True, size=12)

add_table(
    ["#", "Algorithm", "What it does on NHANES data"],
    [
        ["1", "C-Chart DI\nImputation",
         "For each column with missing values, tests 3 imputation strategies (mean, median, mode) "
         "and picks whichever produces the most fair (DI closest to 1.0) result for Gender."],
        ["2", "MICE",
         "Fills remaining missing values by treating each incomplete column as a dependent "
         "variable and predicting it from all other columns. Preserves inter-variable relationships."],
        ["3", "Reweighing",
         "Assigns a Sample_Weight to each patient. Underrepresented groups get higher weights, "
         "overrepresented groups get lower weights."],
        ["4", "Disparate Impact\nRemover",
         "Shifts feature distributions to reduce correlation with Gender. Moves group distributions "
         "closer to the global median (controlled by repair_level=0.8)."],
        ["5", "GAN\nAugmentation",
         "If a demographic group has too few positive cases, generates synthetic patients by "
         "sampling real ones + adding small Gaussian noise."],
        ["6", "ProWSyn",
         "Generates synthetic minority-class samples near the decision boundary using "
         "proximity-weighted interpolation between nearest neighbours."],
        ["7", "Counterfactual\nGeneration",
         "Creates 'what-if' copies: take every male patient, flip gender to female (keep all "
         "clinical features identical), and vice versa. Tests whether predictions would change "
         "based on gender alone."],
    ]
)

heading("7B. Multi-Dataset Preprocessing (multi_dataset_preprocessing.py)", level=2)

para("This script applies the exact same 7 algorithms to ALL 5 datasets, "
     "with dataset-specific configurations:", bold=True)

add_table(
    ["Dataset", "Sensitive Attribute", "Target Variable", "Specifics"],
    [
        ["Breast Cancer\nCoimbra", "Age_Group\n(median split)",
         "Cancer_Risk\n(1=Cancer, 0=Healthy)", "Only 116 patients —\nsensitive to augmentation"],
        ["CDC Diabetes", "Sex\n(0=Female, 1=Male)",
         "Diabetes_binary", "70k patients —\nsampled to 10k for speed"],
        ["CKD (UCI)", "Age_Group\n(median split)",
         "CKD_Risk", "ARFF format, ? as missing\nvalues, needs special parsing"],
        ["eICU", "Gender_Encoded",
         "Mortality\n(0=Alive, 1=Expired)", "Age column has '> 89'\nstring → converted to 90"],
        ["Heart Disease", "sex\n(0=Female, 1=Male)",
         "Heart_Disease", "4 clinic sites merged;\ntarget binarised from 0-4"],
    ]
)

doc.add_page_break()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: BEFORE / AFTER PLOTS
# ═══════════════════════════════════════════════════════════════════

heading("8. Before vs After Preprocessing — What Do the Plots Tell Us?")

para("The visualisation suite generates 5 plot types (Swarm, Violin, Interactive, Strip, Pair), "
     "each run twice: once on raw data (BEFORE) and once on preprocessed data (AFTER). "
     "This lets you visually verify that preprocessing improved the data.")

heading("Swarm Plot — BEFORE Preprocessing", level=2)
add_image_if_exists(os.path.join(PLOTS, "1_swarm_BEFORE.png"),
                    "Figure 2: Swarm Plot — BEFORE Preprocessing (raw data)", width=6.2)

heading("Swarm Plot — AFTER Preprocessing", level=2)
add_image_if_exists(os.path.join(PLOTS, "1_swarm_AFTER.png"),
                    "Figure 3: Swarm Plot — AFTER Preprocessing (cleaned data)", width=6.2)

heading("What to Look For (BEFORE → AFTER):", level=3)

para("Heart Disease panel (left):", bold=True)
bullet("BEFORE: Cholesterol values may show a wider, more irregular spread for one sex. "
       "Extreme outliers (>400 mg/dl) concentrated in one demographic group.")
bullet("AFTER: The Disparate Impact Remover shifts group distributions closer together. "
       "Spread is more symmetrical between Male and Female.")

para("Breast Cancer panel (middle):", bold=True)
bullet("BEFORE: Raw BMI and Age values show natural groupings. Cancer (red) vs Healthy (green) overlap.")
bullet("AFTER: GAN augmentation adds synthetic patients to underrepresented groups. "
       "You may see more balanced red dots in age/BMI ranges that had few cancer cases.")

para("NHANES panel (right):", bold=True)
bullet("BEFORE: Raw age distribution by gender may be asymmetric if NHANES surveyed more older males.")
bullet("AFTER: Augmentation algorithms add synthetic records. Distribution shows more symmetric "
       "age coverage across genders.")

heading("Violin Plot — BEFORE Preprocessing", level=2)
add_image_if_exists(os.path.join(PLOTS, "2_violin_BEFORE.png"),
                    "Figure 4: Violin Plot — BEFORE Preprocessing", width=6.2)

heading("Violin Plot — AFTER Preprocessing", level=2)
add_image_if_exists(os.path.join(PLOTS, "2_violin_AFTER.png"),
                    "Figure 5: Violin Plot — AFTER Preprocessing", width=6.2)

para("How to read violin shapes:", bold=True)
bullet("A fat section means many patients have values in that range")
bullet("A thin section means few patients")
bullet("BEFORE: Male and Female violins may have very different shapes")
bullet("AFTER: Violins should become more similar in shape between demographic groups")

heading("Summary — What Before/After Proves", level=2)

add_table(
    ["Observation", "Before", "After", "What This Proves"],
    [
        ["Feature\ndistributions",
         "Different shapes per\ndemographic group",
         "More similar shapes",
         "DI Remover +\nReweighing worked"],
        ["Outliers",
         "Concentrated in\nspecific groups",
         "More evenly\ndistributed",
         "Imputation filled missing\nvalues that caused artifacts"],
        ["Sample sizes\nper group",
         "Unequal\n(e.g., 3x more males)",
         "More balanced",
         "GAN + ProWSyn\naugmentation worked"],
        ["Decision\nboundaries",
         "Blurred for\nminority groups",
         "Sharper",
         "ProWSyn generated\nboundary-aware samples"],
    ]
)

para("IMPORTANT: The goal is NOT to make all groups identical — biological differences are "
     "real (men and women genuinely have different average cholesterol levels). The goal is "
     "to ensure the model's ability to predict disease is equally good for every group, "
     "not that every group looks the same.", bold=True)


# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════

doc.save(OUT)
print(f"\n{'='*60}")
print(f"  Word document saved: {OUT}")
print(f"{'='*60}")
