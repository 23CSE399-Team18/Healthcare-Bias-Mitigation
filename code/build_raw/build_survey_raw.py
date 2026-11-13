"""Build model-ready Heart (BRFSS 2022), Hypertension (NHIS 2023) and Diabetes
(NHANES 2021-2023) files from the verified CDC sources.

Column convention used in every *_model_ready.csv (also respiratory):
  meta_*  identifiers, survey weights/design, domain or split-group columns.
          Never model inputs.
  sens_*  harmonised fairness attributes for evaluation: sex, age group,
          race/ethnicity (White / Black / Hispanic / Other) plus native detail.
          Not inputs by default (age and sex also appear as input features).
  target  binary outcome (meaning in schema.json).
  every other column is a candidate predictor.

Survey "don't know / refused / not ascertained" codes become missing. Structural
skips with a known answer are recoded (e.g. no drinks in 30 days -> 0 drinks per
occasion). Nothing is imputed, scaled, one-hot encoded or outlier-filtered:
those are the project's pre-processing experiments.

Run: python code/build_raw/build_survey_raw.py   (writes <dataset>/raw/<dataset>_raw.csv)
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]   # D:/complete

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AGE_BINS = [17, 39, 64, 79, 200]
AGE_LABELS = ["18-39", "40-64", "65-79", "80+"]


def age_group(age):
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(object)


def keep(series, valid):
    """Return series where value is in `valid` (iterable) else NaN."""
    return series.where(series.isin(list(valid)))


def in_range(series, low, high):
    return series.where(series.between(low, high))


def yes_no(series, yes=1, no=2):
    return series.map({yes: 1.0, no: 0.0})


def finish(df, name, target_meaning, source, extra_excluded):
    features = [c for c in df.columns if not c.startswith(("meta_", "sens_")) and c != "target"]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in features if c not in categorical]
    if df.target.isna().any() or not set(df.target.unique()) <= {0, 1}:
        raise ValueError(f"{name}: target must be binary with no missing values")
    path = ROOT / name / "raw" / f"{name}_raw.csv"
    if not df.meta_record_id.is_unique:
        raise ValueError(f"{name}: meta_record_id must identify one record")
    df.to_csv(path, index=False)
    print(f"{name}: rows={len(df)} features={len(features)} positive={int(df.target.sum())}")
    return {
        "file": path.name,
        "source": source,
        "rows": int(len(df)),
        "positive": int(df.target.sum()),
        "positive_rate": round(float(df.target.mean()), 4),
        "target_meaning": target_meaning,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "sensitive_attributes": [c for c in df.columns if c.startswith("sens_")],
        "meta_columns_never_inputs": [c for c in df.columns if c.startswith("meta_")],
        "source_columns_excluded": extra_excluded,
        "feature_missing_rate": {f: round(float(df[f].isna().mean()), 4) for f in features},
    }


def build_heart():
    raw = pd.read_csv(ROOT / "heart" / "raw" / "source" / "dataset_verified" / "heart_2022_baseline_raw.csv")
    out = pd.DataFrame(index=raw.index)
    # SEQNO is only unique within a state, so the record key is "<state FIPS>-<SEQNO>".
    out["meta_record_id"] = raw._STATE.astype(int).astype(str) + "-" + raw.SEQNO.astype("int64").astype(str)
    out["meta_domain_state"] = raw._STATE
    out["meta_survey_weight"] = raw._LLCPWT
    out["meta_strata"] = raw._STSTR
    out["meta_psu"] = raw._PSU

    out["sens_sex"] = raw.SEXVAR.map({1: "Male", 2: "Female"})
    out["sens_age_group"] = age_group(raw._AGE80)
    out["sens_race_ethnicity"] = raw._RACEGR4.map({1: "White", 2: "Black", 3: "Other", 4: "Other", 5: "Hispanic"})
    out["sens_race_detail"] = raw._RACEGR4.map({1: "White NH", 2: "Black NH", 3: "Other race only NH",
                                                4: "Multiracial NH", 5: "Hispanic"})

    out["age"] = raw._AGE80                                   # 80 = 80+
    out["male"] = yes_no(raw.SEXVAR, 1, 2)
    out["bmi"] = raw._BMI5 / 100
    out["smoking_status"] = raw._SMOKER3.map({1: "current_daily", 2: "current_someday", 3: "former", 4: "never"})
    out["any_exercise_30d"] = yes_no(raw.EXERANY2)
    out["sleep_hours"] = in_range(raw.SLEPTIM1, 1, 24)
    out["general_health"] = in_range(raw.GENHLTH, 1, 5)       # 1 excellent .. 5 poor
    out["poor_physical_health_days"] = in_range(raw.PHYSHLTH.replace(88, 0), 0, 30)
    out["poor_mental_health_days"] = in_range(raw.MENTHLTH.replace(88, 0), 0, 30)
    out["diabetes_status"] = raw.DIABETE4.map({1: "diabetes", 2: "gestational_only", 3: "no", 4: "prediabetes"})
    out["education"] = in_range(raw.EDUCA, 1, 6)
    out["income_category"] = in_range(raw.INCOME3, 1, 11)
    out["cost_barrier_12m"] = yes_no(raw.MEDCOST1)
    out["checkup_recency"] = in_range(raw.CHECKUP1.replace(8, 5), 1, 5)  # 5 = never
    alc = raw.ALCDAY4
    days = pd.Series(np.nan, index=raw.index)
    days[alc.between(101, 107)] = (alc[alc.between(101, 107)] - 100) * 30 / 7
    days[alc.between(201, 230)] = alc[alc.between(201, 230)] - 200
    days[alc == 888] = 0
    out["alcohol_days_30d"] = days
    drinks = in_range(raw.AVEDRNK3, 1, 76)
    drinks[(raw.AVEDRNK3 == 88) | (alc == 888)] = 0
    out["drinks_per_occasion"] = drinks
    out["target"] = raw.HeartDisease.astype(int)
    return finish(
        out, "heart",
        "Ever told had heart attack (MI) or angina/coronary heart disease (_MICHD); self-reported history",
        "CDC BRFSS 2022 https://www.cdc.gov/brfss/annual_data/annual_2022.html",
        {"_MICHD, CVDINFR4, CVDCRHD4": "label sources",
         "SMOKE100, SMOKDAY2": "already summarised in smoking_status (_SMOKER3); SMOKDAY2 is a structural skip",
         "_BMI5CAT": "redundant with bmi",
         "_RACEGR4": "fairness attribute only (sens_race_*)"})


def build_hypertension():
    base = ROOT / "hypertension" / "raw" / "source" / "raw" / "adult23.csv"
    cols = ["HHX", "REGION", "URBRRL", "WTFA_A", "PSTRAT", "PPSU", "HYPEV_A", "AGEP_A", "SEX_A", "HISPALLP_A",
            "EDUCP_A", "BMICAT_A", "WEIGHTLBTC_A", "HEIGHTTC_A", "SMKCIGST_A", "SMKECIGST_A", "PHSTAT_A",
            "CHLEV_A", "DIBEV_A", "PREDIB_A", "CHDEV_A", "MIEV_A", "ANGEV_A", "STREV_A", "COPDEV_A", "ASEV_A",
            "ARTHEV_A", "CANEV_A", "DEPEV_A", "ANXEV_A", "PHQ2SCREEN_A", "GAD2SCREEN_A", "MARITAL_A",
            "POVRATTC_A", "NOTCOV_A", "NATUSBORN_A", "EMPLASTWK_A", "FDSCAT4_A", "USUALPL_A"]
    raw = pd.read_csv(base, usecols=cols)
    raw = raw[raw.HYPEV_A.isin([1, 2])].reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)
    out["meta_record_id"] = raw.HHX
    out["meta_domain_region"] = raw.REGION.map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"})
    out["meta_survey_weight"] = raw.WTFA_A
    out["meta_strata"] = raw.PSTRAT
    out["meta_psu"] = raw.PPSU

    age = in_range(raw.AGEP_A, 18, 85)                         # 85 = 85+
    out["sens_sex"] = raw.SEX_A.map({1: "Male", 2: "Female"})
    out["sens_age_group"] = age_group(age)
    out["sens_race_ethnicity"] = raw.HISPALLP_A.map({1: "Hispanic", 2: "White", 3: "Black", 4: "Other",
                                                     5: "Other", 6: "Other", 7: "Other"})
    out["sens_race_detail"] = raw.HISPALLP_A.map({1: "Hispanic", 2: "White NH", 3: "Black NH", 4: "Asian NH",
                                                  5: "AIAN NH", 6: "AIAN and other NH", 7: "Other/multiple NH"})

    out["age"] = age
    out["male"] = yes_no(raw.SEX_A, 1, 2)
    out["education"] = in_range(raw.EDUCP_A, 0, 10)
    out["bmi_category"] = in_range(raw.BMICAT_A, 1, 4)         # 1 under .. 4 obese
    out["weight_lb"] = in_range(raw.WEIGHTLBTC_A, 100, 299)    # public-use top/bottom coded
    out["height_in"] = in_range(raw.HEIGHTTC_A, 59, 76)        # public-use top/bottom coded
    out["smoking_status"] = raw.SMKCIGST_A.map({1: "current_daily", 2: "current_someday", 3: "former", 4: "never"})
    out["ecig_status"] = raw.SMKECIGST_A.map({1: "current", 2: "former", 3: "never"})
    out["general_health"] = in_range(raw.PHSTAT_A, 1, 5)       # 1 excellent .. 5 poor
    for src, dst in [("CHLEV_A", "high_cholesterol_ever"), ("DIBEV_A", "diabetes_ever"),
                     ("PREDIB_A", "prediabetes_ever"), ("CHDEV_A", "coronary_heart_disease_ever"),
                     ("MIEV_A", "heart_attack_ever"), ("ANGEV_A", "angina_ever"), ("STREV_A", "stroke_ever"),
                     ("COPDEV_A", "copd_ever"), ("ASEV_A", "asthma_ever"), ("ARTHEV_A", "arthritis_ever"),
                     ("CANEV_A", "cancer_ever"), ("DEPEV_A", "depression_ever"), ("ANXEV_A", "anxiety_ever"),
                     ("PHQ2SCREEN_A", "phq2_positive"), ("GAD2SCREEN_A", "gad2_positive"),
                     ("NATUSBORN_A", "us_born"), ("EMPLASTWK_A", "worked_last_week")]:
        out[dst] = yes_no(raw[src])
    out["marital_status"] = raw.MARITAL_A.map({1: "married", 2: "cohabiting", 3: "neither"})
    out["income_poverty_ratio"] = in_range(raw.POVRATTC_A, 0, 11)   # 11 = 11+
    out["uninsured"] = raw.NOTCOV_A.map({1: 1.0, 2: 0.0})
    out["food_insecurity"] = in_range(raw.FDSCAT4_A, 1, 4)     # 1 high security .. 4 very low
    out["usual_place_of_care"] = raw.USUALPL_A.map({1: 1.0, 3: 1.0, 2: 0.0})
    out["urbanicity"] = in_range(raw.URBRRL, 1, 4)             # 1 large central metro .. 4 nonmetro
    out["target"] = yes_no(raw.HYPEV_A).astype(int)
    return finish(
        out, "hypertension",
        "Ever told by a doctor/health professional of hypertension (HYPEV_A); self-reported history",
        "CDC NHIS 2023 Sample Adult https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html",
        {"HYPEV_A": "label source",
         "HYP12M_A, HYPDIF_A, HYPMED_A": "asked only of people with hypertension (label leakage)",
         "BPLAST_A, LASTDR_A": "care-contact timing strongly driven by a known diagnosis",
         "RATCAT_A": "redundant with income_poverty_ratio",
         "HISPALLP_A": "fairness attribute only (sens_race_*)"})


def build_diabetes():
    raw = pd.read_csv(ROOT / "diabetes" / "raw" / "source" / "diabetes_baseline_raw.csv")
    out = pd.DataFrame(index=raw.index)
    out["meta_record_id"] = raw.SEQN
    out["meta_survey_weight_interview"] = raw.WTINT2YR
    out["meta_survey_weight_exam"] = raw.WTMEC2YR
    out["meta_strata"] = raw.SDMVSTRA
    out["meta_psu"] = raw.SDMVPSU
    out["meta_examined"] = (raw.RIDSTATR == 2).astype(int)

    out["sens_sex"] = raw.RIAGENDR.map({1: "Male", 2: "Female"})
    out["sens_age_group"] = age_group(raw.RIDAGEYR)
    out["sens_race_ethnicity"] = raw.RIDRETH3.map({1: "Hispanic", 2: "Hispanic", 3: "White", 4: "Black",
                                                   6: "Other", 7: "Other"})
    out["sens_race_detail"] = raw.RIDRETH3.map({1: "Mexican American", 2: "Other Hispanic", 3: "White NH",
                                                4: "Black NH", 6: "Asian NH", 7: "Other/multiracial NH"})

    out["age"] = raw.RIDAGEYR                                  # 80 = 80+
    out["male"] = yes_no(raw.RIAGENDR, 1, 2)
    out["education"] = raw.education_level                     # 1 <9th grade .. 5 college+
    out["income_poverty_ratio"] = raw.INDFMPIR                 # 5 = 5+
    out["bmi"] = raw.BMXBMI
    out["waist_cm"] = raw.BMXWAIST
    out["smoking_status"] = raw.smoking_status
    out["target"] = raw.diabetes_diagnosed.astype(int)
    return finish(
        out, "diabetes",
        "Doctor ever told you have diabetes (DIQ010; pregnancy-only excluded by wording); self-reported history",
        "CDC NHANES Aug 2021-Aug 2023 https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023",
        {"DIQ010": "label source",
         "LBXGH, LBXGLU": "HbA1c/glucose are diagnostic criteria; kept only in diabetes_biomarkers_adults.csv",
         "DMDEDUC2, SMQ020, SMQ040": "raw sources of education/smoking_status",
         "BMXWT, BMXHT": "redundant with bmi",
         "RIDEXPRG": "pregnancy status, restricted ages; not a baseline feature",
         "RIDRETH3": "fairness attribute only (sens_race_*)"})


def main():
    schema = {
        "column_convention": {
            "meta_*": "identifiers, survey weights/design, domain and split-group columns; never model inputs",
            "sens_*": "harmonised fairness attributes for evaluation; not inputs by default",
            "target": "binary outcome",
            "other": "candidate predictors (categorical = text columns, encode inside your pipeline)",
        },
        "harmonised_groups": {"sens_age_group": AGE_LABELS,
                              "sens_race_ethnicity": ["White", "Black", "Hispanic", "Other"],
                              "sens_sex": ["Male", "Female"]},
        "datasets": {},
    }
    schema["datasets"]["heart"] = build_heart()
    schema["datasets"]["hypertension"] = build_hypertension()
    schema["datasets"]["diabetes"] = build_diabetes()
    for name, record in schema["datasets"].items():
        path = ROOT / name / "raw" / f"{name}_schema.json"
        path.write_text(json.dumps({**{k: v for k, v in schema.items() if k != "datasets"}, name: record}, indent=2), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
