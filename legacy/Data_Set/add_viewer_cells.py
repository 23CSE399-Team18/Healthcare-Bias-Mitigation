import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"d:\Data_Set\multi_dataset_integration.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}

cells = [

    md("---\n## 👁️ Stage 8: View Full Integrated Data\nRun cells below to see all 868,874 rows across all 11 datasets."),

    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "# Remove ALL display limits so you see the FULL data\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.max_rows', None)\n"
        "pd.set_option('display.width', None)\n"
        "pd.set_option('display.max_colwidth', None)\n"
        "pd.set_option('display.float_format', '{:.4f}'.format)\n"
        "\n"
        "print('Loading full integrated dataset...')\n"
        "df = pd.read_csv(\n"
        "    r'd:\\Data_Set\\integrated_output\\full_integrated_dataset.csv',\n"
        "    low_memory=False\n"
        ")\n"
        "print(f'Loaded: {len(df):,} rows  x  {df.shape[1]} columns')\n"
        "print(f'Datasets: {sorted(df[\"dataset_source\"].unique())}')\n"
        "print(f'Domains : {sorted(df[\"domain\"].unique())}')\n"
    ),

    md("### 8.1 — All Columns in the Dataset"),

    code(
        "# Show every column name and data type\n"
        "col_info = pd.DataFrame({\n"
        "    'Column'  : df.columns,\n"
        "    'Dtype'   : df.dtypes.values,\n"
        "    'NonNull%': (df.notna().mean() * 100).round(1).values,\n"
        "    'Sample'  : [str(df[c].dropna().iloc[0]) if df[c].notna().any() else 'N/A'\n"
        "                 for c in df.columns]\n"
        "})\n"
        "print(f'Total columns: {len(col_info)}')\n"
        "display(col_info)\n"
    ),

    md("### 8.2 — Records per Dataset (All 11)"),

    code(
        "counts = df['dataset_source'].value_counts()\n"
        "print('Records per Dataset:')\n"
        "print('-' * 55)\n"
        "for ds, cnt in counts.items():\n"
        "    pct = cnt / len(df) * 100\n"
        "    bar = '#' * int(pct / 2)\n"
        "    print(f'  {ds:<30} {cnt:>8,}  ({pct:5.1f}%)  {bar}')\n"
        "print('-' * 55)\n"
        "print(f'  {\"TOTAL\":<30} {len(df):>8,}  (100.0%)')\n"
    ),

    md("### 8.3 — Records per Clinical Domain"),

    code(
        "for dom, cnt in df['domain'].value_counts().items():\n"
        "    pct = cnt / len(df) * 100\n"
        "    bar = '#' * int(pct / 2)\n"
        "    print(f'  {dom:<25} {cnt:>8,}  ({pct:5.1f}%)  {bar}')\n"
    ),

    md("### 8.4 — Full Data: All Rows for Every Dataset"),

    code(
        "# Shows ALL rows for EACH dataset with only its real columns (no empty cols)\n"
        "for ds in df['dataset_source'].unique():\n"
        "    sub = df[df['dataset_source'] == ds].reset_index(drop=True)\n"
        "    # Keep only columns that have more than 1 unique value for this dataset\n"
        "    real_cols = (['dataset_source', 'domain', 'target', 'age', 'age_group', 'gender']\n"
        "                 + [c for c in sub.columns\n"
        "                    if c not in ['dataset_source','domain','target','age','age_group','gender']\n"
        "                    and sub[c].nunique() > 1])\n"
        "    sub = sub[real_cols]\n"
        "    print()\n"
        "    print('=' * 80)\n"
        "    print(f'  DATASET: {ds}  |  {len(sub):,} rows  x  {len(real_cols)} columns')\n"
        "    print('=' * 80)\n"
        "    with pd.option_context('display.max_rows', None,\n"
        "                           'display.max_columns', None,\n"
        "                           'display.width', None):\n"
        "        display(sub)\n"
    ),

    md("### 8.5 — Core Schema: All 868,874 Rows (6 Key Columns)"),

    code(
        "# Full view of all records — 6 unified columns only\n"
        "schema = ['dataset_source', 'domain', 'target', 'age', 'age_group', 'gender']\n"
        "core   = df[schema].reset_index(drop=True)\n"
        "print(f'Core dataset: {core.shape}')\n"
        "with pd.option_context('display.max_rows', None,\n"
        "                       'display.max_columns', None,\n"
        "                       'display.width', None):\n"
        "    display(core)\n"
    ),

    md("### 8.6 — Statistical Summary per Dataset"),

    code(
        "summary_rows = []\n"
        "for ds in df['dataset_source'].unique():\n"
        "    sub = df[df['dataset_source'] == ds]\n"
        "    summary_rows.append({\n"
        "        'Dataset'       : ds,\n"
        "        'Domain'        : sub['domain'].iat[0],\n"
        "        'Rows'          : len(sub),\n"
        "        'Columns'       : sub.shape[1],\n"
        "        'Target_Rate_%' : round(sub['target'].mean() * 100, 2),\n"
        "        'Age_Mean'      : round(sub['age'].mean(), 1),\n"
        "        'Age_Min'       : round(sub['age'].min(), 1),\n"
        "        'Age_Max'       : round(sub['age'].max(), 1),\n"
        "        'Male_%'        : round((sub['gender'] == 'Male').mean() * 100, 1),\n"
        "        'Female_%'      : round((sub['gender'] == 'Female').mean() * 100, 1),\n"
        "        'Unknown_%'     : round((sub['gender'] == 'Unknown').mean() * 100, 1),\n"
        "    })\n"
        "stat_df = pd.DataFrame(summary_rows)\n"
        "with pd.option_context('display.max_rows', None, 'display.max_columns', None,\n"
        "                       'display.width', None):\n"
        "    display(stat_df)\n"
    ),

    md("### 8.7 — Filter & View Any Domain (Change DOMAIN below)"),

    code(
        "# Change this to any domain you want:\n"
        "# Options: 'icu', 'cardiovascular', 'metabolic', 'ecg',\n"
        "#          'dermatology', 'population_health', 'renal', 'oncology'\n"
        "DOMAIN = 'cardiovascular'\n"
        "\n"
        "subset    = df[df['domain'] == DOMAIN].reset_index(drop=True)\n"
        "real_cols = (['dataset_source','domain','target','age','age_group','gender']\n"
        "             + [c for c in subset.columns\n"
        "                if c not in ['dataset_source','domain','target','age','age_group','gender']\n"
        "                and subset[c].nunique() > 1])\n"
        "subset    = subset[real_cols]\n"
        "print(f'Domain [{DOMAIN}]: {len(subset):,} rows  x  {len(real_cols)} columns')\n"
        "with pd.option_context('display.max_rows', None, 'display.max_columns', None,\n"
        "                       'display.width', None):\n"
        "    display(subset)\n"
    ),

    md("### 8.8 — Search Any Specific Dataset"),

    code(
        "# Change DATASET_NAME to any of the 11 datasets:\n"
        "# UCI_HeartDisease, CKD_UCI, eICU_original, NHANES, CDC_Diabetes,\n"
        "# BreastCancer_Coimbra, MIMIC_IV, eICU_CRD_v2, PTB_XL_ECG, HAM10000, HeartFailure_EHR\n"
        "DATASET_NAME = 'MIMIC_IV'\n"
        "\n"
        "ds_df     = df[df['dataset_source'] == DATASET_NAME].reset_index(drop=True)\n"
        "real_cols = [c for c in ds_df.columns if ds_df[c].nunique() > 1\n"
        "             or c in ['dataset_source','domain','target','age','age_group','gender']]\n"
        "ds_df     = ds_df[real_cols]\n"
        "print(f'Dataset [{DATASET_NAME}]: {len(ds_df):,} rows  x  {len(real_cols)} columns')\n"
        "with pd.option_context('display.max_rows', None, 'display.max_columns', None,\n"
        "                       'display.width', None):\n"
        "    display(ds_df)\n"
    ),
]

# Remove duplicate Stage 8 if already added before
nb["cells"] = [c for c in nb["cells"]
               if "Stage 8" not in "".join(c.get("source", []))]

# Append all Stage 8 cells
nb["cells"].extend(cells)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — notebook now has {len(nb['cells'])} cells")
print(f"Stage 8 added: {len(cells)} new cells for full data viewing")
