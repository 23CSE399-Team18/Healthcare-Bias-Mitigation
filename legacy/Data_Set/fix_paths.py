import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"d:\Data_Set\multi_dataset_integration.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Fix Cell 4: wrong drive e:/ → d:/ ────────────────────────────────────────
nb["cells"][4]["source"] = [
    'BASE     = Path(r"d:/Data_Set")\n'
    'DS_DIR   = BASE / "Data_Set"\n'
    'ND_DIR   = BASE / "ND"\n'
    'OUT_DIR  = BASE / "integrated_output"\n'
    'PP_DIR   = OUT_DIR / "preprocessed"\n'
    'DOM_DIR  = OUT_DIR / "by_domain"\n'
    'PLOT_DIR = OUT_DIR / "plots"\n'
    "\n"
    "for d in [OUT_DIR, PP_DIR, DOM_DIR, PLOT_DIR]:\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    'print(f"BASE    : {BASE}")\n'
    'print(f"DS_DIR  : {DS_DIR}")\n'
    'print(f"ND_DIR  : {ND_DIR}")\n'
    'print(f"OUT_DIR : {OUT_DIR}")\n'
]
nb["cells"][4]["outputs"] = []
nb["cells"][4]["execution_count"] = None
print("Cell 4 fixed: e:/Data_Set -> d:/Data_Set")

# ── Fix any other cell still referencing e:/ ─────────────────────────────────
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "e:/Data_Set" in src or "e:\\Data_Set" in src:
        new_src = src.replace("e:/Data_Set", "d:/Data_Set").replace("e:\\Data_Set", "d:\\Data_Set")
        cell["source"] = [new_src]
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"  Also fixed Cell {i}")

# ── Find Stage 8 marker and insert quick-load cell right after it ─────────────
stage8_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Stage 8" in src and cell["cell_type"] == "markdown":
        stage8_idx = i
        break

quick_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n"
        "\n"
        "# Load full integrated data (868,874 rows x 362 columns)\n"
        'df = pd.read_csv(\n'
        '    r"d:\\Data_Set\\integrated_output\\full_integrated_dataset.csv",\n'
        '    low_memory=False\n'
        ')\n'
        "\n"
        'print("Shape:", df.shape)           # (868874, 362)\n'
        'print()\n'
        'print("Columns:", df.columns.tolist())\n'
        'print()\n'
        "df.head(10)                          # shows first 10 rows cleanly\n"
    ],
}

if stage8_idx is not None:
    # Remove old quick-load cell if it exists right after Stage 8
    nb["cells"].insert(stage8_idx + 1, quick_cell)
    print(f"Quick-load cell inserted at position {stage8_idx + 1}")
else:
    nb["cells"].append(quick_cell)
    print("Quick-load cell appended at end")

# ── Clear all outputs ─────────────────────────────────────────────────────────
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSaved. Notebook now has {len(nb['cells'])} cells.")
print("Ready — open in Jupyter and do Kernel -> Restart & Run All")
