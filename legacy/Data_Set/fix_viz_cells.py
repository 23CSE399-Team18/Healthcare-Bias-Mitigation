"""
fix_viz_cells.py
Fix visualization cells in multi_dataset_integration.ipynb that can
crash when a domain/dataset has no valid age data after dropna.

Bug: pd.cut(df_clean['age'], bins=10) raises ValueError: Cannot cut empty array
Fix: skip the plot for that domain if df_clean is empty.
"""
import json, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

FRAGMENT = "Cannot cut empty array"   # not in code -- use unique code fragment
FRAGMENT = "df_clean['age_bin'] = pd.cut(df_clean['age'], bins=10)"

fixed = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", ""))
    if FRAGMENT not in src:
        continue

    # Replace the whole age-target plot block with a safe version
    old_block = """\
    for i, (domain, df_d) in enumerate(domains_with_data[:n_plots]):
        df_clean = df_d.dropna(subset=['age','target'])
        df_clean = df_clean.copy()
        df_clean['age_bin'] = pd.cut(df_clean['age'], bins=10)
        grp = df_clean.groupby('age_bin')['target'].mean()
        axes[i].plot(range(len(grp)), grp.values, 'o-', color='steelblue', linewidth=2, markersize=5)
        axes[i].fill_between(range(len(grp)), grp.values, alpha=0.15, color='steelblue')
        axes[i].set_title(f'{domain.replace("_"," ").title()}', fontweight='bold')
        axes[i].set_xlabel('Age Bin Index'); axes[i].set_ylabel('Target Rate')
        axes[i].grid(True, alpha=0.3)"""

    new_block = """\
    for i, (domain, df_d) in enumerate(domains_with_data[:n_plots]):
        df_clean = df_d.dropna(subset=['age','target']).copy()
        if len(df_clean) < 10:  # skip if too few rows after dropna
            axes[i].text(0.5, 0.5, f'{domain}\\nNo age data', ha='center', va='center',
                         transform=axes[i].transAxes)
            continue
        try:
            df_clean['age_bin'] = pd.cut(df_clean['age'], bins=10)
            grp = df_clean.groupby('age_bin', observed=False)['target'].mean()
            axes[i].plot(range(len(grp)), grp.values, 'o-', color='steelblue', linewidth=2, markersize=5)
            axes[i].fill_between(range(len(grp)), grp.values, alpha=0.15, color='steelblue')
        except Exception as _ve:
            axes[i].text(0.5, 0.5, f'{domain}\\nPlot error', ha='center', va='center',
                         transform=axes[i].transAxes)
            continue
        axes[i].set_title(f'{domain.replace("_"," ").title()}', fontweight='bold')
        axes[i].set_xlabel('Age Bin Index'); axes[i].set_ylabel('Target Rate')
        axes[i].grid(True, alpha=0.3)"""

    if old_block in src:
        src = src.replace(old_block, new_block, 1)
        cell["source"] = [src]
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"  [PATCHED] Cell {i}: age_bin empty array guard")
        fixed += 1
    else:
        # Show what the loop looks like
        for ln in src.split("\n"):
            if "age_bin" in ln or "df_clean" in ln:
                print(f"  LINE: {repr(ln)}")

print(f"\nPatches applied: {fixed}")

# Clear all outputs
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("[SAVED]")
