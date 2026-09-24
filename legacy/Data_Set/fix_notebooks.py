"""
fix_notebooks.py
Patches the two existing notebooks to fix the bugs that halted execution.

Bug 1 — nd_preprocessing_and_integration.ipynb
  • Wrong base path: e:/Data_Set  ->  d:/Data_Set
  • KeyError: True in Heart Failure EHR cell:
      df.get('Is_Counterfactual', 0) == 0   ->   correct guard

Bug 2 — multi_dataset_integration.ipynb
  • max(miss_by_ds.values()+1e-6) — can't add float to dict_values
    ->  max(miss_by_ds.values()) + 1e-6 (with empty-dict guard)
"""

import json, copy, pathlib, sys

BASE = pathlib.Path(r"d:/Data_Set")

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [SAVED] {path}")

def patch_source(source_lines, old, new, *, label=""):
    """Replace first occurrence of `old` substring in any source line."""
    for i, line in enumerate(source_lines):
        if old in line:
            source_lines[i] = line.replace(old, new, 1)
            lbl = label or repr(old)[:60]
            print(f"    [PATCHED] {lbl}")
            return True
    return False

def patch_nb_cells(nb, patches):
    """
    patches = list of (old_str, new_str, label)
    Applies each patch to the first cell whose source contains old_str.
    Returns count of successful patches.
    """
    done = 0
    for old, new, label in patches:
        for cell in nb["cells"]:
            src = cell.get("source", [])
            if isinstance(src, list):
                joined = "".join(src)
                if old in joined:
                    patched = joined.replace(old, new, 1)
                    cell["source"] = [patched]
                    print(f"    [PATCHED] {label}")
                    done += 1
                    break
            elif isinstance(src, str):
                if old in src:
                    cell["source"] = src.replace(old, new, 1)
                    print(f"    [PATCHED] {label}")
                    done += 1
                    break
    return done


# ─────────────────────────────────────────────────────────────────
# 1.  nd_preprocessing_and_integration.ipynb
# ─────────────────────────────────────────────────────────────────
nd_nb_path = BASE / "nd_preprocessing_and_integration.ipynb"
print(f"\n{'='*60}")
print(f"Patching: {nd_nb_path.name}")
print(f"{'='*60}")

nd_nb = load_nb(nd_nb_path)

nd_patches = [
    (
        "BASE    = Path(r'e:/Data_Set')",
        "BASE    = Path(r'd:/Data_Set')",
        "Fix base path  e:/ -> d:/"
    ),
    (
        "df_real = df[df.get('Is_Counterfactual', 0) == 0].copy()",
        "df_real = df[df['Is_Counterfactual'] == 0].copy() if 'Is_Counterfactual' in df.columns else df.copy()",
        "Fix KeyError: True in Heart Failure EHR (Is_Counterfactual guard)"
    ),
]

n = patch_nb_cells(nd_nb, nd_patches)
print(f"  Total patches applied: {n}/{len(nd_patches)}")

# Also clear stale outputs & reset execution_count for ALL cells
# so the whole notebook can be re-run cleanly from top
for cell in nd_nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

save_nb(nd_nb, nd_nb_path)


# ─────────────────────────────────────────────────────────────────
# 2.  multi_dataset_integration.ipynb
# ─────────────────────────────────────────────────────────────────
mi_nb_path = BASE / "multi_dataset_integration.ipynb"
print(f"\n{'='*60}")
print(f"Patching: {mi_nb_path.name}")
print(f"{'='*60}")

mi_nb = load_nb(mi_nb_path)

mi_patches = [
    (
        # original broken line
        "colors2 = [plt.cm.RdYlGn_r(v/max(miss_by_ds.values()+1e-6)) for v in miss_by_ds.values()]",
        # fixed: safe guard + correct arithmetic
        "max_miss = max(miss_by_ds.values()) + 1e-6 if miss_by_ds else 1\n"
        "colors2 = [plt.cm.RdYlGn_r(v / max_miss) for v in miss_by_ds.values()]",
        "Fix  max(dict_values + float)  ->  max() + float  with empty-guard"
    ),
]

n = patch_nb_cells(mi_nb, mi_patches)
print(f"  Total patches applied: {n}/{len(mi_patches)}")

save_nb(mi_nb, mi_nb_path)

print("\n✅  All patches complete.\n")
print("Next steps:")
print("  1. Open  nd_preprocessing_and_integration.ipynb  and run all cells (Kernel → Restart & Run All)")
print("  2. Once finished, open  multi_dataset_integration.ipynb  and run all cells")
