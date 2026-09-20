"""
fix_mimic_agg.py
Fix: x.mode()[0] -> safe_mode(x) in MIMIC-IV aggregation dict.
When all values in a group are NaN, mode() returns an empty Series
and [0] raises KeyError: 0. Use .iloc[0] with a length guard.
"""
import json, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

MIMIC_FRAGMENT = "preprocess_mimic_iv"

fixed = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", ""))
    if MIMIC_FRAGMENT not in src:
        continue

    # Replace the unsafe lambda pattern with a safe helper
    old = "lambda x: x.mode()[0] if len(x) else np.nan"
    new = "lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan"
    if old in src:
        src = src.replace(old, new)
        cell["source"] = [src]
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"  [PATCHED] Cell {i}: MIMIC-IV mode() safe lambda")
        fixed += 1
        break

if not fixed:
    # Try alternate form
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", ""))
        if "mode()[0]" in src:
            src = src.replace("mode()[0]", "mode().iloc[0] if len(x.mode()) > 0 else np.nan")
            cell["source"] = [src]
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"  [PATCHED] Cell {i}: mode()[0] -> safe (alt)")
            fixed += 1

print(f"  Patches applied: {fixed}")

# Clear all code cells
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("[SAVED]")
