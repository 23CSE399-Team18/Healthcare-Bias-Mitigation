"""
scan_fragile.py
Scan multi_dataset_integration.ipynb for other common fragile patterns
that could cause crashes downstream, so we can fix them all at once.
"""
import json, re, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

issues = []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", ""))

    # 1. pd.cut without empty guard
    if "pd.cut(" in src and "len(" not in src and "empty" not in src:
        for ln in src.split("\n"):
            if "pd.cut(" in ln:
                issues.append(f"Cell {i} [pd.cut no guard]: {ln.strip()}")

    # 2. .iloc[0] or [0] on mode result
    if "mode()[0]" in src or "mode().iloc[0]" not in src and "mode()" in src:
        pass  # already fixed

    # 3. groupby without observed= (FutureWarning -> error in newer pandas)
    if "groupby(" in src and "observed=" not in src:
        for ln in src.split("\n"):
            if "groupby(" in ln and "observed=" not in ln and not ln.strip().startswith("#"):
                issues.append(f"Cell {i} [groupby no observed]: {ln.strip()[:80]}")

    # 4. .values on empty groupby result
    if ".values, 'o-'" in src:
        issues.append(f"Cell {i}: plot with .values could fail if groupby empty")

print(f"Found {len(issues)} potential issues:\n")
seen = set()
for iss in issues:
    key = iss[:60]
    if key not in seen:
        print(f"  {iss}")
        seen.add(key)
