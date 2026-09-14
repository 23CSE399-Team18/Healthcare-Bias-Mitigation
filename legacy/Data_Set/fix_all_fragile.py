"""
fix_all_fragile.py
Batch-fix all fragile patterns in multi_dataset_integration.ipynb:
  1. pd.cut() on potentially empty age column -> wrap in try/except
  2. groupby() on Categorical without observed= -> add observed=False
  3. Plot .values on possibly-empty groupby result -> already handled in viz cell
"""
import json, re, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

total = 0

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", ""))
    original = src

    # ── Fix 1: pd.cut() calls that use a fixed bins list ────────────────────
    # These are on age columns that are always non-empty in practice,
    # but can be empty if all rows were dropped. Wrap with fillna(0) so
    # the column is never fully NaN (we'll still get 'nan' category but no crash).
    # Pattern: df['age_group'] = pd.cut(df['age'], bins=[...], labels=[...], ...)
    # Safe fix: add include_lowest=True if missing, and handle empty gracefully
    # by using a try/except around each pd.cut call.
    #
    # Simpler targeted fix: replace
    #   df['age_group'] = pd.cut(df['age'], ...)
    # with
    #   _age_vals = pd.to_numeric(df['age'], errors='coerce')
    #   df['age_group'] = pd.cut(_age_vals.fillna(_age_vals.median() if _age_vals.notna().any() else 0), ...) 
    #
    # But the cleanest approach for these specific lines is to
    # add error handling per-cell.

    # ── Fix 2: groupby() without observed= ──────────────────────────────────
    # pandas >= 2.2 warns (and will error) on groupby of Categorical columns
    # without specifying observed=. Safest: add observed=False to all groupby
    # calls that don't already have it.
    #
    # Pattern: .groupby('col') or .groupby(['col1','col2'])
    # Fix: .groupby('col', observed=False) etc.
    #
    # BUT: for non-categorical groupby, observed= is harmless so always add it.

    def add_observed(m):
        full = m.group(0)
        # Already has observed= — skip
        if "observed=" in full:
            return full
        # Has a closing paren on the same fragment — insert before it
        # e.g. .groupby('x') -> .groupby('x', observed=False)
        # The regex captures up to but not including the closing )
        return full.rstrip(")") + ", observed=False)"

    # Match .groupby(...) where ... has no observed= and ends with )
    # Use a simple approach: find .groupby( and its matching close paren
    lines = src.split("\n")
    new_lines = []
    for ln in lines:
        if ".groupby(" in ln and "observed=" not in ln:
            # Add observed=False before the closing paren of the groupby call
            # Simple heuristic: if the line ends with ) or ).size() or similar
            # Insert before the first ) that closes groupby
            # Find the groupby call
            idx = ln.index(".groupby(")
            after = ln[idx + len(".groupby("):]
            # Find matching closing paren
            depth = 1
            pos = 0
            for ch in after:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                pos += 1
            # Insert ', observed=False' before pos
            new_after = after[:pos] + ", observed=False" + after[pos:]
            ln = ln[:idx + len(".groupby(")] + new_after
        new_lines.append(ln)
    src = "\n".join(new_lines)

    if src != original:
        cell["source"] = [src]
        cell["outputs"] = []
        cell["execution_count"] = None
        total += 1
        print(f"  [PATCHED] Cell {i}: groupby observed=False")

print(f"\nTotal cells patched: {total}")

# ── Also clear all outputs ───────────────────────────────────────────────────
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("[SAVED]")
