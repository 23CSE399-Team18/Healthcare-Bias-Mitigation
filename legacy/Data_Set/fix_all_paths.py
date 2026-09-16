import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"d:\Data_Set\multi_dataset_integration.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("Scanning ALL cells for wrong paths...\n")

fixed_count = 0
for i, cell in enumerate(nb["cells"]):
    # Join source lines into one string
    src = "".join(cell.get("source", []))

    # Check if this cell has e: references
    has_e_drive = (
        "e:/Data_Set" in src or
        "e:\\Data_Set" in src or
        "e:\\\\Data_Set" in src or
        'r"e:' in src or
        "r'e:" in src
    )

    if has_e_drive:
        print(f"  [FOUND] Cell {i} ({cell['cell_type']}): {src[:100].strip()}")

        # Replace all variants
        new_src = (src
            .replace("e:/Data_Set",   "d:/Data_Set")
            .replace("e:\\\\Data_Set","d:\\\\Data_Set")
            .replace("e:\\Data_Set",  "d:\\Data_Set")
        )

        cell["source"] = [new_src]
        cell["outputs"] = []
        cell["execution_count"] = None
        fixed_count += 1
        print(f"         -> FIXED\n")

print(f"\nTotal cells fixed: {fixed_count}")

# Verify no e: remaining
remaining = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "e:/Data_Set" in src or "e:\\Data_Set" in src or "e:\\\\Data_Set" in src:
        remaining.append(i)

if remaining:
    print(f"WARNING: e: still found in cells: {remaining}")
else:
    print("VERIFIED: No e:\\Data_Set references remaining.")

# Clear all outputs
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSaved successfully. Notebook has {len(nb['cells'])} cells.")
print("Now do: Kernel -> Restart & Run All")
