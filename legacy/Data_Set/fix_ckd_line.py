"""Fix the one broken line in CKD cell of multi_dataset_integration.ipynb"""
import json, pathlib

NB = pathlib.Path(r"d:/Data_Set/multi_dataset_integration.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

cell = nb["cells"][11]
src = "".join(cell.get("source", ""))

# The broken line uses mixed quotes that cause SyntaxError in the kernel.
# Replace with a simpler strip using replace() calls to avoid the issue.
bad  = "            col = parts[1].strip(\"'\"\")"
good = "            col = parts[1].replace(\"'\", \"\").replace('\"', \"\")"

if bad in src:
    src = src.replace(bad, good, 1)
    cell["source"] = [src]
    cell["outputs"] = []
    cell["execution_count"] = None
    print("[PATCHED] CKD strip quote fix")
else:
    # Try alternative representation
    bad2 = "col = parts[1].strip(\"'\"\")"
    if bad2 in src:
        src = src.replace(bad2, "col = parts[1].replace(\"'\", \"\").replace('\"', \"\")", 1)
        cell["source"] = [src]
        cell["outputs"] = []
        cell["execution_count"] = None
        print("[PATCHED] CKD strip quote fix (alt)")
    else:
        # Show what the line looks like so we can target it precisely
        for ln in src.split("\n"):
            if "strip" in ln and "parts" in ln:
                print(f"  ACTUAL LINE: {repr(ln)}")

# Also clear all other code cell outputs for a clean run
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("[SAVED]")
