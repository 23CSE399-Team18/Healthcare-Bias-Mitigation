"""
fix_drive_path.py
Called by START_JUPYTER.bat to update the BASE path in both notebooks
so they work from any drive letter (D:, E:, F:, G: etc.)
"""
import json, sys, pathlib, re

# Drive letter passed from batch file e.g. "D:" or "E:"
if len(sys.argv) > 1:
    drive = sys.argv[1].rstrip("\\").rstrip("/")  # e.g. "D:"
else:
    # Auto-detect: use the drive where this script lives
    drive = pathlib.Path(__file__).drive  # e.g. "D:"

drive_lower = drive.lower().rstrip(":")
new_base = f"{drive_lower}:/Data_Set"

print(f"  Setting BASE path to: {new_base}")

NOTEBOOKS = [
    pathlib.Path(__file__).parent / "nd_preprocessing_and_integration.ipynb",
    pathlib.Path(__file__).parent / "multi_dataset_integration.ipynb",
]

# All possible old drive letters to replace
OLD_DRIVES = ["d:/Data_Set", "e:/Data_Set", "f:/Data_Set", "g:/Data_Set",
              "D:/Data_Set", "E:/Data_Set", "F:/Data_Set", "G:/Data_Set",
              "d:\\Data_Set", "e:\\Data_Set", "f:\\Data_Set", "g:\\Data_Set"]

for nb_path in NOTEBOOKS:
    if not nb_path.exists():
        print(f"  [SKIP] {nb_path.name} not found")
        continue

    text = nb_path.read_text(encoding="utf-8")
    original = text

    for old in OLD_DRIVES:
        text = text.replace(old, new_base)

    if text != original:
        nb_path.write_text(text, encoding="utf-8")
        print(f"  [UPDATED] {nb_path.name}")
    else:
        print(f"  [OK - no change needed] {nb_path.name}")

print("  Path fix complete.")
