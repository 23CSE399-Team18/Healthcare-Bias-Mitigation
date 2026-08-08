import pathlib

out = pathlib.Path(r"d:/Data_Set/integrated_output")

checks = [
    ("full_integrated_dataset.csv",    out / "full_integrated_dataset.csv"),
    ("integration_metadata.json",      out / "integration_metadata.json"),
    ("preprocessed/ folder",           out / "preprocessed"),
    ("by_domain/ folder",              out / "by_domain"),
    ("plots/ folder",                  out / "plots"),
    ("nd_preprocessed/ folder",        out / "nd_preprocessed"),
]

print("=== INTEGRATION COMPLETE CHECK ===")
all_ok = True
for label, path in checks:
    ok = path.exists()
    if not ok:
        all_ok = False
    status = "OK     " if ok else "MISSING"
    print(f"  [{status}] {label}")

full = out / "full_integrated_dataset.csv"
if full.exists():
    size_mb = full.stat().st_size / 1e6
    print(f"\n  Master file size : {size_mb:.1f} MB")

csv_count = len(list(out.rglob("*.csv")))
png_count = len(list(out.rglob("*.png")))
pp_count  = len(list((out / "preprocessed").glob("*.csv"))) if (out / "preprocessed").exists() else 0
dom_count = len(list((out / "by_domain").glob("*.csv")))    if (out / "by_domain").exists()    else 0

print(f"  Preprocessed CSVs: {pp_count}")
print(f"  Domain CSVs      : {dom_count}")
print(f"  Total CSVs       : {csv_count}")
print(f"  Total Plots      : {png_count}")
print()
print("RESULT:", "FULLY COMPLETE" if all_ok else "INCOMPLETE - some files missing")
