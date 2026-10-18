import json, pathlib, pandas as pd

out = pathlib.Path(r"d:/Data_Set/integrated_output")

csvs  = list(out.rglob("*.csv"))
pngs  = list(out.rglob("*.png"))
jsons = list(out.rglob("*.json"))
total_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6

print("=== FINAL OUTPUT SUMMARY ===")
print(f"Total size : {total_mb:.1f} MB")
print(f"CSV files  : {len(csvs)}")
print(f"PNG plots  : {len(pngs)}")
print(f"JSON files : {len(jsons)}")
print()

print("--- CSVs (largest first) ---")
for f in sorted(csvs, key=lambda x: x.stat().st_size, reverse=True):
    mb = f.stat().st_size / 1e6
    print(f"  {str(f.relative_to(out)):<55} {mb:>7.1f} MB")

print()
print("--- Plots ---")
for f in sorted(pngs):
    print(f"  {f.name}")

print()
full = out / "full_integrated_dataset.csv"
if full.exists():
    df = pd.read_csv(full)
    print("full_integrated_dataset.csv:")
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print("  Records per dataset:")
    for ds, cnt in df["dataset_source"].value_counts().items():
        print(f"    {ds:<30} {cnt:>8,}")
    print("  Records per domain:")
    for dom, cnt in df["domain"].value_counts().items():
        print(f"    {dom:<30} {cnt:>8,}")
