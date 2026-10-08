import pandas as pd

csv_path = r"e:\Data_Set\integrated_output\full_integrated_dataset.csv"
df = pd.read_csv(csv_path, skiprows=range(1, 150165), nrows=2, header=0)
columns = list(df.columns)

for r_idx in range(len(df)):
    row_num = 150166 + r_idx
    row = df.iloc[r_idx]
    
    # Let's rebuild the CSV line for this row and see the raw representation
    # to understand how the commas are structured.
    raw_vals = [str(x) if pd.notna(x) else "" for x in row]
    raw_line = ",".join(raw_vals)
    
    # Find position of BMI and next non-empty value after BMI
    bmi_val = raw_vals[162]
    # Find non-empty values after column 162
    next_filled = [(idx, col, raw_vals[idx]) for idx, col in enumerate(columns) if idx > 162 and raw_vals[idx] != ""]
    
    print(f"\nExcel Row {row_num}:")
    print(f"  BMI Value (Col 162): {bmi_val}")
    print(f"  Next non-null columns after BMI:")
    for idx, col, val in next_filled[:5]:
         print(f"    - Index {idx} ({col}): {val}")
    
    # Comma count between BMI and the first non-null value after it
    first_next_idx = next_filled[0][0]
    commas_between = first_next_idx - 162
    print(f"  Commas between BMI and {columns[first_next_idx]}: {commas_between}")
