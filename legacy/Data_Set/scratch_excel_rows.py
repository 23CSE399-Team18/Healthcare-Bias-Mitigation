import pandas as pd

csv_path = r"e:\Data_Set\integrated_output\full_integrated_dataset.csv"
# Load specific rows to match the Excel row numbers (Excel row 150166 is 0-indexed index 150164, since row 1 is header)
df = pd.read_csv(csv_path, skiprows=range(1, 150164), nrows=3, header=0)
columns = list(df.columns)

for r_idx in range(len(df)):
    row_num = 150165 + r_idx  # Excel row number
    row = df.iloc[r_idx]
    print(f"\nExcel Row {row_num}:")
    print(f"  Col 0 (A) [{columns[0]}]: {row[columns[0]]}")
    print(f"  Col 1 (B) [{columns[1]}]: {row[columns[1]]}")
    print(f"  Col 2 (C) [{columns[2]}]: {row[columns[2]]}")
    print(f"  Col 3 (D) [{columns[3]}]: {row[columns[3]]}")
    print(f"  Col 4 (E) [{columns[4]}]: {row[columns[4]]}")
    print(f"  Col 5 (F) [{columns[5]}]: {row[columns[5]]}")
    print(f"  Col 6 (G) [{columns[6]}]: {row[columns[6]]}")
    print(f"  Col 7 (H) [{columns[7]}]: {row[columns[7]]}")
    print(f"  Col 8 (I) [{columns[8]}]: {row[columns[8]]}")
    print(f"  Col 9 (J) [{columns[9]}]: {row[columns[9]]}")
    print(f"  Col 10 (K) [{columns[10]}]: {row[columns[10]]}")
    print(f"  Col 11 (L) [{columns[11]}]: {row[columns[11]]}")
    print(f"  Col 12 (M) [{columns[12]}]: {row[columns[12]]}")
    print(f"  Col 13 (N) [{columns[13]}]: {row[columns[13]]}")
    print(f"  Col 14 (O) [{columns[14]}]: {row[columns[14]]}")
    print(f"  Col 15 (P) [{columns[15]}]: {row[columns[15]]}")
    print(f"  Col 16 (Q) [{columns[16]}]: {row[columns[16]]}")
    print(f"  Col 17 (R) [{columns[17]}]: {row[columns[17]]}")
    print(f"  Col 18 (S) [{columns[18]}]: {row[columns[18]]}")
    print(f"  Col 19 (T) [{columns[19]}]: {row[columns[19]]}")
    print(f"  Col 20 (U) [{columns[20]}]: {row[columns[20]]}")
    print(f"  Col 21 (V) [{columns[21]}]: {row[columns[21]]}")
    print(f"  Col 22 (W) [{columns[22]}]: {row[columns[22]]}")
