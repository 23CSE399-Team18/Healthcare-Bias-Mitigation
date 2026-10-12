import pandas as pd
import json

csv_path = r"e:\Data_Set\integrated_output\full_integrated_dataset.csv"
print("Reading CSV headers...")
# Read only the first row to get columns quickly
df_head = pd.read_csv(csv_path, nrows=5)
columns = list(df_head.columns)
print(f"Total columns in dataset: {len(columns)}")
print("First 30 columns:", columns[:30])

# Let's inspect the CDC dataset specifically
print("\nInspecting CDC Diabetes rows specifically...")
# We know from Row 150166 in the screenshot that CDC rows start with 'CDC_Diabetes_BRFSS2015' in column 'dataset_source'
# Let's read a chunk to find one CDC row and see which columns have non-null values
chunk_size = 10000
for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
    cdc_rows = chunk[chunk['dataset_source'] == 'CDC_Diabetes_BRFSS2015']
    if not cdc_rows.empty:
        # Find columns that are NOT completely null for CDC rows
        cdc_row = cdc_rows.iloc[0]
        non_null_cols = cdc_row.dropna().index.tolist()
        print(f"Number of non-null columns in this CDC row: {len(non_null_cols)}")
        print("Non-null columns for this CDC row:", non_null_cols)
        
        # Let's see the columns around the gap.
        # The screenshot shows: CDC_Diabetes_BRFSS2015 (col 0/A), metabolic (col 1/B), 0.0 (col 2/C), 0.0 (col 3/D), 60.0 (col 4/E), 60-64 (col 5/F), Female (col 6/G), ... gap ... 29.0 (col 13/N?), ...
        # Let's print the values and columns for the first 40 columns in the CSV for this row:
        print("\nFirst 40 columns and their values for this CDC row:")
        for idx, col in enumerate(columns[:40]):
            print(f"  Col {idx} ({col}): {cdc_row[col]}")
        break
