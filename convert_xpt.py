import pandas as pd
import os

# Define file paths
input_file = "DEMO_L .xpt"
output_file = "NHANES_DEMO_2021_2023.csv"

def convert_xpt_to_csv(input_path, output_path):
    print(f"Loading {input_path}...")
    try:
        # Load XPT file
        # Using pyreadstat via pandas for SAS XPT files
        df = pd.read_sas(input_path)

        # Save as CSV
        df.to_csv(output_path, index=False)
        print(f"Done - CSV file created: {output_path}")
        
        # Verify file creation
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"File size: {size} bytes")
            print("\nFirst 5 rows:")
            print(df.head())
        else:
            print(f"Error: Output file {output_path} not found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_xpt_to_csv(input_file, output_file)
