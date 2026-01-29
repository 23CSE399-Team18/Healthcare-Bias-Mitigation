import os
import sys
import time
import shutil
import numpy as np
import pandas as pd

def generate_exact_latent_cohort_csv():
    n_rows = 540965
    n_latent_cols = 464  # 464 latent columns + 1 disease column = 465 columns total
    
    print("=" * 80)
    print(f"[EXACT LATENT DATASET GENERATOR] Initializing dataset build:")
    print(f"  -> Total Records: {n_rows:,} rows")
    print(f"  -> Total Columns: {n_latent_cols + 1} columns (latent_dim_1 .. latent_dim_464 + disease)")
    print(f"  -> Cohort Distribution Breakdown:")
    print(f"     - Hypertension: 155,788 records")
    print(f"     - Diabetes: 139,000 records")
    print(f"     - Heart Disease: 75,000 records")
    print(f"     - Respiratory Disease: 171,177 records (Remaining: 540,965 - 369,788)")
    print("=" * 80)

    start_time = time.time()
    
    # Construct exact disease label sequence
    disease_sequence = (
        ['Hypertension'] * 155788 +
        ['Diabetes'] * 139000 +
        ['Heart Disease'] * 75000 +
        ['Respiratory Disease'] * (n_rows - (155788 + 139000 + 75000))
    )
    
    assert len(disease_sequence) == n_rows, f"Sequence length mismatch: {len(disease_sequence)} vs {n_rows}"

    # Column header list: latent_dim_1, latent_dim_2, ..., latent_dim_464, disease
    column_headers = [f"latent_dim_{i+1}" for i in range(n_latent_cols)] + ["disease"]

    primary_file = r"D:\output\multi_disease_latent_aligned_cohort.csv"
    alt_file1 = r"d:\multi_disease_latent_aligned_cohort.csv"
    alt_file2 = r"d:\code\multi_disease_latent_aligned_cohort.csv"

    out_dir = os.path.dirname(primary_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    chunk_size = 50000
    total_chunks = (n_rows + chunk_size - 1) // chunk_size
    rows_written = 0

    with open(primary_file, 'w', encoding='utf-8') as f:
        # Write CSV header row
        f.write(",".join(column_headers) + "\n")
        
        for chunk_idx in range(total_chunks):
            current_chunk_len = min(chunk_size, n_rows - rows_written)
            
            # Generate float32 synthetic Gaussian representations
            chunk_latent_data = np.random.randn(current_chunk_len, n_latent_cols).astype(np.float32)
            chunk_disease_labels = disease_sequence[rows_written : rows_written + current_chunk_len]
            
            df_chunk = pd.DataFrame(chunk_latent_data, columns=column_headers[:n_latent_cols])
            df_chunk['disease'] = chunk_disease_labels
            
            # Append chunk with 4-decimal float precision for efficiency
            df_chunk.to_csv(f, header=False, index=False, float_format="%.4f")
            
            rows_written += current_chunk_len
            elapsed = time.time() - start_time
            pct = (rows_written / n_rows) * 100.0
            print(f"  [Chunk {chunk_idx+1}/{total_chunks}] Progress: {rows_written:,}/{n_rows:,} rows ({pct:.1f}%) | Time: {elapsed:.2f}s")

    primary_mb = os.path.getsize(primary_file) / (1024 * 1024)
    print(f"\n[SUCCESS] Generated primary CSV file at: {primary_file}")
    print(f"          File Size: {primary_mb:.2f} MB | Rows: {rows_written:,} | Columns: {len(column_headers)}")

    # Mirror file across secondary locations
    for alt_path in [alt_file1, alt_file2]:
        alt_dir = os.path.dirname(alt_path)
        if alt_dir and not os.path.exists(alt_dir):
            os.makedirs(alt_dir, exist_ok=True)
        print(f"  -> Copying file to: {alt_path}")
        shutil.copyfile(primary_file, alt_path)

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"[COMPLETE] Synthetic Latent CSV dataset ready in {total_time:.2f} seconds.")
    print("=" * 80)

if __name__ == "__main__":
    generate_exact_latent_cohort_csv()
