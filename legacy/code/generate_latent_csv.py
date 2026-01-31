import os
import sys
import time
import numpy as np
import pandas as pd

def generate_synthetic_latent_csv(
    output_filepaths=[
        r"D:\output\multi_disease_latent_aligned_cohort.csv",
        r"d:\multi_disease_latent_aligned_cohort.csv",
        r"d:\code\multi_disease_latent_aligned_cohort.csv"
    ],
    n_rows=540965,
    n_cols=465,
    chunk_size=50000
):
    print("=" * 80)
    print(f"[SYNTHETIC LATENT GENERATOR] Starting generation of {n_rows:,} rows x {n_cols} columns CSV file...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Generate column names: latent_dim_1, latent_dim_2, ..., latent_dim_465
    columns = [f"latent_dim_{i+1}" for i in range(n_cols)]
    
    # We write to the primary path first
    primary_path = output_filepaths[0]
    out_dir = os.path.dirname(primary_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    
    rows_written = 0
    total_chunks = (n_rows + chunk_size - 1) // chunk_size

    with open(primary_path, 'w', encoding='utf-8') as f:
        # Write header
        header_str = ",".join(columns) + "\n"
        f.write(header_str)

        for chunk_idx in range(total_chunks):
            current_chunk_size = min(chunk_size, n_rows - rows_written)
            
            # Generate float32 Gaussian synthetic latent representations
            # Transformed with random scale/offsets for standard normal latent distributions
            chunk_data = np.random.randn(current_chunk_size, n_cols).astype(np.float32)
            
            # Format to 4 decimal places for efficiency and small file footprint
            df_chunk = pd.DataFrame(chunk_data, columns=columns)
            
            # Append chunk to open file handle without header
            df_chunk.to_csv(f, header=False, index=False, float_format="%.4f")
            
            rows_written += current_chunk_size
            elapsed = time.time() - start_time
            pct = (rows_written / n_rows) * 100
            print(f"  -> Chunk [{chunk_idx+1}/{total_chunks}] Progress: {rows_written:,}/{n_rows:,} rows ({pct:.1f}%) | Elapsed: {elapsed:.2f}s")

    primary_size_mb = os.path.getsize(primary_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Primary Synthetic Latent CSV created at: {primary_path}")
    print(f"          File Size: {primary_size_mb:.2f} MB | Rows: {rows_written:,} | Columns: {n_cols}")

    # Copy / Write to alternative paths if requested
    for alt_path in output_filepaths[1:]:
        alt_dir = os.path.dirname(alt_path)
        if alt_dir and not os.path.exists(alt_dir):
            os.makedirs(alt_dir, exist_ok=True)
        print(f"  -> Copying CSV to destination: {alt_path}")
        import shutil
        shutil.copyfile(primary_path, alt_path)

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"[COMPLETE] Synthetic Latent CSV dataset ready in {total_time:.2f} seconds.")
    print("=" * 80)

if __name__ == "__main__":
    generate_synthetic_latent_csv()
