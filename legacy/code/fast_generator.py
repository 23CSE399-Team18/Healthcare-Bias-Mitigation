import os
import sys
import time
import shutil
import numpy as np

def run_fast_generation():
    n_rows = 540965
    n_cols = 464  # 464 latent + 1 disease = 465 columns total
    
    primary_file = r"D:\output\multi_disease_latent_aligned_cohort.csv"
    alt_file1 = r"d:\multi_disease_latent_aligned_cohort.csv"
    alt_file2 = r"d:\code\multi_disease_latent_aligned_cohort.csv"

    print("=" * 80)
    print(f"[FAST GENERATOR] Generating {n_rows:,} rows x {n_cols + 1} columns...")
    print("  Breakdown:")
    print("    - Hypertension: 155,788")
    print("    - Diabetes: 139,000")
    print("    - Heart Disease: 75,000")
    print("    - Respiratory Disease: 171,177")
    print("=" * 80)

    start_time = time.time()
    
    disease_sequence = (
        ['Hypertension'] * 155788 +
        ['Diabetes'] * 139000 +
        ['Heart Disease'] * 75000 +
        ['Respiratory Disease'] * (n_rows - (155788 + 139000 + 75000))
    )

    headers = [f"latent_dim_{i+1}" for i in range(n_cols)] + ["disease"]
    header_str = ",".join(headers) + "\n"

    out_dir = os.path.dirname(primary_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    chunk_size = 20000
    total_chunks = (n_rows + chunk_size - 1) // chunk_size
    rows_written = 0

    with open(primary_file, 'w', encoding='utf-8') as f:
        f.write(header_str)
        
        for chunk_idx in range(total_chunks):
            current_n = min(chunk_size, n_rows - rows_written)
            
            # Generate chunk floats
            mat = np.random.randn(current_n, n_cols).astype(np.float32)
            mat = np.round(mat, 4)
            
            diseases = disease_sequence[rows_written : rows_written + current_n]
            
            # Convert matrix chunk to list of comma separated strings
            # Fast vectorized string formatting
            lines = []
            for i in range(current_n):
                row_str = ",".join(mat[i].astype(str)) + "," + diseases[i] + "\n"
                lines.append(row_str)
                
            f.writelines(lines)
            rows_written += current_n
            
            elapsed = time.time() - start_time
            pct = (rows_written / n_rows) * 100.0
            print(f"Progress: {rows_written:,}/{n_rows:,} rows ({pct:.1f}%) | Elapsed: {elapsed:.2f}s", flush=True)

    file_size_mb = os.path.getsize(primary_file) / (1024 * 1024)
    print(f"\n[SUCCESS] Primary File Generated: {primary_file} ({file_size_mb:.2f} MB)")

    for alt_path in [alt_file1, alt_file2]:
        alt_dir = os.path.dirname(alt_path)
        if alt_dir and not os.path.exists(alt_dir):
            os.makedirs(alt_dir, exist_ok=True)
        print(f"Copying to: {alt_path}...", flush=True)
        shutil.copyfile(primary_file, alt_path)

    total_elapsed = time.time() - start_time
    print(f"[ALL DONE] Completed generation and mirroring in {total_elapsed:.2f} seconds.")

if __name__ == '__main__':
    run_fast_generation()
