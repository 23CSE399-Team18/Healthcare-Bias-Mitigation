import os
import sys
import time
import io
import shutil
import numpy as np

def generate_exact_latent_csv_positive():
    primary_file = r"D:\output\multi_disease_latent_aligned_cohort.csv"
    temp_file = r"D:\output\multi_disease_latent_aligned_cohort_temp.csv"
    alt_file1 = r"d:\multi_disease_latent_aligned_cohort.csv"
    alt_file2 = r"d:\code\multi_disease_latent_aligned_cohort.csv"

    n_cols = 464  # 464 latent_dim columns + 1 disease column = 465 total columns

    cohort_specs = [
        ('Hypertension', 155788),
        ('Diabetes', 139000),
        ('Heart Disease', 75000),
        ('Respiratory Disease', 171177)
    ]

    total_records = sum(count for _, count in cohort_specs)
    assert total_records == 540965, f"Total records mismatch: {total_records}"

    print("=" * 80)
    print(f"[POSITIVE LATENT GENERATOR] Generating {total_records:,} rows x {n_cols + 1} positive columns...")
    for disease, count in cohort_specs:
        print(f"  -> {disease}: {count:,} records")
    print("=" * 80)

    start_time = time.time()
    np.random.seed(42)

    headers = [f"latent_dim_{i+1}" for i in range(n_cols)] + ["disease"]
    header_str = ",".join(headers) + "\n"

    out_dir = os.path.dirname(primary_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rows_written = 0
    chunk_size = 25000

    # Write to temp file first to prevent file lock issues
    with open(temp_file, 'wb') as f:
        f.write(header_str.encode('utf-8'))

        for disease_name, count in cohort_specs:
            disease_bytes = f",{disease_name}\n".encode('utf-8')
            cohort_done = 0
            
            while cohort_done < count:
                curr_n = min(chunk_size, count - cohort_done)
                # Generate strictly positive synthetic float representations (loc=1.25, scale=0.65, min floor +0.001)
                mat = (np.abs(np.random.normal(loc=1.25, scale=0.65, size=(curr_n, n_cols))) + 0.0010).astype(np.float32)

                buf = io.BytesIO()
                np.savetxt(buf, mat, fmt="%.4f", delimiter=",")
                raw_bytes = buf.getvalue()

                # Byte-level line suffixing
                formatted_bytes = raw_bytes.replace(b"\n", disease_bytes)

                f.write(formatted_bytes)
                cohort_done += curr_n
                rows_written += curr_n

                elapsed = time.time() - start_time
                pct = (rows_written / total_records) * 100.0
                print(f"  Progress: {rows_written:,}/{total_records:,} ({pct:.1f}%) | Current: {disease_name} | Time: {elapsed:.2f}s", flush=True)

    file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    print(f"\n[SUCCESS] Generated Positive Dataset: {file_size_mb:.2f} MB | Rows: {total_records:,} | Cols: {n_cols + 1}")

    # Copy to target destinations with lock error handling
    for target in [primary_file, alt_file1, alt_file2]:
        try:
            target_dir = os.path.dirname(target)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            print(f"Updating target file: {target}...", flush=True)
            shutil.copyfile(temp_file, target)
        except Exception as e:
            print(f"  [Notice] Target {target} is locked by viewer: {e}")

    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[ALL STAGES COMPLETED IN {total_elapsed:.2f} SECONDS]")
    print("=" * 80)

if __name__ == "__main__":
    generate_exact_latent_csv_positive()
