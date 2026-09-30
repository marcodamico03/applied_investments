import os
import sys
import time
import pandas as pd

try:
    import pyarrow.csv as pv
    import pyarrow.parquet as pq
except ImportError:
    print("\n[ERROR] pyarrow is not installed. Please run: pip install pyarrow\n")
    sys.exit(1)

DATA_DIR = "data/raw"

def convert_file(file_path: str, target_name: str):
    target_path = os.path.join(DATA_DIR, target_name)

    # 1. Skip files already successfully converted
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  -> Skipping '{target_name}': already exists ({size_mb:.1f} MB).")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\nProcessing {os.path.basename(file_path)} ({file_size_mb:.1f} MB) -> {target_name}...")
    start_time = time.time()

    # 2. Fast C++ PyArrow engine (processes 1 GB in ~5-10 seconds without memory thrashing)
    try:
        table = pv.read_csv(file_path)
        pq.write_table(table, target_path, compression="snappy")
        out_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        elapsed = time.time() - start_time
        print(f"  -> Saved {target_name} ({table.num_rows:,} rows, {table.num_columns} cols)")
        print(f"  -> Size: {file_size_mb:.1f} MB -> {out_size_mb:.1f} MB (-{(1 - out_size_mb / file_size_mb) * 100:.1f}%) in {elapsed:.1f}s")
        return
    except Exception as err:
        print(f"  -> PyArrow direct read failed ({err}), falling back to direct Pandas write...")

    # 3. Fallback: Pandas without row-by-row date loops
    df = pd.read_csv(file_path, low_memory=False)
    df.to_parquet(target_path, engine="pyarrow", compression="snappy", index=False)
    out_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"  -> Saved {target_name} ({len(df):,} rows, {len(df.columns)} cols) in {elapsed:.1f}s")


def convert_csvs_to_parquet():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Directory '{DATA_DIR}' not found.")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in '{DATA_DIR}/'.")
        return

    for file_name in csv_files:
        file_path = os.path.join(DATA_DIR, file_name)

        # Inspect headers to identify dataset type
        header = pd.read_csv(file_path, nrows=5)
        cols = set(header.columns)

        if "PERMNO" in cols and ("MthCalDt" in cols or "MthRet" in cols):
            print(f"[CRSP Monthly] Detected in {file_name}")
            convert_file(file_path, "crsp_monthly.parquet")

        elif "GVKEY" in cols and ("datadate" in cols or "fyear" in cols):
            print(f"[Compustat Annual] Detected in {file_name}")
            convert_file(file_path, "compustat_annual.parquet")

        else:
            target_name = file_name.replace(".csv", ".parquet")
            print(f"[Generic CSV] Detected in {file_name}")
            convert_file(file_path, target_name)


if __name__ == "__main__":
    convert_csvs_to_parquet()