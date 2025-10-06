import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

# ======================================================
# Helper: Clean and normalize labels
# ======================================================
def clean_labels(df, label_col="Label"):
    """Clean and normalize labels for IoT-23 dataset"""
    if label_col not in df.columns:
        print(f"[WARN] Skipping file: no '{label_col}' column found")
        return df

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[label_col] = df[label_col].replace({
        'malicious': 'Malicious',
        'benign': 'Benign',
        'background': 'Benign'
    })
    return df

# ======================================================
# Main Feature Engineering Function
# ======================================================
def engineer_features(input_folder, output_file, chunksize=100000):
    print(f"[INFO] Starting feature engineering from: {input_folder}")

    keep_cols = [
        'duration', 'orig_bytes', 'resp_bytes',
        'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes',
        'Label'
    ]
    
    col_names = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
        'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'tunnel_parents', 'Label', 'detailed-label'
    ]

    # find all conn.log.labeled recursively
    log_files = list(Path(input_folder).rglob("*.conn.log.labeled"))
    print(f"[DEBUG] Found {len(log_files)} conn.log.labeled files under {input_folder}")
    for sample in log_files[:5]:
        print(f"    ↳ {sample}")

    if not log_files:
        print(f"[ERROR] No '*.conn.log.labeled' files found under {input_folder}", file=sys.stderr)
        sys.exit(1)

    is_first_chunk = True
    total_rows = 0

    for i, f in enumerate(log_files):
        print(f"\n[INFO] ({i+1}/{len(log_files)}) Processing: {f}")
        try:
            chunk_iter = pd.read_csv(
                f,
                sep='\t',
                names=col_names,
                na_values='-',
                comment='#',
                chunksize=chunksize,
                on_bad_lines='skip',
                engine='python'
            )

            for chunk in tqdm(chunk_iter, desc=f"Processing {f.name}", unit="chunk"):
                if chunk.empty:
                    continue

                chunk = clean_labels(chunk, label_col="Label")
                if "Label" not in chunk.columns:
                    continue

                chunk = chunk[keep_cols].fillna(0)

                chunk.to_csv(
                    output_file,
                    mode='a' if not is_first_chunk else 'w',
                    header=is_first_chunk,
                    index=False
                )

                is_first_chunk = False
                total_rows += len(chunk)

        except Exception as e:
            print(f"[ERROR] Failed to process {f.name}: {e}", file=sys.stderr)

    print(f"\n[INFO] ✅ Completed successfully.")
    print(f"[INFO] Total processed rows: {total_rows}")
    print(f"[INFO] Final dataset saved to: {output_file}")

# ======================================================
# CLI Entry Point
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient feature engineering for IoT-23 conn logs."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder containing IoT-23 conn.log.labeled files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file for features")
    parser.add_argument("--chunksize", type=int, default=100000,
                        help="Number of rows to process at a time")

    args = parser.parse_args()
    engineer_features(args.input, args.output, args.chunksize)
