import pandas as pd
import numpy as np
from pathlib import Path
from utils import clean_labels  # Assuming clean_labels is in your utils.py
import argparse
import sys

def engineer_features(input_folder, output_file):
    """
    Processes log files one by one to engineer features and saves them to a
    single CSV file, operating in a memory-efficient manner.
    """
    print("[INFO] Starting memory-efficient feature engineering...")

    # Define the columns from the original conn.log files we want to keep.
    keep_cols = [
        'duration', 'orig_bytes', 'resp_bytes',
        'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes',
        'Label'
    ]
    
    # Full column names for Zeek conn logs from the IoT-23 dataset.
    # The last two are the labels added to this specific dataset.
    col_names = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
        'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'tunnel_parents', 'Label', 'detailed-label'
    ]

    # Flag to ensure the header is only written for the first file.
    is_first_file = True

    # Find all labeled connection logs recursively.
    log_files = list(Path(input_folder).rglob("*.conn.log.labeled"))
    if not log_files:
        print(f"[ERROR] No '*.conn.log.labeled' files found in {input_folder}", file=sys.stderr)
        return

    print(f"[INFO] Found {len(log_files)} files to process...")

    for f in log_files:
        try:
            # Read a single log file. Zeek logs are tab-separated, use '-' for null,
            # and have commented headers we need to skip.
            df = pd.read_csv(
                f,
                sep='\t',
                names=col_names,
                na_values='-',
                comment='#'
            )
            
            if df.empty:
                print(f"[WARN] Skipping empty file: {f.name}")
                continue

            # --- Apply your original processing logic to this single file ---
            df = clean_labels(df, label_col="Label")
            df = df[keep_cols].fillna(0)
            
            # -----------------------------------------------------------------

            if is_first_file:
                # For the first file, write with a header and overwrite any existing file.
                print(f"[INFO] Creating '{output_file}' and writing first batch of features...")
                df.to_csv(output_file, mode='w', header=True, index=False)
                is_first_file = False
            else:
                # For all other files, append without the header.
                df.to_csv(output_file, mode='a', header=False, index=False)

        except Exception as e:
            print(f"[ERROR] Failed to process file {f.name}: {e}", file=sys.stderr)

    print(f"\n[INFO] ✅ All files processed. Final dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient feature engineering for IoT-23 conn logs."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Input folder containing IoT-23 conn.log.labeled files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Output CSV file for features"
    )
    args = parser.parse_args()

    engineer_features(args.input, args.output)
