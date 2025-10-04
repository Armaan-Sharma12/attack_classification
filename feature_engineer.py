import pandas as pd
import numpy as np
from pathlib import Path
from utils import clean_labels  # Make sure utils.py is in same folder
import argparse
import sys

def engineer_features(input_folder, output_file, chunksize=100000):
    """
    Memory-efficient feature engineering for IoT-23 conn logs.
    Processes logs chunk-by-chunk to avoid memory overload.
    """

    print("[INFO] Starting memory-efficient feature engineering...")

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

    log_files = sorted(Path(input_folder).rglob("*.conn.log.labeled"))
    if not log_files:
        print(f"[ERROR] No '*.conn.log.labeled' files found in {input_folder}", file=sys.stderr)
        return

    print(f"[INFO] Found {len(log_files)} files to process...")

    is_first_chunk = True
    total_rows = 0

    for i, f in enumerate(log_files):
        print(f"[INFO] Processing file {i+1}/{len(log_files)}: {f}")
        try:
            for chunk in pd.read_csv(
                f,
                sep='\t',
                names=col_names,
                na_values='-',
                comment='#',
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines='skip'
            ):
                if chunk.empty:
                    continue

                chunk = clean_labels(chunk, label_col="Label")
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
            print(f"[ERROR] Failed to process file {f.name}: {e}", file=sys.stderr)

    print(f"\n[INFO] ✅ All files processed. Total rows written: {total_rows}")
    print(f"[INFO] Final dataset saved to {output_file}")


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
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Number of rows to process at a time"
    )
    args = parser.parse_args()

    engineer_features(args.input, args.output, args.chunksize)
