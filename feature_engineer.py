import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_conn_logs, clean_labels
import argparse

def engineer_features(input_folder, output_file):
    print("[INFO] Loading raw logs...")
    df = load_conn_logs(input_folder)
    df = clean_labels(df, label_col="Label")

    print("[INFO] Selecting useful features...")
    keep_cols = [
        'duration', 'orig_bytes', 'resp_bytes',
        'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes',
        'Label'
    ]
    df = df[keep_cols].fillna(0)

    print("[INFO] Saving processed dataset...")
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder containing IoT-23 conn.log.labeled files")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for features")
    args = parser.parse_args()

    engineer_features(args.input, args.output)
