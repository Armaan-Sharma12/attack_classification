import pandas as pd
import numpy as np

def load_conn_logs(input_folder):
    """Load all conn.log.labeled files into one DataFrame."""
    dfs = []
    for f in Path(input_folder).rglob("conn.log.labeled"):
        df = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def clean_labels(df, label_col="Label"):
    """Simplify labels: Attack / Benign, or keep attack classes if available."""
    df[label_col] = df[label_col].str.lower()
    df[label_col] = df[label_col].replace({
        'benign': 'benign',
        '-': 'benign',
        'background': 'benign'
    })
    return df
