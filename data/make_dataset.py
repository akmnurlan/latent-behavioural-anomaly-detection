import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "unsw_nb15_processed.csv")

# Try common label column names used in UNSW/NF-UNSW distributions
LABEL_CANDIDATES = ["label", "Label", "attack", "Attack", "class", "Class"]

def find_label_column(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(
        f"Could not find a label column. Available columns: {list(df.columns)[:30]} ..."
    )

def to_binary_label(series: pd.Series) -> pd.Series:
    # Convert common label encodings to 0/1
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        # common encodings: "benign"/"normal" vs "attack"/types
        return (~s.isin(["benign", "normal", "0", "false"])).astype(int)

    # numeric labels
    s = pd.to_numeric(s, errors="coerce")
    # common convention: 0 benign, 1 attack
    # if there are multiple attack types encoded as 2..n, also treat as attack
    return (s.fillna(0) != 0).astype(int)

def main():
    print("=== UNSW/NF-UNSW make_dataset started ===")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"Looking for CSV in: {os.path.abspath(RAW_DIR)}")
    print(f"Found {len(csv_files)} CSV file(s): {[os.path.basename(x) for x in csv_files]}")

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. Put the NF-UNSW-NB15 CSV there and rerun."
        )

    # If multiple CSVs exist, merge them
    dfs = []
    for fp in csv_files:
        print(f"Loading: {os.path.basename(fp)}")
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged rows: {len(df):,} | columns: {len(df.columns)}")

    # Create binary label
    label_col = find_label_column(df)
    df["label"] = to_binary_label(df[label_col])
    if label_col != "label":
        df = df.drop(columns=[label_col])

    # Keep numeric features + label
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in numeric_cols:
        numeric_cols.append("label")
    df = df[numeric_cols]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved processed dataset to: {OUTPUT_PATH}")
    print("=== Done ===")

if __name__ == "__main__":
    main()
