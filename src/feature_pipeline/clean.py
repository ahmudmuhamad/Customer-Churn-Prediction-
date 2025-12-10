"""
Clean the dataset.

- Reads from: data/interim/
- Logic:
    1. Coerces TotalCharges to numeric (errors='coerce').
    2. Calculates Medians on TRAIN for TotalCharges.
    3. ***SAVES Medians to models/artifacts/imputer.json*** (The Artifact)
    4. Imputes NaNs in all datasets using that saved artifact.
- Writes to:  data/interim/*_cleaned.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

INTERIM_DIR = Path("data/interim")
ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
NUMERICAL_COLS_TO_CLEAN = ["TotalCharges"]

def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Coerce specified columns to numeric, replacing errors with NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _fix_nans_with_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Replace NaNs with provided medians."""
    df = df.copy()
    for col, value in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df

def run_cleaning(
    input_dir: Path | str = INTERIM_DIR,
    output_dir: Path | str = INTERIM_DIR
):
    print("Starting Data Cleaning...")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    # 1. Load Data
    train = pd.read_csv(in_dir / "train.csv")
    val_df = pd.read_csv(in_dir / "val.csv")
    test_df = pd.read_csv(in_dir / "test.csv")

    # 2. Coerce Numeric (Stateless cleaning)
    train = _coerce_numeric(train, NUMERICAL_COLS_TO_CLEAN)
    val_df = _coerce_numeric(val_df, NUMERICAL_COLS_TO_CLEAN)
    test_df = _coerce_numeric(test_df, NUMERICAL_COLS_TO_CLEAN)

    # 3. Calculate Medians (THE ARTIFACT)
    fill_values = {}
    for col in NUMERICAL_COLS_TO_CLEAN:
        if col in train.columns:
            # Median of known values (ignoring NaN)
            median_val = float(train[col].median())
            fill_values[col] = median_val
            print(f"   Calculated median for {col}: {median_val:,.2f}")

    # 4. Save Artifact
    artifact_path = ARTIFACTS_DIR / "imputer.json"
    with open(artifact_path, "w") as f:
        json.dump(fill_values, f)
    print(f"   Saved Imputer Artifact to {artifact_path}")

    # 5. Apply Fixes (Using the artifact values)
    train = _fix_nans_with_medians(train, fill_values)
    val_df = _fix_nans_with_medians(val_df, fill_values)
    test_df = _fix_nans_with_medians(test_df, fill_values)

    # 6. Save Data
    train.to_csv(out_dir / "train_cleaned.csv", index=False)
    val_df.to_csv(out_dir / "val_cleaned.csv", index=False)
    test_df.to_csv(out_dir / "test_cleaned.csv", index=False)

    print("Cleaning complete.")

if __name__ == "__main__":
    run_cleaning()
