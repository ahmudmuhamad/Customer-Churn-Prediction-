"""
Load raw split data.

- Reads from: data/raw/splits/
- Writes to:  data/interim/
"""

import pandas as pd
from pathlib import Path
import os

# Paths
RAW_SPLITS_DIR = Path("data/raw/splits")
INTERIM_DIR = Path("data/interim")

def load_data(
    input_dir: Path | str = RAW_SPLITS_DIR,
    output_dir: Path | str = INTERIM_DIR,
):
    """Load split datasets and save them to interim."""
    
    print(f"Loading data from {input_dir}...")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train = pd.read_csv(in_dir / "train.csv")
    val_df = pd.read_csv(in_dir / "validation.csv")
    test_df = pd.read_csv(in_dir / "test.csv")

    # Save to 'interim'
    train.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False) # Renaming to val.csv for consistency if preferred, or keep validation
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Data loaded and saved to {out_dir}.")
    print(f"   Train: {train.shape}")
    print(f"   Val:   {val_df.shape}")
    print(f"   Test:  {test_df.shape}")

if __name__ == "__main__":
    load_data()
