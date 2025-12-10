"""
Batch Inference Script.

- Reads from input CSV.
- Loads artifacts (Model, Imputer, Preprocessor).
- Applies cleaning and transformation.
- Generates predictions.
- Saves results.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline.clean import _coerce_numeric, _fix_nans_with_medians, NUMERICAL_COLS_TO_CLEAN
from src.feature_pipeline.transform import CATEGORICAL_COLS, NUMERICAL_COLS

# Params
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_production.joblib" # Default to production
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    """Load model, imputer, and preprocessor."""
    print("Loading artifacts...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    imputer_path = ARTIFACTS_DIR / "imputer.json"
    if not imputer_path.exists():
        raise FileNotFoundError(f"Imputer not found at {imputer_path}")
    with open(imputer_path, "r") as f:
        imputer_dict = json.load(f)
        
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    return model, imputer_dict, preprocessor

def preprocess_new_data(df: pd.DataFrame, imputer_dict: dict, preprocessor) -> pd.DataFrame:
    """Clean and Transform new data using saved artifacts."""
    df = df.copy()
    
    # 1. Cleaning (Coerce + Impute)
    print("   Cleaning data...")
    df = _coerce_numeric(df, NUMERICAL_COLS_TO_CLEAN)
    df = _fix_nans_with_medians(df, imputer_dict)
    
    # 2. Transform (Scaling + OHE)
    print("   Transforming data...")
    # The preprocessor expects specific columns. We must ensure they exist.
    # In a real scenario, we might check for missing cols.
    
    X_transformed = preprocessor.transform(df)
    
    # Reconstruct DF with feature names
    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
    all_features = NUMERICAL_COLS + list(onehot_features)
    
    X_processed = pd.DataFrame(X_transformed, columns=all_features)
    
    return X_processed

def run_inference(input_path: Path | str = None):
    print("Starting Inference Pipeline...")
    
    if input_path is None:
        # Default to a sample file or validation set for smoke processing
        input_path = PROJECT_ROOT / "data" / "interim" / "test.csv"
        print(f"   No input provided. Using: {input_path}")
        
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # 1. Load Artifacts
    model, imputer, preprocessor = load_artifacts()
    
    # 2. Load Data
    raw_df = pd.read_csv(input_path)
    print(f"   Raw Data Shape: {raw_df.shape}")
    
    # 3. Process
    try:
        X_new = preprocess_new_data(raw_df, imputer, preprocessor)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return
        
    print(f"   Processed Data Shape: {X_new.shape}")
    
    # 4. Predict
    print("   Generating predictions...")
    preds = model.predict(X_new)
    print("   Generating probabilities...")
    probs = model.predict_proba(X_new)[:, 1]
    
    # 5. Save
    results = raw_df.copy()
    results["predicted_churn"] = preds
    results["churn_probability"] = probs
    
    save_path = OUTPUT_DIR / "predictions.csv"
    results.to_csv(save_path, index=False)
    
    print(f"Predictions saved to: {save_path}")
    print(f"   Sample: {preds[:5]}")

if __name__ == "__main__":
    run_inference()
