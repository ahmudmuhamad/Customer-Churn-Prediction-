import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import shutil
import sys
import joblib

# Add project root to sys.path since src is not installed as a package in some envs
# Use insert(0) to ensure we import the LOCAL src, not a stale installed one
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_pipeline.load import load_data
from src.feature_pipeline.clean import (
    _coerce_numeric,
    _fix_nans_with_medians,
    run_cleaning
)
from src.feature_pipeline.transform import (
    create_preprocessor,
    run_transformation,
    CATEGORICAL_COLS,
    NUMERICAL_COLS
)

# =========================
# 1. TEST: load.py
# =========================
def test_load_data_logic(tmp_path):
    # Setup dummy raw split data
    raw_dir = tmp_path / "splits"
    raw_dir.mkdir()
    
    # Create dummy csvs
    pd.DataFrame({"a": [1]}).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame({"a": [2]}).to_csv(raw_dir / "validation.csv", index=False)
    pd.DataFrame({"a": [3]}).to_csv(raw_dir / "test.csv", index=False)

    # Run function
    out_dir = tmp_path / "interim"
    load_data(input_dir=raw_dir, output_dir=out_dir)

    # Assertions
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "val.csv").exists()
    assert (out_dir / "test.csv").exists()
    
    print("✅ Load logic passed")

# =========================
# 2. TEST: clean.py
# =========================
def test_coerce_numeric():
    df = pd.DataFrame({
        "TotalCharges": ["100", "200", " ", "nan", "300"],
        "Other": ["A", "B", "C", "D", "E"]
    })
    
    cleaned = _coerce_numeric(df, ["TotalCharges"])
    
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 2 # " " and "nan" string
    assert cleaned["TotalCharges"].iloc[0] == 100
    print("✅ Numeric coercion passed")

def test_fix_nans_imputation():
    df = pd.DataFrame({
        "TotalCharges": [100.0, np.nan, 300.0, np.nan], 
        "other_col": [1, 2, 3, 4]
    })
    
    medians = {"TotalCharges": 200.0}
    
    fixed_df = _fix_nans_with_medians(df, medians)
    
    assert fixed_df["TotalCharges"].iloc[1] == 200.0
    assert fixed_df["TotalCharges"].iloc[3] == 200.0
    assert fixed_df["TotalCharges"].iloc[0] == 100.0
    print("✅ NaN imputation passed")

# =========================
# 3. TEST: transform.py
# =========================
def test_create_preprocessor():
    preprocessor = create_preprocessor()
    assert len(preprocessor.transformers) == 2
    names = [t[0] for t in preprocessor.transformers]
    assert 'num' in names
    assert 'cat' in names
    print("✅ Preprocessor creation passed")

# =========================
# 4. INTEGRATION TEST
# =========================
from unittest.mock import patch

@patch("src.feature_pipeline.clean.ARTIFACTS_DIR")
@patch("src.feature_pipeline.transform.ARTIFACTS_DIR")
def test_full_pipeline_integration(mock_trans_artifacts, mock_clean_artifacts, tmp_path):
    """
    Tests the full flow (Load -> Clean -> Transform).
    """
    # 1. Setup Directories
    raw_dir = tmp_path / "raw" / "splits"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    
    for d in [raw_dir, interim_dir, processed_dir, artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True) # parents=True for raw/splits
    
    # Configure Mocks
    mock_clean_artifacts.__truediv__.side_effect = lambda x: artifacts_dir / x
    # For transform, run_transformation handles both preprocessor (joblib) and label_encoder (joblib)
    mock_trans_artifacts.__truediv__.side_effect = lambda x: artifacts_dir / x

    # 2. Create Dummy Data
    # Must contain relevant columns
    df = pd.DataFrame({
        "gender": ["Male", "Female"] * 5,
        "SeniorCitizen": [0, 1] * 5,
        "Partner": ["Yes", "No"] * 5,
        "Dependents": ["Yes", "No"] * 5,
        "PhoneService": ["Yes", "No"] * 5,
        "MultipleLines": ["No phone service"] * 10,
        "InternetService": ["DSL"] * 10,
        "OnlineSecurity": ["No"] * 10,
        "OnlineBackup": ["Yes"] * 10,
        "DeviceProtection": ["No"] * 10,
        "TechSupport": ["No"] * 10,
        "StreamingTV": ["No"] * 10,
        "StreamingMovies": ["No"] * 10,
        "Contract": ["Month-to-month"] * 10,
        "PaperlessBilling": ["Yes"] * 10,
        "PaymentMethod": ["Electronic check"] * 10,
        "tenure": range(10),
        "MonthlyCharges": [50.0] * 10,
        "TotalCharges": ["100", " ", "200", "300", "400", "500", "600", "700", "800", "900"], # Has string/empty
        "Churn": ["Yes", "No"] * 5
    })
    
    df.to_csv(raw_dir / "train.csv", index=False)
    # Validation
    df.to_csv(raw_dir / "validation.csv", index=False)
    # Test
    df.to_csv(raw_dir / "test.csv", index=False)

    # 3. Run Pipeline
    load_data(input_dir=raw_dir, output_dir=interim_dir)
    run_cleaning(input_dir=interim_dir, output_dir=interim_dir)
    run_transformation(input_dir=interim_dir, output_dir=processed_dir)
    
    # 4. Verify Artifacts
    assert (artifacts_dir / "imputer.json").exists()
    assert (artifacts_dir / "preprocessor.joblib").exists()
    assert (artifacts_dir / "label_encoder.joblib").exists()
    
    # 5. Verify Processed Data
    train_proc = pd.read_csv(processed_dir / "train_processed.csv")
    assert not train_proc.empty
    # Check OneHotEncoding happened (gender -> gender_Female, gender_Male)
    assert "gender_Female" in train_proc.columns or "cat__gender_Female" in train_proc.columns
    # Check Scaling happened
    assert abs(train_proc["tenure"].mean()) < 1 # Scaled roughly 0
    # Check Target Encoded
    assert train_proc["Churn"].dtype in [np.int32, np.int64]

    print("✅ Full Pipeline Integration Passed")
