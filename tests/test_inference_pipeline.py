import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference_pipeline.inference import (
    load_artifacts,
    preprocess_new_data,
    run_inference
)

@pytest.fixture
def dummy_artifacts(tmp_path):
    """Creates dummy directory structure and JSON artifact."""
    models_dir = tmp_path / "models"
    artifacts_dir = models_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    
    # 2. Dummy Imputer (JSON is clean to save)
    imputer_path = artifacts_dir / "imputer.json"
    with open(imputer_path, "w") as f:
        json.dump({"TotalCharges": 100.0}, f)
        
    return tmp_path

def test_load_artifacts(dummy_artifacts):
    """Test loading of artifacts."""
    with patch("src.inference_pipeline.inference.PROJECT_ROOT", dummy_artifacts), \
         patch("src.inference_pipeline.inference.ARTIFACTS_DIR", dummy_artifacts / "models" / "artifacts"), \
         patch("src.inference_pipeline.inference.MODEL_PATH", dummy_artifacts / "models" / "xgboost_production.joblib"), \
         patch("src.inference_pipeline.inference.joblib.load") as mock_load:
        
        # joblib.load is called twice: model, preprocessor
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_load.side_effect = [mock_model, mock_preprocessor]
        
        # Create dummy file placeholders so .exists() checks pass
        (dummy_artifacts / "models" / "xgboost_production.joblib").touch()
        (dummy_artifacts / "models" / "artifacts" / "preprocessor.joblib").touch()
        
        model, imputer, preprocessor = load_artifacts()
        
        assert model is mock_model
        assert imputer == {"TotalCharges": 100.0}
        assert preprocessor is mock_preprocessor

def test_preprocess_integration():
    """Test preprocessing glue logic."""
    # Data that needs cleaning and transforming
    df = pd.DataFrame({
        "TotalCharges": [" ", "200"],
        "tenure": [1, 2],
        "MonthlyCharges": [10, 20],
        "gender": ["M", "F"] 
    })
    
    imputer_dict = {"TotalCharges": 100.0}
    
    # Mock preprocessor
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((2, 5)) 
    mock_cat = MagicMock()
    # Dummy feature names: 3 numerical + 2 categorical = 5
    mock_cat.get_feature_names_out.return_value = ["cat_A", "cat_B"]
    mock_preprocessor.named_transformers_ = {'cat': mock_cat}
    
    # Patch the constants used in inference.py
    with patch("src.inference_pipeline.inference.NUMERICAL_COLS", ["num1", "num2", "num3"]), \
         patch("src.inference_pipeline.inference.CATEGORICAL_COLS", ["cat1"]):
             
        processed = preprocess_new_data(df, imputer_dict, mock_preprocessor)
        
        # Should return a dataframe
        assert isinstance(processed, pd.DataFrame)
        assert len(processed.columns) == 5

def test_run_inference_end_to_end(dummy_artifacts, tmp_path):
    """Test run_inference flow."""
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(input_csv, index=False)
    
    # Mock artifacts loading to avoid disk access and pickling issues
    with patch("src.inference_pipeline.inference.PROJECT_ROOT", dummy_artifacts), \
         patch("src.inference_pipeline.inference.OUTPUT_DIR", tmp_path), \
         patch("src.inference_pipeline.inference.load_artifacts") as mock_load_artifacts, \
         patch("src.inference_pipeline.inference.preprocess_new_data") as mock_prep:
         
        # Mock load_artifacts return
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        
        mock_imputer = {}
        mock_preprocessor = MagicMock()
        
        mock_load_artifacts.return_value = (mock_model, mock_imputer, mock_preprocessor)
        
        # Mock preprocessing to return dummy valid input for model
        mock_prep.return_value = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
        
        run_inference(input_path=input_csv)
        
        # Check output
        assert (tmp_path / "predictions.csv").exists()
        res = pd.read_csv(tmp_path / "predictions.csv")
        assert "predicted_churn" in res.columns
        assert "churn_probability" in res.columns

