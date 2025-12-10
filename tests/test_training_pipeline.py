import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock optuna BEFORE importing src.training_pipeline.tune
# This prevents ModuleNotFoundError if optuna is not installed in the test env
sys.modules["optuna"] = MagicMock()

from src.training_pipeline.train import train_model
from src.training_pipeline.tune import run_tuning
from src.training_pipeline.eval import evaluate_model

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Creates dummy processed data for testing."""
    # Dummy classification data
    df = pd.DataFrame({
        "tenure": range(10),
        "MonthlyCharges": [50.0] * 10,
        "TotalCharges": [500.0] * 10,
        "gender_Male": [1, 0] * 5,
        "Churn": [0, 1] * 5  # Binary target
    })
    
    # Save as train, val, and test processed (matching filenames in scripts)
    df.to_csv(tmp_path / "train_processed.csv", index=False)
    df.to_csv(tmp_path / "val_processed.csv", index=False)
    df.to_csv(tmp_path / "test_processed.csv", index=False)
    
    return tmp_path

# ================================
# TEST: train.py
# ================================
@patch("src.training_pipeline.train.joblib.dump")
def test_train_model_runs(mock_dump, dummy_data_dir):
    """Test that train_model runs without error."""
    with patch("src.training_pipeline.train.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Use small params for speed
        test_params = {"n_estimators": 2, "max_depth": 2, "use_label_encoder": False}
        
        # Run function
        model = train_model(
            data_dir=dummy_data_dir,
            params=test_params,
            experiment_name="test_experiment"
        )
        
        # Assertions
        assert model is not None
        mock_dump.assert_called_once()
        print("✅ train_model test passed")

# ================================
# TEST: tune.py
# ================================
def test_tune_model_runs(dummy_data_dir):
    # Configure the mocked optuna to return a valid study with dict best_params
    import optuna 
    mock_study = MagicMock()
    mock_study.best_params = {"n_estimators": 10, "max_depth": 3}
    mock_study.best_value = 0.85
    optuna.create_study.return_value = mock_study

    with patch("src.training_pipeline.tune.mlflow") as mock_mlflow:
        # Mocking context manager for start_run
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        
        # Run tuning with just 1 trial
        best_params = run_tuning(n_trials=1, data_dir=dummy_data_dir)
        
        assert isinstance(best_params, dict)
        assert best_params["n_estimators"] == 10
        print("✅ tune_model test passed")

# ================================
# TEST: eval.py
# ================================
@patch("src.training_pipeline.eval.joblib.load")
def test_eval_model_logic(mock_load, dummy_data_dir):
    with patch("src.training_pipeline.eval.mlflow") as mock_mlflow:
        # Mock the model loading
        fake_model = MagicMock()
        # Predict classes (0 or 1)
        fake_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        # Predict proba
        fake_model.predict_proba.return_value = np.array([[0.1, 0.9]] * 10)
        
        mock_load.return_value = fake_model
        
        # Patch MODEL_DIR to point to dummy_data_dir where we can create a dummy file
        with patch("src.training_pipeline.eval.MODEL_DIR", dummy_data_dir):
             # Create dummy model file so exists() check passes
            (dummy_data_dir / "xgboost_best.joblib").touch()
            
            metrics = evaluate_model(
                model_name="xgboost_best.joblib", 
                data_dir=dummy_data_dir
            )
            
            assert "f1" in metrics
            assert "accuracy" in metrics
            # 0.7 threshold. F1 on (0,1,0,1...) vs (0,1,0,1...) is 1.0
            assert metrics["f1"] == 1.0 
            print("✅ evaluate_model test passed")
