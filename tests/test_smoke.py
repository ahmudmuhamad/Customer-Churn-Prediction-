import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 1. Add Project Root to Path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# 2. Import Pipeline Functions
from src.feature_pipeline.load import load_data
from src.feature_pipeline.clean import run_cleaning
from src.feature_pipeline.transform import run_transformation
from src.training_pipeline.train import train_model
from src.inference_pipeline.inference import run_inference

def test_smoke_end_to_end(tmp_path):
    """
    The Ultimate Sanity Check.
    Runs the full lifecycle on dummy Churn data.
    Mocking MLflow prevents needing a server running.
    Redirects all file I/O to a temp directory.
    """
    print("\n💨 Starting Smoke Test...")

    # --- SETUP TEMP DIRECTORIES ---
    raw_splits_dir = tmp_path / "data" / "raw" / "splits"
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    artifacts_dir = models_dir / "artifacts"
    predictions_dir = tmp_path / "data" / "predictions"
    
    for d in [raw_splits_dir, interim_dir, processed_dir, models_dir, artifacts_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: CREATE DUMMY RAW DATA (Splits) ---
    # Create a small DF with all required columns
    df = pd.DataFrame({
        "gender": ["Male"] * 10,
        "SeniorCitizen": [0] * 10,
        "Partner": ["Yes"] * 10,
        "Dependents": ["No"] * 10,
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "PhoneService": ["Yes"] * 10,
        "MultipleLines": ["No"] * 10,
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
        "MonthlyCharges": [50.5] * 10,
        "TotalCharges": ["100.0", "200.0", "300.0", "400.0", " ", "600.0", "700.0", "800.0", "900.0", "1000.0"], # String with one missing to test coercion
        "Churn": ["No", "Yes"] * 5
    })
    
    # Save dummy splits
    df.to_csv(raw_splits_dir / "train.csv", index=False)
    df.to_csv(raw_splits_dir / "validation.csv", index=False)
    df.to_csv(raw_splits_dir / "test.csv", index=False)

    # --- STEP 2: PATCH PATHS ---
    # We trick the scripts into using our temp folders via patches
    # Note: Inference uses globals, cleaning uses globals.
    
    with patch("src.feature_pipeline.clean.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.feature_pipeline.transform.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.training_pipeline.train.MODEL_DIR", models_dir), \
         patch("src.inference_pipeline.inference.MODEL_PATH", models_dir / "xgboost_production.joblib"), \
         patch("src.inference_pipeline.inference.ARTIFACTS_DIR", artifacts_dir), \
         patch("src.inference_pipeline.inference.OUTPUT_DIR", predictions_dir), \
         patch("src.training_pipeline.train.mlflow") as mock_mlflow:

        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = "smoke_test_run"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        # --- EXECUTE PIPELINE ---
        
        print("\n1️⃣ Running LOAD...")
        # load.py accepts args
        load_data(input_dir=raw_splits_dir, output_dir=interim_dir)
        assert (interim_dir / "train.csv").exists()

        print("2️⃣ Running CLEAN...")
        # clean.py accepts args, but ARTIFACTS_DIR was patched
        run_cleaning(input_dir=interim_dir, output_dir=interim_dir)
        assert (interim_dir / "train_cleaned.csv").exists()
        assert (artifacts_dir / "imputer.json").exists()

        print("3️⃣ Running TRANSFORM...")
        # transform.py accepts args
        run_transformation(input_dir=interim_dir, output_dir=processed_dir)
        assert (processed_dir / "train_processed.csv").exists()
        assert (artifacts_dir / "preprocessor.joblib").exists()
        assert (artifacts_dir / "label_encoder.joblib").exists()

        print("4️⃣ Running TRAIN...")
        # train.py accepts args
        train_model(data_dir=processed_dir)
        assert (models_dir / "xgboost_production.joblib").exists()

        print("5️⃣ Running INFERENCE...")
        # Run inference on the test split we created earlier (the raw one)
        # Note: Inference expects the raw schema (before OHE), which matches our raw input.
        inference_input = raw_splits_dir / "test.csv"
        run_inference(input_path=inference_input)
        
        assert (predictions_dir / "predictions.csv").exists()

        # Check results
        results = pd.read_csv(predictions_dir / "predictions.csv")
        assert "predicted_churn" in results.columns
        assert "churn_probability" in results.columns
        assert len(results) == 10
        
        print("\n✅ SMOKE TEST PASSED! The plumbing works.")
