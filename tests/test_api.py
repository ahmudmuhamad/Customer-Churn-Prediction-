import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.main import app
from src.app.api.v1.state import ml_resources

client = TestClient(app)

# Dummy Payload matching schemas.py
VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

@patch("src.app.main.load_artifacts") # Mock loading in lifespan
def test_health_check(mock_load):
    # Mocking lifespan load to avoid startup errors during test client creation context
    mock_load.return_value = (MagicMock(), {}, MagicMock())
    
    with TestClient(app) as local_client:
        response = local_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_predict_churn_flow():
    """Test full prediction flow with mocked model resources."""
    
    # 1. Setup Mock Resources manually in state (simulating startup)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    mock_imputer = {"TotalCharges": 100.0}
    
    mock_preprocessor = MagicMock()
    # Mock transform return shape: 1 row, 2 columns (1 numeric + 1 categorical)
    mock_preprocessor.transform.return_value = np.zeros((1, 2)) 
    mock_cat = MagicMock()
    # Mock get_feature_names_out to return 1 feature name
    mock_cat.get_feature_names_out.return_value = ["cat1_val"] 
    mock_preprocessor.named_transformers_ = {'cat': mock_cat}
    
    # Inject into state
    ml_resources["model"] = mock_model
    ml_resources["imputer"] = mock_imputer
    ml_resources["preprocessor"] = mock_preprocessor
    
    # We also need to patch the global constants in inference used by preprocessing
    # because preprocess_new_data imports them.
    # A cleaner integration test verifies the logic, here we test the API wiring.
    with patch("src.inference_pipeline.inference.NUMERICAL_COLS", ["num1"]), \
         patch("src.inference_pipeline.inference.CATEGORICAL_COLS", ["cat1"]):
         
         response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
         
         assert response.status_code == 200
         data = response.json()
         assert "prediction" in data
         assert data["prediction"] == 1
         assert "churn_probability" in data
         assert data["churn_label"] == "Yes"

def test_predict_invalid_payload():
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["gender"] # Missing required field
    
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422 # Validation Error
