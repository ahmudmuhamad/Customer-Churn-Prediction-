"""
Evaluate the Production Model.

- Reads from: data/processed/test_processed.csv
- Loads model: locally from models/xgboost_production.joblib (or xgboost_best.joblib if configured)
- Logic:
    1. Loads Holdout Data.
    2. Loads the model.
    3. Generates predictions.
    4. Calculates metrics (F1, Accuracy, ROC-AUC).
    5. Fails pipeline if F1 < 0.70.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET = "Churn"

# Config - User requested to load 'xgboost_best.joblib' or usually production model.
# Since 'train.py' saves 'xgboost_production.joblib', we default to that for the pipeline flow,
# BUT we allow override to check 'xgboost_best.joblib' as per recent user request.
# DEFAULT_MODEL_NAME = "xgboost_best.joblib" 
DEFAULT_MODEL_NAME = "xgboost_production.joblib"

def evaluate_model(
    model_name: str = DEFAULT_MODEL_NAME, 
    data_dir: Path = PROCESSED_DIR
):
    """Evaluate model on holdout set."""
    mlflow.set_tracking_uri("http://localhost:5000")
    print(f"Evaluating model: {model_name}...")

    # 1. Load Holdout Data
    holdout_path = Path(data_dir) / "test_processed.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout data not found at {holdout_path}")
        
    holdout = pd.read_csv(holdout_path)
    y_test = holdout[TARGET]
    X_test = holdout.drop(columns=[TARGET])

    # 2. Load Model
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # 3. Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 4. Metrics
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else 0.0

    print("\nHOLDOUT EVALUATION REPORT")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC-AUC:  {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 5. Log Metrics to MLflow (Optional, usually tied to a run)
    # Since we loaded locally, we might want to start a NEW run just for eval logging
    # or just print. We'll start a run.
    try:
        with mlflow.start_run(run_name="Evaluation_Job"):
            mlflow.log_params({"model_source": model_name})
            mlflow.log_metric("holdout_f1", f1)
            mlflow.log_metric("holdout_accuracy", acc)
            mlflow.log_metric("holdout_roc_auc", auc)
            print("Logged evaluation metrics to MLflow.")
    except Exception as e:
        print(f"Could not log to MLflow: {e}")

    # 6. Quality Gate
    if f1 < 0.70:
        raise ValueError(f"Model quality check failed! F1 {f1:.4f} is below 0.70 threshold.")
    
    print("Model passed quality checks.")
    return {"f1": f1, "accuracy": acc, "auc": auc}

if __name__ == "__main__":
    evaluate_model()
