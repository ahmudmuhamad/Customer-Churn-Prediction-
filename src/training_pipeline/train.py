"""
Train the Production Model.

- Reads from: data/processed/ (train_processed.csv + val_processed.csv)
- Logic:
    1. Combines Train + Val datasets.
    2. Trains the Champion XGBoost model.
    3. Logs to MLflow.
- Saves model to: models/xgboost_production.joblib
"""

import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from mlflow.models import infer_signature

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "Churn"

# Champion Parameters 
# (In a real pipeline, these might be loaded from a config or the best run in MLflow)
CHAMPION_PARAMS = {
    'n_estimators': 155,
    'max_depth': 4,
    'learning_rate': 0.17,
    'subsample': 0.77,
    'colsample_bytree': 0.81,
    'min_child_weight': 2,
    'reg_alpha': 0.0003,
    'reg_lambda': 0.007,
    'random_state': 42,
    'n_jobs': -1,
    'use_label_encoder': False,
    'eval_metric': "logloss"
}

def load_and_combine_data(data_dir: Path):
    """Load train and val, combine them."""
    print(f"Loading datasets from: {data_dir}")
    
    train_path = data_dir / "train_processed.csv"
    val_path = data_dir / "val_processed.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find {train_path}.")

    train = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Concatenate
    full_df = pd.concat([train, val_df], axis=0).reset_index(drop=True)
    
    y = full_df[TARGET]
    X = full_df.drop(columns=[TARGET])
    
    print(f"Combined Data Shape: {X.shape}")
    return X, y

def train_model(
    data_dir: Path = PROCESSED_DIR,
    params: dict = CHAMPION_PARAMS,
    experiment_name: str = "Customer Churn Prediction - Production"
):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    # Disable autologging
    mlflow.autolog(disable=True)
    
    print("Starting Production Training...")
    
    X, y = load_and_combine_data(data_dir)
    
    with mlflow.start_run(run_name="Production_Train_Job"):
        model = XGBClassifier(**params)
        
        print(f"Training XGBoost with params: {params}")
        model.fit(X, y)
        
        # Log to MLflow
        mlflow.log_params(params)
        
        signature = infer_signature(X.head(), model.predict(X.head()))
        
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="ChurnChampion"
        )
        
        print(f"Model trained and logged to MLflow run: {mlflow.active_run().info.run_id}")
        
        # Save Locally
        local_path = MODEL_DIR / "xgboost_production.joblib"
        joblib.dump(model, local_path)
        print(f"Model saved locally to: {local_path}")
        
    return model

if __name__ == "__main__":
    train_model()
