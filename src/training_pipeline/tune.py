"""
Hyperparameter Tuning Script (Optuna).

- Reads from: data/processed/ (train_processed.csv + val_processed.csv)
- Logic:
    1. Runs Optuna optimization on XGBoost.
    2. Logs results to MLflow nested runs.
    3. Prints the best parameters.
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET = "Churn"

def load_data(data_dir: Path):
    """Load train and val data."""
    train = pd.read_csv(Path(data_dir) / "train_processed.csv")
    val_df = pd.read_csv(Path(data_dir) / "val_processed.csv")
    
    y_train = train[TARGET]
    X_train = train.drop(columns=[TARGET])
    
    y_val = val_df[TARGET]
    X_val = val_df.drop(columns=[TARGET])
    
    return X_train, y_train, X_val, y_val

def run_tuning(n_trials: int = 20, data_dir: Path = PROCESSED_DIR):
    """Run Optuna Tuning."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Customer Churn Prediction - Tuning")
    
    # Disable autologging to prevent conflicts
    mlflow.autolog(disable=True)
    
    print(f"Loading data from {data_dir}...")
    try:
        X_train, y_train, X_val, y_val = load_data(data_dir)
    except FileNotFoundError:
        print(f"Data not found in {data_dir}. Run feature pipeline first.")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        
        with mlflow.start_run(nested=True):
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            f1 = f1_score(y_val, preds)
            
            # Log params & metric
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            
        return f1

    print("Starting Optuna Tuning...")
    with mlflow.start_run(run_name="Optuna_Study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("\nBest Params Found:")
        print(study.best_params)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_score", study.best_value)
        
    return study.best_params

if __name__ == "__main__":
    run_tuning(n_trials=10)
