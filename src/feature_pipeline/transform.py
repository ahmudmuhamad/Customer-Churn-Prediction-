"""
Apply Feature Transformations.

- Reads from: data/interim/*_cleaned.csv
- Performs:
    1. OneHotEncoding on Categorical columns.
    2. StandardScaler on Numerical columns.
    3. LabelEncoding on Target.
- Writes to:  data/processed/ (Ready for ML)
- Artifacts: models/artifacts/preprocessor.joblib, models/artifacts/label_encoder.joblib
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Config - Same as Notebook
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TARGET = 'Churn'

def run_transformation(
    input_dir: Path | str = INTERIM_DIR,
    output_dir: Path | str = PROCESSED_DIR
):
    print("Starting Feature Transformation...")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Cleaned Data
    print("   Loading cleaned data...")
    train = pd.read_csv(in_dir / "train_cleaned.csv")
    val_df = pd.read_csv(in_dir / "val_cleaned.csv")
    test_df = pd.read_csv(in_dir / "test_cleaned.csv")

    # 2. Define Transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop' 
    )

    # 3. Fit on Train (The Artifact)
    print("   Fitting Preprocessor on Train...")
    preprocessor.fit(train)

    # 4. Save Artifacts
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    print(f"   Saved Preprocessor to {ARTIFACTS_DIR / 'preprocessor.joblib'}")

    # 5. Transform Data
    print("   Transforming datasets...")
    X_train_transformed = preprocessor.transform(train)
    X_val_transformed = preprocessor.transform(val_df)
    X_test_transformed = preprocessor.transform(test_df) # Assuming we want to process test for final eval if needed, though notebook skipped it. Adding it here for completeness of pipeline.

    # 6. Reconstruct DataFrames
    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
    all_features = NUMERICAL_COLS + list(onehot_features)

    train_processed = pd.DataFrame(X_train_transformed, columns=all_features)
    val_processed = pd.DataFrame(X_val_transformed, columns=all_features)
    test_processed = pd.DataFrame(X_test_transformed, columns=all_features)

    # 7. Target Encoding
    print("   Encoding Target...")
    le = LabelEncoder()
    y_train = le.fit_transform(train[TARGET])
    y_val = le.transform(val_df[TARGET])
    y_test = le.transform(test_df[TARGET])

    train_processed[TARGET] = y_train
    val_processed[TARGET] = y_val
    test_processed[TARGET] = y_test
    
    # Save LabelEncoder
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")
    print(f"   Saved LabelEncoder to {ARTIFACTS_DIR / 'label_encoder.joblib'}")

    # 8. Save Processed Data
    train_processed.to_csv(out_dir / "train_processed.csv", index=False)
    val_processed.to_csv(out_dir / "val_processed.csv", index=False)
    test_processed.to_csv(out_dir / "test_processed.csv", index=False)

    print(f"Transformation complete. Files saved to {out_dir}")
    print(f"   Final Feature Count: {train_processed.shape[1]}")

if __name__ == "__main__":
    run_transformation()
