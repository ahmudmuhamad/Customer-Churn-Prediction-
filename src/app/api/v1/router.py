from fastapi import APIRouter, HTTPException, Depends
from src.app.api.v1.schemas import CustomerData
from src.app.api.v1.state import ml_resources
from src.inference_pipeline.inference import preprocess_new_data
import pandas as pd

router = APIRouter(tags=["Predictions"])

@router.post("/predict")
def predict_churn(data: CustomerData):
    """
    Predict churn for a single customer.
    """
    # 1. Retrieve artifacts
    model = ml_resources.get("model")
    imputer = ml_resources.get("imputer")
    preprocessor = ml_resources.get("preprocessor")
    
    if not model or not imputer or not preprocessor:
        raise HTTPException(status_code=503, detail="ML resources not loaded.")
    
    # 2. Convert input Pydantic model to DataFrame
    # Pydantic -> Dict -> DataFrame (single row)
    input_df = pd.DataFrame([data.model_dump()])
    
    # 3. Preprocess
    # Reusing the EXACT SAME logic from inference pipeline
    try:
        X_processed = preprocess_new_data(input_df, imputer, preprocessor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    # 4. Predict
    try:
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # Convert numpy types to python native for JSON serialization
        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "churn_label": "Yes" if prediction == 1 else "No"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
