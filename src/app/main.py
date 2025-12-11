from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.app.api.v1.router import router as router_v1
from src.app.api.v1.state import ml_resources
from src.inference_pipeline.inference import load_artifacts

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts on startup
    print("Loading ML artifacts...")
    model, imputer, preprocessor = load_artifacts()
    ml_resources["model"] = model
    ml_resources["imputer"] = imputer
    ml_resources["preprocessor"] = preprocessor
    print("ML artifacts loaded.")
    yield
    # Clean up (if needed)
    ml_resources.clear()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer turnover.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router_v1, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn Prediction API is running."}
