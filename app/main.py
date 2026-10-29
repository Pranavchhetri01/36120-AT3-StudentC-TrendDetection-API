
import json
from pathlib import Path
from app.features import router as features_router
from fastapi import FastAPI, HTTPException
from app.model_service import router as prediction_router


app = FastAPI(
    title="Student C - Cryptocurrency Trend Detection API",
    description=(
        "Advanced Machine Learning AT3 - Group 31. "
        "XGBoost-based cryptocurrency trend classification."
    ),
    version="1.0.0",
)
app.include_router(features_router)
app.include_router(prediction_router)
BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


@app.get("/")
def root():
    return {
        "service": "Student C - Trend Category Prediction",
        "group": "Group 31",
        "status": "running",
        "model": "XGBoost Classifier",
        "forecast_horizon_days": 7,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "student-c-trend-detection",
    }


@app.get("/model-metadata")
def model_metadata():
    if not METADATA_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model metadata file is unavailable",
        )

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Model metadata could not be loaded",
        ) from exc
