
"""Student C - Seven-day cryptocurrency trend prediction service."""

import json
import math
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.fetch_crypto import fetch_token, resolve_token
from app.tc_features import engineer_features
from app.team_features import get_shared_features


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "trend_classifier.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

# Exact feature names and order used during Experiment 1 training.
EXPECTED_FEATURES = ["MA5", "MA20", "Return20"]


class PredictionRequest(BaseModel):
    token: str = Field(..., examples=["BTC"])
    forecast_horizon: int = Field(default=7)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load and cache the verified Experiment 1 model and metadata."""

    if not MODEL_PATH.is_file():
        raise RuntimeError("Trained model file is missing")

    if not METADATA_PATH.is_file():
        raise RuntimeError("Model metadata file is missing")

    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("features") != EXPECTED_FEATURES:
        raise RuntimeError(
            "Model feature order does not match Experiment 1"
        )

    if len(metadata.get("classes", [])) != 5:
        raise RuntimeError("Expected five trend categories")

    if getattr(model, "n_features_in_", None) != 3:
        raise RuntimeError(
            "Model must expect exactly three input features"
        )

    return model, metadata


@router.post("/predict")
def predict_trend(request: PredictionRequest):
    """Predict the seven-day trend category for a cryptocurrency."""

    # Step 1: Validate forecast horizon.
    if request.forecast_horizon != 7:
        raise HTTPException(
            status_code=422,
            detail="Only a 7-day forecast horizon is supported",
        )

    # Step 2: Validate cryptocurrency.
    try:
        token = resolve_token(request.token)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Step 3: Load the trained model and metadata.
    try:
        model, metadata = load_artifacts()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Trained model or metadata is unavailable",
        ) from exc

    # Step 4: Fetch live market data.
    try:
        raw_data = fetch_token(token)

        if len(raw_data) < 21:
            raise ValueError(
                "At least 21 completed daily candles are required"
            )

        engineered = engineer_features(raw_data)

        if engineered.empty:
            raise ValueError("No engineered feature rows available")

        latest = engineered.iloc[-1]

        # Step 5: Identify the latest completed Kraken candle.
        #
        # This date will also be used to verify that teammate
        # features refer to the same market observation.
        data_date = pd.Timestamp(latest["Date"]).date()
        data_date_str = data_date.isoformat()

        # Step 6: Compute all three features locally.
        #
        # These are fallback values if teammate services
        # cannot be reached or their data is out of date.
        local_features = {}

        for feature in EXPECTED_FEATURES:
            value = float(latest[feature])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid local feature value: {feature}"
                )

            local_features[feature] = value

        # Step 7: Obtain features from teammate services.
        #
        # Student A provides Return20.
        # Student B provides MA5 and MA20.
        #
        # The helper verifies the remote feature date
        # against the latest local Kraken candle date.
        #
        # If the request fails, the date differs, or the
        # response contains invalid values, it falls back
        # to the locally calculated Kraken features.
        values, feature_sources = get_shared_features(
            token=token,
            local_features=local_features,
            horizon=request.forecast_horizon,
            data_date=data_date_str,
        )

        # Step 8: Validate the final features.
        for feature in EXPECTED_FEATURES:
            if feature not in values:
                raise ValueError(
                    f"Missing required feature: {feature}"
                )

            value = float(values[feature])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid feature received: {feature}"
                )

            values[feature] = value

        # Step 9: Construct the model input.
        #
        # Preserve the exact feature order from Experiment 1.
        model_input = pd.DataFrame(
            [[values[name] for name in EXPECTED_FEATURES]],
            columns=EXPECTED_FEATURES,
        )

        # Step 10: Generate predictions and probabilities.
        predicted_class = int(model.predict(model_input)[0])
        probabilities = model.predict_proba(model_input)[0]

        labels = metadata["classes"]

        if predicted_class < 0 or predicted_class >= len(labels):
            raise ValueError(
                "Unexpected trend category returned by model"
            )

        if len(probabilities) != len(labels):
            raise ValueError(
                "Model probabilities do not match category labels"
            )

        confidence = float(probabilities[predicted_class])

        if not math.isfinite(confidence):
            raise ValueError("Invalid prediction confidence")

        # Step 11: Calculate the seven-day forecast date.
        forecast_date = data_date + timedelta(days=7)

        # Step 12: Return the final prediction.
        return {
            "token": token,
            "forecast_horizon": 7,
            "data_date": data_date_str,
            "forecast_date": forecast_date.isoformat(),
            "prediction": labels[predicted_class],
            "confidence": round(confidence, 6),
            "class_probabilities": {
                label: round(float(probabilities[index]), 6)
                for index, label in enumerate(labels)
            },
            "features_used": {
                name: values[name]
                for name in EXPECTED_FEATURES
            },
            "feature_sources": feature_sources,
            "model": metadata["model_name"],
            "source": "Kraken",
        }

    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction features are unavailable: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Prediction failed while processing market data",
        ) from exc
