
import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.fetch_crypto import fetch_token, resolve_token
from app.tc_features import engineer_features

router = APIRouter()


class FeatureRequest(BaseModel):
    token: str = Field(..., examples=["BTC"])
    forecast_horizon: int = Field(default=7)


@router.post("/features")
def get_features(request: FeatureRequest):
    if request.forecast_horizon != 7:
        raise HTTPException(
            status_code=422,
            detail="Student C supports a 7-day forecast horizon.",
        )

    try:
        token = resolve_token(request.token)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    try:
        raw = fetch_token(token)
        if len(raw) < 21:
            raise ValueError("Insufficient completed daily candles")

        engineered = engineer_features(raw)
        latest = engineered.iloc[-1]

        feature_names = [
            "Volatility20",
            "MaxPrice20",
            "MaxDrawdown",
            "GapRisk",
        ]

        features = {}
        for name in feature_names:
            value = float(latest[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} is unavailable")
            features[name] = value

        return {
            "token": token,
            "forecast_horizon": 7,
            "data_date": latest["Date"].date().isoformat(),
            "source": "Kraken",
            "features": features,
        }

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to calculate features: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to fetch cryptocurrency market data",
        ) from exc
