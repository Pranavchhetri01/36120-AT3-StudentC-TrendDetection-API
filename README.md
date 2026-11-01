# Student C — Cryptocurrency Trend Detection API

**Course:** 36120 Advanced Machine Learning Application  
**Assessment:** AT3 — Cryptocurrency Financial Intelligence Platform  
**Group:** 31  
**Student:** Pranav Chhetri  
**Student ID:** 26030813  
**Role:** Student C — Trend Category (TC) Classification

## 1. Project Overview

This repository contains Student C's independent FastAPI microservice for the Group 31 Cryptocurrency Financial Intelligence Platform.

The service uses a trained **Class-Weighted XGBoost classifier** to predict cryptocurrency trend categories seven days ahead.

The five prediction categories are:

- Strong Downtrend
- Downtrend
- Sideways
- Uptrend
- Strong Uptrend

The API supports eight cryptocurrencies: BTC, ETH, XRP, SOL, DOGE, XMR, XLM and TRX.

## 2. Live Deployment

**Render API:** https://student-c-trend-api-a5fd.onrender.com

**Interactive Swagger documentation:** https://student-c-trend-api-a5fd.onrender.com/docs

The service runs independently on Render and is consumed by the Group 31 Streamlit application.

## 3. Model and Features

**Algorithm:** Class-Weighted XGBoost  
**Forecast horizon:** Seven days  
**Model input features:** MA5, MA20 and Return20

The classifier uses three mandatory predictors:

- MA5 — Five-day moving average of closing prices.
- MA20 — Twenty-day moving average of closing prices.
- Return20 — Twenty-day percentage price change.

The service retrieves live OHLCV data from the Kraken public API.

The deployed prediction service requests Return20 from Student A's API and MA5/MA20 from Student B's API. Remote values are accepted only when their observation dates match the locally fetched Kraken data. Local Kraken-derived features serve as fallbacks when teammate services are unavailable or return incompatible data.

Student C also provides four independently engineered features through `/features`: Volatility20, MaxPrice20, MaxDrawdown and GapRisk.

## 4. Model Performance

The selected model was evaluated on a chronological test dataset of 832 observations.

- Test accuracy: **34.13%**
- Test Macro-F1: **0.3284**
- Test Macro-Precision: **0.3336**
- Test Macro-Recall: **0.3741**

These predictions are experimental analytical indicators and are not guaranteed forecasts or financial advice.

## 5. Repository Structure

```text
app/
    __init__.py
    main.py
    fetch_crypto.py
    tc_features.py
    features.py
    model_service.py
    team_features.py
models/
    trend_classifier.joblib
    metadata.json
README.md
pyproject.toml
uv.lock
.python-version
.gitignore
```

## 6. Installation and Local Setup

Use Python **3.12.13** and the dependencies recorded in `pyproject.toml` and `uv.lock`.

Install `uv` if it is not already available:

```powershell
python -m pip install uv
```

Clone the repository:

```powershell
git clone https://github.com/Pranavchhetri01/36120-AT3-StudentC-TrendDetection-API.git
cd 36120-AT3-StudentC-TrendDetection-API
```

Install the locked dependencies:

```powershell
uv sync --frozen
```

Start the FastAPI server:

```powershell
uv run uvicorn app.main:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

No Kraken API key is required for the public OHLC endpoint.

## 7. Optional Teammate API Configuration

To enable live cross-service feature sharing, configure the following environment variables.

```powershell
$env:STUDENT_A_API_URL = "https://three6120-26sp-at3-group31-studenta-api.onrender.com"
$env:STUDENT_B_API_URL = "https://three6120-26sp-at3-group31-studentb-api.onrender.com"
```

Restart the application after configuring the variables.

If teammate services are unavailable or return features for a different observation date, the service falls back to locally calculated Kraken features.

## 8. API Endpoints

The microservice exposes five endpoints:

- `GET /` — Service information
- `GET /health` — Service health status
- `POST /features` — Student C's four engineered features
- `POST /predict` — Seven-day Trend Category prediction
- `GET /model-metadata` — Model information and evaluation metadata

Example prediction request:

```json
{
  "token": "BTC",
  "forecast_horizon": 7
}
```

The prediction response includes the predicted trend category, confidence, category probabilities, observation date, forecast date, model information and predictor sources.

## 9. Testing

The deployed API was tested successfully using HTTP requests across all eight supported cryptocurrencies.

Testing verified:

- Successful model loading and inference.
- Access to the five required endpoints.
- Successful predictions for the eight supported tokens.
- Student C feature generation.
- Integration with Student A and Student B's live APIs.
- Date-aware feature validation and local fallback functionality.

To test an individual prediction, use the Swagger interface at `/docs` and execute `POST /predict`.

## 10. Related Group Repositories

**Shared experimentation repository:**  
https://github.com/nhinguyenuts/36120-26SP-AT3-Group31-experiments

**Shared Streamlit repository:**  
https://github.com/nhinguyenuts/36120-26SP-AT3-Group31-streamlit

The final group Streamlit application consumes this deployed FastAPI service via HTTP requests.

## 11. Limitations

Model performance is limited by class imbalance, restricted historical coverage and the use of three mandatory technical predictors. Prediction confidence should not be interpreted as a guarantee of accuracy.

The service depends on external market-data availability. Teammate API requests may also experience delays due to free-tier cloud hosting, although local feature fallback helps maintain prediction functionality.

## 12. AI Assistance

Generative AI assistance was used during coding, debugging, documentation and reporting. All generated content and technical outputs remain subject to student review, validation and academic integrity requirements.