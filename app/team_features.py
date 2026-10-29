
"""Fetch shared features from teammate APIs with local fallback."""

import math
import os

import requests

FEATURE_OWNERS = {
    "MA5": "B",
    "MA20": "B",
    "Return20": "A",
}


def get_shared_features(token, local_features, horizon=7):
    """
    Try teammate APIs first; use locally calculated values
    if a service is unavailable or returns invalid data.
    """
    result = {}
    sources = {}

    for student, names in {
        "A": ["Return20"],
        "B": ["MA5", "MA20"],
    }.items():
        url = os.getenv(f"STUDENT_{student}_API_URL", "").rstrip("/")
        remote = {}

        if url:
            try:
                response = requests.post(
                    f"{url}/features",
                    json={"token": token, "forecast_horizon": horizon},
                    timeout=5,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    remote = payload.get("features", {})
            except (requests.RequestException, ValueError):
                remote = {}

        for name in names:
            value = remote.get(name) if isinstance(remote, dict) else None

            try:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Non-finite remote value")
                result[name] = value
                sources[name] = f"Student {student} API"
            except (TypeError, ValueError):
                fallback = float(local_features[name])
                if not math.isfinite(fallback):
                    raise ValueError(f"Invalid local {name}")
                result[name] = fallback
                sources[name] = "Local Kraken fallback"

    return result, sources
