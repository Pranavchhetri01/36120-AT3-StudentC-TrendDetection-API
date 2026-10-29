
"""Retrieve teammate-owned features with date-aware local fallback."""

import math
import os

import requests


FEATURE_OWNERS = {
    "Return20": "A",
    "MA5": "B",
    "MA20": "B",
}

STUDENT_FEATURES = {
    "A": ["Return20"],
    "B": ["MA5", "MA20"],
}


def get_shared_features(
    token,
    local_features,
    horizon=7,
    data_date=None,
):
    """Use teammate features only when valid and date-aligned."""

    result = {}
    sources = {}

    expected_date = str(data_date) if data_date is not None else None

    for student, names in STUDENT_FEATURES.items():
        url = os.getenv(
            f"STUDENT_{student}_API_URL", ""
        ).rstrip("/")

        remote = {}

        if url and expected_date:
            try:
                response = requests.post(
                    f"{url}/features",
                    json={
                        "token": token,
                        "forecast_horizon": horizon,
                    },
                    timeout=15,
                )

                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, dict):
                    raise ValueError("Invalid API response")

                remote_date = payload.get(
                    "date", payload.get("data_date")
                )

                if remote_date != expected_date:
                    raise ValueError(
                        f"Feature date mismatch: {remote_date}"
                    )

                remote = payload.get("features", {})

                if not isinstance(remote, dict):
                    remote = {}

            except (requests.RequestException, ValueError):
                remote = {}

        for name in names:
            value = remote.get(name)

            try:
                value = float(value)

                if not math.isfinite(value):
                    raise ValueError("Invalid remote feature")

                result[name] = value
                sources[name] = f"Student {student} API"

            except (TypeError, ValueError):
                fallback = float(local_features[name])

                if not math.isfinite(fallback):
                    raise ValueError(
                        f"Invalid local feature: {name}"
                    )

                result[name] = fallback
                sources[name] = "Local Kraken fallback"

    return result, sources
