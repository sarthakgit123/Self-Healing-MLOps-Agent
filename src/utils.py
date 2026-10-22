"""Shared project configuration, paths, and artifact helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "taxi_fare_model.joblib"
FEATURES_PATH = MODELS_DIR / "model_features.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
PREDICTIONS_DB_PATH = PROJECT_ROOT / "data" / "predictions.sqlite3"
SAMPLE_SIZE = 300_000
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
NYC_LON_BOUNDS = (-74.30, -73.70)
NYC_LAT_BOUNDS = (40.50, 41.00)
FARE_BOUNDS = (2.50, 200.00)
PASSENGER_BOUNDS = (1, 6)
XGB_PARAMS = {
    "n_estimators": 150, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist",
}


def save_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
