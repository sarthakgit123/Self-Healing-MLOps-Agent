"""Reusable training pipeline for the NYC taxi fare project."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Support direct execution by absolute path from outside the repository.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

from src.evaluate import compare_metrics, regression_metrics
from src.preprocessing import (
    FEATURE_COLUMNS,
    chronological_split,
    clean_taxi_data,
    engineer_features,
)
from src.utils import (
    DATA_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    XGB_PARAMS,
    save_json,
)
from src.tracking import log_result, tracking_run, training_parameters


def fit_models(X_train, y_train, X_val, y_val):
    """Fit Phase 2 baseline and candidate estimator on the training partition."""
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_val)
    model = XGBRegressor(**XGB_PARAMS)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    predictions = model.predict(X_val)
    return {
        "baseline": baseline,
        "baseline_predictions": baseline_predictions,
        "model": model,
        "predictions": predictions,
        "training_time_seconds": train_seconds,
    }


def build_training_metrics(y_val, baseline_predictions, predictions, train_seconds, n_train, n_val):
    """Calculate the project's established metrics and training metadata."""
    baseline_metrics = regression_metrics(y_val, baseline_predictions)
    model_metrics = regression_metrics(y_val, predictions)
    return {
        "project": "NYC Taxi Fare Prediction",
        "level": 2,
        "model_type": "XGBRegressor",
        "model_hyperparameters": XGB_PARAMS,
        "training_samples": n_train,
        "validation_samples": n_val,
        "features_count": len(FEATURE_COLUMNS),
        "features_list": FEATURE_COLUMNS,
        "training_time_seconds": round(train_seconds, 2),
        "evaluation_metrics": compare_metrics(baseline_metrics, model_metrics),
    }


def save_training_artifacts(model, metrics, model_path=MODEL_PATH,
                            features_path=FEATURES_PATH, metrics_path=METRICS_PATH):
    """Write the established Phase 2 artifact trio to the requested paths."""
    model_path = Path(model_path)
    features_path = Path(features_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json(FEATURE_COLUMNS, features_path)
    save_json(metrics, metrics_path)
    return {
        "model": str(model_path),
        "features": str(features_path),
        "metrics": str(metrics_path),
    }


def _run_training(data_path=DATA_PATH, sample_size=SAMPLE_SIZE, save_artifacts=True):
    """Fit the median baseline and XGBoost using the historical chronological split."""
    raw = pd.read_csv(data_path, nrows=sample_size)
    cleaned = clean_taxi_data(raw)
    features = engineer_features(cleaned)
    train_df, val_df = chronological_split(features)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["fare_amount"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["fare_amount"]

    fitted = fit_models(X_train, y_train, X_val, y_val)
    metrics = build_training_metrics(
        y_val,
        fitted["baseline_predictions"],
        fitted["predictions"],
        fitted["training_time_seconds"],
        len(X_train),
        len(X_val),
    )
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_training_artifacts(fitted["model"], metrics)

    return {
        **fitted,
        "metrics": metrics,
        "train_df": train_df,
        "val_df": val_df,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def run_training(data_path=DATA_PATH, sample_size=SAMPLE_SIZE, save_artifacts=True, track=True):
    """Run Phase 2 training and record it once in the local MLflow experiment."""
    if not track:
        return _run_training(data_path, sample_size, save_artifacts)
    with tracking_run("taxi-fare-training", {"orchestrator": "direct_training"}) as (mlflow, run):
        mlflow.log_params(training_parameters(sample_size))
        result = _run_training(data_path, sample_size, save_artifacts)
        log_result(result["metrics"], model=result["model"], status="completed")
        result["mlflow_run_id"] = run.info.run_id
        return result


def main():
    result = run_training()
    print(json.dumps(result["metrics"], indent=2))
    print(f"MLflow run: {result['mlflow_run_id']}\nModel: {MODEL_PATH}\nFeatures: {FEATURES_PATH}\nMetrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
