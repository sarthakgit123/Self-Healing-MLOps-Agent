"""Small LangGraph orchestration for the existing taxi fare pipeline.

The default MAE gate is $2.00: it gives the Phase 2 validation result of about
$1.81 a modest margin while still requiring substantially lower error than the
median baseline (about $6.52). Callers can override the metric and threshold.
Passing candidates are saved under models/candidates/ and never replace the
existing Phase 2 model artifact.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.preprocessing import (
    FEATURE_COLUMNS,
    chronological_split,
    clean_taxi_data,
    engineer_features,
)
from src.train import build_training_metrics, fit_models, save_training_artifacts
from src.tracking import log_result, tracking_run, training_parameters
from src.utils import DATA_PATH, MODELS_DIR, SAMPLE_SIZE

QUALITY_METRIC = "mae"
QUALITY_THRESHOLD = 2.00
SUPPORTED_QUALITY_METRICS = {"mae", "rmse"}


class TaxiWorkflowState(TypedDict, total=False):
    """Inputs, intermediate execution values, final metrics, and status."""

    data_path: str
    sample_size: int | None
    quality_metric: str
    quality_threshold: float
    run_id: str
    status: str
    error: str | None
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    model: Any
    baseline_predictions: Any
    predictions: Any
    training_time_seconds: float
    metrics: dict[str, Any]
    artifact_paths: dict[str, str]
    mlflow_run_id: str


def _error(state: TaxiWorkflowState, stage: str, exc: Exception) -> dict[str, Any]:
    return {"status": "error", "error": f"{stage}: {type(exc).__name__}: {exc}"}


def validate_data(state: TaxiWorkflowState) -> dict[str, Any]:
    """Check the input file and required source columns before loading rows."""
    try:
        path = Path(state["data_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        columns = set(pd.read_csv(path, nrows=0).columns)
        required = {
            "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "passenger_count",
        }
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        metric = state["quality_metric"].lower()
        threshold = float(state["quality_threshold"])
        if metric not in SUPPORTED_QUALITY_METRICS:
            raise ValueError(f"quality_metric must be one of {sorted(SUPPORTED_QUALITY_METRICS)}")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("quality_threshold must be a finite positive number")
        return {
            "status": "validated",
            "error": None,
            "quality_metric": metric,
            "quality_threshold": threshold,
        }
    except Exception as exc:
        return _error(state, "data validation failed", exc)


def preprocess_data(state: TaxiWorkflowState) -> dict[str, Any]:
    """Reuse Phase 2 cleaning, feature generation, and chronological split."""
    try:
        raw = pd.read_csv(state["data_path"], nrows=state["sample_size"])
        cleaned = clean_taxi_data(raw)
        features = engineer_features(cleaned)
        train_df, val_df = chronological_split(features)
        if train_df.empty or val_df.empty:
            raise ValueError("Preprocessing produced an empty training or validation partition")
        return {
            "X_train": train_df[FEATURE_COLUMNS],
            "y_train": train_df["fare_amount"],
            "X_val": val_df[FEATURE_COLUMNS],
            "y_val": val_df["fare_amount"],
            "status": "preprocessed",
        }
    except Exception as exc:
        return _error(state, "preprocessing failed", exc)


def train_model(state: TaxiWorkflowState) -> dict[str, Any]:
    """Reuse the Phase 2 model-fitting function; do not save artifacts here."""
    try:
        fitted = fit_models(state["X_train"], state["y_train"], state["X_val"], state["y_val"])
        return {
            "model": fitted["model"],
            "baseline_predictions": fitted["baseline_predictions"],
            "predictions": fitted["predictions"],
            "training_time_seconds": fitted["training_time_seconds"],
            "status": "trained",
        }
    except Exception as exc:
        return _error(state, "model training failed", exc)


def evaluate_model(state: TaxiWorkflowState) -> dict[str, Any]:
    """Reuse Phase 2 metric definitions and attach the selected gate result."""
    try:
        metrics = build_training_metrics(
            state["y_val"], state["baseline_predictions"], state["predictions"],
            state["training_time_seconds"], len(state["X_train"]), len(state["X_val"]),
        )
        return {"metrics": metrics, "status": "evaluated"}
    except Exception as exc:
        return _error(state, "model evaluation failed", exc)


def route_after_evaluation(state: TaxiWorkflowState) -> str:
    """Send only candidates within the configured error threshold to saving."""
    if state.get("status") != "evaluated" or state.get("error"):
        return "stop"
    score = state["metrics"]["evaluation_metrics"]["trained_model"][state["quality_metric"]]
    return "save" if score <= state["quality_threshold"] else "reject"


def reject_candidate(state: TaxiWorkflowState) -> dict[str, Any]:
    score = state["metrics"]["evaluation_metrics"]["trained_model"][state["quality_metric"]]
    message = (
        f"Quality gate failed: validation {state['quality_metric'].upper()} "
        f"{score:.4f} exceeds threshold {state['quality_threshold']:.4f}; "
        "candidate artifacts were not saved."
    )
    return {"status": "quality_failed", "error": message}


def save_candidate(state: TaxiWorkflowState) -> dict[str, Any]:
    """Save a passing candidate in a run-specific directory, preserving champion."""
    try:
        candidate_dir = MODELS_DIR / "candidates" / state["run_id"]
        paths = save_training_artifacts(
            state["model"],
            state["metrics"],
            model_path=candidate_dir / "taxi_fare_model.joblib",
            features_path=candidate_dir / "model_features.json",
            metrics_path=candidate_dir / "metrics.json",
        )
        return {"artifact_paths": paths, "status": "completed", "error": None}
    except Exception as exc:
        return _error(state, "candidate saving failed", exc)


def _continue_or_stop(state: TaxiWorkflowState) -> str:
    return "continue" if not state.get("error") else "stop"


def _build_graph():
    graph = StateGraph(TaxiWorkflowState)
    graph.add_node("validate_data", validate_data)
    graph.add_node("preprocess_data", preprocess_data)
    graph.add_node("train_model", train_model)
    graph.add_node("evaluate_model", evaluate_model)
    graph.add_node("save_candidate", save_candidate)
    graph.add_node("reject_candidate", reject_candidate)

    graph.add_edge(START, "validate_data")
    for source, target in (
        ("validate_data", "preprocess_data"),
        ("preprocess_data", "train_model"),
        ("train_model", "evaluate_model"),
    ):
        graph.add_conditional_edges(
            source,
            _continue_or_stop,
            {"continue": target, "stop": END},
        )
    graph.add_conditional_edges(
        "evaluate_model",
        route_after_evaluation,
        {"save": "save_candidate", "reject": "reject_candidate", "stop": END},
    )
    graph.add_edge("save_candidate", END)
    graph.add_edge("reject_candidate", END)
    return graph.compile()


workflow_graph = _build_graph()


def run_workflow(
    data_path=DATA_PATH,
    sample_size=SAMPLE_SIZE,
    quality_metric=QUALITY_METRIC,
    quality_threshold=QUALITY_THRESHOLD,
) -> TaxiWorkflowState:
    """Invoke the compiled graph and record one local MLflow run for the attempt."""
    initial_state: TaxiWorkflowState = {
        "data_path": str(data_path),
        "sample_size": sample_size,
        "quality_metric": str(quality_metric).lower(),
        "quality_threshold": quality_threshold,
        "run_id": uuid.uuid4().hex,
        "status": "started",
        "error": None,
    }
    with tracking_run("taxi-fare-langgraph", {"orchestrator": "langgraph"}) as (mlflow, run):
        mlflow.log_params(training_parameters(sample_size, quality_metric, quality_threshold))
        try:
            result = workflow_graph.invoke(initial_state)
        except Exception as exc:
            result = {
                **initial_state,
                "status": "error",
                "error": f"workflow execution failed: {type(exc).__name__}: {exc}",
            }
        if "metrics" in result:
            log_result(
                result["metrics"],
                model=result.get("model"),
                status=result.get("status", "error"),
                error=result.get("error"),
            )
        else:
            mlflow.set_tag("workflow_status", result.get("status", "error"))
            if result.get("error"):
                mlflow.set_tag("workflow_error", result["error"][:500])
        result["mlflow_run_id"] = run.info.run_id
        return result


if __name__ == "__main__":
    result = run_workflow()
    print({key: result[key] for key in ("status", "error", "metrics", "artifact_paths", "mlflow_run_id") if key in result})
