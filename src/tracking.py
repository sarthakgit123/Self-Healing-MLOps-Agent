"""Local MLflow helpers shared by direct training and LangGraph runs."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Any

import joblib
import mlflow

from src.preprocessing import FEATURE_COLUMNS
from src.utils import PROJECT_ROOT, XGB_PARAMS

EXPERIMENT_NAME = "NYC Taxi Fare Prediction"
TRACKING_DIR = PROJECT_ROOT / "mlruns"


def configure_tracking():
    """Select the project-local file store and named experiment."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())
    return mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def tracking_run(run_name: str, tags: dict[str, str] | None = None):
    """Create one run in the local experiment; callers own the run boundary."""
    configure_tracking()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"project": EXPERIMENT_NAME, **(tags or {})})
        yield mlflow, run


def training_parameters(sample_size, quality_metric=None, quality_threshold=None):
    """Return comparable model and workflow configuration as MLflow params."""
    params: dict[str, Any] = dict(XGB_PARAMS)
    params.update({
        "sample_size": "all rows" if sample_size is None else sample_size,
        "validation_split": "chronological 80/20",
        "random_seed": XGB_PARAMS["random_state"],
        "features_count": len(FEATURE_COLUMNS),
    })
    if quality_metric is not None:
        params["quality_metric"] = quality_metric
    if quality_threshold is not None:
        params["quality_threshold"] = quality_threshold
    return {key: str(value) if value is None else value for key, value in params.items()}


def log_result(metrics: dict[str, Any], model=None, status: str = "completed", error=None):
    """Log comparable metrics and JSON artifacts; log a model only on success."""
    evaluation = metrics.get("evaluation_metrics", {})
    baseline = evaluation.get("baseline_median", {})
    candidate = evaluation.get("trained_model", {})
    improvement = evaluation.get("improvement", {})
    numeric_metrics = {
        "baseline_mae": baseline.get("mae"),
        "baseline_rmse": baseline.get("rmse"),
        "baseline_r2": baseline.get("r2"),
        "candidate_mae": candidate.get("mae"),
        "candidate_rmse": candidate.get("rmse"),
        "candidate_r2": candidate.get("r2"),
        "mae_reduction_pct": improvement.get("mae_reduction_pct"),
        "rmse_reduction_pct": improvement.get("rmse_reduction_pct"),
        "training_samples": metrics.get("training_samples"),
        "validation_samples": metrics.get("validation_samples"),
        "features_count": metrics.get("features_count"),
        "training_time_seconds": metrics.get("training_time_seconds"),
    }
    mlflow.log_metrics({key: float(value) for key, value in numeric_metrics.items() if value is not None})
    mlflow.log_dict(metrics, "evaluation_metrics.json")
    mlflow.log_dict({"features": FEATURE_COLUMNS}, "model_features.json")
    mlflow.set_tag("workflow_status", status)
    if error:
        mlflow.set_tag("workflow_error", str(error)[:500])
    # Keep every trained candidate available for comparison, including one
    # rejected by the quality gate. This is an MLflow run artifact, not a
    # replacement for the project's accepted candidate/champion files.
    if model is not None:
        # MLflow 3 stores logged models separately from a run's legacy
        # artifact listing. Keep the loadable MLflow model and also place a
        # serialized copy under the run artifact path for easy discovery.
        mlflow.xgboost.log_model(model, name="candidate_model")
        with tempfile.TemporaryDirectory(prefix="taxi-candidate-") as temp_dir:
            model_file = Path(temp_dir) / "taxi_fare_model.joblib"
            joblib.dump(model, model_file)
            mlflow.log_artifact(str(model_file), artifact_path="candidate_model")
