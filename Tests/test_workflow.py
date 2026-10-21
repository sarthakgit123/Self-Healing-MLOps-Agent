import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

import src.tracking as tracking
import src.workflow as workflow


def _run_artifact_paths(run_id):
    """List artifacts through the supported MLflow tracking client API."""
    client = mlflow.tracking.MlflowClient()
    return client.list_artifacts(run_id)


def _write_sample_csv(path, n=80):
    index = np.arange(n)
    timestamps = pd.date_range("2015-01-01", periods=n, freq="h")
    frame = pd.DataFrame({
        "fare_amount": 5 + index % 15 + np.linspace(0, 2, n),
        "pickup_datetime": timestamps.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pickup_longitude": -73.99 + (index % 8) * 0.002,
        "pickup_latitude": 40.70 + (index % 7) * 0.002,
        "dropoff_longitude": -73.94 + (index % 6) * 0.002,
        "dropoff_latitude": 40.75 + (index % 5) * 0.002,
        "passenger_count": np.ones(n, dtype=int),
    })
    frame.to_csv(path, index=False)
    return path


def test_workflow_saves_passing_candidate_and_preserves_champion(tmp_path, monkeypatch):
    data_path = _write_sample_csv(tmp_path / "sample.csv")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    champion_path = model_dir / "taxi_fare_model.joblib"
    champion_path.write_bytes(b"existing champion")
    monkeypatch.setattr(workflow, "MODELS_DIR", model_dir)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")

    result = workflow.run_workflow(
        data_path=data_path, sample_size=None, quality_threshold=1_000.0
    )

    assert result["status"] == "completed"
    assert result["error"] is None
    candidate_files = result["artifact_paths"]
    assert all(Path(path).is_file() for path in candidate_files.values())
    assert champion_path.read_bytes() == b"existing champion"
    loaded_model = joblib.load(candidate_files["model"])
    assert np.allclose(result["predictions"], loaded_model.predict(result["X_val"]))
    metrics = json.loads(open(candidate_files["metrics"], encoding="utf-8").read())
    assert metrics["evaluation_metrics"]["trained_model"]["mae"] == result["metrics"]["evaluation_metrics"]["trained_model"]["mae"]
    run = mlflow.get_run(result["mlflow_run_id"])
    assert run.data.tags["workflow_status"] == "completed"
    assert "candidate_mae" in run.data.metrics
    assert any(item.path == "candidate_model" for item in _run_artifact_paths(result["mlflow_run_id"]))


def test_workflow_quality_gate_rejects_without_saving(tmp_path, monkeypatch):
    data_path = _write_sample_csv(tmp_path / "sample.csv")
    model_dir = tmp_path / "models"
    monkeypatch.setattr(workflow, "MODELS_DIR", model_dir)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")

    result = workflow.run_workflow(
        data_path=data_path, sample_size=None, quality_threshold=0.000001
    )

    assert result["status"] == "quality_failed"
    assert "Quality gate failed" in result["error"]
    assert "metrics" in result
    assert not (model_dir / "candidates").exists()
    run = mlflow.get_run(result["mlflow_run_id"])
    assert run.data.tags["workflow_status"] == "quality_failed"
    assert "candidate_mae" in run.data.metrics
    assert any(item.path == "candidate_model" for item in _run_artifact_paths(result["mlflow_run_id"]))


def test_workflow_exposes_validation_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")
    result = workflow.run_workflow(
        data_path=tmp_path / "missing.csv", sample_size=None
    )

    assert result["status"] == "error"
    assert "data validation failed" in result["error"]
    assert "Dataset not found" in result["error"]
    run = mlflow.get_run(result["mlflow_run_id"])
    assert run.data.tags["workflow_status"] == "error"
