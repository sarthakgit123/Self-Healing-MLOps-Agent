import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import src.tracking as tracking
from src.evaluate import regression_metrics
from src.train import run_training
from src.utils import save_json


def test_metrics_known_values():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["mae"] == 1 / 3
    assert np.isclose(metrics["rmse"], np.sqrt(1 / 3))
    assert np.isclose(metrics["r2"], 0.5)


def test_small_training_save_reload_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")
    n = 80
    timestamps = pd.date_range("2015-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "fare_amount": 5 + np.arange(n) % 15 + np.linspace(0, 2, n),
        "pickup_datetime": timestamps.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pickup_longitude": -73.99 + (np.arange(n) % 8) * 0.002,
        "pickup_latitude": 40.70 + (np.arange(n) % 7) * 0.002,
        "dropoff_longitude": -73.94 + (np.arange(n) % 6) * 0.002,
        "dropoff_latitude": 40.75 + (np.arange(n) % 5) * 0.002,
        "passenger_count": np.ones(n, dtype=int),
    })
    data_path = tmp_path / "train.csv"
    df.to_csv(data_path, index=False)
    result = run_training(data_path=data_path, sample_size=None, save_artifacts=False)
    model_path, metrics_path = tmp_path / "model.joblib", tmp_path / "metrics.json"
    joblib.dump(result["model"], model_path)
    save_json(result["metrics"], metrics_path)
    loaded = joblib.load(model_path)
    assert np.allclose(result["predictions"], loaded.predict(result["X_val"]))
    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded_metrics["evaluation_metrics"]["trained_model"]["mae"] >= 0
    assert model_path.stat().st_size > 0
    experiment = mlflow.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.workflow_status"] == "completed"
    assert "metrics.candidate_mae" in runs.columns
