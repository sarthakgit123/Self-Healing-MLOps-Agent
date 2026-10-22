import json
import sqlite3

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.prediction_store import PredictionStore
from src.preprocessing import FEATURE_COLUMNS
from src.utils import FEATURES_PATH, MODEL_PATH


class FeatureCheckingModel:
    def predict(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        assert len(frame) == 1
        return np.array([18.75])


@pytest.fixture
def model_artifacts(tmp_path):
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "model_features.json"
    joblib.dump(FeatureCheckingModel(), model_path)
    features_path.write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    return model_path, features_path


@pytest.fixture
def client(model_artifacts):
    model_path, features_path = model_artifacts
    database_path = model_path.parent / "predictions.sqlite3"
    with TestClient(create_app(model_path, features_path, database_path)) as test_client:
        yield test_client


def valid_request():
    return {
        "pickup_datetime": "2015-01-01 12:00:00 UTC",
        "pickup_longitude": -73.99,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.98,
        "dropoff_latitude": 40.76,
        "passenger_count": 1,
    }


def submit_prediction(client):
    response = client.post("/predict", json=valid_request())
    assert response.status_code == 200
    return response.json()


def test_health_reports_ready_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready", "model_ready": True, "storage_ready": True
    }


def test_predict_returns_fare_and_uses_saved_feature_order(client):
    response = client.post("/predict", json=valid_request())
    assert response.status_code == 200
    assert response.json()["predicted_fare"] == 18.75
    assert response.json()["prediction_id"]
    assert response.json()["model_identifier"].startswith("sha256:")


def test_prediction_is_persisted_with_inputs_fare_time_and_model(client):
    response = client.post("/predict", json=valid_request())
    result = response.json()
    stored = client.app.state.prediction_store.get_prediction(result["prediction_id"])
    assert stored["input_features"] == valid_request()
    assert stored["predicted_fare"] == result["predicted_fare"]
    assert stored["created_at"].endswith("+00:00")
    assert stored["model_identifier"] == result["model_identifier"]


def test_each_prediction_gets_a_unique_id(client):
    first = client.post("/predict", json=valid_request()).json()
    second = client.post("/predict", json=valid_request()).json()
    assert first["prediction_id"] != second["prediction_id"]


def test_prediction_record_survives_app_restart(model_artifacts):
    model_path, features_path = model_artifacts
    database_path = model_path.parent / "persistent.sqlite3"
    with TestClient(create_app(model_path, features_path, database_path)) as first_client:
        response = first_client.post("/predict", json=valid_request())
        prediction_id = response.json()["prediction_id"]

    with TestClient(create_app(model_path, features_path, database_path)) as restarted_client:
        assert restarted_client.get("/health").status_code == 200
        saved = restarted_client.app.state.prediction_store.get_prediction(prediction_id)
    assert saved is not None
    assert saved["prediction_id"] == prediction_id


def test_feedback_is_saved_and_returns_absolute_error(client):
    prediction = submit_prediction(client)
    response = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_fare": 20.0},
    )
    assert response.status_code == 200
    assert response.json()["prediction_id"] == prediction["prediction_id"]
    assert response.json()["predicted_fare"] == 18.75
    assert response.json()["actual_fare"] == 20.0
    assert response.json()["absolute_error"] == 1.25

    saved = client.app.state.prediction_store.get_prediction(prediction["prediction_id"])
    assert saved["actual_fare"] == 20.0
    assert saved["absolute_error"] == 1.25
    assert saved["feedback_at"].endswith("+00:00")


def test_feedback_rejects_unknown_prediction_id(client):
    response = client.post(
        "/feedback", json={"prediction_id": "unknown", "actual_fare": 20.0}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction ID was not found"


def test_feedback_rejects_duplicate_submission(client):
    prediction = submit_prediction(client)
    payload = {"prediction_id": prediction["prediction_id"], "actual_fare": 20.0}
    assert client.post("/feedback", json=payload).status_code == 200
    duplicate = client.post("/feedback", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Feedback has already been recorded for this prediction"
    stored = client.app.state.prediction_store.get_prediction(prediction["prediction_id"])
    assert stored["actual_fare"] == 20.0
    assert stored["absolute_error"] == 1.25


def test_end_to_end_health_predict_feedback_flow(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"

    prediction_response = client.post("/predict", json=valid_request())
    assert prediction_response.status_code == 200
    prediction = prediction_response.json()
    assert prediction["prediction_id"]
    assert prediction["predicted_fare"] == 18.75

    store = client.app.state.prediction_store
    saved_prediction = store.get_prediction(prediction["prediction_id"])
    assert saved_prediction is not None
    assert saved_prediction["input_features"] == valid_request()
    assert saved_prediction["predicted_fare"] == prediction["predicted_fare"]

    feedback_response = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_fare": 20.0},
    )
    assert feedback_response.status_code == 200
    feedback = feedback_response.json()
    assert feedback["prediction_id"] == prediction["prediction_id"]
    assert feedback["predicted_fare"] == 18.75
    assert feedback["actual_fare"] == 20.0
    assert feedback["absolute_error"] == 1.25

    saved_feedback = store.get_prediction(prediction["prediction_id"])
    assert saved_feedback["actual_fare"] == 20.0
    assert saved_feedback["absolute_error"] == 1.25
    assert saved_feedback["feedback_at"].endswith("+00:00")


@pytest.mark.parametrize(
    "actual_fare", [-0.01, "20.0", True, float("nan"), 10**400]
)
def test_feedback_rejects_invalid_fares(client, actual_fare):
    prediction = submit_prediction(client)
    response = client.post(
        "/feedback",
        content=json.dumps(
            {"prediction_id": prediction["prediction_id"], "actual_fare": actual_fare}
        ),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_feedback_persists_across_app_restart(model_artifacts):
    model_path, features_path = model_artifacts
    database_path = model_path.parent / "feedback_persistent.sqlite3"
    with TestClient(create_app(model_path, features_path, database_path)) as first_client:
        prediction = submit_prediction(first_client)
        feedback = first_client.post(
            "/feedback",
            json={"prediction_id": prediction["prediction_id"], "actual_fare": 21.0},
        )
        assert feedback.status_code == 200

    with TestClient(create_app(model_path, features_path, database_path)) as restarted_client:
        saved = restarted_client.app.state.prediction_store.get_prediction(
            prediction["prediction_id"]
        )
    assert saved["actual_fare"] == 21.0
    assert saved["absolute_error"] == 2.25


def test_feedback_database_failure_rolls_back_record(client):
    prediction = submit_prediction(client)
    database_path = client.app.state.prediction_store.database_path
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_feedback_update
            BEFORE UPDATE OF actual_fare ON predictions
            WHEN NEW.actual_fare IS NOT NULL
            BEGIN
                SELECT RAISE(ABORT, 'simulated database failure');
            END;
            """
        )

    response = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_fare": 20.0},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Feedback could not be stored"
    saved = client.app.state.prediction_store.get_prediction(prediction["prediction_id"])
    assert saved["actual_fare"] is None
    assert saved["feedback_at"] is None
    assert saved["absolute_error"] is None


def test_existing_prediction_table_is_migrated_for_feedback(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE predictions (
                prediction_id TEXT PRIMARY KEY,
                input_features_json TEXT NOT NULL,
                predicted_fare REAL NOT NULL,
                created_at TEXT NOT NULL,
                model_identifier TEXT
            )
            """
        )

    PredictionStore(database_path).initialize()
    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    assert {"actual_fare", "feedback_at", "absolute_error"} <= columns


@pytest.mark.parametrize(
    "updates",
    [
        {"passenger_count": 7},
        {"passenger_count": 1.5},
        {"pickup_longitude": -80.0},
        {"pickup_longitude": "-73.99"},
        {"pickup_latitude": float("nan")},
        {"pickup_datetime": "not-a-date"},
    ],
)
def test_predict_rejects_invalid_values(client, updates):
    request = valid_request()
    request.update(updates)
    response = client.post(
        "/predict",
        content=json.dumps(request),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    request = valid_request()
    del request["passenger_count"]
    response = client.post("/predict", json=request)
    assert response.status_code == 422


def test_missing_model_artifact_reports_not_ready(tmp_path):
    app = create_app(
        tmp_path / "missing.joblib",
        tmp_path / "missing_features.json",
        tmp_path / "predictions.sqlite3",
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["detail"]["status"] == "not_ready"
        prediction = client.post("/predict", json=valid_request())
        assert prediction.status_code == 503
        assert prediction.json()["detail"]["message"] == "Model artifact is unavailable"


def test_database_write_error_returns_clear_server_error(client, monkeypatch):
    def fail_write(*args, **kwargs):
        raise sqlite3.OperationalError("disk is unavailable")

    monkeypatch.setattr(PredictionStore, "save_prediction", fail_write)
    response = client.post("/predict", json=valid_request())
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction could not be stored"


@pytest.mark.skipif(
    not MODEL_PATH.is_file() or not FEATURES_PATH.is_file(),
    reason="The repository's local champion model artifacts are not present",
)
def test_real_saved_model_integration(tmp_path):
    with TestClient(create_app(database_path=tmp_path / "predictions.sqlite3")) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=valid_request())
    assert response.status_code == 200
    assert np.isfinite(response.json()["predicted_fare"])
    assert response.json()["prediction_id"]
