"""FastAPI serving for the saved NYC taxi fare model."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from src.preprocessing import FEATURE_COLUMNS, engineer_features
from src.prediction_store import (
    DuplicateFeedbackError,
    PredictionStore,
    UnknownPredictionError,
)
from src.utils import (
    FEATURES_PATH,
    MODEL_PATH,
    NYC_LAT_BOUNDS,
    NYC_LON_BOUNDS,
    PASSENGER_BOUNDS,
    PREDICTIONS_DB_PATH,
)

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Raw trip inputs; derived features are built with training code."""

    model_config = ConfigDict(extra="forbid")

    pickup_datetime: str
    pickup_longitude: float = Field(
        ge=NYC_LON_BOUNDS[0], le=NYC_LON_BOUNDS[1], strict=True, allow_inf_nan=False
    )
    pickup_latitude: float = Field(
        ge=NYC_LAT_BOUNDS[0], le=NYC_LAT_BOUNDS[1], strict=True, allow_inf_nan=False
    )
    dropoff_longitude: float = Field(
        ge=NYC_LON_BOUNDS[0], le=NYC_LON_BOUNDS[1], strict=True, allow_inf_nan=False
    )
    dropoff_latitude: float = Field(
        ge=NYC_LAT_BOUNDS[0], le=NYC_LAT_BOUNDS[1], strict=True, allow_inf_nan=False
    )
    passenger_count: Annotated[
        StrictInt,
        Field(ge=PASSENGER_BOUNDS[0], le=PASSENGER_BOUNDS[1]),
    ]

    @field_validator("pickup_datetime")
    @classmethod
    def validate_pickup_datetime(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC")
        except (TypeError, ValueError) as exc:
            raise ValueError("must use format YYYY-MM-DD HH:MM:SS UTC") from exc
        return value


class FeedbackRequest(BaseModel):
    """Actual fare supplied later for a previously served prediction."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(min_length=1)
    actual_fare: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("actual_fare", mode="before")
    @classmethod
    def validate_actual_fare_type(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("actual_fare must be a JSON number")
        try:
            numeric_value = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("actual_fare must be a finite number") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("actual_fare must be finite")
        return numeric_value


def load_model_artifacts(model_path: Path, features_path: Path):
    """Load the existing joblib model and verify its saved feature schema."""
    model_path = Path(model_path)
    features_path = Path(features_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature schema not found: {features_path}")

    saved_features = json.loads(features_path.read_text(encoding="utf-8"))
    if saved_features != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema does not match src.preprocessing.FEATURE_COLUMNS")

    model = joblib.load(model_path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model artifact does not provide a predict() method")
    return model


def _model_identifier(model_path: Path) -> str:
    """Return a stable identifier for the exact model artifact loaded."""
    digest = hashlib.sha256()
    with Path(model_path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _features_from_request(request: PredictionRequest) -> pd.DataFrame:
    raw = pd.DataFrame([request.model_dump()])
    engineered = engineer_features(raw)
    if engineered.empty:
        raise HTTPException(
            status_code=422,
            detail="Trip is invalid for this model: pickup and drop-off must be more than 50 meters apart.",
        )
    return engineered.loc[:, FEATURE_COLUMNS]


def create_app(
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
    database_path: Path = PREDICTIONS_DB_PATH,
) -> FastAPI:
    """Create an app that loads the saved model once at startup, without training."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_error = None
        app.state.model_identifier = None
        app.state.prediction_store = None
        app.state.storage_error = None
        try:
            app.state.model = load_model_artifacts(model_path, features_path)
            app.state.model_identifier = _model_identifier(model_path)
        except Exception as exc:
            app.state.model_error = str(exc)
            logger.exception("Model artifacts could not be loaded")
        try:
            store = PredictionStore(database_path)
            store.initialize()
            app.state.prediction_store = store
        except Exception as exc:
            app.state.storage_error = str(exc)
            logger.exception("Prediction storage could not be initialized")
        yield

    app = FastAPI(title="NYC Taxi Fare Prediction", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        # Do not echo invalid input values; they may include non-finite floats
        # which cannot be represented in a standards-compliant JSON response.
        details = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health")
    def health():
        if app.state.model is None or app.state.prediction_store is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "model_ready": app.state.model is not None,
                    "storage_ready": app.state.prediction_store is not None,
                    "model_error": app.state.model_error,
                    "storage_error": app.state.storage_error,
                },
            )
        return {"status": "ready", "model_ready": True, "storage_ready": True}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        model = app.state.model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail={"message": "Model artifact is unavailable", "error": app.state.model_error},
            )
        features = _features_from_request(request)
        if app.state.prediction_store is None:
            raise HTTPException(status_code=503, detail="Prediction storage is unavailable")
        try:
            prediction = float(model.predict(features)[0])
        except Exception as exc:
            logger.exception("Model prediction failed")
            raise HTTPException(status_code=500, detail="Model prediction failed") from exc
        try:
            record = app.state.prediction_store.save_prediction(
                input_features=request.model_dump(),
                predicted_fare=prediction,
                model_identifier=app.state.model_identifier,
            )
        except Exception as exc:
            logger.exception("Prediction was made but could not be persisted")
            raise HTTPException(status_code=500, detail="Prediction could not be stored") from exc
        return {
            "prediction_id": record["prediction_id"],
            "predicted_fare": prediction,
            "model_identifier": app.state.model_identifier,
        }

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        store = app.state.prediction_store
        if store is None:
            raise HTTPException(status_code=503, detail="Prediction storage is unavailable")
        try:
            return store.save_feedback(request.prediction_id, request.actual_fare)
        except UnknownPredictionError as exc:
            raise HTTPException(status_code=404, detail="Prediction ID was not found") from exc
        except DuplicateFeedbackError as exc:
            raise HTTPException(
                status_code=409, detail="Feedback has already been recorded for this prediction"
            ) from exc
        except sqlite3.Error as exc:
            logger.exception("Feedback could not be persisted")
            raise HTTPException(status_code=500, detail="Feedback could not be stored") from exc

    return app


app = create_app()
