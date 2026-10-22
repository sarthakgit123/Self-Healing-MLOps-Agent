"""Small SQLite persistence layer for served fare predictions."""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UnknownPredictionError(LookupError):
    """Raised when feedback references an ID absent from the prediction table."""


class DuplicateFeedbackError(ValueError):
    """Raised when actual-fare feedback was already recorded for a prediction."""


class PredictionStore:
    """Persist prediction records using a short-lived SQLite connection per operation."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    input_features_json TEXT NOT NULL,
                    predicted_fare REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    model_identifier TEXT,
                    actual_fare REAL,
                    feedback_at TEXT,
                    absolute_error REAL
                )
                """
            )
            existing_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(predictions)")
            }
            for column, sql_type in (
                ("actual_fare", "REAL"),
                ("feedback_at", "TEXT"),
                ("absolute_error", "REAL"),
            ):
                if column not in existing_columns:
                    connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {sql_type}")

    def save_prediction(
        self,
        input_features: dict[str, Any],
        predicted_fare: float,
        model_identifier: str | None,
    ) -> dict[str, Any]:
        record = {
            "prediction_id": str(uuid.uuid4()),
            "input_features": input_features,
            "predicted_fare": float(predicted_fare),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_identifier": model_identifier,
        }
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO predictions
                    (prediction_id, input_features_json, predicted_fare, created_at, model_identifier)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record["prediction_id"],
                    json.dumps(input_features, allow_nan=False),
                    record["predicted_fare"],
                    record["created_at"],
                    record["model_identifier"],
                ),
            )
        return record

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "prediction_id": row["prediction_id"],
            "input_features": json.loads(row["input_features_json"]),
            "predicted_fare": row["predicted_fare"],
            "created_at": row["created_at"],
            "model_identifier": row["model_identifier"],
            "actual_fare": row["actual_fare"],
            "feedback_at": row["feedback_at"],
            "absolute_error": row["absolute_error"],
        }

    def save_feedback(self, prediction_id: str, actual_fare: float) -> dict[str, Any]:
        """Attach actual fare once and calculate absolute error atomically."""
        actual_fare = float(actual_fare)
        if not math.isfinite(actual_fare) or actual_fare < 0:
            raise ValueError("actual_fare must be finite and non-negative")

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT predicted_fare, actual_fare FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                raise UnknownPredictionError(prediction_id)
            if row["actual_fare"] is not None:
                raise DuplicateFeedbackError(prediction_id)

            absolute_error = abs(float(row["predicted_fare"]) - actual_fare)
            feedback_at = datetime.now(timezone.utc).isoformat()
            updated = connection.execute(
                """
                UPDATE predictions
                SET actual_fare = ?, feedback_at = ?, absolute_error = ?
                WHERE prediction_id = ? AND actual_fare IS NULL
                """,
                (actual_fare, feedback_at, absolute_error, prediction_id),
            )
            if updated.rowcount != 1:
                raise DuplicateFeedbackError(prediction_id)

            return {
                "prediction_id": prediction_id,
                "predicted_fare": float(row["predicted_fare"]),
                "actual_fare": actual_fare,
                "absolute_error": absolute_error,
                "feedback_at": feedback_at,
            }
