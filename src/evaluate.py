"""Regression metrics and model comparison."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE, and R² as plain Python floats."""
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def compare_metrics(baseline, model):
    return {"baseline_median": baseline, "trained_model": model,
            "improvement": {"mae_reduction_pct": float((1 - model["mae"] / baseline["mae"]) * 100),
                            "rmse_reduction_pct": float((1 - model["rmse"] / baseline["rmse"]) * 100)}}
