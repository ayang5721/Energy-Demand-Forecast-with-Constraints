"""Forecast metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Return mean absolute percentage error as a percent."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.where(true == 0, np.finfo(float).eps, true)
    return float(np.mean(np.abs((true - pred) / denom)) * 100)


def bias(y_true, y_pred) -> float:
    """Return mean signed forecast error, prediction minus truth."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def compute_metrics(y_true, y_pred) -> dict:
    """Return standard milestone metrics."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def _metrics_series(group: pd.DataFrame) -> pd.Series:
    metrics = compute_metrics(group["true_load_mw"], group["predicted_load_mw"])
    metrics["n"] = len(group)
    return pd.Series(metrics)


def make_metrics_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forecast metrics by model."""
    return predictions_df.groupby("model", sort=False).apply(_metrics_series).reset_index()


def make_metrics_by_load_area(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forecast metrics by model, zone, and load area."""
    return predictions_df.groupby(["model", "zone", "load_area"], sort=False).apply(_metrics_series).reset_index()


def make_error_by_hour(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly mean absolute error, signed error, and RMSE by model."""
    def hourly(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "mean_abs_error": group["abs_error_mw"].mean(),
                "mean_error": group["error_mw"].mean(),
                "rmse": rmse(group["true_load_mw"], group["predicted_load_mw"]),
            }
        )

    return predictions_df.groupby(["model", "hour"], sort=False).apply(hourly).reset_index()
