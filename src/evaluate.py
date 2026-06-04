import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y == 0, np.finfo(float).eps, y)
    return float(np.mean(np.abs((y - pred) / denom)) * 100)


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def _metrics_series(group: pd.DataFrame) -> pd.Series:
    row = compute_metrics(group["true_load_mw"], group["predicted_load_mw"])
    row["n"] = len(group)
    return pd.Series(row)


def make_metrics_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    return predictions_df.groupby("model", sort=False).apply(_metrics_series).reset_index()


def make_metrics_by_load_area(predictions_df: pd.DataFrame) -> pd.DataFrame:
    return predictions_df.groupby(["model", "zone", "load_area"], sort=False).apply(_metrics_series).reset_index()


def make_error_by_hour(predictions_df: pd.DataFrame) -> pd.DataFrame:
    def hourly(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "mean_abs_error": group["abs_error_mw"].mean(),
                "mean_error": group["error_mw"].mean(),
                "rmse": rmse(group["true_load_mw"], group["predicted_load_mw"]),
            }
        )

    return predictions_df.groupby(["model", "hour"], sort=False).apply(hourly).reset_index()
