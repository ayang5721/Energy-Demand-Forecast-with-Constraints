"""Feature engineering for the 24-hour-ahead milestone forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "load_area",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "load_mw",
    "load_lag_1",
    "load_lag_24",
    "load_lag_48",
    "rolling_mean_24",
    "rolling_std_24",
]
FORECAST_HORIZON = 24
LOAD_LAGS = [1, 24, 48]
ROLLING_WINDOWS = [24]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, weekday, month, and weekend indicators from EPT timestamps."""
    out = df.copy()
    out["hour"] = out["timestamp_ept"].dt.hour
    out["day_of_week"] = out["timestamp_ept"].dt.dayofweek
    out["month"] = out["timestamp_ept"].dt.month
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings for hour of day and day of week."""
    out = df.copy()
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_day_of_week"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["cos_day_of_week"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add load lag features separately for each load area."""
    out = df.copy().sort_values(["load_area", "timestamp_utc"])
    grouped = out.groupby("load_area", sort=False)["load_mw"]
    for lag in lags:
        out[f"load_lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean and standard deviation features by load area."""
    out = df.copy().sort_values(["load_area", "timestamp_utc"])
    grouped = out.groupby("load_area", sort=False)["load_mw"]
    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=window)
        out[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        out[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
    return out


def add_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Add same-load-area target values and timestamps at the forecast horizon."""
    out = df.copy().sort_values(["load_area", "timestamp_utc"])
    grouped = out.groupby("load_area", sort=False)
    out["target_load_mw"] = grouped["load_mw"].shift(-horizon)
    out["target_timestamp_utc"] = grouped["timestamp_utc"].shift(-horizon)
    out["target_timestamp_ept"] = grouped["timestamp_ept"].shift(-horizon)
    return out


def make_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full milestone feature pipeline and drop incomplete rows."""
    out = add_time_features(df)
    out = add_cyclical_features(out)
    out = add_lag_features(out, LOAD_LAGS)
    out = add_rolling_features(out, ROLLING_WINDOWS)
    out = add_target(out, horizon=FORECAST_HORIZON)

    required = FEATURE_COLUMNS + [
        "timestamp_utc",
        "timestamp_ept",
        "target_timestamp_utc",
        "target_timestamp_ept",
        "zone",
        "target_load_mw",
    ]
    before = len(out)
    out = out.dropna(subset=required)
    out = out.sort_values(["timestamp_utc", "load_area"]).reset_index(drop=True)

    max_history = max(max(LOAD_LAGS), max(window - 1 for window in ROLLING_WINDOWS))
    expected_rows = (
        df.groupby("load_area").size()
        .sub(max_history + FORECAST_HORIZON)
        .clip(lower=0)
        .sum()
    )
    if len(out) != expected_rows:
        dropped = before - len(out)
        print(f"WARNING: Feature rows are {len(out)}, expected {expected_rows}. Dropped {dropped} incomplete rows.")
    return out
