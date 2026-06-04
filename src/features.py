import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "zone",
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
WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "temperature_lag_24",
    "humidity_lag_24",
]
WEATHER_FEATURE_COLUMNS = FEATURE_COLUMNS + WEATHER_COLUMNS
FORECAST_HORIZON = 24
LOAD_LAGS = [1, 24, 48]
ROLLING_WINDOWS = [24]
WEATHER_LAG_HOURS = 24


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp_ept"].dt.hour
    out["day_of_week"] = out["timestamp_ept"].dt.dayofweek
    out["month"] = out["timestamp_ept"].dt.month
    out["is_weekend"] = out["day_of_week"].isin((5, 6)).astype(int)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour_angle = 2 * np.pi * out["hour"] / 24
    dow_angle = 2 * np.pi * out["day_of_week"] / 7
    out["sin_hour"] = np.sin(hour_angle)
    out["cos_hour"] = np.cos(hour_angle)
    out["sin_day_of_week"] = np.sin(dow_angle)
    out["cos_day_of_week"] = np.cos(dow_angle)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    lookup = out[["zone", "load_area", "timestamp_utc", "load_mw"]].copy()
    for lag in lags:
        lagged = lookup.rename(
            columns={
                "timestamp_utc": "lag_timestamp_utc",
                "load_mw": f"load_lag_{lag}",
            }
        )
        out["lag_timestamp_utc"] = out["timestamp_utc"] - pd.Timedelta(hours=lag)
        out = out.merge(lagged, on=["zone", "load_area", "lag_timestamp_utc"], how="left")
        out = out.drop(columns=["lag_timestamp_utc"])
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy().sort_values(["zone", "load_area", "timestamp_utc"])
    by_area = out.groupby(["zone", "load_area"], sort=False)["load_mw"]
    for window in windows:
        rolling = by_area.rolling(window=window, min_periods=window)
        out[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=[0, 1], drop=True)
        out[f"rolling_std_{window}"] = rolling.std().reset_index(level=[0, 1], drop=True)
    return out


def add_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    out = df.copy()
    target_rows = out[["zone", "load_area", "timestamp_utc", "timestamp_ept", "load_mw"]].rename(
        columns={
            "timestamp_utc": "target_timestamp_utc",
            "timestamp_ept": "target_timestamp_ept",
            "load_mw": "target_load_mw",
        }
    )
    out["target_timestamp_utc"] = out["timestamp_utc"] + pd.Timedelta(hours=horizon)
    out = out.merge(target_rows, on=["zone", "load_area", "target_timestamp_utc"], how="left")
    return out


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wx = weather_df[["timestamp_utc", "temperature_c", "humidity_pct"]].copy()

    out = out.merge(wx, on="timestamp_utc", how="left")

    lagged_weather = wx.rename(
        columns={
            "timestamp_utc": "weather_lag_timestamp_utc",
            "temperature_c": "temperature_lag_24",
            "humidity_pct": "humidity_lag_24",
        }
    )
    out["weather_lag_timestamp_utc"] = out["timestamp_utc"] - pd.Timedelta(hours=WEATHER_LAG_HOURS)
    out = out.merge(lagged_weather, on="weather_lag_timestamp_utc", how="left")
    out = out.drop(columns=["weather_lag_timestamp_utc"])
    return out


def make_feature_dataset(df: pd.DataFrame, weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_cyclical_features(out)
    out = add_lag_features(out, LOAD_LAGS)
    out = add_rolling_features(out, ROLLING_WINDOWS)
    if weather_df is not None:
        out = add_weather_features(out, weather_df)
    out = add_target(out, horizon=FORECAST_HORIZON)

    required_cols = (WEATHER_FEATURE_COLUMNS if weather_df is not None else FEATURE_COLUMNS) + [
        "timestamp_utc",
        "timestamp_ept",
        "target_timestamp_utc",
        "target_timestamp_ept",
        "zone",
        "target_load_mw",
    ]
    before = len(out)
    out = out.dropna(subset=required_cols)
    out = out.sort_values(["timestamp_utc", "zone", "load_area"]).reset_index(drop=True)

    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} rows without complete lag, rolling, or exact 24-hour target values.")
    return out
