from __future__ import annotations

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_weather_data(path: str) -> pd.DataFrame:
    weather = pd.read_csv(path, skiprows=3)
    weather = weather.rename(
        columns={
            "time": "timestamp_utc",
            "temperature_2m (°C)": "temperature_c",
            "relative_humidity_2m (%)": "humidity_pct",
        }
    )
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], errors="coerce")
    weather["temperature_c"] = pd.to_numeric(weather["temperature_c"], errors="coerce")
    weather["humidity_pct"] = pd.to_numeric(weather["humidity_pct"], errors="coerce")
    weather = weather.dropna(subset=["timestamp_utc", "temperature_c", "humidity_pct"])
    weather = weather.drop_duplicates(subset=["timestamp_utc"])
    weather = weather.sort_values("timestamp_utc").reset_index(drop=True)
    return weather[["timestamp_utc", "temperature_c", "humidity_pct"]]


def filter_weather_to_load_range(
    weather_df: pd.DataFrame,
    load_df: pd.DataFrame,
    lag_hours: int = 24,
) -> pd.DataFrame:
    min_load_timestamp = load_df["timestamp_utc"].min()
    max_load_timestamp = load_df["timestamp_utc"].max()
    min_weather_timestamp = min_load_timestamp - pd.Timedelta(hours=lag_hours)

    filtered = weather_df[
        (weather_df["timestamp_utc"] >= min_weather_timestamp)
        & (weather_df["timestamp_utc"] <= max_load_timestamp)
    ].copy()

    weather_timestamps = set(filtered["timestamp_utc"])
    load_timestamps = set(load_df["timestamp_utc"].dropna().unique())
    missing_issue_timestamps = sorted(load_timestamps - weather_timestamps)
    if missing_issue_timestamps:
        first_missing = missing_issue_timestamps[0]
        raise ValueError(f"Weather data is missing issue-time coverage starting at {first_missing}.")

    lag_timestamps = {timestamp - pd.Timedelta(hours=lag_hours) for timestamp in load_timestamps}
    missing_lag_timestamps = sorted(lag_timestamps - weather_timestamps)
    if missing_lag_timestamps:
        first_missing = missing_lag_timestamps[0]
        raise ValueError(f"Weather data is missing {lag_hours}-hour lag coverage starting at {first_missing}.")

    return filtered.reset_index(drop=True)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )


def clean_pjm_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()

    datetime_format = "%m/%d/%Y %I:%M:%S %p"
    cleaned["timestamp_utc"] = pd.to_datetime(
        cleaned["datetime_beginning_utc"], format=datetime_format, errors="coerce"
    )
    cleaned["timestamp_ept"] = pd.to_datetime(
        cleaned["datetime_beginning_ept"], format=datetime_format, errors="coerce"
    )
    cleaned = cleaned.rename(columns={"mkt_region": "market_region", "mw": "load_mw"})
    cleaned["load_mw"] = pd.to_numeric(cleaned["load_mw"], errors="coerce")

    if "is_verified" in cleaned.columns:
        cleaned["is_verified"] = _to_bool(cleaned["is_verified"])

    string_cols = ["nerc_region", "market_region", "zone", "load_area"]
    for col in string_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.dropna(subset=["timestamp_utc", "timestamp_ept", "load_area", "zone", "load_mw"])
    cleaned = cleaned.drop_duplicates(subset=["timestamp_utc", "zone", "load_area"])
    cleaned = cleaned.sort_values(["timestamp_utc", "zone", "load_area"]).reset_index(drop=True)

    columns = [
        "timestamp_utc",
        "timestamp_ept",
        "nerc_region",
        "market_region",
        "zone",
        "load_area",
        "load_mw",
        "is_verified",
    ]
    return cleaned[columns]


def validate_clean_data(df: pd.DataFrame) -> dict:
    duplicate_count = int(df.duplicated(subset=["timestamp_utc", "zone", "load_area"]).sum())
    summary = {
        "n_rows": int(len(df)),
        "n_unique_timestamps": int(df["timestamp_utc"].nunique()),
        "zone_values": sorted(df["zone"].dropna().unique().tolist()),
        "load_area_values": sorted(df["load_area"].dropna().unique().tolist()),
        "rows_per_zone": df.groupby("zone").size().to_dict(),
        "rows_per_load_area": df.groupby("load_area").size().to_dict(),
        "rows_per_zone_load_area": df.groupby(["zone", "load_area"]).size().to_dict(),
        "min_timestamp_utc": df["timestamp_utc"].min(),
        "max_timestamp_utc": df["timestamp_utc"].max(),
        "all_verified": bool(df["is_verified"].fillna(False).all()),
        "duplicate_count_after_cleaning": duplicate_count,
        "missing_values_by_column": df.isna().sum().to_dict(),
    }

    print("\nClean data validation")
    print(f"Rows: {summary['n_rows']}")
    print(f"Unique timestamps: {summary['n_unique_timestamps']}")
    print(f"Zones: {summary['zone_values']}")
    print(f"Load areas: {summary['load_area_values']}")
    print(f"Rows per zone: {summary['rows_per_zone']}")
    print(f"Rows per load area: {summary['rows_per_load_area']}")
    print(f"UTC range: {summary['min_timestamp_utc']} to {summary['max_timestamp_utc']}")
    print(f"All verified: {summary['all_verified']}")
    print(f"Duplicates after cleaning: {summary['duplicate_count_after_cleaning']}")

    checks = [
        (summary["all_verified"], "Expected all rows to be verified."),
        (summary["duplicate_count_after_cleaning"] == 0, "Expected no duplicate zone/load-area timestamps."),
    ]
    for ok, message in checks:
        if not ok:
            print(f"WARNING: {message}")

    return summary
