"""Data loading, cleaning, and validation for PJM load data."""

from __future__ import annotations

import pandas as pd


EXPECTED_LOAD_AREAS = ["AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT"]


def load_raw_data(path: str) -> pd.DataFrame:
    """Read the raw PJM CSV from disk."""
    return pd.read_csv(path)


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean-like values to pandas booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )


def clean_pjm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw PJM load data and return standard milestone columns."""
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

    cleaned = cleaned[cleaned["zone"] == "AEP"]
    cleaned = cleaned.dropna(subset=["timestamp_utc", "timestamp_ept", "load_area", "zone", "load_mw"])
    cleaned = cleaned.drop_duplicates(subset=["timestamp_utc", "zone", "load_area"])
    cleaned = cleaned.sort_values(["timestamp_utc", "load_area"]).reset_index(drop=True)

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
    """Return and print a validation summary for cleaned PJM data."""
    duplicate_count = int(df.duplicated(subset=["timestamp_utc", "zone", "load_area"]).sum())
    summary = {
        "n_rows": int(len(df)),
        "n_unique_timestamps": int(df["timestamp_utc"].nunique()),
        "zone_values": sorted(df["zone"].dropna().unique().tolist()),
        "load_area_values": sorted(df["load_area"].dropna().unique().tolist()),
        "rows_per_load_area": df.groupby("load_area").size().to_dict(),
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
    print(f"Rows per load area: {summary['rows_per_load_area']}")
    print(f"UTC range: {summary['min_timestamp_utc']} to {summary['max_timestamp_utc']}")
    print(f"All verified: {summary['all_verified']}")
    print(f"Duplicates after cleaning: {summary['duplicate_count_after_cleaning']}")

    expected_rows_per_area = {
        area: summary["n_unique_timestamps"]
        for area in summary["load_area_values"]
    }
    checks = [
        (summary["zone_values"] == ["AEP"], "Expected only AEP zone."),
        (summary["load_area_values"] == EXPECTED_LOAD_AREAS, "Expected four known AEP load areas."),
        (
            summary["rows_per_load_area"] == expected_rows_per_area,
            "Expected each load area to have one row per unique timestamp.",
        ),
        (summary["all_verified"], "Expected all rows to be verified."),
    ]
    for ok, message in checks:
        if not ok:
            print(f"WARNING: {message}")

    return summary
