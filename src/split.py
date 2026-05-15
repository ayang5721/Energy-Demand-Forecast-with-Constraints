"""Chronological train, validation, and test splitting utilities."""

from __future__ import annotations

import pandas as pd

from features import FEATURE_COLUMNS


METADATA_COLUMNS = [
    "timestamp_utc",
    "timestamp_ept",
    "target_timestamp_utc",
    "target_timestamp_ept",
    "zone",
    "load_area",
    "load_mw",
    "target_load_mw",
]


def time_based_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split a feature dataframe by sorted unique input timestamps."""
    timestamps = sorted(df["timestamp_utc"].dropna().unique())
    n_timestamps = len(timestamps)
    train_end = int(n_timestamps * train_frac)
    val_end = int(n_timestamps * (train_frac + val_frac))

    train_ts = set(timestamps[:train_end])
    val_ts = set(timestamps[train_end:val_end])
    test_ts = set(timestamps[val_end:])

    train_df = df[df["timestamp_utc"].isin(train_ts)].copy()
    val_df = df[df["timestamp_utc"].isin(val_ts)].copy()
    test_df = df[df["timestamp_utc"].isin(test_ts)].copy()
    return train_df, val_df, test_df


def get_feature_target_metadata(df: pd.DataFrame):
    """Return X, y, and metadata dataframes for modeling."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["target_load_mw"].copy()
    metadata = df[METADATA_COLUMNS].copy()
    return X, y, metadata


def validate_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate chronological ordering and timestamp exclusivity of splits."""
    print("\nSplit validation")
    print(f"Train rows: {len(train_df)}, timestamps: {train_df['timestamp_utc'].nunique()}")
    print(f"Validation rows: {len(val_df)}, timestamps: {val_df['timestamp_utc'].nunique()}")
    print(f"Test rows: {len(test_df)}, timestamps: {test_df['timestamp_utc'].nunique()}")

    if not train_df.empty and not val_df.empty:
        assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
    if not val_df.empty and not test_df.empty:
        assert val_df["timestamp_utc"].max() < test_df["timestamp_utc"].min()

    train_ts = set(train_df["timestamp_utc"])
    val_ts = set(val_df["timestamp_utc"])
    test_ts = set(test_df["timestamp_utc"])
    assert train_ts.isdisjoint(val_ts)
    assert train_ts.isdisjoint(test_ts)
    assert val_ts.isdisjoint(test_ts)
    print("Chronological split checks passed.")
