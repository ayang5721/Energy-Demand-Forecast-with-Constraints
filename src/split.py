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
    """Split each zone/load-area series chronologically."""
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(["zone", "load_area"], sort=False):
        timestamps = sorted(group["timestamp_utc"].dropna().unique())
        n_timestamps = len(timestamps)
        train_end = int(n_timestamps * train_frac)
        val_end = int(n_timestamps * (train_frac + val_frac))

        train_ts = set(timestamps[:train_end])
        val_ts = set(timestamps[train_end:val_end])
        test_ts = set(timestamps[val_end:])

        train_parts.append(group[group["timestamp_utc"].isin(train_ts)])
        val_parts.append(group[group["timestamp_utc"].isin(val_ts)])
        test_parts.append(group[group["timestamp_utc"].isin(test_ts)])

    train_df = pd.concat(train_parts, ignore_index=True).sort_values(["timestamp_utc", "zone", "load_area"])
    val_df = pd.concat(val_parts, ignore_index=True).sort_values(["timestamp_utc", "zone", "load_area"])
    test_df = pd.concat(test_parts, ignore_index=True).sort_values(["timestamp_utc", "zone", "load_area"])
    return train_df, val_df, test_df


def get_feature_target_metadata(df: pd.DataFrame, feature_columns=None):
    """Return X, y, and metadata dataframes for modeling."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    X = df[feature_columns].copy()
    y = df["target_load_mw"].copy()
    metadata = df[METADATA_COLUMNS].copy()
    return X, y, metadata


def validate_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate chronological ordering within each zone/load-area split."""
    print("\nSplit validation")
    print(f"Train rows: {len(train_df)}, timestamps: {train_df['timestamp_utc'].nunique()}")
    print(f"Validation rows: {len(val_df)}, timestamps: {val_df['timestamp_utc'].nunique()}")
    print(f"Test rows: {len(test_df)}, timestamps: {test_df['timestamp_utc'].nunique()}")
    print(f"Train series: {train_df[['zone', 'load_area']].drop_duplicates().shape[0]}")
    print(f"Validation series: {val_df[['zone', 'load_area']].drop_duplicates().shape[0]}")
    print(f"Test series: {test_df[['zone', 'load_area']].drop_duplicates().shape[0]}")

    keys = ["zone", "load_area"]
    series_keys = pd.concat([train_df[keys], val_df[keys], test_df[keys]]).drop_duplicates()
    for row in series_keys.itertuples(index=False):
        train_group = train_df[(train_df["zone"] == row.zone) & (train_df["load_area"] == row.load_area)]
        val_group = val_df[(val_df["zone"] == row.zone) & (val_df["load_area"] == row.load_area)]
        test_group = test_df[(test_df["zone"] == row.zone) & (test_df["load_area"] == row.load_area)]

        if not train_group.empty and not val_group.empty:
            assert train_group["timestamp_utc"].max() < val_group["timestamp_utc"].min()
        if not val_group.empty and not test_group.empty:
            assert val_group["timestamp_utc"].max() < test_group["timestamp_utc"].min()

        train_ts = set(train_group["timestamp_utc"])
        val_ts = set(val_group["timestamp_utc"])
        test_ts = set(test_group["timestamp_utc"])
        assert train_ts.isdisjoint(val_ts)
        assert train_ts.isdisjoint(test_ts)
        assert val_ts.isdisjoint(test_ts)
    print("Chronological split checks passed.")
