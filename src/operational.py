"""Zone aggregation utilities for load-area forecasts."""

from __future__ import annotations

import pandas as pd


def aggregate_to_zone(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load-area forecasts into zone-level forecasts by target timestamp."""
    return (
        predictions_df.groupby(["target_timestamp_utc", "target_timestamp_ept", "zone", "model"], sort=False)
        .agg(
            true_zone_load_mw=("true_load_mw", "sum"),
            predicted_zone_load_mw=("predicted_load_mw", "sum"),
            n_load_areas=("load_area", "nunique"),
        )
        .reset_index()
    )


def aggregate_predictions_to_zone(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Alias for the constraint-layer plan's zone aggregation function name."""
    return aggregate_to_zone(predictions_df)
