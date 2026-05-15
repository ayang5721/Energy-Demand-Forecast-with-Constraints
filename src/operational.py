"""Zone aggregation and simple under/over generation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_to_zone(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load-area forecasts into zone-level forecasts by target timestamp."""
    return (
        predictions_df.groupby(["target_timestamp_utc", "target_timestamp_ept", "model"], sort=False)
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


def add_under_over_generation(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Add signed, absolute, under-generation, and over-generation columns."""
    out = zone_df.copy()
    out["zone_error_mw"] = out["predicted_zone_load_mw"] - out["true_zone_load_mw"]
    out["zone_abs_error_mw"] = out["zone_error_mw"].abs()
    out["under_generation_mw"] = np.maximum(0, out["true_zone_load_mw"] - out["predicted_zone_load_mw"])
    out["over_generation_mw"] = np.maximum(0, out["predicted_zone_load_mw"] - out["true_zone_load_mw"])
    return out


def make_operational_metrics(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple operational metrics by model."""
    def metrics(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_hours": len(group),
                "under_generation_hours": int((group["under_generation_mw"] > 0).sum()),
                "under_generation_rate": float((group["under_generation_mw"] > 0).mean()),
                "total_under_generation_mw": group["under_generation_mw"].sum(),
                "max_under_generation_mw": group["under_generation_mw"].max(),
                "total_over_generation_mw": group["over_generation_mw"].sum(),
                "max_over_generation_mw": group["over_generation_mw"].max(),
                "mean_zone_abs_error_mw": group["zone_abs_error_mw"].mean(),
                "rmse_zone_error_mw": float(np.sqrt(np.mean(group["zone_error_mw"] ** 2))),
                "bias_zone_error_mw": group["zone_error_mw"].mean(),
            }
        )

    return zone_df.groupby("model", sort=False).apply(metrics).reset_index()
