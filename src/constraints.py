"""Hard-constraint dispatch layer for zone-level forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


DISPATCH_TOLERANCE = 1e-6


def create_generator_fleet(max_zone_load_mw: float) -> pd.DataFrame:
    """Create a synthetic generator fleet sized to observed peak zone load."""
    fleet = pd.DataFrame(
        [
            {"generator": "cheap_base", "max_mw": 0.35 * max_zone_load_mw, "cost_per_mwh": 25.0},
            {"generator": "mid_cost", "max_mw": 0.35 * max_zone_load_mw, "cost_per_mwh": 50.0},
            {"generator": "high_cost", "max_mw": 0.35 * max_zone_load_mw, "cost_per_mwh": 85.0},
            {"generator": "peaker", "max_mw": 0.25 * max_zone_load_mw, "cost_per_mwh": 150.0},
        ]
    )
    fleet = fleet.sort_values("cost_per_mwh").reset_index(drop=True)
    fleet["dispatch_order"] = np.arange(1, len(fleet) + 1)
    return fleet[["generator", "max_mw", "cost_per_mwh", "dispatch_order"]]


def greedy_dispatch(demand_mw: float, generator_fleet: pd.DataFrame) -> dict:
    """Dispatch generators from cheapest to most expensive to meet demand."""
    demand = max(0.0, float(demand_mw))
    remaining_demand = demand
    total_generation_mw = 0.0
    dispatch_cost = 0.0
    result = {"demand_mw": demand}

    ordered_fleet = generator_fleet.sort_values(["cost_per_mwh", "dispatch_order"])
    for row in ordered_fleet.itertuples(index=False):
        generation_mw = min(float(row.max_mw), remaining_demand)
        generation_mw = max(0.0, generation_mw)
        remaining_demand -= generation_mw
        total_generation_mw += generation_mw
        dispatch_cost += generation_mw * float(row.cost_per_mwh)
        result[f"{row.generator}_generation_mw"] = generation_mw

    unmet_demand_mw = max(0.0, remaining_demand)
    feasible = unmet_demand_mw <= DISPATCH_TOLERANCE
    result.update(
        {
            "total_generation_mw": total_generation_mw,
            "dispatch_cost": dispatch_cost,
            "feasible": bool(feasible),
            "unmet_demand_mw": 0.0 if feasible else unmet_demand_mw,
        }
    )
    return result


def run_constrained_dispatch(zone_predictions_df: pd.DataFrame, generator_fleet: pd.DataFrame) -> pd.DataFrame:
    """Run forecast and oracle dispatch for every model-hour zone forecast."""
    rows = []
    for row in zone_predictions_df.itertuples(index=False):
        forecast_dispatch = greedy_dispatch(row.predicted_zone_load_mw, generator_fleet)
        oracle_dispatch = greedy_dispatch(row.true_zone_load_mw, generator_fleet)

        scheduled_generation_mw = forecast_dispatch["total_generation_mw"]
        oracle_generation_mw = oracle_dispatch["total_generation_mw"]
        out = {
            "target_timestamp_utc": row.target_timestamp_utc,
            "target_timestamp_ept": row.target_timestamp_ept,
            "model": row.model,
            "true_zone_load_mw": row.true_zone_load_mw,
            "predicted_zone_load_mw": row.predicted_zone_load_mw,
            "scheduled_generation_mw": scheduled_generation_mw,
            "dispatch_cost": forecast_dispatch["dispatch_cost"],
            "oracle_generation_mw": oracle_generation_mw,
            "oracle_dispatch_cost": oracle_dispatch["dispatch_cost"],
            "cost_gap": forecast_dispatch["dispatch_cost"] - oracle_dispatch["dispatch_cost"],
            "feasible_for_forecast": forecast_dispatch["feasible"],
            "unmet_forecast_demand_mw": forecast_dispatch["unmet_demand_mw"],
            "under_generation_mw": max(0.0, row.true_zone_load_mw - scheduled_generation_mw),
            "over_generation_mw": max(0.0, scheduled_generation_mw - row.true_zone_load_mw),
        }

        for generator in generator_fleet["generator"]:
            column = f"{generator}_generation_mw"
            out[column] = forecast_dispatch.get(column, 0.0)
            out[f"oracle_{column}"] = oracle_dispatch.get(column, 0.0)

        rows.append(out)

    return pd.DataFrame(rows)


def make_post_constraint_metrics(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute post-constraint operational metrics by model."""
    def metrics(group: pd.DataFrame) -> pd.Series:
        zone_error = group["scheduled_generation_mw"] - group["true_zone_load_mw"]
        return pd.Series(
            {
                "n_hours": len(group),
                "feasible_hours": int(group["feasible_for_forecast"].sum()),
                "infeasible_hours": int((~group["feasible_for_forecast"]).sum()),
                "under_generation_hours": int((group["under_generation_mw"] > 0).sum()),
                "under_generation_rate": float((group["under_generation_mw"] > 0).mean()),
                "total_under_generation_mw": group["under_generation_mw"].sum(),
                "max_under_generation_mw": group["under_generation_mw"].max(),
                "total_over_generation_mw": group["over_generation_mw"].sum(),
                "max_over_generation_mw": group["over_generation_mw"].max(),
                "total_dispatch_cost": group["dispatch_cost"].sum(),
                "total_oracle_dispatch_cost": group["oracle_dispatch_cost"].sum(),
                "total_cost_gap": group["cost_gap"].sum(),
                "mean_cost_gap": group["cost_gap"].mean(),
                "mean_abs_zone_error_after_dispatch_mw": zone_error.abs().mean(),
                "rmse_zone_error_after_dispatch_mw": float(np.sqrt(np.mean(zone_error**2))),
                "bias_zone_error_after_dispatch_mw": zone_error.mean(),
            }
        )

    return dispatch_df.groupby("model", sort=False).apply(metrics).reset_index()
