from __future__ import annotations

import numpy as np
import pandas as pd


DISPATCH_TOLERANCE = 1e-6
UNDER_GENERATION_PENALTY_PER_MWH = 500.0
OVER_GENERATION_PENALTY_PER_MWH = 50.0


def create_generator_fleet(max_zone_load_mw: float) -> pd.DataFrame:
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


def _dispatch_many(demand_mw, generator_fleet: pd.DataFrame) -> dict:
    demand = np.maximum(0.0, np.asarray(demand_mw, dtype=float))
    remaining_demand = demand.copy()
    total_generation_mw = np.zeros_like(demand)
    dispatch_cost = np.zeros_like(demand)
    result = {}

    ordered_fleet = generator_fleet.sort_values(["cost_per_mwh", "dispatch_order"])
    for row in ordered_fleet.itertuples(index=False):
        generation_mw = np.minimum(float(row.max_mw), remaining_demand)
        generation_mw = np.maximum(0.0, generation_mw)
        remaining_demand -= generation_mw
        total_generation_mw += generation_mw
        dispatch_cost += generation_mw * float(row.cost_per_mwh)
        result[f"{row.generator}_generation_mw"] = generation_mw

    unmet_demand_mw = np.maximum(0.0, remaining_demand)
    feasible = unmet_demand_mw <= DISPATCH_TOLERANCE
    result.update(
        {
            "total_generation_mw": total_generation_mw,
            "dispatch_cost": dispatch_cost,
            "feasible": feasible,
            "unmet_demand_mw": np.where(feasible, 0.0, unmet_demand_mw),
        }
    )
    return result


def run_constrained_dispatch(zone_predictions_df: pd.DataFrame, generator_fleet: pd.DataFrame) -> pd.DataFrame:
    out = zone_predictions_df[
        [
            "target_timestamp_utc",
            "target_timestamp_ept",
            "zone",
            "model",
            "true_zone_load_mw",
            "predicted_zone_load_mw",
        ]
    ].copy()

    forecast_dispatch = _dispatch_many(out["predicted_zone_load_mw"], generator_fleet)
    oracle_dispatch = _dispatch_many(out["true_zone_load_mw"], generator_fleet)

    out["scheduled_generation_mw"] = forecast_dispatch["total_generation_mw"]
    out["base_generator_cost"] = forecast_dispatch["dispatch_cost"]
    out["dispatch_cost"] = out["base_generator_cost"]
    out["oracle_generation_mw"] = oracle_dispatch["total_generation_mw"]
    out["oracle_base_generator_cost"] = oracle_dispatch["dispatch_cost"]
    out["oracle_dispatch_cost"] = out["oracle_base_generator_cost"]
    out["cost_gap"] = out["base_generator_cost"] - out["oracle_base_generator_cost"]
    out["feasible_for_forecast"] = forecast_dispatch["feasible"]
    out["unmet_forecast_demand_mw"] = forecast_dispatch["unmet_demand_mw"]
    out["under_generation_mw"] = np.maximum(0.0, out["true_zone_load_mw"] - out["scheduled_generation_mw"])
    out["over_generation_mw"] = np.maximum(0.0, out["scheduled_generation_mw"] - out["true_zone_load_mw"])
    out["under_generation_mwh"] = out["under_generation_mw"]
    out["over_generation_mwh"] = out["over_generation_mw"]
    out["under_generation_penalty_cost"] = (
        out["under_generation_mwh"] * UNDER_GENERATION_PENALTY_PER_MWH
    )
    out["over_generation_penalty_cost"] = (
        out["over_generation_mwh"] * OVER_GENERATION_PENALTY_PER_MWH
    )
    out["penalty_cost"] = out["under_generation_penalty_cost"] + out["over_generation_penalty_cost"]
    out["total_operational_cost"] = out["base_generator_cost"] + out["penalty_cost"]
    out["oracle_penalty_cost"] = 0.0
    out["oracle_total_operational_cost"] = out["oracle_base_generator_cost"] + out["oracle_penalty_cost"]
    out["constraint_regret"] = out["total_operational_cost"] - out["oracle_total_operational_cost"]

    for generator in generator_fleet["generator"]:
        column = f"{generator}_generation_mw"
        out[column] = forecast_dispatch.get(column, 0.0)
        out[f"oracle_{column}"] = oracle_dispatch.get(column, 0.0)

    return out


def make_post_constraint_metrics(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    def metrics(group: pd.DataFrame) -> pd.Series:
        zone_error = group["scheduled_generation_mw"] - group["true_zone_load_mw"]
        total_base_generator_cost = group["base_generator_cost"].sum()
        total_penalty_cost = group["penalty_cost"].sum()
        total_operational_cost = group["total_operational_cost"].sum()
        oracle_total_operational_cost = group["oracle_total_operational_cost"].sum()
        total_constraint_regret = group["constraint_regret"].sum()
        return pd.Series(
            {
                "n_hours": len(group),
                "feasible_hours": int(group["feasible_for_forecast"].sum()),
                "infeasible_hours": int((~group["feasible_for_forecast"]).sum()),
                "under_generation_hours": int((group["under_generation_mwh"] > 0).sum()),
                "over_generation_hours": int((group["over_generation_mwh"] > 0).sum()),
                "under_generation_rate": float((group["under_generation_mwh"] > 0).mean()),
                "over_generation_rate": float((group["over_generation_mwh"] > 0).mean()),
                "total_under_generation_mwh": group["under_generation_mwh"].sum(),
                "total_over_generation_mwh": group["over_generation_mwh"].sum(),
                "total_under_generation_mw": group["under_generation_mw"].sum(),
                "total_over_generation_mw": group["over_generation_mw"].sum(),
                "max_under_generation_mw": group["under_generation_mw"].max(),
                "max_over_generation_mw": group["over_generation_mw"].max(),
                "total_base_generator_cost": total_base_generator_cost,
                "total_oracle_base_generator_cost": group["oracle_base_generator_cost"].sum(),
                "total_under_generation_penalty_cost": group["under_generation_penalty_cost"].sum(),
                "total_over_generation_penalty_cost": group["over_generation_penalty_cost"].sum(),
                "total_penalty_cost": total_penalty_cost,
                "total_operational_cost": total_operational_cost,
                "oracle_total_operational_cost": oracle_total_operational_cost,
                "total_constraint_regret": total_constraint_regret,
                "mean_constraint_regret": group["constraint_regret"].mean(),
                "total_dispatch_cost": total_base_generator_cost,
                "total_oracle_dispatch_cost": group["oracle_dispatch_cost"].sum(),
                "total_cost_gap": group["cost_gap"].sum(),
                "mean_cost_gap": group["cost_gap"].mean(),
                "mean_abs_zone_error_after_dispatch_mw": zone_error.abs().mean(),
                "rmse_zone_error_after_dispatch_mw": float(np.sqrt(np.mean(zone_error**2))),
                "bias_zone_error_after_dispatch_mw": zone_error.mean(),
            }
        )

    return dispatch_df.groupby("model", sort=False).apply(metrics).reset_index()


def validate_constraint_costs(dispatch_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    print("\nConstraint cost validation")

    assert np.allclose(dispatch_df["oracle_penalty_cost"], 0.0)
    assert np.allclose(
        dispatch_df["total_operational_cost"],
        dispatch_df["base_generator_cost"] + dispatch_df["penalty_cost"],
    )
    assert np.allclose(
        dispatch_df["constraint_regret"],
        dispatch_df["total_operational_cost"] - dispatch_df["oracle_total_operational_cost"],
    )
    assert not (
        (dispatch_df["under_generation_mwh"] > 0)
        & (dispatch_df["over_generation_mwh"] > 0)
    ).any()
    assert (dispatch_df["under_generation_penalty_cost"] >= 0).all()
    assert (dispatch_df["over_generation_penalty_cost"] >= 0).all()
    assert (dispatch_df["penalty_cost"] >= 0).all()
    assert np.allclose(
        metrics_df["total_operational_cost"],
        metrics_df["total_base_generator_cost"] + metrics_df["total_penalty_cost"],
    )
    assert np.allclose(
        metrics_df["total_constraint_regret"],
        metrics_df["total_operational_cost"] - metrics_df["oracle_total_operational_cost"],
    )

    print("Constraint cost checks passed.")
