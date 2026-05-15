"""Matplotlib plots for milestone outputs."""

from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp/matplotlib-energy-demand-forecast").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd


MODEL_STYLES = {
    "Persistence": {"linestyle": "--", "linewidth": 1.8, "alpha": 0.9},
    "OLS": {"linestyle": ":", "linewidth": 2.4, "alpha": 0.95},
    "Ridge": {"linestyle": "-.", "linewidth": 1.8, "alpha": 0.9},
}


def plot_true_vs_predicted_load_area(predictions_df, output_path, load_area=None, max_points=96) -> None:
    """Plot pre-constraint actual and model predictions for one load area."""
    df = predictions_df.copy()
    if load_area is None:
        load_area = sorted(df["load_area"].unique())[0]
    df = df[df["load_area"] == load_area].sort_values("target_timestamp_ept")
    sample_times = df["target_timestamp_ept"].drop_duplicates().head(max_points)
    sample = df[df["target_timestamp_ept"].isin(sample_times)]

    plt.figure(figsize=(12, 6))
    actual = sample.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    plt.plot(actual["target_timestamp_ept"], actual["true_load_mw"], label="Actual", color="black", linewidth=2.5)
    for model in ["Persistence", "OLS", "Ridge"]:
        model_df = sample[sample["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            plt.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    plt.title(f"Pre-Constraint Layer: True vs Predicted Load - {load_area}")
    plt.xlabel("Target timestamp EPT")
    plt.ylabel("MW")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_true_vs_predicted_average(predictions_df, output_path, max_points=96) -> None:
    """Plot pre-constraint average actual and predicted load across all load areas."""
    df = predictions_df.copy().sort_values("target_timestamp_ept")
    sample_times = df["target_timestamp_ept"].drop_duplicates().head(max_points)
    sample = df[df["target_timestamp_ept"].isin(sample_times)]

    averaged = (
        sample.groupby(["target_timestamp_ept", "model"], sort=False)
        .agg(true_load_mw=("true_load_mw", "mean"), predicted_load_mw=("predicted_load_mw", "mean"))
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    actual = averaged.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    plt.plot(actual["target_timestamp_ept"], actual["true_load_mw"], label="Actual", color="black", linewidth=2.5)
    for model in ["Persistence", "OLS", "Ridge"]:
        model_df = averaged[averaged["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            plt.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    plt.title("Pre-Constraint Layer: True vs Predicted Load - Average Across Load Areas")
    plt.xlabel("Target timestamp EPT")
    plt.ylabel("Average MW")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_by_hour(error_by_hour_df, output_path) -> None:
    """Plot pre-constraint mean absolute forecast error by target hour for each model."""
    plt.figure(figsize=(10, 6))
    for model, group in error_by_hour_df.groupby("model", sort=False):
        ordered = group.sort_values("hour")
        plt.plot(ordered["hour"], ordered["mean_abs_error"], marker="o", label=model)
    plt.xlabel("Target hour")
    plt.ylabel("Mean absolute error (MW)")
    plt.title("Pre-Constraint Layer: Mean Absolute Error by Hour")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_forecast_metrics_bar(metrics_df: pd.DataFrame, output_path, metric="rmse") -> None:
    """Plot a selected pre-constraint forecast metric by model."""
    ordered = metrics_df.sort_values(metric)
    plt.figure(figsize=(8, 5))
    plt.bar(ordered["model"], ordered[metric])
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(f"Pre-Constraint Layer: {metric.upper()} by Model")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_post_constraint_dispatch_cost(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total dispatch cost by model."""
    ordered = post_metrics_df.sort_values("total_dispatch_cost")
    plt.figure(figsize=(8, 5))
    plt.bar(ordered["model"], ordered["total_dispatch_cost"])
    plt.xlabel("Model")
    plt.ylabel("Total dispatch cost ($)")
    plt.title("Post-Constraint Layer: Total Dispatch Cost by Model")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_post_constraint_under_generation(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total under-generation by model."""
    ordered = post_metrics_df.sort_values("total_under_generation_mw")
    plt.figure(figsize=(8, 5))
    plt.bar(ordered["model"], ordered["total_under_generation_mw"])
    plt.xlabel("Model")
    plt.ylabel("Total under-generation (MW)")
    plt.title("Post-Constraint Layer: Total Under-Generation by Model")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_post_constraint_scheduled_vs_true(dispatch_df: pd.DataFrame, output_path, max_points=96) -> None:
    """Plot scheduled generation against true zone load for an initial sample window."""
    df = dispatch_df.copy().sort_values("target_timestamp_ept")
    sample_times = df["target_timestamp_ept"].drop_duplicates().head(max_points)
    sample = df[df["target_timestamp_ept"].isin(sample_times)]

    plt.figure(figsize=(12, 6))
    actual = sample.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    plt.plot(actual["target_timestamp_ept"], actual["true_zone_load_mw"], label="Actual", color="black", linewidth=2.5)
    for model in ["Persistence", "OLS", "Ridge"]:
        model_df = sample[sample["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            plt.plot(
                model_df["target_timestamp_ept"],
                model_df["scheduled_generation_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    plt.title("Post-Constraint Layer: Scheduled vs True Zone Load")
    plt.xlabel("Target timestamp EPT")
    plt.ylabel("MW")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
