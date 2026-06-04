"""Matplotlib plots for forecast and dispatch outputs."""

from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp/matplotlib-energy-demand-forecast").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd


MODEL_STYLES = {
    "Persistence": {"color": "#4C78A8", "linestyle": "--", "linewidth": 1.8, "alpha": 0.9},
    "OLS": {"color": "#F58518", "linestyle": ":", "linewidth": 2.4, "alpha": 0.95},
    "Neural Network": {"color": "#E45756", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "Neural Network + Weather": {"color": "#72B7B2", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "Ridge": {"color": "#54A24B", "linestyle": "-.", "linewidth": 1.8, "alpha": 0.9},
    "Lasso": {"color": "#B279A2", "linestyle": "-", "linewidth": 1.7, "alpha": 0.85},
}
MODEL_ORDER = ["Persistence", "OLS", "Neural Network", "Neural Network + Weather", "Ridge", "Lasso"]


def _ordered_models(df: pd.DataFrame) -> list[str]:
    """Return known models in preferred order, followed by any new models."""
    models = df["model"].dropna().unique().tolist()
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(model for model in models if model not in MODEL_ORDER)
    return ordered


def plot_true_vs_predicted_load_area(predictions_df, output_path, zone=None, load_area=None, max_points=96) -> None:
    """Plot pre-constraint actual and model predictions for one load area."""
    df = predictions_df.copy()
    if load_area is None:
        load_area = sorted(df["load_area"].unique())[0]
    if zone is None:
        zone = sorted(df[df["load_area"] == load_area]["zone"].unique())[0]
    df = df[(df["zone"] == zone) & (df["load_area"] == load_area)].sort_values("target_timestamp_ept")
    sample_times = df["target_timestamp_ept"].drop_duplicates().head(max_points)
    sample = df[df["target_timestamp_ept"].isin(sample_times)]

    fig, (ax_load, ax_error) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    actual = sample.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    ax_load.plot(
        actual["target_timestamp_ept"],
        actual["true_load_mw"],
        label="Actual",
        color="black",
        linewidth=2.5,
    )
    for model in _ordered_models(sample):
        model_df = sample[sample["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            ax_load.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
            ax_error.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"] - model_df["true_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    ax_error.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    ax_load.set_title(f"Pre-Constraint Layer: True vs Predicted Load - {zone} / {load_area}")
    ax_load.set_ylabel("MW")
    ax_error.set_xlabel("Target timestamp EPT")
    ax_error.set_ylabel("Error MW")
    ax_load.legend(ncol=3)
    ax_error.legend(ncol=3)
    plt.xticks(rotation=30, ha="right")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
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

    fig, (ax_load, ax_error) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    actual = averaged.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    ax_load.plot(
        actual["target_timestamp_ept"],
        actual["true_load_mw"],
        label="Actual",
        color="black",
        linewidth=2.5,
    )
    for model in _ordered_models(averaged):
        model_df = averaged[averaged["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            ax_load.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
            ax_error.plot(
                model_df["target_timestamp_ept"],
                model_df["predicted_load_mw"] - model_df["true_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    ax_error.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    ax_load.set_title("Pre-Constraint Layer: True vs Predicted Load - Average Across Load Areas")
    ax_load.set_ylabel("Average MW")
    ax_error.set_xlabel("Target timestamp EPT")
    ax_error.set_ylabel("Error MW")
    ax_load.legend(ncol=3)
    ax_error.legend(ncol=3)
    plt.xticks(rotation=30, ha="right")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_error_by_hour(error_by_hour_df, output_path) -> None:
    """Plot pre-constraint mean absolute forecast error by target hour for each model."""
    plt.figure(figsize=(10, 6))
    for model in _ordered_models(error_by_hour_df):
        group = error_by_hour_df[error_by_hour_df["model"] == model]
        if group.empty:
            continue
        ordered = group.sort_values("hour")
        plt.plot(
            ordered["hour"],
            ordered["mean_abs_error"],
            marker="o",
            label=model,
            **MODEL_STYLES.get(model, {}),
        )
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
    colors = [MODEL_STYLES.get(model, {}).get("color", "#777777") for model in ordered["model"]]
    plt.bar(ordered["model"], ordered[metric], color=colors)
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(f"Pre-Constraint Layer: {metric.upper()} by Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_post_constraint_metric_bar(
    post_metrics_df: pd.DataFrame,
    output_path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot one post-constraint metric by model."""
    ordered = post_metrics_df.sort_values(metric)
    plt.figure(figsize=(8, 5))
    colors = [MODEL_STYLES.get(model, {}).get("color", "#777777") for model in ordered["model"]]
    plt.bar(ordered["model"], ordered[metric], color=colors)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_post_constraint_base_generator_cost(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total base generator cost by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_base_generator_cost",
        "Total base generator cost ($)",
        "Post-Constraint Layer: Total Base Generator Cost by Model",
    )


def plot_post_constraint_under_generation(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total under-generation by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_under_generation_mwh",
        "Total under-generation (MWh)",
        "Post-Constraint Layer: Total Under-Generation by Model",
    )


def plot_post_constraint_over_generation(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total over-generation by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_over_generation_mwh",
        "Total over-generation (MWh)",
        "Post-Constraint Layer: Total Over-Generation by Model",
    )


def plot_post_constraint_penalty_cost(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total penalty cost by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_penalty_cost",
        "Total penalty cost ($)",
        "Post-Constraint Layer: Total Penalty Cost by Model",
    )


def plot_post_constraint_total_operational_cost(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total operational cost by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_operational_cost",
        "Total operational cost ($)",
        "Post-Constraint Layer: Total Operational Cost by Model",
    )


def plot_post_constraint_constraint_regret(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot post-constraint total constraint regret by model."""
    _plot_post_constraint_metric_bar(
        post_metrics_df,
        output_path,
        "total_constraint_regret",
        "Total constraint regret ($)",
        "Post-Constraint Layer: Total Constraint Regret by Model",
    )


def plot_post_constraint_penalty_cost_stacked(post_metrics_df: pd.DataFrame, output_path) -> None:
    """Plot under- and over-generation penalty costs stacked by model."""
    ordered = post_metrics_df.sort_values("total_penalty_cost")
    plt.figure(figsize=(8, 5))
    plt.bar(
        ordered["model"],
        ordered["total_under_generation_penalty_cost"],
        label="Under-generation penalty",
        color="#E45756",
    )
    plt.bar(
        ordered["model"],
        ordered["total_over_generation_penalty_cost"],
        bottom=ordered["total_under_generation_penalty_cost"],
        label="Over-generation penalty",
        color="#72B7B2",
    )
    plt.xlabel("Model")
    plt.ylabel("Total penalty cost ($)")
    plt.title("Post-Constraint Layer: Penalty Cost Components by Model")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()



def plot_post_constraint_scheduled_vs_true(dispatch_df: pd.DataFrame, output_path, max_points=96) -> None:
    """Plot total scheduled generation against total true load for an initial sample window."""
    df = dispatch_df.copy().sort_values("target_timestamp_ept")
    sample_times = df["target_timestamp_ept"].drop_duplicates().head(max_points)
    sample = df[df["target_timestamp_ept"].isin(sample_times)]
    sample = (
        sample.groupby(["target_timestamp_ept", "model"], sort=False)
        .agg(
            true_zone_load_mw=("true_zone_load_mw", "sum"),
            scheduled_generation_mw=("scheduled_generation_mw", "sum"),
        )
        .reset_index()
    )

    fig, (ax_load, ax_error) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    actual = sample.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")
    ax_load.plot(
        actual["target_timestamp_ept"],
        actual["true_zone_load_mw"],
        label="Actual",
        color="black",
        linewidth=2.5,
    )
    for model in _ordered_models(sample):
        model_df = sample[sample["model"] == model].sort_values("target_timestamp_ept")
        if not model_df.empty:
            ax_load.plot(
                model_df["target_timestamp_ept"],
                model_df["scheduled_generation_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
            ax_error.plot(
                model_df["target_timestamp_ept"],
                model_df["scheduled_generation_mw"] - model_df["true_zone_load_mw"],
                label=model,
                **MODEL_STYLES.get(model, {}),
            )
    ax_error.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    ax_load.set_title("Post-Constraint Layer: Scheduled vs True Total Load Across Zones")
    ax_load.set_ylabel("MW")
    ax_error.set_xlabel("Target timestamp EPT")
    ax_error.set_ylabel("Error MW")
    ax_load.legend(ncol=3)
    ax_error.legend(ncol=3)
    plt.xticks(rotation=30, ha="right")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()
