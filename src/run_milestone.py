"""Run the CS229 milestone load forecasting pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from constraints import create_generator_fleet, make_post_constraint_metrics, run_constrained_dispatch
from data import clean_pjm_data, load_raw_data, validate_clean_data
from evaluate import make_error_by_hour, make_metrics_by_load_area, make_metrics_table
from features import make_feature_dataset
from models import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, predict_persistence, train_ols, tune_lasso, tune_ridge
from operational import aggregate_predictions_to_zone
from plots import (
    plot_error_by_hour,
    plot_forecast_metrics_bar,
    plot_post_constraint_dispatch_cost,
    plot_post_constraint_scheduled_vs_true,
    plot_post_constraint_under_generation,
    plot_true_vs_predicted_average,
    plot_true_vs_predicted_load_area,
)
from split import get_feature_target_metadata, time_based_split, validate_split


RAW_DATA_PATH = "data/hrl_load_metered.csv"
RESULTS_DIR = "results/milestone"


def _make_output_dirs(base_dir: Path) -> dict[str, Path]:
    """Create and return milestone output directories."""
    dirs = {
        "metrics": base_dir / "metrics",
        "predictions": base_dir / "predictions",
        "figures": base_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _build_prediction_frame(metadata: pd.DataFrame, y_pred, model_name: str) -> pd.DataFrame:
    """Build long-format prediction rows for one model."""
    out = metadata.copy()
    out["model"] = model_name
    out["true_load_mw"] = out["target_load_mw"]
    out["predicted_load_mw"] = y_pred
    out["error_mw"] = out["predicted_load_mw"] - out["true_load_mw"]
    out["abs_error_mw"] = out["error_mw"].abs()
    out["hour"] = out["target_timestamp_ept"].dt.hour
    out["day_of_week"] = out["target_timestamp_ept"].dt.dayofweek
    out["month"] = out["target_timestamp_ept"].dt.month
    columns = [
        "timestamp_utc",
        "timestamp_ept",
        "target_timestamp_utc",
        "target_timestamp_ept",
        "zone",
        "load_area",
        "model",
        "true_load_mw",
        "predicted_load_mw",
        "error_mw",
        "abs_error_mw",
        "hour",
        "day_of_week",
        "month",
    ]
    return out[columns]


def main() -> None:
    """Run the full milestone pipeline and save metrics, predictions, and figures."""
    results_dir = Path(RESULTS_DIR)
    dirs = _make_output_dirs(results_dir)

    raw = load_raw_data(RAW_DATA_PATH)
    clean = clean_pjm_data(raw)
    clean_summary = validate_clean_data(clean)
    clean.to_csv(dirs["predictions"] / "pre_constraint_layer_cleaned_data_snapshot.csv", index=False)

    feature_df = make_feature_dataset(clean)
    feature_df.to_csv(dirs["predictions"] / "pre_constraint_layer_feature_data_snapshot.csv", index=False)

    train_df, val_df, test_df = time_based_split(feature_df)
    validate_split(train_df, val_df, test_df)

    X_train, y_train, _ = get_feature_target_metadata(train_df)
    X_val, y_val, _ = get_feature_target_metadata(val_df)
    X_test, y_test, test_metadata = get_feature_target_metadata(test_df)

    print("\nTraining models")
    persistence_test_pred = predict_persistence(X_test)
    print("Persistence predictions complete.")

    print("Training OLS...")
    ols_model = train_ols(X_train, y_train, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)
    ols_test_pred = ols_model.predict(X_test)
    print("OLS complete.")

    print("Tuning Ridge...")
    best_ridge_model, best_alpha, ridge_val_results = tune_ridge(
        X_train,
        y_train,
        X_val,
        y_val,
        [0.01, 0.1, 1.0, 10.0, 100.0],
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )
    ridge_test_pred = best_ridge_model.predict(X_test)
    ridge_val_results.to_csv(dirs["metrics"] / "pre_constraint_layer_ridge_validation_results.csv", index=False)
    print(f"Ridge complete. Best alpha: {best_alpha}")

    print("Tuning Lasso...")
    best_lasso_model, best_lasso_alpha, lasso_val_results = tune_lasso(
        X_train,
        y_train,
        X_val,
        y_val,
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )
    lasso_test_pred = best_lasso_model.predict(X_test)
    lasso_val_results.to_csv(dirs["metrics"] / "pre_constraint_layer_lasso_validation_results.csv", index=False)
    print(f"Lasso complete. Best alpha: {best_lasso_alpha}")

    predictions = pd.concat(
        [
            _build_prediction_frame(test_metadata, persistence_test_pred, "Persistence"),
            _build_prediction_frame(test_metadata, ols_test_pred, "OLS"),
            _build_prediction_frame(test_metadata, ridge_test_pred, "Ridge"),
            _build_prediction_frame(test_metadata, lasso_test_pred, "Lasso"),
        ],
        ignore_index=True,
    )
    predictions.to_csv(dirs["predictions"] / "pre_constraint_layer_test_predictions.csv", index=False)

    metrics = make_metrics_table(predictions)
    metrics_by_area = make_metrics_by_load_area(predictions)
    error_by_hour = make_error_by_hour(predictions)
    metrics.to_csv(dirs["metrics"] / "pre_constraint_layer_forecast_metrics.csv", index=False)
    metrics_by_area.to_csv(dirs["metrics"] / "pre_constraint_layer_forecast_metrics_by_load_area.csv", index=False)
    error_by_hour.to_csv(dirs["metrics"] / "pre_constraint_layer_error_by_hour.csv", index=False)

    for load_area in sorted(predictions["load_area"].unique()):
        plot_true_vs_predicted_load_area(
            predictions,
            dirs["figures"] / f"pre_constraint_layer_true_vs_predicted_{load_area}.png",
            load_area=load_area,
        )
    plot_true_vs_predicted_average(
        predictions,
        dirs["figures"] / "pre_constraint_layer_true_vs_predicted_average_load_area.png",
    )
    plot_error_by_hour(error_by_hour, dirs["figures"] / "pre_constraint_layer_error_by_hour.png")
    plot_forecast_metrics_bar(metrics, dirs["figures"] / "pre_constraint_layer_rmse_by_model.png", metric="rmse")
    plot_forecast_metrics_bar(metrics, dirs["figures"] / "pre_constraint_layer_mape_by_model.png", metric="mape")

    pre_constraint_zone_predictions = aggregate_predictions_to_zone(predictions)
    pre_constraint_zone_predictions.to_csv(
        dirs["predictions"] / "pre_constraint_layer_zone_predictions.csv",
        index=False,
    )

    generator_fleet = create_generator_fleet(pre_constraint_zone_predictions["true_zone_load_mw"].max())
    generator_fleet.to_csv(dirs["metrics"] / "post_constraint_layer_generator_fleet.csv", index=False)

    dispatch_hourly = run_constrained_dispatch(pre_constraint_zone_predictions, generator_fleet)
    post_constraint_metrics = make_post_constraint_metrics(dispatch_hourly)
    dispatch_hourly.to_csv(dirs["predictions"] / "post_constraint_layer_dispatch_hourly.csv", index=False)
    post_constraint_metrics.to_csv(dirs["metrics"] / "post_constraint_layer_dispatch_metrics.csv", index=False)

    pre_post_summary = metrics.merge(
        post_constraint_metrics[
            [
                "model",
                "under_generation_rate",
                "total_under_generation_mw",
                "total_over_generation_mw",
                "total_dispatch_cost",
                "total_cost_gap",
            ]
        ],
        on="model",
        how="left",
    ).rename(
        columns={
            "rmse": "pre_constraint_layer_rmse",
            "mae": "pre_constraint_layer_mae",
            "mape": "pre_constraint_layer_mape",
            "bias": "pre_constraint_layer_bias",
            "under_generation_rate": "post_constraint_layer_under_generation_rate",
            "total_under_generation_mw": "post_constraint_layer_total_under_generation_mw",
            "total_over_generation_mw": "post_constraint_layer_total_over_generation_mw",
            "total_dispatch_cost": "post_constraint_layer_total_dispatch_cost",
            "total_cost_gap": "post_constraint_layer_total_cost_gap",
        }
    )
    pre_post_summary = pre_post_summary[
        [
            "model",
            "pre_constraint_layer_rmse",
            "pre_constraint_layer_mae",
            "pre_constraint_layer_mape",
            "pre_constraint_layer_bias",
            "post_constraint_layer_under_generation_rate",
            "post_constraint_layer_total_under_generation_mw",
            "post_constraint_layer_total_over_generation_mw",
            "post_constraint_layer_total_dispatch_cost",
            "post_constraint_layer_total_cost_gap",
        ]
    ]
    pre_post_summary.to_csv(dirs["metrics"] / "pre_post_constraint_layer_summary.csv", index=False)

    plot_post_constraint_dispatch_cost(
        post_constraint_metrics,
        dirs["figures"] / "post_constraint_layer_dispatch_cost_by_model.png",
    )
    plot_post_constraint_under_generation(
        post_constraint_metrics,
        dirs["figures"] / "post_constraint_layer_under_generation_by_model.png",
    )
    plot_post_constraint_scheduled_vs_true(
        dispatch_hourly,
        dirs["figures"] / "post_constraint_layer_scheduled_vs_true_zone_load.png",
    )

    print("\nMilestone summary")
    print(f"Data rows after cleaning: {clean_summary['n_rows']}")
    print(f"Feature rows: {len(feature_df)}")
    print(f"Train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Best ridge alpha: {best_alpha}")
    print(f"Best lasso alpha: {best_lasso_alpha}")
    print("\nPre-constraint layer forecast metrics")
    print(metrics.to_string(index=False))
    print("\nPost-constraint layer dispatch metrics")
    print(post_constraint_metrics.to_string(index=False))
    print("\nSaved outputs")
    print(f"Metrics: {dirs['metrics']}")
    print(f"Predictions: {dirs['predictions']}")
    print(f"Figures: {dirs['figures']}")


if __name__ == "__main__":
    main()
