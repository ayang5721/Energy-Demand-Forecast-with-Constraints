"""Run the load forecasting and constrained dispatch pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from constraints import (
    create_generator_fleet,
    make_post_constraint_metrics,
    run_constrained_dispatch,
    validate_constraint_costs,
)
from data import (
    clean_pjm_data,
    filter_weather_to_load_range,
    load_raw_data,
    load_weather_data,
    validate_clean_data,
)
from evaluate import make_error_by_hour, make_metrics_by_load_area, make_metrics_table
from features import FEATURE_COLUMNS, WEATHER_FEATURE_COLUMNS, make_feature_dataset
from models import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    WEATHER_NUMERIC_COLUMNS,
    predict_persistence,
    train_neural_network,
    train_ols,
    tune_lasso,
    tune_ridge,
)
from operational import aggregate_predictions_to_zone
from plots import (
    plot_error_by_hour,
    plot_forecast_metrics_bar,
    plot_post_constraint_base_generator_cost,
    plot_post_constraint_constraint_regret,
    plot_post_constraint_over_generation,
    plot_post_constraint_penalty_cost,
    plot_post_constraint_penalty_cost_stacked,
    plot_post_constraint_scheduled_vs_true,
    plot_post_constraint_total_operational_cost,
    plot_post_constraint_under_generation,
    plot_true_vs_predicted_average,
    plot_true_vs_predicted_load_area,
)
from split import get_feature_target_metadata, time_based_split, validate_split


RAW_DATA_PATH = "data/hrl_load_metered.csv"
WEATHER_DATA_PATH = "data/4_year_kentucky_weather_data.csv"
RESULTS_DIR = "results"


def _make_output_dirs(base_dir: Path) -> dict[str, Path]:
    """Create and return output directories."""
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


def _safe_filename(value: str) -> str:
    """Return a filesystem-friendly label."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def main() -> None:
    """Run the full pipeline and save metrics, predictions, and figures."""
    results_dir = Path(RESULTS_DIR)
    dirs = _make_output_dirs(results_dir)

    raw = load_raw_data(RAW_DATA_PATH)
    weather = load_weather_data(WEATHER_DATA_PATH)
    clean = clean_pjm_data(raw)
    weather = filter_weather_to_load_range(weather, clean)
    clean_summary = validate_clean_data(clean)
    clean.to_csv(dirs["predictions"] / "pre_constraint_layer_cleaned_data_snapshot.csv", index=False)

    feature_df = make_feature_dataset(clean, weather)
    feature_df.to_csv(dirs["predictions"] / "pre_constraint_layer_feature_data_snapshot.csv", index=False)

    train_df, val_df, test_df = time_based_split(feature_df)
    validate_split(train_df, val_df, test_df)

    X_train, y_train, _ = get_feature_target_metadata(train_df, FEATURE_COLUMNS)
    X_val, y_val, _ = get_feature_target_metadata(val_df, FEATURE_COLUMNS)
    X_test, y_test, test_metadata = get_feature_target_metadata(test_df, FEATURE_COLUMNS)
    X_weather_train, _, _ = get_feature_target_metadata(train_df, WEATHER_FEATURE_COLUMNS)
    X_weather_test, _, _ = get_feature_target_metadata(test_df, WEATHER_FEATURE_COLUMNS)

    print("\nTraining models")
    persistence_test_pred = predict_persistence(X_test)
    print("Persistence predictions complete.")

    print("Training OLS...")
    ols_model = train_ols(X_train, y_train, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)
    ols_test_pred = ols_model.predict(X_test)
    print("OLS complete.")

    print("Training Neural Network...")
    neural_network_model = train_neural_network(X_train, y_train, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)
    neural_network_test_pred = neural_network_model.predict(X_test)
    print("Neural Network complete.")

    print("Training Neural Network + Weather...")
    weather_neural_network_model = train_neural_network(
        X_weather_train,
        y_train,
        CATEGORICAL_COLUMNS,
        WEATHER_NUMERIC_COLUMNS,
    )
    weather_neural_network_test_pred = weather_neural_network_model.predict(X_weather_test)
    print("Neural Network + Weather complete.")

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
            _build_prediction_frame(test_metadata, neural_network_test_pred, "Neural Network"),
            _build_prediction_frame(test_metadata, weather_neural_network_test_pred, "Neural Network + Weather"),
            _build_prediction_frame(test_metadata, ridge_test_pred, "Ridge"),
            _build_prediction_frame(test_metadata, lasso_test_pred, "Lasso"),
        ],
        ignore_index=True,
    )
    print("Saving pre-constraint predictions and metrics...")
    predictions.to_csv(dirs["predictions"] / "pre_constraint_layer_test_predictions.csv", index=False)

    metrics = make_metrics_table(predictions)
    metrics_by_area = make_metrics_by_load_area(predictions)
    error_by_hour = make_error_by_hour(predictions)
    metrics.to_csv(dirs["metrics"] / "pre_constraint_layer_forecast_metrics.csv", index=False)
    metrics_by_area.to_csv(dirs["metrics"] / "pre_constraint_layer_forecast_metrics_by_load_area.csv", index=False)
    error_by_hour.to_csv(dirs["metrics"] / "pre_constraint_layer_error_by_hour.csv", index=False)

    print("Generating pre-constraint plots...")
    pre_constraint_load_area_dir = dirs["figures"] / "pre_constraint_load_area"
    pre_constraint_error_dir = dirs["figures"] / "pre_constraint_error"
    post_constraint_analysis_dir = dirs["figures"] / "post_constraint_analysis"
    for row in (
        predictions[["zone", "load_area"]]
        .drop_duplicates()
        .sort_values(["zone", "load_area"])
        .itertuples(index=False)
    ):
        plot_true_vs_predicted_load_area(
            predictions,
            pre_constraint_load_area_dir
            / f"pre_constraint_layer_true_vs_predicted_{_safe_filename(row.zone)}_{_safe_filename(row.load_area)}.png",
            zone=row.zone,
            load_area=row.load_area,
        )
    plot_true_vs_predicted_average(
        predictions,
        pre_constraint_load_area_dir / "pre_constraint_layer_true_vs_predicted_average_load_area.png",
    )
    plot_error_by_hour(error_by_hour, pre_constraint_error_dir / "pre_constraint_layer_error_by_hour.png")
    plot_forecast_metrics_bar(metrics, pre_constraint_error_dir / "pre_constraint_layer_rmse_by_model.png", metric="rmse")
    plot_forecast_metrics_bar(metrics, pre_constraint_error_dir / "pre_constraint_layer_mape_by_model.png", metric="mape")
    plot_forecast_metrics_bar(metrics, pre_constraint_error_dir / "pre_constraint_layer_bias_by_model.png", metric="bias")

    print("Aggregating load-area forecasts to zones...")
    pre_constraint_zone_predictions = aggregate_predictions_to_zone(predictions)
    pre_constraint_zone_predictions.to_csv(
        dirs["predictions"] / "pre_constraint_layer_zone_predictions.csv",
        index=False,
    )

    generator_fleet = create_generator_fleet(pre_constraint_zone_predictions["true_zone_load_mw"].max())
    generator_fleet.to_csv(dirs["metrics"] / "post_constraint_layer_generator_fleet.csv", index=False)

    print("Running post-constraint dispatch...")
    dispatch_hourly = run_constrained_dispatch(pre_constraint_zone_predictions, generator_fleet)
    post_constraint_metrics = make_post_constraint_metrics(dispatch_hourly)
    validate_constraint_costs(dispatch_hourly, post_constraint_metrics)
    print("Saving post-constraint dispatch outputs...")
    dispatch_hourly.to_csv(dirs["predictions"] / "post_constraint_layer_dispatch_hourly.csv", index=False)
    post_constraint_metrics.to_csv(dirs["metrics"] / "post_constraint_layer_dispatch_metrics.csv", index=False)

    pre_post_summary = metrics.merge(
        post_constraint_metrics[
            [
                "model",
                "under_generation_rate",
                "over_generation_rate",
                "total_under_generation_mwh",
                "total_over_generation_mwh",
                "total_base_generator_cost",
                "total_penalty_cost",
                "total_operational_cost",
                "total_constraint_regret",
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
            "over_generation_rate": "post_constraint_layer_over_generation_rate",
            "total_under_generation_mwh": "post_constraint_layer_total_under_generation_mwh",
            "total_over_generation_mwh": "post_constraint_layer_total_over_generation_mwh",
            "total_base_generator_cost": "post_constraint_layer_total_base_generator_cost",
            "total_penalty_cost": "post_constraint_layer_total_penalty_cost",
            "total_operational_cost": "post_constraint_layer_total_operational_cost",
            "total_constraint_regret": "post_constraint_layer_total_constraint_regret",
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
            "post_constraint_layer_over_generation_rate",
            "post_constraint_layer_total_under_generation_mwh",
            "post_constraint_layer_total_over_generation_mwh",
            "post_constraint_layer_total_base_generator_cost",
            "post_constraint_layer_total_penalty_cost",
            "post_constraint_layer_total_operational_cost",
            "post_constraint_layer_total_constraint_regret",
        ]
    ]
    pre_post_summary.to_csv(dirs["metrics"] / "pre_post_constraint_layer_summary.csv", index=False)

    print("Generating post-constraint plots...")
    plot_post_constraint_base_generator_cost(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_base_generator_cost_by_model.png",
    )
    plot_post_constraint_under_generation(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_under_generation_by_model.png",
    )
    plot_post_constraint_over_generation(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_over_generation_by_model.png",
    )
    plot_post_constraint_penalty_cost(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_penalty_cost_by_model.png",
    )
    plot_post_constraint_penalty_cost_stacked(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_penalty_cost_stacked_by_model.png",
    )
    plot_post_constraint_total_operational_cost(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_total_operational_cost_by_model.png",
    )
    plot_post_constraint_constraint_regret(
        post_constraint_metrics,
        post_constraint_analysis_dir / "post_constraint_layer_constraint_regret_by_model.png",
    )
    plot_post_constraint_scheduled_vs_true(
        dispatch_hourly,
        post_constraint_analysis_dir / "post_constraint_layer_scheduled_vs_true_zone_load.png",
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
