# Energy Demand Forecast with Constraints

This repository contains the CS229 milestone pipeline for 24-hour-ahead AEP load forecasting.

## Run

Install dependencies (venv contains dependencies):

```bash
pip install -r requirements.txt
```

Run the milestone pipeline from the repository root (using venv for dependencies):

```bash
.venv/bin/python src/run_milestone.py
```

## Outputs

The pipeline writes all milestone artifacts under `results/milestone/`:

- `metrics/pre_constraint_layer_forecast_metrics.csv`
- `metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv`
- `metrics/pre_constraint_layer_error_by_hour.csv`
- `metrics/pre_constraint_layer_ridge_validation_results.csv`
- `metrics/post_constraint_layer_generator_fleet.csv`
- `metrics/post_constraint_layer_dispatch_metrics.csv`
- `metrics/pre_post_constraint_layer_summary.csv`
- `predictions/pre_constraint_layer_test_predictions.csv`
- `predictions/pre_constraint_layer_zone_predictions.csv`
- `predictions/post_constraint_layer_dispatch_hourly.csv`
- `figures/pre_constraint_layer_true_vs_predicted_average_load_area.png`
- `figures/pre_constraint_layer_error_by_hour.png`
- `figures/pre_constraint_layer_rmse_by_model.png`
- `figures/post_constraint_layer_dispatch_cost_by_model.png`
- `figures/post_constraint_layer_under_generation_by_model.png`
- `figures/post_constraint_layer_scheduled_vs_true_zone_load.png`

The milestone models are Persistence, Ordinary Least Squares, and Ridge regression.
