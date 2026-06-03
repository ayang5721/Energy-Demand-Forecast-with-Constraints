# Energy Demand Forecast with Constraints

This repository contains the CS229 milestone pipeline for 24-hour-ahead PJM load forecasting across all zones and load areas present in the input CSV.

## Run

Install dependencies (venv contains dependencies):

```bash
pip install -r requirements.txt
```

Run the milestone pipeline from the repository root (using venv for dependencies):

```bash
.venv/bin/python -u src/run.py
```

## Outputs

The pipeline writes all milestone artifacts under `results/milestone/`:

- `metrics/pre_constraint_layer_forecast_metrics.csv`
- `metrics/pre_constraint_layer_forecast_metrics_by_load_area.csv`
- `metrics/pre_constraint_layer_error_by_hour.csv`
- `metrics/pre_constraint_layer_ridge_validation_results.csv`
- `metrics/pre_constraint_layer_lasso_validation_results.csv`
- `metrics/post_constraint_layer_generator_fleet.csv`
- `metrics/post_constraint_layer_dispatch_metrics.csv`
- `metrics/pre_post_constraint_layer_summary.csv`
- `predictions/pre_constraint_layer_test_predictions.csv`
- `predictions/pre_constraint_layer_zone_predictions.csv`
- `predictions/post_constraint_layer_dispatch_hourly.csv`
- `figures/pre_constraint_load_area/`
- `figures/pre_constraint_error/`
- `figures/post_constraint_analysis/`

The milestone models are Persistence, Ordinary Least Squares, Ridge regression, and Lasso regression.
