# Energy Demand Forecast with Constraints

This repository contains the CS229 milestone pipeline for 24-hour-ahead AEP load forecasting.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the milestone pipeline from the repository root:

```bash
python src/run_milestone.py
```

## Outputs

The pipeline writes all milestone artifacts under `results/milestone/`:

- `metrics/forecast_metrics.csv`
- `metrics/forecast_metrics_by_load_area.csv`
- `metrics/error_by_hour.csv`
- `metrics/ridge_validation_results.csv`
- `metrics/zone_under_over_generation.csv`
- `predictions/test_predictions.csv`
- `predictions/zone_predictions.csv`
- `figures/true_vs_predicted_sample.png`
- `figures/error_by_hour.png`
- `figures/rmse_by_model.png`

The milestone models are Persistence, Ordinary Least Squares, and Ridge regression.
