# EKPC 24-Hour-Ahead Load Forecast — Neural Network

## Overview

This document describes a small feedforward neural network that forecasts hourly electricity load for the PJM Western region **EKPC** load area. The model predicts load **24 hours ahead** using lagged load history, calendar features, and optional Kentucky weather data. Results are compared against a **persistence** baseline (current load as the forecast).

**Load data:** `2022_load_ekpc.csv` through `Western_EKPC_load_metered.csv` (Jun 2022–May 2026)  
**Weather data:** `data/4_year_kentucky_weather_data.csv` (hourly 2 m temperature and humidity, Jun 2022–Jun 2026, UTC)  
**Script:** `src/ekpc_neural_network.py`  
**Results directory:** `results/ekpc_neural_network/`

## Data

| Item | Value |
|------|-------|
| Zone | EKPC |
| Load area | EKPC |
| UTC range | 2022-06-01 04:00:00 to 2026-05-01 03:00:00 |
| Clean hourly rows | 33552 |
| Feature rows (after lags/target) | 24768 |

Rows are hourly metered load (`mw` → `load_mw`). Only verified EKPC rows are retained after cleaning.

## Forecasting task

At decision time `t`, the model uses features built from time `t` and past load to predict:

```text
target_load_mw(t) = load_mw at t + 24 hours
```

## Features

### Calendar

| Feature | Description |
|---------|-------------|
| `month` | Calendar month |
| `is_weekend` | 1 if Saturday or Sunday |
| `sin_hour`, `cos_hour` | Cyclical hour encoding |
| `sin_day_of_week`, `cos_day_of_week` | Cyclical weekday encoding |
| `sin_day_of_year`, `cos_day_of_year` | Cyclical annual encoding |

### Load history

| Feature group | Description |
|---------------|-------------|
| `load_lag_1` … `load_lag_24` | Past 24 hours of load |
| `load_lag_48` | Load 48 hours before `t` |
| `load_lag_168` | Load one week before `t` |
| `load_lag_8760` | Load same hour one year before `t` |
| `load_mw` | Current load at `t` (same clock hour as target on previous day) |
| `rolling_mean_24`, `rolling_std_24` | 24-hour rolling statistics |
| `rolling_mean_168`, `rolling_std_168` | 168-hour rolling statistics |

### Weather (Kentucky, UTC-aligned)

| Feature | Description |
|---------|-------------|
| `temperature_c` | 2 m air temperature (°C) at forecast issue time `t` |
| `humidity_pct` | Relative humidity (%) at `t` |
| `temperature_lag_24` | Temperature 24 hours before `t` |
| `humidity_lag_24` | Humidity 24 hours before `t` |

Weather is merged on `timestamp_utc` with load. Only weather known at `t` is used (no target-hour weather, avoiding lookahead).

**Input dimension (no weather):** 40 numeric features  
**Input dimension (with weather):** 44 numeric features  

Features are standardized before the network.

## Model architecture

| Layer | Units | Activation |
|-------|------:|------------|
| Input | 40 or 44 | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Output | 1 | linear |

**Implementation:** `sklearn.neural_network.MLPRegressor` inside a `Pipeline` with `StandardScaler`.

The network is trained to predict the **residual** between the 24-hour-ahead target and the persistence forecast (`load_mw` at time `t`). Final predictions are `load_mw + predicted_residual`, which anchors the model on the strong daily persistence signal.

| Hyperparameter | Value |
|----------------|-------|
| Solver | Adam |
| Training loss | Squared error (MSE) |
| L2 penalty (`alpha`) | 0.01 |
| Learning rate | 0.0001 |
| Batch size | 128 |
| Max iterations | 600 |
| Early stopping | Yes (10% of training rows held out internally) |
| Converged | False |
| Random seed | 42 |

## Train / validation / test split

Chronological split by unique `timestamp_utc` (no shuffling):

| Split | Timestamps | Rows |
|-------|----------:|-----:|
| Train | 17337 | 17337 |
| Validation | 3715 | 3715 |
| Test | 3716 | 3716 |

Fractions: 70% train, 15% validation, 15% test.

## Evaluation metrics

- **MAE** — mean absolute error (MW)
- **RMSE** — root mean squared error (MW)
- **MAPE** — mean absolute percentage error
- **Bias** — mean signed error (prediction − actual)
- **Peak-hour MAE** — MAE on test hours in the top 10% of realized load

## Test-set performance

### Persistence baseline

Forecasts `load_mw` at time `t` as the prediction for load at `t + 24`.

| MAE (MW) | 218.080 |
| RMSE (MW) | 303.388 |
| MAPE (%) | 12.641 |
| Bias (MW) | 0.614 |
| Peak-hour MAE (MW) | 440.459 |
| N | 3716 |

### Neural network (load + calendar only)

| MAE (MW) | 203.217 |
| RMSE (MW) | 276.713 |
| MAPE (%) | 12.118 |
| Bias (MW) | 14.922 |
| Peak-hour MAE (MW) | 392.699 |
| N | 3716 |

### Neural network + weather

| MAE (MW) | 204.484 |
| RMSE (MW) | 278.802 |
| MAPE (%) | 12.122 |
| Bias (MW) | 27.509 |
| Peak-hour MAE (MW) | 393.356 |
| N | 3716 |

### Comparison (Persistence − Neural Network)

| Metric | Δ (positive = neural network better) |
|--------|--------------------------------------|
| MAE | +14.863 MW |
| RMSE | +26.675 MW |

### Weather uplift (base NN − NN + weather)

| Metric | Δ (positive = weather model better) |
|--------|-------------------------------------|
| MAE | -1.267 MW |
| RMSE | -2.089 MW |

**Training iterations (base NN):** 600  
**Training iterations (NN + weather):** 600

## Output artifacts

| File | Description |
|------|-------------|
| `results/ekpc_neural_network/figures/model_comparison.png` | Bar chart of errors and test-period time series |
| `results/ekpc_neural_network/predictions/test_predictions.csv` | Test-set predictions for all models |
| `results/ekpc_neural_network/metrics/test_metrics.json` | Test metrics by model |
| `results/ekpc_neural_network/metrics/validation_metrics.json` | Validation metrics (model selection reference) |
| `docs/ekpc_neural_network.md` | This report |

## How to run

From the repository root:

```bash
python src/ekpc_neural_network.py
```
