# EKPC 24-Hour-Ahead Load Forecast — Neural Network

## Overview

This document describes a small feedforward neural network that forecasts hourly electricity load for the PJM Western region **EKPC** load area. The model predicts load **24 hours ahead** using lagged load history and calendar features. Results are compared against a **persistence** baseline (current load as the forecast).

**Data file:** `data/Western_EKPC_load_metered.csv`  
**Script:** `src/ekpc_neural_network.py`  
**Results directory:** `results/ekpc_neural_network/`

## Data

| Item | Value |
|------|-------|
| Zone | EKPC |
| Load area | EKPC |
| UTC range | 2025-06-01 04:00:00 to 2026-05-01 03:00:00 |
| Clean hourly rows | 7248 |
| Feature rows (after lags/target) | 7056 |

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

### Load history

| Feature group | Description |
|---------------|-------------|
| `load_lag_1` … `load_lag_24` | Past 24 hours of load |
| `load_lag_48` | Load 48 hours before `t` |
| `load_lag_168` | Load one week before `t` |
| `load_mw` | Current load at `t` (same clock hour as target on previous day) |
| `rolling_mean_24`, `rolling_std_24` | 24-hour rolling statistics |
| `rolling_mean_168`, `rolling_std_168` | 168-hour rolling statistics |

**Input dimension:** 37 numeric features (standardized before the network).

## Model architecture

| Layer | Units | Activation |
|-------|------:|------------|
| Input | 37 | — |
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
| Train | 4939 | 4939 |
| Validation | 1058 | 1058 |
| Test | 1059 | 1059 |

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

| MAE (MW) | 136.165 |
| RMSE (MW) | 183.211 |
| MAPE (%) | 10.149 |
| Bias (MW) | 23.876 |
| Peak-hour MAE (MW) | 223.881 |
| N | 1059 |

### Neural network

| MAE (MW) | 139.308 |
| RMSE (MW) | 182.530 |
| MAPE (%) | 10.365 |
| Bias (MW) | -5.761 |
| Peak-hour MAE (MW) | 200.651 |
| N | 1059 |

### Comparison (Persistence − Neural Network)

| Metric | Δ (positive = neural network better) |
|--------|--------------------------------------|
| MAE | -3.143 MW |
| RMSE | +0.681 MW |

**Training iterations used:** 600

## Output artifacts

| File | Description |
|------|-------------|
| `results/ekpc_neural_network/predictions/test_predictions.csv` | Test-set predictions for both models |
| `results/ekpc_neural_network/metrics/test_metrics.json` | Test metrics by model |
| `results/ekpc_neural_network/metrics/validation_metrics.json` | Validation metrics (model selection reference) |
| `docs/ekpc_neural_network.md` | This report |

## How to run

From the repository root:

```bash
python src/ekpc_neural_network.py
```
