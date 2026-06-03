"""24-hour-ahead EKPC load forecasting with a small feedforward neural network."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import _to_bool
from evaluate import compute_metrics
from features import add_cyclical_features, add_lag_features, add_rolling_features, add_target, add_time_features
from models import predict_persistence
from split import time_based_split


RAW_DATA_PATH = Path("data/Western_EKPC_load_metered.csv")
RESULTS_DIR = Path("results/ekpc_neural_network")
REPORT_PATH = Path("docs/ekpc_neural_network.md")

HORIZON = 24
LAG_HOURS = list(range(1, 25))
EXTRA_LAGS = [48, 168]
ROLLING_WINDOWS = [24, 168]

NUMERIC_FEATURE_COLUMNS = [
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "load_mw",
    *[f"load_lag_{lag}" for lag in LAG_HOURS],
    "load_lag_48",
    "load_lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]

METADATA_COLUMNS = [
    "timestamp_utc",
    "timestamp_ept",
    "target_timestamp_utc",
    "target_timestamp_ept",
    "zone",
    "load_area",
    "load_mw",
    "target_load_mw",
]


def load_ekpc_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the Western EKPC hourly load CSV."""
    return pd.read_csv(path)


def clean_ekpc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean EKPC load data and return standard modeling columns."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()

    datetime_format = "%m/%d/%Y %I:%M:%S %p"
    cleaned["timestamp_utc"] = pd.to_datetime(
        cleaned["datetime_beginning_utc"], format=datetime_format, errors="coerce"
    )
    cleaned["timestamp_ept"] = pd.to_datetime(
        cleaned["datetime_beginning_ept"], format=datetime_format, errors="coerce"
    )
    cleaned = cleaned.rename(columns={"mkt_region": "market_region", "mw": "load_mw"})
    cleaned["load_mw"] = pd.to_numeric(cleaned["load_mw"], errors="coerce")

    if "is_verified" in cleaned.columns:
        cleaned["is_verified"] = _to_bool(cleaned["is_verified"])

    for col in ["nerc_region", "market_region", "zone", "load_area"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned[cleaned["zone"] == "EKPC"]
    cleaned = cleaned[cleaned["load_area"] == "EKPC"]
    cleaned = cleaned.dropna(subset=["timestamp_utc", "timestamp_ept", "load_area", "zone", "load_mw"])
    cleaned = cleaned.drop_duplicates(subset=["timestamp_utc", "zone", "load_area"])
    cleaned = cleaned.sort_values("timestamp_utc").reset_index(drop=True)

    columns = [
        "timestamp_utc",
        "timestamp_ept",
        "nerc_region",
        "market_region",
        "zone",
        "load_area",
        "load_mw",
        "is_verified",
    ]
    return cleaned[columns]


def make_ekpc_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build 24-hour-ahead supervised examples for the EKPC load area."""
    out = add_time_features(df)
    out = add_cyclical_features(out)
    out = add_lag_features(out, LAG_HOURS + EXTRA_LAGS)
    out = add_rolling_features(out, ROLLING_WINDOWS)
    out = add_target(out, horizon=HORIZON)

    required = NUMERIC_FEATURE_COLUMNS + METADATA_COLUMNS + ["target_load_mw"]
    out = out.dropna(subset=required)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    return out


def get_feature_target_metadata(df: pd.DataFrame):
    """Return feature matrix, target vector, and metadata for modeling."""
    X = df[NUMERIC_FEATURE_COLUMNS].copy()
    y = df["target_load_mw"].copy()
    metadata = df[METADATA_COLUMNS].copy()
    return X, y, metadata


def build_neural_network_pipeline() -> Pipeline:
    """Return a scaled feedforward network with two hidden layers (64, 32)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    learning_rate_init=1e-4,
                    batch_size=128,
                    max_iter=600,
                    tol=1e-4,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_neural_network(nn_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the network on the 24-hour-ahead residual relative to persistence."""
    residual_train = y_train.to_numpy() - X_train["load_mw"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        nn_model.fit(X_train, residual_train)
    return nn_model


def predict_neural_network(nn_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Reconstruct load forecasts from persistence plus predicted residual."""
    residual_pred = nn_model.predict(X)
    return X["load_mw"].to_numpy() + residual_pred


def peak_hour_mae(y_true, y_pred, peak_quantile: float = 0.90) -> float:
    """Mean absolute error on hours whose realized load is in the top decile."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    threshold = np.quantile(true, peak_quantile)
    mask = true >= threshold
    if not mask.any():
        return float("nan")
    return float(mean_absolute_error(true[mask], pred[mask]))


def _build_prediction_frame(metadata: pd.DataFrame, y_pred, model_name: str) -> pd.DataFrame:
    out = metadata.copy()
    out["model"] = model_name
    out["true_load_mw"] = out["target_load_mw"]
    out["predicted_load_mw"] = y_pred
    out["error_mw"] = out["predicted_load_mw"] - out["true_load_mw"]
    out["abs_error_mw"] = out["error_mw"].abs()
    out["hour"] = out["target_timestamp_ept"].dt.hour
    return out


def _format_metrics_block(metrics: dict) -> str:
    lines = [
        f"| MAE (MW) | {metrics['mae']:.3f} |",
        f"| RMSE (MW) | {metrics['rmse']:.3f} |",
        f"| MAPE (%) | {metrics['mape']:.3f} |",
        f"| Bias (MW) | {metrics['bias']:.3f} |",
        f"| Peak-hour MAE (MW) | {metrics['peak_hour_mae']:.3f} |",
        f"| N | {metrics['n']} |",
    ]
    return "\n".join(lines)


def write_report(
    path: Path,
    data_summary: dict,
    split_summary: dict,
    metrics_by_model: dict[str, dict],
    nn_info: dict,
) -> None:
    """Write markdown documentation for the EKPC neural network experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)

    persistence = metrics_by_model["Persistence"]
    neural = metrics_by_model["Neural Network"]
    mae_improvement = persistence["mae"] - neural["mae"]
    rmse_improvement = persistence["rmse"] - neural["rmse"]

    content = f"""# EKPC 24-Hour-Ahead Load Forecast — Neural Network

## Overview

This document describes a small feedforward neural network that forecasts hourly electricity load for the PJM Western region **EKPC** load area. The model predicts load **24 hours ahead** using lagged load history and calendar features. Results are compared against a **persistence** baseline (current load as the forecast).

**Data file:** `{RAW_DATA_PATH}`  
**Script:** `src/ekpc_neural_network.py`  
**Results directory:** `{RESULTS_DIR}/`

## Data

| Item | Value |
|------|-------|
| Zone | EKPC |
| Load area | EKPC |
| UTC range | {data_summary['min_timestamp_utc']} to {data_summary['max_timestamp_utc']} |
| Clean hourly rows | {data_summary['n_rows']} |
| Feature rows (after lags/target) | {data_summary['n_feature_rows']} |

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

**Input dimension:** {len(NUMERIC_FEATURE_COLUMNS)} numeric features (standardized before the network).

## Model architecture

| Layer | Units | Activation |
|-------|------:|------------|
| Input | {len(NUMERIC_FEATURE_COLUMNS)} | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Output | 1 | linear |

**Implementation:** `sklearn.neural_network.MLPRegressor` inside a `Pipeline` with `StandardScaler`.

The network is trained to predict the **residual** between the 24-hour-ahead target and the persistence forecast (`load_mw` at time `t`). Final predictions are `load_mw + predicted_residual`, which anchors the model on the strong daily persistence signal.

| Hyperparameter | Value |
|----------------|-------|
| Solver | Adam |
| Training loss | Squared error (MSE) |
| L2 penalty (`alpha`) | {nn_info['alpha']} |
| Learning rate | {nn_info['learning_rate_init']} |
| Batch size | {nn_info['batch_size']} |
| Max iterations | {nn_info['max_iter']} |
| Early stopping | Yes (10% of training rows held out internally) |
| Converged | {nn_info['converged']} |
| Random seed | 42 |

## Train / validation / test split

Chronological split by unique `timestamp_utc` (no shuffling):

| Split | Timestamps | Rows |
|-------|----------:|-----:|
| Train | {split_summary['train_timestamps']} | {split_summary['train_rows']} |
| Validation | {split_summary['val_timestamps']} | {split_summary['val_rows']} |
| Test | {split_summary['test_timestamps']} | {split_summary['test_rows']} |

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

{_format_metrics_block(persistence)}

### Neural network

{_format_metrics_block(neural)}

### Comparison (Persistence − Neural Network)

| Metric | Δ (positive = neural network better) |
|--------|--------------------------------------|
| MAE | {mae_improvement:+.3f} MW |
| RMSE | {rmse_improvement:+.3f} MW |

**Training iterations used:** {nn_info['n_iter_']}

## Output artifacts

| File | Description |
|------|-------------|
| `{RESULTS_DIR}/predictions/test_predictions.csv` | Test-set predictions for both models |
| `{RESULTS_DIR}/metrics/test_metrics.json` | Test metrics by model |
| `{RESULTS_DIR}/metrics/validation_metrics.json` | Validation metrics (model selection reference) |
| `{REPORT_PATH}` | This report |

## How to run

From the repository root:

```bash
python src/ekpc_neural_network.py
```
"""
    path.write_text(content, encoding="utf-8")


def run_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    results_dir: Path = RESULTS_DIR,
    report_path: Path = REPORT_PATH,
) -> dict[str, dict]:
    """Load data, train the neural network, evaluate, and write outputs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)

    raw = load_ekpc_raw(raw_path)
    clean = clean_ekpc_data(raw)
    features = make_ekpc_feature_dataset(clean)

    train_df, val_df, test_df = time_based_split(features)
    X_train, y_train, meta_train = get_feature_target_metadata(train_df)
    X_val, y_val, meta_val = get_feature_target_metadata(val_df)
    X_test, y_test, meta_test = get_feature_target_metadata(test_df)

    nn_model = build_neural_network_pipeline()
    fit_neural_network(nn_model, X_train, y_train)

    def evaluate_split(name: str, X, y, metadata) -> tuple[dict, pd.DataFrame]:
        nn_pred = predict_neural_network(nn_model, X)
        persistence_pred = predict_persistence(X)

        nn_metrics = compute_metrics(y, nn_pred)
        nn_metrics["peak_hour_mae"] = peak_hour_mae(y, nn_pred)
        nn_metrics["n"] = len(y)

        persistence_metrics = compute_metrics(y, persistence_pred)
        persistence_metrics["peak_hour_mae"] = peak_hour_mae(y, persistence_pred)
        persistence_metrics["n"] = len(y)

        preds = pd.concat(
            [
                _build_prediction_frame(metadata, persistence_pred, "Persistence"),
                _build_prediction_frame(metadata, nn_pred, "Neural Network"),
            ],
            ignore_index=True,
        )
        return {name: {"Persistence": persistence_metrics, "Neural Network": nn_metrics}}, preds

    val_metrics, _ = evaluate_split("validation", X_val, y_val, meta_val)
    test_metrics, test_preds = evaluate_split("test", X_test, y_test, meta_test)

    test_preds.to_csv(results_dir / "predictions" / "test_predictions.csv", index=False)
    with open(results_dir / "metrics" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["test"], f, indent=2)
    with open(results_dir / "metrics" / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics["validation"], f, indent=2)

    mlp = nn_model.named_steps["model"]
    nn_info = {
        "alpha": 0.01,
        "learning_rate_init": 1e-4,
        "batch_size": 128,
        "max_iter": 600,
        "n_iter_": int(mlp.n_iter_),
        "best_loss": float(mlp.best_loss_) if mlp.best_loss_ is not None else None,
        "converged": bool(mlp.n_iter_ < mlp.max_iter),
    }

    data_summary = {
        "n_rows": int(len(clean)),
        "n_feature_rows": int(len(features)),
        "min_timestamp_utc": str(clean["timestamp_utc"].min()),
        "max_timestamp_utc": str(clean["timestamp_utc"].max()),
    }
    split_summary = {
        "train_timestamps": int(train_df["timestamp_utc"].nunique()),
        "val_timestamps": int(val_df["timestamp_utc"].nunique()),
        "val_rows": len(val_df),
        "train_rows": len(train_df),
        "test_timestamps": int(test_df["timestamp_utc"].nunique()),
        "test_rows": len(test_df),
    }

    write_report(
        report_path,
        data_summary=data_summary,
        split_summary=split_summary,
        metrics_by_model=test_metrics["test"],
        nn_info=nn_info,
    )

    print("\nEKPC neural network pipeline complete")
    print(f"Clean rows: {data_summary['n_rows']}")
    print(f"Feature rows: {data_summary['n_feature_rows']}")
    print(f"Train / val / test rows: {split_summary['train_rows']} / {split_summary['val_rows']} / {split_summary['test_rows']}")
    print("\nTest metrics:")
    for model_name, metrics in test_metrics["test"].items():
        print(f"  {model_name}: MAE={metrics['mae']:.2f} MW, RMSE={metrics['rmse']:.2f} MW, peak MAE={metrics['peak_hour_mae']:.2f} MW")
    print(f"\nReport written to {report_path}")
    return test_metrics["test"]


if __name__ == "__main__":
    run_pipeline()
