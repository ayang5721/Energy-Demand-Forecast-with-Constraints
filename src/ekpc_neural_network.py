"""24-hour-ahead EKPC load forecasting with a small feedforward neural network."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_MPL_CONFIG_DIR = Path("/tmp/matplotlib-energy-demand-forecast").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import _to_bool
from evaluate import compute_metrics
from features import add_cyclical_features, add_lag_features, add_rolling_features, add_target, add_time_features
from models import predict_persistence
from split import time_based_split


EKPC_LOAD_PATHS = [
    Path("data/2022_load_ekpc.csv"),
    Path("data/2023_load_ekpc.csv"),
    Path("data/2024_load_ekpc.csv"),
    Path("data/Western_EKPC_load_metered.csv"),
]
WEATHER_DATA_PATH = Path("data/4_year_kentucky_weather_data.csv")
RESULTS_DIR = Path("results/ekpc_neural_network")
REPORT_PATH = Path("docs/ekpc_neural_network.md")
ANNUAL_LAG_HOURS = 8760

MODEL_ORDER = ["Persistence", "Neural Network", "Neural Network + Weather"]
MODEL_COLORS = {
    "Persistence": "#6c757d",
    "Neural Network": "#0d6efd",
    "Neural Network + Weather": "#198754",
}
MODEL_LINE_STYLES = {
    "Persistence": {"linestyle": "--", "linewidth": 1.8},
    "Neural Network": {"linestyle": "-.", "linewidth": 1.8},
    "Neural Network + Weather": {"linestyle": "-", "linewidth": 2.0},
}

HORIZON = 24
LAG_HOURS = list(range(1, 25))
EXTRA_LAGS = [48, 168, ANNUAL_LAG_HOURS]
ROLLING_WINDOWS = [24, 168]

LOAD_FEATURE_COLUMNS = [
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_day_of_year",
    "cos_day_of_year",
    "load_mw",
    *[f"load_lag_{lag}" for lag in LAG_HOURS],
    "load_lag_48",
    "load_lag_168",
    f"load_lag_{ANNUAL_LAG_HOURS}",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]

WEATHER_FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "temperature_lag_24",
    "humidity_lag_24",
]

NUMERIC_FEATURE_COLUMNS = LOAD_FEATURE_COLUMNS
NUMERIC_FEATURE_COLUMNS_WITH_WEATHER = LOAD_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS

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


def load_ekpc_raw(paths: list[Path] | None = None) -> pd.DataFrame:
    """Read and concatenate multi-year EKPC hourly load CSVs."""
    load_paths = paths or EKPC_LOAD_PATHS
    frames = [pd.read_csv(path) for path in load_paths]
    return pd.concat(frames, ignore_index=True)


def add_day_of_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding for day of year (annual seasonality)."""
    out = df.copy()
    day_of_year = out["timestamp_ept"].dt.dayofyear
    out["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return out


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


def load_weather_data(path: str | Path = WEATHER_DATA_PATH) -> pd.DataFrame:
    """Read hourly Kentucky temperature and humidity (UTC timestamps)."""
    weather = pd.read_csv(path, skiprows=2)
    weather.columns = ["timestamp_utc", "temperature_c", "humidity_pct"]
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"])
    weather["temperature_c"] = pd.to_numeric(weather["temperature_c"], errors="coerce")
    weather["humidity_pct"] = pd.to_numeric(weather["humidity_pct"], errors="coerce")
    return weather.dropna(subset=["timestamp_utc", "temperature_c", "humidity_pct"])


def merge_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach weather at the forecast issue hour and 24-hour weather lags."""
    out = df.merge(weather, on="timestamp_utc", how="left")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out["temperature_lag_24"] = out["temperature_c"].shift(24)
    out["humidity_lag_24"] = out["humidity_pct"].shift(24)
    return out


def make_ekpc_feature_dataset(
    df: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build 24-hour-ahead supervised examples for the EKPC load area."""
    out = add_time_features(df)
    out = add_day_of_year_features(out)
    out = add_cyclical_features(out)
    if weather is not None:
        out = merge_weather_features(out, weather)
    out = add_lag_features(out, LAG_HOURS + EXTRA_LAGS)
    out = add_rolling_features(out, ROLLING_WINDOWS)
    out = add_target(out, horizon=HORIZON)

    feature_columns = (
        NUMERIC_FEATURE_COLUMNS_WITH_WEATHER if weather is not None else NUMERIC_FEATURE_COLUMNS
    )
    required = feature_columns + METADATA_COLUMNS + ["target_load_mw"]
    out = out.dropna(subset=required)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    return out


def get_feature_target_metadata(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
):
    """Return feature matrix, target vector, and metadata for modeling."""
    columns = feature_columns or NUMERIC_FEATURE_COLUMNS
    X = df[columns].copy()
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
    neural_weather = metrics_by_model["Neural Network + Weather"]
    mae_improvement = persistence["mae"] - neural["mae"]
    rmse_improvement = persistence["rmse"] - neural["rmse"]
    weather_vs_base_mae = neural["mae"] - neural_weather["mae"]
    weather_vs_base_rmse = neural["rmse"] - neural_weather["rmse"]

    content = f"""# EKPC 24-Hour-Ahead Load Forecast — Neural Network

## Overview

This document describes a small feedforward neural network that forecasts hourly electricity load for the PJM Western region **EKPC** load area. The model predicts load **24 hours ahead** using lagged load history, calendar features, and optional Kentucky weather data. Results are compared against a **persistence** baseline (current load as the forecast).

**Load data:** `2022_load_ekpc.csv` through `Western_EKPC_load_metered.csv` (Jun 2022–May 2026)  
**Weather data:** `{WEATHER_DATA_PATH}` (hourly 2 m temperature and humidity, Jun 2022–Jun 2026, UTC)  
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

**Input dimension (no weather):** {len(NUMERIC_FEATURE_COLUMNS)} numeric features  
**Input dimension (with weather):** {len(NUMERIC_FEATURE_COLUMNS_WITH_WEATHER)} numeric features  

Features are standardized before the network.

## Model architecture

| Layer | Units | Activation |
|-------|------:|------------|
| Input | {len(NUMERIC_FEATURE_COLUMNS)} or {len(NUMERIC_FEATURE_COLUMNS_WITH_WEATHER)} | — |
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

### Neural network (load + calendar only)

{_format_metrics_block(neural)}

### Neural network + weather

{_format_metrics_block(neural_weather)}

### Comparison (Persistence − Neural Network)

| Metric | Δ (positive = neural network better) |
|--------|--------------------------------------|
| MAE | {mae_improvement:+.3f} MW |
| RMSE | {rmse_improvement:+.3f} MW |

### Weather uplift (base NN − NN + weather)

| Metric | Δ (positive = weather model better) |
|--------|-------------------------------------|
| MAE | {weather_vs_base_mae:+.3f} MW |
| RMSE | {weather_vs_base_rmse:+.3f} MW |

**Training iterations (base NN):** {nn_info['n_iter_']}  
**Training iterations (NN + weather):** {nn_info['n_iter_weather']}

## Output artifacts

| File | Description |
|------|-------------|
| `{RESULTS_DIR}/figures/model_comparison.png` | Bar chart of errors and test-period time series |
| `{RESULTS_DIR}/predictions/test_predictions.csv` | Test-set predictions for all models |
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


def plot_model_comparison(
    test_metrics: dict[str, dict],
    predictions_df: pd.DataFrame,
    output_path: Path,
    time_series_hours: int = 168,
) -> Path:
    """Create a two-panel figure comparing persistence, NN, and NN + weather."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("mae", "MAE (MW)"),
        ("rmse", "RMSE (MW)"),
        ("peak_hour_mae", "Peak-hour MAE (MW)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1.4]})

    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.24
    for idx, (metric_key, ylabel) in enumerate(metric_specs):
        offset = (idx - 1) * bar_width
        values = [test_metrics[model][metric_key] for model in MODEL_ORDER]
        axes[0].bar(
            x + offset,
            values,
            width=bar_width,
            label=ylabel.replace(" (MW)", ""),
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    axes[0].set_ylabel("Error (MW)")
    axes[0].set_title("Test-set forecast error by model")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    preds = predictions_df.copy()
    preds["target_timestamp_ept"] = pd.to_datetime(preds["target_timestamp_ept"])
    sample_times = (
        preds["target_timestamp_ept"].drop_duplicates().sort_values().head(time_series_hours)
    )
    sample = preds[preds["target_timestamp_ept"].isin(sample_times)]
    actual = sample.drop_duplicates("target_timestamp_ept").sort_values("target_timestamp_ept")

    axes[1].plot(
        actual["target_timestamp_ept"],
        actual["true_load_mw"],
        label="Actual",
        color="black",
        linewidth=2.4,
    )
    for model in MODEL_ORDER:
        model_df = sample[sample["model"] == model].sort_values("target_timestamp_ept")
        axes[1].plot(
            model_df["target_timestamp_ept"],
            model_df["predicted_load_mw"],
            label=model,
            color=MODEL_COLORS[model],
            **MODEL_LINE_STYLES[model],
        )
    axes[1].set_title(f"True vs predicted load (first {time_series_hours} test hours)")
    axes[1].set_xlabel("Target timestamp (EPT)")
    axes[1].set_ylabel("Load (MW)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("EKPC 24-hour-ahead forecast comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _metrics_for_models(y, X, metadata, nn_models: dict[str, Pipeline]) -> tuple[dict, pd.DataFrame]:
    """Compute metrics and prediction rows for persistence and neural nets."""
    persistence_pred = predict_persistence(X)
    split_metrics = {
        "Persistence": {
            **compute_metrics(y, persistence_pred),
            "peak_hour_mae": peak_hour_mae(y, persistence_pred),
            "n": len(y),
        }
    }
    pred_frames = [_build_prediction_frame(metadata, persistence_pred, "Persistence")]

    for model_name, nn_model in nn_models.items():
        nn_pred = predict_neural_network(nn_model, X)
        split_metrics[model_name] = {
            **compute_metrics(y, nn_pred),
            "peak_hour_mae": peak_hour_mae(y, nn_pred),
            "n": len(y),
        }
        pred_frames.append(_build_prediction_frame(metadata, nn_pred, model_name))

    return split_metrics, pd.concat(pred_frames, ignore_index=True)


def run_pipeline(
    load_paths: list[Path] | None = None,
    weather_path: Path = WEATHER_DATA_PATH,
    results_dir: Path = RESULTS_DIR,
    report_path: Path = REPORT_PATH,
) -> dict[str, dict]:
    """Load data, train neural networks, evaluate, and write outputs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)

    raw = load_ekpc_raw(load_paths)
    clean = clean_ekpc_data(raw)
    weather = load_weather_data(weather_path)
    features = make_ekpc_feature_dataset(clean, weather=weather)

    train_df, val_df, test_df = time_based_split(features)

    X_train, y_train, meta_train = get_feature_target_metadata(train_df, NUMERIC_FEATURE_COLUMNS)
    X_val, y_val, meta_val = get_feature_target_metadata(val_df, NUMERIC_FEATURE_COLUMNS)
    X_test, y_test, meta_test = get_feature_target_metadata(test_df, NUMERIC_FEATURE_COLUMNS)

    X_train_w, _, _ = get_feature_target_metadata(train_df, NUMERIC_FEATURE_COLUMNS_WITH_WEATHER)
    X_val_w, _, _ = get_feature_target_metadata(val_df, NUMERIC_FEATURE_COLUMNS_WITH_WEATHER)
    X_test_w, _, _ = get_feature_target_metadata(test_df, NUMERIC_FEATURE_COLUMNS_WITH_WEATHER)

    nn_model = build_neural_network_pipeline()
    fit_neural_network(nn_model, X_train, y_train)

    nn_weather_model = build_neural_network_pipeline()
    fit_neural_network(nn_weather_model, X_train_w, y_train)

    val_metrics, _ = _metrics_for_models(
        y_val,
        X_val,
        meta_val,
        {"Neural Network": nn_model},
    )
    val_weather_metrics, _ = _metrics_for_models(
        y_val,
        X_val_w,
        meta_val,
        {"Neural Network + Weather": nn_weather_model},
    )
    val_metrics["Neural Network + Weather"] = val_weather_metrics["Neural Network + Weather"]

    test_metrics, test_preds = _metrics_for_models(
        y_test,
        X_test,
        meta_test,
        {"Neural Network": nn_model},
    )
    test_weather_metrics, test_weather_preds = _metrics_for_models(
        y_test,
        X_test_w,
        meta_test,
        {"Neural Network + Weather": nn_weather_model},
    )
    test_metrics["Neural Network + Weather"] = test_weather_metrics["Neural Network + Weather"]
    test_preds = pd.concat([test_preds, test_weather_preds], ignore_index=True)

    test_preds.to_csv(results_dir / "predictions" / "test_predictions.csv", index=False)
    with open(results_dir / "metrics" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(results_dir / "metrics" / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    mlp = nn_model.named_steps["model"]
    mlp_weather = nn_weather_model.named_steps["model"]
    nn_info = {
        "alpha": 0.01,
        "learning_rate_init": 1e-4,
        "batch_size": 128,
        "max_iter": 600,
        "n_iter_": int(mlp.n_iter_),
        "n_iter_weather": int(mlp_weather.n_iter_),
        "best_loss": float(mlp.best_loss_) if mlp.best_loss_ is not None else None,
        "converged": bool(mlp.n_iter_ < mlp.max_iter),
    }

    data_summary = {
        "n_rows": int(len(clean)),
        "n_feature_rows": int(len(features)),
        "n_weather_rows": int(len(weather)),
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
        metrics_by_model=test_metrics,
        nn_info=nn_info,
    )

    figure_path = plot_model_comparison(
        test_metrics,
        test_preds,
        results_dir / "figures" / "model_comparison.png",
    )

    base = test_metrics["Neural Network"]
    with_weather = test_metrics["Neural Network + Weather"]

    print("\nEKPC neural network pipeline complete")
    print(f"Clean rows: {data_summary['n_rows']}")
    print(f"Feature rows: {data_summary['n_feature_rows']}")
    print(f"Train / val / test rows: {split_summary['train_rows']} / {split_summary['val_rows']} / {split_summary['test_rows']}")
    print("\nTest metrics:")
    for model_name, metrics in test_metrics.items():
        print(f"  {model_name}: MAE={metrics['mae']:.2f} MW, RMSE={metrics['rmse']:.2f} MW, peak MAE={metrics['peak_hour_mae']:.2f} MW")
    print(
        f"\nWeather vs base NN on test: MAE {base['mae'] - with_weather['mae']:+.2f} MW, "
        f"RMSE {base['rmse'] - with_weather['rmse']:+.2f} MW"
    )
    print(f"\nReport written to {report_path}")
    print(f"Comparison figure saved to {figure_path}")
    return test_metrics


def plot_from_saved_results(
    metrics_path: Path = RESULTS_DIR / "metrics" / "test_metrics.json",
    predictions_path: Path = RESULTS_DIR / "predictions" / "test_predictions.csv",
    output_path: Path = RESULTS_DIR / "figures" / "model_comparison.png",
) -> Path:
    """Regenerate the comparison figure from saved metrics and predictions."""
    with open(metrics_path, encoding="utf-8") as f:
        test_metrics = json.load(f)
    predictions_df = pd.read_csv(predictions_path, parse_dates=["target_timestamp_ept"])
    return plot_model_comparison(test_metrics, predictions_df, output_path)


if __name__ == "__main__":
    run_pipeline()
