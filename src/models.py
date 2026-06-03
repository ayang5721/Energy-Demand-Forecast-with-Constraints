"""Milestone forecasting models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLars, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import rmse


CATEGORICAL_COLUMNS = ["zone", "load_area"]
NUMERIC_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "load_mw",
    "load_lag_1",
    "load_lag_24",
    "load_lag_48",
    "rolling_mean_24",
    "rolling_std_24",
]


def get_preprocessor(categorical_cols, numeric_cols) -> ColumnTransformer:
    """Return a sklearn preprocessor for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )


def predict_persistence(X: pd.DataFrame) -> np.ndarray:
    """Return current load as the 24-hour-ahead persistence forecast."""
    return X["load_mw"].to_numpy()


def train_ols(X_train, y_train, categorical_cols, numeric_cols) -> Pipeline:
    """Train an ordinary least squares regression pipeline."""
    model = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor(categorical_cols, numeric_cols)),
            ("model", LinearRegression()),
        ]
    )
    return model.fit(X_train, y_train)


def train_ridge(X_train, y_train, alpha, categorical_cols, numeric_cols) -> Pipeline:
    """Train a Ridge regression pipeline."""
    model = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor(categorical_cols, numeric_cols)),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    return model.fit(X_train, y_train)


def train_lasso(X_train, y_train, alpha, categorical_cols, numeric_cols) -> Pipeline:
    """Train a Lasso regression pipeline."""
    model = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor(categorical_cols, numeric_cols)),
            ("model", LassoLars(alpha=alpha, max_iter=500)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X_train, y_train)


def tune_ridge(X_train, y_train, X_val, y_val, alpha_grid, categorical_cols, numeric_cols):
    """Tune Ridge alpha by validation RMSE and return the best fitted model."""
    rows = []
    best_model = None
    best_alpha = None
    best_rmse = np.inf
    for alpha in alpha_grid:
        model = train_ridge(X_train, y_train, alpha, categorical_cols, numeric_cols)
        pred = model.predict(X_val)
        val_rmse = rmse(y_val, pred)
        rows.append({"alpha": alpha, "validation_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
            best_model = model
    return best_model, best_alpha, pd.DataFrame(rows)


def tune_lasso(X_train, y_train, X_val, y_val, alpha_grid, categorical_cols, numeric_cols):
    """Tune Lasso alpha by validation RMSE and return the best fitted model."""
    rows = []
    best_model = None
    best_alpha = None
    best_rmse = np.inf
    for alpha in alpha_grid:
        model = train_lasso(X_train, y_train, alpha, categorical_cols, numeric_cols)
        pred = model.predict(X_val)
        val_rmse = rmse(y_val, pred)
        rows.append({"alpha": alpha, "validation_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
            best_model = model
    return best_model, best_alpha, pd.DataFrame(rows)
