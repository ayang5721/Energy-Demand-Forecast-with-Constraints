"""Milestone forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

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
WEATHER_NUMERIC_COLUMNS = NUMERIC_COLUMNS + [
    "temperature_c",
    "humidity_pct",
    "temperature_lag_24",
    "humidity_lag_24",
]


def _soft_threshold(value: float, penalty: float) -> float:
    """Return the Lasso soft-thresholded coefficient update."""
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


class ModelPreprocessor:
    """One-hot encode categorical columns and standardize numeric columns."""

    def __init__(self, categorical_cols, numeric_cols):
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.categories_ = {}
        self.numeric_mean_ = None
        self.numeric_scale_ = None

    def fit(self, X: pd.DataFrame):
        for col in self.categorical_cols:
            self.categories_[col] = sorted(X[col].astype(str).dropna().unique().tolist())

        numeric = X[self.numeric_cols].astype(float)
        self.numeric_mean_ = numeric.mean()
        self.numeric_scale_ = numeric.std(ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        columns = []
        for col in self.categorical_cols:
            values = X[col].astype(str).to_numpy()
            for category in self.categories_[col]:
                columns.append((values == category).astype(float))

        numeric = X[self.numeric_cols].astype(float)
        scaled_numeric = ((numeric - self.numeric_mean_) / self.numeric_scale_).to_numpy()
        columns.extend(scaled_numeric[:, idx] for idx in range(scaled_numeric.shape[1]))
        return np.column_stack(columns)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearModel:
    """Linear regression model with explicit OLS, Ridge, and Lasso training."""

    def __init__(
        self,
        model_type: str,
        categorical_cols,
        numeric_cols,
        alpha: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-4,
    ):
        self.model_type = model_type
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.preprocessor = ModelPreprocessor(categorical_cols, numeric_cols)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_iter_ = 0
        self.converged_ = True

    def fit(self, X_train: pd.DataFrame, y_train):
        X = self.preprocessor.fit_transform(X_train)
        y = np.asarray(y_train, dtype=float)

        if self.model_type == "ols":
            self._fit_ols(X, y)
        elif self.model_type == "ridge":
            self._fit_ridge(X, y)
        elif self.model_type == "lasso":
            self._fit_lasso(X, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        return self.intercept_ + transformed @ self.coef_

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        design = np.column_stack([np.ones(X.shape[0]), X])
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> None:
        design = np.column_stack([np.ones(X.shape[0]), X])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        left = design.T @ design + self.alpha * penalty
        right = design.T @ y
        params = np.linalg.solve(left, right)
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]

    def _fit_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        n_rows, n_features = X.shape
        coef = np.zeros(n_features)
        intercept = float(np.mean(y))
        prediction = np.full(n_rows, intercept)
        feature_norms = np.mean(X * X, axis=0)
        feature_norms = np.where(feature_norms == 0.0, 1.0, feature_norms)

        self.converged_ = False
        for iteration in range(1, self.max_iter + 1):
            old_intercept = intercept
            old_coef = coef.copy()

            intercept = float(np.mean(y - (prediction - intercept)))
            prediction += intercept - old_intercept

            for idx in range(n_features):
                prediction_without_feature = prediction - X[:, idx] * coef[idx]
                partial_residual = y - prediction_without_feature
                rho = float(np.mean(X[:, idx] * partial_residual))
                new_coef = _soft_threshold(rho, self.alpha) / feature_norms[idx]
                prediction = prediction_without_feature + X[:, idx] * new_coef
                coef[idx] = new_coef

            max_change = max(
                abs(intercept - old_intercept),
                float(np.max(np.abs(coef - old_coef))) if n_features else 0.0,
            )
            if max_change < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.intercept_ = intercept
        self.coef_ = coef


class ResidualNeuralNetworkModel:
    """Residual neural network anchored to the 24-hour persistence forecast."""

    def __init__(
        self,
        categorical_cols,
        numeric_cols,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        alpha: float = 0.01,
        learning_rate_init: float = 0.0001,
        batch_size: int = 128,
        max_iter: int = 600,
        random_state: int = 42,
    ):
        self.preprocessor = ModelPreprocessor(categorical_cols, numeric_cols)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=10,
            random_state=random_state,
        )

    def fit(self, X_train: pd.DataFrame, y_train):
        X = self.preprocessor.fit_transform(X_train)
        y = np.asarray(y_train, dtype=float)
        residual = y - X_train["load_mw"].to_numpy(dtype=float)
        self.model.fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        residual_pred = self.model.predict(transformed)
        return X["load_mw"].to_numpy(dtype=float) + residual_pred


def get_preprocessor(categorical_cols, numeric_cols) -> ModelPreprocessor:
    """Return the project preprocessor."""
    return ModelPreprocessor(categorical_cols, numeric_cols)


def predict_persistence(X: pd.DataFrame) -> np.ndarray:
    """Return current load as the 24-hour-ahead persistence forecast."""
    return X["load_mw"].to_numpy()


def train_ols(X_train, y_train, categorical_cols, numeric_cols) -> LinearModel:
    """Train ordinary least squares with the normal equation."""
    model = LinearModel("ols", categorical_cols, numeric_cols)
    return model.fit(X_train, y_train)


def train_ridge(X_train, y_train, alpha, categorical_cols, numeric_cols) -> LinearModel:
    """Train Ridge regression with explicit L2-regularized least squares."""
    model = LinearModel("ridge", categorical_cols, numeric_cols, alpha=alpha)
    return model.fit(X_train, y_train)


def train_lasso(X_train, y_train, alpha, categorical_cols, numeric_cols) -> LinearModel:
    """Train Lasso regression with explicit coordinate descent."""
    model = LinearModel("lasso", categorical_cols, numeric_cols, alpha=alpha, max_iter=500)
    return model.fit(X_train, y_train)


def train_neural_network(X_train, y_train, categorical_cols, numeric_cols) -> ResidualNeuralNetworkModel:
    """Train the residual MLP neural network."""
    model = ResidualNeuralNetworkModel(categorical_cols, numeric_cols)
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
