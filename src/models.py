"""Forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    """Explicit NumPy residual MLP anchored to the 24-hour persistence forecast."""

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
        validation_fraction: float = 0.10,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
    ):
        self.preprocessor = ModelPreprocessor(categorical_cols, numeric_cols)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.tol = float(tol)
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.residual_mean_ = 0.0
        self.residual_scale_ = 1.0
        self.n_iter_ = 0
        self.converged_ = False
        self.loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    @staticmethod
    def _relu_grad(values: np.ndarray) -> np.ndarray:
        return (values > 0.0).astype(float)

    def _initialize_parameters(self, n_features: int, rng: np.random.Generator) -> None:
        layer_sizes = [n_features, *self.hidden_layer_sizes, 1]
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights_.append(rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []
        current = X
        for idx, (weights, bias) in enumerate(zip(self.weights_, self.biases_)):
            z = current @ weights + bias
            pre_activations.append(z)
            if idx == len(self.weights_) - 1:
                current = z
            else:
                current = self._relu(z)
            activations.append(current)
        return activations, pre_activations

    def _predict_scaled_residual(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1].ravel()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self._predict_scaled_residual(X)
        mse_loss = 0.5 * float(np.mean((pred - y) ** 2))
        l2_loss = 0.5 * self.alpha * sum(float(np.sum(weights * weights)) for weights in self.weights_) / len(X)
        return mse_loss + l2_loss

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y_batch: np.ndarray,
        n_train: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_size = len(y_batch)
        delta = (activations[-1].ravel() - y_batch)[:, None] / batch_size
        weight_grads = [np.zeros_like(weights) for weights in self.weights_]
        bias_grads = [np.zeros_like(bias) for bias in self.biases_]

        for idx in range(len(self.weights_) - 1, -1, -1):
            weight_grads[idx] = activations[idx].T @ delta + (self.alpha / n_train) * self.weights_[idx]
            bias_grads[idx] = delta.sum(axis=0)
            if idx > 0:
                delta = (delta @ self.weights_[idx].T) * self._relu_grad(pre_activations[idx - 1])
        return weight_grads, bias_grads

    def _fit_mlp(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n_rows, n_features = X_train.shape
        n_val = int(n_rows * self.validation_fraction)
        if n_val > 0:
            X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        else:
            X_fit, y_fit = X_train, y_train
            X_val, y_val = None, None

        self._initialize_parameters(n_features, rng)
        first_moments_w = [np.zeros_like(weights) for weights in self.weights_]
        second_moments_w = [np.zeros_like(weights) for weights in self.weights_]
        first_moments_b = [np.zeros_like(bias) for bias in self.biases_]
        second_moments_b = [np.zeros_like(bias) for bias in self.biases_]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        step = 0
        best_validation_loss = np.inf
        best_weights = [weights.copy() for weights in self.weights_]
        best_biases = [bias.copy() for bias in self.biases_]
        epochs_without_improvement = 0

        for epoch in range(1, self.max_iter + 1):
            for start in range(0, len(X_fit), self.batch_size):
                indices = rng.permutation(len(X_fit)) if start == 0 else indices
                batch_indices = indices[start : start + self.batch_size]
                X_batch = X_fit[batch_indices]
                y_batch = y_fit[batch_indices]
                activations, pre_activations = self._forward(X_batch)
                weight_grads, bias_grads = self._backward(
                    activations,
                    pre_activations,
                    y_batch,
                    n_train=len(X_fit),
                )

                step += 1
                for idx in range(len(self.weights_)):
                    first_moments_w[idx] = beta1 * first_moments_w[idx] + (1.0 - beta1) * weight_grads[idx]
                    second_moments_w[idx] = (
                        beta2 * second_moments_w[idx] + (1.0 - beta2) * (weight_grads[idx] ** 2)
                    )
                    first_moments_b[idx] = beta1 * first_moments_b[idx] + (1.0 - beta1) * bias_grads[idx]
                    second_moments_b[idx] = (
                        beta2 * second_moments_b[idx] + (1.0 - beta2) * (bias_grads[idx] ** 2)
                    )

                    corrected_first_w = first_moments_w[idx] / (1.0 - beta1**step)
                    corrected_second_w = second_moments_w[idx] / (1.0 - beta2**step)
                    corrected_first_b = first_moments_b[idx] / (1.0 - beta1**step)
                    corrected_second_b = second_moments_b[idx] / (1.0 - beta2**step)

                    self.weights_[idx] -= (
                        self.learning_rate_init * corrected_first_w / (np.sqrt(corrected_second_w) + epsilon)
                    )
                    self.biases_[idx] -= (
                        self.learning_rate_init * corrected_first_b / (np.sqrt(corrected_second_b) + epsilon)
                    )

            self.n_iter_ = epoch
            self.loss_curve_.append(self._loss(X_fit, y_fit))
            validation_loss = self._loss(X_val, y_val) if X_val is not None else self.loss_curve_[-1]
            self.validation_loss_curve_.append(validation_loss)

            if validation_loss < best_validation_loss - self.tol:
                best_validation_loss = validation_loss
                best_weights = [weights.copy() for weights in self.weights_]
                best_biases = [bias.copy() for bias in self.biases_]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.n_iter_no_change:
                    self.converged_ = True
                    break

        self.weights_ = best_weights
        self.biases_ = best_biases

    def fit(self, X_train: pd.DataFrame, y_train):
        X = self.preprocessor.fit_transform(X_train)
        y = np.asarray(y_train, dtype=float)
        residual = y - X_train["load_mw"].to_numpy(dtype=float)
        self.residual_mean_ = float(np.mean(residual))
        self.residual_scale_ = float(np.std(residual))
        if self.residual_scale_ == 0.0:
            self.residual_scale_ = 1.0
        scaled_residual = (residual - self.residual_mean_) / self.residual_scale_
        self._fit_mlp(X, scaled_residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        scaled_residual_pred = self._predict_scaled_residual(transformed)
        residual_pred = scaled_residual_pred * self.residual_scale_ + self.residual_mean_
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
