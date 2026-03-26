"""QuantumMLAgent — Hybrid classical-quantum time series prediction.

Compares classical baselines (linear regression, rolling mean) against
a variational quantum regressor for predicting next-day returns.
Uses walk-forward evaluation: train on past, predict one step ahead.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from quantum.ml.variational_regressor import VariationalQuantumRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature preparation (no look-ahead)
# ---------------------------------------------------------------------------

def prepare_features(returns: np.ndarray, n_lags: int) -> tuple:
    """Create lagged feature matrix from 1-D return series.

    X[i] = [returns[i+n_lags-1], returns[i+n_lags-2], ..., returns[i]]
    y[i] = returns[i + n_lags]

    No look-ahead: every feature at index i uses only data <= i+n_lags-1.
    """
    n = len(returns)
    n_samples = n - n_lags
    X = np.zeros((n_samples, n_lags))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Lags: most recent first
        for lag in range(n_lags):
            X[i, lag] = returns[i + n_lags - 1 - lag]
        y[i] = returns[i + n_lags]

    return X, y


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def rolling_mean_predictor(
    X: np.ndarray, y: np.ndarray, window: int = 10
) -> np.ndarray:
    """Predict next return as rolling mean of recent returns."""
    preds = np.zeros(len(y))
    all_returns = np.concatenate([X[0, ::-1], y])  # reconstruct series order

    for i in range(len(y)):
        start = max(0, i + X.shape[1] - window)
        end = i + X.shape[1]
        preds[i] = np.mean(all_returns[start:end])

    return preds


def linear_regression_predictor(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Walk-forward linear regression: train on past, predict one ahead."""
    n = len(y)
    preds = np.zeros(n)
    min_train = max(X.shape[1] + 1, 20)

    for i in range(n):
        if i < min_train:
            preds[i] = np.mean(y[:max(i, 1)])
        else:
            X_train = X[:i]
            y_train = y[:i]
            # Solve normal equations: w = (X^T X)^-1 X^T y
            XtX = X_train.T @ X_train
            reg = 1e-6 * np.eye(X_train.shape[1])
            w = np.linalg.solve(XtX + reg, X_train.T @ y_train)
            preds[i] = float(X[i] @ w)

    return preds


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QuantumMLAgent(BaseAgent):
    """Hybrid classical-quantum ML agent for time series prediction."""

    DEFAULT_CONFIG = {
        "methods": ["linear", "rolling_mean", "vqr"],
        "n_lags": 5,
        "n_qubits": 4,
        "n_layers": 2,
        "maxiter": 100,
        "seed": 42,
        "rolling_window": 10,
        "train_ratio": 0.7,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "returns": "pd.DataFrame — daily returns",
            "target_column": "str — column name to predict",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "linear": "(if configured) dict with mse, predictions",
            "rolling_mean": "(if configured) dict with mse, predictions",
            "vqr": "(if configured) dict with mse, predictions",
            "comparison": "(if multiple methods) dict",
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        returns_df: pd.DataFrame = inputs["returns"]
        target_col = inputs.get("target_column", returns_df.columns[0])
        cfg = self._config
        methods = cfg["methods"]

        series = returns_df[target_col].values
        n_lags = cfg["n_lags"]
        X, y = prepare_features(series, n_lags)

        # Train/test split (walk-forward, no shuffling)
        split = int(len(y) * cfg.get("train_ratio", 0.7))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        result: Dict[str, Any] = {"n_samples": len(y), "n_lags": n_lags}

        if "rolling_mean" in methods:
            start = time.perf_counter()
            preds = rolling_mean_predictor(X, y, window=cfg.get("rolling_window", 10))
            test_preds = preds[split:]
            elapsed = (time.perf_counter() - start) * 1000
            mse = float(np.mean((test_preds - y_test) ** 2))
            result["rolling_mean"] = {
                "mse": mse,
                "runtime_ms": elapsed,
                "n_test": len(y_test),
            }

        if "linear" in methods:
            start = time.perf_counter()
            preds = linear_regression_predictor(X, y)
            test_preds = preds[split:]
            elapsed = (time.perf_counter() - start) * 1000
            mse = float(np.mean((test_preds - y_test) ** 2))
            result["linear"] = {
                "mse": mse,
                "runtime_ms": elapsed,
                "n_test": len(y_test),
            }

        if "vqr" in methods:
            start = time.perf_counter()
            vqr = VariationalQuantumRegressor(
                n_qubits=cfg.get("n_qubits", 4),
                n_layers=cfg.get("n_layers", 2),
                maxiter=cfg.get("maxiter", 100),
                seed=cfg.get("seed"),
            )
            vqr.fit(X_train, y_train)
            test_preds = vqr.predict(X_test)
            elapsed = (time.perf_counter() - start) * 1000
            mse = float(np.mean((test_preds - y_test) ** 2))
            result["vqr"] = {
                "mse": mse,
                "runtime_ms": elapsed,
                "n_test": len(y_test),
                "n_params": vqr.n_params,
            }

        # Comparison
        method_results = {
            m: result[m] for m in methods if m in result and "mse" in result.get(m, {})
        }
        if len(method_results) >= 2:
            best = min(method_results, key=lambda m: method_results[m]["mse"])
            result["comparison"] = {
                "best_method": best,
                "best_mse": method_results[best]["mse"],
                "all_mse": {m: method_results[m]["mse"] for m in method_results},
            }

        self._metrics = result
        return result

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        if "returns" not in inputs:
            raise ValueError("Missing required input: 'returns'")
        return True

    def log_metrics(self) -> None:
        if not self._metrics:
            logger.info("QuantumMLAgent: no metrics (run() not called)")
            return
        logger.info("QuantumMLAgent metrics: %s", {
            k: v for k, v in self._metrics.items()
            if k not in ("predictions",)
        })
