"""Tests for Quantum ML Agent — TDD: tests written first.

Covers quantum feature maps, variational quantum regressor,
classical baselines, and the QuantumMLAgent BaseAgent contract.
"""

import numpy as np
import pandas as pd
import pytest

from quantum.ml.feature_map import angle_encode, amplitude_encode, QuantumFeatureMap
from quantum.ml.variational_regressor import VariationalQuantumRegressor
from agents.quantum.quantum_ml_agent import (
    QuantumMLAgent,
    prepare_features,
    rolling_mean_predictor,
    linear_regression_predictor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns():
    """252 days of synthetic daily returns for 1 asset."""
    rng = np.random.default_rng(42)
    n_days = 252
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    returns = rng.normal(0.0005, 0.015, n_days)
    return pd.DataFrame({"SPY": returns}, index=dates)


@pytest.fixture
def multi_asset_returns():
    """252 days of synthetic daily returns for 3 assets."""
    rng = np.random.default_rng(42)
    n_days = 252
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {
        "AAPL": rng.normal(0.0005, 0.015, n_days),
        "MSFT": rng.normal(0.0004, 0.012, n_days),
        "GOOG": rng.normal(0.0003, 0.018, n_days),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def simple_features():
    """Small feature matrix for quantum encoding tests."""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])


# ===========================================================================
# 1. Quantum Feature Maps
# ===========================================================================

class TestAngleEncoding:

    def test_output_shape(self, simple_features):
        """Angle encoding produces 2^n_qubits statevector per sample."""
        sv = angle_encode(simple_features[0], n_qubits=3)
        assert sv.shape == (2 ** 3,)

    def test_statevector_normalized(self, simple_features):
        """Output statevector has unit norm."""
        sv = angle_encode(simple_features[0], n_qubits=3)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_different_inputs_different_states(self, simple_features):
        """Different feature vectors produce different quantum states."""
        sv1 = angle_encode(simple_features[0], n_qubits=3)
        sv2 = angle_encode(simple_features[1], n_qubits=3)
        assert not np.allclose(sv1, sv2)

    def test_deterministic(self, simple_features):
        """Same input always gives same output."""
        sv1 = angle_encode(simple_features[0], n_qubits=3)
        sv2 = angle_encode(simple_features[0], n_qubits=3)
        assert np.allclose(sv1, sv2)


class TestAmplitudeEncoding:

    def test_output_shape(self):
        """Amplitude encoding uses ceil(log2(n_features)) qubits."""
        features = np.array([0.1, 0.2, 0.3, 0.4])
        sv = amplitude_encode(features)
        assert sv.shape == (4,)  # 2^2 = 4

    def test_statevector_normalized(self):
        sv = amplitude_encode(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_zero_padded(self):
        """Non-power-of-2 features get zero-padded."""
        features = np.array([1.0, 2.0, 3.0])
        sv = amplitude_encode(features)
        assert sv.shape == (4,)  # padded to 2^2


class TestQuantumFeatureMap:

    def test_batch_encode(self, simple_features):
        """Feature map encodes a batch of samples."""
        fmap = QuantumFeatureMap(n_qubits=3, encoding="angle")
        encoded = fmap.encode_batch(simple_features)
        assert encoded.shape == (3, 2 ** 3)

    def test_kernel_matrix_shape(self, simple_features):
        """Kernel matrix is n_samples x n_samples."""
        fmap = QuantumFeatureMap(n_qubits=3, encoding="angle")
        K = fmap.kernel_matrix(simple_features)
        assert K.shape == (3, 3)

    def test_kernel_matrix_symmetric(self, simple_features):
        fmap = QuantumFeatureMap(n_qubits=3, encoding="angle")
        K = fmap.kernel_matrix(simple_features)
        assert np.allclose(K, K.T)

    def test_kernel_diagonal_is_one(self, simple_features):
        """K(x,x) = |<phi(x)|phi(x)>|^2 = 1."""
        fmap = QuantumFeatureMap(n_qubits=3, encoding="angle")
        K = fmap.kernel_matrix(simple_features)
        assert np.allclose(np.diag(K), 1.0, atol=1e-10)


# ===========================================================================
# 2. Variational Quantum Regressor
# ===========================================================================

class TestVariationalQuantumRegressor:

    def test_fit_does_not_raise(self):
        """Fitting on small data should complete."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 3))
        y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.1, 20)

        vqr = VariationalQuantumRegressor(n_qubits=3, n_layers=2, maxiter=50, seed=42)
        vqr.fit(X, y)

    def test_predict_shape(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (20, 3))
        y_train = X_train @ np.array([1.0, -0.5, 0.3])
        X_test = rng.normal(0, 1, (5, 3))

        vqr = VariationalQuantumRegressor(n_qubits=3, n_layers=2, maxiter=50, seed=42)
        vqr.fit(X_train, y_train)
        preds = vqr.predict(X_test)
        assert preds.shape == (5,)

    def test_predictions_are_finite(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 2))
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3

        vqr = VariationalQuantumRegressor(n_qubits=2, n_layers=2, maxiter=50, seed=42)
        vqr.fit(X, y)
        preds = vqr.predict(X[:5])
        assert np.all(np.isfinite(preds))

    def test_has_parameters(self):
        vqr = VariationalQuantumRegressor(n_qubits=3, n_layers=2, seed=42)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 3))
        y = rng.normal(0, 1, 20)
        vqr.fit(X, y)
        assert vqr.parameters is not None
        assert len(vqr.parameters) > 0


# ===========================================================================
# 3. Classical baselines
# ===========================================================================

class TestClassicalBaselines:

    def test_rolling_mean_returns_array(self, synthetic_returns):
        X, y = prepare_features(synthetic_returns["SPY"].values, n_lags=5)
        preds = rolling_mean_predictor(X, y, window=10)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)

    def test_linear_regression_returns_array(self, synthetic_returns):
        X, y = prepare_features(synthetic_returns["SPY"].values, n_lags=5)
        preds = linear_regression_predictor(X, y)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)

    def test_linear_regression_predictions_finite(self, synthetic_returns):
        X, y = prepare_features(synthetic_returns["SPY"].values, n_lags=5)
        preds = linear_regression_predictor(X, y)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# 4. Feature preparation
# ===========================================================================

class TestPrepareFeatures:

    def test_output_shapes(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        X, y = prepare_features(returns, n_lags=5)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 5  # n_lags features

    def test_no_look_ahead(self):
        """Features at index i should only use data up to index i."""
        returns = np.arange(20, dtype=float)
        X, y = prepare_features(returns, n_lags=3)
        # y[0] should be returns[3] (first target after 3 lags)
        # X[0] should be [returns[2], returns[1], returns[0]]
        assert y[0] == returns[3]
        assert X[0, 0] == returns[2]  # most recent lag


# ===========================================================================
# 5. QuantumMLAgent (BaseAgent contract)
# ===========================================================================

class TestQuantumMLAgent:

    def test_has_base_agent_contract(self):
        agent = QuantumMLAgent()
        assert hasattr(agent, "run") and callable(agent.run)
        assert hasattr(agent, "validate") and callable(agent.validate)
        assert hasattr(agent, "log_metrics") and callable(agent.log_metrics)
        assert isinstance(agent.input_schema, dict)
        assert isinstance(agent.output_schema, dict)

    def test_run_returns_dict(self, synthetic_returns):
        agent = QuantumMLAgent(config={
            "n_lags": 5,
            "n_qubits": 3,
            "n_layers": 1,
            "maxiter": 20,
            "seed": 42,
        })
        result = agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        assert isinstance(result, dict)

    def test_run_contains_classical_baseline(self, synthetic_returns):
        agent = QuantumMLAgent(config={
            "methods": ["linear"],
            "n_lags": 5,
            "seed": 42,
        })
        result = agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        assert "linear" in result
        assert "mse" in result["linear"]

    def test_run_contains_quantum_predictions(self, synthetic_returns):
        agent = QuantumMLAgent(config={
            "methods": ["vqr"],
            "n_lags": 3,
            "n_qubits": 3,
            "n_layers": 1,
            "maxiter": 20,
            "seed": 42,
        })
        result = agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        assert "vqr" in result
        assert "mse" in result["vqr"]

    def test_run_comparison_mode(self, synthetic_returns):
        agent = QuantumMLAgent(config={
            "methods": ["linear", "vqr"],
            "n_lags": 3,
            "n_qubits": 3,
            "n_layers": 1,
            "maxiter": 20,
            "seed": 42,
        })
        result = agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        assert "comparison" in result

    def test_validate_passes(self, synthetic_returns):
        agent = QuantumMLAgent(config={"methods": ["linear"], "n_lags": 5})
        result = agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        assert agent.validate(
            {"returns": synthetic_returns, "target_column": "SPY"}, result
        )

    def test_validate_rejects_missing_returns(self):
        agent = QuantumMLAgent()
        with pytest.raises((ValueError, KeyError)):
            agent.validate({}, {})

    def test_log_metrics_after_run(self, synthetic_returns):
        agent = QuantumMLAgent(config={"methods": ["linear"], "n_lags": 5})
        agent.run({"returns": synthetic_returns, "target_column": "SPY"})
        agent.log_metrics()  # should not raise
