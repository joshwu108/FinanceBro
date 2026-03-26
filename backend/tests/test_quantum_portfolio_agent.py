"""Tests for QuantumPortfolioAgent — TDD: tests written first.

Tests the full BaseAgent contract, classical vs quantum comparison,
efficient frontier generation, and integration with the existing pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from agents.quantum.quantum_portfolio_agent import QuantumPortfolioAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_asset_returns():
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
def five_asset_returns():
    """400 days of synthetic daily returns for 5 assets."""
    rng = np.random.default_rng(99)
    n_days = 400
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = ["A", "B", "C", "D", "E"]
    data = {t: rng.normal(0.0003, 0.01 + i * 0.003, n_days) for i, t in enumerate(tickers)}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def default_agent():
    return QuantumPortfolioAgent()


@pytest.fixture
def qaoa_agent():
    return QuantumPortfolioAgent(config={
        "methods": ["markowitz_cvxpy", "qaoa"],
        "qaoa_layers": 2,
        "weight_precision_bits": 3,
        "qaoa_maxiter": 200,
        "qaoa_seed": 42,
    })


# ===========================================================================
# 1. BaseAgent contract
# ===========================================================================

class TestBaseAgentContract:
    """Verify QuantumPortfolioAgent satisfies the BaseAgent interface."""

    def test_has_run_method(self, default_agent):
        assert hasattr(default_agent, "run") and callable(default_agent.run)

    def test_has_validate_method(self, default_agent):
        assert hasattr(default_agent, "validate") and callable(default_agent.validate)

    def test_has_log_metrics_method(self, default_agent):
        assert hasattr(default_agent, "log_metrics") and callable(default_agent.log_metrics)

    def test_has_input_schema(self, default_agent):
        schema = default_agent.input_schema
        assert isinstance(schema, dict)
        assert "returns" in schema

    def test_has_output_schema(self, default_agent):
        schema = default_agent.output_schema
        assert isinstance(schema, dict)
        assert "classical_weights" in schema or "weights" in schema


# ===========================================================================
# 2. Configuration
# ===========================================================================

class TestConfiguration:

    def test_default_config(self):
        agent = QuantumPortfolioAgent()
        assert agent._config["max_weight"] == 0.10
        assert "markowitz_cvxpy" in agent._config["methods"]

    def test_custom_config_override(self):
        agent = QuantumPortfolioAgent(config={"max_weight": 0.25, "qaoa_layers": 5})
        assert agent._config["max_weight"] == 0.25
        assert agent._config["qaoa_layers"] == 5

    def test_methods_configurable(self):
        agent = QuantumPortfolioAgent(config={"methods": ["qaoa"]})
        assert agent._config["methods"] == ["qaoa"]


# ===========================================================================
# 3. Run — classical only
# ===========================================================================

class TestRunClassical:
    """Test run() with classical methods only."""

    def test_run_returns_dict(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={"methods": ["markowitz_cvxpy"]})
        result = agent.run({"returns": three_asset_returns})
        assert isinstance(result, dict)

    def test_classical_weights_present(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={"methods": ["markowitz_cvxpy"]})
        result = agent.run({"returns": three_asset_returns})
        assert "classical_weights" in result
        w = result["classical_weights"]
        assert isinstance(w, np.ndarray)
        assert len(w) == 3

    def test_classical_weights_sum_to_one(self, three_asset_returns):
        # max_weight must be >= 1/n_assets for budget constraint to be feasible
        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "max_weight": 0.50,
        })
        result = agent.run({"returns": three_asset_returns})
        w = result["classical_weights"]
        assert np.isclose(np.sum(w), 1.0, atol=0.02)

    def test_classical_weights_respect_max_weight(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "max_weight": 0.50,
        })
        result = agent.run({"returns": three_asset_returns})
        w = result["classical_weights"]
        assert np.all(w <= 0.50 + 1e-6)

    def test_classical_weights_non_negative(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={"methods": ["markowitz_cvxpy"]})
        result = agent.run({"returns": three_asset_returns})
        w = result["classical_weights"]
        assert np.all(w >= -1e-6)

    def test_classical_objective_present(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={"methods": ["markowitz_cvxpy"]})
        result = agent.run({"returns": three_asset_returns})
        assert "classical_objective" in result
        assert result["classical_objective"] > 0  # variance is positive


# ===========================================================================
# 4. Run — QAOA
# ===========================================================================

class TestRunQAOA:
    """Test run() with QAOA method."""

    def test_qaoa_weights_present(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert "quantum_weights" in result
        w = result["quantum_weights"]
        assert isinstance(w, np.ndarray)
        assert len(w) == 3

    def test_qaoa_weights_non_negative(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        w = result["quantum_weights"]
        assert np.all(w >= -1e-6)

    def test_qaoa_weights_bounded(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={
            "methods": ["qaoa"],
            "max_weight": 0.50,
            "qaoa_layers": 2,
            "weight_precision_bits": 3,
            "qaoa_seed": 42,
        })
        result = agent.run({"returns": three_asset_returns})
        w = result["quantum_weights"]
        assert np.all(w <= 0.50 + 1e-6)

    def test_qaoa_runtime_recorded(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert "quantum_runtime_ms" in result
        assert result["quantum_runtime_ms"] > 0

    def test_qaoa_metadata_present(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert "quantum_metadata" in result
        meta = result["quantum_metadata"]
        assert "optimal_gamma" in meta
        assert "optimal_beta" in meta


# ===========================================================================
# 5. Comparison metrics
# ===========================================================================

class TestComparison:
    """Test classical vs quantum comparison output."""

    def test_comparison_metrics_present(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert "comparison" in result

    def test_comparison_has_required_fields(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        comp = result["comparison"]
        assert "weight_distance" in comp
        assert "runtime_ratio" in comp

    def test_weight_distance_is_non_negative(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert result["comparison"]["weight_distance"] >= 0.0


# ===========================================================================
# 6. Efficient frontier
# ===========================================================================

class TestEfficientFrontier:

    def test_frontier_generated(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "frontier_points": 5,
        })
        result = agent.run({"returns": three_asset_returns})
        assert "efficient_frontier" in result

    def test_frontier_has_risks_and_returns(self, three_asset_returns):
        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "frontier_points": 5,
        })
        result = agent.run({"returns": three_asset_returns})
        frontier = result["efficient_frontier"]
        assert "risks" in frontier
        assert "returns" in frontier
        assert len(frontier["risks"]) == 5
        assert len(frontier["returns"]) == 5


# ===========================================================================
# 7. Validate
# ===========================================================================

class TestValidate:

    def test_validate_passes_valid_output(self, three_asset_returns, qaoa_agent):
        result = qaoa_agent.run({"returns": three_asset_returns})
        assert qaoa_agent.validate({"returns": three_asset_returns}, result)

    def test_validate_rejects_missing_returns(self, qaoa_agent):
        with pytest.raises((ValueError, KeyError)):
            qaoa_agent.validate({}, {"classical_weights": np.array([0.5, 0.5])})


# ===========================================================================
# 8. Log metrics
# ===========================================================================

class TestLogMetrics:

    def test_log_metrics_does_not_raise(self, three_asset_returns, qaoa_agent):
        qaoa_agent.run({"returns": three_asset_returns})
        qaoa_agent.log_metrics()  # should not raise

    def test_metrics_stored_after_run(self, three_asset_returns, qaoa_agent):
        qaoa_agent.run({"returns": three_asset_returns})
        assert qaoa_agent._metrics  # should be non-empty


# ===========================================================================
# 9. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_two_assets(self):
        """Works with just 2 assets."""
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({
            "A": rng.normal(0.001, 0.01, 252),
            "B": rng.normal(0.002, 0.02, 252),
        }, index=pd.bdate_range("2024-01-01", periods=252))

        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy", "qaoa"],
            "max_weight": 0.80,
            "qaoa_layers": 2,
            "weight_precision_bits": 3,
            "qaoa_seed": 42,
        })
        result = agent.run({"returns": returns})
        assert "classical_weights" in result
        assert "quantum_weights" in result

    def test_single_asset_classical(self):
        """Single asset should get weight = 1.0."""
        returns = pd.DataFrame({
            "SPY": np.random.default_rng(0).normal(0.001, 0.01, 252),
        }, index=pd.bdate_range("2024-01-01", periods=252))

        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "max_weight": 1.0,
        })
        result = agent.run({"returns": returns})
        assert np.isclose(result["classical_weights"][0], 1.0, atol=0.01)
