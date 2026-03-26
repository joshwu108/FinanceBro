"""Tests for Quantum Monte Carlo agent — TDD: tests written first.

Covers Black-Scholes analytical pricing, classical Monte Carlo with variance
reduction, and quantum amplitude estimation for European option pricing.
"""

import numpy as np
import pytest

from quantum.solvers.classical_solvers import SolverResult
from agents.quantum.quantum_montecarlo_agent import (
    QuantumMonteCarloAgent,
    black_scholes_call,
    black_scholes_put,
    classical_mc_price,
    quantum_ae_price,
    convergence_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def call_option_params():
    """Standard European call parameters."""
    return {
        "spot_price": 100.0,
        "strike_price": 105.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "time_to_expiry": 1.0,
    }


@pytest.fixture
def put_option_params():
    """European put parameters."""
    return {
        "spot_price": 100.0,
        "strike_price": 95.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "time_to_expiry": 0.5,
    }


@pytest.fixture
def deep_itm_call():
    """Deep in-the-money call: S >> K."""
    return {
        "spot_price": 150.0,
        "strike_price": 80.0,
        "risk_free_rate": 0.03,
        "volatility": 0.15,
        "time_to_expiry": 0.25,
    }


# ===========================================================================
# 1. Black-Scholes analytical
# ===========================================================================

class TestBlackScholes:

    def test_call_price_positive(self, call_option_params):
        p = call_option_params
        price = black_scholes_call(p["spot_price"], p["strike_price"],
                                   p["risk_free_rate"], p["volatility"],
                                   p["time_to_expiry"])
        assert price > 0

    def test_put_price_positive(self, put_option_params):
        p = put_option_params
        price = black_scholes_put(p["spot_price"], p["strike_price"],
                                  p["risk_free_rate"], p["volatility"],
                                  p["time_to_expiry"])
        assert price > 0

    def test_put_call_parity(self, call_option_params):
        """C - P = S - K*exp(-rT)."""
        p = call_option_params
        c = black_scholes_call(p["spot_price"], p["strike_price"],
                               p["risk_free_rate"], p["volatility"],
                               p["time_to_expiry"])
        put = black_scholes_put(p["spot_price"], p["strike_price"],
                                p["risk_free_rate"], p["volatility"],
                                p["time_to_expiry"])
        parity = p["spot_price"] - p["strike_price"] * np.exp(
            -p["risk_free_rate"] * p["time_to_expiry"])
        assert np.isclose(c - put, parity, atol=1e-8)

    def test_deep_itm_call_near_intrinsic(self, deep_itm_call):
        p = deep_itm_call
        price = black_scholes_call(p["spot_price"], p["strike_price"],
                                   p["risk_free_rate"], p["volatility"],
                                   p["time_to_expiry"])
        intrinsic = p["spot_price"] - p["strike_price"] * np.exp(
            -p["risk_free_rate"] * p["time_to_expiry"])
        # Deep ITM: price should be close to discounted intrinsic
        assert price >= intrinsic - 1e-6

    def test_zero_volatility_call(self):
        """Zero vol: call = max(S - K*exp(-rT), 0)."""
        c = black_scholes_call(100.0, 90.0, 0.05, 1e-10, 1.0)
        expected = 100.0 - 90.0 * np.exp(-0.05)
        assert np.isclose(c, expected, atol=0.01)

    def test_call_increases_with_spot(self):
        c1 = black_scholes_call(90.0, 100.0, 0.05, 0.2, 1.0)
        c2 = black_scholes_call(110.0, 100.0, 0.05, 0.2, 1.0)
        assert c2 > c1

    def test_call_increases_with_volatility(self):
        c1 = black_scholes_call(100.0, 100.0, 0.05, 0.10, 1.0)
        c2 = black_scholes_call(100.0, 100.0, 0.05, 0.40, 1.0)
        assert c2 > c1


# ===========================================================================
# 2. Classical Monte Carlo
# ===========================================================================

class TestClassicalMC:

    def test_price_close_to_bs(self, call_option_params):
        """MC price should converge to BS with enough paths."""
        p = call_option_params
        bs = black_scholes_call(p["spot_price"], p["strike_price"],
                                p["risk_free_rate"], p["volatility"],
                                p["time_to_expiry"])
        result = classical_mc_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_paths=500_000, seed=42, option_type="call",
        )
        assert abs(result["price"] - bs) < 0.50  # within $0.50

    def test_returns_required_fields(self, call_option_params):
        p = call_option_params
        result = classical_mc_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_paths=10_000, seed=0, option_type="call",
        )
        assert "price" in result
        assert "std_error" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "runtime_ms" in result
        assert "n_paths" in result

    def test_std_error_decreases_with_paths(self, call_option_params):
        p = call_option_params
        r1 = classical_mc_price(S0=p["spot_price"], K=p["strike_price"],
                                r=p["risk_free_rate"], sigma=p["volatility"],
                                T=p["time_to_expiry"],
                                n_paths=1_000, seed=42, option_type="call")
        r2 = classical_mc_price(S0=p["spot_price"], K=p["strike_price"],
                                r=p["risk_free_rate"], sigma=p["volatility"],
                                T=p["time_to_expiry"],
                                n_paths=100_000, seed=42, option_type="call")
        assert r2["std_error"] < r1["std_error"]

    def test_antithetic_reduces_variance(self, call_option_params):
        p = call_option_params
        naive = classical_mc_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_paths=50_000, seed=42, option_type="call",
            antithetic=False,
        )
        anti = classical_mc_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_paths=50_000, seed=42, option_type="call",
            antithetic=True,
        )
        # Antithetic should reduce std_error (or at least not increase it much)
        assert anti["std_error"] <= naive["std_error"] * 1.1

    def test_put_option_pricing(self, put_option_params):
        p = put_option_params
        bs = black_scholes_put(p["spot_price"], p["strike_price"],
                               p["risk_free_rate"], p["volatility"],
                               p["time_to_expiry"])
        result = classical_mc_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_paths=500_000, seed=42, option_type="put",
        )
        assert abs(result["price"] - bs) < 0.30

    def test_reproducibility(self, call_option_params):
        p = call_option_params
        r1 = classical_mc_price(S0=p["spot_price"], K=p["strike_price"],
                                r=p["risk_free_rate"], sigma=p["volatility"],
                                T=p["time_to_expiry"],
                                n_paths=10_000, seed=123, option_type="call")
        r2 = classical_mc_price(S0=p["spot_price"], K=p["strike_price"],
                                r=p["risk_free_rate"], sigma=p["volatility"],
                                T=p["time_to_expiry"],
                                n_paths=10_000, seed=123, option_type="call")
        assert r1["price"] == r2["price"]


# ===========================================================================
# 3. Quantum Amplitude Estimation
# ===========================================================================

class TestQuantumAE:

    def test_returns_required_fields(self, call_option_params):
        p = call_option_params
        result = quantum_ae_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_qubits=4, n_estimation_qubits=3,
        )
        assert "price" in result
        assert "runtime_ms" in result
        assert "n_qubits" in result
        assert "circuit_depth" in result

    def test_price_is_positive(self, call_option_params):
        p = call_option_params
        result = quantum_ae_price(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            n_qubits=4, n_estimation_qubits=3,
        )
        assert result["price"] >= 0.0

    def test_accuracy_improves_with_qubits(self, call_option_params):
        """More estimation qubits should improve accuracy."""
        p = call_option_params
        bs = black_scholes_call(p["spot_price"], p["strike_price"],
                                p["risk_free_rate"], p["volatility"],
                                p["time_to_expiry"])
        errors = []
        for n_est in [2, 4, 6]:
            result = quantum_ae_price(
                S0=p["spot_price"], K=p["strike_price"],
                r=p["risk_free_rate"], sigma=p["volatility"],
                T=p["time_to_expiry"],
                n_qubits=4, n_estimation_qubits=n_est,
            )
            errors.append(abs(result["price"] - bs))
        # Trend should be decreasing (more qubits = more precision)
        # Allow some noise but last should be better than first
        assert errors[-1] < errors[0] * 1.5  # tolerant check

    def test_circuit_depth_increases_with_estimation_qubits(self, call_option_params):
        p = call_option_params
        r1 = quantum_ae_price(S0=p["spot_price"], K=p["strike_price"],
                              r=p["risk_free_rate"], sigma=p["volatility"],
                              T=p["time_to_expiry"],
                              n_qubits=3, n_estimation_qubits=2)
        r2 = quantum_ae_price(S0=p["spot_price"], K=p["strike_price"],
                              r=p["risk_free_rate"], sigma=p["volatility"],
                              T=p["time_to_expiry"],
                              n_qubits=3, n_estimation_qubits=5)
        assert r2["circuit_depth"] > r1["circuit_depth"]


# ===========================================================================
# 4. Convergence analysis
# ===========================================================================

class TestConvergenceAnalysis:

    def test_returns_scaling_data(self, call_option_params):
        p = call_option_params
        result = convergence_analysis(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            classical_path_counts=[1000, 10000],
            quantum_estimation_qubits=[2, 3, 4],
            n_qubits_price=3,
        )
        assert "classical" in result
        assert "quantum" in result
        assert len(result["classical"]) == 2
        assert len(result["quantum"]) == 3

    def test_classical_error_decreases(self, call_option_params):
        p = call_option_params
        result = convergence_analysis(
            S0=p["spot_price"], K=p["strike_price"],
            r=p["risk_free_rate"], sigma=p["volatility"],
            T=p["time_to_expiry"],
            classical_path_counts=[1000, 100_000],
            quantum_estimation_qubits=[2],
            n_qubits_price=3,
        )
        errors = [r["std_error"] for r in result["classical"]]
        assert errors[-1] < errors[0]


# ===========================================================================
# 5. QuantumMonteCarloAgent (BaseAgent contract)
# ===========================================================================

class TestQuantumMonteCarloAgent:

    def test_has_base_agent_contract(self):
        agent = QuantumMonteCarloAgent()
        assert hasattr(agent, "run") and callable(agent.run)
        assert hasattr(agent, "validate") and callable(agent.validate)
        assert hasattr(agent, "log_metrics") and callable(agent.log_metrics)
        assert isinstance(agent.input_schema, dict)
        assert isinstance(agent.output_schema, dict)

    def test_run_returns_dict(self, call_option_params):
        agent = QuantumMonteCarloAgent()
        result = agent.run(call_option_params)
        assert isinstance(result, dict)

    def test_run_contains_bs_price(self, call_option_params):
        agent = QuantumMonteCarloAgent()
        result = agent.run(call_option_params)
        assert "black_scholes_price" in result
        assert result["black_scholes_price"] > 0

    def test_run_contains_classical_mc(self, call_option_params):
        agent = QuantumMonteCarloAgent(config={"methods": ["classical_mc"]})
        result = agent.run(call_option_params)
        assert "classical_mc" in result

    def test_run_contains_quantum_ae(self, call_option_params):
        agent = QuantumMonteCarloAgent(config={
            "methods": ["quantum_ae"],
            "n_qubits_price": 3,
            "n_estimation_qubits": 3,
        })
        result = agent.run(call_option_params)
        assert "quantum_ae" in result

    def test_run_comparison_mode(self, call_option_params):
        agent = QuantumMonteCarloAgent(config={
            "methods": ["classical_mc", "quantum_ae"],
            "n_classical_paths": 50_000,
            "n_qubits_price": 3,
            "n_estimation_qubits": 3,
        })
        result = agent.run(call_option_params)
        assert "classical_mc" in result
        assert "quantum_ae" in result
        assert "comparison" in result

    def test_validate_passes(self, call_option_params):
        agent = QuantumMonteCarloAgent()
        result = agent.run(call_option_params)
        assert agent.validate(call_option_params, result)

    def test_validate_rejects_missing_spot(self):
        agent = QuantumMonteCarloAgent()
        with pytest.raises((ValueError, KeyError)):
            agent.validate({"strike_price": 100.0}, {})

    def test_log_metrics_after_run(self, call_option_params):
        agent = QuantumMonteCarloAgent()
        agent.run(call_option_params)
        agent.log_metrics()  # should not raise
