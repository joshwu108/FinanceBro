"""Tests for Quantum Backtesting Engine — TDD: tests written first.

Event-driven backtester that plugs in quantum subroutines
(portfolio optimization, signal generation) and evaluates
performance over time with transaction costs and slippage.
"""

import numpy as np
import pandas as pd
import pytest

from agents.quantum.quantum_backtester import (
    QuantumBacktester,
    BacktestResult,
    Trade,
    rebalance_portfolio,
    compute_performance_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_data():
    """1 year of synthetic daily prices for 3 assets."""
    rng = np.random.default_rng(42)
    n_days = 252
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    prices = {
        "AAPL": 150 * np.cumprod(1 + rng.normal(0.0004, 0.015, n_days)),
        "MSFT": 300 * np.cumprod(1 + rng.normal(0.0003, 0.012, n_days)),
        "GOOG": 130 * np.cumprod(1 + rng.normal(0.0002, 0.018, n_days)),
    }
    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def returns_data(price_data):
    """Daily returns from price data."""
    return price_data.pct_change().dropna()


@pytest.fixture
def default_backtester():
    return QuantumBacktester(config={
        "initial_capital": 100_000,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
        "rebalance_frequency": 21,  # monthly
        "optimizer": "classical",
        "max_weight": 0.50,
    })


@pytest.fixture
def quantum_backtester():
    return QuantumBacktester(config={
        "initial_capital": 100_000,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
        "rebalance_frequency": 21,
        "optimizer": "qaoa",
        "max_weight": 0.50,
        "qaoa_layers": 2,
        "weight_precision_bits": 3,
        "qaoa_seed": 42,
        "lookback_window": 60,
    })


# ===========================================================================
# 1. Trade data structure
# ===========================================================================

class TestTrade:

    def test_trade_creation(self):
        t = Trade(date=pd.Timestamp("2024-06-01"), ticker="AAPL",
                  shares=10.0, price=150.0, cost=0.75)
        assert t.ticker == "AAPL"
        assert t.shares == 10.0
        assert t.cost == 0.75

    def test_trade_value(self):
        t = Trade(date=pd.Timestamp("2024-06-01"), ticker="AAPL",
                  shares=10.0, price=150.0, cost=0.75)
        assert t.notional == 1500.0


# ===========================================================================
# 2. Rebalancing logic
# ===========================================================================

class TestRebalance:

    def test_rebalance_returns_trades(self):
        current_weights = np.array([0.5, 0.3, 0.2])
        target_weights = np.array([0.33, 0.34, 0.33])
        prices = np.array([150.0, 300.0, 130.0])
        capital = 100_000.0

        trades = rebalance_portfolio(
            current_weights, target_weights, prices,
            capital, cost_bps=5, slippage_bps=3,
        )
        assert isinstance(trades, list)
        assert len(trades) > 0

    def test_rebalance_same_weights_no_trades(self):
        weights = np.array([0.5, 0.5])
        prices = np.array([100.0, 200.0])

        trades = rebalance_portfolio(
            weights, weights, prices,
            100_000, cost_bps=5, slippage_bps=3,
        )
        assert len(trades) == 0

    def test_transaction_costs_positive(self):
        current = np.array([0.6, 0.4])
        target = np.array([0.4, 0.6])
        prices = np.array([100.0, 200.0])

        trades = rebalance_portfolio(
            current, target, prices,
            100_000, cost_bps=5, slippage_bps=3,
        )
        total_cost = sum(t.cost for t in trades)
        assert total_cost > 0


# ===========================================================================
# 3. Performance metrics
# ===========================================================================

class TestPerformanceMetrics:

    def test_returns_required_fields(self):
        portfolio_values = np.array([100_000, 101_000, 99_500, 102_000, 103_000])
        metrics = compute_performance_metrics(portfolio_values)
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics

    def test_positive_returns(self):
        values = np.linspace(100_000, 120_000, 252)
        metrics = compute_performance_metrics(values)
        assert metrics["total_return"] > 0
        assert metrics["annualized_return"] > 0

    def test_max_drawdown_non_negative(self):
        rng = np.random.default_rng(42)
        values = 100_000 * np.cumprod(1 + rng.normal(0.0003, 0.01, 252))
        metrics = compute_performance_metrics(values)
        assert metrics["max_drawdown"] >= 0

    def test_flat_portfolio_zero_return(self):
        values = np.full(252, 100_000.0)
        metrics = compute_performance_metrics(values)
        assert np.isclose(metrics["total_return"], 0.0, atol=1e-10)


# ===========================================================================
# 4. Backtester — classical optimizer
# ===========================================================================

class TestBacktesterClassical:

    def test_run_returns_backtest_result(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert isinstance(result, BacktestResult)

    def test_result_has_portfolio_values(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert len(result.portfolio_values) > 0

    def test_result_has_trades(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert isinstance(result.trades, list)

    def test_result_has_metrics(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert result.metrics is not None
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics

    def test_initial_capital_matches(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert np.isclose(result.portfolio_values[0], 100_000, atol=1)

    def test_transaction_costs_deducted(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert result.total_transaction_costs > 0

    def test_rebalance_dates_recorded(self, returns_data, default_backtester):
        result = default_backtester.run(returns_data)
        assert len(result.rebalance_dates) > 0


# ===========================================================================
# 5. Backtester — QAOA optimizer
# ===========================================================================

class TestBacktesterQuantum:

    def test_qaoa_run_completes(self, returns_data, quantum_backtester):
        result = quantum_backtester.run(returns_data)
        assert isinstance(result, BacktestResult)

    def test_qaoa_has_optimizer_metadata(self, returns_data, quantum_backtester):
        result = quantum_backtester.run(returns_data)
        assert result.optimizer_name == "qaoa"


# ===========================================================================
# 6. Comparison: classical vs quantum
# ===========================================================================

class TestBacktesterComparison:

    def test_compare_returns_both(self, returns_data):
        bt = QuantumBacktester(config={
            "initial_capital": 100_000,
            "transaction_cost_bps": 5,
            "slippage_bps": 3,
            "rebalance_frequency": 21,
            "max_weight": 0.50,
        })
        comparison = bt.compare(returns_data, optimizers=["classical", "qaoa"],
                                qaoa_config={"qaoa_layers": 2, "weight_precision_bits": 3, "qaoa_seed": 42, "lookback_window": 60})
        assert "classical" in comparison
        assert "qaoa" in comparison
        assert "classical" in comparison and isinstance(comparison["classical"], BacktestResult)
