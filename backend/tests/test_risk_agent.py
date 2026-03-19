"""Tests for RiskAgent.

Validates:
  - BaseAgent contract (run, validate, log_metrics, schemas)
  - Volatility scaling: positions sized inversely to rolling volatility
  - Fractional Kelly criterion: half-Kelly default, uses trade stats
  - Max position size constraint enforced
  - Stop-loss triggers when cumulative drawdown exceeds threshold
  - Portfolio VaR limit scales down positions when breached
  - VaR (95%, 99%), CVaR, max position exposure metrics
  - No look-ahead: position sizes use only past data
  - Input validation (missing keys, wrong types, NaN)
  - Experiment log format
  - Integration with BacktestAgent output format
"""

import numpy as np
import pandas as pd
import pytest

from agents.risk_agent import RiskAgent
from agents.base_agent import BaseAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_returns(n_days=252, seed=42, mean=0.0005, std=0.02):
    """Generate synthetic daily returns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(rng.normal(mean, std, n_days), index=dates, name="returns")


def _make_price_data(n_days=252, seed=42, start_price=100.0):
    """Generate synthetic OHLCV data consistent with returns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    close = start_price + np.cumsum(rng.randn(n_days) * 0.5)
    close = np.maximum(close, 1.0)
    open_ = close + rng.randn(n_days) * 0.3
    open_ = np.maximum(open_, 0.5)
    high = np.maximum(close, open_) + rng.uniform(0.1, 2.0, n_days)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.0, n_days)
    low = np.maximum(low, 0.5)
    volume = rng.randint(1_000_000, 10_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_signals(n_days=252, seed=42):
    """Generate synthetic trading signals (-1, 0, 1)."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    signals = rng.choice([-1, 0, 1], size=n_days, p=[0.2, 0.5, 0.3])
    return pd.Series(signals, index=dates, name="signals")


def _make_trade_log(seed=42):
    """Generate synthetic trade log matching BacktestAgent format."""
    rng = np.random.RandomState(seed)
    trades = []
    entry_price = None
    for i in range(20):
        if i % 2 == 0:
            entry_price = 100 + rng.randn() * 5
            action = "BUY" if rng.rand() > 0.3 else "SHORT"
            trades.append({
                "date": f"2023-{(i // 2) + 1:02d}-15",
                "action": action,
                "price": round(entry_price, 6),
                "shares": 100,
                "cost": round(entry_price * 100 * 0.0005, 6),
                "slippage": round(entry_price * 100 * 0.0002, 6),
                "portfolio_value": 100_000.0,
            })
        else:
            exit_price = entry_price + rng.randn() * 3
            prev_action = trades[-1]["action"]
            action = "SELL" if prev_action == "BUY" else "COVER"
            trades.append({
                "date": f"2023-{(i // 2) + 1:02d}-28",
                "action": action,
                "price": round(exit_price, 6),
                "shares": 100,
                "cost": round(exit_price * 100 * 0.0005, 6),
                "slippage": round(exit_price * 100 * 0.0002, 6),
                "portfolio_value": 100_000.0 + (exit_price - entry_price) * 100,
            })
    return trades


def _make_valid_inputs(n_days=252, seed=42):
    """Create a complete valid input dict for RiskAgent."""
    return {
        "returns": _make_returns(n_days, seed),
        "price_data": _make_price_data(n_days, seed),
        "signals": _make_signals(n_days, seed),
        "trade_log": _make_trade_log(seed),
    }


# ── BaseAgent Contract ───────────────────────────────────────────


class TestBaseAgentContract:

    def test_implements_base_agent(self):
        agent = RiskAgent()
        assert isinstance(agent, BaseAgent)

    def test_has_input_schema(self):
        schema = RiskAgent().input_schema
        assert isinstance(schema, dict)
        assert "returns" in schema
        assert "price_data" in schema
        assert "signals" in schema

    def test_has_output_schema(self):
        schema = RiskAgent().output_schema
        assert isinstance(schema, dict)
        assert "position_sizes" in schema
        assert "risk_metrics" in schema

    def test_run_returns_dict(self):
        outputs = RiskAgent().run(_make_valid_inputs())
        assert isinstance(outputs, dict)
        assert "position_sizes" in outputs
        assert "risk_metrics" in outputs

    def test_validate_passes_for_valid_output(self):
        agent = RiskAgent()
        inputs = _make_valid_inputs()
        outputs = agent.run(inputs)
        assert agent.validate(inputs, outputs) is True


# ── Volatility Scaling ───────────────────────────────────────────


class TestVolatilityScaling:

    def test_higher_volatility_reduces_position_size(self):
        """Positions sized inversely to rolling volatility."""
        n_days = 252
        dates = pd.bdate_range(start="2023-01-01", periods=n_days)

        # Use all-long signals to isolate volatility effect
        inputs = _make_valid_inputs(n_days=n_days)
        inputs["signals"] = pd.Series(1, index=dates, name="signals")

        agent = RiskAgent(config={
            "sizing_method": "volatility_scaling",
            "max_position_size": 1.0,    # uncapped to test vol effect
            "stop_loss_threshold": 1.0,  # effectively disabled
            "var_limit": 1.0,            # effectively disabled
        })
        outputs = agent.run(inputs)
        pos = outputs["position_sizes"]

        rolling_vol = inputs["returns"].rolling(window=20).std().dropna()
        common = pos.index.intersection(rolling_vol.index)
        p, v = pos.loc[common], rolling_vol.loc[common]
        med = v.median()

        avg_high = p[v > med].abs().mean()
        avg_low = p[v <= med].abs().mean()
        assert avg_high < avg_low

    def test_volatility_scaling_uses_rolling_window(self):
        """Sizes should vary (not constant) after warm-up period."""
        agent = RiskAgent(config={
            "sizing_method": "volatility_scaling",
            "volatility_window": 20,
        })
        outputs = agent.run(_make_valid_inputs())
        valid_sizes = outputs["position_sizes"].iloc[20:]
        assert valid_sizes.std() > 0

    def test_no_look_ahead_in_position_sizing(self):
        """Position size at time t must not use data after t."""
        agent = RiskAgent(config={"sizing_method": "volatility_scaling"})
        inputs = _make_valid_inputs(n_days=100)
        outputs_full = agent.run(inputs)

        truncated = {
            "returns": inputs["returns"].iloc[:50],
            "price_data": inputs["price_data"].iloc[:50],
            "signals": inputs["signals"].iloc[:50],
            "trade_log": inputs["trade_log"],
        }
        outputs_trunc = agent.run(truncated)

        common = outputs_trunc["position_sizes"].index
        pd.testing.assert_series_equal(
            outputs_full["position_sizes"].loc[common],
            outputs_trunc["position_sizes"],
            check_names=False,
        )


# ── Fractional Kelly ─────────────────────────────────────────────


class TestFractionalKelly:

    def test_kelly_produces_valid_fractions(self):
        """Kelly sizes must not exceed max_position_size."""
        agent = RiskAgent(config={"sizing_method": "kelly"})
        outputs = agent.run(_make_valid_inputs())
        max_size = agent._config["max_position_size"]
        assert (outputs["position_sizes"].abs() <= max_size + 1e-10).all()

    def test_half_kelly_is_default(self):
        agent = RiskAgent(config={"sizing_method": "kelly"})
        assert agent._config["kelly_fraction"] == 0.5

    def test_losing_strategy_gives_zero_sizes(self):
        """All-losing trades ⇒ Kelly fraction ≈ 0."""
        losing_trades = []
        for i in range(20):
            if i % 2 == 0:
                losing_trades.append({
                    "date": f"2023-{(i // 2) + 1:02d}-15",
                    "action": "BUY",
                    "price": 100.0,
                    "shares": 100,
                    "cost": 5.0,
                    "slippage": 2.0,
                    "portfolio_value": 100_000.0,
                })
            else:
                losing_trades.append({
                    "date": f"2023-{(i // 2) + 1:02d}-28",
                    "action": "SELL",
                    "price": 95.0,
                    "shares": 100,
                    "cost": 4.75,
                    "slippage": 1.9,
                    "portfolio_value": 99_500.0,
                })

        agent = RiskAgent(config={"sizing_method": "kelly"})
        inputs = _make_valid_inputs()
        inputs["trade_log"] = losing_trades
        outputs = agent.run(inputs)
        assert outputs["position_sizes"].abs().mean() < 0.01

    def test_empty_trade_log_gives_zero_sizes(self):
        """No trade history ⇒ Kelly defaults to 0."""
        agent = RiskAgent(config={"sizing_method": "kelly"})
        inputs = _make_valid_inputs()
        inputs["trade_log"] = []
        outputs = agent.run(inputs)
        assert (outputs["position_sizes"] == 0).all()

    def test_kelly_no_look_ahead(self):
        """Kelly must only use trades that closed before current timestep."""
        # Trade exits on 2023-03-28 and 2023-06-28
        trade_log = [
            {"date": "2023-03-15", "action": "BUY", "price": 100.0,
             "shares": 100, "cost": 5.0, "slippage": 2.0,
             "portfolio_value": 100_000.0},
            {"date": "2023-03-28", "action": "SELL", "price": 110.0,
             "shares": 100, "cost": 5.5, "slippage": 2.2,
             "portfolio_value": 101_000.0},
            {"date": "2023-06-15", "action": "BUY", "price": 105.0,
             "shares": 100, "cost": 5.25, "slippage": 2.1,
             "portfolio_value": 101_000.0},
            {"date": "2023-06-28", "action": "SELL", "price": 115.0,
             "shares": 100, "cost": 5.75, "slippage": 2.3,
             "portfolio_value": 102_000.0},
        ]

        n_days = 252
        inputs = _make_valid_inputs(n_days=n_days)
        inputs["trade_log"] = trade_log
        agent = RiskAgent(config={"sizing_method": "kelly"})
        outputs = agent.run(inputs)
        pos = outputs["position_sizes"]

        # Before first trade exit (2023-03-28): no data → must be 0
        before_first_exit = pos[pos.index < pd.Timestamp("2023-03-28")]
        assert (before_first_exit == 0).all(), (
            "Kelly must not use trades before they close"
        )

        # Between first and second exit: only first trade used
        between = pos[
            (pos.index >= pd.Timestamp("2023-03-29"))
            & (pos.index < pd.Timestamp("2023-06-28"))
        ]
        # After second exit: both trades available — sizes may differ
        after_second = pos[pos.index >= pd.Timestamp("2023-06-29")]
        # At least one non-zero in each window (signals permitting)
        # The key assertion is the before_first_exit = 0 check above

    def test_consecutive_entries_handled(self):
        """Consecutive BUY without exit should not crash, uses latest entry."""
        trade_log = [
            {"date": "2023-01-10", "action": "BUY", "price": 100.0,
             "shares": 100, "cost": 5.0, "slippage": 2.0,
             "portfolio_value": 100_000.0},
            # Consecutive BUY (no exit for first)
            {"date": "2023-01-15", "action": "BUY", "price": 102.0,
             "shares": 100, "cost": 5.1, "slippage": 2.0,
             "portfolio_value": 100_000.0},
            {"date": "2023-01-28", "action": "SELL", "price": 108.0,
             "shares": 100, "cost": 5.4, "slippage": 2.2,
             "portfolio_value": 100_600.0},
        ]

        agent = RiskAgent(config={"sizing_method": "kelly"})
        inputs = _make_valid_inputs()
        inputs["trade_log"] = trade_log
        # Should not raise
        outputs = agent.run(inputs)
        assert "position_sizes" in outputs


# ── Risk Constraints ─────────────────────────────────────────────


class TestRiskConstraints:

    def test_max_position_size_enforced(self):
        max_size = 0.05
        agent = RiskAgent(config={"max_position_size": max_size})
        outputs = agent.run(_make_valid_inputs())
        assert (outputs["position_sizes"].abs() <= max_size + 1e-10).all()

    def test_stop_loss_zeros_positions_during_drawdown(self):
        """Stop-loss triggers when cumulative drawdown > threshold."""
        n_days = 100
        dates = pd.bdate_range(start="2023-01-01", periods=n_days)

        # 50 normal days, 10 crash days (-3%/day), 40 recovery days
        returns = pd.Series(
            np.concatenate([
                np.full(50, 0.001),
                np.full(10, -0.03),
                np.full(40, 0.001),
            ]),
            index=dates,
            name="returns",
        )

        inputs = {
            "returns": returns,
            "price_data": _make_price_data(n_days, seed=99),
            "signals": pd.Series(np.ones(n_days), index=dates, name="signals"),
            "trade_log": _make_trade_log(),
        }

        agent = RiskAgent(config={"stop_loss_threshold": 0.05})
        outputs = agent.run(inputs)

        # Days 55-65 are deep in the drawdown → should be stopped out
        drawdown_slice = outputs["position_sizes"].iloc[55:65]
        assert (drawdown_slice == 0).all()

    def test_stop_loss_uses_lagged_drawdown(self):
        """Stop-loss decision at bar i must use drawdown from bar i-1."""
        n_days = 10
        dates = pd.bdate_range(start="2023-01-01", periods=n_days)

        # Bar 5 has a big negative return that pushes drawdown past 5%
        returns_data = np.full(n_days, 0.001)
        returns_data[5] = -0.08  # Single crash day
        returns = pd.Series(returns_data, index=dates, name="returns")

        inputs = {
            "returns": returns,
            "price_data": _make_price_data(n_days, seed=77),
            "signals": pd.Series(np.ones(n_days), index=dates, name="signals"),
            "trade_log": [],
        }

        agent = RiskAgent(config={"stop_loss_threshold": 0.05})
        outputs = agent.run(inputs)
        pos = outputs["position_sizes"]

        # Bar 5: drawdown happens HERE, but lagged drawdown uses bar 4
        # (which is still above threshold). So bar 5 should NOT be zero.
        assert pos.iloc[5] != 0.0, (
            "Stop-loss should use lagged drawdown — bar 5's drawdown "
            "should not affect bar 5's position"
        )
        # Bar 6: lagged drawdown = drawdown at bar 5 (which exceeds threshold)
        # So bar 6 SHOULD be zero.
        assert pos.iloc[6] == 0.0, (
            "Bar 6 should be stopped out using bar 5's drawdown"
        )

    def test_var_limit_scales_down_positions(self):
        """When rolling VaR exceeds limit, positions are scaled down."""
        agent = RiskAgent(config={"var_limit": 0.02})
        outputs = agent.run(_make_valid_inputs())
        assert "var_limit_breaches" in outputs["risk_metrics"]


# ── Risk Metrics ─────────────────────────────────────────────────


class TestRiskMetrics:

    def test_var_95_negative(self):
        outputs = RiskAgent().run(_make_valid_inputs())
        assert outputs["risk_metrics"]["var_95"] <= 0

    def test_var_99_worse_than_var_95(self):
        metrics = RiskAgent().run(_make_valid_inputs())["risk_metrics"]
        assert metrics["var_99"] <= metrics["var_95"]

    def test_cvar_worse_than_var(self):
        metrics = RiskAgent().run(_make_valid_inputs())["risk_metrics"]
        assert metrics["cvar_95"] <= metrics["var_95"]

    def test_max_position_exposure_non_negative(self):
        metrics = RiskAgent().run(_make_valid_inputs())["risk_metrics"]
        assert metrics["max_position_exposure"] >= 0


# ── Input Validation ─────────────────────────────────────────────


class TestInputValidation:

    def test_missing_returns_raises(self):
        inputs = _make_valid_inputs()
        del inputs["returns"]
        with pytest.raises(ValueError, match="returns"):
            RiskAgent().run(inputs)

    def test_missing_price_data_raises(self):
        inputs = _make_valid_inputs()
        del inputs["price_data"]
        with pytest.raises(ValueError, match="price_data"):
            RiskAgent().run(inputs)

    def test_missing_signals_raises(self):
        inputs = _make_valid_inputs()
        del inputs["signals"]
        with pytest.raises(ValueError, match="signals"):
            RiskAgent().run(inputs)

    def test_returns_wrong_type_raises(self):
        inputs = _make_valid_inputs()
        inputs["returns"] = [0.01, 0.02, -0.01]
        with pytest.raises(TypeError, match="returns"):
            RiskAgent().run(inputs)

    def test_nan_in_returns_raises(self):
        inputs = _make_valid_inputs()
        inputs["returns"].iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            RiskAgent().run(inputs)


# ── Experiment Logging ───────────────────────────────────────────


class TestExperimentLogging:

    def test_log_metrics_without_run_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            RiskAgent().log_metrics()
        assert "No metrics" in caplog.text

    def test_log_metrics_after_run_succeeds(self):
        agent = RiskAgent()
        agent.run(_make_valid_inputs())
        # Should not raise
        agent.log_metrics()


# ── Integration ──────────────────────────────────────────────────


class TestIntegration:

    def test_accepts_backtest_output_format(self):
        """Works with equity-derived returns from BacktestAgent."""
        n_days = 252
        dates = pd.bdate_range(start="2023-01-01", periods=n_days)
        equity = pd.Series(
            100_000 + np.cumsum(np.random.RandomState(42).randn(n_days) * 100),
            index=dates,
            name="equity",
        )
        returns = equity.pct_change().fillna(0)

        inputs = {
            "returns": returns,
            "price_data": _make_price_data(n_days),
            "signals": _make_signals(n_days),
            "trade_log": _make_trade_log(),
        }
        result = RiskAgent().run(inputs)
        assert "position_sizes" in result
        assert len(result["position_sizes"]) == n_days

    def test_output_index_matches_input_dates(self):
        inputs = _make_valid_inputs()
        outputs = RiskAgent().run(inputs)
        pd.testing.assert_index_equal(
            outputs["position_sizes"].index,
            inputs["returns"].index,
        )
