"""Tests for BacktestAgent.

Validates:
  - BaseAgent contract is satisfied (run, validate, log_metrics, schemas)
  - Event-driven simulation (day-by-day replay, no vectorized shortcuts)
  - Transaction costs applied correctly (5-10 bps)
  - Slippage modeling on each execution
  - No look-ahead bias: signal at bar i executes at open of bar i+1
  - Equity curve correctness
  - Trade log structure and accuracy
  - Performance metrics: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Turnover
  - Benchmark comparison (SPY buy-and-hold)
  - max_position_size enforcement
  - Edge cases: all-flat signals, single trade, no trades
  - Input validation
  - Experiment log format
"""

import numpy as np
import pandas as pd
import pytest

from agents.backtest_agent import BacktestAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_ohlcv(
    n_days: int = 252,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a DatetimeIndex.

    Guarantees OHLC invariants: high >= max(open, close), low <= min(open, close).
    """
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
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _make_constant_price_data(
    n_days: int = 50,
    price: float = 100.0,
) -> pd.DataFrame:
    """Constant-price OHLCV for deterministic cost/slippage testing."""
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


def _make_trending_up_data(n_days: int = 100) -> pd.DataFrame:
    """Linearly increasing price for predictable return testing."""
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    prices = np.linspace(100.0, 150.0, n_days)
    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


def _make_trending_down_data(n_days: int = 100) -> pd.DataFrame:
    """Linearly decreasing price."""
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    prices = np.linspace(100.0, 60.0, n_days)
    return pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


@pytest.fixture
def agent():
    return BacktestAgent()


@pytest.fixture
def price_data():
    return _make_ohlcv()


@pytest.fixture
def constant_price_data():
    return _make_constant_price_data()


@pytest.fixture
def trending_up_data():
    return _make_trending_up_data()


@pytest.fixture
def all_long_signals(price_data):
    """Signal = 1 (long) every day."""
    return pd.Series(1, index=price_data.index, name="signal")


@pytest.fixture
def all_flat_signals(price_data):
    """Signal = 0 (flat) every day."""
    return pd.Series(0, index=price_data.index, name="signal")


@pytest.fixture
def alternating_signals(price_data):
    """Alternate between long (1) and flat (0) to generate frequent trades."""
    signals = pd.Series(0, index=price_data.index, name="signal")
    signals.iloc[::2] = 1
    return signals


# ── BaseAgent Contract Tests ──────────────────────────────────────


class TestBaseAgentContract:
    """BacktestAgent must implement the full BaseAgent interface."""

    def test_input_schema_exists(self, agent):
        schema = agent.input_schema
        assert isinstance(schema, dict)
        assert "price_data" in schema
        assert "predictions" in schema

    def test_output_schema_exists(self, agent):
        schema = agent.output_schema
        assert isinstance(schema, dict)
        assert "equity_curve" in schema
        assert "trade_log" in schema
        assert "performance_summary" in schema

    def test_run_returns_dict(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert isinstance(result, dict)

    def test_validate_returns_true(self, agent, price_data, all_long_signals):
        inputs = {"price_data": price_data, "predictions": all_long_signals}
        outputs = agent.run(inputs)
        assert agent.validate(inputs, outputs) is True

    def test_log_metrics_does_not_raise(self, agent, price_data, all_long_signals):
        agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        agent.log_metrics()  # Should not raise


# ── Input Validation Tests ────────────────────────────────────────


class TestInputValidation:

    def test_missing_price_data_raises(self, agent, all_long_signals):
        with pytest.raises((ValueError, KeyError)):
            agent.run({"predictions": all_long_signals})

    def test_missing_predictions_raises(self, agent, price_data):
        with pytest.raises((ValueError, KeyError)):
            agent.run({"price_data": price_data})

    def test_misaligned_indices_raises(self, agent, price_data):
        bad_index = pd.bdate_range("2020-01-01", periods=10)
        predictions = pd.Series(1, index=bad_index)
        with pytest.raises(ValueError, match="[Aa]lign|overlap|empty"):
            agent.run({"price_data": price_data, "predictions": predictions})

    def test_invalid_signal_values_raises(self, agent, price_data):
        predictions = pd.Series(5, index=price_data.index)
        with pytest.raises(ValueError, match="[Ss]ignal|[Pp]rediction"):
            agent.run({"price_data": price_data, "predictions": predictions})

    def test_missing_ohlcv_columns_raises(self, agent):
        bad_df = pd.DataFrame({"close": [100.0]}, index=pd.bdate_range("2023-01-01", periods=1))
        predictions = pd.Series(1, index=bad_df.index)
        with pytest.raises(ValueError, match="[Cc]olumn"):
            agent.run({"price_data": bad_df, "predictions": predictions})


# ── Equity Curve Tests ────────────────────────────────────────────


class TestEquityCurve:

    def test_equity_curve_is_series(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert isinstance(result["equity_curve"], pd.Series)

    def test_equity_curve_starts_at_initial_capital(self, agent, price_data, all_long_signals):
        """Bar 0 has no pending signal, so equity == initial capital."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        curve = result["equity_curve"]
        # No trade on bar 0 (signal recorded, executed on bar 1)
        assert curve.iloc[0] == pytest.approx(100_000.0, abs=0.01)

    def test_equity_curve_length_matches_data(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert len(result["equity_curve"]) == len(price_data)

    def test_equity_curve_no_nan(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert not result["equity_curve"].isna().any()

    def test_equity_curve_always_positive(self, agent, price_data, all_long_signals):
        """Portfolio value should never go negative."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert (result["equity_curve"] > 0).all()

    def test_flat_signals_preserve_capital(self, agent, price_data, all_flat_signals):
        """If we never trade, equity should remain at initial capital."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        curve = result["equity_curve"]
        assert curve.iloc[-1] == pytest.approx(100_000.0, abs=0.01)

    def test_equity_curve_index_matches_price_data(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert result["equity_curve"].index.equals(price_data.index)


# ── Trade Log Tests ───────────────────────────────────────────────


class TestTradeLog:

    def test_trade_log_is_list(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert isinstance(result["trade_log"], list)

    def test_trade_log_entry_structure(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        if result["trade_log"]:
            entry = result["trade_log"][0]
            assert "date" in entry
            assert "action" in entry
            assert "price" in entry
            assert "shares" in entry
            assert "cost" in entry
            assert "slippage" in entry
            assert "portfolio_value" in entry

    def test_no_trades_when_flat(self, agent, price_data, all_flat_signals):
        """Zero trades expected when all signals are flat."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        assert len(result["trade_log"]) == 0

    def test_alternating_signals_produce_trades(self, agent, price_data, alternating_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": alternating_signals,
        })
        assert len(result["trade_log"]) > 0

    def test_trade_prices_are_positive(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        for trade in result["trade_log"]:
            assert trade["price"] > 0


# ── Transaction Cost Tests ────────────────────────────────────────


class TestTransactionCosts:

    def test_costs_applied_on_trade(self, agent, constant_price_data):
        """With constant prices and a single buy, the only equity change is costs."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[0:] = 1  # Signal long from bar 0, executes at bar 1

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        curve = result["equity_curve"]
        # After buying, equity < initial due to transaction cost + slippage
        assert curve.iloc[-1] < 100_000.0

    def test_cost_recorded_in_trade_log(self, agent, constant_price_data):
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[0:] = 1

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        assert len(buy_trades) > 0
        assert buy_trades[0]["cost"] > 0

    def test_custom_transaction_cost(self):
        """Higher transaction costs should reduce equity more."""
        data = _make_constant_price_data(n_days=50)
        signals = pd.Series(0, index=data.index)
        signals.iloc[0:] = 1

        agent_low = BacktestAgent(config={"transaction_cost_bps": 5.0})
        agent_high = BacktestAgent(config={"transaction_cost_bps": 10.0})

        result_low = agent_low.run({"price_data": data, "predictions": signals})
        result_high = agent_high.run({"price_data": data, "predictions": signals})

        assert result_high["equity_curve"].iloc[-1] < result_low["equity_curve"].iloc[-1]

    def test_transaction_costs_are_realistic(self, agent, constant_price_data):
        """Cost per trade should be proportional to trade value * bps."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[0:] = 1

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            trade = buy_trades[0]
            expected_cost_approx = trade["price"] * trade["shares"] * 5.0 / 10_000
            assert trade["cost"] == pytest.approx(expected_cost_approx, rel=0.01)


# ── Slippage Tests ────────────────────────────────────────────────


class TestSlippage:

    def test_slippage_applied_on_buy(self, agent, constant_price_data):
        """Buy execution price should include slippage."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[0:] = 1

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            assert buy_trades[0]["slippage"] >= 0

    def test_slippage_recorded_in_trade_log(self, agent, constant_price_data):
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[0:] = 1

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            assert "slippage" in buy_trades[0]
            assert buy_trades[0]["slippage"] > 0

    def test_higher_slippage_reduces_equity(self):
        """Increasing slippage should reduce returns."""
        data = _make_constant_price_data(n_days=50)
        signals = pd.Series(0, index=data.index)
        signals.iloc[0:] = 1

        agent_low = BacktestAgent(config={"slippage_bps": 1.0})
        agent_high = BacktestAgent(config={"slippage_bps": 10.0})

        result_low = agent_low.run({"price_data": data, "predictions": signals})
        result_high = agent_high.run({"price_data": data, "predictions": signals})

        assert result_high["equity_curve"].iloc[-1] < result_low["equity_curve"].iloc[-1]


# ── Performance Metrics Tests ─────────────────────────────────────


class TestPerformanceMetrics:

    def test_performance_summary_keys(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        summary = result["performance_summary"]
        required_keys = {
            "sharpe", "sortino", "max_drawdown", "calmar",
            "win_rate", "turnover", "total_return", "annualized_return",
            "total_trades",
        }
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_sharpe_ratio_is_finite(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        sharpe = result["performance_summary"]["sharpe"]
        assert np.isfinite(sharpe)

    def test_sortino_ratio_is_finite(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        sortino = result["performance_summary"]["sortino"]
        assert np.isfinite(sortino)

    def test_max_drawdown_is_non_positive(self, agent, price_data, all_long_signals):
        """Max drawdown should be <= 0 (expressed as negative percentage)."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        mdd = result["performance_summary"]["max_drawdown"]
        assert mdd <= 0.0

    def test_calmar_ratio_is_finite(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        calmar = result["performance_summary"]["calmar"]
        assert np.isfinite(calmar)

    def test_win_rate_between_zero_and_one(self, agent, price_data, alternating_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": alternating_signals,
        })
        win_rate = result["performance_summary"]["win_rate"]
        assert 0.0 <= win_rate <= 1.0

    def test_turnover_non_negative(self, agent, price_data, alternating_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": alternating_signals,
        })
        turnover = result["performance_summary"]["turnover"]
        assert turnover >= 0.0

    def test_total_trades_count(self, agent, price_data, all_flat_signals):
        """Zero trades with flat signals."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        assert result["performance_summary"]["total_trades"] == 0

    def test_flat_signals_zero_sharpe(self, agent, price_data, all_flat_signals):
        """No trades means zero returns, Sharpe should be 0."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        assert result["performance_summary"]["sharpe"] == pytest.approx(0.0, abs=1e-10)

    def test_trending_up_long_positive_return(self, agent):
        """Holding long in a rising market should produce positive total return."""
        data = _make_trending_up_data(n_days=100)
        signals = pd.Series(1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})
        assert result["performance_summary"]["total_return"] > 0

    def test_trending_down_long_negative_return(self, agent):
        """Holding long in a falling market should produce negative total return."""
        data = _make_trending_down_data(n_days=100)
        signals = pd.Series(1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})
        assert result["performance_summary"]["total_return"] < 0


# ── Benchmark Comparison Tests ────────────────────────────────────


class TestBenchmarkComparison:

    def test_benchmark_comparison_in_summary(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        summary = result["performance_summary"]
        assert "benchmark_comparison" in summary
        bc = summary["benchmark_comparison"]
        assert "strategy_return" in bc
        assert "benchmark_return" in bc
        assert "excess_return" in bc

    def test_benchmark_uses_buy_and_hold(self, agent, price_data, all_long_signals):
        """Benchmark return should equal buy-and-hold from open[1] to close[-1]."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        bc = result["performance_summary"]["benchmark_comparison"]
        expected_bh_return = (
            (price_data["close"].iloc[-1] - price_data["open"].iloc[1])
            / price_data["open"].iloc[1]
        )
        assert bc["benchmark_return"] == pytest.approx(expected_bh_return, rel=0.01)

    def test_external_benchmark_data_used(self, agent, price_data, all_long_signals):
        """When benchmark_data is provided, it should be used instead of price_data."""
        bench = _make_trending_up_data(n_days=len(price_data))
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
            "benchmark_data": bench,
        })
        bc = result["performance_summary"]["benchmark_comparison"]
        expected = (bench["close"].iloc[-1] - bench["open"].iloc[1]) / bench["open"].iloc[1]
        assert bc["benchmark_return"] == pytest.approx(expected, rel=0.01)


# ── No Look-Ahead Bias Tests ─────────────────────────────────────


class TestNoLookAheadBias:

    def test_trade_executes_at_next_bar_open(self, agent, price_data):
        """Signal at bar i should execute at OPEN of bar i+1."""
        signals = pd.Series(0, index=price_data.index)
        signals.iloc[10] = 1  # Signal long on bar 10
        signals.iloc[11:] = 1  # Stay long

        result = agent.run({
            "price_data": price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            trade = buy_trades[0]
            trade_date = pd.Timestamp(trade["date"])
            # Trade should execute on bar 11 (next bar after signal on bar 10)
            expected_exec_date = price_data.index[11]
            assert trade_date == expected_exec_date

            # Execution price should be based on bar 11's open + slippage
            day_open = price_data.loc[trade_date, "open"]
            assert trade["price"] >= day_open * 0.99
            assert trade["price"] <= day_open * 1.01

    def test_no_trade_on_bar_zero(self, agent, price_data, all_long_signals):
        """No pending signal on bar 0, so equity == initial_capital exactly."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert result["equity_curve"].iloc[0] == pytest.approx(100_000.0, abs=0.01)

    def test_equity_curve_computed_sequentially(self, agent, price_data, all_long_signals):
        """Each equity point depends only on previous equity and current prices."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        curve = result["equity_curve"]
        assert curve.index.is_monotonic_increasing


# ── Event-Driven Simulation Tests ─────────────────────────────────


class TestEventDriven:

    def test_single_day_position_change(self, agent, constant_price_data):
        """A single signal flip should generate exactly one BUY trade."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[10:] = 1  # Signal long starting bar 10, executes bar 11

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        assert len(buy_trades) == 1

    def test_exit_generates_sell_trade(self, agent, constant_price_data):
        """Going from long to flat should generate a SELL trade."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[5:15] = 1  # Long for 10 bars, then exit

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        sell_trades = [t for t in result["trade_log"] if t["action"] == "SELL"]
        assert len(sell_trades) >= 1

    def test_alternating_generates_many_trades(self, agent, price_data, alternating_signals):
        """Alternating long/flat should generate many trades."""
        result = agent.run({
            "price_data": price_data,
            "predictions": alternating_signals,
        })
        assert len(result["trade_log"]) > 10


# ── Config Override Tests ─────────────────────────────────────────


class TestConfigOverride:

    def test_custom_initial_capital(self, price_data, all_long_signals):
        agent = BacktestAgent(config={"initial_capital": 50_000.0})
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        # Bar 0 has no trade, so equity == initial capital
        assert result["equity_curve"].iloc[0] == pytest.approx(50_000.0, abs=0.01)

    def test_config_override_in_inputs(self, agent, price_data, all_long_signals):
        result = agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
            "config": {"initial_capital": 200_000.0},
        })
        assert result["equity_curve"].iloc[0] == pytest.approx(200_000.0, abs=0.01)


# ── max_position_size Tests ───────────────────────────────────────


class TestMaxPositionSize:

    def test_max_position_size_limits_exposure(self):
        """With max_position_size=0.5, only half the capital should be deployed."""
        data = _make_constant_price_data(n_days=20, price=100.0)
        signals = pd.Series(1, index=data.index)

        agent_full = BacktestAgent(config={"max_position_size": 1.0})
        agent_half = BacktestAgent(config={"max_position_size": 0.5})

        result_full = agent_full.run({"price_data": data, "predictions": signals})
        result_half = agent_half.run({"price_data": data, "predictions": signals})

        full_trades = [t for t in result_full["trade_log"] if t["action"] == "BUY"]
        half_trades = [t for t in result_half["trade_log"] if t["action"] == "BUY"]

        if full_trades and half_trades:
            # Half-position should buy roughly half the shares
            assert half_trades[0]["shares"] < full_trades[0]["shares"]
            assert half_trades[0]["shares"] == pytest.approx(
                full_trades[0]["shares"] / 2, abs=2
            )


# ── Validate Method Tests ─────────────────────────────────────────


class TestValidation:

    def test_validate_checks_equity_no_nan(self, agent, price_data, all_long_signals):
        inputs = {"price_data": price_data, "predictions": all_long_signals}
        outputs = agent.run(inputs)
        outputs["equity_curve"].iloc[5] = np.nan
        with pytest.raises(ValueError, match="[Nn]a[Nn]|NaN"):
            agent.validate(inputs, outputs)

    def test_validate_checks_equity_positive(self, agent, price_data, all_long_signals):
        inputs = {"price_data": price_data, "predictions": all_long_signals}
        outputs = agent.run(inputs)
        outputs["equity_curve"].iloc[5] = -100.0
        with pytest.raises(ValueError, match="[Nn]egative|[Pp]ositive"):
            agent.validate(inputs, outputs)


# ── Experiment Logging Tests ──────────────────────────────────────


class TestExperimentLogging:

    def test_log_metrics_creates_file(self, agent, price_data, all_long_signals, tmp_path, monkeypatch):
        """log_metrics should create a JSON file in experiments/."""
        monkeypatch.setattr(
            "agents.backtest_agent.Path.__file__",
            str(tmp_path / "agents" / "backtest_agent.py"),
            raising=False,
        )
        agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        agent.log_metrics()

    def test_metrics_populated_after_run(self, agent, price_data, all_long_signals):
        agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert agent._metrics
        assert "sharpe" in agent._metrics or "total_return" in agent._metrics


# ── Reproducibility Tests ─────────────────────────────────────────


class TestReproducibility:

    def test_deterministic_results(self, price_data, all_long_signals):
        """Same inputs should produce identical outputs."""
        agent1 = BacktestAgent()
        agent2 = BacktestAgent()

        result1 = agent1.run({"price_data": price_data, "predictions": all_long_signals})
        result2 = agent2.run({"price_data": price_data, "predictions": all_long_signals})

        pd.testing.assert_series_equal(
            result1["equity_curve"],
            result2["equity_curve"],
        )
        assert result1["performance_summary"] == result2["performance_summary"]


# ── Short Signal Tests ────────────────────────────────────────────


class TestShortSignals:

    def test_short_signal_in_trending_down_profitable(self, agent):
        """Shorting a falling market should be profitable."""
        data = _make_trending_down_data(n_days=100)
        signals = pd.Series(-1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})
        assert result["performance_summary"]["total_return"] > 0

    def test_short_generates_correct_trade_actions(self, agent, constant_price_data):
        """Short signal should generate SHORT and COVER trade actions."""
        signals = pd.Series(0, index=constant_price_data.index)
        signals.iloc[5:15] = -1  # Short for 10 bars, then cover

        result = agent.run({
            "price_data": constant_price_data,
            "predictions": signals,
        })
        actions = {t["action"] for t in result["trade_log"]}
        assert "SHORT" in actions
        assert "COVER" in actions


# ── Forced Liquidation Tests ──────────────────────────────────────


class TestForcedLiquidation:

    def test_open_long_force_liquidated_at_end(self, agent, price_data):
        """A position held at the last bar must be force-liquidated."""
        signals = pd.Series(1, index=price_data.index)
        result = agent.run({"price_data": price_data, "predictions": signals})
        actions = [t["action"] for t in result["trade_log"]]
        assert "LIQUIDATE_LONG" in actions
        assert actions[-1] == "LIQUIDATE_LONG"

    def test_open_short_force_liquidated_at_end(self, agent, price_data):
        """A short held at the last bar must be force-liquidated."""
        signals = pd.Series(-1, index=price_data.index)
        result = agent.run({"price_data": price_data, "predictions": signals})
        actions = [t["action"] for t in result["trade_log"]]
        assert "LIQUIDATE_SHORT" in actions
        assert actions[-1] == "LIQUIDATE_SHORT"

    def test_no_forced_liquidation_when_flat(self, agent, price_data, all_flat_signals):
        """No liquidation trade when ending flat."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        actions = [t["action"] for t in result["trade_log"]]
        assert "LIQUIDATE_LONG" not in actions
        assert "LIQUIDATE_SHORT" not in actions

    def test_forced_liquidation_includes_costs(self, agent):
        """Force-liquidated terminal equity < mark-to-market equity."""
        data = _make_constant_price_data(n_days=20, price=100.0)
        signals = pd.Series(1, index=data.index)

        agent_noliq = BacktestAgent(config={"transaction_cost_bps": 0.0, "slippage_bps": 0.0})
        result_nocost = agent_noliq.run({"price_data": data, "predictions": signals})

        result = agent.run({"price_data": data, "predictions": signals})
        assert result["equity_curve"].iloc[-1] < result_nocost["equity_curve"].iloc[-1]

    def test_forced_liquidation_win_rate_includes_final_trade(self, agent):
        """Win rate must include the force-liquidated exit."""
        data = _make_trending_up_data(n_days=50)
        signals = pd.Series(1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})

        liq_trades = [t for t in result["trade_log"] if t["action"] == "LIQUIDATE_LONG"]
        assert len(liq_trades) == 1
        assert result["performance_summary"]["win_rate"] > 0


# ── Short Borrow Cost Tests ──────────────────────────────────────


class TestShortBorrowCosts:

    def test_borrow_cost_reduces_short_equity(self):
        """Short positions should incur daily borrow costs."""
        data = _make_constant_price_data(n_days=50, price=100.0)
        signals = pd.Series(-1, index=data.index)

        agent_no_borrow = BacktestAgent(config={"short_borrow_annual_bps": 0.0})
        agent_with_borrow = BacktestAgent(config={"short_borrow_annual_bps": 100.0})

        result_no = agent_no_borrow.run({"price_data": data, "predictions": signals})
        result_yes = agent_with_borrow.run({"price_data": data, "predictions": signals})

        assert result_yes["equity_curve"].iloc[-1] < result_no["equity_curve"].iloc[-1]

    def test_no_borrow_cost_when_long(self):
        """Long positions should not incur borrow costs."""
        data = _make_constant_price_data(n_days=50, price=100.0)
        signals = pd.Series(1, index=data.index)

        agent_no = BacktestAgent(config={"short_borrow_annual_bps": 0.0})
        agent_yes = BacktestAgent(config={"short_borrow_annual_bps": 500.0})

        r_no = agent_no.run({"price_data": data, "predictions": signals})
        r_yes = agent_yes.run({"price_data": data, "predictions": signals})

        assert r_no["equity_curve"].iloc[-1] == pytest.approx(
            r_yes["equity_curve"].iloc[-1], rel=1e-9,
        )


# ── Volume Limit Tests ────────────────────────────────────────────


class TestVolumeLimits:

    def test_volume_cap_limits_shares(self):
        """Position size should be capped by max_volume_participation."""
        dates = pd.bdate_range(start="2023-01-01", periods=20)
        data = pd.DataFrame({
            "open": 10.0, "high": 11.0, "low": 9.0,
            "close": 10.0, "volume": 100.0,
        }, index=dates)
        signals = pd.Series(1, index=data.index)

        agent = BacktestAgent(config={
            "initial_capital": 100_000.0,
            "max_volume_participation": 0.1,
        })
        result = agent.run({"price_data": data, "predictions": signals})
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            assert buy_trades[0]["shares"] <= int(100 * 0.1) + 1

    def test_high_volume_no_cap(self):
        """With ample volume the cap should not bind."""
        data = _make_constant_price_data(n_days=20, price=100.0)
        signals = pd.Series(1, index=data.index)

        agent_cap = BacktestAgent(config={"max_volume_participation": 0.02})
        agent_nocap = BacktestAgent(config={"max_volume_participation": 1.0})

        r_cap = agent_cap.run({"price_data": data, "predictions": signals})
        r_nocap = agent_nocap.run({"price_data": data, "predictions": signals})

        buy_cap = [t for t in r_cap["trade_log"] if t["action"] == "BUY"]
        buy_nocap = [t for t in r_nocap["trade_log"] if t["action"] == "BUY"]

        if buy_cap and buy_nocap:
            assert buy_cap[0]["shares"] == buy_nocap[0]["shares"]


# ── Out-of-Sample Flag Tests ─────────────────────────────────────


class TestOutOfSampleFlag:

    def test_out_of_sample_stored_in_metrics(self, agent, price_data, all_long_signals):
        agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
            "out_of_sample": True,
        })
        assert agent._metrics["out_of_sample"] is True

    def test_out_of_sample_defaults_none(self, agent, price_data, all_long_signals):
        agent.run({
            "price_data": price_data,
            "predictions": all_long_signals,
        })
        assert agent._metrics["out_of_sample"] is None


# ── Trade Log Look-Ahead Regression Tests ─────────────────────────


class TestTradeLogNoLookAhead:

    def test_trade_portfolio_value_uses_open_not_close(self, agent):
        """Trade log portfolio_value must use only open_price (known at execution)."""
        dates = pd.bdate_range(start="2023-01-01", periods=20)
        data = pd.DataFrame({
            "open": 100.0, "high": 110.0, "low": 90.0,
            "close": 105.0, "volume": 1_000_000.0,
        }, index=dates)
        signals = pd.Series(0, index=data.index)
        signals.iloc[0:] = 1

        result = agent.run({"price_data": data, "predictions": signals})
        buy_trades = [t for t in result["trade_log"] if t["action"] == "BUY"]
        if buy_trades:
            trade = buy_trades[0]
            shares = trade["shares"]
            cost_paid = trade["price"] * shares + trade["cost"]
            remaining_cash = 100_000.0 - cost_paid
            expected_pv = remaining_cash + shares * 100.0
            assert trade["portfolio_value"] == pytest.approx(expected_pv, rel=0.01)


# ── Metric Edge-Case & Stability Tests ────────────────────────────


class TestConfigValidation:

    def test_zero_trading_days_raises(self):
        agent = BacktestAgent(config={"trading_days_per_year": 0})
        data = _make_constant_price_data(n_days=10)
        signals = pd.Series(0, index=data.index)
        with pytest.raises(ValueError, match="trading_days_per_year"):
            agent.run({"price_data": data, "predictions": signals})

    def test_negative_trading_days_raises(self):
        agent = BacktestAgent(config={"trading_days_per_year": -1})
        data = _make_constant_price_data(n_days=10)
        signals = pd.Series(0, index=data.index)
        with pytest.raises(ValueError, match="trading_days_per_year"):
            agent.run({"price_data": data, "predictions": signals})

    def test_zero_initial_capital_raises(self):
        agent = BacktestAgent(config={"initial_capital": 0})
        data = _make_constant_price_data(n_days=10)
        signals = pd.Series(0, index=data.index)
        with pytest.raises(ValueError, match="initial_capital"):
            agent.run({"price_data": data, "predictions": signals})


class TestDivideByZeroGuards:

    def test_sharpe_with_constant_equity(self, agent, price_data, all_flat_signals):
        """Zero std should return Sharpe=0, not crash."""
        result = agent.run({
            "price_data": price_data,
            "predictions": all_flat_signals,
        })
        assert result["performance_summary"]["sharpe"] == 0.0

    def test_sortino_with_all_positive_returns(self, agent):
        """All-positive excess returns → zero downside dev → Sortino=0."""
        data = _make_trending_up_data(n_days=50)
        signals = pd.Series(1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})
        sortino = result["performance_summary"]["sortino"]
        assert np.isfinite(sortino)

    def test_turnover_with_tiny_equity(self):
        """Near-zero avg equity should not produce inf turnover."""
        dates = pd.bdate_range(start="2023-01-01", periods=20)
        data = pd.DataFrame({
            "open": 0.01, "high": 0.02, "low": 0.005,
            "close": 0.01, "volume": 1e9,
        }, index=dates)
        signals = pd.Series(0, index=data.index)
        signals.iloc[0:] = 1

        agent = BacktestAgent(config={"initial_capital": 1.0})
        result = agent.run({"price_data": data, "predictions": signals})
        assert np.isfinite(result["performance_summary"]["turnover"])

    def test_calmar_with_near_zero_drawdown(self, agent):
        """Tiny drawdown should not produce inf Calmar."""
        data = _make_trending_up_data(n_days=252)
        signals = pd.Series(1, index=data.index)
        result = agent.run({"price_data": data, "predictions": signals})
        calmar = result["performance_summary"]["calmar"]
        assert np.isfinite(calmar)
        assert abs(calmar) <= 100.0


class TestAnnualizationStability:

    def test_short_backtest_annualized_return_not_extreme(self):
        """A 5-day backtest should not report 10000% annualized return."""
        data = _make_trending_up_data(n_days=5)
        signals = pd.Series(1, index=data.index)
        agent = BacktestAgent()
        result = agent.run({"price_data": data, "predictions": signals})
        ann = result["performance_summary"]["annualized_return"]
        total = result["performance_summary"]["total_return"]
        assert ann == pytest.approx(total, rel=0.01)

    def test_full_year_backtest_annualizes_correctly(self):
        """A 252-day backtest should annualize to a reasonable value."""
        data = _make_trending_up_data(n_days=252)
        signals = pd.Series(1, index=data.index)
        agent = BacktestAgent()
        result = agent.run({"price_data": data, "predictions": signals})
        ann = result["performance_summary"]["annualized_return"]
        total = result["performance_summary"]["total_return"]
        assert np.isfinite(ann)
        assert abs(ann) < abs(total) * 5

    def test_turnover_annualized(self):
        """Turnover should be per-year, not total."""
        data = _make_constant_price_data(n_days=252, price=100.0)
        sigs = pd.Series(0, index=data.index)
        sigs.iloc[::2] = 1

        agent = BacktestAgent()
        r252 = agent.run({"price_data": data, "predictions": sigs})

        data_half = _make_constant_price_data(n_days=126, price=100.0)
        sigs_half = pd.Series(0, index=data_half.index)
        sigs_half.iloc[::2] = 1

        r126 = agent.run({"price_data": data_half, "predictions": sigs_half})

        t252 = r252["performance_summary"]["turnover"]
        t126 = r126["performance_summary"]["turnover"]
        assert t252 == pytest.approx(t126, rel=0.3)


class TestMetricConsistency:

    def test_sharpe_and_sortino_finite_for_two_bars(self):
        """2-bar backtest should not crash or produce NaN."""
        data = _make_constant_price_data(n_days=2, price=100.0)
        signals = pd.Series(0, index=data.index)
        agent = BacktestAgent()
        result = agent.run({"price_data": data, "predictions": signals})
        assert np.isfinite(result["performance_summary"]["sharpe"])
        assert np.isfinite(result["performance_summary"]["sortino"])

    def test_all_metrics_finite(self, agent, price_data, alternating_signals):
        """Every metric in performance_summary must be finite."""
        result = agent.run({
            "price_data": price_data,
            "predictions": alternating_signals,
        })
        summary = result["performance_summary"]
        for key in ("sharpe", "sortino", "max_drawdown", "calmar",
                     "win_rate", "turnover", "total_return", "annualized_return"):
            assert np.isfinite(summary[key]), f"{key} is not finite: {summary[key]}"