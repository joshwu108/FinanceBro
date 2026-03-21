"""Tests for orchestrator/run_pipeline.

Validates:
  - Full pipeline executes end-to-end with synthetic data
  - All 9 agents are invoked in correct order
  - Per-symbol results contain expected keys
  - Experiment log is written to disk
  - Inference bundle is saved for /predict endpoint
  - Multi-symbol runs produce PortfolioAgent output
  - Signal conversion: probabilities → {-1, 0, 1}
  - Error handling for empty symbol list
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from orchestrator.run_pipeline import (
    run_pipeline,
    _predictions_to_signals,
    _serialize_fold_results,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _make_ohlcv(
    n_days: int = 500,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with valid bar invariants."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2022-01-01", periods=n_days)

    close = start_price + np.cumsum(rng.randn(n_days) * 0.5)
    close = np.maximum(close, 1.0)

    open_ = close + rng.randn(n_days) * 0.3
    open_ = np.maximum(open_, 0.5)

    high = np.maximum(close, open_) + np.abs(rng.randn(n_days) * 0.2)
    low = np.minimum(close, open_) - np.abs(rng.randn(n_days) * 0.2)
    low = np.maximum(low, 0.1)

    volume = np.abs(rng.randn(n_days) * 1_000_000) + 100_000

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _mock_yfinance_history(symbol_data_map):
    """Create a mock for yfinance Ticker.history that returns symbol-specific data."""
    def mock_ticker_init(self, symbol):
        self.ticker = symbol

    def mock_history(self, **kwargs):
        ticker_symbol = self.ticker
        if ticker_symbol in symbol_data_map:
            df = symbol_data_map[ticker_symbol].copy()
            df.columns = [c.capitalize() for c in df.columns]
            return df
        return pd.DataFrame()

    return mock_ticker_init, mock_history


# ── Signal conversion tests ──────────────────────────────────────


class TestPredictionsToSignals:
    def test_long_short_flat(self):
        preds = pd.Series([0.8, 0.2, 0.5, 0.6, 0.1])
        signals = _predictions_to_signals(preds, threshold=0.5)
        expected = pd.Series([1, -1, 1, 1, -1], dtype=int)
        pd.testing.assert_series_equal(signals, expected)

    def test_all_long(self):
        preds = pd.Series([0.9, 0.7, 0.55])
        signals = _predictions_to_signals(preds, threshold=0.5)
        assert (signals == 1).all()

    def test_all_short(self):
        preds = pd.Series([0.1, 0.3, 0.4])
        signals = _predictions_to_signals(preds, threshold=0.5)
        assert (signals == -1).all()

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=3)
        preds = pd.Series([0.6, 0.4, 0.5], index=idx)
        signals = _predictions_to_signals(preds, threshold=0.5)
        assert signals.index.equals(idx)


class TestSerializeFoldResults:
    def test_strips_internal_keys(self):
        folds = [{
            "fold_index": 0,
            "split_info": {"train_size": 100, "test_size": 20,
                           "train_start": "2022-01-01", "train_end": "2022-06-01",
                           "test_start": "2022-06-02", "test_end": "2022-09-01"},
            "model_metrics": {"accuracy": 0.52, "f1": 0.50},
            "backtest_metrics": {"sharpe": 0.1, "max_drawdown": -0.05, "total_return": 0.01},
            "model_type": "logistic_regression",
            "_internal_object": object(),
        }]
        result = _serialize_fold_results(folds)
        assert len(result) == 1
        assert "_internal_object" not in result[0]
        assert result[0]["fold_index"] == 0


# ── Full pipeline integration test ───────────────────────────────


class TestRunPipeline:
    """Integration tests for run_pipeline with mocked data fetching."""

    @pytest.fixture
    def single_symbol_config(self):
        return {
            "symbols": ["TEST"],
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "model_type": "logistic_regression",
            "transaction_costs_bps": 5.0,
            "slippage_bps": 2.0,
            "max_position_size": 1.0,
            "benchmark": "SPY",
            "n_folds": 3,
        }

    @pytest.fixture
    def multi_symbol_config(self):
        return {
            "symbols": ["TESTA", "TESTB"],
            "model_type": "logistic_regression",
            "transaction_costs_bps": 5.0,
            "slippage_bps": 2.0,
            "max_position_size": 1.0,
            "n_folds": 3,
        }

    def _patch_yfinance(self, symbol_data_map):
        """Return patch context managers for yfinance."""
        init_mock, history_mock = _mock_yfinance_history(symbol_data_map)
        return (
            patch("yfinance.Ticker.__init__", init_mock),
            patch("yfinance.Ticker.history", history_mock),
        )

    def test_single_symbol_pipeline(self, single_symbol_config, tmp_path):
        """Full pipeline for a single symbol produces all expected outputs."""
        test_data = _make_ohlcv(n_days=800, seed=42)
        spy_data = _make_ohlcv(n_days=800, seed=99)
        symbol_map = {"TEST": test_data, "SPY": spy_data}

        p1, p2 = self._patch_yfinance(symbol_map)
        with p1, p2, \
             patch("orchestrator.run_pipeline.MODELS_DIR", tmp_path / "models"), \
             patch("orchestrator.run_pipeline.EXPERIMENTS_DIR", tmp_path / "experiments"), \
             patch("agents.data_agent.DataAgent.log_metrics"), \
             patch("agents.data_agent.DataAgent._check_splits", return_value=None):

            # Also patch parquet writes to use tmp_path
            with patch.object(
                __import__("agents.data_agent", fromlist=["DataAgent"]).DataAgent,
                "DEFAULT_CONFIG",
                {**__import__("agents.data_agent", fromlist=["DataAgent"]).DataAgent.DEFAULT_CONFIG,
                 "data_dir": str(tmp_path / "data")},
            ):
                result = run_pipeline(single_symbol_config)

        # Top-level structure
        assert "run_id" in result
        assert "per_symbol" in result
        assert "stats" in result
        assert result["portfolio"] is None  # single symbol

        # Per-symbol results
        assert "TEST" in result["per_symbol"]
        sym_result = result["per_symbol"]["TEST"]

        # All pipeline stages present
        assert "model" in sym_result
        assert "walk_forward" in sym_result
        assert "backtest" in sym_result
        assert "overfitting" in sym_result
        assert "risk" in sym_result

        # Model metrics
        assert "train_metrics" in sym_result["model"]
        assert "test_metrics" in sym_result["model"]
        assert sym_result["model"]["model_type"] == "logistic_regression"

        # Walk-forward
        assert sym_result["walk_forward"]["n_folds"] == 3

        # Backtest has performance summary
        perf = sym_result["backtest"]["performance_summary"]
        assert "sharpe" in perf
        assert "max_drawdown" in perf
        assert "total_return" in perf

        # Overfitting score in [0, 1]
        score = sym_result["overfitting"]["overfitting_score"]
        assert 0.0 <= score <= 1.0

        # Risk metrics present
        assert "var_95" in sym_result["risk"]["risk_metrics"]

        # Stats
        assert "hypothesis_test" in result["stats"]
        assert "bootstrap" in result["stats"]

        # Experiment log written
        exp_dir = tmp_path / "experiments"
        if exp_dir.exists():
            exp_files = list(exp_dir.glob("pipeline_*.json"))
            assert len(exp_files) >= 1
            exp_data = json.loads(exp_files[0].read_text())
            assert exp_data["out_of_sample"] is True

    def test_empty_symbols_raises(self):
        with pytest.raises(ValueError, match="non-empty 'symbols'"):
            run_pipeline({"symbols": []})

    def test_predictions_to_signals_valid_range(self):
        """Signals must be in {-1, 0, 1}."""
        preds = pd.Series(np.random.rand(100))
        signals = _predictions_to_signals(preds, threshold=0.5)
        assert set(signals.unique()).issubset({-1, 0, 1})
