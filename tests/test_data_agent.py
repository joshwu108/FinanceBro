"""Tests for DataAgent.

Validates:
  - BaseAgent contract is satisfied
  - OHLCV schema enforcement (columns, DatetimeIndex)
  - Temporal ordering (monotonically increasing)
  - No NaN / inf in cleaned output
  - Anomaly detection (>5% daily price jumps)
  - Non-positive price rejection
  - Survivorship bias reporting (failed symbols tracked)
  - Parquet round-trip persistence
  - No forward-filling (ffill forbidden)
  - Multi-symbol date alignment
  - Data quality report structure
  - Duplicate timestamp handling
  - OHLC bar invariant enforcement
  - Date input validation
  - Timezone-aware data normalization
  - All-symbols-fail scenario
  - Experiment log format
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agents.data_agent import DataAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_ohlcv(
    n_days: int = 200,
    seed: int = 42,
    start_price: float = 100.0,
    symbol: str = "TEST",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a DatetimeIndex.

    Guarantees OHLC invariants: high >= max(open, close), low <= min(open, close).
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)

    close = start_price + np.cumsum(rng.randn(n_days) * 0.5)
    # Ensure all prices are positive
    close = np.maximum(close, 1.0)
    open_ = close + rng.randn(n_days) * 0.3
    open_ = np.maximum(open_, 0.5)

    # Ensure high >= max(open, close) and low <= min(open, close)
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


def _make_yahoo_response(
    n_days: int = 200,
    seed: int = 42,
    symbol: str = "TEST",
) -> pd.DataFrame:
    """Simulate what yfinance ticker.history() returns (Date index, title-case cols)."""
    df = _make_ohlcv(n_days=n_days, seed=seed, symbol=symbol)
    # yfinance returns title-case columns and extra columns
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index.name = "Date"
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def agent(tmp_data_dir: Path, tmp_path: Path, monkeypatch) -> DataAgent:
    """Create a DataAgent with a temporary data directory.

    Redirects experiment logging to tmp_path so tests don't pollute
    the real experiments/ directory.
    """
    monkeypatch.setattr(
        "agents.data_agent.__file__",
        str(tmp_path / "agents" / "data_agent.py"),
    )
    return DataAgent(config={"data_dir": str(tmp_data_dir)})


# ── BaseAgent contract ───────────────────────────────────────────


class TestBaseAgentContract:
    def test_input_schema_exists(self, agent: DataAgent):
        schema = agent.input_schema
        assert "symbols" in schema
        assert "start_date" in schema
        assert "end_date" in schema

    def test_output_schema_exists(self, agent: DataAgent):
        schema = agent.output_schema
        assert "cleaned_data" in schema
        assert "data_quality_report" in schema

    def test_run_returns_expected_keys(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        assert "cleaned_data" in result
        assert "data_quality_report" in result

    def test_validate_passes_on_good_data(self, agent: DataAgent):
        cleaned = {"AAPL": _make_ohlcv(100)}
        quality = {
            "symbols_requested": ["AAPL"],
            "symbols_fetched": ["AAPL"],
            "symbols_failed": [],
            "survivorship_bias_warnings": [],
            "per_symbol": {
                "AAPL": {
                    "rows": 100,
                    "rows_dropped_nan": 0,
                    "date_range_start": "2023-01-01",
                    "date_range_end": "2023-06-01",
                    "source_timezone": None,
                    "missing_day_gaps": [],
                    "missing_business_days": [],
                    "anomalies": [],
                    "anomaly_count": 0,
                    "split_warning": None,
                },
            },
        }
        outputs = {"cleaned_data": cleaned, "data_quality_report": quality}
        assert agent.validate({"symbols": ["AAPL"]}, outputs) is True

    def test_log_metrics_runs(self, agent: DataAgent, tmp_path: Path):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        # run() calls log_metrics() automatically
        experiments_dir = tmp_path / "experiments"
        logs = list(experiments_dir.glob("data_agent_*.json"))
        assert len(logs) >= 1

        # Verify log content matches experiment template structure
        content = json.loads(logs[0].read_text())
        assert content["agent"] == "DataAgent"
        assert content["experiment_id"].startswith("data_")
        assert "date" in content
        assert "symbols" in content
        assert "metrics" in content
        assert content["stage"] == "data"


# ── Schema validation ────────────────────────────────────────────


class TestSchemaValidation:
    def test_output_has_correct_columns(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        expected_cols = {"open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_cols

    def test_output_has_datetime_index(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert isinstance(df.index, pd.DatetimeIndex)


# ── Temporal ordering ────────────────────────────────────────────


class TestTemporalOrdering:
    def test_index_monotonic_increasing(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert df.index.is_monotonic_increasing


# ── No NaN / inf ─────────────────────────────────────────────────


class TestNoMissingValues:
    def test_no_nan_in_output(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert not df.isna().any().any(), "Cleaned data contains NaN"

    def test_no_inf_in_output(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert np.isfinite(df.values).all(), "Cleaned data contains inf"


# ── Anomaly detection ────────────────────────────────────────────


class TestAnomalyDetection:
    def test_flags_large_price_jumps(self, agent: DataAgent):
        """Inject a >5% daily jump and verify the specific date is flagged."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        # Inject a 10% jump at row 50
        prev_close = yahoo_data.iloc[49, yahoo_data.columns.get_loc("Close")]
        new_close = prev_close * 1.10
        yahoo_data.iloc[50, yahoo_data.columns.get_loc("Close")] = new_close

        # Ensure OHLC invariants hold after injection
        cur_high = yahoo_data.iloc[50, yahoo_data.columns.get_loc("High")]
        if cur_high < new_close:
            yahoo_data.iloc[50, yahoo_data.columns.get_loc("High")] = new_close + 0.1

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        report = result["data_quality_report"]
        anomalies = report["per_symbol"]["AAPL"]["anomalies"]
        assert len(anomalies) > 0, "Should flag >5% price jump"

        # Verify the specific injected date is in the anomaly list
        anomaly_dates = [a["date"] for a in anomalies]
        injected_date = str(yahoo_data.index[50].date())
        assert injected_date in anomaly_dates, (
            f"Expected {injected_date} in anomalies, got {anomaly_dates}"
        )


# ── Non-positive price rejection ─────────────────────────────────


class TestNonPositivePrices:
    def test_rejects_negative_prices(self, agent: DataAgent):
        """Data with negative close prices should fail the pipeline."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        yahoo_data.iloc[10, yahoo_data.columns.get_loc("Close")] = -5.0

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            with pytest.raises(ValueError, match="All symbols failed"):
                agent.run({
                    "symbols": ["AAPL"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                })

    def test_rejects_zero_prices(self, agent: DataAgent):
        """Data with zero close prices should fail the pipeline."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        yahoo_data.iloc[10, yahoo_data.columns.get_loc("Close")] = 0.0

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            with pytest.raises(ValueError, match="All symbols failed"):
                agent.run({
                    "symbols": ["AAPL"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                })


# ── Survivorship bias ────────────────────────────────────────────


class TestSurvivorshipBias:
    def test_failed_symbols_tracked(self, agent: DataAgent):
        """Symbols that return no data should appear in symbols_failed."""
        good_data = _make_yahoo_response(n_days=100, seed=42)

        def side_effect(symbol):
            mock = MagicMock()
            if symbol == "DELISTED":
                mock.history.return_value = pd.DataFrame()
            else:
                mock.history.return_value = good_data
            return mock

        with patch("agents.data_agent.yf") as mock_yf:
            mock_yf.Ticker.side_effect = side_effect

            result = agent.run({
                "symbols": ["AAPL", "DELISTED"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        report = result["data_quality_report"]
        assert "DELISTED" in report["symbols_failed"]
        assert "AAPL" in report["symbols_fetched"]


# ── Parquet persistence ──────────────────────────────────────────


class TestParquetPersistence:
    def test_saves_parquet_files(self, agent: DataAgent, tmp_data_dir: Path):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        parquet_file = tmp_data_dir / "AAPL.parquet"
        assert parquet_file.exists(), "Parquet file not saved"

    def test_parquet_round_trip(self, agent: DataAgent, tmp_data_dir: Path):
        """Data read back from parquet should match the output."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        original = result["cleaned_data"]["AAPL"]
        loaded = pd.read_parquet(tmp_data_dir / "AAPL.parquet")
        # Parquet does not preserve index freq — compare values only
        pd.testing.assert_frame_equal(original, loaded, check_freq=False)


# ── No forward-filling ───────────────────────────────────────────


class TestNoForwardFill:
    def test_nan_rows_dropped_not_filled(self, agent: DataAgent):
        """Rows with NaN should be dropped, not forward-filled."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        # Inject NaN at row 50
        yahoo_data.iloc[50, yahoo_data.columns.get_loc("Close")] = np.nan

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        # Should have fewer rows than input (the NaN row was dropped)
        assert len(df) < 100
        assert not df.isna().any().any()


# ── Multi-symbol alignment ───────────────────────────────────────


class TestMultiSymbolAlignment:
    def test_common_date_index(self, agent: DataAgent):
        """Multiple symbols should share an identical date index."""
        data_a = _make_yahoo_response(n_days=100, seed=42)
        data_b = _make_yahoo_response(n_days=100, seed=99)

        def side_effect(symbol):
            mock = MagicMock()
            if symbol == "AAPL":
                mock.history.return_value = data_a
            else:
                mock.history.return_value = data_b
            return mock

        with patch("agents.data_agent.yf") as mock_yf:
            mock_yf.Ticker.side_effect = side_effect

            result = agent.run({
                "symbols": ["AAPL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        idx_a = result["cleaned_data"]["AAPL"].index
        idx_b = result["cleaned_data"]["MSFT"].index
        assert idx_a.equals(idx_b)


# ── Data quality report structure ────────────────────────────────


class TestQualityReport:
    def test_report_has_required_fields(self, agent: DataAgent):
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        report = result["data_quality_report"]
        assert "symbols_requested" in report
        assert "symbols_fetched" in report
        assert "symbols_failed" in report
        assert "per_symbol" in report
        assert "run_id" in report
        assert "survivorship_bias_warnings" in report

        sym_report = report["per_symbol"]["AAPL"]
        assert "rows" in sym_report
        assert "rows_dropped_nan" in sym_report
        assert "date_range_start" in sym_report
        assert "date_range_end" in sym_report
        assert "source_timezone" in sym_report
        assert "missing_day_gaps" in sym_report
        assert "missing_business_days" in sym_report
        assert "anomalies" in sym_report
        assert "anomaly_count" in sym_report
        assert "split_warning" in sym_report


# ── Input validation ─────────────────────────────────────────────


class TestInputValidation:
    def test_missing_symbols(self, agent: DataAgent):
        with pytest.raises(ValueError, match="symbols"):
            agent.run({"start_date": "2023-01-01", "end_date": "2023-12-31"})

    def test_empty_symbols(self, agent: DataAgent):
        with pytest.raises(ValueError, match="symbols"):
            agent.run({"symbols": [], "start_date": "2023-01-01", "end_date": "2023-12-31"})


# ── Duplicate timestamp handling ─────────────────────────────────


class TestDuplicateTimestamps:
    def test_normalize_deduplicates(self, agent: DataAgent):
        """Duplicate timestamps should be deduplicated during normalization."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        # Create a duplicate timestamp by overwriting row 11 index with row 10's
        new_index = yahoo_data.index.tolist()
        new_index[11] = new_index[10]
        yahoo_data.index = pd.DatetimeIndex(new_index)

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert df.index.is_unique, "Output should have unique timestamps"
        assert len(df) == 99  # One duplicate removed

    def test_validate_rejects_duplicates(self, agent: DataAgent):
        """validate() should catch duplicate timestamps as defense-in-depth."""
        df = _make_ohlcv(50)
        dup_row = df.iloc[[10]]
        df_with_dup = pd.concat([df, dup_row]).sort_index()

        outputs = {
            "cleaned_data": {"TEST": df_with_dup},
            "data_quality_report": {
                "symbols_requested": ["TEST"],
                "symbols_fetched": ["TEST"],
                "symbols_failed": [],
                "per_symbol": {},
            },
        }
        with pytest.raises(ValueError, match="duplicate timestamps"):
            agent.validate({"symbols": ["TEST"]}, outputs)


# ── All symbols fail ─────────────────────────────────────────────


class TestAllSymbolsFail:
    def test_raises_when_all_symbols_fail(self, agent: DataAgent):
        """Pipeline should raise when no symbols have valid data."""
        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.Ticker.return_value = mock_ticker

            with pytest.raises(ValueError, match="All symbols failed"):
                agent.run({
                    "symbols": ["BAD1", "BAD2"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                })


# ── Timezone handling ────────────────────────────────────────────


class TestTimezoneHandling:
    def test_timezone_aware_input_normalized(self, agent: DataAgent):
        """yfinance data with timezone-aware index should be normalized to naive."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        yahoo_data.index = yahoo_data.index.tz_localize("US/Eastern")

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            result = agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            })

        df = result["cleaned_data"]["AAPL"]
        assert df.index.tz is None, "Index should be timezone-naive after normalization"

        report = result["data_quality_report"]["per_symbol"]["AAPL"]
        assert report["source_timezone"] is not None


# ── OHLC bar invariants ──────────────────────────────────────────


class TestOHLCInvariants:
    def test_rejects_high_less_than_low(self, agent: DataAgent):
        """OHLC violation (high < low) should be caught by validate."""
        yahoo_data = _make_yahoo_response(n_days=100, seed=42)
        # Swap High and Low at row 10 to create violation
        orig_high = yahoo_data.iloc[10, yahoo_data.columns.get_loc("High")]
        orig_low = yahoo_data.iloc[10, yahoo_data.columns.get_loc("Low")]
        yahoo_data.iloc[10, yahoo_data.columns.get_loc("High")] = orig_low
        yahoo_data.iloc[10, yahoo_data.columns.get_loc("Low")] = orig_high

        with patch("agents.data_agent.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = yahoo_data
            mock_yf.Ticker.return_value = mock_ticker

            with pytest.raises(ValueError, match="high < low"):
                agent.run({
                    "symbols": ["AAPL"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                })


# ── Date input validation ────────────────────────────────────────


class TestDateValidation:
    def test_invalid_start_date_format(self, agent: DataAgent):
        with pytest.raises(ValueError, match="start_date must be ISO format"):
            agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023/01/01",
                "end_date": "2023-12-31",
            })

    def test_invalid_end_date_format(self, agent: DataAgent):
        with pytest.raises(ValueError, match="end_date must be ISO format"):
            agent.run({
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "not-a-date",
            })

    def test_start_after_end(self, agent: DataAgent):
        with pytest.raises(ValueError, match="start_date.*before.*end_date"):
            agent.run({
                "symbols": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2023-12-31",
            })
