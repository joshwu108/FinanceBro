"""Tests for WalkForwardAgent.

Validates:
  - BaseAgent contract is satisfied (run, validate, log_metrics, schemas)
  - Expanding window walk-forward validation (train grows, test slides)
  - No temporal leakage: each fold's test data is strictly after train data
  - Per-fold metrics include Sharpe, drawdown, model metrics
  - Aggregated metrics across all folds
  - Fold results list with correct structure
  - Input validation (missing keys, wrong types, insufficient data)
  - Edge cases: minimum data, single fold, all-same target
  - Experiment logging to /experiments/
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agents.walkforward_agent import WalkForwardAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_feature_matrix(
    n_days: int = 504,
    n_features: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic feature matrix with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2021-01-01", periods=n_days)
    data = rng.randn(n_days, n_features)
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=dates, columns=columns)


def _make_target(
    index: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic binary target aligned with index."""
    rng = np.random.RandomState(seed)
    return pd.Series(
        rng.randint(0, 2, len(index)),
        index=index,
        name="target",
    )


def _make_ohlcv(
    index: pd.DatetimeIndex,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data aligned with a DatetimeIndex."""
    rng = np.random.RandomState(seed)
    n_days = len(index)

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
        index=index,
    )


@pytest.fixture
def sample_inputs():
    """Standard inputs for WalkForwardAgent tests."""
    X = _make_feature_matrix(n_days=504, n_features=5, seed=42)
    y = _make_target(X.index, seed=42)
    price_data = _make_ohlcv(X.index, seed=42)
    return {
        "feature_matrix": X,
        "target": y,
        "price_data": price_data,
    }


@pytest.fixture
def agent():
    return WalkForwardAgent()


@pytest.fixture
def configured_agent():
    """Agent with small folds for faster testing."""
    return WalkForwardAgent(config={
        "n_folds": 3,
        "min_train_size": 100,
        "test_size": 50,
        "model_config": {"model_type": "logistic_regression"},
    })


# ── BaseAgent contract ───────────────────────────────────────────


class TestBaseAgentContract:
    """WalkForwardAgent must satisfy the BaseAgent interface."""

    def test_has_run_method(self, agent):
        assert hasattr(agent, "run") and callable(agent.run)

    def test_has_validate_method(self, agent):
        assert hasattr(agent, "validate") and callable(agent.validate)

    def test_has_log_metrics_method(self, agent):
        assert hasattr(agent, "log_metrics") and callable(agent.log_metrics)

    def test_has_input_schema(self, agent):
        schema = agent.input_schema
        assert isinstance(schema, dict)
        assert "feature_matrix" in schema
        assert "target" in schema
        assert "price_data" in schema

    def test_has_output_schema(self, agent):
        schema = agent.output_schema
        assert isinstance(schema, dict)
        assert "fold_results" in schema
        assert "aggregated_metrics" in schema


# ── Core walk-forward validation ─────────────────────────────────


class TestExpandingWindowValidation:
    """Verify expanding window logic is correct."""

    def test_run_returns_expected_keys(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        assert "fold_results" in result
        assert "aggregated_metrics" in result
        assert "n_folds" in result

    def test_fold_count_matches_config(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        assert result["n_folds"] == 3
        assert len(result["fold_results"]) == 3

    def test_train_window_expands_across_folds(self, configured_agent, sample_inputs):
        """Each successive fold must have a larger or equal training set."""
        result = configured_agent.run(sample_inputs)
        folds = result["fold_results"]

        train_sizes = [f["split_info"]["train_size"] for f in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Fold {i} train_size ({train_sizes[i]}) < "
                f"fold {i-1} ({train_sizes[i-1]})"
            )

    def test_test_windows_do_not_overlap(self, configured_agent, sample_inputs):
        """Test windows across folds must not overlap."""
        result = configured_agent.run(sample_inputs)
        folds = result["fold_results"]

        for i in range(1, len(folds)):
            prev_test_end = pd.Timestamp(folds[i - 1]["split_info"]["test_end"])
            curr_test_start = pd.Timestamp(folds[i]["split_info"]["test_start"])
            assert curr_test_start > prev_test_end, (
                f"Fold {i} test_start ({curr_test_start}) overlaps "
                f"fold {i-1} test_end ({prev_test_end})"
            )


# ── No temporal leakage ──────────────────────────────────────────


class TestNoTemporalLeakage:
    """Each fold's test data must be strictly after its train data."""

    def test_test_start_after_train_end_per_fold(
        self, configured_agent, sample_inputs
    ):
        result = configured_agent.run(sample_inputs)
        for i, fold in enumerate(result["fold_results"]):
            train_end = pd.Timestamp(fold["split_info"]["train_end"])
            test_start = pd.Timestamp(fold["split_info"]["test_start"])
            assert test_start > train_end, (
                f"Fold {i}: test_start ({test_start}) <= "
                f"train_end ({train_end}) — temporal leak!"
            )

    def test_no_future_data_in_training(self, configured_agent, sample_inputs):
        """ModelAgent must not receive data beyond the fold's test_end."""
        from agents.model_agent import ModelAgent

        captured_inputs = []
        original_run = ModelAgent.run

        def capturing_run(self_inner, inputs):
            captured_inputs.append({
                "max_index": inputs["feature_matrix"].index.max(),
                "train_end_date": inputs.get("train_end_date"),
            })
            return original_run(self_inner, inputs)

        with patch.object(ModelAgent, "run", capturing_run):
            result = configured_agent.run(sample_inputs)

        for i, (fold, captured) in enumerate(
            zip(result["fold_results"], captured_inputs)
        ):
            train_end = pd.Timestamp(fold["split_info"]["train_end"])
            test_end = pd.Timestamp(fold["split_info"]["test_end"])

            # ModelAgent receives combined train+test, but must split at train_end_date
            assert captured["train_end_date"] == str(train_end.date()), (
                f"Fold {i}: train_end_date not set correctly"
            )
            # Combined data should not extend beyond the fold's test window
            assert captured["max_index"] <= test_end, (
                f"Fold {i}: ModelAgent received data beyond test_end ({test_end})"
            )


# ── Per-fold metrics ─────────────────────────────────────────────


class TestPerFoldMetrics:
    """Each fold result must contain model and backtest metrics."""

    def test_fold_has_model_metrics(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        for fold in result["fold_results"]:
            assert "model_metrics" in fold
            metrics = fold["model_metrics"]
            assert "accuracy" in metrics
            assert "f1" in metrics

    def test_fold_has_backtest_metrics(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        for fold in result["fold_results"]:
            assert "backtest_metrics" in fold
            bt = fold["backtest_metrics"]
            assert "sharpe" in bt
            assert "max_drawdown" in bt
            assert "total_return" in bt

    def test_fold_has_split_info(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        for fold in result["fold_results"]:
            assert "split_info" in fold
            info = fold["split_info"]
            assert "train_size" in info
            assert "test_size" in info
            assert "train_start" in info
            assert "train_end" in info
            assert "test_start" in info
            assert "test_end" in info

    def test_fold_has_fold_index(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        for i, fold in enumerate(result["fold_results"]):
            assert fold["fold_index"] == i


# ── Aggregated metrics ───────────────────────────────────────────


class TestAggregatedMetrics:
    """Aggregated metrics must summarize all folds."""

    def test_aggregated_has_sharpe(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        agg = result["aggregated_metrics"]
        assert "mean_sharpe" in agg
        assert "std_sharpe" in agg

    def test_aggregated_has_drawdown(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        agg = result["aggregated_metrics"]
        assert "mean_max_drawdown" in agg
        assert "worst_max_drawdown" in agg

    def test_aggregated_has_return(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        agg = result["aggregated_metrics"]
        assert "mean_total_return" in agg

    def test_aggregated_has_model_metrics(self, configured_agent, sample_inputs):
        result = configured_agent.run(sample_inputs)
        agg = result["aggregated_metrics"]
        assert "mean_accuracy" in agg
        assert "mean_f1" in agg

    def test_aggregated_sharpe_is_mean_of_folds(
        self, configured_agent, sample_inputs
    ):
        result = configured_agent.run(sample_inputs)
        fold_sharpes = [
            f["backtest_metrics"]["sharpe"] for f in result["fold_results"]
        ]
        expected_mean = np.mean(fold_sharpes)
        assert abs(result["aggregated_metrics"]["mean_sharpe"] - expected_mean) < 1e-4

    def test_aggregated_max_drawdown_is_worst_across_folds(
        self, configured_agent, sample_inputs
    ):
        result = configured_agent.run(sample_inputs)
        fold_dds = [
            f["backtest_metrics"]["max_drawdown"] for f in result["fold_results"]
        ]
        expected_worst = min(fold_dds)  # max_drawdown is negative
        assert (
            abs(
                result["aggregated_metrics"]["worst_max_drawdown"] - expected_worst
            )
            < 1e-4
        )


# ── Input validation ─────────────────────────────────────────────


class TestInputValidation:
    """Agent must reject invalid inputs with clear errors."""

    def test_missing_feature_matrix(self, agent, sample_inputs):
        del sample_inputs["feature_matrix"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(sample_inputs)

    def test_missing_target(self, agent, sample_inputs):
        del sample_inputs["target"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(sample_inputs)

    def test_missing_price_data(self, agent, sample_inputs):
        del sample_inputs["price_data"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(sample_inputs)

    def test_feature_matrix_wrong_type(self, agent, sample_inputs):
        sample_inputs["feature_matrix"] = "not a dataframe"
        with pytest.raises((TypeError, ValueError)):
            agent.run(sample_inputs)

    def test_target_wrong_type(self, agent, sample_inputs):
        sample_inputs["target"] = [1, 0, 1]
        with pytest.raises((TypeError, ValueError)):
            agent.run(sample_inputs)

    def test_price_data_wrong_type(self, agent, sample_inputs):
        sample_inputs["price_data"] = "not a dataframe"
        with pytest.raises((TypeError, ValueError)):
            agent.run(sample_inputs)

    def test_insufficient_data_for_folds(self, agent):
        """Too few rows for even a single fold should raise."""
        X = _make_feature_matrix(n_days=10, n_features=3)
        y = _make_target(X.index)
        price_data = _make_ohlcv(X.index)
        with pytest.raises(ValueError, match="[Ii]nsufficient|[Nn]ot enough|too (few|small)"):
            agent.run({
                "feature_matrix": X,
                "target": y,
                "price_data": price_data,
            })

    def test_misaligned_indices(self, agent):
        """Feature matrix and target with different indices should raise."""
        X = _make_feature_matrix(n_days=200)
        y = _make_target(
            pd.bdate_range(start="2025-01-01", periods=200), seed=42
        )
        price_data = _make_ohlcv(X.index)
        with pytest.raises(ValueError):
            agent.run({
                "feature_matrix": X,
                "target": y,
                "price_data": price_data,
            })

    def test_price_data_missing_columns(self, agent, sample_inputs):
        """Price data missing OHLCV columns should raise."""
        sample_inputs["price_data"] = sample_inputs["price_data"][["close"]]
        with pytest.raises(ValueError):
            agent.run(sample_inputs)

    def test_misaligned_price_data_index(self):
        """price_data with different index than feature_matrix should raise."""
        X = _make_feature_matrix(n_days=252, n_features=5, seed=42)
        y = _make_target(X.index, seed=42)
        # price_data with completely different dates
        different_index = pd.bdate_range(start="2025-01-01", periods=252)
        price_data = _make_ohlcv(different_index, seed=42)

        agent = WalkForwardAgent(config={
            "n_folds": 1,
            "min_train_size": 150,
            "test_size": 50,
            "model_config": {"model_type": "logistic_regression"},
        })
        with pytest.raises(ValueError, match="signals index does not match"):
            agent.run({
                "feature_matrix": X,
                "target": y,
                "price_data": price_data,
            })


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Walk-forward validation with edge-case scenarios."""

    def test_single_fold(self):
        """A single fold should still work."""
        agent = WalkForwardAgent(config={
            "n_folds": 1,
            "min_train_size": 150,
            "test_size": 50,
            "model_config": {"model_type": "logistic_regression"},
        })
        X = _make_feature_matrix(n_days=252, n_features=5)
        y = _make_target(X.index)
        price_data = _make_ohlcv(X.index)

        result = agent.run({
            "feature_matrix": X,
            "target": y,
            "price_data": price_data,
        })
        assert result["n_folds"] == 1
        assert len(result["fold_results"]) == 1

    def test_custom_model_config_passed_through(self, sample_inputs):
        """model_config should be forwarded to ModelAgent."""
        agent = WalkForwardAgent(config={
            "n_folds": 2,
            "min_train_size": 100,
            "test_size": 50,
            "model_config": {"model_type": "logistic_regression"},
        })
        result = agent.run(sample_inputs)
        for fold in result["fold_results"]:
            assert fold["model_type"] == "logistic_regression"


# ── Validate method ──────────────────────────────────────────────


class TestValidateMethod:
    """The validate() method checks output structure."""

    def test_validate_passes_on_good_output(self, configured_agent, sample_inputs):
        outputs = configured_agent.run(sample_inputs)
        assert configured_agent.validate(sample_inputs, outputs) is True

    def test_validate_rejects_empty_fold_results(self, configured_agent, sample_inputs):
        outputs = configured_agent.run(sample_inputs)
        outputs["fold_results"] = []
        with pytest.raises(ValueError):
            configured_agent.validate(sample_inputs, outputs)

    def test_validate_rejects_missing_aggregated_metrics(
        self, configured_agent, sample_inputs
    ):
        outputs = configured_agent.run(sample_inputs)
        del outputs["aggregated_metrics"]
        with pytest.raises((ValueError, KeyError)):
            configured_agent.validate(sample_inputs, outputs)


# ── Experiment logging ───────────────────────────────────────────


class TestExperimentLogging:
    """log_metrics must persist to /experiments/."""

    def test_log_metrics_creates_file(self, configured_agent, sample_inputs, tmp_path):
        configured_agent.run(sample_inputs)

        with patch.object(
            type(configured_agent),
            "_experiments_dir",
            new_callable=lambda: property(lambda self: tmp_path),
        ):
            configured_agent.log_metrics()

        log_files = list(tmp_path.glob("walkforward_agent_*.json"))
        assert len(log_files) == 1

        content = json.loads(log_files[0].read_text())
        assert content["agent"] == "WalkForwardAgent"
        assert "metrics" in content
        assert "n_folds" in content["metrics"]

    def test_log_metrics_without_run_is_safe(self, agent):
        """Calling log_metrics before run should not crash."""
        agent.log_metrics()  # Should log a warning, not raise
