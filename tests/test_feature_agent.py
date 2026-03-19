"""Tests for FeatureAgent.

Validates:
  - Look-ahead bias is fixed (features at time t cannot see t+1)
  - Target construction is correct
  - NaN handling uses dropna, not ffill/bfill
  - Feature/target separation is enforced
  - BaseAgent contract is satisfied
  - Inf handling works
  - Idempotency
"""

import numpy as np
import pandas as pd
import pytest

from agents.feature_agent import FeatureAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_ohlcv(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    close = 100.0 + np.cumsum(rng.randn(n_days) * 0.5)
    high = close + rng.uniform(0.1, 2.0, n_days)
    low = close - rng.uniform(0.1, 2.0, n_days)
    open_ = close + rng.randn(n_days) * 0.3
    volume = rng.randint(1_000_000, 10_000_000, n_days).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def agent() -> FeatureAgent:
    return FeatureAgent()


@pytest.fixture
def run_result(agent: FeatureAgent, ohlcv_data: pd.DataFrame):
    return agent.run({"cleaned_data": ohlcv_data})


# ── BaseAgent contract ───────────────────────────────────────────


class TestBaseAgentContract:
    def test_input_schema_exists(self, agent: FeatureAgent):
        schema = agent.input_schema
        assert "cleaned_data" in schema

    def test_output_schema_exists(self, agent: FeatureAgent):
        schema = agent.output_schema
        assert "feature_matrix" in schema
        assert "target" in schema
        assert "feature_metadata" in schema

    def test_run_returns_expected_keys(self, run_result):
        assert "feature_matrix" in run_result
        assert "target" in run_result
        assert "feature_metadata" in run_result
        assert "all_targets" in run_result

    def test_validate_passes(self, agent: FeatureAgent, ohlcv_data: pd.DataFrame):
        result = agent.run({"cleaned_data": ohlcv_data})
        assert agent.validate({"cleaned_data": ohlcv_data}, result) is True

    def test_log_metrics_runs(self, agent: FeatureAgent, ohlcv_data: pd.DataFrame):
        agent.run({"cleaned_data": ohlcv_data})
        agent.log_metrics()
        from pathlib import Path
        experiments_dir = Path(__file__).parent.parent / "experiments"
        logs = list(experiments_dir.glob("feature_agent_*.json"))
        assert len(logs) >= 1
        # Clean up the log file we just created
        logs[-1].unlink()


# ── Look-ahead bias ─────────────────────────────────────────────


class TestNoLookAheadBias:
    """Features at time t must not use data from t+1 or later."""

    def test_features_only_use_past_data(self, ohlcv_data: pd.DataFrame):
        """Truncate the last 10 rows and verify that features for all
        earlier rows are identical — proving no future data leaked."""
        full_agent = FeatureAgent()
        truncated_agent = FeatureAgent()

        full = full_agent.run({"cleaned_data": ohlcv_data})
        truncated = truncated_agent.run({"cleaned_data": ohlcv_data.iloc[:-10]})

        # Overlapping dates
        common_idx = full["feature_matrix"].index.intersection(
            truncated["feature_matrix"].index
        )
        assert len(common_idx) > 0

        full_sub = full["feature_matrix"].loc[common_idx]
        trunc_sub = truncated["feature_matrix"].loc[common_idx]

        pd.testing.assert_frame_equal(full_sub, trunc_sub)

    def test_no_shift_negative_in_features(self):
        """Verify feature columns don't contain target keywords."""
        agent = FeatureAgent()
        result = agent.run({"cleaned_data": _make_ohlcv()})
        target_kw = ("fwd_return", "fwd_direction", "target")
        for col in result["feature_matrix"].columns:
            for kw in target_kw:
                assert kw not in col, f"Leaked target column: {col}"


# ── Target construction ──────────────────────────────────────────


class TestTargetConstruction:
    def test_forward_return_manually(self, ohlcv_data: pd.DataFrame):
        """Verify fwd_return_5d matches manual calculation."""
        agent = FeatureAgent(config={"target_horizons": [5], "default_target": "fwd_direction_5d"})
        result = agent.run({"cleaned_data": ohlcv_data})
        all_targets = result["all_targets"]

        # Manual: (close[t+5] - close[t]) / close[t]
        close = ohlcv_data["close"]
        manual_fwd = (close.shift(-5) - close) / close

        # Compare on overlapping valid indices
        common = all_targets.index.intersection(manual_fwd.dropna().index)
        actual = all_targets.loc[common, "fwd_return_5d"]
        expected = manual_fwd.loc[common]

        pd.testing.assert_series_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_direction_matches_return_sign(self, run_result):
        """fwd_direction should be 1 when fwd_return > 0, else 0."""
        targets = run_result["all_targets"]
        for col in targets.columns:
            if col.startswith("fwd_direction_"):
                horizon = col.replace("fwd_direction_", "").replace("d", "")
                return_col = f"fwd_return_{horizon}d"
                expected = (targets[return_col] > 0).astype(float)
                pd.testing.assert_series_equal(
                    targets[col], expected, check_names=False
                )

    def test_direction_is_nan_at_boundary(self):
        """The last k rows of fwd_direction_kd should be NaN (no future data),
        not silently converted to 0."""
        agent = FeatureAgent(config={"target_horizons": [5]})
        data = _make_ohlcv(300)
        df = agent._ensure_datetime_index(data.copy())
        targets = agent._compute_targets(df, {**agent._config, "target_horizons": [5]})
        # Last 5 rows must be NaN — NOT 0
        assert targets["fwd_direction_5d"].iloc[-5:].isna().all(), (
            "Last 5 rows of fwd_direction_5d should be NaN (no future data)"
        )
        # And the rows before that should be valid (0 or 1)
        valid_direction = targets["fwd_direction_5d"].iloc[:-5].dropna()
        assert valid_direction.isin([0, 1]).all()


# ── NaN handling ─────────────────────────────────────────────────


class TestNaNHandling:
    def test_no_nan_in_feature_matrix(self, run_result):
        assert not run_result["feature_matrix"].isna().any().any()

    def test_no_nan_in_target(self, run_result):
        assert not run_result["target"].isna().any()

    def test_no_inf_in_feature_matrix(self, run_result):
        assert np.isfinite(run_result["feature_matrix"].values).all()

    def test_rows_dropped_not_filled(self, ohlcv_data: pd.DataFrame):
        """Output should have fewer rows than input (warmup rows dropped)."""
        agent = FeatureAgent()
        result = agent.run({"cleaned_data": ohlcv_data})
        assert len(result["feature_matrix"]) < len(ohlcv_data)


# ── Feature / target separation ──────────────────────────────────


class TestSeparation:
    def test_no_raw_ohlcv_in_features(self, run_result):
        raw = {"open", "high", "low", "close", "volume"}
        feature_cols = set(run_result["feature_matrix"].columns)
        assert not raw.intersection(feature_cols)

    def test_target_not_in_features(self, run_result):
        feature_cols = set(run_result["feature_matrix"].columns)
        target_cols = set(run_result["all_targets"].columns)
        assert not feature_cols.intersection(target_cols)

    def test_indices_aligned(self, run_result):
        assert run_result["feature_matrix"].index.equals(
            run_result["target"].index
        )


# ── Idempotency ──────────────────────────────────────────────────


class TestIdempotency:
    def test_two_runs_same_result(self, ohlcv_data: pd.DataFrame):
        agent = FeatureAgent()
        r1 = agent.run({"cleaned_data": ohlcv_data})
        r2 = agent.run({"cleaned_data": ohlcv_data})
        pd.testing.assert_frame_equal(r1["feature_matrix"], r2["feature_matrix"])
        pd.testing.assert_series_equal(r1["target"], r2["target"])


# ── Feature metadata ─────────────────────────────────────────────


class TestMetadata:
    def test_metadata_count_matches_columns(self, run_result):
        n_meta = len(run_result["feature_metadata"])
        n_cols = len(run_result["feature_matrix"].columns)
        assert n_meta == n_cols

    def test_every_feature_has_lookback(self, run_result):
        for m in run_result["feature_metadata"]:
            assert "lookback_window" in m
            assert isinstance(m["lookback_window"], int)


# ── Temporal ordering ────────────────────────────────────────────


class TestTemporalOrder:
    def test_index_monotonic_increasing(self, run_result):
        assert run_result["feature_matrix"].index.is_monotonic_increasing


# ── Config overrides ─────────────────────────────────────────────


class TestConfig:
    def test_disable_categories(self, ohlcv_data: pd.DataFrame):
        agent = FeatureAgent(config={
            "include_momentum": False,
            "include_volume": False,
        })
        result = agent.run({"cleaned_data": ohlcv_data})
        cols = set(result["feature_matrix"].columns)
        assert "rsi" not in cols
        assert "obv_roc" not in cols
        # Trend should still be present
        assert "close_sma20_ratio" in cols

    def test_custom_target_horizon(self, ohlcv_data: pd.DataFrame):
        agent = FeatureAgent(config={
            "target_horizons": [3, 10],
            "default_target": "fwd_direction_3d",
        })
        result = agent.run({"cleaned_data": ohlcv_data})
        assert "fwd_return_3d" in result["all_targets"].columns
        assert "fwd_return_10d" in result["all_targets"].columns


# ── Date column input (not just DatetimeIndex) ───────────────────


class TestDateColumnInput:
    def test_date_as_column_not_index(self, ohlcv_data: pd.DataFrame):
        """Agent should handle 'date' as a regular column."""
        df = ohlcv_data.copy()
        df["date"] = df.index
        df = df.reset_index(drop=True)
        agent = FeatureAgent()
        result = agent.run({"cleaned_data": df})
        assert isinstance(result["feature_matrix"].index, pd.DatetimeIndex)


# ── Validation catches bad data ──────────────────────────────────


class TestValidationRejects:
    def test_rejects_nan_in_features(self, agent: FeatureAgent):
        X = pd.DataFrame({"a": [1.0, np.nan]}, index=pd.bdate_range("2020-01-01", periods=2))
        y = pd.Series([1, 0], index=X.index)
        with pytest.raises(ValueError, match="NaN"):
            agent.validate({}, {"feature_matrix": X, "target": y, "feature_metadata": []})

    def test_rejects_target_col_in_features(self, agent: FeatureAgent):
        idx = pd.bdate_range("2020-01-01", periods=2)
        X = pd.DataFrame({"fwd_return_5d": [0.01, 0.02]}, index=idx)
        y = pd.Series([1, 0], index=idx)
        with pytest.raises(ValueError, match="Target column"):
            agent.validate({}, {"feature_matrix": X, "target": y, "feature_metadata": []})

    def test_rejects_raw_ohlcv_in_features(self, agent: FeatureAgent):
        idx = pd.bdate_range("2020-01-01", periods=2)
        X = pd.DataFrame({"close": [100.0, 101.0]}, index=idx)
        y = pd.Series([1, 0], index=idx)
        with pytest.raises(ValueError, match="Raw OHLCV"):
            agent.validate({}, {"feature_matrix": X, "target": y, "feature_metadata": []})


# ── Input validation ─────────────────────────────────────────────


class TestInputValidation:
    def test_missing_cleaned_data(self, agent: FeatureAgent):
        with pytest.raises(ValueError, match="cleaned_data"):
            agent.run({})

    def test_wrong_type(self, agent: FeatureAgent):
        with pytest.raises(TypeError, match="pd.DataFrame"):
            agent.run({"cleaned_data": "not a dataframe"})

    def test_missing_columns(self, agent: FeatureAgent):
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            agent.run({"cleaned_data": df})


# ── Config mutation ──────────────────────────────────────────────


class TestConfigNotMutated:
    def test_run_with_override_does_not_mutate_config(self, ohlcv_data: pd.DataFrame):
        agent = FeatureAgent()
        original_horizons = agent._config["target_horizons"].copy()
        agent.run({
            "cleaned_data": ohlcv_data,
            "feature_config": {"target_horizons": [3, 10], "default_target": "fwd_direction_3d"},
        })
        assert agent._config["target_horizons"] == original_horizons
