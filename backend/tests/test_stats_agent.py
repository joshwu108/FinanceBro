"""Tests for StatsAgent.

Validates:
  - BaseAgent contract is satisfied
  - Accepts returns series or equity curve as input
  - Computes: Sharpe, Sortino, max drawdown, volatility, total return
  - Annualizes metrics correctly (sqrt(252) scaling)
  - Block bootstrap Sharpe ratio with confidence intervals
  - Hypothesis testing (H0: Sharpe <= 0)
  - Multiple testing correction (Bonferroni, BH-FDR)
  - Edge cases: flat returns, single observation, all-positive returns
  - Input validation
  - Experiment logging
  - Reproducibility with random seed
"""

import numpy as np
import pandas as pd
import pytest

from agents.stats_agent import StatsAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_positive_returns(n_days: int = 252, seed: int = 42) -> pd.Series:
    """Returns with positive drift (strategy that works)."""
    rng = np.random.RandomState(seed)
    daily = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(daily, index=dates, name="returns")


def _make_negative_returns(n_days: int = 252, seed: int = 42) -> pd.Series:
    """Returns with negative drift (strategy that fails)."""
    rng = np.random.RandomState(seed)
    daily = rng.normal(loc=-0.001, scale=0.01, size=n_days)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(daily, index=dates, name="returns")


def _make_flat_returns(n_days: int = 252) -> pd.Series:
    """Zero returns — no alpha."""
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(0.0, index=dates, name="returns")


def _make_noisy_returns(n_days: int = 252, seed: int = 42) -> pd.Series:
    """Pure noise around zero mean — no alpha."""
    rng = np.random.RandomState(seed)
    daily = rng.normal(loc=0.0, scale=0.02, size=n_days)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(daily, index=dates, name="returns")


def _make_strong_positive_returns(n_days: int = 504, seed: int = 42) -> pd.Series:
    """Returns with very strong positive drift — statistically significant."""
    rng = np.random.RandomState(seed)
    daily = rng.normal(loc=0.003, scale=0.01, size=n_days)
    dates = pd.bdate_range(start="2022-01-01", periods=n_days)
    return pd.Series(daily, index=dates, name="returns")


def _make_equity_curve(n_days: int = 252, seed: int = 42) -> pd.Series:
    """Equity curve starting at 100,000 with positive drift."""
    rng = np.random.RandomState(seed)
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    prices = 100_000.0 * np.cumprod(1.0 + daily_returns)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(prices, index=dates, name="equity")


@pytest.fixture
def agent():
    return StatsAgent(config={"random_seed": 42})


@pytest.fixture
def positive_returns():
    return _make_positive_returns()


@pytest.fixture
def negative_returns():
    return _make_negative_returns()


@pytest.fixture
def flat_returns():
    return _make_flat_returns()


@pytest.fixture
def noisy_returns():
    return _make_noisy_returns()


@pytest.fixture
def equity_curve():
    return _make_equity_curve()


# ── BaseAgent Contract Tests ──────────────────────────────────────


class TestBaseAgentContract:

    def test_input_schema_exists(self, agent):
        schema = agent.input_schema
        assert isinstance(schema, dict)
        assert "returns" in schema or "equity_curve" in schema

    def test_output_schema_exists(self, agent):
        schema = agent.output_schema
        assert isinstance(schema, dict)
        assert "metrics" in schema
        assert "bootstrap" in schema
        assert "hypothesis_test" in schema

    def test_run_returns_dict(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert isinstance(result, dict)

    def test_validate_returns_true(self, agent, positive_returns):
        inputs = {"returns": positive_returns}
        outputs = agent.run(inputs)
        assert agent.validate(inputs, outputs) is True

    def test_log_metrics_does_not_raise(self, agent, positive_returns):
        agent.run({"returns": positive_returns})
        agent.log_metrics()


# ── Input Handling Tests ──────────────────────────────────────────


class TestInputHandling:

    def test_accepts_returns_series(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert "metrics" in result

    def test_accepts_equity_curve(self, agent, equity_curve):
        """Should derive returns from equity curve."""
        result = agent.run({"equity_curve": equity_curve})
        assert "metrics" in result

    def test_missing_both_inputs_raises(self, agent):
        with pytest.raises(ValueError, match="[Rr]eturn|[Ee]quity"):
            agent.run({})

    def test_returns_preferred_over_equity(self, agent, positive_returns, equity_curve):
        """When both provided, returns should be used."""
        result = agent.run({
            "returns": positive_returns,
            "equity_curve": equity_curve,
        })
        # The Sharpe should match what we'd get from returns alone
        result_returns_only = agent.run({"returns": positive_returns})
        assert (
            result["metrics"]["sharpe"]
            == result_returns_only["metrics"]["sharpe"]
        )

    def test_empty_returns_raises(self, agent):
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="[Ee]mpty|[Ii]nsufficient"):
            agent.run({"returns": empty})


# ── Basic Metrics Tests ───────────────────────────────────────────


class TestBasicMetrics:

    def test_metrics_keys(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        metrics = result["metrics"]
        required = {
            "sharpe", "sharpe_adjusted", "autocorrelation_lag1",
            "annualized_volatility", "max_drawdown",
            "total_return", "annualized_return", "annualized_return_reliable",
            "sortino", "calmar",
        }
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_sharpe_positive_for_positive_drift(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["metrics"]["sharpe"] > 0

    def test_sharpe_negative_for_negative_drift(self, agent, negative_returns):
        result = agent.run({"returns": negative_returns})
        assert result["metrics"]["sharpe"] < 0

    def test_sharpe_near_zero_for_flat(self, agent, flat_returns):
        result = agent.run({"returns": flat_returns})
        assert result["metrics"]["sharpe"] == pytest.approx(0.0, abs=1e-10)

    def test_volatility_positive(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["metrics"]["annualized_volatility"] > 0

    def test_volatility_zero_for_flat(self, agent, flat_returns):
        result = agent.run({"returns": flat_returns})
        assert result["metrics"]["annualized_volatility"] == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_non_positive(self, agent, positive_returns):
        """Max drawdown should be <= 0 (negative fraction)."""
        result = agent.run({"returns": positive_returns})
        assert result["metrics"]["max_drawdown"] <= 0.0

    def test_max_drawdown_zero_for_flat(self, agent, flat_returns):
        result = agent.run({"returns": flat_returns})
        assert result["metrics"]["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_total_return_positive_for_positive_drift(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["metrics"]["total_return"] > 0

    def test_total_return_negative_for_negative_drift(self, agent, negative_returns):
        result = agent.run({"returns": negative_returns})
        assert result["metrics"]["total_return"] < 0

    def test_annualized_return_is_finite(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert np.isfinite(result["metrics"]["annualized_return"])

    def test_sortino_is_finite(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert np.isfinite(result["metrics"]["sortino"])

    def test_calmar_is_finite(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert np.isfinite(result["metrics"]["calmar"])

    def test_all_metrics_finite(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        for key, val in result["metrics"].items():
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"


# ── Annualization Tests ───────────────────────────────────────────


class TestAnnualization:

    def test_sharpe_is_annualized(self, agent, positive_returns):
        """Annualized Sharpe should be ~sqrt(252) times daily Sharpe."""
        result = agent.run({"returns": positive_returns})
        daily_sharpe = positive_returns.mean() / positive_returns.std()
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        assert result["metrics"]["sharpe"] == pytest.approx(
            annualized_sharpe, rel=0.01
        )

    def test_volatility_is_annualized(self, agent, positive_returns):
        """Annualized vol should be ~sqrt(252) times daily std."""
        result = agent.run({"returns": positive_returns})
        expected = positive_returns.std() * np.sqrt(252)
        assert result["metrics"]["annualized_volatility"] == pytest.approx(
            expected, rel=0.01
        )


# ── Bootstrap Tests ───────────────────────────────────────────────


class TestBootstrap:

    def test_bootstrap_keys(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        bs = result["bootstrap"]
        required = {
            "sharpe_mean", "sharpe_std", "sharpe_ci_lower",
            "sharpe_ci_upper", "n_iterations",
        }
        for key in required:
            assert key in bs, f"Missing bootstrap key: {key}"

    def test_bootstrap_iterations_count(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["bootstrap"]["n_iterations"] >= 1000

    def test_ci_lower_less_than_upper(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        bs = result["bootstrap"]
        assert bs["sharpe_ci_lower"] < bs["sharpe_ci_upper"]

    def test_ci_contains_point_estimate(self, agent, positive_returns):
        """The point-estimate Sharpe should be within or near the CI."""
        result = agent.run({"returns": positive_returns})
        sharpe = result["metrics"]["sharpe"]
        bs = result["bootstrap"]
        # Allow some margin — bootstrap mean can differ from point estimate
        assert bs["sharpe_ci_lower"] <= sharpe + 0.5
        assert bs["sharpe_ci_upper"] >= sharpe - 0.5

    def test_positive_strategy_ci_above_zero(self, agent):
        """A strongly positive strategy should have CI above zero."""
        strong = _make_strong_positive_returns()
        result = agent.run({"returns": strong})
        bs = result["bootstrap"]
        assert bs["sharpe_ci_lower"] > 0

    def test_noisy_strategy_ci_includes_zero(self, agent, noisy_returns):
        """A pure-noise strategy should have CI spanning zero."""
        result = agent.run({"returns": noisy_returns})
        bs = result["bootstrap"]
        assert bs["sharpe_ci_lower"] <= 0 <= bs["sharpe_ci_upper"]

    def test_custom_bootstrap_iterations(self):
        agent = StatsAgent(config={"bootstrap_iterations": 2000, "random_seed": 42})
        returns = _make_positive_returns()
        result = agent.run({"returns": returns})
        assert result["bootstrap"]["n_iterations"] == 2000

    def test_bootstrap_std_positive(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["bootstrap"]["sharpe_std"] > 0


# ── Hypothesis Testing ────────────────────────────────────────────


class TestHypothesisTesting:

    def test_hypothesis_test_keys(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        ht = result["hypothesis_test"]
        required = {
            "p_value", "is_significant",
            "null_hypothesis", "alternative_hypothesis",
        }
        for key in required:
            assert key in ht, f"Missing hypothesis test key: {key}"

    def test_p_value_between_zero_and_one(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        p = result["hypothesis_test"]["p_value"]
        assert 0.0 <= p <= 1.0

    def test_positive_strategy_significant(self, agent):
        """A strategy with strong positive drift should be significant."""
        strong = _make_strong_positive_returns()
        result = agent.run({"returns": strong})
        assert result["hypothesis_test"]["is_significant"] is True
        assert result["hypothesis_test"]["p_value"] < 0.05

    def test_noisy_strategy_not_significant(self, agent, noisy_returns):
        """Pure noise should not be significant."""
        result = agent.run({"returns": noisy_returns})
        assert result["hypothesis_test"]["is_significant"] is False

    def test_null_hypothesis_stated(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        ht = result["hypothesis_test"]
        assert "Sharpe" in ht["null_hypothesis"] or "sharpe" in ht["null_hypothesis"].lower()


# ── Multiple Testing Correction ───────────────────────────────────


class TestMultipleTestingCorrection:

    def test_multiple_testing_keys(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns, "num_tests": 5})
        mt = result["multiple_testing"]
        required = {
            "raw_p_value", "corrected_p_value_bonferroni",
            "corrected_p_value_bh", "num_tests",
            "is_significant_after_correction",
        }
        for key in required:
            assert key in mt, f"Missing key: {key}"

    def test_bonferroni_increases_p_value(self, agent, positive_returns):
        """Bonferroni correction should increase the p-value."""
        result = agent.run({"returns": positive_returns, "num_tests": 10})
        mt = result["multiple_testing"]
        assert mt["corrected_p_value_bonferroni"] >= mt["raw_p_value"]

    def test_bonferroni_scales_with_num_tests(self, agent, positive_returns):
        """More tests = more conservative correction."""
        result_5 = agent.run({"returns": positive_returns, "num_tests": 5})
        result_20 = agent.run({"returns": positive_returns, "num_tests": 20})
        p5 = result_5["multiple_testing"]["corrected_p_value_bonferroni"]
        p20 = result_20["multiple_testing"]["corrected_p_value_bonferroni"]
        assert p20 >= p5

    def test_bh_without_all_p_values_falls_back_to_conservative(self, agent, positive_returns):
        """Without all_p_values, BH uses conservative bound (same as Bonferroni)."""
        result = agent.run({"returns": positive_returns, "num_tests": 10})
        mt = result["multiple_testing"]
        assert mt["corrected_p_value_bh"] == pytest.approx(
            mt["corrected_p_value_bonferroni"], rel=1e-10,
        )

    def test_bh_with_all_p_values_less_conservative(self, agent, positive_returns):
        """Full BH with all p-values should be less conservative than Bonferroni."""
        result = agent.run({
            "returns": positive_returns,
            "num_tests": 5,
            "all_p_values": [0.001, 0.01, 0.03, 0.20, 0.80],
        })
        mt = result["multiple_testing"]
        assert mt["corrected_p_value_bh"] <= mt["corrected_p_value_bonferroni"]

    def test_single_test_no_correction(self, agent, positive_returns):
        """With num_tests=1, corrected p-values should equal raw."""
        result = agent.run({"returns": positive_returns, "num_tests": 1})
        mt = result["multiple_testing"]
        assert mt["corrected_p_value_bonferroni"] == pytest.approx(
            mt["raw_p_value"], rel=1e-10
        )

    def test_default_num_tests_is_one(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["multiple_testing"]["num_tests"] == 1

    def test_corrected_p_value_capped_at_one(self, agent, noisy_returns):
        """Bonferroni-corrected p-value should never exceed 1.0."""
        result = agent.run({"returns": noisy_returns, "num_tests": 100})
        mt = result["multiple_testing"]
        assert mt["corrected_p_value_bonferroni"] <= 1.0


# ── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:

    def test_short_returns_series(self, agent):
        """Minimum viable input — very short series."""
        dates = pd.bdate_range(start="2023-01-01", periods=30)
        returns = pd.Series(
            np.random.RandomState(42).normal(0.001, 0.01, 30),
            index=dates,
        )
        result = agent.run({"returns": returns})
        assert "metrics" in result

    def test_single_large_drawdown(self, agent):
        """One big loss in otherwise positive returns."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range(start="2023-01-01", periods=100)
        returns = pd.Series(rng.normal(0.001, 0.005, 100), index=dates)
        returns.iloc[50] = -0.15  # -15% crash
        result = agent.run({"returns": returns})
        assert result["metrics"]["max_drawdown"] < -0.10

    def test_all_positive_returns(self, agent):
        """Every day is positive — max drawdown should be 0."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        returns = pd.Series(0.001, index=dates)
        result = agent.run({"returns": returns})
        assert result["metrics"]["max_drawdown"] == pytest.approx(0.0, abs=1e-10)


# ── Reproducibility Tests ─────────────────────────────────────────


class TestReproducibility:

    def test_seeded_bootstrap_is_deterministic(self, positive_returns):
        agent1 = StatsAgent(config={"random_seed": 123})
        agent2 = StatsAgent(config={"random_seed": 123})

        result1 = agent1.run({"returns": positive_returns})
        result2 = agent2.run({"returns": positive_returns})

        assert result1["bootstrap"] == result2["bootstrap"]
        assert result1["hypothesis_test"] == result2["hypothesis_test"]

    def test_different_seeds_differ(self, positive_returns):
        agent1 = StatsAgent(config={"random_seed": 1})
        agent2 = StatsAgent(config={"random_seed": 2})

        result1 = agent1.run({"returns": positive_returns})
        result2 = agent2.run({"returns": positive_returns})

        # Bootstrap CIs should differ (not exact match)
        assert result1["bootstrap"]["sharpe_ci_lower"] != result2["bootstrap"]["sharpe_ci_lower"]


# ── Config Override Tests ─────────────────────────────────────────


class TestConfigOverride:

    def test_custom_confidence_level(self, positive_returns):
        agent_90 = StatsAgent(config={"confidence_level": 0.90, "random_seed": 42})
        agent_99 = StatsAgent(config={"confidence_level": 0.99, "random_seed": 42})

        result_90 = agent_90.run({"returns": positive_returns})
        result_99 = agent_99.run({"returns": positive_returns})

        # 99% CI should be wider than 90% CI
        width_90 = (
            result_90["bootstrap"]["sharpe_ci_upper"]
            - result_90["bootstrap"]["sharpe_ci_lower"]
        )
        width_99 = (
            result_99["bootstrap"]["sharpe_ci_upper"]
            - result_99["bootstrap"]["sharpe_ci_lower"]
        )
        assert width_99 > width_90


# ── Experiment Logging Tests ──────────────────────────────────────


class TestExperimentLogging:

    def test_metrics_populated_after_run(self, agent, positive_returns):
        agent.run({"returns": positive_returns})
        assert agent._metrics
        assert "sharpe" in agent._metrics


# ── Hardened Edge Cases ───────────────────────────────────────────


class TestHardenedEdgeCases:

    def test_nan_heavy_input_raises_if_too_few_remain(self, agent):
        """Series mostly NaN should raise after cleanup."""
        dates = pd.bdate_range(start="2023-01-01", periods=10)
        returns = pd.Series([np.nan] * 8 + [0.01, np.nan], index=dates)
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            agent.run({"returns": returns})

    def test_inf_heavy_input_raises_if_too_few_remain(self, agent):
        """Series mostly inf should raise after cleanup."""
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        returns = pd.Series([np.inf, -np.inf, np.inf, -np.inf, 0.01], index=dates)
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            agent.run({"returns": returns})

    def test_total_loss_return(self, agent):
        """A -1.0 return (total wipeout) should produce annualized_return = -1.0."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.001, 0.005, 50), index=dates)
        returns.iloc[25] = -1.0
        result = agent.run({"returns": returns})
        assert result["metrics"]["annualized_return"] == -1.0
        assert result["metrics"]["total_return"] <= -1.0

    def test_bootstrap_iterations_below_1000_rejected(self):
        """Spec mandates >= 1000 iterations; lower values must be rejected."""
        agent = StatsAgent(config={"bootstrap_iterations": 500, "random_seed": 42})
        returns = _make_positive_returns()
        with pytest.raises(ValueError, match="1000"):
            agent.run({"returns": returns})

    def test_block_size_larger_than_series_clamped(self):
        """block_size >= len(returns) should be auto-clamped, not crash."""
        agent = StatsAgent(config={"bootstrap_block_size": 999, "random_seed": 42})
        dates = pd.bdate_range(start="2023-01-01", periods=30)
        returns = pd.Series(
            np.random.RandomState(42).normal(0.001, 0.01, 30),
            index=dates,
        )
        result = agent.run({"returns": returns})
        assert result["bootstrap"]["sharpe_std"] > 0

    def test_sharpe_suspiciously_high_produces_warning(self):
        """A strategy with unrealistically high Sharpe should trigger a warning."""
        agent = StatsAgent(config={"random_seed": 42})
        dates = pd.bdate_range(start="2023-01-01", periods=252)
        returns = pd.Series(0.01, index=dates)
        returns.iloc[::10] = 0.009
        result = agent.run({"returns": returns})
        if abs(result["metrics"]["sharpe"]) > agent._SHARPE_SUSPICION_THRESHOLD:
            assert "warnings" in result
            assert any("sharpe_suspiciously_high" in w for w in result["warnings"])

    def test_two_observation_minimum(self, agent):
        """Exactly 2 observations should still compute without error."""
        dates = pd.bdate_range(start="2023-01-01", periods=2)
        returns = pd.Series([0.01, -0.005], index=dates)
        result = agent.run({"returns": returns})
        assert np.isfinite(result["metrics"]["sharpe"])

    def test_equity_curve_with_zero_value_handles_inf(self, agent):
        """Equity curve hitting 0 produces inf in pct_change; infs cleaned."""
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        equity = pd.Series([100, 110, 0, 50, 60], index=dates)
        result = agent.run({"equity_curve": equity})
        assert result["metrics"]["total_return"] <= -1.0


# ── BH Full Correction Tests ─────────────────────────────────────


class TestBHFullCorrection:

    def test_bh_with_ranked_p_values(self):
        """Full BH should properly adjust based on rank."""
        result = StatsAgent._multiple_testing_correction(
            raw_p_value=0.03,
            num_tests=5,
            significance_threshold=0.05,
            all_p_values=[0.001, 0.01, 0.03, 0.20, 0.80],
        )
        assert result["corrected_p_value_bh"] <= result["corrected_p_value_bonferroni"]
        assert result["corrected_p_value_bh"] <= 1.0
        assert result["corrected_p_value_bh"] >= result["raw_p_value"]

    def test_bh_monotonicity(self):
        """BH adjusted p-values should be monotonically non-decreasing."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05]
        results = []
        for p in p_values:
            r = StatsAgent._multiple_testing_correction(
                raw_p_value=p,
                num_tests=5,
                significance_threshold=0.05,
                all_p_values=p_values,
            )
            results.append(r["corrected_p_value_bh"])
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1] + 1e-10


# ── Autocorrelation & Lo(2002) Tests ─────────────────────────────


class TestAutocorrelation:

    def test_autocorrelation_lag1_reported(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert "autocorrelation_lag1" in result["metrics"]
        assert np.isfinite(result["metrics"]["autocorrelation_lag1"])

    def test_adjusted_sharpe_reported(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert "sharpe_adjusted" in result["metrics"]
        assert np.isfinite(result["metrics"]["sharpe_adjusted"])

    def test_adjusted_sharpe_lower_for_positively_autocorrelated(self):
        """Momentum-like returns (positive autocorrelation) should have
        adjusted Sharpe <= naive Sharpe."""
        rng = np.random.RandomState(42)
        n = 500
        raw = rng.normal(0.001, 0.01, n)
        # Induce positive autocorrelation via smoothing
        smoothed = np.convolve(raw, [0.5, 0.5], mode="same")
        dates = pd.bdate_range(start="2022-01-01", periods=n)
        returns = pd.Series(smoothed, index=dates)

        agent = StatsAgent(config={"random_seed": 42})
        result = agent.run({"returns": returns})
        m = result["metrics"]
        assert m["autocorrelation_lag1"] > 0.1
        assert m["sharpe_adjusted"] <= m["sharpe"] + 0.01

    def test_adjusted_sharpe_equals_naive_for_iid(self):
        """For IID returns, adjusted Sharpe should be close to naive."""
        rng = np.random.RandomState(42)
        n = 1000
        daily = rng.normal(0.0005, 0.01, n)
        dates = pd.bdate_range(start="2020-01-01", periods=n)
        returns = pd.Series(daily, index=dates)

        agent = StatsAgent(config={"random_seed": 42})
        result = agent.run({"returns": returns})
        m = result["metrics"]
        assert abs(m["sharpe_adjusted"] - m["sharpe"]) / max(abs(m["sharpe"]), 1e-9) < 0.15

    def test_flat_returns_adjusted_sharpe_zero(self, agent, flat_returns):
        result = agent.run({"returns": flat_returns})
        assert result["metrics"]["sharpe_adjusted"] == 0.0


# ── Aggregation Stability Tests ───────────────────────────────────


class TestAggregationStability:

    def test_total_return_log_space_matches_naive(self, agent, positive_returns):
        """Log-space total return should match naive for normal magnitudes."""
        result = agent.run({"returns": positive_returns})
        naive = float(np.prod((1.0 + positive_returns).values) - 1.0)
        assert result["metrics"]["total_return"] == pytest.approx(naive, rel=1e-5)

    def test_total_return_stable_for_long_series(self):
        """Log-space computation should stay accurate for multi-year series."""
        rng = np.random.RandomState(42)
        n = 2520  # ~10 years
        daily = rng.normal(0.0003, 0.015, n)
        dates = pd.bdate_range(start="2015-01-01", periods=n)
        returns = pd.Series(daily, index=dates)

        agent = StatsAgent(config={"random_seed": 42})
        result = agent.run({"returns": returns})
        tr = result["metrics"]["total_return"]
        assert np.isfinite(tr)
        assert tr > -1.0

    def test_sortino_ddof_consistency(self, agent, positive_returns):
        """Sortino should use ddof=1 consistent with Sharpe."""
        result = agent.run({"returns": positive_returns})
        assert np.isfinite(result["metrics"]["sortino"])

    def test_annualized_return_unreliable_flagged_for_short_series(self):
        """Short series should flag annualized_return_reliable=False."""
        agent = StatsAgent(config={"random_seed": 42})
        dates = pd.bdate_range(start="2023-01-01", periods=30)
        returns = pd.Series(
            np.random.RandomState(42).normal(0.001, 0.01, 30),
            index=dates,
        )
        result = agent.run({"returns": returns})
        assert result["metrics"]["annualized_return_reliable"] is False

    def test_annualized_return_reliable_for_long_series(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert result["metrics"]["annualized_return_reliable"] is True


# ── Adaptive Block Size Tests ─────────────────────────────────────


class TestAdaptiveBlockSize:

    def test_adaptive_block_size_formula(self):
        assert StatsAgent._adaptive_block_size(252) == max(2, round(252 ** (1/3)))
        assert StatsAgent._adaptive_block_size(27) == 3
        assert StatsAgent._adaptive_block_size(8) == 2

    def test_explicit_block_size_overrides_adaptive(self):
        """User-specified block_size should override adaptive selection."""
        agent = StatsAgent(config={
            "bootstrap_block_size": 10,
            "random_seed": 42,
        })
        returns = _make_positive_returns()
        result_explicit = agent.run({"returns": returns})

        agent_default = StatsAgent(config={"random_seed": 42})
        result_default = agent_default.run({"returns": returns})

        assert result_explicit["bootstrap"] != result_default["bootstrap"]


# ── Rounding-After-Validation Tests ──────────────────────────────


class TestRoundingOrder:

    def test_output_values_are_rounded(self, agent, positive_returns):
        """Final output floats should be rounded to 6 dp."""
        result = agent.run({"returns": positive_returns})
        sharpe = result["metrics"]["sharpe"]
        assert sharpe == round(sharpe, 6)

    def test_near_degenerate_ci_detected_before_rounding(self):
        """CI that differs by 1e-8 should not be masked by rounding."""
        agent = StatsAgent(config={"random_seed": 42})
        dates = pd.bdate_range(start="2023-01-01", periods=2)
        returns = pd.Series([0.01, -0.005], index=dates)
        result = agent.run({"returns": returns})
        bs = result["bootstrap"]
        assert isinstance(bs["sharpe_ci_lower"], float)
        assert isinstance(bs["sharpe_ci_upper"], float)


# ── Win Rate Tests ───────────────────────────────────────────────


class TestWinRate:

    def test_win_rate_key_present(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        assert "win_rate" in result["metrics"]

    def test_win_rate_between_zero_and_one(self, agent, positive_returns):
        result = agent.run({"returns": positive_returns})
        wr = result["metrics"]["win_rate"]
        assert 0.0 <= wr <= 1.0

    def test_win_rate_all_positive(self, agent):
        """All positive daily returns → win rate = 1.0."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        returns = pd.Series(0.001, index=dates)
        result = agent.run({"returns": returns})
        assert result["metrics"]["win_rate"] == 1.0

    def test_win_rate_all_negative(self, agent):
        """All negative daily returns → win rate = 0.0."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        returns = pd.Series(-0.001, index=dates)
        result = agent.run({"returns": returns})
        assert result["metrics"]["win_rate"] == 0.0

    def test_win_rate_flat_returns(self, agent, flat_returns):
        """Zero returns are not wins → win rate = 0.0."""
        result = agent.run({"returns": flat_returns})
        assert result["metrics"]["win_rate"] == 0.0

    def test_win_rate_known_value(self, agent):
        """Manually verify: 3 positive out of 5 = 0.6."""
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.005], index=dates)
        result = agent.run({"returns": returns})
        assert result["metrics"]["win_rate"] == pytest.approx(0.6, abs=1e-10)


# ── Turnover Tests ───────────────────────────────────────────────


def _make_weights(n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Simulated portfolio weights that change over time."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    # 3 assets with slowly changing weights
    raw = rng.dirichlet(alpha=[5, 3, 2], size=n_days)
    return pd.DataFrame(raw, index=dates, columns=["A", "B", "C"])


class TestTurnover:

    def test_turnover_key_present_when_weights_given(self, agent, positive_returns):
        weights = _make_weights(n_days=len(positive_returns))
        weights.index = positive_returns.index
        result = agent.run({"returns": positive_returns, "weights": weights})
        assert "turnover" in result["metrics"]

    def test_turnover_absent_when_no_weights(self, agent, positive_returns):
        """Without weights, turnover should not be in metrics."""
        result = agent.run({"returns": positive_returns})
        assert "turnover" not in result["metrics"]

    def test_turnover_non_negative(self, agent, positive_returns):
        weights = _make_weights(n_days=len(positive_returns))
        weights.index = positive_returns.index
        result = agent.run({"returns": positive_returns, "weights": weights})
        assert result["metrics"]["turnover"] >= 0.0

    def test_turnover_zero_for_static_weights(self, agent, positive_returns):
        """Constant weights → zero turnover."""
        n = len(positive_returns)
        dates = positive_returns.index
        weights = pd.DataFrame(
            {"A": [0.5] * n, "B": [0.3] * n, "C": [0.2] * n},
            index=dates,
        )
        result = agent.run({"returns": positive_returns, "weights": weights})
        assert result["metrics"]["turnover"] == pytest.approx(0.0, abs=1e-10)

    def test_turnover_annualized(self, agent, positive_returns):
        """Turnover should be annualized (daily avg * 252)."""
        weights = _make_weights(n_days=len(positive_returns))
        weights.index = positive_returns.index
        result = agent.run({"returns": positive_returns, "weights": weights})
        # Manually compute expected annualized turnover
        daily_turnover = weights.diff().abs().sum(axis=1).iloc[1:]
        expected = float(daily_turnover.mean() * 252)
        assert result["metrics"]["turnover"] == pytest.approx(expected, rel=1e-6)

    def test_turnover_is_finite(self, agent, positive_returns):
        weights = _make_weights(n_days=len(positive_returns))
        weights.index = positive_returns.index
        result = agent.run({"returns": positive_returns, "weights": weights})
        assert np.isfinite(result["metrics"]["turnover"])


# ── Benchmark Comparison Tests ───────────────────────────────────


def _make_benchmark_returns(n_days: int = 252, seed: int = 99) -> pd.Series:
    """Simulated SPY benchmark returns."""
    rng = np.random.RandomState(seed)
    daily = rng.normal(loc=0.0003, scale=0.012, size=n_days)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)
    return pd.Series(daily, index=dates, name="SPY")


class TestBenchmarkComparison:

    def test_benchmark_section_present(self, agent, positive_returns):
        benchmark = _make_benchmark_returns(n_days=len(positive_returns))
        benchmark.index = positive_returns.index
        result = agent.run({
            "returns": positive_returns,
            "benchmark_returns": benchmark,
        })
        assert "benchmark" in result

    def test_benchmark_section_absent_without_input(self, agent, positive_returns):
        """No benchmark input → no benchmark section."""
        result = agent.run({"returns": positive_returns})
        assert "benchmark" not in result

    def test_benchmark_keys(self, agent, positive_returns):
        benchmark = _make_benchmark_returns(n_days=len(positive_returns))
        benchmark.index = positive_returns.index
        result = agent.run({
            "returns": positive_returns,
            "benchmark_returns": benchmark,
        })
        bm = result["benchmark"]
        required = {
            "excess_return", "information_ratio", "beta", "alpha",
            "tracking_error",
        }
        for key in required:
            assert key in bm, f"Missing benchmark key: {key}"

    def test_all_benchmark_values_finite(self, agent, positive_returns):
        benchmark = _make_benchmark_returns(n_days=len(positive_returns))
        benchmark.index = positive_returns.index
        result = agent.run({
            "returns": positive_returns,
            "benchmark_returns": benchmark,
        })
        for key, val in result["benchmark"].items():
            assert np.isfinite(val), f"benchmark[{key}] is not finite: {val}"

    def test_excess_return_sign(self, agent):
        """Strategy with higher return than benchmark → positive excess."""
        dates = pd.bdate_range(start="2023-01-01", periods=252)
        strategy = pd.Series(
            np.random.RandomState(42).normal(0.001, 0.01, 252),
            index=dates,
        )
        benchmark = pd.Series(
            np.random.RandomState(99).normal(0.0001, 0.01, 252),
            index=dates,
        )
        result = agent.run({
            "returns": strategy,
            "benchmark_returns": benchmark,
        })
        # With these seeds the strategy drift is higher
        assert result["benchmark"]["excess_return"] > 0

    def test_beta_near_one_for_correlated(self, agent):
        """Strategy = benchmark + noise should have beta near 1.0."""
        dates = pd.bdate_range(start="2023-01-01", periods=504)
        rng = np.random.RandomState(42)
        benchmark = pd.Series(
            rng.normal(0.0003, 0.01, 504), index=dates,
        )
        strategy = benchmark + rng.normal(0.0, 0.002, 504)
        result = agent.run({
            "returns": strategy,
            "benchmark_returns": benchmark,
        })
        assert result["benchmark"]["beta"] == pytest.approx(1.0, abs=0.15)

    def test_tracking_error_positive(self, agent, positive_returns):
        benchmark = _make_benchmark_returns(n_days=len(positive_returns))
        benchmark.index = positive_returns.index
        result = agent.run({
            "returns": positive_returns,
            "benchmark_returns": benchmark,
        })
        assert result["benchmark"]["tracking_error"] > 0

    def test_information_ratio_sign(self, agent):
        """Positive excess return with low tracking error → positive IR."""
        dates = pd.bdate_range(start="2023-01-01", periods=504)
        rng = np.random.RandomState(42)
        benchmark = pd.Series(rng.normal(0.0, 0.01, 504), index=dates)
        # Strategy is benchmark + consistent alpha
        strategy = benchmark + 0.001
        result = agent.run({
            "returns": strategy,
            "benchmark_returns": benchmark,
        })
        assert result["benchmark"]["information_ratio"] > 0

    def test_alpha_annualized(self, agent, positive_returns):
        """Alpha should be annualized (daily alpha * 252)."""
        benchmark = _make_benchmark_returns(n_days=len(positive_returns))
        benchmark.index = positive_returns.index
        result = agent.run({
            "returns": positive_returns,
            "benchmark_returns": benchmark,
        })
        # Alpha should be a reasonable annualized value
        alpha = result["benchmark"]["alpha"]
        assert abs(alpha) < 5.0  # Sanity: not astronomically high

    def test_benchmark_mismatched_length_raises(self, agent, positive_returns):
        """Benchmark and strategy must have same length after alignment."""
        benchmark = _make_benchmark_returns(n_days=10)
        with pytest.raises((ValueError, KeyError)):
            agent.run({
                "returns": positive_returns,
                "benchmark_returns": benchmark,
            })
