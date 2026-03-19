"""StatsAgent — Statistical validation of strategy performance.

Determines whether strategy performance is statistically significant
or could arise by chance. Uses block bootstrap for Sharpe ratio
confidence intervals, hypothesis testing, and multiple testing correction.

Constraints:
  - Block bootstrap preserves return autocorrelation
  - Minimum 1000 bootstrap iterations
  - Multiple testing correction mandatory when comparing >1 model/symbol
  - Reports both raw and corrected p-values
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class StatsAgent(BaseAgent):
    """Statistical validation agent for strategy performance.

    Computes performance metrics, bootstrap confidence intervals,
    hypothesis tests, and multiple testing corrections.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "risk_free_rate": 0.0,
        "trading_days_per_year": 252,
        "bootstrap_iterations": 1000,
        "bootstrap_block_size": 5,
        "confidence_level": 0.95,
        "significance_threshold": 0.05,
        "random_seed": None,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "returns": (
                "pd.Series — daily returns series (preferred), OR"
            ),
            "equity_curve": (
                "pd.Series — portfolio equity curve (returns derived if provided)"
            ),
            "num_tests": (
                "(optional) int — number of models/symbols tested, "
                "for multiple testing correction. Default 1."
            ),
            "all_p_values": (
                "(optional) list[float] — p-values from all tested "
                "models/symbols for proper BH-FDR correction. "
                "Without this, BH falls back to conservative Bonferroni."
            ),
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "metrics": (
                "dict — sharpe, annualized_volatility, max_drawdown, "
                "total_return, annualized_return, sortino, calmar"
            ),
            "bootstrap": (
                "dict — sharpe_mean, sharpe_std, sharpe_ci_lower, "
                "sharpe_ci_upper, n_iterations"
            ),
            "hypothesis_test": (
                "dict — p_value, is_significant, null_hypothesis, "
                "alternative_hypothesis"
            ),
            "multiple_testing": (
                "dict — raw_p_value, corrected_p_value_bonferroni, "
                "corrected_p_value_bh, num_tests, is_significant_after_correction"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical validation pipeline.

        Args:
            inputs: dict with 'returns' (Series) or 'equity_curve' (Series),
            optional 'num_tests' (int), optional 'config' (dict).

        Returns:
            dict with 'metrics', 'bootstrap', 'hypothesis_test',
            'multiple_testing'.
        """
        returns, cfg = self._validate_inputs(inputs)
        num_tests = inputs.get("num_tests", 1)
        if not isinstance(num_tests, int) or num_tests < 1:
            num_tests = 1

        trading_days = cfg["trading_days_per_year"]
        risk_free = cfg["risk_free_rate"]
        n_iter = cfg["bootstrap_iterations"]
        if n_iter < 1000:
            raise ValueError(
                f"bootstrap_iterations must be >= 1000 per spec, got {n_iter}"
            )
        n = len(returns)
        user_block_size = cfg.get("bootstrap_block_size")
        if user_block_size is None or user_block_size == self.DEFAULT_CONFIG["bootstrap_block_size"]:
            block_size = self._adaptive_block_size(n)
        else:
            block_size = user_block_size

        if block_size >= n:
            block_size = max(1, n // 3)
            logger.warning(
                "bootstrap_block_size (%d) >= series length (%d), "
                "clamped to %d to avoid degenerate bootstrap",
                cfg.get("bootstrap_block_size", block_size), n, block_size,
            )
        conf_level = cfg["confidence_level"]
        sig_threshold = cfg["significance_threshold"]
        seed = cfg["random_seed"]

        # Phase 1: Basic performance metrics
        metrics = self._compute_metrics(returns, trading_days, risk_free)

        # Phase 1b: Turnover (optional — requires weights input)
        weights = inputs.get("weights")
        if weights is not None and isinstance(weights, pd.DataFrame) and len(weights) > 1:
            daily_turnover = weights.diff().abs().sum(axis=1).iloc[1:]
            metrics["turnover"] = float(daily_turnover.mean() * trading_days)

        # Phase 2: Block bootstrap for Sharpe ratio
        bootstrap = self._block_bootstrap_sharpe(
            returns=returns,
            trading_days=trading_days,
            risk_free=risk_free,
            n_iterations=n_iter,
            block_size=block_size,
            confidence_level=conf_level,
            seed=seed,
        )

        # Phase 3: Hypothesis testing (H0: Sharpe <= 0)
        hypothesis_test = self._hypothesis_test(
            bootstrap_sharpes=bootstrap["_sharpe_distribution"],
            significance_threshold=sig_threshold,
        )

        # Phase 4: Multiple testing correction
        raw_p = hypothesis_test["p_value"]
        all_p_values = inputs.get("all_p_values")
        multiple_testing = self._multiple_testing_correction(
            raw_p_value=raw_p,
            num_tests=num_tests,
            significance_threshold=sig_threshold,
            all_p_values=all_p_values,
        )

        # Clean internal fields from bootstrap
        bootstrap_output = {k: v for k, v in bootstrap.items() if not k.startswith("_")}

        outputs: Dict[str, Any] = {
            "metrics": metrics,
            "bootstrap": bootstrap_output,
            "hypothesis_test": hypothesis_test,
            "multiple_testing": multiple_testing,
        }

        # Phase 5: Benchmark comparison (optional)
        benchmark_returns = inputs.get("benchmark_returns")
        if benchmark_returns is not None:
            outputs["benchmark"] = self._benchmark_comparison(
                returns, benchmark_returns, trading_days,
            )

        # Validate on full-precision values, then round for output
        self.validate(inputs, outputs)
        self._round_outputs(outputs)

        # Record metrics
        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            **metrics,
            "p_value": hypothesis_test["p_value"],
            "is_significant": hypothesis_test["is_significant"],
            "sharpe_ci_lower": bootstrap_output["sharpe_ci_lower"],
            "sharpe_ci_upper": bootstrap_output["sharpe_ci_upper"],
            "num_tests": num_tests,
        }

        logger.info(
            "StatsAgent complete: Sharpe=%.3f, p=%.4f, significant=%s",
            metrics["sharpe"],
            hypothesis_test["p_value"],
            hypothesis_test["is_significant"],
        )

        return outputs

    # Annualized Sharpe above this threshold triggers a warning.
    _SHARPE_SUSPICION_THRESHOLD: float = 3.0

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:  # noqa: ARG002
        """Validate output structure and value ranges."""
        _ = inputs
        metrics = outputs["metrics"]

        # Numeric metrics must all be finite
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if not np.isfinite(val):
                    raise ValueError(f"Metric '{key}' is not finite: {val}")

        # Max drawdown must be <= 0
        if metrics["max_drawdown"] > 0:
            raise ValueError(
                f"max_drawdown must be <= 0, got {metrics['max_drawdown']}"
            )

        # Critic mindset: flag suspiciously high Sharpe
        sharpe = metrics["sharpe"]
        if abs(sharpe) > self._SHARPE_SUSPICION_THRESHOLD:
            logger.warning(
                "OVERFITTING WARNING: Sharpe ratio %.2f exceeds %.1f — "
                "likely overfitting, look-ahead bias, or insufficient data. "
                "Investigate before trusting this result.",
                sharpe, self._SHARPE_SUSPICION_THRESHOLD,
            )
            outputs.setdefault("warnings", []).append(
                f"sharpe_suspiciously_high: {sharpe:.4f} "
                f"(threshold: {self._SHARPE_SUSPICION_THRESHOLD})"
            )

        # Bootstrap CI: lower <= upper (equality allowed for degenerate cases)
        bs = outputs["bootstrap"]
        if bs["sharpe_ci_lower"] > bs["sharpe_ci_upper"]:
            raise ValueError("Bootstrap CI lower > upper")

        # p-value in [0, 1]
        p = outputs["hypothesis_test"]["p_value"]
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p_value out of range [0, 1]: {p}")

        # Corrected p-values in [0, 1]
        mt = outputs["multiple_testing"]
        for key in ("corrected_p_value_bonferroni", "corrected_p_value_bh"):
            cp = mt[key]
            if not (0.0 <= cp <= 1.0):
                raise ValueError(f"multiple_testing['{key}'] out of [0, 1]: {cp}")

        return True

    def log_metrics(self) -> None:
        """Persist metrics from the most recent run to experiments/."""
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        experiments_dir = Path(__file__).parent.parent / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        run_id = self._metrics.get("run_id", uuid.uuid4().hex[:12])

        log_entry = {
            "experiment_id": f"stats_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "StatsAgent",
            "stage": "statistical_validation",
            "timestamp": now.isoformat(),
            "metrics": {
                k: v for k, v in self._metrics.items()
                if k not in ("run_id",)
            },
            "statistical_significance": {
                "sharpe_p_value": self._metrics.get("p_value"),
                "sharpe_ci_lower": self._metrics.get("sharpe_ci_lower"),
                "sharpe_ci_upper": self._metrics.get("sharpe_ci_upper"),
                "is_significant": self._metrics.get("is_significant"),
            },
            "notes": "StatsAgent statistical validation run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"stats_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: benchmark comparison ──────────────────────────────

    @staticmethod
    def _benchmark_comparison(
        returns: pd.Series,
        benchmark: pd.Series,
        trading_days: int,
    ) -> Dict[str, Any]:
        """Compute strategy performance relative to a benchmark."""
        if not isinstance(benchmark, pd.Series):
            raise TypeError(f"'benchmark_returns' must be pd.Series, got {type(benchmark)}")

        common_idx = returns.index.intersection(benchmark.index)
        min_required = max(2, len(returns) // 2)
        if len(common_idx) < min_required:
            raise ValueError(
                f"Strategy and benchmark share only {len(common_idx)} dates "
                f"(need at least {min_required}). Check index alignment."
            )

        strat = returns.loc[common_idx].values
        bench = benchmark.loc[common_idx].values
        excess = strat - bench

        excess_return = float(np.mean(excess) * trading_days)

        tracking_error = float(np.std(excess, ddof=1) * np.sqrt(trading_days))
        if tracking_error > 0:
            information_ratio = float(excess_return / tracking_error)
        else:
            information_ratio = 0.0

        # OLS regression: strategy = alpha + beta * benchmark
        bench_mean = bench.mean()
        strat_mean = strat.mean()
        cov = float(np.mean((strat - strat_mean) * (bench - bench_mean)))
        bench_var = float(np.mean((bench - bench_mean) ** 2))
        if bench_var > 0:
            beta = cov / bench_var
        else:
            beta = 0.0
        alpha_daily = strat_mean - beta * bench_mean
        alpha = float(alpha_daily * trading_days)

        return {
            "excess_return": excess_return,
            "information_ratio": information_ratio,
            "beta": beta,
            "alpha": alpha,
            "tracking_error": tracking_error,
        }

    # ── Internal: rounding (applied AFTER validation) ──────────────

    @staticmethod
    def _round_outputs(outputs: Dict[str, Any], decimals: int = 6) -> None:
        """Round all float values in outputs to *decimals* places in-place."""
        for section_key in ("metrics", "bootstrap", "hypothesis_test", "multiple_testing"):
            section = outputs.get(section_key, {})
            for k, v in section.items():
                if isinstance(v, float):
                    section[k] = round(v, decimals)

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Extract and validate returns series from inputs."""
        cfg = {**self._config, **(inputs.get("config") or {})}

        returns = inputs.get("returns")
        equity_curve = inputs.get("equity_curve")

        if returns is not None:
            if not isinstance(returns, pd.Series):
                raise TypeError(
                    f"'returns' must be pd.Series, got {type(returns)}"
                )
        elif equity_curve is not None:
            if not isinstance(equity_curve, pd.Series):
                raise TypeError(
                    f"'equity_curve' must be pd.Series, got {type(equity_curve)}"
                )
            # Derive returns from equity curve
            returns = equity_curve.pct_change().dropna()
        else:
            raise ValueError(
                "inputs must contain 'returns' or 'equity_curve'"
            )

        # Replace inf with NaN and drop before length check
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 2:
            raise ValueError(
                f"Insufficient data: need at least 2 observations "
                f"after cleaning, got {len(returns)}"
            )

        return returns, cfg

    # ── Internal: basic metrics ───────────────────────────────────

    _MIN_DAYS_FOR_ANNUALIZATION: int = 63  # ~1 quarter

    def _compute_metrics(
        self,
        returns: pd.Series,
        trading_days: int,
        risk_free: float,
    ) -> Dict[str, Any]:
        """Compute standard performance metrics with annualization."""
        n_days = len(returns)
        daily_excess = returns - risk_free / trading_days
        excess_std = float(daily_excess.std())  # ddof=1 (pandas default)
        sqrt_t = np.sqrt(trading_days)

        # ── Sharpe ratio (annualized, naive IID assumption) ──
        if excess_std == 0 or np.isnan(excess_std):
            sharpe = 0.0
        else:
            sharpe = float((daily_excess.mean() / excess_std) * sqrt_t)

        # ── Autocorrelation-adjusted Sharpe — Lo (2002) ──
        sharpe_adjusted = self._lo_adjusted_sharpe(
            daily_excess, sharpe, trading_days,
        )

        # ── Autocorrelation diagnostics ──
        rho_1 = float(daily_excess.autocorr(lag=1)) if n_days > 2 else 0.0
        if np.isnan(rho_1):
            rho_1 = 0.0

        # Annualized volatility (of raw returns, not excess)
        ann_vol = float(returns.std() * sqrt_t)

        # Total return (log-space for numerical stability)
        ret_vals = returns.values
        if np.any(ret_vals <= -1.0):
            total_return = -1.0
        else:
            log_returns = np.log1p(ret_vals)
            total_return = float(np.expm1(np.sum(log_returns)))

        # Annualized return
        years = n_days / trading_days
        if total_return <= -1.0:
            ann_return = -1.0
            logger.warning(
                "Total loss detected (total_return=%.4f). "
                "Annualized return set to -1.0.",
                total_return,
            )
        elif years > 0:
            ann_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        else:
            ann_return = 0.0

        ann_return_reliable = n_days >= self._MIN_DAYS_FOR_ANNUALIZATION
        if not ann_return_reliable:
            logger.warning(
                "Only %d trading days — annualized return (%.2f%%) is "
                "unreliable. Minimum %d days recommended.",
                n_days, ann_return * 100, self._MIN_DAYS_FOR_ANNUALIZATION,
            )

        # Max drawdown
        max_dd = self._max_drawdown(returns)

        # Sortino ratio (annualized) — lower partial moment with ddof=1
        downside_vals = np.minimum(daily_excess.values, 0.0)
        n_down = len(downside_vals)
        downside_var = float(np.sum(downside_vals ** 2) / max(n_down - 1, 1))
        downside_std = np.sqrt(downside_var) if downside_var > 0 else 0.0
        if downside_std == 0:
            sortino = 0.0
        else:
            sortino = float((daily_excess.mean() / downside_std) * sqrt_t)

        # Calmar ratio
        if max_dd != 0.0:
            calmar = float(ann_return / abs(max_dd))
        else:
            calmar = 0.0

        # Win rate — fraction of positive return days
        win_rate = float(np.mean(returns.values > 0))

        return {
            "sharpe": sharpe,
            "sharpe_adjusted": sharpe_adjusted,
            "autocorrelation_lag1": rho_1,
            "annualized_volatility": ann_vol,
            "max_drawdown": max_dd,
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_return_reliable": ann_return_reliable,
            "sortino": sortino,
            "calmar": calmar,
            "win_rate": win_rate,
        }

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Compute max drawdown from a returns series."""
        cumulative = (1.0 + returns).cumprod()
        values = cumulative.values.astype(float)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / np.where(cummax == 0, 1.0, cummax)
        return float(np.min(drawdown))

    @staticmethod
    def _lo_adjusted_sharpe(
        daily_excess: pd.Series,
        naive_sharpe: float,
        trading_days: int,
        max_lag: Optional[int] = None,
    ) -> float:
        """Autocorrelation-adjusted Sharpe ratio (Lo, 2002).

        Corrects the naive sqrt(T) annualization by accounting for
        serial correlation in returns.  Uses Bartlett-kernel weighting
        on the first *q* sample autocorrelations.

        SR_adj = SR_naive * sqrt(1 / eta)
        where eta = 1 + 2 * sum_{k=1}^{q} (1 - k/(q+1)) * rho_k

        When returns are positively autocorrelated (momentum), eta > 1
        and the adjusted Sharpe is *lower* — the IID version overstates
        significance.
        """
        n = len(daily_excess)
        if n < 10 or naive_sharpe == 0.0:
            return naive_sharpe

        q = max_lag if max_lag is not None else min(n // 5, int(np.sqrt(n)))
        q = max(1, q)

        vals = daily_excess.values
        demeaned = vals - vals.mean()
        var = float(np.dot(demeaned, demeaned) / n)
        if var == 0:
            return naive_sharpe

        correction = 0.0
        for k in range(1, q + 1):
            rho_k = float(np.dot(demeaned[k:], demeaned[:-k]) / (n * var))
            weight = 1.0 - k / (q + 1)
            correction += 2.0 * weight * rho_k

        eta = 1.0 + correction
        if eta <= 0:
            return naive_sharpe

        return float(naive_sharpe * np.sqrt(1.0 / eta))

    @staticmethod
    def _adaptive_block_size(n: int) -> int:
        """Data-driven block size — n^(1/3) rule (Politis & White, 2004)."""
        return max(2, int(round(n ** (1.0 / 3.0))))

    # ── Internal: block bootstrap ─────────────────────────────────

    @staticmethod
    def _block_bootstrap_sharpe(
        returns: pd.Series,
        trading_days: int,
        risk_free: float,
        n_iterations: int,
        block_size: int,
        confidence_level: float,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Block bootstrap for Sharpe ratio distribution.

        Uses moving block bootstrap (MBB) — overlapping blocks sampled
        with replacement — to preserve the autocorrelation structure.
        """
        rng = np.random.RandomState(seed)
        ret_values = returns.values
        n = len(ret_values)
        sqrt_t = np.sqrt(trading_days)
        rf_daily = risk_free / trading_days

        # Number of blocks needed to cover the full series
        n_blocks = max(1, int(np.ceil(n / block_size)))

        bootstrap_sharpes: List[float] = []

        for _ in range(n_iterations):
            # Sample block start indices
            starts = rng.randint(0, max(1, n - block_size + 1), size=n_blocks)
            # Concatenate blocks
            sample_indices = np.concatenate([
                np.arange(s, min(s + block_size, n)) for s in starts
            ])[:n]

            sample = ret_values[sample_indices]
            excess = sample - rf_daily
            std = excess.std(ddof=1)
            if std > 0:
                sample_sharpe = float((excess.mean() / std) * sqrt_t)
            else:
                sample_sharpe = 0.0
            bootstrap_sharpes.append(sample_sharpe)

        bs_array = np.array(bootstrap_sharpes)
        alpha = 1.0 - confidence_level
        ci_lower = float(np.percentile(bs_array, 100 * alpha / 2))
        ci_upper = float(np.percentile(bs_array, 100 * (1 - alpha / 2)))

        return {
            "sharpe_mean": float(np.mean(bs_array)),
            "sharpe_std": float(np.std(bs_array, ddof=1)),
            "sharpe_ci_lower": ci_lower,
            "sharpe_ci_upper": ci_upper,
            "n_iterations": n_iterations,
            "_sharpe_distribution": bs_array,
        }

    # ── Internal: hypothesis testing ──────────────────────────────

    @staticmethod
    def _hypothesis_test(
        bootstrap_sharpes: np.ndarray,
        significance_threshold: float,
    ) -> Dict[str, Any]:
        """Test H0: Sharpe <= 0 vs H1: Sharpe > 0.

        p-value = proportion of bootstrap Sharpe values <= 0.
        """
        p_value = float(np.mean(bootstrap_sharpes <= 0))
        is_significant = p_value < significance_threshold

        return {
            "p_value": p_value,
            "is_significant": is_significant,
            "null_hypothesis": "Sharpe ratio <= 0",
            "alternative_hypothesis": "Sharpe ratio > 0",
        }

    # ── Internal: multiple testing correction ─────────────────────

    @staticmethod
    def _multiple_testing_correction(
        raw_p_value: float,
        num_tests: int,
        significance_threshold: float,
        all_p_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Apply Bonferroni and Benjamini-Hochberg corrections.

        Bonferroni: p_corrected = min(p * num_tests, 1.0)

        BH-FDR requires all m p-values ranked jointly.  When
        ``all_p_values`` is provided, full BH is applied.  When only a
        single p-value is available, the conservative bound is used:
        p_bh = min(p * num_tests, 1.0)  (rank-1 assumption, same as
        Bonferroni) because we cannot determine the true rank without
        the other p-values.
        """
        if num_tests > 1:
            logger.info(
                "Multiple testing correction assumes independent tests. "
                "If strategies are correlated (e.g., same underlying assets), "
                "Bonferroni is overly conservative and BH may not control FDR. "
                "Consider permutation-based corrections for correlated tests."
            )

        bonferroni_p = min(raw_p_value * num_tests, 1.0)

        if all_p_values is not None and len(all_p_values) > 1:
            m = len(all_p_values)
            sorted_p = np.sort(all_p_values)
            # BH adjusted p-values with monotonicity enforcement
            adjusted = np.empty(m)
            adjusted[-1] = sorted_p[-1]
            for i in range(m - 2, -1, -1):
                adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))
            adjusted = np.clip(adjusted, 0.0, 1.0)
            # Find raw_p_value's rank (first occurrence in sorted order)
            rank_idx = int(np.searchsorted(sorted_p, raw_p_value))
            bh_p = float(adjusted[min(rank_idx, m - 1)])
        else:
            # Conservative: assume worst-case rank=1, giving p*m/1 = p*m
            bh_p = min(raw_p_value * num_tests, 1.0)

        is_significant = bonferroni_p < significance_threshold

        return {
            "raw_p_value": raw_p_value,
            "corrected_p_value_bonferroni": bonferroni_p,
            "corrected_p_value_bh": bh_p,
            "num_tests": num_tests,
            "is_significant_after_correction": is_significant,
        }
