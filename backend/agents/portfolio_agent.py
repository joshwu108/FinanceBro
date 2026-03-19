"""PortfolioAgent — Portfolio construction and weight optimization.

Combines multiple assets into a portfolio using configurable weighting
methods (equal weight, risk parity, minimum variance).  Computes
portfolio returns, equity curve, and risk-adjusted performance metrics.

Constraints (from specs/portfolio_spec.md):
  - Covariance estimated via Ledoit-Wolf shrinkage (not raw sample)
  - Minimum estimation window: 252 trading days
  - Long-only by default (no shorting)
  - Max position weight: configurable, default 10%
  - Weights sum to ≤ 1.0 (cash position allowed)
  - Rebalancing incurs transaction costs
  - No look-ahead: weights at time t use only data up to t
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PortfolioAgent(BaseAgent):
    """Portfolio construction agent.

    Supported methods:
      - equal_weight: 1/N allocation (baseline)
      - risk_parity: weights inversely proportional to asset volatility
      - minimum_variance: minimize portfolio variance via Ledoit-Wolf covariance
    """

    DEFAULT_CONFIG: MappingProxyType = MappingProxyType({
        "method": "equal_weight",
        "max_weight": 0.10,
        "rebalance_frequency": 21,  # trading days (~monthly)
        "covariance_window": 252,   # 1 year minimum
        "transaction_cost_bps": 5,  # basis points per trade
        "initial_capital": 100_000.0,
        "trading_days_per_year": 252,
        "risk_free_rate": 0.0,
    })

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ────────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "returns": "pd.DataFrame — daily returns per asset (columns=tickers, index=dates)",
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "weights": "pd.DataFrame — portfolio weights per asset at each rebalance date",
            "portfolio_returns": "pd.Series — daily portfolio returns",
            "equity_curve": "pd.Series — cumulative portfolio equity",
            "portfolio_metrics": (
                "dict — annualized_return, annualized_volatility, "
                "sharpe_ratio, diversification_ratio"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        returns, cfg = self._validate_inputs(inputs)

        method = cfg["method"]
        max_weight = cfg["max_weight"]
        rebal_freq = cfg["rebalance_frequency"]
        cov_window = cfg["covariance_window"]
        tx_cost_bps = cfg["transaction_cost_bps"]
        initial_capital = cfg["initial_capital"]

        # Compute weights at each rebalance date
        weights = self._compute_weights(
            returns=returns,
            method=method,
            max_weight=max_weight,
            rebal_freq=rebal_freq,
            cov_window=cov_window,
        )

        # Compute portfolio returns from weights and asset returns
        portfolio_returns = self._compute_portfolio_returns(
            returns=returns,
            weights=weights,
            tx_cost_bps=tx_cost_bps,
        )

        # Compute equity curve
        equity_curve = self._compute_equity_curve(
            portfolio_returns=portfolio_returns,
            initial_capital=initial_capital,
        )

        # Compute portfolio metrics
        portfolio_metrics = self._compute_portfolio_metrics(
            portfolio_returns=portfolio_returns,
            weights=weights,
            returns=returns,
            cfg=cfg,
        )

        outputs: Dict[str, Any] = {
            "weights": weights,
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity_curve,
            "portfolio_metrics": portfolio_metrics,
        }

        self.validate(inputs, outputs)

        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "method": method,
            "max_weight": max_weight,
            "rebalance_frequency": rebal_freq,
            "transaction_cost_bps": tx_cost_bps,
            "n_assets": len(returns.columns),
            "n_rebalances": len(weights),
            **portfolio_metrics,
        }

        logger.info(
            "PortfolioAgent complete: method=%s, Sharpe=%.4f, n_assets=%d",
            method,
            portfolio_metrics["sharpe_ratio"],
            len(returns.columns),
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        weights = outputs["weights"]
        max_weight = self._config["max_weight"]

        if weights.isna().any().any():
            raise ValueError("Weights contain NaN values")

        if (weights < -1e-10).any().any():
            raise ValueError("Weights contain negative values (long-only constraint)")

        if (weights > max_weight + 1e-10).any().any():
            raise ValueError(
                f"Weights exceed max_weight ({max_weight})"
            )

        for _, row in weights.iterrows():
            if row.sum() > 1.0 + 1e-10:
                raise ValueError(
                    f"Weights sum to {row.sum():.6f}, exceeding 1.0"
                )

        required = {
            "annualized_return", "annualized_volatility",
            "sharpe_ratio", "diversification_ratio",
        }
        missing = required - set(outputs["portfolio_metrics"].keys())
        if missing:
            raise ValueError(f"Portfolio metrics missing keys: {missing}")

        return True

    def log_metrics(self) -> None:
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        experiments_dir = Path(__file__).parent.parent / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        run_id = self._metrics.get("run_id", uuid.uuid4().hex[:12])

        log_entry = {
            "experiment_id": f"portfolio_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "PortfolioAgent",
            "stage": "portfolio_construction",
            "timestamp": now.isoformat(),
            "parameters": {
                "method": self._metrics.get("method"),
                "max_weight": self._metrics.get("max_weight"),
                "rebalance_frequency": self._metrics.get("rebalance_frequency"),
                "transaction_cost_bps": self._metrics.get("transaction_cost_bps"),
                "n_assets": self._metrics.get("n_assets"),
            },
            "metrics": {
                "annualized_return": self._metrics.get("annualized_return"),
                "annualized_volatility": self._metrics.get("annualized_volatility"),
                "sharpe_ratio": self._metrics.get("sharpe_ratio"),
                "diversification_ratio": self._metrics.get("diversification_ratio"),
                "n_rebalances": self._metrics.get("n_rebalances"),
            },
            "notes": "PortfolioAgent portfolio construction run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"portfolio_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if "returns" not in inputs:
            raise ValueError("inputs must contain 'returns'")

        returns = inputs["returns"]

        if not isinstance(returns, pd.DataFrame):
            raise TypeError(
                f"'returns' must be a pd.DataFrame, got {type(returns)}"
            )
        if returns.isna().any().any():
            raise ValueError("returns contain NaN values")

        cfg = {**self._config, **(inputs.get("config") or {})}

        method = cfg["method"]
        cov_window = cfg["covariance_window"]

        if method in ("risk_parity", "minimum_variance"):
            if len(returns) < cov_window:
                raise ValueError(
                    f"Need at least {cov_window} (252) trading days for "
                    f"covariance estimation, got {len(returns)}"
                )

        return returns, cfg

    # ── Internal: weight computation ──────────────────────────────

    def _compute_weights(
        self,
        returns: pd.DataFrame,
        method: str,
        max_weight: float,
        rebal_freq: int,
        cov_window: int,
    ) -> pd.DataFrame:
        """Compute portfolio weights at each rebalance date.

        No look-ahead: at rebalance date t, only returns up to t are used.
        """
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Determine rebalance dates
        if method == "equal_weight":
            # Equal weight can start from the first date
            start_idx = 0
        else:
            # Other methods need cov_window data points
            start_idx = cov_window

        rebal_indices = list(range(start_idx, len(returns), rebal_freq))
        if not rebal_indices:
            rebal_indices = [start_idx]

        weight_rows = []
        weight_dates = []

        for idx in rebal_indices:
            if idx >= len(returns):
                break

            date = returns.index[idx]
            past_returns = returns.iloc[:idx + 1]

            if method == "equal_weight":
                raw_weights = self._equal_weight(n_assets)
            elif method == "risk_parity":
                raw_weights = self._risk_parity(past_returns, cov_window)
            elif method == "minimum_variance":
                raw_weights = self._minimum_variance(past_returns, cov_window)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Apply constraints
            constrained = self._apply_weight_constraints(raw_weights, max_weight)
            weight_rows.append(constrained)
            weight_dates.append(date)

        return pd.DataFrame(weight_rows, index=weight_dates, columns=tickers)

    @staticmethod
    def _equal_weight(n_assets: int) -> np.ndarray:
        """Uniform 1/N allocation."""
        return np.full(n_assets, 1.0 / n_assets)

    @staticmethod
    def _risk_parity(returns: pd.DataFrame, cov_window: int) -> np.ndarray:
        """Weights inversely proportional to asset volatility.

        Uses rolling volatility from the most recent cov_window days.
        """
        recent = returns.iloc[-cov_window:]
        vols = recent.std().values

        # Handle zero-volatility assets
        vols = np.maximum(vols, 1e-10)

        inverse_vol = 1.0 / vols
        return inverse_vol / inverse_vol.sum()

    @staticmethod
    def _minimum_variance(returns: pd.DataFrame, cov_window: int) -> np.ndarray:
        """Approximate minimum variance portfolio using Ledoit-Wolf covariance.

        Computes the unconstrained minimum variance weights analytically,
        then clips negative weights and renormalizes as a heuristic
        simplex projection.  This is an approximation — not a true QP
        solution to: min w'Σw  s.t.  w >= 0, sum(w) = 1.
        """
        recent = returns.iloc[-cov_window:]
        lw = LedoitWolf().fit(recent.values)
        cov_matrix = lw.covariance_

        n = len(cov_matrix)
        cov_inv = np.linalg.inv(cov_matrix + np.eye(n) * 1e-8)
        ones = np.ones(n)

        # Unconstrained minimum variance weights: w = Σ^{-1} 1 / (1' Σ^{-1} 1)
        raw = cov_inv @ ones
        raw = raw / raw.sum()

        # Project onto simplex (long-only: clip negatives, renormalize)
        raw = np.maximum(raw, 0.0)
        total = raw.sum()
        if total > 0:
            raw = raw / total
        else:
            raw = np.full(n, 1.0 / n)

        return raw

    @staticmethod
    def _apply_weight_constraints(
        weights: np.ndarray, max_weight: float
    ) -> np.ndarray:
        """Enforce max weight and long-only constraints.

        Iteratively clips weights to max_weight and redistributes excess
        equally to all non-capped assets (including zero-weight ones).
        When all assets are at max_weight, remaining weight becomes cash.
        """
        w = weights.copy()
        w = np.maximum(w, 0.0)
        n = len(w)

        for _ in range(n + 1):  # at most n iterations to cap all assets
            excess_mask = w > max_weight
            if not excess_mask.any():
                break

            excess = (w[excess_mask] - max_weight).sum()
            w[excess_mask] = max_weight

            # Redistribute excess to all non-capped assets
            free_mask = ~excess_mask
            n_free = free_mask.sum()
            if n_free > 0:
                w[free_mask] += excess / n_free
            else:
                # All assets are capped — remainder is cash
                break

        # Final clip to ensure constraints
        w = np.clip(w, 0.0, max_weight)

        # If sum > 1 (shouldn't happen with proper constraints), normalize
        total = w.sum()
        if total > 1.0:
            w = w / total

        if total < 0.95:
            logger.info(
                "Portfolio %.1f%% invested (%.1f%% cash) due to max_weight=%.2f",
                total * 100, (1 - total) * 100, max_weight,
            )

        return w

    # ── Internal: portfolio returns ───────────────────────────────

    @staticmethod
    def _compute_portfolio_returns(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        tx_cost_bps: float,
    ) -> pd.Series:
        """Compute daily portfolio returns from weights and asset returns.

        Between rebalance dates, weights are held constant (forward-filled).
        Transaction costs are deducted on rebalance dates based on turnover.
        """
        tx_cost = tx_cost_bps / 10_000.0

        # Forward-fill weights to every trading day
        weights_daily = weights.reindex(returns.index, method="ffill")

        # Days before first rebalance have NaN — fill with 0 (cash)
        weights_daily = weights_daily.fillna(0.0)

        # Daily portfolio return = sum(w_i * r_i)
        portfolio_returns = (weights_daily * returns).sum(axis=1)

        # Deduct transaction costs on rebalance dates
        if tx_cost > 0:
            rebal_dates_set = set(weights.index)
            prev_w = None
            for date in weights.index:
                if date not in returns.index:
                    continue
                current_w = weights.loc[date].values
                if prev_w is not None:
                    turnover = np.abs(current_w - prev_w).sum()
                else:
                    turnover = np.abs(current_w).sum()
                portfolio_returns.loc[date] -= turnover * tx_cost
                prev_w = current_w.copy()

        return portfolio_returns

    @staticmethod
    def _compute_equity_curve(
        portfolio_returns: pd.Series,
        initial_capital: float,
    ) -> pd.Series:
        """Compute equity curve from portfolio returns."""
        return initial_capital * (1 + portfolio_returns).cumprod()

    @staticmethod
    def _compute_portfolio_metrics(
        portfolio_returns: pd.Series,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute portfolio performance metrics."""
        trading_days = cfg["trading_days_per_year"]
        risk_free = cfg["risk_free_rate"]

        ann_return = float(portfolio_returns.mean() * trading_days)
        ann_vol = float(portfolio_returns.std() * np.sqrt(trading_days))

        if ann_vol > 0:
            sharpe = (ann_return - risk_free) / ann_vol
        else:
            sharpe = 0.0

        # Diversification ratio = weighted avg of individual vols / portfolio vol
        # Use only data up to last rebalance date (no look-ahead)
        last_rebal_date = weights.index[-1]
        last_weights = weights.iloc[-1].values
        past_returns = returns.loc[:last_rebal_date]
        asset_vols = past_returns.std().values * np.sqrt(trading_days)
        weighted_avg_vol = (last_weights * asset_vols).sum()

        if ann_vol > 0:
            div_ratio = weighted_avg_vol / ann_vol
        else:
            div_ratio = 1.0

        return {
            "annualized_return": round(ann_return, 6),
            "annualized_volatility": round(ann_vol, 6),
            "sharpe_ratio": round(sharpe, 6),
            "diversification_ratio": round(max(div_ratio, 1.0), 6),
        }
