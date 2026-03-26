"""Quantum Backtesting Engine — event-driven strategy evaluation.

Steps through time, rebalances at configured frequency using either
classical (Markowitz) or quantum (QAOA) portfolio optimization.
Includes transaction costs, slippage, and realistic constraints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quantum.solvers.classical_solvers import solve_markowitz_scipy
from quantum.solvers.problem_encodings import decode_binary_weights, portfolio_to_qubo
from quantum.solvers.qaoa_solver import QAOASolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade record."""
    date: pd.Timestamp
    ticker: str
    shares: float
    price: float
    cost: float

    @property
    def notional(self) -> float:
        return abs(self.shares * self.price)


@dataclass
class BacktestResult:
    """Full backtest output."""
    portfolio_values: np.ndarray
    trades: List[Trade]
    rebalance_dates: List[pd.Timestamp]
    metrics: Dict[str, float]
    total_transaction_costs: float
    optimizer_name: str
    weight_history: List[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

def rebalance_portfolio(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    prices: np.ndarray,
    capital: float,
    cost_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> List[Trade]:
    """Compute trades needed to move from current to target weights.

    Returns list of Trade objects with transaction costs and slippage.
    """
    n = len(current_weights)
    weight_diff = target_weights - current_weights
    trades: List[Trade] = []

    cost_rate = cost_bps / 10_000
    slip_rate = slippage_bps / 10_000

    for i in range(n):
        if abs(weight_diff[i]) < 1e-6:
            continue

        notional = abs(weight_diff[i]) * capital
        shares = notional / prices[i] * np.sign(weight_diff[i])
        tc = notional * (cost_rate + slip_rate)

        trades.append(Trade(
            date=pd.Timestamp.now(),
            ticker=f"ASSET_{i}",
            shares=float(shares),
            price=float(prices[i]),
            cost=float(tc),
        ))

    return trades


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(portfolio_values: np.ndarray) -> Dict[str, float]:
    """Compute standard performance metrics from portfolio value series."""
    if len(portfolio_values) < 2:
        return {"total_return": 0.0, "annualized_return": 0.0,
                "sharpe_ratio": 0.0, "max_drawdown": 0.0, "volatility": 0.0}

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = float(portfolio_values[-1] / portfolio_values[0] - 1)
    n_days = len(portfolio_values) - 1
    annualized_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

    vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = float(annualized_return / vol) if vol > 1e-10 else 0.0

    # Max drawdown
    peak = portfolio_values[0]
    max_dd = 0.0
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(max_dd),
        "volatility": vol,
    }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class QuantumBacktester:
    """Event-driven backtester with pluggable quantum/classical optimizers."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "initial_capital": 100_000,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
        "rebalance_frequency": 21,
        "optimizer": "classical",
        "max_weight": 0.50,
        "lookback_window": 60,
        "qaoa_layers": 2,
        "weight_precision_bits": 3,
        "qaoa_seed": None,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}

    def run(self, returns: pd.DataFrame) -> BacktestResult:
        """Run backtest on daily returns DataFrame.

        Walks through time, rebalances at configured frequency using
        only past data (no look-ahead).
        """
        cfg = self._config
        n_assets = returns.shape[1]
        n_days = len(returns)
        lookback = cfg["lookback_window"]
        rebal_freq = cfg["rebalance_frequency"]
        initial_capital = cfg["initial_capital"]

        portfolio_values = [float(initial_capital)]
        current_weights = np.ones(n_assets) / n_assets  # equal weight start
        all_trades: List[Trade] = []
        rebalance_dates: List[pd.Timestamp] = []
        weight_history: List[np.ndarray] = [current_weights.copy()]
        total_tc = 0.0

        capital = float(initial_capital)

        for t in range(n_days):
            date = returns.index[t]
            day_returns = returns.iloc[t].values

            # Update portfolio value
            capital *= float(1.0 + np.dot(current_weights, day_returns))

            # Rebalance check
            if t > 0 and t % rebal_freq == 0 and t >= lookback:
                past_returns = returns.iloc[t - lookback:t]
                target_weights = self._optimize(past_returns, cfg)

                prices = np.ones(n_assets) * 100.0  # normalized
                trades = rebalance_portfolio(
                    current_weights, target_weights, prices,
                    capital, cfg["transaction_cost_bps"], cfg["slippage_bps"],
                )

                # Apply transaction costs
                tc = sum(tr.cost for tr in trades)
                capital -= tc
                total_tc += tc

                # Update trade records
                for tr in trades:
                    all_trades.append(Trade(
                        date=date, ticker=tr.ticker,
                        shares=tr.shares, price=tr.price, cost=tr.cost,
                    ))

                current_weights = target_weights
                rebalance_dates.append(date)
                weight_history.append(current_weights.copy())

            portfolio_values.append(capital)

        pv = np.array(portfolio_values)
        metrics = compute_performance_metrics(pv)

        return BacktestResult(
            portfolio_values=pv,
            trades=all_trades,
            rebalance_dates=rebalance_dates,
            metrics=metrics,
            total_transaction_costs=total_tc,
            optimizer_name=cfg["optimizer"],
            weight_history=weight_history,
        )

    def compare(
        self,
        returns: pd.DataFrame,
        optimizers: List[str],
        qaoa_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, BacktestResult]:
        """Run backtests with multiple optimizers and compare."""
        results = {}
        for opt_name in optimizers:
            cfg = {**self._config, "optimizer": opt_name}
            if opt_name == "qaoa" and qaoa_config:
                cfg.update(qaoa_config)
            bt = QuantumBacktester(config=cfg)
            results[opt_name] = bt.run(returns)
        return results

    def _optimize(
        self, past_returns: pd.DataFrame, cfg: Dict[str, Any]
    ) -> np.ndarray:
        """Run portfolio optimization on historical returns."""
        mu = np.array(past_returns.mean())
        lw = LedoitWolf().fit(past_returns.values)
        cov = lw.covariance_
        n_assets = len(mu)
        target_return = float(np.mean(mu))
        max_weight = cfg["max_weight"]

        if cfg["optimizer"] == "qaoa":
            return self._optimize_qaoa(mu, cov, target_return, cfg)

        # Classical fallback
        result = solve_markowitz_scipy(
            expected_returns=mu,
            covariance=cov,
            target_return=target_return,
            max_weight=max_weight,
        )
        return result.weights

    def _optimize_qaoa(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        """Run QAOA portfolio optimization."""
        solver = QAOASolver(config={
            "n_layers": cfg.get("qaoa_layers", 2),
            "optimizer": "COBYLA",
            "maxiter": 200,
            "n_shots": 1024,
            "seed": cfg.get("qaoa_seed"),
        })

        result = solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=target_return,
            max_weight=cfg["max_weight"],
            n_bits=cfg.get("weight_precision_bits", 3),
        )

        decoded = decode_binary_weights(
            result.weights, len(mu),
            cfg.get("weight_precision_bits", 3),
            cfg["max_weight"],
        )
        return decoded
