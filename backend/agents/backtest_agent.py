"""BacktestAgent — Event-driven backtesting engine.

Simulates trading day-by-day over historical data using model predictions.
Applies realistic transaction costs, volume-dependent slippage, market
impact, and short borrow costs.  Produces equity curves, trade logs, and
performance metrics.  Benchmarks against buy-and-hold (external benchmark
data or same-asset fallback).

Constraints:
  - No look-ahead bias: signal on bar i executes at open of bar i+1
  - Transaction costs: 5-10 bps per trade
  - Slippage + sqrt market-impact modeling on each execution
  - Volume participation limits (default 2 % of daily volume)
  - Short borrow costs (annualized, accrued daily)
  - Event-driven (step through time, no vectorized shortcuts that leak)
  - Forced liquidation at end of backtest period
  - max_position_size limits capital deployed per trade
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

VALID_SIGNALS = {-1, 0, 1}
_BPS_DIVISOR: float = 10_000.0


class BacktestAgent(BaseAgent):
    """Event-driven backtesting engine.

    Iterates through each timestep, generating trade decisions from
    predictions, applying transaction costs, slippage, market impact,
    and short borrow costs, and tracking portfolio value over time.

    Execution model
    ---------------
    * Signal generated at bar *i* is executed at the **open** price of
      bar *i + 1* to avoid look-ahead bias.
    * Open positions are force-liquidated at the end of the backtest so
      that all round-trip costs are captured.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "initial_capital": 100_000.0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "max_position_size": 1.0,
        "max_volume_participation": 0.02,
        "market_impact_factor": 0.1,
        "short_borrow_annual_bps": 50.0,
        "risk_free_rate": 0.0,
        "trading_days_per_year": 252,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "price_data": (
                "pd.DataFrame — OHLCV with DatetimeIndex, columns: "
                "open, high, low, close, volume"
            ),
            "predictions": (
                "pd.Series — trading signal per timestep. "
                "1 = long, 0 = flat, -1 = short. "
                "Index must align with price_data."
            ),
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
            "benchmark_data": (
                "(optional) pd.DataFrame — benchmark OHLCV for comparison. "
                "If not provided, buy-and-hold on price_data is used."
            ),
            "out_of_sample": (
                "(optional) bool — whether this run uses out-of-sample data"
            ),
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "equity_curve": "pd.Series — portfolio value indexed by date",
            "trade_log": (
                "list[dict] — each entry: date, action, price, shares, "
                "cost, slippage, portfolio_value"
            ),
            "performance_summary": (
                "dict — sharpe, sortino, max_drawdown, calmar, "
                "win_rate, turnover, total_return, annualized_return, "
                "total_trades, benchmark_comparison"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the event-driven backtest simulation.

        Execution model: signal at bar i executes at open of bar i+1.
        Bar 0's equity is always initial_capital (no trade on first bar).
        Open positions are force-liquidated at the end.

        Args:
            inputs: dict with 'price_data' (DataFrame), 'predictions'
                    (Series), optional 'config', 'benchmark_data', and
                    'out_of_sample'.

        Returns:
            dict with 'equity_curve', 'trade_log', 'performance_summary'.
        """
        price_data, predictions, cfg = self._validate_inputs(inputs)
        benchmark_data = inputs.get("benchmark_data")
        out_of_sample = inputs.get("out_of_sample")

        initial_capital = cfg["initial_capital"]
        tc_rate = cfg["transaction_cost_bps"] / _BPS_DIVISOR
        slip_rate = cfg["slippage_bps"] / _BPS_DIVISOR
        max_pos_size = cfg["max_position_size"]
        max_vol_part = cfg["max_volume_participation"]
        impact_factor = cfg["market_impact_factor"]
        borrow_daily = (
            cfg["short_borrow_annual_bps"]
            / _BPS_DIVISOR
            / cfg["trading_days_per_year"]
        )
        trading_days = cfg["trading_days_per_year"]
        risk_free = cfg["risk_free_rate"]

        # ── Event-driven simulation ──
        cash = initial_capital
        shares_held = 0
        equity_values: List[float] = []
        trade_log: List[Dict[str, Any]] = []
        pending_signal: Optional[int] = None

        for i in range(len(price_data)):
            date = price_data.index[i]
            open_price = float(price_data["open"].iloc[i])
            close_price = float(price_data["close"].iloc[i])
            volume = float(price_data["volume"].iloc[i])

            # Execute pending signal from previous bar
            if pending_signal is not None:
                target_direction = pending_signal
                current_direction = (
                    1 if shares_held > 0
                    else (-1 if shares_held < 0 else 0)
                )

                if target_direction != current_direction:
                    if shares_held != 0:
                        cash = self._close_position(
                            shares_held=shares_held,
                            exec_ref_price=open_price,
                            slip_rate=slip_rate,
                            tc_rate=tc_rate,
                            volume=volume,
                            impact_factor=impact_factor,
                            date=date,
                            trade_log=trade_log,
                            cash=cash,
                        )
                        shares_held = 0

                    if target_direction != 0:
                        cash, shares_held = self._open_position(
                            direction=target_direction,
                            open_price=open_price,
                            slip_rate=slip_rate,
                            tc_rate=tc_rate,
                            max_pos_size=max_pos_size,
                            volume=volume,
                            max_vol_participation=max_vol_part,
                            impact_factor=impact_factor,
                            date=date,
                            trade_log=trade_log,
                            cash=cash,
                        )

            # Accrue daily short borrow cost
            if shares_held < 0:
                cash -= abs(shares_held) * close_price * borrow_daily

            # Mark-to-market portfolio value
            if shares_held >= 0:
                portfolio_value = cash + shares_held * close_price
            else:
                portfolio_value = cash - abs(shares_held) * close_price

            equity_values.append(portfolio_value)

            # Record signal for next bar execution (no look-ahead)
            pending_signal = int(predictions.iloc[i])

        # ── Force-liquidate open positions at end of backtest ──
        if shares_held != 0:
            last_close = float(price_data["close"].iloc[-1])
            last_volume = float(price_data["volume"].iloc[-1])
            last_date = price_data.index[-1]

            cash = self._close_position(
                shares_held=shares_held,
                exec_ref_price=last_close,
                slip_rate=slip_rate,
                tc_rate=tc_rate,
                volume=last_volume,
                impact_factor=impact_factor,
                date=last_date,
                trade_log=trade_log,
                cash=cash,
                forced_liquidation=True,
            )
            shares_held = 0
            equity_values[-1] = cash

        equity_curve = pd.Series(
            equity_values, index=price_data.index, name="equity"
        )

        # ── Compute performance metrics ──
        performance = self._compute_metrics(
            equity_curve=equity_curve,
            trade_log=trade_log,
            price_data=price_data,
            benchmark_data=benchmark_data,
            initial_capital=initial_capital,
            trading_days=trading_days,
            risk_free=risk_free,
        )

        outputs: Dict[str, Any] = {
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "performance_summary": performance,
        }

        # ── Validate ──
        self.validate(inputs, outputs)

        # ── Record metrics ──
        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "total_return": performance["total_return"],
            "sharpe": performance["sharpe"],
            "sortino": performance["sortino"],
            "max_drawdown": performance["max_drawdown"],
            "calmar": performance["calmar"],
            "win_rate": performance["win_rate"],
            "turnover": performance["turnover"],
            "total_trades": performance["total_trades"],
            "initial_capital": initial_capital,
            "transaction_cost_bps": cfg["transaction_cost_bps"],
            "slippage_bps": cfg["slippage_bps"],
            "short_borrow_annual_bps": cfg["short_borrow_annual_bps"],
            "out_of_sample": out_of_sample,
        }

        logger.info(
            "BacktestAgent complete: Sharpe=%.3f, MaxDD=%.2f%%, Trades=%d",
            performance["sharpe"],
            performance["max_drawdown"] * 100,
            performance["total_trades"],
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:  # noqa: ARG002
        """Validate backtest outputs for integrity."""
        _ = inputs
        equity_curve = outputs["equity_curve"]

        if equity_curve.isna().any():
            raise ValueError("Equity curve contains NaN values")

        if (equity_curve <= 0).any():
            raise ValueError("Equity curve contains negative or zero values")

        required_trade_keys = {
            "date", "action", "price", "shares", "cost", "slippage",
            "portfolio_value",
        }
        for trade in outputs["trade_log"]:
            missing = required_trade_keys - set(trade.keys())
            if missing:
                raise ValueError(f"Trade log entry missing keys: {missing}")

        summary = outputs["performance_summary"]
        required_keys = {
            "sharpe", "sortino", "max_drawdown", "calmar",
            "win_rate", "turnover", "total_return", "annualized_return",
            "total_trades",
        }
        missing_keys = required_keys - set(summary.keys())
        if missing_keys:
            raise ValueError(
                f"Performance summary missing keys: {missing_keys}"
            )

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
            "experiment_id": f"backtest_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "BacktestAgent",
            "stage": "backtest",
            "timestamp": now.isoformat(),
            "parameters": {
                "transaction_costs_bps": self._metrics.get(
                    "transaction_cost_bps"
                ),
                "slippage_bps": self._metrics.get("slippage_bps"),
                "initial_capital": self._metrics.get("initial_capital"),
                "short_borrow_annual_bps": self._metrics.get(
                    "short_borrow_annual_bps"
                ),
            },
            "out_of_sample": self._metrics.get("out_of_sample"),
            "metrics": {
                "sharpe": self._metrics.get("sharpe"),
                "sortino": self._metrics.get("sortino"),
                "max_drawdown": self._metrics.get("max_drawdown"),
                "calmar": self._metrics.get("calmar"),
                "win_rate": self._metrics.get("win_rate"),
                "turnover": self._metrics.get("turnover"),
                "total_return": self._metrics.get("total_return"),
                "total_trades": self._metrics.get("total_trades"),
            },
            "notes": "BacktestAgent event-driven simulation run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"backtest_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: slippage / volume helpers ────────────────────────

    @staticmethod
    def _effective_slippage(
        base_slip_rate: float,
        shares: int,
        volume: float,
        impact_factor: float,
    ) -> float:
        """Volume-adjusted slippage rate.

        effective = base_rate * (1 + impact_factor * sqrt(shares / volume))

        The sqrt market-impact component penalises large orders relative
        to daily volume, while the base rate covers fixed microstructure
        costs.
        """
        if volume <= 0 or shares <= 0:
            return base_slip_rate
        participation = shares / volume
        return base_slip_rate * (1.0 + impact_factor * np.sqrt(participation))

    @staticmethod
    def _cap_shares_by_volume(
        desired_shares: int,
        volume: float,
        max_vol_participation: float,
    ) -> int:
        """Limit order size to a fraction of daily volume."""
        if volume <= 0 or max_vol_participation <= 0:
            return desired_shares
        max_from_volume = int(volume * max_vol_participation)
        return min(desired_shares, max(max_from_volume, 1))

    # ── Internal: position management ─────────────────────────────

    @staticmethod
    def _close_position(
        shares_held: int,
        exec_ref_price: float,
        slip_rate: float,
        tc_rate: float,
        volume: float,
        impact_factor: float,
        date: Any,
        trade_log: List[Dict[str, Any]],
        cash: float,
        forced_liquidation: bool = False,
    ) -> float:
        """Close an existing long or short position.

        Always closes the *full* position.  Market impact is computed on
        the full order size (no volume cap — you must exit).

        Returns updated cash.
        """
        abs_shares = abs(shares_held)
        eff_slip = BacktestAgent._effective_slippage(
            slip_rate, abs_shares, volume, impact_factor,
        )

        if shares_held > 0:
            exec_price = exec_ref_price * (1.0 - eff_slip)
            trade_value = abs_shares * exec_price
            cost = trade_value * tc_rate
            slippage_amount = abs_shares * exec_ref_price * eff_slip
            cash += trade_value - cost
            action = "LIQUIDATE_LONG" if forced_liquidation else "SELL"
        else:
            exec_price = exec_ref_price * (1.0 + eff_slip)
            trade_value = abs_shares * exec_price
            cost = trade_value * tc_rate
            slippage_amount = abs_shares * exec_ref_price * eff_slip
            cash -= trade_value + cost
            action = "LIQUIDATE_SHORT" if forced_liquidation else "COVER"

        trade_log.append({
            "date": str(date),
            "action": action,
            "price": round(exec_price, 6),
            "shares": abs_shares,
            "cost": round(cost, 6),
            "slippage": round(slippage_amount, 6),
            "portfolio_value": round(cash, 2),
        })

        return cash

    @staticmethod
    def _open_position(
        direction: int,
        open_price: float,
        slip_rate: float,
        tc_rate: float,
        max_pos_size: float,
        volume: float,
        max_vol_participation: float,
        impact_factor: float,
        date: Any,
        trade_log: List[Dict[str, Any]],
        cash: float,
    ) -> Tuple[float, int]:
        """Open a new long or short position at the bar's open price.

        Sizing uses exec_price (slippage-adjusted) for both long and
        short to ensure symmetric cost accounting.  Shares are further
        capped by ``max_volume_participation * daily_volume``.

        Portfolio value in the trade log uses ``open_price`` (the only
        price known at execution time) — never the close.

        Returns updated (cash, shares_held).
        """
        deployable_cash = cash * max_pos_size
        shares_held = 0

        if direction == 1:
            # Phase 1 — estimate size with base slippage
            base_exec = open_price * (1.0 + slip_rate)
            est_shares = int(deployable_cash / (base_exec * (1.0 + tc_rate)))
            est_shares = BacktestAgent._cap_shares_by_volume(
                est_shares, volume, max_vol_participation,
            )

            if est_shares > 0:
                # Phase 2 — volume-adjusted slippage
                eff_slip = BacktestAgent._effective_slippage(
                    slip_rate, est_shares, volume, impact_factor,
                )
                exec_price = open_price * (1.0 + eff_slip)

                # Phase 3 — re-size conservatively with effective cost
                max_shares = min(
                    int(deployable_cash / (exec_price * (1.0 + tc_rate))),
                    est_shares,
                )

                if max_shares > 0:
                    trade_value = max_shares * exec_price
                    cost = trade_value * tc_rate
                    slippage_amount = max_shares * open_price * eff_slip
                    cash -= trade_value + cost
                    shares_held = max_shares

                    trade_log.append({
                        "date": str(date),
                        "action": "BUY",
                        "price": round(exec_price, 6),
                        "shares": max_shares,
                        "cost": round(cost, 6),
                        "slippage": round(slippage_amount, 6),
                        "portfolio_value": round(
                            cash + shares_held * open_price, 2
                        ),
                    })

        else:
            # Phase 1 — estimate size with base slippage (symmetric)
            base_exec = open_price * (1.0 - slip_rate)
            est_shares = int(deployable_cash / (base_exec * (1.0 + tc_rate)))
            est_shares = BacktestAgent._cap_shares_by_volume(
                est_shares, volume, max_vol_participation,
            )

            if est_shares > 0:
                eff_slip = BacktestAgent._effective_slippage(
                    slip_rate, est_shares, volume, impact_factor,
                )
                exec_price = open_price * (1.0 - eff_slip)

                max_shares = min(
                    int(deployable_cash / (exec_price * (1.0 + tc_rate))),
                    est_shares,
                )

                if max_shares > 0:
                    trade_value = max_shares * exec_price
                    cost = trade_value * tc_rate
                    slippage_amount = max_shares * open_price * eff_slip
                    cash += trade_value - cost
                    shares_held = -max_shares

                    trade_log.append({
                        "date": str(date),
                        "action": "SHORT",
                        "price": round(exec_price, 6),
                        "shares": max_shares,
                        "cost": round(cost, 6),
                        "slippage": round(slippage_amount, 6),
                        "portfolio_value": round(
                            cash - abs(shares_held) * open_price, 2
                        ),
                    })

        return cash, shares_held

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Validate and extract inputs."""
        if "price_data" not in inputs:
            raise ValueError("inputs must contain 'price_data'")
        if "predictions" not in inputs:
            raise ValueError("inputs must contain 'predictions'")

        price_data = inputs["price_data"]
        predictions = inputs["predictions"]

        if not isinstance(price_data, pd.DataFrame):
            raise TypeError(
                f"'price_data' must be a pd.DataFrame, got {type(price_data)}"
            )

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(price_data.columns)
        if missing:
            raise ValueError(f"'price_data' missing required columns: {missing}")

        if not isinstance(predictions, pd.Series):
            raise TypeError(
                f"'predictions' must be a pd.Series, got {type(predictions)}"
            )

        # Align predictions to price_data index first
        common_index = price_data.index.intersection(predictions.index)
        if len(common_index) == 0:
            raise ValueError(
                "No overlap between price_data and predictions indices — "
                "cannot align"
            )

        price_data = price_data.loc[common_index]
        predictions = predictions.loc[common_index]

        # Validate signal values AFTER alignment
        if predictions.isna().any():
            raise ValueError("Predictions contain NaN values after alignment")

        unique_signals = set(predictions.unique())
        invalid = unique_signals - VALID_SIGNALS
        if invalid:
            raise ValueError(
                f"Signal values must be in {{-1, 0, 1}}, got invalid values: {invalid}"
            )

        cfg = {**self._config, **(inputs.get("config") or {})}

        return price_data, predictions, cfg

    # ── Internal: performance metric computation ──────────────────

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        trade_log: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        initial_capital: float,
        trading_days: int,
        risk_free: float,
    ) -> Dict[str, Any]:
        """Compute all performance metrics from equity curve and trade log."""
        daily_returns = equity_curve.pct_change().fillna(0.0)
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        n_days = len(equity_curve)
        years = n_days / trading_days

        if years > 0 and equity_curve.iloc[-1] > 0:
            annualized_return = (
                (equity_curve.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0
            )
        else:
            annualized_return = 0.0

        sharpe = self._sharpe_ratio(daily_returns, risk_free, trading_days)
        sortino = self._sortino_ratio(daily_returns, risk_free, trading_days)
        max_drawdown = self._max_drawdown(equity_curve)

        if max_drawdown != 0.0:
            calmar = annualized_return / abs(max_drawdown)
        else:
            calmar = 0.0

        win_rate = self._win_rate(trade_log)
        turnover = self._turnover(trade_log, equity_curve)
        total_trades = len(trade_log)

        # Benchmark: use external data if provided, else same-asset
        # buy-and-hold.  Entry at bar 1's open (first possible execution)
        # to match strategy timing.
        bm = benchmark_data if benchmark_data is not None else price_data
        if len(bm) >= 2:
            bench_entry = float(bm["open"].iloc[1])
            bench_exit = float(bm["close"].iloc[-1])
        else:
            bench_entry = bench_exit = 1.0

        benchmark_return = (
            (bench_exit - bench_entry) / bench_entry
            if bench_entry != 0
            else 0.0
        )

        return {
            "sharpe": round(sharpe, 6),
            "sortino": round(sortino, 6),
            "max_drawdown": round(max_drawdown, 6),
            "calmar": round(calmar, 6),
            "win_rate": round(win_rate, 6),
            "turnover": round(turnover, 6),
            "total_return": round(total_return, 6),
            "annualized_return": round(annualized_return, 6),
            "total_trades": total_trades,
            "benchmark_comparison": {
                "strategy_return": round(total_return, 6),
                "benchmark_return": round(benchmark_return, 6),
                "excess_return": round(total_return - benchmark_return, 6),
            },
        }

    @staticmethod
    def _sharpe_ratio(
        daily_returns: pd.Series, risk_free: float, trading_days: int
    ) -> float:
        """Annualized Sharpe ratio."""
        excess = daily_returns - risk_free / trading_days
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * np.sqrt(trading_days))

    @staticmethod
    def _sortino_ratio(
        daily_returns: pd.Series, risk_free: float, trading_days: int
    ) -> float:
        """Annualized Sortino ratio using proper downside deviation.

        DD = sqrt( mean( min(excess_i, 0)^2 ) )  over ALL observations.
        """
        excess = daily_returns - risk_free / trading_days
        downside_diff = np.minimum(excess.values, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_dev == 0 or np.isnan(downside_dev):
            return 0.0
        return float((excess.mean() / downside_dev) * np.sqrt(trading_days))

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        """Maximum drawdown as a negative fraction (e.g., -0.15 = 15% drawdown)."""
        values = equity_curve.values.astype(float)
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / np.where(cummax == 0, 1.0, cummax)
        return float(np.min(drawdown))

    @staticmethod
    def _win_rate(trade_log: List[Dict[str, Any]]) -> float:
        """Win rate from paired round trips.

        Pairs entries (BUY / SHORT) with exits (SELL / COVER /
        LIQUIDATE_LONG / LIQUIDATE_SHORT).
        """
        if not trade_log:
            return 0.0

        entry_actions = {"BUY", "SHORT"}
        exit_actions = {"SELL", "COVER", "LIQUIDATE_LONG", "LIQUIDATE_SHORT"}

        profits: List[float] = []
        open_trade: Optional[Dict[str, Any]] = None

        for trade in trade_log:
            action = trade["action"]
            if action in entry_actions:
                open_trade = trade
            elif action in exit_actions and open_trade is not None:
                matched_shares = min(trade["shares"], open_trade["shares"])
                if open_trade["action"] == "BUY":
                    pnl = (
                        matched_shares * (trade["price"] - open_trade["price"])
                        - trade["cost"] - open_trade["cost"]
                    )
                else:  # SHORT → COVER / LIQUIDATE_SHORT
                    pnl = (
                        matched_shares * (open_trade["price"] - trade["price"])
                        - trade["cost"] - open_trade["cost"]
                    )
                profits.append(pnl)
                open_trade = None

        if not profits:
            return 0.0
        wins = sum(1 for p in profits if p > 0)
        return wins / len(profits)

    @staticmethod
    def _turnover(
        trade_log: List[Dict[str, Any]],
        equity_curve: pd.Series,
    ) -> float:
        """Turnover as total traded value / average portfolio value."""
        if not trade_log:
            return 0.0
        total_traded = sum(t["price"] * t["shares"] for t in trade_log)
        avg_equity = equity_curve.mean()
        if avg_equity == 0:
            return 0.0
        return total_traded / avg_equity
