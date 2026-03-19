"""RiskAgent — Risk-aware position sizing.

Computes position sizes per timestep using volatility scaling or fractional
Kelly criterion, subject to risk constraints (max position size, stop-loss,
portfolio VaR limit).  Produces risk metrics including VaR, CVaR, and max
exposure.

Constraints (from specs/risk_spec.md):
  - Position sizes computed using only past data (no look-ahead)
  - Kelly sizing uses out-of-sample win rate and payoff ratio
  - Volatility estimates use rolling windows (not full-sample std)
  - Max single position: configurable, default 10% of portfolio
  - Stop-loss: zero positions when cumulative drawdown exceeds threshold
  - Portfolio-level VaR limit: scale down positions when breached
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """Risk-aware position sizing agent.

    Supports two sizing methods:
      - volatility_scaling: size inversely to recent rolling volatility
      - kelly: fractional Kelly criterion from trade statistics

    Both methods respect max position size, stop-loss, and VaR constraints.
    """

    DEFAULT_CONFIG: MappingProxyType = MappingProxyType({
        "sizing_method": "volatility_scaling",
        "volatility_window": 20,
        "target_volatility": 0.15,
        "kelly_fraction": 0.5,
        "max_position_size": 0.10,
        "stop_loss_threshold": 0.10,
        "var_limit": 0.05,
        "trading_days_per_year": 252,
    })

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ────────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "returns": "pd.Series — daily returns indexed by date",
            "price_data": "pd.DataFrame — OHLCV with DatetimeIndex",
            "signals": "pd.Series — trading signals (-1, 0, 1)",
            "trade_log": "(optional) list[dict] — trades from BacktestAgent",
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "position_sizes": "pd.Series — fraction of capital per timestep",
            "risk_metrics": (
                "dict — var_95 (negative = loss), var_99 (negative = loss), "
                "cvar_95 (negative = loss, <= var_95), "
                "max_position_exposure (non-negative fraction), "
                "var_limit_breaches (int count). "
                "Sign convention: VaR/CVaR are left-tail percentiles of "
                "position-weighted returns; negative values indicate losses."
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        returns, price_data, signals, trade_log, cfg = self._validate_inputs(
            inputs
        )

        sizing_method = cfg["sizing_method"]
        vol_window = cfg["volatility_window"]
        max_pos = cfg["max_position_size"]
        stop_loss = cfg["stop_loss_threshold"]
        var_limit = cfg["var_limit"]
        trading_days = cfg["trading_days_per_year"]

        if sizing_method == "kelly":
            raw_sizes = self._kelly_sizing(
                signals=signals,
                trade_log=trade_log,
                kelly_fraction=cfg["kelly_fraction"],
            )
        else:
            raw_sizes = self._volatility_scaling(
                returns=returns,
                signals=signals,
                vol_window=vol_window,
                target_vol=cfg["target_volatility"],
                trading_days=trading_days,
            )

        position_sizes = self._apply_constraints(
            raw_sizes=raw_sizes,
            returns=returns,
            max_pos=max_pos,
            stop_loss=stop_loss,
            var_limit=var_limit,
            vol_window=vol_window,
        )

        risk_metrics = self._compute_risk_metrics(
            returns=returns,
            position_sizes=position_sizes,
            var_limit=var_limit,
        )

        outputs: Dict[str, Any] = {
            "position_sizes": position_sizes,
            "risk_metrics": risk_metrics,
        }

        self.validate(inputs, outputs)

        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "sizing_method": sizing_method,
            "max_position_size": max_pos,
            "stop_loss_threshold": stop_loss,
            "var_95": risk_metrics["var_95"],
            "var_99": risk_metrics["var_99"],
            "cvar_95": risk_metrics["cvar_95"],
            "max_position_exposure": risk_metrics["max_position_exposure"],
            "var_limit_breaches": risk_metrics["var_limit_breaches"],
            "mean_position_size": float(position_sizes.abs().mean()),
        }

        logger.info(
            "RiskAgent complete: method=%s, VaR95=%.4f, max_exposure=%.4f",
            sizing_method,
            risk_metrics["var_95"],
            risk_metrics["max_position_exposure"],
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        _ = inputs
        position_sizes = outputs["position_sizes"]

        if position_sizes.isna().any():
            raise ValueError("Position sizes contain NaN values")

        max_pos = self._config["max_position_size"]
        if (position_sizes.abs() > max_pos + 1e-10).any():
            raise ValueError(
                f"Position sizes exceed max_position_size ({max_pos})"
            )

        required = {"var_95", "var_99", "cvar_95", "max_position_exposure"}
        missing = required - set(outputs["risk_metrics"].keys())
        if missing:
            raise ValueError(f"Risk metrics missing keys: {missing}")

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
            "experiment_id": f"risk_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "RiskAgent",
            "stage": "risk_management",
            "timestamp": now.isoformat(),
            "parameters": {
                "sizing_method": self._metrics.get("sizing_method"),
                "max_position_size": self._metrics.get("max_position_size"),
                "stop_loss_threshold": self._metrics.get("stop_loss_threshold"),
            },
            "metrics": {
                "var_95": self._metrics.get("var_95"),
                "var_99": self._metrics.get("var_99"),
                "cvar_95": self._metrics.get("cvar_95"),
                "max_position_exposure": self._metrics.get(
                    "max_position_exposure"
                ),
                "var_limit_breaches": self._metrics.get("var_limit_breaches"),
                "mean_position_size": self._metrics.get("mean_position_size"),
            },
            "notes": "RiskAgent position sizing run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"risk_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series, List[Dict[str, Any]], Dict[str, Any]]:
        if "returns" not in inputs:
            raise ValueError("inputs must contain 'returns'")
        if "price_data" not in inputs:
            raise ValueError("inputs must contain 'price_data'")
        if "signals" not in inputs:
            raise ValueError("inputs must contain 'signals'")

        returns = inputs["returns"]
        price_data = inputs["price_data"]
        signals = inputs["signals"]
        trade_log = inputs.get("trade_log", [])

        if not isinstance(returns, pd.Series):
            raise TypeError(
                f"'returns' must be a pd.Series, got {type(returns)}"
            )
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError(
                f"'price_data' must be a pd.DataFrame, got {type(price_data)}"
            )
        if not isinstance(signals, pd.Series):
            raise TypeError(
                f"'signals' must be a pd.Series, got {type(signals)}"
            )
        if returns.isna().any():
            raise ValueError("returns contain NaN values")

        cfg = {**self._config, **(inputs.get("config") or {})}
        return returns, price_data, signals, trade_log, cfg

    # ── Internal: sizing methods ──────────────────────────────────

    @staticmethod
    def _volatility_scaling(
        returns: pd.Series,
        signals: pd.Series,
        vol_window: int,
        target_vol: float,
        trading_days: int,
    ) -> pd.Series:
        """Size positions inversely to rolling annualized volatility.

        position_size = signal * target_vol / rolling_annualized_vol

        Uses rolling window with expanding fallback for warm-up period.
        No look-ahead: at time t, only data up to t is used.
        """
        annualized_factor = np.sqrt(trading_days)
        rolling_vol = (
            returns.rolling(window=vol_window, min_periods=2).std()
            * annualized_factor
        )

        raw_sizes = pd.Series(0.0, index=returns.index)
        valid_mask = (rolling_vol > 0) & rolling_vol.notna()
        raw_sizes[valid_mask] = (
            signals[valid_mask] * target_vol / rolling_vol[valid_mask]
        )

        return raw_sizes.replace([np.inf, -np.inf], 0.0)

    @staticmethod
    def _kelly_sizing(
        signals: pd.Series,
        trade_log: List[Dict[str, Any]],
        kelly_fraction: float,
    ) -> pd.Series:
        """Fractional Kelly criterion with expanding window over trade history.

        At each timestep t, only round-trip trades that closed *before* t are
        used to compute win rate and payoff ratio.  This prevents look-ahead
        bias per specs/risk_spec.md: "Kelly sizing must use out-of-sample
        win rate and payoff ratio, not in-sample."
        """
        round_trips = RiskAgent._build_round_trips(trade_log)
        raw_sizes = pd.Series(0.0, index=signals.index)

        if not round_trips:
            return raw_sizes

        round_trips.sort(key=lambda rt: rt["exit_date"])

        trip_idx = 0
        wins: List[float] = []
        losses: List[float] = []

        for i, date in enumerate(signals.index):
            date_str = str(date)[:10]

            # Accumulate only trips that closed strictly before this date
            while (
                trip_idx < len(round_trips)
                and round_trips[trip_idx]["exit_date"] < date_str
            ):
                pnl = round_trips[trip_idx]["pnl"]
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
                trip_idx += 1

            total = len(wins) + len(losses)
            if total == 0:
                continue

            win_rate = len(wins) / total
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0

            kelly_f = RiskAgent._compute_kelly_fraction(
                win_rate, avg_win, avg_loss
            )
            kelly_f = max(kelly_f * kelly_fraction, 0.0)
            raw_sizes.iloc[i] = signals.iloc[i] * kelly_f

        return raw_sizes

    @staticmethod
    def _build_round_trips(
        trade_log: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pair entry/exit trades into round trips with PnL and exit date.

        Handles consecutive entries by logging a warning and keeping the
        latest entry (conservative: earlier entry is treated as abandoned).
        Break-even trades (PnL == 0 after costs) are counted as losses.

        Returns:
            list of dicts with keys: exit_date (str), pnl (float).
        """
        entry_actions = {"BUY", "SHORT"}
        exit_actions = {"SELL", "COVER", "LIQUIDATE_LONG", "LIQUIDATE_SHORT"}

        round_trips: List[Dict[str, Any]] = []
        open_trade: Optional[Dict[str, Any]] = None

        for trade in trade_log:
            action = trade["action"]
            if action in entry_actions:
                if open_trade is not None:
                    logger.warning(
                        "Consecutive entry at %s without prior exit — "
                        "dropping earlier entry at %s",
                        trade["date"],
                        open_trade["date"],
                    )
                open_trade = trade
            elif action in exit_actions and open_trade is not None:
                shares = min(trade["shares"], open_trade["shares"])
                if open_trade["action"] == "BUY":
                    pnl = shares * (trade["price"] - open_trade["price"])
                else:
                    pnl = shares * (open_trade["price"] - trade["price"])
                pnl -= trade["cost"] + open_trade["cost"]

                round_trips.append({
                    "exit_date": trade["date"],
                    "pnl": pnl,
                })
                open_trade = None

        return round_trips

    @staticmethod
    def _compute_kelly_fraction(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Compute raw Kelly fraction from trade statistics.

        Args:
            win_rate: fraction of winning trades (0-1).
            avg_win: mean profit on winning trades (positive magnitude).
            avg_loss: mean loss on losing trades (positive magnitude).

        Returns:
            Raw Kelly fraction (may be negative for losing strategies).
        """
        if avg_loss == 0 and avg_win == 0:
            return 0.0
        if avg_loss == 0:
            return win_rate
        if avg_win == 0:
            return 0.0
        payoff = avg_win / avg_loss
        return win_rate - (1 - win_rate) / payoff

    # ── Internal: constraints ─────────────────────────────────────

    @staticmethod
    def _apply_stop_loss(
        position_sizes: pd.Series,
        returns: pd.Series,
        stop_loss: float,
    ) -> pd.Series:
        """Zero out positions during stop-loss periods.

        Uses lagged drawdown (shift by 1) so the decision at bar i is
        based on drawdown known at the close of bar i-1, preventing
        look-ahead into bar i's return.
        """
        result = position_sizes.copy()

        cumulative = (1 + returns).cumprod()
        rolling_high = cumulative.expanding().max()
        drawdown = (cumulative - rolling_high) / rolling_high
        drawdown_lagged = drawdown.shift(1)

        stopped_out = False
        for i in range(len(result)):
            dd = drawdown_lagged.iloc[i]
            if pd.isna(dd):
                continue

            if stopped_out:
                if dd > -stop_loss * 0.5:
                    stopped_out = False
                else:
                    result.iloc[i] = 0.0
                    continue

            if dd < -stop_loss:
                stopped_out = True
                result.iloc[i] = 0.0

        return result

    @staticmethod
    def _apply_var_scaling(
        position_sizes: pd.Series,
        returns: pd.Series,
        var_limit: float,
        vol_window: int,
    ) -> pd.Series:
        """Scale down positions when rolling VaR exceeds limit.

        Uses lagged rolling VaR (shift by 1) to prevent look-ahead.
        """
        result = position_sizes.copy()

        if vol_window < 2 or len(returns) <= vol_window:
            return result

        rolling_var = returns.rolling(window=vol_window).quantile(0.05)
        rolling_var_lagged = rolling_var.shift(1)

        for i in range(vol_window + 1, len(result)):
            rv = rolling_var_lagged.iloc[i]
            if pd.notna(rv) and rv < -var_limit:
                scale = var_limit / abs(rv)
                result.iloc[i] *= min(scale, 1.0)

        return result

    @staticmethod
    def _apply_constraints(
        raw_sizes: pd.Series,
        returns: pd.Series,
        max_pos: float,
        stop_loss: float,
        var_limit: float,
        vol_window: int,
    ) -> pd.Series:
        """Apply risk constraints to raw position sizes.

        1. Clamp to max_position_size
        2. Zero out during stop-loss periods (lagged drawdown > threshold)
        3. Scale down when lagged rolling VaR exceeds limit
        """
        clamped = raw_sizes.clip(-max_pos, max_pos)
        after_stop = RiskAgent._apply_stop_loss(clamped, returns, stop_loss)
        return RiskAgent._apply_var_scaling(
            after_stop, returns, var_limit, vol_window
        )

    # ── Internal: risk metrics ────────────────────────────────────

    @staticmethod
    def _compute_risk_metrics(
        returns: pd.Series,
        position_sizes: pd.Series,
        var_limit: float,
    ) -> Dict[str, Any]:
        """Compute VaR, CVaR, max exposure, and VaR limit breach count.

        Sign convention: VaR and CVaR are left-tail percentiles of
        position-weighted returns.  Negative values indicate losses.
        Example: var_95 = -0.012 means the worst 5% of days lose >= 1.2%.
        """
        adjusted_returns = returns * position_sizes.abs()
        clean = adjusted_returns.dropna()

        if len(clean) == 0:
            return {
                "var_95": 0.0,
                "var_99": 0.0,
                "cvar_95": 0.0,
                "max_position_exposure": 0.0,
                "var_limit_breaches": 0,
            }

        var_95 = float(np.percentile(clean, 5))
        var_99 = float(np.percentile(clean, 1))

        tail = clean[clean <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

        max_exposure = float(position_sizes.abs().max())
        var_limit_breaches = int((clean < -var_limit).sum())

        return {
            "var_95": round(var_95, 6),
            "var_99": round(var_99, 6),
            "cvar_95": round(cvar_95, 6),
            "max_position_exposure": round(max_exposure, 6),
            "var_limit_breaches": var_limit_breaches,
        }
