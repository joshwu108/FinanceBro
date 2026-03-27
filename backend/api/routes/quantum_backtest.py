"""Quantum backtester API — 4-mode noise-aware comparison."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


class BacktestRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=8, example=["AAPL", "MSFT", "GOOG"])
    period: str = Field(default="2y", description="Data lookback period")
    initial_capital: float = Field(default=100_000, gt=0)
    transaction_cost_bps: float = Field(default=5, ge=0, le=50)
    slippage_bps: float = Field(default=3, ge=0, le=50)
    rebalance_frequency: int = Field(default=21, ge=5, le=63)
    max_weight: float = Field(default=0.50, ge=0.10, le=1.0)
    lookback_window: int = Field(default=60, ge=20, le=252)
    qaoa_layers: int = Field(default=2, ge=1, le=4)
    single_qubit_error: float = Field(default=0.01, ge=0, le=0.2)
    two_qubit_error: float = Field(default=0.02, ge=0, le=0.3)
    readout_error: float = Field(default=0.01, ge=0, le=0.2)


def _fetch_returns(tickers: List[str], period: str):
    """Fetch real returns, fall back to synthetic."""
    try:
        import yfinance as yf
        import pandas as pd

        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        if len(returns) < 100:
            raise ValueError("Insufficient data for backtesting")
        return returns
    except Exception as e:
        logger.warning("yfinance failed (%s), using synthetic returns", e)
        import pandas as pd

        rng = np.random.default_rng(42)
        n_days = 504
        dates = pd.bdate_range("2022-01-01", periods=n_days)
        data = {}
        for i, t in enumerate(tickers):
            mu = 0.0003 + i * 0.0001
            sigma = 0.01 + i * 0.002
            data[t] = rng.normal(mu, sigma, n_days)
        return pd.DataFrame(data, index=dates)


def _serialize_backtest_result(result) -> Dict[str, Any]:
    """Convert BacktestResult dataclass to JSON-safe dict."""
    return {
        "portfolio_values": result.portfolio_values.tolist(),
        "metrics": result.metrics,
        "total_transaction_costs": result.total_transaction_costs,
        "optimizer_name": result.optimizer_name,
        "n_trades": len(result.trades),
        "n_rebalances": len(result.rebalance_dates),
        "weight_history": [w.tolist() for w in result.weight_history],
    }


@router.post("/backtest")
def run_quantum_backtest(req: BacktestRequest) -> Dict[str, Any]:
    """Run 4-mode noise-aware backtest comparison."""
    try:
        from agents.quantum.quantum_backtester import QuantumBacktester
        from quantum.noise.noise_model import NoiseModel

        returns_df = _fetch_returns(req.tickers, req.period)

        noise_model = NoiseModel(
            single_qubit_error=req.single_qubit_error,
            two_qubit_error=req.two_qubit_error,
            readout_error=req.readout_error,
        )

        backtester = QuantumBacktester(config={
            "initial_capital": req.initial_capital,
            "transaction_cost_bps": req.transaction_cost_bps,
            "slippage_bps": req.slippage_bps,
            "rebalance_frequency": req.rebalance_frequency,
            "max_weight": req.max_weight,
            "lookback_window": req.lookback_window,
            "qaoa_layers": req.qaoa_layers,
        })

        raw = backtester.noise_aware_compare(
            returns=returns_df,
            noise_model=noise_model,
        )

        # Serialize each BacktestResult
        response: Dict[str, Any] = {
            "tickers": req.tickers,
            "n_days": len(returns_df),
        }

        for key in ("classical", "qaoa_ideal", "qaoa_noisy", "qaoa_mitigated"):
            if key in raw:
                response[key] = _serialize_backtest_result(raw[key])

        if "summary" in raw:
            response["summary"] = raw["summary"]

        return response
    except Exception as e:
        logger.exception("Quantum backtest failed")
        raise HTTPException(status_code=500, detail=str(e))
