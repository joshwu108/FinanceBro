"""Quantum portfolio optimization API — QAOA vs classical Markowitz."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=10, example=["AAPL", "MSFT", "GOOG", "AMZN"])
    period: str = Field(default="1y", description="Data lookback period for yfinance")
    max_weight: float = Field(default=0.40, ge=0.05, le=1.0)
    qaoa_layers: int = Field(default=2, ge=1, le=5)
    weight_precision_bits: int = Field(default=3, ge=2, le=5)
    qaoa_maxiter: int = Field(default=300, ge=50, le=1000)
    frontier_points: int = Field(default=0, ge=0, le=30, description="0 = skip frontier")


class FrontierRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=10)
    period: str = Field(default="1y")
    max_weight: float = Field(default=0.40, ge=0.05, le=1.0)
    n_points: int = Field(default=15, ge=5, le=30)


def _fetch_returns(tickers: List[str], period: str):
    """Fetch real returns via yfinance, fall back to synthetic."""
    try:
        import yfinance as yf
        import pandas as pd

        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        if len(returns) < 60:
            raise ValueError("Insufficient data")
        return returns
    except Exception as e:
        logger.warning("yfinance failed (%s), using synthetic returns", e)
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n_days = 252
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        data = {}
        for i, t in enumerate(tickers):
            mu = 0.0003 + i * 0.0001
            sigma = 0.01 + i * 0.002
            data[t] = rng.normal(mu, sigma, n_days)
        return pd.DataFrame(data, index=dates)


def _serialize(obj: Any) -> Any:
    """Make numpy types JSON-serializable."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


@router.post("/portfolio")
def run_quantum_portfolio(req: PortfolioRequest) -> Dict[str, Any]:
    """Run QAOA vs classical Markowitz portfolio optimization."""
    try:
        from agents.quantum.quantum_portfolio_agent import QuantumPortfolioAgent

        returns_df = _fetch_returns(req.tickers, req.period)

        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy", "qaoa"],
            "max_weight": req.max_weight,
            "qaoa_layers": req.qaoa_layers,
            "weight_precision_bits": req.weight_precision_bits,
            "qaoa_maxiter": req.qaoa_maxiter,
            "frontier_points": req.frontier_points,
        })

        result = agent.run({"returns": returns_df})
        agent.validate({"returns": returns_df}, result)

        return _serialize(result)
    except Exception as e:
        logger.exception("Quantum portfolio optimization failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/frontier")
def run_efficient_frontier(req: FrontierRequest) -> Dict[str, Any]:
    """Generate efficient frontier with classical solver."""
    try:
        from agents.quantum.quantum_portfolio_agent import QuantumPortfolioAgent

        returns_df = _fetch_returns(req.tickers, req.period)

        agent = QuantumPortfolioAgent(config={
            "methods": ["markowitz_cvxpy"],
            "max_weight": req.max_weight,
            "frontier_points": req.n_points,
        })

        result = agent.run({"returns": returns_df})
        return _serialize({
            "tickers": req.tickers,
            "efficient_frontier": result.get("efficient_frontier", {}),
            "classical_weights": result.get("classical_weights"),
            "classical_objective": result.get("classical_objective"),
        })
    except Exception as e:
        logger.exception("Efficient frontier generation failed")
        raise HTTPException(status_code=500, detail=str(e))
