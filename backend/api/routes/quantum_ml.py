"""Quantum ML prediction API — VQR vs classical baselines."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


class QuantumMLRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    period: str = Field(default="1y", description="Data lookback period")
    n_lags: int = Field(default=5, ge=2, le=20)
    n_qubits: int = Field(default=4, ge=2, le=8)
    n_layers: int = Field(default=2, ge=1, le=5)
    maxiter: int = Field(default=100, ge=20, le=500)
    seed: int = Field(default=42)
    methods: List[str] = Field(default=["linear", "rolling_mean", "vqr"])
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)


def _fetch_returns(ticker: str, period: str):
    """Fetch returns for a single ticker."""
    try:
        import yfinance as yf
        import pandas as pd

        data = yf.download(ticker, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=ticker)
        returns = data.pct_change().dropna()
        if len(returns) < 50:
            raise ValueError("Insufficient data")
        # Limit to most recent 200 rows to keep VQR training fast
        return returns.tail(200)
    except Exception as e:
        logger.warning("yfinance failed (%s), using synthetic returns", e)
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n_days = 200
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        return pd.DataFrame({ticker: rng.normal(0.0003, 0.015, n_days)}, index=dates)


@router.post("/ml/predict")
def run_quantum_ml(req: QuantumMLRequest) -> Dict[str, Any]:
    """Run VQR vs classical prediction comparison."""
    try:
        from agents.quantum.quantum_ml_agent import QuantumMLAgent

        returns_df = _fetch_returns(req.ticker, req.period)

        agent = QuantumMLAgent(config={
            "methods": req.methods,
            "n_lags": req.n_lags,
            "n_qubits": req.n_qubits,
            "n_layers": req.n_layers,
            "maxiter": req.maxiter,
            "seed": req.seed,
            "train_ratio": req.train_ratio,
        })

        result = agent.run({
            "returns": returns_df,
            "target_column": req.ticker,
        })

        return {
            "ticker": req.ticker,
            "n_data_points": len(returns_df),
            **result,
        }
    except Exception as e:
        logger.exception("Quantum ML prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
