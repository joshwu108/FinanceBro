"""Quantum Monte Carlo option pricing API — BS + MC + QAE comparison."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


class OptionPriceRequest(BaseModel):
    spot_price: float = Field(..., gt=0, example=100.0)
    strike_price: float = Field(..., gt=0, example=105.0)
    risk_free_rate: float = Field(default=0.05, ge=0, le=0.5)
    volatility: float = Field(..., gt=0, le=5.0, example=0.2)
    time_to_expiry: float = Field(..., gt=0, le=10.0, example=1.0)
    n_classical_paths: int = Field(default=100_000, ge=1000, le=1_000_000)
    n_qubits_price: int = Field(default=4, ge=2, le=8)
    n_estimation_qubits: int = Field(default=4, ge=2, le=10)
    seed: int = Field(default=42)


class ConvergenceRequest(BaseModel):
    spot_price: float = Field(default=100.0, gt=0)
    strike_price: float = Field(default=105.0, gt=0)
    risk_free_rate: float = Field(default=0.05, ge=0, le=0.5)
    volatility: float = Field(default=0.2, gt=0, le=5.0)
    time_to_expiry: float = Field(default=1.0, gt=0, le=10.0)
    classical_path_counts: List[int] = Field(
        default=[100, 500, 1000, 5000, 10000, 50000, 100000],
    )
    quantum_estimation_qubits: List[int] = Field(
        default=[2, 3, 4, 5, 6, 7, 8],
    )
    n_qubits_price: int = Field(default=4, ge=2, le=8)


@router.post("/options/price")
def price_option(req: OptionPriceRequest) -> Dict[str, Any]:
    """Price a European call option using BS, classical MC, and quantum AE."""
    try:
        from agents.quantum.quantum_montecarlo_agent import QuantumMonteCarloAgent

        agent = QuantumMonteCarloAgent(config={
            "methods": ["classical_mc", "quantum_ae"],
            "n_classical_paths": req.n_classical_paths,
            "n_qubits_price": req.n_qubits_price,
            "n_estimation_qubits": req.n_estimation_qubits,
            "seed": req.seed,
        })

        result = agent.run({
            "spot_price": req.spot_price,
            "strike_price": req.strike_price,
            "risk_free_rate": req.risk_free_rate,
            "volatility": req.volatility,
            "time_to_expiry": req.time_to_expiry,
        })

        return result
    except Exception as e:
        logger.exception("Option pricing failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/options/convergence")
def convergence_analysis(req: ConvergenceRequest) -> Dict[str, Any]:
    """Compare convergence: classical MC O(1/sqrt(N)) vs quantum AE O(1/M)."""
    try:
        from agents.quantum.quantum_montecarlo_agent import (
            black_scholes_call,
            convergence_analysis as run_convergence,
        )

        bs_price = black_scholes_call(
            req.spot_price, req.strike_price,
            req.risk_free_rate, req.volatility, req.time_to_expiry,
        )

        result = run_convergence(
            S0=req.spot_price,
            K=req.strike_price,
            r=req.risk_free_rate,
            sigma=req.volatility,
            T=req.time_to_expiry,
            classical_path_counts=req.classical_path_counts,
            quantum_estimation_qubits=req.quantum_estimation_qubits,
            n_qubits_price=req.n_qubits_price,
        )

        # Attach errors relative to BS
        for r in result["classical"]:
            r["error"] = abs(r["price"] - bs_price)
        for r in result["quantum"]:
            r["error"] = abs(r["price"] - bs_price)

        return {
            "bs_price": bs_price,
            **result,
        }
    except Exception as e:
        logger.exception("Convergence analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
