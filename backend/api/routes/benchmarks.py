"""Quantum benchmarks API — scaling experiments and C++ speedup."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


class PortfolioScalingRequest(BaseModel):
    asset_counts: List[int] = Field(default=[3, 4, 5, 6, 8])
    n_bits: int = Field(default=3, ge=2, le=5)
    qaoa_layers: int = Field(default=2, ge=1, le=4)
    qaoa_maxiter: int = Field(default=200, ge=50, le=500)


class MaxCutScalingRequest(BaseModel):
    node_counts: List[int] = Field(default=[4, 6, 8, 10, 12])
    edge_prob: float = Field(default=0.5, ge=0.1, le=1.0)
    qaoa_layers: int = Field(default=2, ge=1, le=4)
    qaoa_maxiter: int = Field(default=300, ge=50, le=500)


class CppSpeedupRequest(BaseModel):
    qubit_counts: List[int] = Field(default=[4, 6, 8, 10, 12, 14])
    n_iterations: int = Field(default=10, ge=1, le=50)


@router.post("/benchmarks/portfolio-scaling")
def run_portfolio_scaling(req: PortfolioScalingRequest) -> Dict[str, Any]:
    """Run portfolio optimization scaling experiment."""
    try:
        from quantum.benchmarks.scaling_experiments import run_portfolio_scaling as run_exp

        return run_exp(
            asset_counts=req.asset_counts,
            n_bits=req.n_bits,
            qaoa_layers=req.qaoa_layers,
            qaoa_maxiter=req.qaoa_maxiter,
        )
    except Exception as e:
        logger.exception("Portfolio scaling benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks/maxcut-scaling")
def run_maxcut_scaling(req: MaxCutScalingRequest) -> Dict[str, Any]:
    """Run Max-Cut scaling experiment."""
    try:
        from quantum.benchmarks.scaling_experiments import run_maxcut_scaling as run_exp

        return run_exp(
            node_counts=req.node_counts,
            edge_prob=req.edge_prob,
            qaoa_layers=req.qaoa_layers,
            qaoa_maxiter=req.qaoa_maxiter,
        )
    except Exception as e:
        logger.exception("Max-Cut scaling benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks/cpp-speedup")
def run_cpp_speedup(req: CppSpeedupRequest) -> Dict[str, Any]:
    """Run C++ vs Python speedup benchmark."""
    try:
        from quantum.benchmarks.scaling_experiments import run_cpp_speedup_benchmark as run_exp

        return run_exp(
            qubit_counts=req.qubit_counts,
            n_iterations=req.n_iterations,
        )
    except Exception as e:
        logger.exception("C++ speedup benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))
