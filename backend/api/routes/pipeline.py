from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
from orchestrator.run_pipeline import run_pipeline

router = APIRouter()

class PipelineConfig(BaseModel):
    symbols: List[str] = Field(..., example=["AAPL", "MSFT", "GOOG"])
    start_date: str = Field(..., example="2020-01-01")
    end_date: str = Field(..., example="2025-01-01")
    model_type: str = Field(default="random_forest")
    transaction_costs_bps: float = Field(default=5.0)
    slippage_bps:float = Field(default=2.0)
    max_position_size: float = Field(default=0.1)
    benchmark: str = Field(default="SPY")


@router.post("/run_pipeline")
def run(config: PipelineConfig):
    result = run_pipeline(config.model_dump())
    return {
        "status": "success",
        "data": result
    }