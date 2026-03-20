import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from orchestrator.run_pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Rough guardrail for extremely narrow date ranges.
# Note: WalkForwardAgent may auto-adjust folds/train windows for short
# histories, but FeatureAgent still needs enough data to compute rolling
# indicators (e.g., SMA-200).
_MIN_CALENDAR_DAYS = 150


class PipelineConfig(BaseModel):
    symbols: List[str] = Field(..., example=["AAPL", "MSFT", "GOOG"])
    start_date: str = Field(..., example="2020-01-01")
    end_date: str = Field(..., example="2025-01-01")
    model_type: str = Field(default="random_forest")
    transaction_costs_bps: float = Field(default=5.0)
    slippage_bps: float = Field(default=2.0)
    max_position_size: float = Field(default=0.1)
    benchmark: str = Field(default="SPY")


@router.post("/run_pipeline")
def run(config: PipelineConfig):
    # Validate date range is wide enough for indicator warm-up + walk-forward
    try:
        start = date.fromisoformat(config.start_date)
        end = date.fromisoformat(config.end_date)
        delta_days = (end - start).days
        if delta_days < _MIN_CALENDAR_DAYS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Date range too narrow ({delta_days} days). "
                    f"The pipeline requires at least {_MIN_CALENDAR_DAYS} calendar days "
                    f"(~1 year) for indicator warm-up (SMA-200) and 5-fold walk-forward "
                    f"validation. Please widen your start_date/end_date range."
                ),
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}")

    try:
        result = run_pipeline(config.model_dump())
    except ValueError as exc:
        logger.exception("Pipeline validation error")
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "pipeline_validation_error",
                "message": str(exc),
                "hint": (
                    "This usually means your requested date range does not provide "
                    "enough usable history after feature/indicator warm-ups. "
                    "Try widening start_date/end_date, or reduce requested data window "
                    "(if supported by the UI)."
                ),
            },
        )
    except Exception as exc:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "data": result
    }