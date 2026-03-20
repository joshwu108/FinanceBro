"""Data route — Serve OHLCV price data for individual symbols."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from agents.data_agent import DataAgent
from api.serializers import serialize_ohlcv

router = APIRouter()


@router.get("/api/data/{symbol}")
def get_symbol_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="ISO date, e.g. 2020-01-01"),
    end_date: Optional[str] = Query(None, description="ISO date, e.g. 2025-01-01"),
):
    """Fetch OHLCV data for a single symbol via DataAgent.

    Returns:
        {symbol: str, data: [{date, open, high, low, close, volume}, ...]}
    """
    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=422, detail="symbol is required")

    data_inputs = {"symbols": [sym]}
    if start_date:
        data_inputs["start_date"] = start_date
    if end_date:
        data_inputs["end_date"] = end_date

    try:
        data_agent = DataAgent()
        data_outputs = data_agent.run(data_inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch data: {exc}"
        ) from exc

    cleaned_data = data_outputs.get("cleaned_data", {})
    if sym not in cleaned_data:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for symbol '{sym}'",
        )

    df = cleaned_data[sym]
    return {
        "symbol": sym,
        "data": serialize_ohlcv(df),
    }
