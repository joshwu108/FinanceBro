import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


# using .env in the root directory
def _get_alpaca_env() -> tuple[str, str, str]:
    api_key = (os.getenv("ALPACA_KEY") or os.getenv("ALPACA_API_KEY") or "").strip()
    api_secret = (os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
    feed = os.getenv("ALPACA_FEED", "iex").strip().lower() or "iex"
    return api_key, api_secret, feed


def _normalize_bar(symbol: str, bar: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": bar["t"],
        "open": float(bar["o"]),
        "high": float(bar["h"]),
        "low": float(bar["l"]),
        "close": float(bar["c"]),
        "volume": int(bar["v"]),
        "symbol": symbol,
    }


@router.get("/api/market/status")
async def market_status(request: Request) -> Dict[str, Any]:
    hub = getattr(request.app.state, "market_data_hub", None)
    if hub is None:
        return {"enabled": False, "alpaca_connected": False, "market_open": False}
    return hub.status()

@router.get("/api/market/snapshot/{symbol}")
async def market_snapshot(symbol: str) -> Dict[str, Any]:
    """Function to get the current stock price of a symbol"""
    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=422, detail="symbol is required")

    api_key, api_secret, feed = _get_alpaca_env()
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=503,
            detail="Alpaca credentials are not configured. Set ALPACA_KEY and ALPACA_SECRET.",
        )

    # Fetch enough history to cover the last trading session even when
    # market is closed (evenings, weekends, holidays).
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    alpaca_base_url = (os.getenv("ALPACA_BASE_URL") or "").strip().lower()
    data_base_url = os.getenv("ALPACA_DATA_BASE_URL", "").strip()
    if not data_base_url:
        alpaca_is_sandbox = "sandbox" in alpaca_base_url
        data_base_url = (
            "https://data.sandbox.alpaca.markets"
            if alpaca_is_sandbox
            else "https://data.alpaca.markets"
        )

    url = f"{data_base_url}/v2/stocks/bars"
    params = {
        "symbols": sym,
        "timeframe": "1Min",
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 1000,
        "feed": feed,
        "sort": "asc",
    }
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Alpaca snapshot fetch failed: {resp.text}",
            )
        payload = resp.json()

    bars_by_symbol = payload.get("bars") or {}
    bars = bars_by_symbol.get(sym, [])
    return {
        "symbol": sym,
        "bars": [_normalize_bar(sym, b) for b in bars],
    }

