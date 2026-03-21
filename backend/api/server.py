import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from workers.tasks import evaluate_tick

ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=False)

from api.routes import data, experiments, market_data, pipeline, predict
from api.websocket_manager import manager

app = FastAPI(title="FinanceBro Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(predict.router)
app.include_router(data.router)
app.include_router(experiments.router)
app.include_router(market_data.router)


# ── Live Market Data Hub ──────────────────────────────────────────────────────

alpaca_key = (os.getenv("ALPACA_KEY") or os.getenv("ALPACA_API_KEY") or "").strip()
alpaca_secret = (os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
alpaca_feed = os.getenv("ALPACA_FEED", "iex").strip().lower() or "iex"
alpaca_base_url = (os.getenv("ALPACA_BASE_URL") or "").strip().lower()
alpaca_stream_base_url_override = (os.getenv("ALPACA_STREAM_BASE_URL") or "").strip()


alpaca_is_sandbox = "sandbox" in alpaca_base_url
alpaca_stream_base_url = (
    alpaca_stream_base_url_override
    if alpaca_stream_base_url_override
    else (
        "wss://stream.data.sandbox.alpaca.markets"
        if alpaca_is_sandbox
        else "wss://stream.data.alpaca.markets"
    )
)

market_data_hub = None

try:
    from services.market_data_hub import MarketDataHub

    market_data_hub = MarketDataHub(
        alpaca_api_key_id=alpaca_key if alpaca_key else None,
        alpaca_api_secret=alpaca_secret if alpaca_secret else None,
        alpaca_feed=alpaca_feed,
        alpaca_stream_base_url=alpaca_stream_base_url,
    )
    manager.configure_hub(market_data_hub)
    app.state.market_data_hub = market_data_hub
except Exception:
    # Server should still start in research mode if Alpaca config is missing.
    market_data_hub = None
    app.state.market_data_hub = None


# ── WebSocket: /ws/live/{symbol} ─────────────────────────────────────────────


@app.websocket("/ws/live/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    queue = await manager.connect(websocket, symbol)
    last_pong = time.monotonic()

    async def recv_loop() -> None:
        nonlocal last_pong
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "pong":
                    last_pong = time.monotonic()
                    continue
                try:
                    parsed = json.loads(msg)
                except Exception:
                    continue
                if isinstance(parsed, dict) and parsed.get("type") == "pong":
                    last_pong = time.monotonic()
        except WebSocketDisconnect:
            pass

    async def send_loop() -> None:
        ping_interval_s = 15
        pong_timeout_s = 30
        next_ping_at = time.monotonic() + ping_interval_s

        while True:
            remaining = max(0, next_ping_at - time.monotonic())
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=remaining)
                await websocket.send_json(payload)
                if payload.get("type") == "trade":
                    trade_data = payload.get("trade", {})
                    price = trade_data.get("price")
                    evaluate_tick.delay(symbol, price, {})
            except asyncio.TimeoutError:
                # Time to ping (and validate that the client is responding).
                await websocket.send_json({"type": "ping", "ts": datetime.now(timezone.utc).isoformat()})
                next_ping_at = time.monotonic() + ping_interval_s

                if time.monotonic() - last_pong > pong_timeout_s:
                    await websocket.close(code=1001)
                    break

    try:
        recv_task = asyncio.create_task(recv_loop())
        send_task = asyncio.create_task(send_loop())
        _done, pending = await asyncio.wait(
            {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event() -> None:
    if market_data_hub is None:
        return
    await market_data_hub.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if market_data_hub is None:
        return
    await market_data_hub.shutdown()

