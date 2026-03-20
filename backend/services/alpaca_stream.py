import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Set

import websockets

logger = logging.getLogger(__name__)


def _parse_rfc3339_timestamp(ts: str) -> datetime:
    """Parse an RFC3339 timestamp (possibly with nanoseconds) into UTC datetime."""
    if not ts:
        raise ValueError("Empty timestamp")

    # Python's fromisoformat may not support nanosecond fractions; trim to microseconds.
    # Example: 2024-07-24T07:56:53.639713735Z
    ts2 = ts
    if ts2.endswith("Z"):
        ts2 = ts2[:-1] + "+00:00"

    # If fractional seconds exceed 6 digits, trim.
    m = re.match(r"^(.*T\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:\d{2})$", ts2)
    if m:
        prefix, frac, tz = m.groups()
        if len(frac) > 6:
            frac = frac[:6]
        # Right-pad to 6 to satisfy fromisoformat variability.
        frac = (frac + "0" * 6)[:6]
        ts2 = f"{prefix}.{frac}{tz}"

    dt = datetime.fromisoformat(ts2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class AlpacaBar:
    symbol: str
    date: str  # ISO string (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class AlpacaQuote:
    symbol: str
    date: str  # ISO string (UTC)
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass(frozen=True)
class AlpacaTrade:
    symbol: str
    date: str  # ISO string (UTC)
    price: float
    size: float


class AlpacaStreamClient:
    """
    Single upstream Alpaca IEX/SIP market data WebSocket client.

    - Maintains desired symbol subscription sets.
    - Reconnects with exponential backoff.
    - Dispatches decoded trade/quote/bar events to async callbacks.
    """

    def __init__(
        self,
        api_key_id: str,
        api_secret: str,
        *,
        feed: str = "iex",
        stream_base_url: str = "wss://stream.data.alpaca.markets",
        subscribe_trades: bool = True,
        subscribe_quotes: bool = True,
        subscribe_bars: bool = True,
        heartbeat_timeout_s: int = 30,
    ):
        if not api_key_id or not api_secret:
            raise ValueError("api_key_id and api_secret are required")

        self.api_key_id = api_key_id
        self.api_secret = api_secret
        self.feed = feed
        self.stream_base_url = stream_base_url.rstrip("/")

        self.subscribe_trades = subscribe_trades
        self.subscribe_quotes = subscribe_quotes
        self.subscribe_bars = subscribe_bars

        self.heartbeat_timeout_s = heartbeat_timeout_s

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._stop_event = asyncio.Event()
        self._run_task: Optional[asyncio.Task[None]] = None

        self._wanted_symbols: Set[str] = set()
        self._connected = asyncio.Event()

        self._on_bar: Optional[Callable[[AlpacaBar], Awaitable[None]]] = None
        self._on_quote: Optional[Callable[[AlpacaQuote], Awaitable[None]]] = None
        self._on_trade: Optional[Callable[[AlpacaTrade], Awaitable[None]]] = None

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def set_callbacks(
        self,
        *,
        on_bar: Optional[Callable[[AlpacaBar], Awaitable[None]]] = None,
        on_quote: Optional[Callable[[AlpacaQuote], Awaitable[None]]] = None,
        on_trade: Optional[Callable[[AlpacaTrade], Awaitable[None]]] = None,
    ) -> None:
        self._on_bar = on_bar
        self._on_quote = on_quote
        self._on_trade = on_trade

    async def start(self) -> None:
        if self._run_task is not None:
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self._run_loop(), name="alpaca_stream_client")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            finally:
                self._ws = None
        self._connected.clear()

    async def subscribe(self, symbols: Set[str]) -> None:
        symbols = {s.strip().upper() for s in symbols if s.strip()}
        if not symbols:
            return
        newly_added = symbols - self._wanted_symbols
        if not newly_added:
            return

        self._wanted_symbols |= newly_added

        if self._ws is not None and self._connected.is_set():
            msg = {"action": "subscribe"}
            if self.subscribe_trades:
                msg["trades"] = sorted(newly_added)
            if self.subscribe_quotes:
                msg["quotes"] = sorted(newly_added)
            if self.subscribe_bars:
                msg["bars"] = sorted(newly_added)
            await self._send_json(msg)

    async def unsubscribe(self, symbols: Set[str]) -> None:
        symbols = {s.strip().upper() for s in symbols if s.strip()}
        if not symbols:
            return
        actually_removed = symbols & self._wanted_symbols
        if not actually_removed:
            return
        self._wanted_symbols -= actually_removed

        if self._ws is not None and self._connected.is_set():
            msg = {"action": "unsubscribe"}
            if self.subscribe_trades:
                msg["trades"] = sorted(actually_removed)
            if self.subscribe_quotes:
                msg["quotes"] = sorted(actually_removed)
            if self.subscribe_bars:
                msg["bars"] = sorted(actually_removed)
            await self._send_json(msg)

    async def _run_loop(self) -> None:
        # Exponential backoff: 1s -> 30s
        backoff_s = 1
        while not self._stop_event.is_set():
            try:
                await self._connect_and_listen()
                backoff_s = 1
            except asyncio.CancelledError:
                break
            except Exception:
                self._connected.clear()
                logger.exception("Alpaca stream disconnected; will reconnect")
                await asyncio.sleep(backoff_s)
                backoff_s = min(30, backoff_s * 2)

    async def _connect_and_listen(self) -> None:
        url = f"{self.stream_base_url}/v2/{self.feed}"
        logger.info("Connecting to Alpaca stream: %s", url)

        self._connected.clear()
        async with websockets.connect(
            url,
            compression=None,
            ping_interval=None,  # We implement heartbeat timeout ourselves.
            close_timeout=10,
            max_size=4 * 1024 * 1024,
        ) as ws:
            self._ws = ws

            # Authenticate within 10-second window.
            await self._send_json(
                {
                    "action": "auth",
                    "key": self.api_key_id,
                    "secret": self.api_secret,
                }
            )

            # Wait for welcome/auth response (first messages are small arrays).
            await self._drain_initial_messages()
            self._connected.set()

            await self._resubscribe_all()
            await self._read_loop()

    async def _drain_initial_messages(self) -> None:
        # We don't require a specific success message body, but we consume
        # whatever the server returns immediately after auth/subscription.
        end = asyncio.get_running_loop().time() + 10
        while asyncio.get_running_loop().time() < end:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=1.5)  # type: ignore[union-attr]
            except asyncio.TimeoutError:
                return
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, list) and parsed:
                # If we got a subscription-related message, it's fine; we proceed.
                return

    async def _resubscribe_all(self) -> None:
        if not self._wanted_symbols:
            return
        msg = {"action": "subscribe"}
        symbols = sorted(self._wanted_symbols)
        if self.subscribe_trades:
            msg["trades"] = symbols
        if self.subscribe_quotes:
            msg["quotes"] = symbols
        if self.subscribe_bars:
            msg["bars"] = symbols
        await self._send_json(msg)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        while not self._stop_event.is_set():
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.heartbeat_timeout_s)
            if raw is None:
                raise ConnectionError("Alpaca websocket closed")

            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to decode Alpaca message as JSON: %s", raw)
                continue

            if not isinstance(messages, list):
                continue

            for entry in messages:
                if not isinstance(entry, dict):
                    continue

                msg_type = entry.get("T")
                if msg_type == "b":
                    await self._handle_bar(entry)
                elif msg_type == "q":
                    await self._handle_quote(entry)
                elif msg_type == "t":
                    await self._handle_trade(entry)
                elif msg_type in {"success", "error", "subscription"}:
                    # Control messages; ignore or log lightly.
                    if msg_type == "error":
                        logger.error("Alpaca stream error message: %s", entry)
                else:
                    # Unknown message type, ignore.
                    continue

    async def _handle_bar(self, entry: dict) -> None:
        if self._on_bar is None:
            return
        symbol = str(entry.get("S", "")).upper()
        if not symbol:
            return
        try:
            dt = _parse_rfc3339_timestamp(str(entry["t"]))
        except Exception:
            return
        try:
            bar = AlpacaBar(
                symbol=symbol,
                date=dt.isoformat(),
                open=float(entry["o"]),
                high=float(entry["h"]),
                low=float(entry["l"]),
                close=float(entry["c"]),
                volume=int(entry["v"]),
            )
        except Exception:
            return
        await self._on_bar(bar)

    async def _handle_quote(self, entry: dict) -> None:
        if self._on_quote is None:
            return
        symbol = str(entry.get("S", "")).upper()
        if not symbol:
            return
        try:
            dt = _parse_rfc3339_timestamp(str(entry["t"]))
        except Exception:
            return
        try:
            quote = AlpacaQuote(
                symbol=symbol,
                date=dt.isoformat(),
                bid_price=float(entry["bp"]),
                bid_size=float(entry["bs"]),
                ask_price=float(entry["ap"]),
                ask_size=float(entry["as"]),
            )
        except Exception:
            return
        await self._on_quote(quote)

    async def _handle_trade(self, entry: dict) -> None:
        if self._on_trade is None:
            return
        symbol = str(entry.get("S", "")).upper()
        if not symbol:
            return
        try:
            dt = _parse_rfc3339_timestamp(str(entry["t"]))
        except Exception:
            return
        try:
            trade = AlpacaTrade(
                symbol=symbol,
                date=dt.isoformat(),
                price=float(entry["p"]),
                size=float(entry["s"]),
            )
        except Exception:
            return
        await self._on_trade(trade)

    async def _send_json(self, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(payload))

