import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, time, timezone
from typing import Any, Dict, Optional, Set
from zoneinfo import ZoneInfo

from services.alpaca_stream import AlpacaBar, AlpacaQuote, AlpacaStreamClient, AlpacaTrade

logger = logging.getLogger(__name__)


class MarketDataHub:
    """
    Fan-out layer for live market data.

    - Keeps a single upstream AlpacaStreamClient connection.
    - Reference-counts per-symbol clients (asyncio queues).
    - Broadcasts normalized JSON envelopes to per-client WebSocket send loops.
    """

    def __init__(
        self,
        *,
        alpaca_api_key_id: Optional[str] = None,
        alpaca_api_secret: Optional[str] = None,
        alpaca_feed: str = "iex",
        alpaca_stream_base_url: str = "wss://stream.data.alpaca.markets",
        queue_maxsize: int = 250,
    ):
        self._queue_maxsize = queue_maxsize
        self._lock = asyncio.Lock()

        self._queues_by_symbol: Dict[str, Set[asyncio.Queue[dict[str, Any]]]] = {}
        self._last_message_at: Optional[datetime] = None

        self._alpaca_enabled = bool(alpaca_api_key_id and alpaca_api_secret)
        self._alpaca_feed = alpaca_feed
        self._alpaca_client: Optional[AlpacaStreamClient] = None

        self._alpaca_connected = False

        if self._alpaca_enabled:
            self._alpaca_client = AlpacaStreamClient(
                alpaca_api_key_id or "",
                alpaca_api_secret or "",
                feed=alpaca_feed,
                stream_base_url=alpaca_stream_base_url,
            )

            # Bind callbacks.
            self._alpaca_client.set_callbacks(
                on_bar=self._handle_bar,
                on_quote=self._handle_quote,
                on_trade=self._handle_trade,
            )

    async def start(self) -> None:
        if not self._alpaca_client:
            return
        await self._alpaca_client.start()

    async def shutdown(self) -> None:
        if self._alpaca_client:
            await self._alpaca_client.stop()

    @property
    def enabled(self) -> bool:
        return self._alpaca_enabled

    def _market_open(self, *, now: Optional[datetime] = None) -> bool:
        # 9:30am-4:00pm ET, weekdays only. (Approximation; doesn't handle holidays.)
        now = now or datetime.now(timezone.utc)
        et = now.astimezone(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return False
        start = time(9, 30)
        end = time(16, 0)
        return start <= et.time() <= end

    def status(self) -> dict[str, Any]:
        subscribed_symbols = sorted(self._queues_by_symbol.keys())
        return {
            "enabled": self._alpaca_enabled,
            "alpaca_connected": self._alpaca_client.is_connected() if self._alpaca_client else False,
            "market_open": self._market_open(),
            "subscribed_symbols": subscribed_symbols,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
        }

    async def add_client(self, symbol: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        sym = symbol.strip().upper()
        if not sym:
            return

        subscribe_needed = False
        async with self._lock:
            if sym not in self._queues_by_symbol:
                self._queues_by_symbol[sym] = set()
            self._queues_by_symbol[sym].add(queue)
            subscribe_needed = len(self._queues_by_symbol[sym]) == 1

        if subscribe_needed and self._alpaca_client:
            try:
                await self._alpaca_client.subscribe({sym})
            except Exception:
                logger.exception("Failed to subscribe symbol to Alpaca: %s", sym)

    async def remove_client(self, symbol: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        sym = symbol.strip().upper()
        if not sym:
            return

        unsubscribe_needed = False
        async with self._lock:
            qs = self._queues_by_symbol.get(sym)
            if not qs:
                return
            qs.discard(queue)
            unsubscribe_needed = len(qs) == 0
            if unsubscribe_needed:
                self._queues_by_symbol.pop(sym, None)

        if unsubscribe_needed and self._alpaca_client:
            try:
                await self._alpaca_client.unsubscribe({sym})
            except Exception:
                logger.exception("Failed to unsubscribe symbol from Alpaca: %s", sym)

    async def _broadcast(self, symbol: str, envelope: dict[str, Any]) -> None:
        # Keep broadcast O(queues) and non-blocking for slow clients.
        queues: list[asyncio.Queue[dict[str, Any]]] = []
        async with self._lock:
            queues = list(self._queues_by_symbol.get(symbol, []))

        for q in queues:
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                # Drop messages for that client rather than blocking the hub.
                continue

        self._last_message_at = datetime.now(timezone.utc)

    async def _handle_bar(self, bar: AlpacaBar) -> None:
        envelope = {
            "type": "bar",
            "symbol": bar.symbol,
            "bar": {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            },
            "timestamp": bar.date,
        }
        await self._broadcast(bar.symbol, envelope)

    async def _handle_quote(self, quote: AlpacaQuote) -> None:
        envelope = {
            "type": "quote",
            "symbol": quote.symbol,
            "quote": {
                "date": quote.date,
                "bidPrice": quote.bid_price,
                "bidSize": quote.bid_size,
                "askPrice": quote.ask_price,
                "askSize": quote.ask_size,
            },
            "timestamp": quote.date,
        }
        await self._broadcast(quote.symbol, envelope)

    async def _handle_trade(self, trade: AlpacaTrade) -> None:
        envelope = {
            "type": "trade",
            "symbol": trade.symbol,
            "trade": {
                "date": trade.date,
                "price": trade.price,
                "size": trade.size,
            },
            "timestamp": trade.date,
        }
        await self._broadcast(trade.symbol, envelope)

