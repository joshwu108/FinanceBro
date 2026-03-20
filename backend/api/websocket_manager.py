import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages per-WebSocket client state and per-symbol queues.

    The hub does the fan-out. The websocket endpoint owns the send/receive loops.
    """

    def __init__(self) -> None:
        self._hub: Optional[Any] = None
        self._connections: Dict[int, Dict[str, Any]] = {}

    def configure_hub(self, hub: Any) -> None:
        self._hub = hub

    async def connect(self, websocket: WebSocket, symbol: str) -> asyncio.Queue[dict[str, Any]]:
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=250)

        sym = symbol.strip().upper()
        if not sym:
            sym = "UNKNOWN"

        if self._hub is not None:
            try:
                await self._hub.add_client(sym, queue)
            except Exception:
                logger.exception("Failed to register websocket client with hub: %s", sym)

        self._connections[id(websocket)] = {"symbol": sym, "queue": queue}
        return queue

    async def disconnect(self, websocket: WebSocket) -> None:
        info = self._connections.pop(id(websocket), None)
        if not info:
            return

        if self._hub is not None:
            try:
                await self._hub.remove_client(info["symbol"], info["queue"])
            except Exception:
                logger.exception("Failed to unregister websocket client with hub")


manager = ConnectionManager()