import asyncio
import json
import logging
from typing import Dict, Set, List
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "portfolio": set(),
            "alerts": set(),
            "general": set()
        }
        self.price_cache: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
    
    async def connect(self, websocket: WebSocket, client_type: str = "general"):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_type].add(websocket)
        logger.info(f"Client connected to {client_type} channel. Total connections: {len(self.active_connections[client_type])}")
    
    async def disconnect(self, websocket: WebSocket, client_type: str = "general"):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections[client_type]:
            self.active_connections[client_type].remove(websocket)
        logger.info(f"Client disconnected from {client_type} channel. Total connections: {len(self.active_connections[client_type])}")
    
    async def broadcast_to_portfolio(self, message: dict):
        """Broadcast portfolio updates to all portfolio clients"""
        if not self.active_connections["portfolio"]:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections["portfolio"]:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to portfolio client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            await self.disconnect(connection, "portfolio")
    
    async def broadcast_to_alerts(self, message: dict):
        """Broadcast alert updates to all alert clients"""
        if not self.active_connections["alerts"]:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections["alerts"]:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to alert client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            await self.disconnect(connection, "alerts")
    
    async def broadcast_price_update(self, symbol: str, price: float, change: float, change_percent: float):
        """Broadcast price updates to all connected clients"""
        message = {
            "type": "price_update",
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": change_percent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update cache
        self.price_cache[symbol] = price
        self.last_update[symbol] = datetime.utcnow()
        
        # Broadcast to all channels
        await self.broadcast_to_portfolio(message)
        await self.broadcast_to_alerts(message)
    
    async def broadcast_portfolio_update(self, portfolio_id: int, holdings: List[dict]):
        """Broadcast portfolio updates"""
        message = {
            "type": "portfolio_update",
            "portfolio_id": portfolio_id,
            "holdings": holdings,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_portfolio(message)
    
    async def broadcast_alert_trigger(self, alert_id: int, symbol: str, price: float, message: str):
        """Broadcast alert triggers"""
        alert_message = {
            "type": "alert_trigger",
            "alert_id": alert_id,
            "symbol": symbol,
            "price": price,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_alerts(alert_message)
    
    def get_connection_count(self, client_type: str = "general") -> int:
        """Get the number of active connections for a client type"""
        return len(self.active_connections[client_type])
    
    def get_price_cache(self) -> Dict[str, float]:
        """Get the current price cache"""
        return self.price_cache.copy()

# Global WebSocket manager instance
websocket_manager = WebSocketManager() 