from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import websocket_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/portfolio")
async def websocket_portfolio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for portfolio updates"""
    await websocket_manager.connect(websocket, "portfolio")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # You can handle incoming messages here if needed
            logger.info(f"Received portfolio message: {data}")
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, "portfolio")

@router.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """WebSocket endpoint for alert updates"""
    await websocket_manager.connect(websocket, "alerts")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # You can handle incoming messages here if needed
            logger.info(f"Received alert message: {data}")
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, "alerts")

@router.websocket("/ws/general")
async def websocket_general_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general updates"""
    await websocket_manager.connect(websocket, "general")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # You can handle incoming messages here if needed
            logger.info(f"Received general message: {data}")
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket, "general") 