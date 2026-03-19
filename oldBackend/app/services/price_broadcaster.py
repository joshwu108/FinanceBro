import asyncio
import logging
from typing import Dict, Set
from datetime import datetime, timedelta
from app.services.data_collector import DataCollector
from app.services.websocket_manager import websocket_manager
from app.services.database import get_db
from app.models.portfolio import PortfolioHolding

logger = logging.getLogger(__name__)

class PriceBroadcaster:
    def __init__(self):
        self.data_collector = DataCollector()
        self.price_cache: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_interval = 10  # seconds
        self.is_running = False
        
    async def start(self):
        """Start the price broadcasting service"""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Starting price broadcaster service...")
        
        while self.is_running:
            try:
                await self.broadcast_price_updates()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in price broadcaster: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self):
        """Stop the price broadcasting service"""
        self.is_running = False
        logger.info("Stopping price broadcaster service...")
    
    async def broadcast_price_updates(self):
        """Fetch and broadcast price updates for all portfolio holdings"""
        try:
            # Get all unique stock symbols from portfolios
            symbols = await self.get_portfolio_symbols()
            
            if not symbols:
                logger.info("No portfolio symbols found")
                return
            
            logger.info(f"Broadcasting prices for {len(symbols)} symbols: {symbols}")
            
            for symbol in symbols:
                try:
                    # Fetch current price with retry logic and fallback
                    stock_data = await self.data_collector.get_stock_data_with_fallback(
                        symbol, period="1d", interval="1m"
                    )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        current_price = stock_data.iloc[-1]['Close']
                        previous_price = self.price_cache.get(symbol, current_price)
                        
                        # Calculate change
                        price_change = current_price - previous_price
                        price_change_percent = (price_change / previous_price * 100) if previous_price > 0 else 0
                        
                        # Update cache
                        self.price_cache[symbol] = current_price
                        self.last_update[symbol] = datetime.utcnow()
                        
                        # Broadcast price update
                        await websocket_manager.broadcast_price_update(
                            symbol=symbol,
                            price=current_price,
                            change=price_change,
                            change_percent=price_change_percent
                        )
                        
                        logger.info(f"Broadcasted {symbol}: ${current_price:.2f} ({price_change_percent:+.2f}%)")
                    else:
                        logger.warning(f"Failed to fetch data for {symbol}, skipping broadcast")
                        
                except Exception as e:
                    logger.error(f"Error fetching price for {symbol}: {e}")
                    # Continue with other symbols instead of stopping the entire loop
                    continue
                    
        except Exception as e:
            logger.error(f"Error in broadcast_price_updates: {e}")
    
    async def get_portfolio_symbols(self) -> Set[str]:
        """Get all unique stock symbols from all portfolios"""
        try:
            db = next(get_db())
            holdings = db.query(PortfolioHolding).all()
            symbols = {holding.stock_symbol for holding in holdings}
            return symbols
        except Exception as e:
            logger.error(f"Error getting portfolio symbols: {e}")
            return set()
    
    async def broadcast_portfolio_update(self, portfolio_id: int):
        """Broadcast a specific portfolio update"""
        try:
            db = next(get_db())
            holdings = db.query(PortfolioHolding).filter(
                PortfolioHolding.portfolio_id == portfolio_id
            ).all()
            
            holdings_data = []
            for holding in holdings:
                current_price = self.price_cache.get(holding.stock_symbol, holding.average_price)
                holdings_data.append({
                    "id": holding.id,
                    "stock_symbol": holding.stock_symbol,
                    "shares": holding.shares,
                    "average_price": holding.average_price,
                    "current_price": current_price,
                    "purchase_date": holding.purchase_date.isoformat() if holding.purchase_date else None
                })
            
            await websocket_manager.broadcast_portfolio_update(portfolio_id, holdings_data)
            logger.info(f"Broadcasted portfolio update for portfolio {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error broadcasting portfolio update: {e}")

# Global price broadcaster instance
price_broadcaster = PriceBroadcaster() 