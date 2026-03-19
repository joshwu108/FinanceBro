"""
Data Collection Service
Fetches stock data from multiple APIs (Yahoo Finance, Alpha Vantage)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import os
import time
import random
from alpha_vantage.timeseries import TimeSeries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects stock data from multiple sources"""
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.cache = {}  # Simple in-memory cache
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
    async def _rate_limit(self):
        """Ensure minimum time between requests to avoid rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()
    
    async def get_stock_data_yahoo(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance with retry logic
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            max_retries: Maximum number of retry attempts
        
        Returns:
            DataFrame with OHLCV data
        """
        for attempt in range(max_retries):
            try:
                await self._rate_limit()
                logger.info(f"Fetching data for {symbol} from Yahoo Finance (attempt {attempt + 1})")
                
                # Add random delay to avoid synchronized requests
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Rename columns for consistency
                data.columns = [col.lower() for col in data.columns]
                
                # Add symbol column
                data['symbol'] = symbol
                
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None
    
    async def get_stock_data_alpha_vantage(
        self, 
        symbol: str, 
        function: str = "TIME_SERIES_DAILY"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            function: API function ('TIME_SERIES_DAILY', 'TIME_SERIES_INTRADAY', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
            
        try:
            logger.info(f"Fetching data for {symbol} from Alpha Vantage")
            
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            
            if function == "TIME_SERIES_DAILY":
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            elif function == "TIME_SERIES_INTRADAY":
                data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
            else:
                logger.error(f"Unsupported function: {function}")
                return None
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns for consistency
            data.columns = [col.lower() for col in data.columns]
            data = data.rename(columns={'date': 'date'})
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    async def get_stock_data_with_fallback(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with fallback to alternative sources
        
        Args:
            symbol: Stock symbol
            period: Time period for Yahoo Finance
            interval: Data interval
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try Yahoo Finance first
        data = await self.get_stock_data_yahoo(symbol, period, interval)
        
        if data is not None and not data.empty:
            return data
        
        # Fallback to Alpha Vantage if Yahoo Finance fails
        logger.info(f"Yahoo Finance failed for {symbol}, trying Alpha Vantage...")
        data = await self.get_stock_data_alpha_vantage(symbol)
        
        if data is not None and not data.empty:
            return data
        
        logger.error(f"All data sources failed for {symbol}")
        return None
    
    async def get_multiple_stocks(
        self, 
        symbols: List[str], 
        source: str = "yahoo",
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks concurrently
        
        Args:
            symbols: List of stock symbols
            source: Data source ('yahoo' or 'alpha_vantage')
            period: Time period for Yahoo Finance
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Fetching data for {len(symbols)} stocks from {source}")
        
        tasks = []
        for symbol in symbols:
            if source == "yahoo":
                task = self.get_stock_data_yahoo(symbol, period)
            elif source == "alpha_vantage":
                task = self.get_stock_data_alpha_vantage(symbol)
            else:
                logger.error(f"Unsupported source: {source}")
                continue
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
            elif result is not None:
                data_dict[symbol] = result
        
        logger.info(f"Successfully fetched data for {len(data_dict)} stocks")
        return data_dict
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate stock data quality
        
        Args:
            data: Stock data DataFrame
        
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False
        
        # Check for required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return False
        
        # Check for missing values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts.to_dict()}")
            return False
        
        # Check for reasonable price values
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.warning(f"Found non-positive prices in {col}")
                return False
        
        return True
    
    def save_data(self, data: pd.DataFrame, symbol: str, format: str = "csv"):
        """
        Save stock data to file
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
            format: File format ('csv', 'parquet')
        """
        try:
            os.makedirs("data/raw", exist_ok=True)
            
            filename = f"data/raw/{symbol}_{datetime.now().strftime('%Y%m%d')}.{format}"
            
            if format == "csv":
                data.to_csv(filename, index=False)
            elif format == "parquet":
                data.to_parquet(filename, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return
            
            logger.info(f"Saved data to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")


# Global instance
data_collector = DataCollector()


# Example usage and testing
async def test_data_collection():
    """Test the data collection service"""
    collector = DataCollector()
    
    # Test with popular stocks
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    print("Testing Yahoo Finance data collection...")
    yahoo_data = await collector.get_multiple_stocks(symbols, source="yahoo", period="6mo")
    
    for symbol, data in yahoo_data.items():
        print(f"{symbol}: {len(data)} records")
        if collector.validate_data(data):
            collector.save_data(data, symbol)
    
    return yahoo_data


if __name__ == "__main__":
    # Run test
    asyncio.run(test_data_collection()) 