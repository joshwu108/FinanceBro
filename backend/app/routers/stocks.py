from fastapi import APIRouter, HTTPException, Query
from app.services.data_collector import DataCollector
from app.services.ml_models import MLModelManager
from app.services.feature_engineering import FeatureEngineer
from app.services.financial_analyzer import FinancialAnalyzer
import numpy as np
import json
import logging
import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_stocks():
    """Get list of available stocks"""
    return {
        "stocks": ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "message": "Available stocks for prediction"
    }

@router.get("/{symbol}/realtime")
async def get_stock_realtime(symbol: str):
    """Get real-time stock data"""
    try:
        data_collector = DataCollector()
        stock_data = await data_collector.get_stock_data_yahoo(symbol, period="1d", interval="1m")
        
        if stock_data is None or stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No real-time data found for {symbol}")
        
        # Get the latest data point
        latest_data = stock_data.iloc[-1]
        
        return {
            "symbol": symbol,
            "price": float(latest_data['close']),
            "change": float(latest_data['close'] - stock_data.iloc[-2]['close']) if len(stock_data) > 1 else 0,
            "change_percent": float(((latest_data['close'] - stock_data.iloc[-2]['close']) / stock_data.iloc[-2]['close'] * 100)) if len(stock_data) > 1 else 0,
            "volume": float(latest_data['volume']),
            "timestamp": latest_data.name.isoformat() if hasattr(latest_data.name, 'isoformat') else str(latest_data.name)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/chart")
async def get_stock_chart_data(
    symbol: str,
    period: str = Query("1mo", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y"),
    interval: str = Query("1d", description="Data interval: 1m, 5m, 15m, 30m, 1h, 1d")
):
    """Get historical chart data for a stock"""
    try:
        data_collector = DataCollector()
        stock_data = await data_collector.get_stock_data_yahoo(symbol, period=period, interval=interval)
        print(f"Raw stock data for {symbol}:")
        print(stock_data)
        
        if stock_data is None or stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No chart data found for {symbol}")
        
        stock_data = stock_data[::-1][:100]
        
        # Check if we have any data after filtering
        if stock_data.empty:
            logger.warning(f"No data remaining after filtering for {symbol}, using original data")
            stock_data = await data_collector.get_stock_data_yahoo(symbol, period=period, interval=interval)
            if stock_data is None or stock_data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Transform data for chart format
        chart_data = []
        for index, row in stock_data.iterrows():
            try:
                chart_data.append({
                    "timestamp": str(row['datetime']) if 'datetime' in row.index else str(row['date']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })
            except Exception as e:
                logger.error(f"Error processing row for {symbol}: {e}")
                continue
        
        # Log the date range for debugging
        if chart_data:
            first_date = chart_data[0]["timestamp"]
            last_date = chart_data[-1]["timestamp"]
            logger.info(f"Date range for {symbol}: {first_date} to {last_date}")
        print('chart_data', chart_data)
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data_points": len(chart_data),
            "data": chart_data,
            "latest_price": float(stock_data['close'].iloc[-1]),
            "data_source": "yahoo"
        }
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}")
async def get_stock_data(symbol: str):
    """Get stock data for a specific symbol"""
    try:
        data_collector = DataCollector()
        stock_data = await data_collector.get_stock_data_yahoo(symbol, period="1y", interval="1d")
        
        # If Yahoo Finance fails, return error instead of fake data
        if stock_data is None or stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        return {
            "symbol": symbol,
            "data_points": len(stock_data),
            "columns": list(stock_data.columns),
            "latest_price": float(stock_data['close'].iloc[-1]) if 'close' in stock_data.columns else None,
            "data_source": "yahoo"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/features")
async def get_stock_features(symbol: str):
    """Get engineered features for a stock"""
    try:
        # Collect data
        data_collector = DataCollector()
        data = await data_collector.get_stock_data_yahoo(symbol, period="1y", interval="1d")
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.engineer_features(data)
        
        if features is None or features.empty:
            raise HTTPException(status_code=500, detail="Failed to engineer features")
        
        return {
            "symbol": symbol,
            "features_count": features.shape[1],
            "samples_count": features.shape[0],
            "feature_names": list(features.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/predict")
async def get_stock_prediction(
    symbol: str, 
    model: str = Query("random_forest", description="Model to use for prediction")
):
    """Get prediction for a stock"""
    try:
        # Collect and engineer data
        data_collector = DataCollector()
        data = await data_collector.get_stock_data_yahoo(symbol, period="1y", interval="1d")
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        engineer = FeatureEngineer()
        features_df = engineer.calculate_technical_indicators(data)
        features_df = engineer.create_target_variables(features_df)
        if features_df is None or features_df.empty:
            raise HTTPException(status_code=500, detail="Failed to engineer features")
        
        
        ml_manager = MLModelManager()

        X_train, X_test, y_train, y_test = engineer.prepare_ml_data(
            features_df, target_column='target_direction_1d', test_size=0.2, sequence_length=3
        )

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Train the model with the same name that will be used for prediction
        model_name = f"{symbol}_{model}"
        ml_manager.train_random_forest(
            X_train_flat, y_train, X_test_flat, y_test, 
            model_name=model_name, task="classification"
        )
        
        # Get latest features for prediction
        latest_features = X_train_flat[-1:].reshape(1, -1)
        prediction = ml_manager.predict(model_name, latest_features, return_probability=True)
        logger.info(f"Successfully predicted stock: {prediction}")
        
        # Handle prediction result
        if prediction is not None and len(prediction) > 0:
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Classification with probabilities
                pred_value = float(prediction[0][1])  # Probability of positive class
                confidence = "high" if abs(pred_value - 0.5) > 0.2 else "medium" if abs(pred_value - 0.5) > 0.1 else "low"
            else:
                # Regression or single value
                pred_value = float(prediction[0])
                confidence = "high" if abs(pred_value) > 0.7 else "medium" if abs(pred_value) > 0.5 else "low"
        else:
            pred_value = None
            confidence = "unknown"
        
        return {
            "symbol": symbol,
            "model": model,
            "prediction": pred_value,
            "confidence": confidence,
            "model_trained": model_name in ml_manager.models,
            "model_name": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    
@router.get("/{symbol}/analysis")
async def get_stock_analysis(symbol: str):
    """Fetch a detailed analysis of a stock"""
    try:
        financial_analyzer = FinancialAnalyzer()
        stock_analysis = await financial_analyzer.get_stock_analysis(symbol)
        logger.info(f"successfully fetched stock analysis {stock_analysis}")
        return stock_analysis
    except Exception as e:
        logger.error(f"Error fetching stock analysis for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{symbol}/train")
async def train_stock_models(symbol: str):
    """Train models for a specific stock"""
    try:
        # Collect data
        data_collector = DataCollector()
        data = await data_collector.get_stock_data_yahoo(symbol, period="1y", interval="1d")
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.engineer_features(data)
        
        if features is None or features.empty:
            raise HTTPException(status_code=500, detail="Failed to engineer features")
        
        # Prepare data
        X = features.drop(['target'], axis=1, errors='ignore')
        y = features['target'] if 'target' in features.columns else features.iloc[:, -1]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        ml_manager = MLModelManager()
        results = ml_manager.train_all_models_for_stock(
            X_train.values, y_train.values, 
            X_test.values, y_test.values,
            symbol=symbol,
            task="classification"
        )
        
        return {
            "symbol": symbol,
            "training_completed": results['success'],
            "models_trained": list(results['models_trained'].keys()),
            "message": f"Training completed for {symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
