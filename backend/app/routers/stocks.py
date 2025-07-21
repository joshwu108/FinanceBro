from fastapi import APIRouter, HTTPException
from app.services.data_collector import DataCollector
from app.services.ml_models import MLModelManager
import numpy as np

router = APIRouter()

@router.get("/stocks")
async def get_stocks():
    return ""

@router.get("/stocks/{symbol}")
async def get_stock_data(symbol: str):
    try:
        data_collector = DataCollector()
        stock_data = data_collector.get_stock_data(symbol)
        return {"symbol": symbol, "data": stock_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

@router.get("/predictions/{symbol}")
async def get_stock_prediction(symbol: str, X: list):
    try:
        mm_manager = MLModelManager()
        prediction = mm_manager.predict(symbol, X, return_probability=True)
        return {"symbol": symbol, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)