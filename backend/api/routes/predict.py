from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.predict_service import get_inference_bundle, predict_for_symbol

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, description="Ticker symbol, e.g. AAPL")
    refresh_data: bool = Field(
        default=False,
        description="If true, refetch OHLCV via DataAgent (slower)",
    )


class PredictResponse(BaseModel):
    symbol: str
    prediction: float = Field(..., description="P(class=1), upward direction")
    signal: str = Field(..., description="long or short (threshold 0.5)")
    timestamp: str = Field(
        ...,
        description="ISO timestamp of the bar used for features",
    )
    model_type: str | None = None


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    """Score the latest bar with a trained model (no training on this path)."""
    sym = body.symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=422, detail="symbol is required")

    try:
        get_inference_bundle(sym)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No inference artifact for {sym}. Run the training pipeline for "
                f"this symbol first (expected models/{sym}_inference.joblib)."
            ),
        ) from None

    try:
        out = predict_for_symbol(
            sym,
            force_refresh_data=body.refresh_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        ) from e

    return PredictResponse(**out)

@router.get("/api/predict/{symbol}")
async def predict_latest(symbol: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="1m")
    feature_agent = FeatureAgent()
    features = feature_agent.run({"cleaned_data": df})
    model_agent = ModelAgent()
    model_agent.load_model(f"models/{symbol}_final.joblib")
    prediction = model_agent.predict(features.tail(1))
    return {
        "symbol": symbol,
        "prediction": prediction,
        "timestamp": df.index[-1]
    }