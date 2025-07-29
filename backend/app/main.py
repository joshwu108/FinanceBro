from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from app.routers import stocks, alerts, portfolio
from app.services.database import init_db
from app.services.redis_client import init_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting FinanceBro API...")
    await init_db()
    await init_redis()
    print("Database and Redis initialized")
    
    yield
    # Shutdown
    print("Shutting down FinanceBro API...")


app = FastAPI(
    title="FinanceBro API",
    description="AI-Powered Financial Agent API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stocks.router, prefix="/api/v1/stocks", tags=["stocks"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
#app.include_router(models.router, prefix="/api/v1/models", tags=["models"])


@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse({
        "message": "Welcome to FinanceBro API",
        "version": "1.0.0",
        "status": "running"
    })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    })


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 