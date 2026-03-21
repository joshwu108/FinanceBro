# FinanceBro

FinanceBro is a quantitative research and trading system focused on statistical rigor, realistic backtesting, and out-of-sample validation.

The project uses a modular agent pipeline for data, feature engineering, modeling, walk-forward validation, backtesting, risk, portfolio construction, and significance testing.

## Current Stack

### Backend
- FastAPI (API server)
- Python agents in backend/agents
- Orchestrator in backend/orchestrator
- HTTPX/WebSockets for market data transport
- pandas, NumPy, scikit-learn, XGBoost, PyTorch

### Frontend
- Next.js + React + TypeScript
- Tailwind CSS
- Lightweight charts and dashboard components

## Repository Layout

```text
FinanceBro/
	backend/
		agents/                 # Data/feature/model/backtest/risk/stats agents
		api/                    # FastAPI server and routes
		orchestrator/           # End-to-end pipeline runner
		specs/                  # Agent and pipeline specifications
		experiments/            # Logged experiment outputs
		tests/                  # Agent and pipeline tests
		requirements.txt
	frontend/                 # Next.js application
	env.example               # Environment variable template
	Agents.MD                 # Master quant agent spec
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- A root environment file at FinanceBro/.env

## Environment Variables

Create a root .env file from the template:

```bash
cp env.example .env
```

Important:
- The backend loads the root .env file (FinanceBro/.env), not a backend/.env or frontend/.env.
- For Alpaca market snapshot/live data, set either:
	- ALPACA_KEY + ALPACA_SECRET
	- or ALPACA_API_KEY + ALPACA_SECRET_KEY
- Optional Alpaca runtime configuration:
	- ALPACA_BASE_URL (contains "sandbox" to auto-select sandbox defaults)
	- ALPACA_FEED (default: iex)
	- ALPACA_DATA_BASE_URL (explicit REST data endpoint override)
	- ALPACA_STREAM_BASE_URL (explicit websocket endpoint override)

## Local Development

### 1. Backend

From repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn api.server:app --reload --port 8000
```

### 2. Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend default: http://localhost:3000
Backend default: http://localhost:8000

## Running Tests

From backend directory:

```bash
cd backend
PYTHONPATH=. pytest -q tests
```

If you run an individual test module directly, include PYTHONPATH=. so imports like agents.* resolve.

## Pipeline Overview

Core order:

```text
DataAgent -> FeatureAgent -> ModelAgent -> WalkForwardAgent ->
BacktestAgent -> OverfittingAgent -> RiskAgent -> PortfolioAgent ->
StatsAgent
```

The orchestrator implementation is in backend/orchestrator/run_pipeline.py.

## API Endpoints

- POST /run_pipeline
- GET /api/market/status
- GET /api/market/snapshot/{symbol}
- WebSocket /ws/live/{symbol}

## Notes

- Experiments are logged under backend/experiments.
- This project prioritizes validation, risk control, and statistical significance over UI polish or indicator count.

