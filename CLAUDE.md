# FinanceBro — Claude Code Project Guide

## Project Overview

FinanceBro is a quantitative research and trading system that demonstrates statistical rigor, proper backtesting methodology, risk-aware portfolio construction, and robust out-of-sample validation. The system is built using modular agents, each responsible for a specific stage of the quant pipeline.

**Master spec**: `Agents.MD` — read this first for the full agent contract, pipeline, and research standards.

---

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL (primary) + SQLite (fallback), SQLAlchemy ORM
- **Caching**: Redis
- **Background tasks**: Celery
- **ML**: scikit-learn, XGBoost, PyTorch (LSTM)
- **NLP**: HuggingFace Transformers (FinBERT), sentence-transformers
- **LLMs**: LangChain, Together AI
- **Data**: yfinance, Alpha Vantage, Pandas, NumPy

### Frontend
- **Framework**: React 18, TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Syncfusion EJ2 React Charts
- **Real-time**: Socket.io, WebSockets

---

## Commands

```bash
# Backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload

# Frontend
cd frontend && npm start

# ML Pipeline
cd backend && python -m ml_pipeline.train_models

# Tests
cd backend && pytest
```

---

## Project Structure

```
backend/                    # FastAPI app, existing services & routers
  app/
    routers/                # stocks, alerts, portfolio, websocket
    services/               # data_collector, feature_engineering, ml_models,
                            # financial_analyzer, sentiment_analyzer,
                            # price_broadcaster, websocket_manager, database, redis
  ml_pipeline/              # Existing ML training pipeline
frontend/                   # React 18 + TypeScript dashboard
agents/                     # Agent implementations (BaseAgent subclasses)
subagents/                  # Role-based agent persona specs (markdown)
specs/                      # Technical requirement documents per agent
orchestrator/               # Pipeline runner that chains agents
experiments/                # Experiment logs (JSON), template, results
data/                       # Cached datasets, intermediate artifacts
```

---

## Hard Constraints (NEVER violate)

### 1. No Look-Ahead Bias
- No future data at time t. No `.shift(-k)` in feature construction.
- All rolling calculations use past data only.
- Violations must be flagged immediately and blocked.
- **Known bug**: `backend/app/services/feature_engineering.py` uses `shift(-days)` for target variables — this MUST be fixed before any agent work begins.

### 2. Time-Series Validation
- No random train/test splits. Ever.
- Always use walk-forward (expanding or sliding window) validation.
- Maintain strict temporal ordering in all data operations.

### 3. Backtesting Realism
- Event-driven (step through time, no vectorized shortcuts that leak).
- Transaction costs: 5–10 bps per trade.
- Slippage modeling required.
- Only information available at each timestep may be used.

### 4. Experiment Tracking
- Every pipeline run MUST log to `/experiments/` using the template.
- No silent experiments. No unlogged results.
- Always mark whether results are out-of-sample.

---

## Development Rules

### BaseAgent Contract
All agents MUST implement:
```python
class BaseAgent:
    def run(self, inputs): pass
    def validate(self, inputs, outputs): pass
    def log_metrics(self): pass

    @property
    def input_schema(self): pass

    @property
    def output_schema(self): pass
```
No agent may bypass this interface.

### Separation of Concerns
- Data logic → DataAgent
- Feature engineering → FeatureAgent
- Modeling → ModelAgent
- Validation → WalkForwardAgent
- Backtesting → BacktestAgent
- Overfitting detection → OverfittingAgent
- Risk management → RiskAgent
- Portfolio construction → PortfolioAgent
- Statistical validation → StatsAgent
- Regime detection → RegimeAgent
- Explainability → ExplainabilityAgent

Do NOT mix responsibilities across agents.

### Spec-Driven Development
Before implementing ANY agent or feature:
1. Check `/specs/` for an existing spec
2. If no spec exists → create one first
3. Implement strictly against the spec
4. Validate against all hard constraints
5. Log experiment results

### Simplicity Over Complexity
- Start with simple models (logistic regression baseline).
- Validate rigorously before adding complexity.
- Do not add ML models or indicators without justification.
- A simple model with rigorous validation > complex model with flawed methodology.

### Critic Mindset (Mandatory)
Always act as BOTH builder and reviewer:
- Question results. Look for overfitting.
- Flag unrealistic assumptions.
- If a Sharpe ratio looks too good, it probably is.

---

## Pipeline Execution Order

```
DataAgent → FeatureAgent → ModelAgent → WalkForwardAgent →
BacktestAgent → OverfittingAgent → RiskAgent → PortfolioAgent →
StatsAgent
```

Optional advanced stages (implement after core pipeline works):
```
→ RegimeAgent → ExplainabilityAgent → RLAgent
```

---

## Building Agents from Existing Code

The `backend/app/services/` directory contains working but unvalidated implementations. Agents should **wrap, refactor, and harden** this code — not rewrite from scratch:

| Agent | Existing Service | What to Fix |
|-------|-----------------|-------------|
| DataAgent | `data_collector.py` | Add survivorship bias checks, validate time alignment |
| FeatureAgent | `feature_engineering.py` | **Fix look-ahead bias** (`shift(-days)`), enforce rolling-only windows |
| ModelAgent | `ml_models.py` | Remove debug prints, add early stopping, add baseline model |
| BacktestAgent | (none — build new) | Event-driven simulation against specs |
| WalkForwardAgent | (none — build new) | Expanding window CV per spec |
| OverfittingAgent | (none — build new) | Train/test gap, PBO, parameter sensitivity |
| RiskAgent | (none — build new) | Kelly sizing, VaR/CVaR, position limits |
| PortfolioAgent | `portfolio.py` (router only) | Add Markowitz optimization, risk parity, covariance shrinkage |
| StatsAgent | (none — build new) | Bootstrap Sharpe, hypothesis testing, multiple testing correction |
| RegimeAgent | (none — build new) | HMM, volatility clustering |
| ExplainabilityAgent | (none — build new) | SHAP values, feature importance over time |

---

## Known Issues to Fix

1. **Look-ahead bias in feature_engineering.py** — `shift(-days)` leaks future data into targets
2. **Debug prints in ml_models.py** — lines with "reached here 2/3/4" left in production
3. **Prediction endpoint uses training data** — `stocks.py` predicts on `X_train_flat[-1:]`, not live data
4. **No early stopping** — XGBoost and LSTM train without validation-based stopping
5. **Sentiment is decorative** — sentiment scores are never used as model features
6. **Aggressive NaN handling** — `ffill().bfill()` masks data quality issues
7. **Hardcoded targets** — financial_analyzer.py uses fixed +7%/-4% targets, not volatility-derived

---

## What NOT to Prioritize

- LLM features, chat interfaces, or agent narratives
- Frontend/UI polish
- Adding more technical indicators (30+ already exist)
- WebSocket or real-time features

Focus exclusively on: **validation, risk, statistical rigor, backtesting**.

---

## Research Standards

- Always report out-of-sample performance
- Always include transaction costs in backtests
- Always benchmark against SPY buy-and-hold
- Always disclose when performance degrades
- Always use walk-forward validation, never single splits
