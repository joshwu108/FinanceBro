# Pipeline Orchestrator

## Description
Executes the full quant research pipeline, chaining all agents in sequence with validation at each step.

## Core Pipeline Steps
1. **DataAgent** — Load and clean historical market data (no survivorship bias)
2. **FeatureAgent** — Generate features using only past data (no look-ahead bias)
3. **ModelAgent** — Train predictive model (baseline first, then complex)
4. **WalkForwardAgent** — Expanding window out-of-sample validation
5. **BacktestAgent** — Event-driven trading simulation with transaction costs and slippage
6. **OverfittingAgent** — Detect overfitting via train/test gap, parameter sensitivity, PBO
7. **RiskAgent** — Apply risk-adjusted position sizing (Kelly, volatility scaling)
8. **PortfolioAgent** — Allocate capital across assets (Markowitz, risk parity)
9. **StatsAgent** — Test statistical significance (bootstrap Sharpe, hypothesis testing)

## Advanced Extensions (after core works)
10. **RegimeAgent** — Detect market regimes, condition model selection
11. **ExplainabilityAgent** — SHAP values, feature attribution over time
12. **RLAgent** — Reinforcement learning strategies (experimental)

## Config
```python
config = {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "model_type": "random_forest",
    "transaction_costs_bps": 5.0,
    "slippage_bps": 2.0,
    "max_position_size": 0.1,
    "benchmark": "SPY",
}
```

## Output
- Full results dict with walk-forward, backtest, overfitting, risk, portfolio, and stats
- Experiment JSON logged automatically to `/experiments/`

## Validation
Every agent calls `validate()` after `run()`. If validation fails, the pipeline halts with a clear error — no silent failures.
