# FinanceBro — Agents.md

## Overview

FinanceBro is a quantitative research and trading system designed to demonstrate:
- Statistical rigor
- Proper backtesting methodology
- Risk-aware portfolio construction
- Robust out-of-sample validation

This system is built using modular "agents", each responsible for a specific stage of the pipeline.

---

## Core Principles

1. No look-ahead bias
2. All performance must be out-of-sample
3. Backtests must include realistic costs
4. Overfitting must be measured, not assumed
5. Risk management > prediction accuracy

---

## System Pipeline

### Core Pipeline (implement in this order)
```
DataAgent → FeatureAgent → ModelAgent → WalkForwardAgent →
BacktestAgent → OverfittingAgent → RiskAgent → PortfolioAgent →
StatsAgent
```

### Advanced Extensions (after core pipeline works)
```
→ RegimeAgent → ExplainabilityAgent → RLAgent
```

Each stage is handled by a dedicated agent.

---

## Agent Interface Contract

Every agent must implement:

class BaseAgent:
def run(self, inputs): pass
def validate(self, inputs, outputs): pass
def log_metrics(self): pass

@property
def input_schema(self): pass

@property
def output_schema(self): pass


---

## Agents

---

### 1. DataAgent

**Responsibilities**
- Load historical market data (price, volume)
- Integrate factor + alternative datasets
- Ensure clean, aligned time-series

**Constraints**
- No survivorship bias
- No forward-filled labels

**Outputs**
- Cleaned dataframe indexed by timestamp

---

### 2. FeatureAgent

**Responsibilities**
- Generate predictive features using ONLY past data

**Examples**
- Returns (1d, 5d, 20d)
- Rolling volatility
- Momentum signals
- Factor exposures

**Rules**
- All features must use rolling windows
- No future leakage

**Outputs**
- Feature matrix `X(t)`
- Target `y(t + k)`

---

### 3. ModelAgent

**Responsibilities**
- Train predictive models

**Models**
- Baseline: Logistic Regression
- Tree-based: Random Forest, XGBoost
- Advanced: Bayesian models

**Validation**
- Walk-forward cross-validation ONLY

**Outputs**
- Predictions (probabilities or signals)

---

### 4. BacktestAgent (CRITICAL)

**Responsibilities**
- Simulate trading over time

**Requirements**
- Event-driven simulation (step through time)
- Execute trades using model outputs

**Market Realism**
- Transaction costs: 5–10 bps
- Slippage modeling

**Metrics**
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Calmar ratio
- Win rate
- Turnover

**Benchmark**
- SPY buy-and-hold

**Outputs**
- Equity curve
- Trade log
- Performance summary

---

### 5. WalkForwardAgent

**Responsibilities**
- Perform time-series validation

**Method**
- Expanding window:
  - Train: months 1–12 → Test: month 13
  - Train: months 1–13 → Test: month 14

**Outputs**
- Aggregated out-of-sample metrics

---

### 6. OverfittingAgent

**Responsibilities**
- Detect and quantify overfitting

**Techniques**
- Train vs test performance gap
- Parameter sensitivity analysis
- Bootstrap resampling

**Advanced**
- Probability of Backtest Overfitting (PBO)
- Combinatorial Cross-Validation

**Outputs**
- Overfitting score
- Stability diagnostics

---

### 7. RiskAgent

**Responsibilities**
- Apply risk-aware position sizing

**Methods**
- Volatility scaling
- Fractional Kelly Criterion

**Risk Controls**
- Max position size
- Stop-loss thresholds

**Metrics**
- VaR / CVaR

**Outputs**
- Position sizes per asset

---

### 8. PortfolioAgent

**Responsibilities**
- Allocate capital across assets

**Methods**
- Mean-Variance Optimization (Markowitz)
- Risk parity

**Enhancements**
- Covariance shrinkage (Ledoit-Wolf)

**Constraints**
- Position limits
- Sector exposure caps

**Outputs**
- Portfolio weights

---

### 9. RegimeAgent

**Responsibilities**
- Detect market regimes

**Methods**
- Hidden Markov Models
- Volatility clustering

**Use Cases**
- Switch strategies based on regime
- Condition model selection

---

### 10. StatsAgent

**Responsibilities**
- Validate statistical significance

**Techniques**
- Bootstrap Sharpe ratio
- Hypothesis testing (Sharpe > 0)
- Multiple testing correction (Bonferroni, FDR)

**Outputs**
- Confidence intervals
- P-values

---

### 11. ExplainabilityAgent

**Responsibilities**
- Interpret model predictions

**Methods**
- SHAP values
- Feature importance over time

**Outputs**
- Feature attribution reports

---

### 12. RLAgent (Advanced)

**Responsibilities**
- Learn trading strategies without labels

**Methods**
- DQN, PPO (model-free RL)

**Reward Function**
- Risk-adjusted returns

**Challenges**
- Non-stationarity
- Exploration vs exploitation

---

## Workflow Execution

### Core Pipeline
1. **DataAgent** → clean, aligned dataset (no survivorship bias)
2. **FeatureAgent** → feature matrix X(t) + target y(t+k) (no leakage)
3. **ModelAgent** → predictions / signals / probabilities
4. **WalkForwardAgent** → out-of-sample validation across expanding folds
5. **BacktestAgent** → event-driven trading simulation with costs
6. **OverfittingAgent** → robustness check, overfitting score
7. **RiskAgent** → risk-adjusted position sizing
8. **PortfolioAgent** → capital allocation across assets
9. **StatsAgent** → statistical significance testing

### Advanced Extensions
10. **RegimeAgent** → market regime detection, conditional model selection
11. **ExplainabilityAgent** → SHAP values, feature attribution
12. **RLAgent** → reinforcement learning strategies (experimental)


---

## Research Standards

- Always report out-of-sample performance
- Always include transaction costs
- Always benchmark against SPY
- Always disclose when performance degrades

---

## Common Failure Modes (Must Avoid)

- Look-ahead bias (automatic rejection)
- Single train/test split
- Ignoring transaction costs
- Overfitting via parameter tuning
- Confusing correlation with alpha

---

## Interview Narrative Target

You should be able to say:

"I built an end-to-end quant system. Initially, my model had look-ahead bias, which I identified and fixed. After implementing walk-forward validation and a proper backtester, performance dropped, revealing overfitting. I added robustness checks, risk-aware position sizing, and portfolio optimization, achieving statistically significant out-of-sample performance."

---

## What NOT to Prioritize

- LLM features / chat interfaces
- Frontend/UI polish
- Adding more technical indicators

Focus on:
- Validation
- Risk
- Statistical rigor

---

## Future Extensions

- Alternative data (news, sentiment)
- Multi-asset portfolios
- High-frequency simulation
- Execution optimization

---

## Final Note

A simple model with rigorous validation > complex model with flawed methodology.

