# Portfolio Construction Specification

## Goal
Optimally allocate capital across assets to maximize risk-adjusted returns.

## Methods
- **Mean-Variance Optimization (Markowitz)**: Minimize portfolio variance for target return
- **Risk Parity**: Equalize marginal risk contribution across assets
- **Minimum Variance**: Global minimum variance portfolio (no return forecast needed)

## Inputs
- Risk-adjusted positions from RiskAgent
- Historical returns for covariance estimation
- Config (position limits, sector caps, rebalancing frequency)

## Outputs
- Portfolio weights per asset
- Efficient frontier curve (for Markowitz)
- Portfolio expected return, risk, Sharpe ratio
- Diversification ratio

## Covariance Estimation
- Use Ledoit-Wolf shrinkage (not raw sample covariance)
- Minimum estimation window: 252 trading days (1 year)
- Re-estimate at each rebalancing point using only past data

## Constraints
- Long-only (no shorting unless config explicitly allows)
- Max position weight: configurable, default 10%
- Weights sum to ≤ 1.0 (allow cash position)
- Rebalancing incurs transaction costs (must be accounted for)

## Success Criteria
- Portfolio Sharpe exceeds equal-weight benchmark
- No single position exceeds max weight
- Turnover is reasonable (not rebalancing excessively)
