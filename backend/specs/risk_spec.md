# Risk Management Specification

## Goal
Implement risk-aware position sizing that controls downside exposure.

## Methods
- **Volatility scaling**: Size positions inversely to recent volatility
- **Fractional Kelly Criterion**: Optimal sizing with conservative fraction (half-Kelly or less)

## Inputs
- Backtest results (returns, signals)
- Volatility estimates (rolling window)
- Config (max position size, stop-loss thresholds)

## Outputs
- Position sizes per asset per timestep
- Risk metrics: VaR (95%, 99%), CVaR, max position exposure

## Risk Controls
- Max single position: configurable, default 10% of portfolio
- Stop-loss: exit if position drawdown exceeds threshold
- Portfolio-level VaR limit: halt new positions if breached

## Constraints
- Position sizes must be computable using only past data
- Kelly sizing must use out-of-sample win rate and payoff ratio, not in-sample
- Volatility estimates must use rolling windows (not full-sample std)

## Success Criteria
- Reduces max drawdown compared to equal-weight positions
- VaR estimates are calibrated (actual breaches ≈ expected frequency)
- Position sizes are stable (low turnover)
