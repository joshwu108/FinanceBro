# Backtesting Specification

## Goal
Build an event-driven backtesting engine.

## Requirements
- Simulate trades step-by-step over time
- Include transaction costs (5–10 bps)
- Include slippage

## Inputs
- Historical price data
- Model predictions

## Outputs
- Equity curve
- Sharpe ratio
- Max drawdown

## Constraints
- No look-ahead bias
- Realistic execution assumptions

## Success Criteria
- Produces stable and reproducible results
- Matches known benchmarks on simple strategies