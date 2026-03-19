# Regime Detection Specification

## Goal
Identify distinct market regimes to enable conditional strategy selection.

## Methods
- **Hidden Markov Model (HMM)**: 2-3 state model on returns and volatility
- **Volatility threshold**: Simple high/low volatility classification using rolling std
- **Structural break detection**: CUSUM test for regime transitions

## Inputs
- Price data (OHLCV)
- Returns series
- Rolling volatility estimates

## Outputs
- Regime labels per timestamp
- Regime probabilities per timestamp
- Transition matrix
- Current (most recent) regime

## Constraints
- All regime labeling must be causal (no future data)
- HMM must be fit using expanding window, not full dataset
- Minimum regime duration: 20 trading days (avoid noise)
- Start with 2 states (high-vol / low-vol) before attempting more

## Validation
- Report backtest performance broken down by regime
- If performance is similar across regimes, the detection adds no value — report this honestly

## Success Criteria
- Regimes are stable and interpretable
- Regime-conditioned strategies outperform unconditional strategies
- Regime transitions are detectable in near-real-time (not just in hindsight)
