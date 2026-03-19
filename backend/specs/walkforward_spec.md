# Walk-Forward Validation Specification

## Goal
Implement time-series cross-validation.

## Method
- Expanding window validation

## Requirements
- Train on past only
- Test on unseen future

## Outputs
- Per-fold metrics
- Aggregated performance

## Success Criteria
- No leakage
- Consistent evaluation across folds