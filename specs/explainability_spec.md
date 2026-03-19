# Explainability Specification

## Goal
Interpret model predictions and verify that models rely on economically sensible features.

## Methods
- **SHAP values**: Model-agnostic feature attribution per prediction
- **Feature importance over time**: Compare importances across walk-forward folds
- **Permutation importance**: Measure out-of-sample performance drop when feature is shuffled

## Inputs
- Trained model
- Feature matrix (out-of-sample only)
- Walk-forward fold results

## Outputs
- SHAP summary plot data (global feature ranking)
- Per-prediction SHAP values
- Feature importance stability scores (rank correlation across folds)
- Flagged spurious features (high importance in one fold, low in another)

## Constraints
- Compute SHAP only on out-of-sample predictions
- Must evaluate importance stability across at least 3 time folds
- Flag any feature that ranks top-5 in one fold but bottom-50% in another

## Success Criteria
- Top features are economically interpretable (momentum, volatility, volume — not noise)
- Feature importance is stable across time periods
- No single feature dominates predictions (concentration risk)
