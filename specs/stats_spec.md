# Statistical Validation Specification

## Goal
Determine whether strategy performance is statistically significant or could arise by chance.

## Methods
- **Bootstrap Sharpe ratio**: Resample returns with replacement, compute Sharpe distribution
- **Hypothesis testing**: H0: Sharpe ≤ 0, H1: Sharpe > 0
- **Multiple testing correction**: Bonferroni or Benjamini-Hochberg FDR when testing multiple models/symbols

## Inputs
- Returns series (out-of-sample)
- Sharpe ratio
- Number of models/symbols tested (for multiple testing correction)

## Outputs
- Sharpe confidence interval (95%)
- p-value for Sharpe > 0
- significance_flag (boolean)
- corrected_p_values (if multiple tests)

## Constraints
- Bootstrap must use block bootstrap (preserve autocorrelation in returns)
- Minimum 1000 bootstrap iterations
- Multiple testing correction is mandatory when comparing >1 model or symbol
- Report both raw and corrected p-values

## Success Criteria
- Sharpe ratio is significantly > 0 at p < 0.05 after correction
- Confidence interval does not include 0
- Results are robust to bootstrap seed variation
