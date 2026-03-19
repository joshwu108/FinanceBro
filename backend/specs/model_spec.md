# ModelAgent Spec

## Role
Train predictive models on the feature matrix produced by FeatureAgent. Always start with a simple baseline (logistic regression) before adding complexity. Report honest, unbiased metrics with skepticism toward strong results.

## Pipeline Position
```
DataAgent → FeatureAgent → **ModelAgent** → WalkForwardAgent → BacktestAgent
```

## Inputs
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `feature_matrix` | `pd.DataFrame` (DatetimeIndex) | Yes | Features from FeatureAgent — no target columns, no raw OHLCV |
| `target` | `pd.Series` (DatetimeIndex) | Yes | Binary target (0/1) from FeatureAgent, aligned to feature_matrix |
| `model_config` | `dict` | No | Overrides for model_type, hyperparameters, train_ratio |
| `train_end_date` | `str` (ISO date) | No | Temporal cutoff — train on data <= this date, test on data after. Overrides train_ratio. Used by WalkForwardAgent. |

## Outputs
| Field | Type | Description |
|-------|------|-------------|
| `predictions` | `pd.Series` (DatetimeIndex) | P(class=1) probabilities on the test set, aligned to timestamps |
| `predicted_classes` | `pd.Series` (DatetimeIndex) | Binary 0/1 predictions on the test set |
| `train_metrics` | `dict` | accuracy, precision, recall, f1, log_loss |
| `test_metrics` | `dict` | accuracy, precision, recall, f1, log_loss |
| `train_test_gap` | `dict` | Per-metric (train − test) differences — overfitting signal |
| `feature_importances` | `dict` or `None` | feature_name → importance score (abs coef for LR, Gini for RF) |
| `model_type` | `str` | Which model was trained |
| `trained_model` | sklearn estimator | Serializable trained model |
| `scaler` | `StandardScaler` | Fitted on training data only |
| `split_info` | `dict` | train_size, test_size, train_start, train_end, test_start, test_end |

## Model Progression
1. **Logistic Regression** (baseline) — must beat this to justify complexity
2. **Random Forest** (conservative: n_estimators=100, max_depth=5, min_samples_leaf=20)

## Hard Constraints
1. **No random splits** — temporal split ONLY. All train data must precede all test data.
2. **Scaler fit on train only** — StandardScaler.fit_transform(train), transform(test).
3. **No look-ahead bias** — model never sees future data during training.
4. **Skepticism mandate** — if test accuracy > 60% on daily direction, warn about possible leakage. If train-test gap > 10%, warn about overfitting.
5. **Conservative hyperparameters** — no 500-estimator / max_depth=10 forests on small datasets.

## Validation Checks
- Predictions in [0, 1], no NaN
- Predicted classes in {0, 1}
- Test timestamps strictly after train timestamps (no temporal overlap)
- All metrics finite

## Metrics to Report
- Accuracy, precision, recall, F1 (binary classification)
- Log loss (calibration quality)
- Directional accuracy vs 50% random baseline
- Train-test gap per metric (overfitting signal)

## Fixes vs. oldBackend/app/services/ml_models.py
- Debug prints removed ("reached here 2/3/4")
- Logistic regression baseline added
- 500 estimators / max_depth=10 → 100 / 5
- Predictions output as timestamp-aligned probabilities
- Train-test gap tracked explicitly
- Scaler strictly train-only
