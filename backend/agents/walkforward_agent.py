"""WalkForwardAgent — Expanding window walk-forward validation.

Performs time-series cross-validation by:
  1. Splitting data into expanding train windows and fixed-size test windows
  2. For each fold: training a model, generating predictions, running a backtest
  3. Aggregating per-fold metrics into summary statistics

Constraints:
  - No temporal leakage: each fold trains on past data only
  - No random splits — strictly expanding window
  - Each fold's test window is strictly after its train window
  - Test windows do not overlap across folds
  - Uses ModelAgent for training and BacktestAgent for evaluation
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from agents.backtest_agent import BacktestAgent
from agents.model_agent import ModelAgent

logger = logging.getLogger(__name__)


class WalkForwardAgent(BaseAgent):
    """Expanding window walk-forward validation engine.

    Train on [t0 -> tN], test on [tN+1 -> tN+k].
    Advance the window, repeat for n_folds.
    Aggregate Sharpe, drawdown, model metrics across all folds.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_folds": 5,
        "min_train_size": 126,   # ~6 months of trading days
        "test_size": 63,         # ~3 months of trading days
        "model_config": {
            "model_type": "logistic_regression",
        },
        "backtest_config": {},
        "signal_threshold": 0.5,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "feature_matrix": (
                "pd.DataFrame — features from FeatureAgent, DatetimeIndex"
            ),
            "target": (
                "pd.Series — binary target (0/1) from FeatureAgent, DatetimeIndex"
            ),
            "price_data": (
                "pd.DataFrame — OHLCV with DatetimeIndex, columns: "
                "open, high, low, close, volume"
            ),
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "fold_results": (
                "list[dict] — per-fold results, each containing: "
                "fold_index, split_info, model_metrics, backtest_metrics, model_type"
            ),
            "aggregated_metrics": (
                "dict — mean_sharpe, std_sharpe, mean_max_drawdown, "
                "worst_max_drawdown, mean_total_return, mean_accuracy, mean_f1"
            ),
            "n_folds": "int — number of folds executed",
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute walk-forward validation across expanding windows.

        For each fold:
          1. Temporal split (expanding train, fixed test)
          2. Train model via ModelAgent
          3. Convert predictions to trading signals
          4. Run backtest via BacktestAgent
          5. Collect metrics

        Returns dict with fold_results, aggregated_metrics, n_folds.
        """
        X, y, price_data, cfg = self._validate_inputs(inputs)

        n_folds = cfg["n_folds"]
        min_train_size = cfg["min_train_size"]
        test_size = cfg["test_size"]
        model_config = cfg.get("model_config", {})
        backtest_config = cfg.get("backtest_config", {})
        signal_threshold = cfg.get("signal_threshold", 0.5)

        total_rows = len(X)
        required_rows = min_train_size + n_folds * test_size
        if total_rows < required_rows:
            raise ValueError(
                f"Insufficient data: have {total_rows} rows, need at least "
                f"{required_rows} (min_train_size={min_train_size} + "
                f"{n_folds} folds * test_size={test_size})"
            )

        fold_boundaries = self._compute_fold_boundaries(
            total_rows=total_rows,
            n_folds=n_folds,
            min_train_size=min_train_size,
            test_size=test_size,
        )

        fold_results: List[Dict[str, Any]] = []

        for fold_idx, (train_end_idx, test_start_idx, test_end_idx) in enumerate(
            fold_boundaries
        ):
            logger.info(
                "WalkForward fold %d/%d: train=[0:%d], test=[%d:%d]",
                fold_idx + 1, n_folds,
                train_end_idx, test_start_idx, test_end_idx,
            )

            fold_result = self._run_single_fold(
                fold_index=fold_idx,
                X=X,
                y=y,
                price_data=price_data,
                train_end_idx=train_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                model_config=model_config,
                backtest_config=backtest_config,
                signal_threshold=signal_threshold,
            )
            fold_results.append(fold_result)

        aggregated = self._aggregate_fold_metrics(fold_results)

        outputs: Dict[str, Any] = {
            "fold_results": fold_results,
            "aggregated_metrics": aggregated,
            "n_folds": len(fold_results),
        }

        self.validate(inputs, outputs)

        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "n_folds": len(fold_results),
            **aggregated,
        }

        logger.info(
            "WalkForwardAgent complete: %d folds, mean_sharpe=%.4f, "
            "worst_dd=%.4f",
            len(fold_results),
            aggregated["mean_sharpe"],
            aggregated["worst_max_drawdown"],
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate walk-forward outputs for structure and temporal integrity."""
        if "fold_results" not in outputs:
            raise ValueError("outputs missing 'fold_results'")
        if "aggregated_metrics" not in outputs:
            raise ValueError("outputs missing 'aggregated_metrics'")

        fold_results = outputs["fold_results"]
        if not fold_results:
            raise ValueError("fold_results is empty — no folds were executed")

        for i, fold in enumerate(fold_results):
            required = {"fold_index", "split_info", "model_metrics", "backtest_metrics"}
            missing = required - set(fold.keys())
            if missing:
                raise ValueError(f"Fold {i} missing keys: {missing}")

            split = fold["split_info"]
            train_end = pd.Timestamp(split["train_end"])
            test_start = pd.Timestamp(split["test_start"])
            if test_start <= train_end:
                raise ValueError(
                    f"Fold {i}: temporal leak — test_start ({test_start}) "
                    f"<= train_end ({train_end})"
                )

        for i in range(1, len(fold_results)):
            prev_test_end = pd.Timestamp(
                fold_results[i - 1]["split_info"]["test_end"]
            )
            curr_test_start = pd.Timestamp(
                fold_results[i]["split_info"]["test_start"]
            )
            if curr_test_start <= prev_test_end:
                raise ValueError(
                    f"Fold {i} test overlaps fold {i-1}: "
                    f"test_start={curr_test_start} <= prev_test_end={prev_test_end}"
                )

        agg = outputs["aggregated_metrics"]
        required_agg = {
            "mean_sharpe", "std_sharpe", "mean_max_drawdown",
            "worst_max_drawdown", "mean_total_return",
            "mean_accuracy", "mean_f1",
        }
        missing_agg = required_agg - set(agg.keys())
        if missing_agg:
            raise ValueError(f"aggregated_metrics missing keys: {missing_agg}")

        return True

    def log_metrics(self) -> None:
        """Persist metrics from the most recent run to experiments/."""
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        experiments_dir = self._experiments_dir
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        run_id = self._metrics.get("run_id", uuid.uuid4().hex[:12])

        log_entry = {
            "experiment_id": f"walkforward_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "WalkForwardAgent",
            "stage": "walk_forward_validation",
            "timestamp": now.isoformat(),
            "parameters": {
                "n_folds": self._metrics.get("n_folds"),
            },
            "out_of_sample": True,
            "metrics": {
                "n_folds": self._metrics.get("n_folds"),
                "mean_sharpe": self._metrics.get("mean_sharpe"),
                "std_sharpe": self._metrics.get("std_sharpe"),
                "mean_max_drawdown": self._metrics.get("mean_max_drawdown"),
                "worst_max_drawdown": self._metrics.get("worst_max_drawdown"),
                "mean_total_return": self._metrics.get("mean_total_return"),
                "mean_accuracy": self._metrics.get("mean_accuracy"),
                "mean_f1": self._metrics.get("mean_f1"),
            },
            "notes": "WalkForwardAgent expanding window validation run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"walkforward_agent_{ts}_{run_id}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    @property
    def _experiments_dir(self) -> Path:
        """Directory for experiment logs. Property for easy test patching."""
        return Path(__file__).parent.parent / "experiments"

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
        """Validate and extract inputs."""
        if "feature_matrix" not in inputs:
            raise ValueError("inputs must contain 'feature_matrix'")
        if "target" not in inputs:
            raise ValueError("inputs must contain 'target'")
        if "price_data" not in inputs:
            raise ValueError("inputs must contain 'price_data'")

        X = inputs["feature_matrix"]
        y = inputs["target"]
        price_data = inputs["price_data"]

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"'feature_matrix' must be pd.DataFrame, got {type(X)}"
            )
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"'target' must be pd.Series, got {type(y)}"
            )
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError(
                f"'price_data' must be pd.DataFrame, got {type(price_data)}"
            )

        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("feature_matrix must have DatetimeIndex")

        if not X.index.equals(y.index):
            raise ValueError("feature_matrix and target indices don't match")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(price_data.columns)
        if missing_cols:
            raise ValueError(
                f"price_data missing required columns: {missing_cols}"
            )

        cfg = {**self._config, **(inputs.get("config") or {})}

        return X, y, price_data, cfg

    # ── Internal: fold boundary computation ───────────────────────

    @staticmethod
    def _compute_fold_boundaries(
        total_rows: int,
        n_folds: int,
        min_train_size: int,
        test_size: int,
    ) -> List[tuple]:
        """Compute (train_end_idx, test_start_idx, test_end_idx) for each fold.

        Uses expanding window: train grows by test_size each fold.
        First fold trains on [0 : min_train_size), tests on
        [min_train_size : min_train_size + test_size).
        """
        boundaries = []
        for fold in range(n_folds):
            train_end_idx = min_train_size + fold * test_size - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_size - 1

            if test_end_idx >= total_rows:
                test_end_idx = total_rows - 1

            if test_start_idx >= total_rows:
                break

            boundaries.append((train_end_idx, test_start_idx, test_end_idx))

        return boundaries

    # ── Internal: single fold execution ───────────────────────────

    def _run_single_fold(
        self,
        fold_index: int,
        X: pd.DataFrame,
        y: pd.Series,
        price_data: pd.DataFrame,
        train_end_idx: int,
        test_start_idx: int,
        test_end_idx: int,
        model_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        signal_threshold: float,
    ) -> Dict[str, Any]:
        """Execute a single walk-forward fold.

        1. Split data temporally
        2. Train model via ModelAgent (using train_end_date)
        3. Convert probabilities to trading signals
        4. Run backtest via BacktestAgent on the test window
        5. Return fold result dict
        """
        # Temporal slicing
        X_train = X.iloc[: train_end_idx + 1]
        y_train = y.iloc[: train_end_idx + 1]
        X_test = X.iloc[test_start_idx: test_end_idx + 1]
        y_test = y.iloc[test_start_idx: test_end_idx + 1]
        price_test = price_data.iloc[test_start_idx: test_end_idx + 1]

        train_end_date = str(X_train.index[-1].date())

        # Combine train + test for ModelAgent (it splits internally via train_end_date)
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])

        # Phase 1: Train model
        model_agent = ModelAgent()
        model_output = model_agent.run({
            "feature_matrix": X_combined,
            "target": y_combined,
            "train_end_date": train_end_date,
            "model_config": model_config,
        })

        # Phase 2: Convert probability predictions to trading signals
        # P(up) > threshold => long (1), P(up) < (1-threshold) => short (-1), else flat (0)
        predictions_proba = model_output["predictions"]
        signals = pd.Series(0, index=predictions_proba.index, dtype=int)
        signals[predictions_proba >= signal_threshold] = 1
        signals[predictions_proba < (1 - signal_threshold)] = -1

        # Verify signals and price_test share the same index to prevent
        # silent data misalignment that would corrupt backtest metrics.
        if not signals.index.equals(price_test.index):
            raise ValueError(
                f"Fold {fold_index}: signals index does not match price_test index. "
                "Ensure price_data shares the same DatetimeIndex as feature_matrix."
            )

        # Phase 3: Run backtest on the test window
        backtest_agent = BacktestAgent(config=backtest_config)
        backtest_output = backtest_agent.run({
            "price_data": price_test,
            "predictions": signals,
        })

        # Phase 4: Collect results
        split_info = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_start": str(X_train.index[0]),
            "train_end": str(X_train.index[-1]),
            "test_start": str(X_test.index[0]),
            "test_end": str(X_test.index[-1]),
        }

        return {
            "fold_index": fold_index,
            "split_info": split_info,
            "model_metrics": model_output["test_metrics"],
            "backtest_metrics": backtest_output["performance_summary"],
            "model_type": model_output["model_type"],
        }

    # ── Internal: aggregation ─────────────────────────────────────

    @staticmethod
    def _aggregate_fold_metrics(
        fold_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate per-fold metrics into summary statistics."""
        sharpes = [f["backtest_metrics"]["sharpe"] for f in fold_results]
        drawdowns = [f["backtest_metrics"]["max_drawdown"] for f in fold_results]
        returns = [f["backtest_metrics"]["total_return"] for f in fold_results]
        accuracies = [f["model_metrics"]["accuracy"] for f in fold_results]
        f1s = [f["model_metrics"]["f1"] for f in fold_results]

        return {
            "mean_sharpe": round(float(np.mean(sharpes)), 6),
            "std_sharpe": round(float(np.std(sharpes)), 6),
            "mean_max_drawdown": round(float(np.mean(drawdowns)), 6),
            "worst_max_drawdown": round(float(np.min(drawdowns)), 6),
            "mean_total_return": round(float(np.mean(returns)), 6),
            "mean_accuracy": round(float(np.mean(accuracies)), 6),
            "mean_f1": round(float(np.mean(f1s)), 6),
        }
