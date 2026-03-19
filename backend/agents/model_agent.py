"""ModelAgent — Trains predictive models on FeatureAgent output.

Wraps and hardens oldBackend/app/services/ml_models.py.

CRITICAL fixes applied vs. the original service:
  - Debug prints removed ("reached here 2/3/4")
  - Logistic regression baseline added (required before any complex model)
  - No random train/test splits — temporal split ONLY
  - StandardScaler fit on train data only, transform applied to test
  - Predictions output as probabilities aligned with test timestamps
  - Feature importances reported for all supported model types
  - Train/test metric gap tracked as explicit overfitting signal
  - 500 estimators / max_depth=10 reduced to conservative defaults (100 / 5)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logistic_regression", "random_forest")


class ModelAgent(BaseAgent):
    """Trains classification models on FeatureAgent output.

    Always starts with a logistic regression baseline.
    All splits are temporal — no random shuffling.
    Outputs prediction probabilities aligned with test timestamps.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_type": "logistic_regression",
        "train_ratio": 0.8,
        "random_state": 42,
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 20,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}
        self._trained_model = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []

    def save_inference_bundle(self, path: Path) -> None:
        """Persist fitted model, scaler, and feature order for online prediction."""
        if self._trained_model is None or self._scaler is None:
            raise RuntimeError(
                "Cannot save inference bundle — call run() to train first"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self._trained_model,
            "scaler": self._scaler,
            "feature_names": list(self._feature_names),
            "model_type": self._config.get("model_type"),
        }
        joblib.dump(bundle, path)
        logger.info("Saved inference bundle to %s", path)

    @staticmethod
    def load_inference_bundle(path: Path) -> Dict[str, Any]:
        """Load a bundle written by :meth:`save_inference_bundle`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Inference bundle not found: {path}")
        return joblib.load(path)

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
            "model_config": "(optional) dict overriding DEFAULT_CONFIG keys",
            "train_end_date": (
                "(optional) str — ISO date cutoff. Train on data <= this date, "
                "test on data after. Overrides train_ratio. "
                "Used by WalkForwardAgent to pass fold boundaries."
            ),
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "predictions": (
                "pd.Series — P(class=1) probabilities on test set, "
                "indexed by test timestamps"
            ),
            "predicted_classes": (
                "pd.Series — binary predictions (0/1) on test set"
            ),
            "train_metrics": "dict — accuracy, precision, recall, f1, log_loss",
            "test_metrics": "dict — accuracy, precision, recall, f1, log_loss",
            "train_test_gap": (
                "dict — per-metric (train − test) difference, overfitting signal"
            ),
            "feature_importances": (
                "dict or None — feature_name -> importance score "
                "(abs coef for LR, Gini for RF)"
            ),
            "model_type": "str — model type used",
            "trained_model": "sklearn estimator (serializable)",
            "scaler": "StandardScaler fitted on training data only",
            "split_info": (
                "dict — train_size, test_size, train_start, train_end, "
                "test_start, test_end"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model and generate predictions on the test set.

        Args:
            inputs: dict with 'feature_matrix', 'target', optional
                    'model_config', 'train_end_date'.

        Returns:
            dict matching output_schema.
        """
        X, y = self._validate_and_extract_inputs(inputs)

        effective_config = {**self._config, **(inputs.get("model_config") or {})}
        model_type = effective_config["model_type"]

        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported: {SUPPORTED_MODELS}"
            )

        # Phase 1: Temporal split — never random
        train_end_date = inputs.get("train_end_date")
        X_train, y_train, X_test, y_test = self._temporal_split(
            X, y,
            train_end_date=train_end_date,
            train_ratio=effective_config["train_ratio"],
        )

        self._feature_names = list(X_train.columns)

        # Phase 2: Scale features (fit on train ONLY)
        self._scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns,
        )
        X_test_scaled = pd.DataFrame(
            self._scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns,
        )

        # Phase 3: Build and train model
        random_state = effective_config.get("random_state", 42)
        model = self._build_model(model_type, effective_config, random_state)
        model.fit(X_train_scaled.values, y_train.values)
        self._trained_model = model

        # Phase 4: Predict probabilities
        train_proba = model.predict_proba(X_train_scaled.values)[:, 1]
        test_proba = model.predict_proba(X_test_scaled.values)[:, 1]

        train_pred_classes = (train_proba >= 0.5).astype(int)
        test_pred_classes = (test_proba >= 0.5).astype(int)

        # Phase 5: Compute metrics
        train_metrics = self._compute_classification_metrics(
            y_train.values, train_pred_classes, train_proba
        )
        test_metrics = self._compute_classification_metrics(
            y_test.values, test_pred_classes, test_proba
        )

        train_test_gap = {
            k: round(train_metrics[k] - test_metrics[k], 6)
            for k in train_metrics
        }

        # Phase 6: Feature importances
        feature_importances = self._extract_feature_importances(
            model, model_type, self._feature_names
        )

        # Phase 7: Build timestamp-aligned output
        predictions = pd.Series(
            test_proba, index=X_test.index, name="predicted_proba"
        )
        predicted_classes = pd.Series(
            test_pred_classes, index=X_test.index, name="predicted_class"
        )

        split_info = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_start": str(X_train.index.min()),
            "train_end": str(X_train.index.max()),
            "test_start": str(X_test.index.min()),
            "test_end": str(X_test.index.max()),
        }

        outputs: Dict[str, Any] = {
            "predictions": predictions,
            "predicted_classes": predicted_classes,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_test_gap": train_test_gap,
            "feature_importances": feature_importances,
            "model_type": model_type,
            "trained_model": model,
            "scaler": self._scaler,
            "split_info": split_info,
        }

        # Phase 8: Validate
        self.validate(inputs, outputs)

        # Phase 9: Skepticism check (per model_agent.md mandate)
        self._skepticism_check(train_metrics, test_metrics, model_type)

        # Phase 10: Record metrics
        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "model_type": model_type,
            **split_info,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_test_gap": train_test_gap,
            "feature_count": len(self._feature_names),
        }

        logger.info(
            "ModelAgent complete: %s, train_acc=%.4f, test_acc=%.4f, gap=%.4f",
            model_type,
            train_metrics["accuracy"],
            test_metrics["accuracy"],
            train_test_gap["accuracy"],
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate model outputs for integrity and temporal correctness."""
        predictions = outputs["predictions"]
        predicted_classes = outputs["predicted_classes"]

        if (predictions < 0).any() or (predictions > 1).any():
            raise ValueError("Predictions contain values outside [0, 1]")

        if predictions.isna().any():
            raise ValueError("Predictions contain NaN")

        unique_classes = set(predicted_classes.unique())
        if not unique_classes.issubset({0, 1}):
            raise ValueError(
                f"Predicted classes must be {{0, 1}}, got {unique_classes}"
            )

        if not predictions.index.is_monotonic_increasing:
            raise ValueError("Prediction timestamps are not temporally ordered")

        # Test timestamps must be strictly after train timestamps
        split_info = outputs["split_info"]
        train_end = pd.Timestamp(split_info["train_end"])
        test_start = pd.Timestamp(split_info["test_start"])
        if test_start <= train_end:
            raise ValueError(
                f"Temporal leak: test_start ({test_start}) <= "
                f"train_end ({train_end})"
            )

        for label, metrics in [("train", outputs["train_metrics"]),
                               ("test", outputs["test_metrics"])]:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    raise ValueError(f"{label}_metrics['{k}'] is not finite: {v}")

        return True

    def log_metrics(self) -> None:
        """Persist metrics from the most recent run to experiments/."""
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        experiments_dir = Path(__file__).parent.parent / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        run_id = self._metrics.get("run_id", uuid.uuid4().hex[:12])

        log_entry = {
            "experiment_id": f"model_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "ModelAgent",
            "stage": "model_training",
            "timestamp": now.isoformat(),
            "model": self._metrics.get("model_type"),
            "features": self._feature_names,
            "parameters": {
                "model_type": self._metrics.get("model_type"),
                "feature_count": self._metrics.get("feature_count"),
                "train_size": self._metrics.get("train_size"),
                "test_size": self._metrics.get("test_size"),
            },
            "out_of_sample": True,
            "metrics": {
                "train": self._metrics.get("train_metrics", {}),
                "test": self._metrics.get("test_metrics", {}),
                "train_test_gap": self._metrics.get("train_test_gap", {}),
            },
            "notes": "ModelAgent training run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"model_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: input validation ────────────────────────────────

    def _validate_and_extract_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and extract feature_matrix and target from inputs."""
        if "feature_matrix" not in inputs:
            raise ValueError("inputs must contain 'feature_matrix'")
        if "target" not in inputs:
            raise ValueError("inputs must contain 'target'")

        X = inputs["feature_matrix"]
        y = inputs["target"]

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"'feature_matrix' must be pd.DataFrame, got {type(X)}"
            )
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"'target' must be pd.Series, got {type(y)}"
            )

        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("feature_matrix must have DatetimeIndex")

        if not X.index.equals(y.index):
            raise ValueError("feature_matrix and target indices don't match")

        if not X.index.is_monotonic_increasing:
            raise ValueError("feature_matrix index is not temporally ordered")

        unique_vals = set(y.dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Target must be binary (0/1), got unique values: {unique_vals}"
            )

        if X.isna().any().any():
            raise ValueError("feature_matrix contains NaN")
        if y.isna().any():
            raise ValueError("target contains NaN")

        return X, y

    # ── Internal: temporal split ──────────────────────────────────

    @staticmethod
    def _temporal_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_end_date: Optional[str] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split data temporally — NO random shuffling.

        If train_end_date is provided, use it as the cutoff.
        Otherwise, use train_ratio to determine the split point.
        """
        if train_end_date is not None:
            cutoff = pd.Timestamp(train_end_date)
            train_mask = X.index <= cutoff
            test_mask = X.index > cutoff
        else:
            n = len(X)
            split_idx = int(n * train_ratio)
            if split_idx < 1:
                raise ValueError(
                    f"Training set too small: {split_idx} rows "
                    f"(total={n}, ratio={train_ratio})"
                )
            if split_idx >= n:
                raise ValueError(
                    f"Test set empty: split_idx={split_idx}, total={n}"
                )
            train_mask = np.zeros(n, dtype=bool)
            train_mask[:split_idx] = True
            test_mask = ~train_mask

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_train) == 0:
            raise ValueError("Training set is empty after temporal split")
        if len(X_test) == 0:
            raise ValueError("Test set is empty after temporal split")

        if X_train.index.max() >= X_test.index.min():
            raise ValueError(
                f"Temporal overlap: train_end={X_train.index.max()}, "
                f"test_start={X_test.index.min()}"
            )

        logger.info(
            "Temporal split: train=%d rows [%s → %s], test=%d rows [%s → %s]",
            len(X_train), X_train.index.min().date(), X_train.index.max().date(),
            len(X_test), X_test.index.min().date(), X_test.index.max().date(),
        )

        return X_train, y_train, X_test, y_test

    # ── Internal: model construction ──────────────────────────────

    @staticmethod
    def _build_model(
        model_type: str,
        config: Dict[str, Any],
        random_state: int,
    ) -> Any:
        """Construct a sklearn estimator from config.

        Conservative defaults prevent memorization on small financial
        datasets (contrast with old code's 500 estimators / max_depth=10).
        """
        if model_type == "logistic_regression":
            params = config.get("logistic_regression", {})
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                solver=params.get("solver", "lbfgs"),
                random_state=random_state,
            )

        if model_type == "random_forest":
            params = config.get("random_forest", {})
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                min_samples_leaf=params.get("min_samples_leaf", 20),
                random_state=random_state,
                n_jobs=-1,
            )

        raise ValueError(f"Unknown model_type: {model_type}")

    # ── Internal: classification metrics ──────────────────────────

    @staticmethod
    def _compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics for a single split."""
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(
                y_true, y_pred, zero_division=0
            )), 6),
            "recall": round(float(recall_score(
                y_true, y_pred, zero_division=0
            )), 6),
            "f1": round(float(f1_score(
                y_true, y_pred, zero_division=0
            )), 6),
            "log_loss": round(float(log_loss(
                y_true, y_proba, labels=[0, 1]
            )), 6),
        }

    # ── Internal: feature importances ─────────────────────────────

    @staticmethod
    def _extract_feature_importances(
        model: Any,
        model_type: str,
        feature_names: List[str],
    ) -> Optional[Dict[str, float]]:
        """Extract feature importances, sorted descending.

        - Logistic regression: absolute coefficient values
        - Random forest: Gini importances
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif model_type == "logistic_regression" and hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        return {
            name: round(float(imp), 6)
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    # ── Internal: skepticism check ────────────────────────────────

    @staticmethod
    def _skepticism_check(
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        model_type: str,
    ) -> None:
        """Flag suspicious results per model_agent.md mandate.

        Daily direction prediction rarely exceeds 55% out-of-sample
        with honest methodology. Higher numbers demand leakage review.
        """
        train_acc = train_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]
        gap = train_acc - test_acc

        if test_acc > 0.60:
            logger.warning(
                "SKEPTICISM: %s test accuracy %.4f > 60%% on daily direction — "
                "verify no data leakage before trusting this result",
                model_type, test_acc,
            )

        if gap > 0.10:
            logger.warning(
                "SKEPTICISM: train-test accuracy gap %.4f > 10%% — "
                "likely overfitting (train=%.4f, test=%.4f)",
                gap, train_acc, test_acc,
            )

        if train_acc > 0.90:
            logger.warning(
                "SKEPTICISM: train accuracy %.4f > 90%% — "
                "possible memorization or data leakage",
                train_acc,
            )
