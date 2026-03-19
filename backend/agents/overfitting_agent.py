"""OverfittingAgent — Detects overfitting and quantifies model robustness.

Implements the spec at specs/overfitting_spec.md and the persona at
subagents/overfitting_agent.md.

Philosophy: Assume the model is wrong until proven otherwise.

Methods:
  1. Train vs test gap analysis — large accuracy/f1/log_loss gaps signal memorization
  2. Fold stability analysis — high variance in Sharpe/accuracy across folds
  3. Suspiciously good detection — results too good for daily direction prediction
  4. Degradation pattern — declining performance in later folds
  5. Per-fold train-test gap — when fold-level train metrics are available

Outputs:
  - overfitting_score (0–1): weighted combination of sub-scores
  - warnings: list of human-readable warning strings
  - diagnostics: detailed numerical breakdown
  - failure_modes: list of detected failure mode labels
  - recommendations: actionable suggestions
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OverfittingAgent(BaseAgent):
    """Skeptical quant reviewer that flags overfitting, memorization,
    and unreliable models.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "accuracy_gap_threshold": 0.10,
        "f1_gap_threshold": 0.15,
        "log_loss_gap_threshold": 0.50,
        "memorization_threshold": 0.90,
        "suspicious_accuracy_threshold": 0.60,
        "suspicious_sharpe_threshold": 2.0,
        "sharpe_std_threshold": 0.50,
        "negative_sharpe_tolerance": 0,
        "degradation_correlation_threshold": -0.80,
        "weights": {
            "gap": 0.35,
            "fold_instability": 0.25,
            "suspicious": 0.20,
            "degradation": 0.10,
            "per_fold_gap": 0.10,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "train_metrics": (
                "dict — accuracy, precision, recall, f1, log_loss from training set"
            ),
            "test_metrics": (
                "dict — accuracy, precision, recall, f1, log_loss from test set"
            ),
            "fold_results": (
                "list[dict] — per-fold results from WalkForwardAgent, each with "
                "backtest_metrics (sharpe, max_drawdown, total_return) and "
                "model_metrics (accuracy, f1)"
            ),
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "overfitting_score": "float 0–1 (0 = no overfitting, 1 = severe)",
            "warnings": "list[str] — human-readable warnings",
            "diagnostics": (
                "dict — train_test_gap, fold_sharpes, fold_accuracies, "
                "per_fold_train_test_gaps, sub_scores"
            ),
            "failure_modes": "list[str] — detected failure mode labels",
            "recommendations": "list[str] — actionable suggestions",
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze inputs for overfitting signals.

        Returns overfitting_score, warnings, diagnostics, failure_modes,
        and recommendations.
        """
        train_metrics, test_metrics, fold_results, cfg = self._validate_inputs(
            inputs
        )

        warnings: List[str] = []
        failure_modes: List[str] = []
        recommendations: List[str] = []

        # Sub-analysis 1: Train vs test gap
        gap_score, gap_diag = self._analyze_train_test_gap(
            train_metrics, test_metrics, cfg, warnings, failure_modes, recommendations
        )

        # Sub-analysis 2: Fold stability
        instability_score, fold_diag = self._analyze_fold_stability(
            fold_results, cfg, warnings, failure_modes, recommendations
        )

        # Sub-analysis 3: Suspiciously good results
        suspicious_score, sus_diag = self._analyze_suspicious_results(
            test_metrics, fold_results, cfg, warnings, failure_modes, recommendations
        )

        # Sub-analysis 4: Degradation pattern
        degradation_score, deg_diag = self._analyze_degradation(
            fold_results, cfg, warnings, failure_modes, recommendations
        )

        # Sub-analysis 5: Per-fold train-test gap
        per_fold_gap_score, pf_diag = self._analyze_per_fold_gaps(
            fold_results, cfg, warnings, failure_modes, recommendations
        )

        # Weighted composite score
        weights = cfg.get("weights", self.DEFAULT_CONFIG["weights"])
        overfitting_score = float(np.clip(
            weights["gap"] * gap_score
            + weights["fold_instability"] * instability_score
            + weights["suspicious"] * suspicious_score
            + weights["degradation"] * degradation_score
            + weights["per_fold_gap"] * per_fold_gap_score,
            0.0,
            1.0,
        ))

        diagnostics = {
            "train_test_gap": gap_diag,
            **fold_diag,
            **sus_diag,
            **deg_diag,
            **pf_diag,
            "sub_scores": {
                "gap": round(gap_score, 4),
                "fold_instability": round(instability_score, 4),
                "suspicious": round(suspicious_score, 4),
                "degradation": round(degradation_score, 4),
                "per_fold_gap": round(per_fold_gap_score, 4),
            },
        }

        outputs: Dict[str, Any] = {
            "overfitting_score": round(overfitting_score, 4),
            "warnings": warnings,
            "diagnostics": diagnostics,
            "failure_modes": failure_modes,
            "recommendations": recommendations,
        }

        self.validate(inputs, outputs)

        self._metrics = {
            "run_id": uuid.uuid4().hex[:12],
            "overfitting_score": outputs["overfitting_score"],
            "n_warnings": len(warnings),
            "n_failure_modes": len(failure_modes),
            "sub_scores": diagnostics["sub_scores"],
        }

        logger.info(
            "OverfittingAgent complete: score=%.4f, warnings=%d, failure_modes=%d",
            overfitting_score,
            len(warnings),
            len(failure_modes),
        )

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate output structure and value ranges."""
        required_keys = {"overfitting_score", "warnings", "diagnostics",
                         "failure_modes", "recommendations"}
        missing = required_keys - set(outputs.keys())
        if missing:
            raise ValueError(f"Output missing required keys: {missing}")

        score = outputs["overfitting_score"]
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"overfitting_score must be in [0, 1], got {score}"
            )

        if not isinstance(outputs["warnings"], list):
            raise ValueError("warnings must be a list")
        if not isinstance(outputs["failure_modes"], list):
            raise ValueError("failure_modes must be a list")
        if not isinstance(outputs["recommendations"], list):
            raise ValueError("recommendations must be a list")

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
            "experiment_id": f"overfitting_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "OverfittingAgent",
            "stage": "overfitting_detection",
            "timestamp": now.isoformat(),
            "out_of_sample": True,
            "parameters": {
                "accuracy_gap_threshold": self._config.get("accuracy_gap_threshold"),
                "f1_gap_threshold": self._config.get("f1_gap_threshold"),
            },
            "metrics": {
                "overfitting_score": self._metrics.get("overfitting_score"),
                "n_warnings": self._metrics.get("n_warnings"),
                "n_failure_modes": self._metrics.get("n_failure_modes"),
                "sub_scores": self._metrics.get("sub_scores"),
            },
            "notes": "OverfittingAgent detection run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"overfitting_agent_{ts}_{run_id}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    @property
    def _experiments_dir(self) -> Path:
        """Directory for experiment logs. Property for easy test patching."""
        return Path(__file__).parent.parent / "experiments"

    # ── Internal: input validation ────────────────────────────────

    def _validate_inputs(
        self, inputs: Dict[str, Any]
    ) -> tuple:
        """Validate and extract required inputs."""
        if not isinstance(inputs, dict):
            raise TypeError(f"inputs must be dict, got {type(inputs)}")

        if "train_metrics" not in inputs:
            raise ValueError("inputs must contain 'train_metrics'")
        if "test_metrics" not in inputs:
            raise ValueError("inputs must contain 'test_metrics'")
        if "fold_results" not in inputs:
            raise ValueError("inputs must contain 'fold_results'")

        train_metrics = inputs["train_metrics"]
        test_metrics = inputs["test_metrics"]
        fold_results = inputs["fold_results"]

        if not isinstance(train_metrics, dict):
            raise TypeError(
                f"'train_metrics' must be dict, got {type(train_metrics)}"
            )
        if not isinstance(test_metrics, dict):
            raise TypeError(
                f"'test_metrics' must be dict, got {type(test_metrics)}"
            )
        if not isinstance(fold_results, list):
            raise TypeError(
                f"'fold_results' must be list, got {type(fold_results)}"
            )
        if len(fold_results) == 0:
            raise ValueError("fold_results is empty — no folds to analyze")

        for i, fold in enumerate(fold_results):
            if not isinstance(fold, dict):
                raise TypeError(f"fold_results[{i}] must be dict, got {type(fold)}")
            if "backtest_metrics" not in fold:
                raise ValueError(f"fold_results[{i}] missing 'backtest_metrics'")
            if "model_metrics" not in fold:
                raise ValueError(f"fold_results[{i}] missing 'model_metrics'")

        cfg = {**self._config, **(inputs.get("config") or {})}

        return train_metrics, test_metrics, fold_results, cfg

    # ── Sub-analysis 1: Train vs test gap ─────────────────────────

    @staticmethod
    def _analyze_train_test_gap(
        train_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        cfg: Dict[str, Any],
        warnings: List[str],
        failure_modes: List[str],
        recommendations: List[str],
    ) -> tuple:
        """Compute train-test gap and flag large discrepancies.

        Returns (score 0-1, diagnostics dict).
        """
        gap = {}
        for key in ("accuracy", "f1", "log_loss", "precision", "recall"):
            train_val = train_metrics.get(key)
            test_val = test_metrics.get(key)
            if train_val is not None and test_val is not None:
                gap[key] = round(float(train_val - test_val), 6)

        score = 0.0
        components = 0

        # Accuracy gap
        acc_gap = gap.get("accuracy", 0.0)
        acc_threshold = cfg.get("accuracy_gap_threshold", 0.10)
        if acc_gap > acc_threshold:
            warnings.append(
                f"Large accuracy gap: train={train_metrics.get('accuracy', '?'):.4f} "
                f"vs test={test_metrics.get('accuracy', '?'):.4f} "
                f"(gap={acc_gap:.4f} > threshold={acc_threshold:.2f})"
            )
            failure_modes.append("train_test_accuracy_gap")
            recommendations.append(
                "Reduce model complexity or add regularization to close the "
                "accuracy gap between train and test"
            )
        score += min(abs(acc_gap) / max(acc_threshold, 1e-9), 1.0)
        components += 1

        # F1 gap
        f1_gap = gap.get("f1", 0.0)
        f1_threshold = cfg.get("f1_gap_threshold", 0.15)
        if f1_gap > f1_threshold:
            warnings.append(
                f"Large f1 gap: train={train_metrics.get('f1', '?'):.4f} "
                f"vs test={test_metrics.get('f1', '?'):.4f} "
                f"(gap={f1_gap:.4f} > threshold={f1_threshold:.2f})"
            )
            failure_modes.append("train_test_f1_gap")
            recommendations.append(
                "F1 degradation suggests class imbalance sensitivity — "
                "consider stratified validation or rebalancing"
            )
        score += min(abs(f1_gap) / max(f1_threshold, 1e-9), 1.0)
        components += 1

        # Memorization
        train_acc = train_metrics.get("accuracy", 0.0)
        mem_threshold = cfg.get("memorization_threshold", 0.90)
        if train_acc > mem_threshold:
            warnings.append(
                f"Possible memorization: train accuracy {train_acc:.4f} > "
                f"{mem_threshold:.0%} — model may have memorized training data"
            )
            failure_modes.append("memorization")
            recommendations.append(
                "Train accuracy above 90% on financial data strongly suggests "
                "memorization — reduce model capacity or increase regularization"
            )
            score += 1.0
            components += 1

        avg_score = score / max(components, 1)
        return float(np.clip(avg_score, 0.0, 1.0)), gap

    # ── Sub-analysis 2: Fold stability ────────────────────────────

    @staticmethod
    def _analyze_fold_stability(
        fold_results: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        warnings: List[str],
        failure_modes: List[str],
        recommendations: List[str],
    ) -> tuple:
        """Assess variance of key metrics across folds.

        Returns (score 0-1, diagnostics dict).
        """
        sharpes = [
            f["backtest_metrics"]["sharpe"] for f in fold_results
        ]
        accuracies = [
            f["model_metrics"]["accuracy"] for f in fold_results
        ]

        sharpe_std = float(np.std(sharpes))
        sharpe_mean = float(np.mean(sharpes))
        acc_std = float(np.std(accuracies))

        sharpe_std_threshold = cfg.get("sharpe_std_threshold", 0.50)
        neg_tolerance = cfg.get("negative_sharpe_tolerance", 0)

        score = 0.0

        # Sharpe instability
        if sharpe_std > sharpe_std_threshold:
            warnings.append(
                f"Unstable Sharpe across folds: std={sharpe_std:.4f} "
                f"(threshold={sharpe_std_threshold:.2f}), "
                f"values={[round(s, 3) for s in sharpes]}"
            )
            failure_modes.append("fold_instability_sharpe")
            recommendations.append(
                "High Sharpe variance suggests model is sensitive to market "
                "regime — consider regime-conditional modeling"
            )
            score += min(sharpe_std / sharpe_std_threshold, 2.0) / 2.0

        # Negative Sharpe folds
        neg_count = sum(1 for s in sharpes if s < 0)
        if neg_count > neg_tolerance:
            warnings.append(
                f"{neg_count}/{len(sharpes)} folds have negative Sharpe — "
                f"model loses money in some periods"
            )
            failure_modes.append("negative_sharpe_folds")
            recommendations.append(
                "Negative Sharpe folds indicate the strategy is not robust "
                "across time periods — investigate which regimes fail"
            )
            score += neg_count / len(sharpes)

        # Normalize
        score = float(np.clip(score / 2.0, 0.0, 1.0))

        diag = {
            "fold_sharpes": [round(s, 6) for s in sharpes],
            "fold_accuracies": [round(a, 6) for a in accuracies],
            "sharpe_mean": round(sharpe_mean, 6),
            "sharpe_std": round(sharpe_std, 6),
            "accuracy_std": round(acc_std, 6),
            "negative_sharpe_folds": neg_count,
        }

        return score, diag

    # ── Sub-analysis 3: Suspiciously good ─────────────────────────

    @staticmethod
    def _analyze_suspicious_results(
        test_metrics: Dict[str, Any],
        fold_results: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        warnings: List[str],
        failure_modes: List[str],
        recommendations: List[str],
    ) -> tuple:
        """Flag results that are too good for daily direction prediction.

        Returns (score 0-1, diagnostics dict).
        """
        score = 0.0
        diag: Dict[str, Any] = {}

        # Test accuracy > threshold
        test_acc = test_metrics.get("accuracy", 0.0)
        acc_threshold = cfg.get("suspicious_accuracy_threshold", 0.60)
        if test_acc > acc_threshold:
            warnings.append(
                f"Suspiciously high test accuracy: {test_acc:.4f} > "
                f"{acc_threshold:.0%} — daily direction prediction rarely "
                f"exceeds this honestly. Check for data leakage."
            )
            failure_modes.append("suspicious_accuracy")
            recommendations.append(
                "Verify no look-ahead bias in features. Check that test data "
                "was not used during feature selection or hyperparameter tuning."
            )
            score += min((test_acc - acc_threshold) / (1.0 - acc_threshold), 1.0)

        # Mean Sharpe across folds
        sharpes = [f["backtest_metrics"]["sharpe"] for f in fold_results]
        mean_sharpe = float(np.mean(sharpes))
        sharpe_threshold = cfg.get("suspicious_sharpe_threshold", 2.0)

        diag["mean_fold_sharpe"] = round(mean_sharpe, 6)

        if mean_sharpe > sharpe_threshold:
            warnings.append(
                f"Suspiciously high mean Sharpe: {mean_sharpe:.4f} > "
                f"{sharpe_threshold:.1f} — too good for daily equity returns"
            )
            failure_modes.append("suspicious_sharpe")
            recommendations.append(
                "Sharpe > 2.0 on equity daily data is extremely rare with "
                "honest methodology. Audit the entire pipeline for data leakage."
            )
            score += min((mean_sharpe - sharpe_threshold) / sharpe_threshold, 1.0)

        score = float(np.clip(score / 2.0, 0.0, 1.0))
        return score, diag

    # ── Sub-analysis 4: Degradation pattern ───────────────────────

    @staticmethod
    def _analyze_degradation(
        fold_results: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        warnings: List[str],
        failure_modes: List[str],
        recommendations: List[str],
    ) -> tuple:
        """Detect declining performance across later folds.

        Uses Spearman-like correlation between fold index and Sharpe.
        Returns (score 0-1, diagnostics dict).
        """
        if len(fold_results) < 3:
            return 0.0, {"degradation_correlation": None}

        sharpes = np.array([
            f["backtest_metrics"]["sharpe"] for f in fold_results
        ])
        indices = np.arange(len(sharpes), dtype=float)

        # Pearson correlation between fold index and Sharpe
        if np.std(sharpes) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(indices, sharpes)[0, 1])

        corr_threshold = cfg.get("degradation_correlation_threshold", -0.80)
        score = 0.0

        if corr < corr_threshold:
            warnings.append(
                f"Declining performance across folds: "
                f"correlation(fold_index, sharpe)={corr:.4f} "
                f"(threshold={corr_threshold:.2f}). "
                f"Strategy may be decaying over time."
            )
            failure_modes.append("performance_degradation")
            recommendations.append(
                "Declining fold performance suggests the signal is decaying — "
                "consider adaptive retraining or regime detection"
            )
            score = min(abs(corr - corr_threshold) / abs(corr_threshold), 1.0)

        diag = {"degradation_correlation": round(corr, 6)}
        return float(np.clip(score, 0.0, 1.0)), diag

    # ── Sub-analysis 5: Per-fold train-test gap ───────────────────

    @staticmethod
    def _analyze_per_fold_gaps(
        fold_results: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        warnings: List[str],
        failure_modes: List[str],
        recommendations: List[str],
    ) -> tuple:
        """Analyze train-test accuracy gaps within individual folds.

        Only available when fold results include 'train_metrics'.
        Returns (score 0-1, diagnostics dict).
        """
        per_fold_gaps: List[float] = []

        for fold in fold_results:
            train_m = fold.get("train_metrics")
            test_m = fold.get("model_metrics")
            if train_m and test_m:
                train_acc = train_m.get("accuracy", 0.0)
                test_acc = test_m.get("accuracy", 0.0)
                per_fold_gaps.append(round(float(train_acc - test_acc), 6))

        if not per_fold_gaps:
            return 0.0, {"per_fold_train_test_gaps": []}

        mean_gap = float(np.mean(per_fold_gaps))
        max_gap = float(np.max(per_fold_gaps))
        acc_threshold = cfg.get("accuracy_gap_threshold", 0.10)

        score = min(mean_gap / max(acc_threshold, 1e-9), 1.0)
        score = float(np.clip(score, 0.0, 1.0))

        if mean_gap > acc_threshold:
            failure_modes.append("per_fold_gap_large")
            recommendations.append(
                "Per-fold train-test gaps are consistently large — "
                "model complexity exceeds what the data can support"
            )

        diag = {
            "per_fold_train_test_gaps": per_fold_gaps,
            "mean_per_fold_gap": round(mean_gap, 6),
            "max_per_fold_gap": round(max_gap, 6),
        }
        return score, diag
