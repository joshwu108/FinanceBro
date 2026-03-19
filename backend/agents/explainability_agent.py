"""ExplainabilityAgent — Interpret model predictions via SHAP.

Computes SHAP-based feature importance (global and per-prediction),
tracks importance drift across walk-forward folds, and flags
spurious or overly concentrated features.

Constraints (from explainability_spec.md):
  - Compute SHAP only on out-of-sample predictions
  - Must evaluate importance stability across at least 3 time folds
  - Flag any feature that ranks top-5 in one fold but bottom-50% in another
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance as sklearn_perm_importance

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_MIN_FOLDS_FOR_DRIFT = 3
_CONCENTRATION_THRESHOLD = 0.5  # single feature > 50% of total importance
_CORRELATION_THRESHOLD = 0.85  # warn if |corr| exceeds this
_ADDITIVITY_RTOL = 0.05  # relative tolerance for SHAP additivity check


def _is_tree_model(model) -> bool:
    """Detect tree-based models that support shap.TreeExplainer.

    Covers sklearn ensembles, XGBoost, LightGBM, and CatBoost.
    """
    if hasattr(model, "estimators_"):
        return True
    model_type = type(model).__name__
    return model_type in {
        "XGBClassifier", "XGBRegressor", "XGBRanker", "XGBRFClassifier",
        "LGBMClassifier", "LGBMRegressor", "LGBMRanker",
        "CatBoostClassifier", "CatBoostRegressor",
    }


class ExplainabilityAgent(BaseAgent):
    """SHAP-based model explainability agent.

    Produces global feature importance, per-prediction SHAP values,
    importance drift across folds, and spurious feature detection.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "random_seed": None,
        "max_samples_for_shap": 500,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}
        self._experiments_dir = Path("experiments")

    # ── BaseAgent contract ────────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": "Trained sklearn-compatible model",
            "scaler": "(optional) StandardScaler used during training",
            "feature_names": "list[str] — feature column names",
            "feature_matrix": "pd.DataFrame — out-of-sample feature values",
            "y_test": "(optional) pd.Series — out-of-sample labels for permutation importance",
            "X_train": "(optional) pd.DataFrame — training features for KernelExplainer background",
            "fold_results": (
                "(optional) list[dict] — walk-forward fold results, each "
                "containing model, scaler, feature_names, X_test, y_test, "
                "and optionally X_train"
            ),
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "feature_importance": "dict[str, float] — mean |SHAP| per feature",
            "shap_values": "pd.DataFrame — per-prediction SHAP values",
            "explanation_summary": (
                "dict with top_features, model_type, n_samples, "
                "concentration_warning, correlated_feature_groups, "
                "additivity_check"
            ),
            "permutation_importance": (
                "(present when y_test provided) dict[str, float] — "
                "performance drop when each feature is shuffled"
            ),
            "importance_drift": (
                "(present when fold_results has >= 3 folds) dict with "
                "stability_scores, mean_rank_correlation, per_fold_importances, "
                "spurious_features"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_inputs(inputs)

        model = inputs["model"]
        scaler = inputs.get("scaler")
        feature_names = inputs["feature_names"]
        feature_matrix: pd.DataFrame = inputs["feature_matrix"]
        y_test: Optional[pd.Series] = inputs.get("y_test")
        X_train: Optional[pd.DataFrame] = inputs.get("X_train")
        fold_results: Optional[List[Dict]] = inputs.get("fold_results")

        shap_values_array, expected_value = self._compute_shap_values(
            model, scaler, feature_matrix, feature_names,
            X_train=X_train,
        )

        shap_df = pd.DataFrame(
            shap_values_array,
            index=feature_matrix.index,
            columns=feature_names,
        )

        feature_importance = {
            name: float(np.mean(np.abs(shap_values_array[:, i])))
            for i, name in enumerate(feature_names)
        }

        additivity_ok = self._check_additivity(
            model, scaler, feature_matrix, feature_names,
            shap_values_array, expected_value,
        )

        correlated_groups = self._detect_correlated_features(
            feature_matrix, feature_names,
        )

        explanation_summary = self._build_summary(
            feature_importance, model, len(feature_matrix),
            correlated_groups=correlated_groups,
            additivity_ok=additivity_ok,
        )

        result: Dict[str, Any] = {
            "feature_importance": feature_importance,
            "shap_values": shap_df,
            "explanation_summary": explanation_summary,
        }

        if y_test is not None:
            result["permutation_importance"] = self._compute_permutation_importance(
                model, scaler, feature_matrix, y_test, feature_names,
            )

        if fold_results is not None and len(fold_results) >= _MIN_FOLDS_FOR_DRIFT:
            result["importance_drift"] = self._compute_drift(
                fold_results, feature_names,
            )

        self._metrics = {
            "feature_importance": feature_importance,
            "explanation_summary": explanation_summary,
            "importance_drift": result.get("importance_drift"),
            "permutation_importance": result.get("permutation_importance"),
        }

        return result

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        if "feature_importance" not in outputs:
            raise ValueError("Output missing 'feature_importance'")
        if "explanation_summary" not in outputs:
            raise ValueError("Output missing 'explanation_summary'")
        if "shap_values" not in outputs:
            raise ValueError("Output missing 'shap_values'")

        fi = outputs["feature_importance"]
        if not isinstance(fi, dict) or len(fi) == 0:
            raise ValueError("feature_importance must be a non-empty dict")

        for name, val in fi.items():
            if not np.isfinite(val):
                raise ValueError(f"Non-finite importance for {name}: {val}")

        summary = outputs["explanation_summary"]
        if not summary.get("additivity_check", {}).get("passed", True):
            logger.warning(
                "SHAP additivity check FAILED — values may be unreliable. "
                "max_error=%.4f",
                summary["additivity_check"].get("max_relative_error", float("nan")),
            )

        if summary.get("correlated_feature_groups"):
            groups = summary["correlated_feature_groups"]
            logger.warning(
                "Detected %d correlated feature group(s). SHAP distributes "
                "credit among correlated features — individual importances "
                "understate their collective contribution: %s",
                len(groups), groups,
            )

        return True

    def log_metrics(self) -> None:
        self._experiments_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        filename = f"explainability_{run_id}.json"
        log_data = {
            "agent": "ExplainabilityAgent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_importance": self._metrics.get("feature_importance", {}),
            "explanation_summary": self._metrics.get("explanation_summary", {}),
            "importance_drift": self._metrics.get("importance_drift"),
            "permutation_importance": self._metrics.get("permutation_importance"),
        }
        filepath = self._experiments_dir / filename
        filepath.write_text(json.dumps(log_data, indent=2, default=str))
        logger.info("Logged explainability metrics to %s", filepath)

    # ── Internal helpers ──────────────────────────────────────────

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        if "model" not in inputs:
            raise ValueError("Missing required input: 'model'")
        if "feature_matrix" not in inputs:
            raise ValueError("Missing required input: 'feature_matrix'")
        fm = inputs["feature_matrix"]
        if not isinstance(fm, pd.DataFrame) or fm.empty:
            raise ValueError("feature_matrix must be a non-empty DataFrame")

    def _compute_shap_values(
        self,
        model,
        scaler: Optional[Any],
        feature_matrix: pd.DataFrame,
        feature_names: List[str],
        X_train: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute SHAP values and expected value.

        Returns (shap_array, expected_value) where shap_array has shape
        (n_samples, n_features).
        """
        max_samples = self._config["max_samples_for_shap"]
        X = feature_matrix.reindex(columns=feature_names).copy()

        if scaler is not None:
            X_scaled = pd.DataFrame(
                scaler.transform(X), index=X.index, columns=feature_names
            )
        else:
            X_scaled = X

        n_classes = _get_n_classes(model)
        if n_classes is not None and n_classes > 2:
            raise ValueError(
                f"Model has {n_classes} classes. ExplainabilityAgent currently "
                "supports binary classification only. Multiclass requires "
                "per-class SHAP analysis."
            )

        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X_scaled)
        else:
            background = self._prepare_background(
                X_train, scaler, feature_names, X_scaled, max_samples
            )
            background_summary = shap.kmeans(
                background, min(10, len(background))
            )
            explainer = shap.KernelExplainer(
                model.predict_proba, background_summary
            )
            raw_shap = explainer.shap_values(X_scaled)

        shap_array = _extract_positive_class_shap(raw_shap)
        expected_value = _extract_expected_value(explainer.expected_value)

        return shap_array, expected_value

    def _prepare_background(
        self,
        X_train: Optional[pd.DataFrame],
        scaler: Optional[Any],
        feature_names: List[str],
        X_scaled_fallback: pd.DataFrame,
        max_samples: int,
    ) -> pd.DataFrame:
        """Build background dataset for KernelExplainer.

        Prefers training data; falls back to OOS data with a warning.
        """
        seed = self._config.get("random_seed")
        rng = np.random.RandomState(seed)

        if X_train is not None:
            bg = X_train.reindex(columns=feature_names).copy()
            if scaler is not None:
                bg = pd.DataFrame(
                    scaler.transform(bg), index=bg.index, columns=feature_names
                )
        else:
            logger.warning(
                "No X_train provided for KernelExplainer background — "
                "falling back to OOS data. SHAP baseline will shift with "
                "each evaluation set, making values less comparable."
            )
            bg = X_scaled_fallback

        if len(bg) > max_samples:
            idx = rng.choice(len(bg), max_samples, replace=False)
            bg = bg.iloc[sorted(idx)]

        return bg

    def _build_summary(
        self,
        feature_importance: Dict[str, float],
        model,
        n_samples: int,
        *,
        correlated_groups: List[List[str]],
        additivity_ok: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build explanation summary dict."""
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        top_features = [
            {"feature": name, "importance": imp} for name, imp in sorted_features
        ]

        total_importance = sum(feature_importance.values())
        max_importance = max(feature_importance.values()) if feature_importance else 0.0

        concentration_warning = False
        if total_importance > 0 and (max_importance / total_importance) > _CONCENTRATION_THRESHOLD:
            concentration_warning = True

        model_type = type(model).__name__

        return {
            "top_features": top_features,
            "model_type": model_type,
            "n_samples": n_samples,
            "concentration_warning": concentration_warning,
            "correlated_feature_groups": correlated_groups,
            "additivity_check": additivity_ok,
        }

    def _compute_drift(
        self,
        fold_results: List[Dict[str, Any]],
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Compute feature importance drift across walk-forward folds."""
        per_fold_importances: List[Dict[str, float]] = []

        common_features = set(feature_names)
        for fold in fold_results:
            fold_features = fold.get("feature_names", feature_names)
            common_features &= set(fold_features)
        common_features_list = sorted(common_features)

        if len(common_features_list) < 2:
            logger.warning(
                "Fewer than 2 common features across folds — "
                "drift analysis requires consistent feature sets."
            )
            return {
                "stability_scores": {},
                "mean_rank_correlation": 0.0,
                "per_fold_importances": [],
                "spurious_features": [],
            }

        for fold in fold_results:
            fold_model = fold["model"]
            fold_scaler = fold.get("scaler")
            fold_features = fold.get("feature_names", feature_names)
            fold_X = fold["X_test"]
            fold_X_train = fold.get("X_train")

            shap_array, _ = self._compute_shap_values(
                fold_model, fold_scaler, fold_X, fold_features,
                X_train=fold_X_train,
            )

            fold_importance = {
                name: float(np.mean(np.abs(shap_array[:, i])))
                for i, name in enumerate(fold_features)
            }
            per_fold_importances.append(fold_importance)

        n_folds = len(per_fold_importances)
        rank_correlations = []
        for i, j in combinations(range(n_folds), 2):
            ranks_i = _importance_to_ranks(per_fold_importances[i], common_features_list)
            ranks_j = _importance_to_ranks(per_fold_importances[j], common_features_list)
            corr, _ = spearmanr(ranks_i, ranks_j)
            rank_correlations.append(corr)

        mean_rank_corr = float(np.mean(rank_correlations)) if rank_correlations else 0.0

        n = len(common_features_list)
        stability_scores = {}
        for fname in common_features_list:
            per_fold_ranks = [
                _get_feature_rank(fi, fname) for fi in per_fold_importances
            ]
            rank_std = float(np.std(per_fold_ranks))
            max_possible_std = (n - 1) / 2.0 if n > 1 else 1.0
            stability = max(0.0, 1.0 - rank_std / max_possible_std)
            stability_scores[fname] = round(stability, 4)

        spurious_features = _detect_spurious(per_fold_importances, common_features_list)

        return {
            "stability_scores": stability_scores,
            "mean_rank_correlation": round(mean_rank_corr, 4),
            "per_fold_importances": per_fold_importances,
            "spurious_features": spurious_features,
        }


    # ── New methods: additivity, correlation, permutation ──────────

    def _check_additivity(
        self,
        model,
        scaler: Optional[Any],
        feature_matrix: pd.DataFrame,
        feature_names: List[str],
        shap_values_array: np.ndarray,
        expected_value: float,
    ) -> Dict[str, Any]:
        """Verify SHAP additivity: sum(shap) + E[f(x)] ≈ f(x).

        Returns a dict with 'passed', 'max_relative_error',
        'mean_absolute_error'.
        """
        X = feature_matrix.reindex(columns=feature_names).copy()
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values

        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, "predict"):
            preds = model.predict(X_scaled)
        else:
            return {"passed": True, "skipped": True}

        shap_sum = shap_values_array.sum(axis=1) + expected_value
        abs_errors = np.abs(shap_sum - preds)
        denom = np.maximum(np.abs(preds), 1e-10)
        rel_errors = abs_errors / denom

        max_rel = float(np.max(rel_errors))
        mean_abs = float(np.mean(abs_errors))
        passed = max_rel <= _ADDITIVITY_RTOL

        if not passed:
            logger.warning(
                "SHAP additivity check FAILED: max_relative_error=%.4f "
                "(threshold=%.4f). Feature attributions may be unreliable.",
                max_rel, _ADDITIVITY_RTOL,
            )

        return {
            "passed": passed,
            "max_relative_error": round(max_rel, 6),
            "mean_absolute_error": round(mean_abs, 6),
        }

    @staticmethod
    def _detect_correlated_features(
        feature_matrix: pd.DataFrame,
        feature_names: List[str],
    ) -> List[List[str]]:
        """Find groups of features with |correlation| > threshold.

        SHAP distributes credit among correlated features, making
        individual importances misleading for correlated groups.
        """
        X = feature_matrix.reindex(columns=feature_names)
        corr_matrix = X.corr().abs()

        visited = set()
        groups: List[List[str]] = []

        for i, fname_i in enumerate(feature_names):
            if fname_i in visited:
                continue
            group = [fname_i]
            for j in range(i + 1, len(feature_names)):
                fname_j = feature_names[j]
                if fname_j in visited:
                    continue
                if corr_matrix.loc[fname_i, fname_j] > _CORRELATION_THRESHOLD:
                    group.append(fname_j)
            if len(group) > 1:
                visited.update(group)
                groups.append(sorted(group))

        return groups

    def _compute_permutation_importance(
        self,
        model,
        scaler: Optional[Any],
        feature_matrix: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Permutation importance on out-of-sample data.

        Measures actual predictive impact by shuffling each feature and
        observing the performance drop — a direct cross-check on SHAP.
        """
        X = feature_matrix.reindex(columns=feature_names).copy()
        if scaler is not None:
            X_vals = scaler.transform(X)
        else:
            X_vals = X.values

        seed = self._config.get("random_seed", 42)

        scoring = "neg_log_loss" if hasattr(model, "predict_proba") else "accuracy"
        pi_result = sklearn_perm_importance(
            model, X_vals, y_test,
            n_repeats=10,
            random_state=seed,
            scoring=scoring,
        )

        importance = {}
        for i, fname in enumerate(feature_names):
            importance[fname] = float(pi_result.importances_mean[i])

        return importance


# ── Module-level helpers ──────────────────────────────────────────


def _importance_to_ranks(
    importance: Dict[str, float], feature_names: List[str]
) -> List[float]:
    """Convert importance dict to rank list (1 = most important)."""
    sorted_names = sorted(feature_names, key=lambda n: importance.get(n, 0.0), reverse=True)
    rank_map = {name: rank + 1 for rank, name in enumerate(sorted_names)}
    return [rank_map[n] for n in feature_names]


def _get_feature_rank(importance: Dict[str, float], feature_name: str) -> int:
    """Get rank of a specific feature (1 = most important)."""
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, _) in enumerate(sorted_features, 1):
        if name == feature_name:
            return rank
    return len(sorted_features)


def _detect_spurious(
    per_fold_importances: List[Dict[str, float]],
    feature_names: List[str],
) -> List[str]:
    """Flag features that rank top-5 in one fold but bottom-50% in another."""
    n_features = len(feature_names)
    bottom_half_cutoff = n_features // 2
    spurious = set()

    for fname in feature_names:
        ranks = [_get_feature_rank(fi, fname) for fi in per_fold_importances]
        has_top5 = any(r <= min(5, n_features) for r in ranks)
        has_bottom_half = any(r > bottom_half_cutoff for r in ranks)
        if has_top5 and has_bottom_half:
            spurious.add(fname)

    return sorted(spurious)


def _extract_positive_class_shap(raw_shap) -> np.ndarray:
    """Extract SHAP values for the positive class (binary classification).

    Handles both old-style list-of-arrays and new-style 3D array outputs.
    """
    if isinstance(raw_shap, list):
        if len(raw_shap) == 2:
            return np.array(raw_shap[1])
        if len(raw_shap) == 1:
            return np.array(raw_shap[0])
        raise ValueError(
            f"Expected binary (2-class) SHAP output, got {len(raw_shap)} "
            "classes. Multiclass is not supported."
        )
    arr = np.array(raw_shap)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[:, :, 1]
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        raise ValueError(
            f"Expected binary SHAP output, got shape {arr.shape}."
        )
    return arr


def _extract_expected_value(ev) -> float:
    """Extract scalar expected value for the positive class."""
    if isinstance(ev, (list, np.ndarray)):
        arr = np.array(ev).ravel()
        if len(arr) == 2:
            return float(arr[1])
        return float(arr[0])
    return float(ev)


def _get_n_classes(model) -> Optional[int]:
    """Return number of classes if the model exposes them, else None."""
    if hasattr(model, "classes_"):
        return len(model.classes_)
    if hasattr(model, "n_classes_"):
        return int(model.n_classes_)
    return None
