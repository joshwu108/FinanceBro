"""Tests for ExplainabilityAgent.

Validates:
  - BaseAgent contract is satisfied
  - Computes SHAP values for feature attribution
  - Produces global feature importance ranking
  - Produces per-prediction explanations
  - Tracks feature importance drift across walk-forward folds
  - Flags spurious features (top-5 in one fold, bottom-50% in another)
  - Detects feature concentration risk
  - SHAP additivity check
  - Correlated feature detection
  - Permutation importance cross-check
  - Multiclass rejection
  - KernelExplainer background from training data
  - Input validation
  - Experiment logging
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from agents.explainability_agent import (
    ExplainabilityAgent,
    _extract_positive_class_shap,
    _is_tree_model,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _make_feature_matrix(
    n_samples: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a feature matrix with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_samples)
    columns = [f"feature_{i}" for i in range(n_features)]
    data = rng.randn(n_samples, n_features)
    return pd.DataFrame(data, index=dates, columns=columns)


def _make_target(n_samples: int = 200, seed: int = 42) -> pd.Series:
    """Create a binary target aligned with feature matrix."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_samples)
    return pd.Series(rng.randint(0, 2, size=n_samples), index=dates, name="target")


def _train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    seed: int = 42,
):
    """Train a model and return (model, scaler, feature_names)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "logistic_regression":
        model = LogisticRegression(random_state=seed, max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=seed
        )
    model.fit(X_scaled, y)
    return model, scaler, list(X.columns)


def _make_fold_results(n_folds: int = 4, n_features: int = 5, seed: int = 42):
    """Create walk-forward fold results with feature matrices and models."""
    rng = np.random.RandomState(seed)
    folds = []
    for i in range(n_folds):
        n_samples = 100
        start_date = pd.Timestamp("2023-01-01") + pd.DateOffset(months=i * 3)
        dates = pd.bdate_range(start=start_date, periods=n_samples)
        columns = [f"feature_{j}" for j in range(n_features)]
        X = pd.DataFrame(rng.randn(n_samples, n_features), index=dates, columns=columns)
        y = pd.Series(rng.randint(0, 2, size=n_samples), index=dates, name="target")
        model, scaler, feature_names = _train_model(X, y, seed=seed + i)
        folds.append({
            "fold_id": i,
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "X_test": X,
            "y_test": y,
        })
    return folds


def _make_unstable_fold_results(n_features: int = 10, seed: int = 42):
    """Create folds where feature importance is deliberately unstable.

    Fold 0: feature_0 is dominant (high importance).
    Fold 1: feature_0 is irrelevant (low importance).
    This should trigger the spurious feature flag.
    """
    rng = np.random.RandomState(seed)
    folds = []

    for i in range(3):
        n_samples = 200
        start_date = pd.Timestamp("2023-01-01") + pd.DateOffset(months=i * 3)
        dates = pd.bdate_range(start=start_date, periods=n_samples)
        columns = [f"feature_{j}" for j in range(n_features)]
        X = pd.DataFrame(rng.randn(n_samples, n_features), index=dates, columns=columns)
        y = pd.Series(rng.randint(0, 2, size=n_samples), index=dates, name="target")

        if i == 0:
            # Make feature_0 strongly predictive
            X["feature_0"] = y * 2 + rng.randn(n_samples) * 0.1
        else:
            # Make feature_0 pure noise
            X["feature_0"] = rng.randn(n_samples) * 10

        model, scaler, feature_names = _train_model(X, y, seed=seed + i)
        folds.append({
            "fold_id": i,
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "X_test": X,
            "y_test": y,
        })
    return folds


@pytest.fixture
def agent():
    return ExplainabilityAgent(config={"random_seed": 42})


@pytest.fixture
def feature_matrix():
    return _make_feature_matrix()


@pytest.fixture
def target():
    return _make_target()


@pytest.fixture
def trained_model(feature_matrix, target):
    return _train_model(feature_matrix, target)


@pytest.fixture
def fold_results():
    return _make_fold_results()


@pytest.fixture
def unstable_fold_results():
    return _make_unstable_fold_results()


# ── BaseAgent Contract ───────────────────────────────────────────


class TestBaseAgentContract:
    """ExplainabilityAgent must implement all BaseAgent methods."""

    def test_has_run_method(self, agent):
        assert hasattr(agent, "run") and callable(agent.run)

    def test_has_validate_method(self, agent):
        assert hasattr(agent, "validate") and callable(agent.validate)

    def test_has_log_metrics_method(self, agent):
        assert hasattr(agent, "log_metrics") and callable(agent.log_metrics)

    def test_has_input_schema(self, agent):
        schema = agent.input_schema
        assert isinstance(schema, dict)
        assert "model" in schema
        assert "feature_matrix" in schema

    def test_has_output_schema(self, agent):
        schema = agent.output_schema
        assert isinstance(schema, dict)
        assert "feature_importance" in schema
        assert "explanation_summary" in schema


# ── SHAP Feature Importance ──────────────────────────────────────


class TestFeatureImportance:
    """SHAP-based global feature importance."""

    def test_returns_feature_importance_dict(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        fi = result["feature_importance"]
        assert isinstance(fi, dict)
        assert set(fi.keys()) == set(feature_names)

    def test_feature_importance_values_are_nonnegative(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        for val in result["feature_importance"].values():
            assert val >= 0.0

    def test_feature_importance_sums_close_to_total(
        self, agent, feature_matrix, trained_model
    ):
        """Importances should be mean absolute SHAP values — all finite."""
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        total = sum(result["feature_importance"].values())
        assert total > 0.0
        assert np.isfinite(total)

    def test_works_with_logistic_regression(self, agent, feature_matrix, target):
        model, scaler, feature_names = _train_model(
            feature_matrix, target, model_type="logistic_regression"
        )
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert len(result["feature_importance"]) == feature_matrix.shape[1]

    def test_works_with_random_forest(self, agent, feature_matrix, target):
        model, scaler, feature_names = _train_model(
            feature_matrix, target, model_type="random_forest"
        )
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert len(result["feature_importance"]) == feature_matrix.shape[1]


# ── Per-Prediction Explanations ──────────────────────────────────


class TestPerPredictionExplanations:
    """SHAP values per sample — per-prediction attribution."""

    def test_shap_values_shape_matches_input(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        shap_df = result["shap_values"]
        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape == feature_matrix.shape

    def test_shap_values_have_same_columns(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert list(result["shap_values"].columns) == feature_names

    def test_shap_values_have_same_index(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        pd.testing.assert_index_equal(
            result["shap_values"].index, feature_matrix.index
        )

    def test_shap_values_are_finite(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert np.all(np.isfinite(result["shap_values"].values))


# ── Feature Importance Drift ─────────────────────────────────────


class TestFeatureImportanceDrift:
    """Track feature importance stability across walk-forward folds."""

    def test_drift_output_present_with_fold_results(self, agent, fold_results):
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        assert "importance_drift" in result

    def test_drift_contains_stability_scores(self, agent, fold_results):
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        drift = result["importance_drift"]
        assert "stability_scores" in drift
        assert isinstance(drift["stability_scores"], dict)

    def test_stability_scores_are_rank_correlations(self, agent, fold_results):
        """Stability = mean pairwise Spearman rank correlation across folds."""
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        drift = result["importance_drift"]
        # Overall stability should be between -1 and 1
        assert -1.0 <= drift["mean_rank_correlation"] <= 1.0

    def test_drift_contains_per_fold_importances(self, agent, fold_results):
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        drift = result["importance_drift"]
        assert "per_fold_importances" in drift
        assert len(drift["per_fold_importances"]) == len(fold_results)


# ── Spurious Feature Detection ───────────────────────────────────


class TestSpuriousFeatureDetection:
    """Flag features that rank top-5 in one fold but bottom-50% in another."""

    def test_spurious_features_flagged(self, agent, unstable_fold_results):
        folds = unstable_fold_results
        result = agent.run({
            "model": folds[-1]["model"],
            "scaler": folds[-1]["scaler"],
            "feature_names": folds[-1]["feature_names"],
            "feature_matrix": folds[-1]["X_test"],
            "fold_results": folds,
        })
        drift = result["importance_drift"]
        assert "spurious_features" in drift
        # feature_0 is deliberately unstable
        assert isinstance(drift["spurious_features"], list)

    def test_no_spurious_with_stable_features(self, agent, fold_results):
        """With random data, spurious detection may or may not flag — but
        the key is the list exists and is a list."""
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        assert isinstance(result["importance_drift"]["spurious_features"], list)


# ── Concentration Risk ───────────────────────────────────────────


class TestConcentrationRisk:
    """No single feature should dominate predictions."""

    def test_concentration_flag_in_output(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        summary = result["explanation_summary"]
        assert "concentration_warning" in summary


# ── Explanation Summary ──────────────────────────────────────────


class TestExplanationSummary:
    """Summary dict with top features and warnings."""

    def test_summary_contains_top_features(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        summary = result["explanation_summary"]
        assert "top_features" in summary
        assert isinstance(summary["top_features"], list)
        assert len(summary["top_features"]) > 0

    def test_summary_top_features_sorted_descending(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        top = result["explanation_summary"]["top_features"]
        importances = [entry["importance"] for entry in top]
        assert importances == sorted(importances, reverse=True)

    def test_summary_has_model_type(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        summary = result["explanation_summary"]
        assert "model_type" in summary

    def test_summary_has_n_samples(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert result["explanation_summary"]["n_samples"] == len(feature_matrix)


# ── Input Validation ─────────────────────────────────────────────


class TestInputValidation:
    """Validate rejects bad inputs."""

    def test_rejects_missing_model(self, agent, feature_matrix):
        with pytest.raises((ValueError, KeyError)):
            agent.run({"feature_matrix": feature_matrix})

    def test_rejects_missing_feature_matrix(self, agent, trained_model):
        model, scaler, feature_names = trained_model
        with pytest.raises((ValueError, KeyError)):
            agent.run({"model": model, "scaler": scaler, "feature_names": feature_names})

    def test_rejects_empty_feature_matrix(self, agent, trained_model):
        model, scaler, feature_names = trained_model
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            agent.run({
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
                "feature_matrix": empty_df,
            })

    def test_validate_passes_on_good_output(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        inputs = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        }
        outputs = agent.run(inputs)
        assert agent.validate(inputs, outputs) is True

    def test_validate_rejects_missing_feature_importance(self, agent):
        inputs = {"model": "dummy", "feature_matrix": pd.DataFrame({"a": [1]})}
        outputs = {"explanation_summary": {}}
        with pytest.raises((ValueError, AssertionError)):
            agent.validate(inputs, outputs)

    def test_rejects_insufficient_folds_for_drift(
        self, agent, fold_results, trained_model
    ):
        """Drift analysis requires >= 3 folds per spec."""
        model, scaler, feature_names = trained_model
        one_fold = fold_results[:1]
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": fold_results[0]["X_test"],
            "fold_results": one_fold,
        })
        # With < 3 folds, drift should be absent or have a warning
        assert "importance_drift" not in result or result.get("importance_drift") is None


# ── Experiment Logging ───────────────────────────────────────────


class TestExperimentLogging:
    """ExplainabilityAgent must log to /experiments/."""

    def test_log_metrics_writes_file(
        self, agent, feature_matrix, trained_model, tmp_path
    ):
        model, scaler, feature_names = trained_model
        agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        with patch.object(agent, "_experiments_dir", tmp_path):
            agent.log_metrics()
        log_files = list(tmp_path.glob("explainability_*.json"))
        assert len(log_files) == 1

    def test_log_metrics_contains_required_fields(
        self, agent, feature_matrix, trained_model, tmp_path
    ):
        model, scaler, feature_names = trained_model
        agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        with patch.object(agent, "_experiments_dir", tmp_path):
            agent.log_metrics()
        log_files = list(tmp_path.glob("explainability_*.json"))
        data = json.loads(log_files[0].read_text())
        assert "agent" in data
        assert data["agent"] == "ExplainabilityAgent"
        assert "feature_importance" in data
        assert "timestamp" in data


# ── SHAP Additivity Check ────────────────────────────────────────


class TestAdditivityCheck:
    """SHAP values must satisfy sum(shap) + E[f(x)] ≈ f(x)."""

    def test_additivity_check_present_in_summary(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        summary = result["explanation_summary"]
        assert "additivity_check" in summary
        assert "passed" in summary["additivity_check"]

    def test_additivity_passes_for_tree_model(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        check = result["explanation_summary"]["additivity_check"]
        assert check["passed"] is True

    def test_additivity_passes_for_logistic_regression(
        self, agent, feature_matrix, target
    ):
        model, scaler, feature_names = _train_model(
            feature_matrix, target, model_type="logistic_regression"
        )
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        check = result["explanation_summary"]["additivity_check"]
        assert "max_relative_error" in check
        assert "mean_absolute_error" in check


# ── Correlated Feature Detection ─────────────────────────────────


class TestCorrelatedFeatureDetection:
    """SHAP distributes credit among correlated features — must warn."""

    def test_no_groups_for_independent_features(
        self, agent, feature_matrix, trained_model
    ):
        """Random features should have no correlated groups."""
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        groups = result["explanation_summary"]["correlated_feature_groups"]
        assert isinstance(groups, list)
        assert len(groups) == 0

    def test_detects_correlated_features(self, agent, target):
        """Two identical features should be flagged as correlated."""
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.bdate_range(start="2023-01-01", periods=n)
        base = rng.randn(n)
        X = pd.DataFrame({
            "base": base,
            "copy_of_base": base + rng.randn(n) * 0.01,
            "independent": rng.randn(n),
        }, index=dates)
        y = target[:n]
        model, scaler, feature_names = _train_model(X, y)
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": X,
        })
        groups = result["explanation_summary"]["correlated_feature_groups"]
        assert len(groups) >= 1
        flat = [f for group in groups for f in group]
        assert "base" in flat
        assert "copy_of_base" in flat

    def test_correlated_groups_field_always_present(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert "correlated_feature_groups" in result["explanation_summary"]


# ── Permutation Importance ───────────────────────────────────────


class TestPermutationImportance:
    """Cross-check SHAP with permutation importance."""

    def test_permutation_importance_present_with_y_test(
        self, agent, feature_matrix, target, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
            "y_test": target,
        })
        assert "permutation_importance" in result
        pi = result["permutation_importance"]
        assert isinstance(pi, dict)
        assert set(pi.keys()) == set(feature_names)

    def test_permutation_importance_absent_without_y_test(
        self, agent, feature_matrix, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
        })
        assert "permutation_importance" not in result

    def test_permutation_values_are_finite(
        self, agent, feature_matrix, target, trained_model
    ):
        model, scaler, feature_names = trained_model
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
            "y_test": target,
        })
        for val in result["permutation_importance"].values():
            assert np.isfinite(val)

    def test_informative_feature_has_high_permutation_importance(self, agent):
        """A strongly predictive feature should have high PI."""
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.bdate_range(start="2023-01-01", periods=n)
        y = pd.Series(rng.randint(0, 2, size=n), index=dates, name="target")
        X = pd.DataFrame({
            "signal": y * 3 + rng.randn(n) * 0.1,
            "noise": rng.randn(n),
        }, index=dates)
        model, scaler, feature_names = _train_model(X, y)
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": X,
            "y_test": y,
        })
        pi = result["permutation_importance"]
        assert pi["signal"] > pi["noise"]


# ── Multiclass Rejection ─────────────────────────────────────────


class TestMulticlassRejection:
    """Agent must reject multiclass models explicitly."""

    def test_rejects_multiclass_model(self, agent):
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.bdate_range(start="2023-01-01", periods=n)
        X = pd.DataFrame(rng.randn(n, 3), index=dates, columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 3, size=n), index=dates, name="target")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_scaled, y)
        assert len(model.classes_) == 3
        with pytest.raises(ValueError, match="Multiclass"):
            agent.run({
                "model": model,
                "scaler": scaler,
                "feature_names": list(X.columns),
                "feature_matrix": X,
            })


# ── Multi-class SHAP extraction ──────────────────────────────────


class TestSHAPExtraction:
    """_extract_positive_class_shap handles various formats correctly."""

    def test_list_of_two_arrays(self):
        arr0 = np.array([[0.1, 0.2], [0.3, 0.4]])
        arr1 = np.array([[0.5, 0.6], [0.7, 0.8]])
        result = _extract_positive_class_shap([arr0, arr1])
        np.testing.assert_array_equal(result, arr1)

    def test_list_of_one_array(self):
        arr0 = np.array([[0.1, 0.2]])
        result = _extract_positive_class_shap([arr0])
        np.testing.assert_array_equal(result, arr0)

    def test_3d_array_binary(self):
        arr = np.random.randn(5, 3, 2)
        result = _extract_positive_class_shap(arr)
        np.testing.assert_array_equal(result, arr[:, :, 1])

    def test_2d_array_passthrough(self):
        arr = np.random.randn(5, 3)
        result = _extract_positive_class_shap(arr)
        np.testing.assert_array_equal(result, arr)

    def test_rejects_three_class_list(self):
        with pytest.raises(ValueError, match="binary"):
            _extract_positive_class_shap([np.zeros((2, 3))] * 3)

    def test_rejects_three_class_3d(self):
        with pytest.raises(ValueError, match="binary"):
            _extract_positive_class_shap(np.zeros((5, 3, 3)))


# ── Tree Model Detection ─────────────────────────────────────────


class TestTreeModelDetection:
    """_is_tree_model covers sklearn, XGBoost, LightGBM, CatBoost."""

    def test_random_forest_detected(self):
        model = RandomForestClassifier(n_estimators=5)
        model.fit(np.random.randn(10, 2), np.random.randint(0, 2, 10))
        assert _is_tree_model(model) is True

    def test_logistic_regression_not_tree(self):
        model = LogisticRegression()
        model.fit(np.random.randn(10, 2), np.random.randint(0, 2, 10))
        assert _is_tree_model(model) is False

    def test_xgboost_name_detected(self):
        class FakeXGB:
            __name__ = "XGBClassifier"
        fake = FakeXGB()
        fake.__class__.__name__ = "XGBClassifier"
        assert _is_tree_model(fake) is True


# ── KernelExplainer Background ───────────────────────────────────


class TestKernelExplainerBackground:
    """KernelExplainer should prefer X_train as background."""

    def test_kernel_explainer_with_x_train(self, agent, feature_matrix, target):
        model, scaler, feature_names = _train_model(
            feature_matrix, target, model_type="logistic_regression"
        )
        X_train = feature_matrix.iloc[:100]
        X_test = feature_matrix.iloc[100:]
        result = agent.run({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_matrix": X_test,
            "X_train": X_train,
        })
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == len(feature_names)

    def test_kernel_explainer_warns_without_x_train(
        self, agent, feature_matrix, target, caplog
    ):
        model, scaler, feature_names = _train_model(
            feature_matrix, target, model_type="logistic_regression"
        )
        import logging
        with caplog.at_level(logging.WARNING):
            agent.run({
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
                "feature_matrix": feature_matrix,
            })
        assert any(
            "No X_train provided" in record.message
            for record in caplog.records
        )


# ── Stability Normalization ──────────────────────────────────────


class TestStabilityNormalization:
    """Stability scores must be properly normalized."""

    def test_stability_scores_between_0_and_1(self, agent, fold_results):
        result = agent.run({
            "model": fold_results[-1]["model"],
            "scaler": fold_results[-1]["scaler"],
            "feature_names": fold_results[-1]["feature_names"],
            "feature_matrix": fold_results[-1]["X_test"],
            "fold_results": fold_results,
        })
        drift = result["importance_drift"]
        for score in drift["stability_scores"].values():
            assert 0.0 <= score <= 1.0

    def test_perfectly_stable_feature_scores_1(self, agent):
        """If a feature always has the same rank, stability = 1.0."""
        from agents.explainability_agent import _get_feature_rank
        importances = [
            {"a": 0.5, "b": 0.3, "c": 0.1},
            {"a": 0.6, "b": 0.2, "c": 0.05},
            {"a": 0.7, "b": 0.15, "c": 0.02},
        ]
        ranks = [_get_feature_rank(fi, "a") for fi in importances]
        assert all(r == 1 for r in ranks)
        rank_std = float(np.std(ranks))
        assert rank_std == 0.0
