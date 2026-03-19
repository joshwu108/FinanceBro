"""Tests for OverfittingAgent.

Validates:
  - BaseAgent contract (run, validate, log_metrics, schemas)
  - Train vs test gap detection (accuracy, f1, log_loss)
  - Fold instability detection (high variance across folds)
  - Suspiciously good results flagging (>60% accuracy, Sharpe > 2)
  - Degradation pattern detection (declining fold performance)
  - Overfitting score calculation (0-1 range, monotonic with severity)
  - Warning generation for each failure mode
  - Input validation (missing keys, wrong types)
  - Experiment log format
"""

import json
import numpy as np
import pytest

from agents.overfitting_agent import OverfittingAgent


# ── Fixtures ─────────────────────────────────────────────────────


def _make_train_metrics(
    accuracy: float = 0.85,
    precision: float = 0.83,
    recall: float = 0.80,
    f1: float = 0.81,
    log_loss: float = 0.40,
) -> dict:
    """Create train metrics dict matching ModelAgent output format."""
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss,
    }


def _make_test_metrics(
    accuracy: float = 0.52,
    precision: float = 0.50,
    recall: float = 0.48,
    f1: float = 0.49,
    log_loss: float = 0.69,
) -> dict:
    """Create test metrics dict matching ModelAgent output format."""
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss,
    }


def _make_fold_result(
    fold_index: int,
    sharpe: float = 0.5,
    max_drawdown: float = -0.10,
    total_return: float = 0.05,
    accuracy: float = 0.52,
    f1: float = 0.50,
    train_accuracy: float = 0.80,
    train_f1: float = 0.78,
) -> dict:
    """Create a single fold result matching WalkForwardAgent output format."""
    return {
        "fold_index": fold_index,
        "split_info": {
            "train_size": 126 + fold_index * 63,
            "test_size": 63,
            "train_start": "2023-01-01",
            "train_end": f"2023-0{fold_index + 6}-30",
            "test_start": f"2023-0{fold_index + 7}-01",
            "test_end": f"2023-0{fold_index + 9}-30",
        },
        "model_metrics": {
            "accuracy": accuracy,
            "precision": accuracy - 0.02,
            "recall": accuracy - 0.04,
            "f1": f1,
            "log_loss": 0.69,
        },
        "backtest_metrics": {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "sortino": sharpe * 1.2,
            "calmar": abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0,
            "win_rate": 0.52,
            "turnover": 0.10,
        },
        "model_type": "logistic_regression",
        "train_metrics": {
            "accuracy": train_accuracy,
            "precision": train_accuracy - 0.02,
            "recall": train_accuracy - 0.05,
            "f1": train_f1,
            "log_loss": 0.40,
        },
    }


def _make_stable_fold_results(n_folds: int = 5) -> list:
    """Create fold results with stable, modest performance (no overfitting)."""
    return [
        _make_fold_result(
            fold_index=i,
            sharpe=0.3 + np.random.RandomState(i).uniform(-0.1, 0.1),
            max_drawdown=-0.08 + np.random.RandomState(i).uniform(-0.02, 0.02),
            total_return=0.03 + np.random.RandomState(i).uniform(-0.01, 0.01),
            accuracy=0.52 + np.random.RandomState(i).uniform(-0.02, 0.02),
            f1=0.50 + np.random.RandomState(i).uniform(-0.02, 0.02),
            train_accuracy=0.55 + np.random.RandomState(i).uniform(-0.02, 0.02),
            train_f1=0.53 + np.random.RandomState(i).uniform(-0.02, 0.02),
        )
        for i in range(n_folds)
    ]


def _make_overfit_fold_results(n_folds: int = 5) -> list:
    """Create fold results with clear overfitting: high train, low test, unstable."""
    return [
        _make_fold_result(
            fold_index=i,
            sharpe=0.2 + np.random.RandomState(i).uniform(-2.0, 2.0),
            max_drawdown=-0.20 + np.random.RandomState(i).uniform(-0.15, 0.0),
            total_return=0.00 + np.random.RandomState(i).uniform(-0.08, 0.04),
            accuracy=0.46 + np.random.RandomState(i).uniform(-0.06, 0.06),
            f1=0.43 + np.random.RandomState(i).uniform(-0.06, 0.06),
            train_accuracy=0.95 + np.random.RandomState(i).uniform(-0.02, 0.02),
            train_f1=0.93 + np.random.RandomState(i).uniform(-0.02, 0.02),
        )
        for i in range(n_folds)
    ]


def _make_suspicious_fold_results(n_folds: int = 5) -> list:
    """Create fold results with suspiciously good performance."""
    return [
        _make_fold_result(
            fold_index=i,
            sharpe=2.5 + np.random.RandomState(i).uniform(-0.3, 0.3),
            max_drawdown=-0.03,
            total_return=0.25 + np.random.RandomState(i).uniform(-0.02, 0.02),
            accuracy=0.68 + np.random.RandomState(i).uniform(-0.02, 0.02),
            f1=0.65 + np.random.RandomState(i).uniform(-0.02, 0.02),
            train_accuracy=0.70 + np.random.RandomState(i).uniform(-0.02, 0.02),
            train_f1=0.68 + np.random.RandomState(i).uniform(-0.02, 0.02),
        )
        for i in range(n_folds)
    ]


def _make_degrading_fold_results(n_folds: int = 5) -> list:
    """Create fold results with a clear degradation pattern over time."""
    return [
        _make_fold_result(
            fold_index=i,
            sharpe=1.0 - i * 0.4,
            max_drawdown=-0.05 - i * 0.03,
            total_return=0.08 - i * 0.03,
            accuracy=0.58 - i * 0.03,
            f1=0.55 - i * 0.03,
            train_accuracy=0.80,
            train_f1=0.78,
        )
        for i in range(n_folds)
    ]


def _make_good_inputs() -> dict:
    """Standard well-formed inputs for the OverfittingAgent."""
    fold_results = _make_stable_fold_results()
    return {
        "train_metrics": _make_train_metrics(accuracy=0.55, f1=0.53),
        "test_metrics": _make_test_metrics(accuracy=0.52, f1=0.50),
        "fold_results": fold_results,
    }


def _make_overfit_inputs() -> dict:
    """Inputs that exhibit clear overfitting signals."""
    fold_results = _make_overfit_fold_results()
    return {
        "train_metrics": _make_train_metrics(accuracy=0.95, f1=0.93),
        "test_metrics": _make_test_metrics(accuracy=0.46, f1=0.43),
        "fold_results": fold_results,
    }


# ── BaseAgent contract ────────────────────────────────────────────


class TestBaseAgentContract:
    """OverfittingAgent must satisfy BaseAgent interface."""

    def test_has_run_method(self):
        agent = OverfittingAgent()
        assert callable(getattr(agent, "run", None))

    def test_has_validate_method(self):
        agent = OverfittingAgent()
        assert callable(getattr(agent, "validate", None))

    def test_has_log_metrics_method(self):
        agent = OverfittingAgent()
        assert callable(getattr(agent, "log_metrics", None))

    def test_has_input_schema(self):
        agent = OverfittingAgent()
        schema = agent.input_schema
        assert isinstance(schema, dict)
        assert "train_metrics" in schema
        assert "test_metrics" in schema
        assert "fold_results" in schema

    def test_has_output_schema(self):
        agent = OverfittingAgent()
        schema = agent.output_schema
        assert isinstance(schema, dict)
        assert "overfitting_score" in schema
        assert "warnings" in schema
        assert "diagnostics" in schema
        assert "failure_modes" in schema
        assert "recommendations" in schema

    def test_run_returns_dict(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        assert isinstance(result, dict)

    def test_run_output_matches_schema(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        for key in agent.output_schema:
            assert key in result, f"Output missing key: {key}"


# ── Overfitting score range ───────────────────────────────────────


class TestOverfittingScoreRange:
    """overfitting_score must be in [0, 1]."""

    def test_score_between_zero_and_one(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        assert 0.0 <= result["overfitting_score"] <= 1.0

    def test_low_score_for_stable_model(self):
        """A model with small train-test gap and stable folds should score low."""
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        result = agent.run(inputs)
        assert result["overfitting_score"] < 0.4

    def test_high_score_for_overfit_model(self):
        """A model with large train-test gap and unstable folds should score high."""
        agent = OverfittingAgent()
        inputs = _make_overfit_inputs()
        result = agent.run(inputs)
        assert result["overfitting_score"] > 0.5

    def test_score_monotonic_with_gap_severity(self):
        """Larger train-test gaps should produce higher scores."""
        agent = OverfittingAgent()
        fold_results = _make_stable_fold_results()

        # Small gap
        inputs_small = {
            "train_metrics": _make_train_metrics(accuracy=0.55),
            "test_metrics": _make_test_metrics(accuracy=0.52),
            "fold_results": fold_results,
        }
        score_small = agent.run(inputs_small)["overfitting_score"]

        # Large gap
        inputs_large = {
            "train_metrics": _make_train_metrics(accuracy=0.90),
            "test_metrics": _make_test_metrics(accuracy=0.48),
            "fold_results": fold_results,
        }
        score_large = agent.run(inputs_large)["overfitting_score"]

        assert score_large > score_small


# ── Train vs test gap detection ───────────────────────────────────


class TestTrainTestGapDetection:
    """Detect overfitting via train-test performance gaps."""

    def test_large_accuracy_gap_flagged(self):
        """accuracy gap > 10% should produce a warning."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.85),
            "test_metrics": _make_test_metrics(accuracy=0.50),
            "fold_results": _make_stable_fold_results(),
        }
        result = agent.run(inputs)
        gap_warnings = [w for w in result["warnings"] if "gap" in w.lower()]
        assert len(gap_warnings) > 0

    def test_large_f1_gap_flagged(self):
        """f1 gap > 15% should produce a warning."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(f1=0.85),
            "test_metrics": _make_test_metrics(f1=0.50),
            "fold_results": _make_stable_fold_results(),
        }
        result = agent.run(inputs)
        gap_warnings = [w for w in result["warnings"] if "f1" in w.lower() or "gap" in w.lower()]
        assert len(gap_warnings) > 0

    def test_small_gap_no_warning(self):
        """A small train-test gap should NOT flag gap-related warnings."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.54, f1=0.52),
            "test_metrics": _make_test_metrics(accuracy=0.52, f1=0.50),
            "fold_results": _make_stable_fold_results(),
        }
        result = agent.run(inputs)
        gap_warnings = [w for w in result["warnings"] if "gap" in w.lower()]
        assert len(gap_warnings) == 0

    def test_gap_diagnostics_reported(self):
        """Diagnostics should include per-metric gap values."""
        agent = OverfittingAgent()
        result = agent.run(_make_overfit_inputs())
        diag = result["diagnostics"]
        assert "train_test_gap" in diag
        assert "accuracy" in diag["train_test_gap"]
        assert "f1" in diag["train_test_gap"]

    def test_memorization_detected(self):
        """Train accuracy > 90% should flag memorization."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.95),
            "test_metrics": _make_test_metrics(accuracy=0.50),
            "fold_results": _make_stable_fold_results(),
        }
        result = agent.run(inputs)
        mem_warnings = [w for w in result["warnings"] if "memoriz" in w.lower()]
        assert len(mem_warnings) > 0


# ── Fold instability detection ────────────────────────────────────


class TestFoldInstability:
    """Detect unstable performance across walk-forward folds."""

    def test_high_sharpe_variance_flagged(self):
        """Large variance in Sharpe across folds should warn."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.55),
            "test_metrics": _make_test_metrics(accuracy=0.52),
            "fold_results": _make_overfit_fold_results(),
        }
        result = agent.run(inputs)
        instability_warnings = [
            w for w in result["warnings"]
            if "unstable" in w.lower() or "instab" in w.lower() or "variance" in w.lower()
        ]
        assert len(instability_warnings) > 0

    def test_stable_folds_no_warning(self):
        """Stable folds should NOT flag instability warnings."""
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        result = agent.run(inputs)
        instability_warnings = [
            w for w in result["warnings"]
            if "unstable" in w.lower() or "instab" in w.lower()
        ]
        assert len(instability_warnings) == 0

    def test_fold_metrics_in_diagnostics(self):
        """Diagnostics should include per-fold sharpe and accuracy arrays."""
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        diag = result["diagnostics"]
        assert "fold_sharpes" in diag
        assert "fold_accuracies" in diag
        assert len(diag["fold_sharpes"]) == 5

    def test_negative_sharpe_folds_flagged(self):
        """Folds with negative Sharpe should be called out."""
        fold_results = _make_stable_fold_results()
        # Force two folds to have negative Sharpe
        fold_results[1]["backtest_metrics"]["sharpe"] = -0.5
        fold_results[3]["backtest_metrics"]["sharpe"] = -0.3

        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.55),
            "test_metrics": _make_test_metrics(accuracy=0.52),
            "fold_results": fold_results,
        }
        result = agent.run(inputs)
        neg_warnings = [
            w for w in result["warnings"]
            if "negative" in w.lower() and "sharpe" in w.lower()
        ]
        assert len(neg_warnings) > 0


# ── Suspiciously good results ────────────────────────────────────


class TestSuspiciouslyGood:
    """Flag results that are too good to be true for daily direction prediction."""

    def test_high_test_accuracy_flagged(self):
        """Test accuracy > 60% on daily direction should be suspicious."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.65),
            "test_metrics": _make_test_metrics(accuracy=0.62),
            "fold_results": _make_suspicious_fold_results(),
        }
        result = agent.run(inputs)
        sus_warnings = [
            w for w in result["warnings"]
            if "suspici" in w.lower() or "too good" in w.lower() or "leakage" in w.lower()
        ]
        assert len(sus_warnings) > 0

    def test_high_sharpe_flagged(self):
        """Mean Sharpe > 2.0 across folds should be flagged."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.65),
            "test_metrics": _make_test_metrics(accuracy=0.62),
            "fold_results": _make_suspicious_fold_results(),
        }
        result = agent.run(inputs)
        sharpe_warnings = [
            w for w in result["warnings"]
            if "sharpe" in w.lower() and ("suspici" in w.lower() or "too" in w.lower())
        ]
        assert len(sharpe_warnings) > 0

    def test_modest_results_not_flagged(self):
        """Modest, realistic results should not be flagged as suspicious."""
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        result = agent.run(inputs)
        sus_warnings = [
            w for w in result["warnings"]
            if "suspici" in w.lower() or "too good" in w.lower()
        ]
        assert len(sus_warnings) == 0


# ── Degradation pattern ──────────────────────────────────────────


class TestDegradationPattern:
    """Detect declining performance over later folds."""

    def test_declining_performance_flagged(self):
        """If fold performance degrades monotonically, flag it."""
        agent = OverfittingAgent()
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.80),
            "test_metrics": _make_test_metrics(accuracy=0.50),
            "fold_results": _make_degrading_fold_results(),
        }
        result = agent.run(inputs)
        degrade_warnings = [
            w for w in result["warnings"]
            if "degrad" in w.lower() or "declin" in w.lower()
        ]
        assert len(degrade_warnings) > 0

    def test_stable_folds_no_degradation_warning(self):
        """Stable folds should not trigger degradation warnings."""
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        degrade_warnings = [
            w for w in result["warnings"]
            if "degrad" in w.lower() or "declin" in w.lower()
        ]
        assert len(degrade_warnings) == 0


# ── Failure modes and recommendations ────────────────────────────


class TestFailureModesAndRecommendations:
    """Output should include actionable failure_modes and recommendations."""

    def test_failure_modes_populated_on_overfit(self):
        agent = OverfittingAgent()
        result = agent.run(_make_overfit_inputs())
        assert isinstance(result["failure_modes"], list)
        assert len(result["failure_modes"]) > 0

    def test_recommendations_populated_on_overfit(self):
        agent = OverfittingAgent()
        result = agent.run(_make_overfit_inputs())
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_no_failure_modes_for_clean_model(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        assert isinstance(result["failure_modes"], list)
        # A clean model may still have minor modes, but should be fewer
        overfit_result = agent.run(_make_overfit_inputs())
        assert len(result["failure_modes"]) < len(overfit_result["failure_modes"])

    def test_warnings_list_type(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        assert isinstance(result["warnings"], list)
        for w in result["warnings"]:
            assert isinstance(w, str)


# ── Input validation ──────────────────────────────────────────────


class TestInputValidation:
    """OverfittingAgent must reject malformed inputs."""

    def test_missing_train_metrics(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        del inputs["train_metrics"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(inputs)

    def test_missing_test_metrics(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        del inputs["test_metrics"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(inputs)

    def test_missing_fold_results(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        del inputs["fold_results"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(inputs)

    def test_empty_fold_results(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        inputs["fold_results"] = []
        with pytest.raises(ValueError):
            agent.run(inputs)

    def test_wrong_type_train_metrics(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        inputs["train_metrics"] = "not a dict"
        with pytest.raises((TypeError, ValueError)):
            agent.run(inputs)

    def test_wrong_type_fold_results(self):
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        inputs["fold_results"] = "not a list"
        with pytest.raises((TypeError, ValueError)):
            agent.run(inputs)

    def test_fold_result_missing_backtest_metrics(self):
        """Each fold must have backtest_metrics."""
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        del inputs["fold_results"][0]["backtest_metrics"]
        with pytest.raises((ValueError, KeyError)):
            agent.run(inputs)


# ── Validate method ──────────────────────────────────────────────


class TestValidateMethod:
    """validate() should check structural correctness of outputs."""

    def test_validate_returns_true_on_good_output(self):
        agent = OverfittingAgent()
        result = agent.run(_make_good_inputs())
        assert agent.validate(_make_good_inputs(), result) is True

    def test_validate_rejects_score_out_of_range(self):
        agent = OverfittingAgent()
        bad_output = {
            "overfitting_score": 1.5,
            "warnings": [],
            "diagnostics": {},
            "failure_modes": [],
            "recommendations": [],
        }
        with pytest.raises(ValueError):
            agent.validate(_make_good_inputs(), bad_output)

    def test_validate_rejects_missing_key(self):
        agent = OverfittingAgent()
        bad_output = {
            "overfitting_score": 0.5,
            # missing warnings, diagnostics, etc.
        }
        with pytest.raises(ValueError):
            agent.validate(_make_good_inputs(), bad_output)


# ── Experiment logging ───────────────────────────────────────────


class TestExperimentLogging:
    """log_metrics must persist to experiments/ in the correct format."""

    def test_log_metrics_creates_file(self, tmp_path, monkeypatch):
        agent = OverfittingAgent()
        monkeypatch.setattr(
            type(agent),
            "_experiments_dir",
            property(lambda self: tmp_path),
        )
        agent.run(_make_good_inputs())
        agent.log_metrics()

        log_files = list(tmp_path.glob("overfitting_agent_*.json"))
        assert len(log_files) == 1

    def test_log_metrics_format(self, tmp_path, monkeypatch):
        agent = OverfittingAgent()
        monkeypatch.setattr(
            type(agent),
            "_experiments_dir",
            property(lambda self: tmp_path),
        )
        agent.run(_make_overfit_inputs())
        agent.log_metrics()

        log_files = list(tmp_path.glob("overfitting_agent_*.json"))
        data = json.loads(log_files[0].read_text())

        assert "experiment_id" in data
        assert data["agent"] == "OverfittingAgent"
        assert "overfitting_score" in data.get("metrics", {})
        assert "timestamp" in data
        assert "out_of_sample" in data

    def test_log_metrics_noop_before_run(self, tmp_path, monkeypatch):
        """log_metrics before run() should not create a file."""
        agent = OverfittingAgent()
        monkeypatch.setattr(
            type(agent),
            "_experiments_dir",
            property(lambda self: tmp_path),
        )
        agent.log_metrics()
        log_files = list(tmp_path.glob("*.json"))
        assert len(log_files) == 0


# ── Config override ──────────────────────────────────────────────


class TestConfigOverride:
    """Agent should accept config overrides via inputs or constructor."""

    def test_custom_gap_threshold(self):
        """Custom gap threshold should change sensitivity."""
        agent = OverfittingAgent(config={"accuracy_gap_threshold": 0.50})
        inputs = {
            "train_metrics": _make_train_metrics(accuracy=0.85),
            "test_metrics": _make_test_metrics(accuracy=0.50),
            "fold_results": _make_stable_fold_results(),
        }
        result = agent.run(inputs)
        # With threshold at 0.50, a 0.35 gap should NOT trigger
        gap_warnings = [w for w in result["warnings"] if "accuracy" in w.lower() and "gap" in w.lower()]
        assert len(gap_warnings) == 0

    def test_config_via_inputs(self):
        """Config passed in inputs should override defaults."""
        agent = OverfittingAgent()
        inputs = _make_good_inputs()
        inputs["config"] = {"accuracy_gap_threshold": 0.01}
        result = agent.run(inputs)
        # With very tight threshold, even small gap triggers
        gap_warnings = [w for w in result["warnings"] if "gap" in w.lower()]
        assert len(gap_warnings) > 0


# ── Per-fold train-test gap analysis ─────────────────────────────


class TestPerFoldTrainTestGap:
    """When fold_results include train_metrics, analyze per-fold gaps too."""

    def test_per_fold_gap_in_diagnostics(self):
        """Diagnostics should include per-fold train-test accuracy gaps."""
        agent = OverfittingAgent()
        result = agent.run(_make_overfit_inputs())
        diag = result["diagnostics"]
        assert "per_fold_train_test_gaps" in diag
        assert len(diag["per_fold_train_test_gaps"]) == 5

    def test_large_per_fold_gap_flagged(self):
        """If individual folds show large train-test gaps, flag them."""
        agent = OverfittingAgent()
        result = agent.run(_make_overfit_inputs())
        # The overfit data has ~0.92 train vs ~0.48 test in each fold
        assert result["overfitting_score"] > 0.5
