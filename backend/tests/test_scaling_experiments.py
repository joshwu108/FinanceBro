"""Tests for scaling experiments — verifies experiment runners work end-to-end."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantum.benchmarks.scaling_experiments import (
    run_portfolio_scaling,
    run_maxcut_scaling,
    run_cpp_speedup_benchmark,
    synthetic_returns,
    _log_experiment,
)


# ===========================================================================
# 1. Synthetic data generation
# ===========================================================================

class TestSyntheticReturns:

    def test_shape(self):
        df = synthetic_returns(5, n_days=100, seed=42)
        assert df.shape == (100, 5)

    def test_reproducible(self):
        df1 = synthetic_returns(3, seed=42)
        df2 = synthetic_returns(3, seed=42)
        assert df1.equals(df2)


# ===========================================================================
# 2. Portfolio scaling experiment
# ===========================================================================

class TestPortfolioScaling:

    def test_returns_results(self):
        result = run_portfolio_scaling(
            asset_counts=[3, 4],
            n_bits=2,
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        assert "results" in result
        assert len(result["results"]) == 2

    def test_result_fields(self):
        result = run_portfolio_scaling(
            asset_counts=[3],
            n_bits=2,
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        r = result["results"][0]
        assert "n_assets" in r
        assert "classical_runtime_ms" in r
        assert "qaoa_runtime_ms" in r
        assert "runtime_ratio" in r

    def test_scaling_exponents_computed(self):
        result = run_portfolio_scaling(
            asset_counts=[3, 4, 5],
            n_bits=2,
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        assert "scaling" in result
        assert "qaoa" in result["scaling"]
        assert "exponent" in result["scaling"]["qaoa"]

    def test_logs_experiment(self, tmp_path, monkeypatch):
        """Experiment gets logged to JSON."""
        monkeypatch.setattr(
            "quantum.benchmarks.scaling_experiments.EXPERIMENTS_DIR", tmp_path
        )
        run_portfolio_scaling(
            asset_counts=[3],
            n_bits=2,
            qaoa_layers=1,
            qaoa_maxiter=20,
            seed=42,
        )
        files = list(tmp_path.glob("portfolio_scaling_*.json"))
        assert len(files) >= 1
        data = json.loads(files[0].read_text())
        assert data["experiment"] == "portfolio_scaling"


# ===========================================================================
# 3. Max-Cut scaling experiment
# ===========================================================================

class TestMaxCutScaling:

    def test_returns_results(self):
        result = run_maxcut_scaling(
            node_counts=[4, 6],
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        assert len(result["results"]) == 2

    def test_has_approx_ratios(self):
        result = run_maxcut_scaling(
            node_counts=[4],
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        r = result["results"][0]
        assert "qaoa_approx_ratio" in r
        assert "sa_approx_ratio" in r
        assert "greedy_approx_ratio" in r

    def test_all_methods_have_runtimes(self):
        result = run_maxcut_scaling(
            node_counts=[4],
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        r = result["results"][0]
        assert r["qaoa_runtime_ms"] > 0
        assert r["sa_runtime_ms"] > 0
        assert r["greedy_runtime_ms"] > 0

    def test_scaling_exponents(self):
        result = run_maxcut_scaling(
            node_counts=[4, 6, 8],
            qaoa_layers=1,
            qaoa_maxiter=30,
            seed=42,
        )
        for method in ["qaoa", "sa", "greedy"]:
            assert method in result["scaling"]
            assert "exponent" in result["scaling"][method]


# ===========================================================================
# 4. C++ speedup benchmark
# ===========================================================================

class TestCppSpeedup:

    def test_returns_results(self):
        result = run_cpp_speedup_benchmark(
            qubit_counts=[4, 6],
            n_iterations=3,
        )
        assert len(result["results"]) == 2

    def test_python_timing_positive(self):
        result = run_cpp_speedup_benchmark(
            qubit_counts=[4],
            n_iterations=3,
        )
        assert result["results"][0]["python_ms"] > 0

    def test_cpp_speedup_if_available(self):
        result = run_cpp_speedup_benchmark(
            qubit_counts=[8],
            n_iterations=3,
        )
        r = result["results"][0]
        if result["has_cpp"]:
            assert r["cpp_ms"] is not None
            assert r["speedup"] is not None
            assert r["speedup"] > 0


# ===========================================================================
# 5. Experiment logging
# ===========================================================================

class TestExperimentLogging:

    def test_log_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "quantum.benchmarks.scaling_experiments.EXPERIMENTS_DIR", tmp_path
        )
        path = _log_experiment("test_exp", {"key": "value", "arr": np.array([1, 2])})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["key"] == "value"
        assert data["arr"] == [1, 2]
