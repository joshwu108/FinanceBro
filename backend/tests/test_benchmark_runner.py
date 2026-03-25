"""Tests for quantum benchmarking framework.

TDD: These tests are written FIRST, before implementation.
Covers BenchmarkResult, ComparisonMetrics, BenchmarkRunner, and scaling analysis.
"""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantum.solvers.classical_solvers import SolverResult
from quantum.benchmarks.metrics import ComparisonMetrics, compute_comparison_metrics
from quantum.benchmarks.benchmark_runner import BenchmarkResult, BenchmarkRunner
from quantum.benchmarks.scaling_analysis import (
    compute_scaling_exponents,
    generate_maxcut_qubo,
    generate_portfolio_qubo,
    generate_random_qubo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_experiments_dir():
    """Create a temporary directory for experiment logs, clean up after."""
    d = tempfile.mkdtemp(prefix="bench_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def small_qubo():
    """A small 3x3 QUBO matrix with known optimal."""
    # Diagonal-dominant => optimal is all zeros (obj = 0)
    Q = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 2.0, 0.3],
        [0.0, 0.0, 1.5],
    ])
    return Q


@pytest.fixture
def make_solver_result():
    """Factory fixture for creating SolverResult instances."""
    def _make(
        weights=None,
        objective_value=0.0,
        runtime_ms=1.0,
        method="test_solver",
        converged=True,
    ):
        if weights is None:
            weights = np.array([0.0, 1.0, 0.0])
        return SolverResult(
            weights=weights,
            objective_value=objective_value,
            runtime_ms=runtime_ms,
            converged=converged,
            method=method,
        )
    return _make


# ===========================================================================
# 1. BenchmarkResult dataclass
# ===========================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation_with_required_fields(self):
        """BenchmarkResult can be created with all required fields."""
        result = BenchmarkResult(
            experiment_id="test-001",
            timestamp="2026-03-25T12:00:00Z",
            config={"solver": "brute_force", "n_repetitions": 5},
            classical_results=[{"objective_value": 1.0}],
            quantum_results=[{"objective_value": 1.2}],
            comparison={"approximation_ratio": 0.83},
        )
        assert result.experiment_id == "test-001"
        assert result.timestamp == "2026-03-25T12:00:00Z"
        assert result.config["solver"] == "brute_force"
        assert len(result.classical_results) == 1
        assert len(result.quantum_results) == 1
        assert result.comparison["approximation_ratio"] == 0.83
        # Defaults
        assert result.problem_description == ""
        assert result.notes == ""

    def test_serialization_to_dict(self):
        """to_dict() produces a JSON-serializable dictionary."""
        result = BenchmarkResult(
            experiment_id="test-002",
            timestamp="2026-03-25T12:00:00Z",
            config={"n": 3},
            classical_results=[{"obj": 1.0}],
            quantum_results=[{"obj": 1.1}],
            comparison={"ratio": 0.9},
            problem_description="test problem",
            notes="a note",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        # Round-trip check
        loaded = json.loads(json_str)
        assert loaded["experiment_id"] == "test-002"
        assert loaded["problem_description"] == "test problem"
        assert loaded["notes"] == "a note"

    def test_from_solver_results_factory(self, make_solver_result):
        """from_solver_results() creates BenchmarkResult from SolverResult lists."""
        classical = [
            make_solver_result(objective_value=5.0, runtime_ms=10.0, method="brute_force"),
            make_solver_result(objective_value=5.0, runtime_ms=11.0, method="brute_force"),
        ]
        quantum = [
            make_solver_result(objective_value=5.5, runtime_ms=2.0, method="qaoa"),
            make_solver_result(objective_value=5.2, runtime_ms=3.0, method="qaoa"),
        ]
        config = {"n_repetitions": 2}

        result = BenchmarkResult.from_solver_results(
            classical=classical,
            quantum=quantum,
            config=config,
            problem_description="test qubo",
        )

        assert isinstance(result, BenchmarkResult)
        assert result.experiment_id  # non-empty
        assert result.timestamp  # non-empty
        assert len(result.classical_results) == 2
        assert len(result.quantum_results) == 2
        assert "approximation_ratio" in result.comparison
        assert result.problem_description == "test qubo"

    def test_from_solver_results_numpy_serializable(self, make_solver_result):
        """from_solver_results() produces JSON-serializable output (no numpy)."""
        classical = [make_solver_result(
            weights=np.array([1.0, 0.0]),
            objective_value=3.0,
            runtime_ms=5.0,
        )]
        quantum = [make_solver_result(
            weights=np.array([0.0, 1.0]),
            objective_value=3.5,
            runtime_ms=1.0,
        )]

        result = BenchmarkResult.from_solver_results(
            classical=classical,
            quantum=quantum,
            config={},
        )

        d = result.to_dict()
        # This must not raise (numpy arrays are not JSON-serializable)
        json.dumps(d)


# ===========================================================================
# 2. ComparisonMetrics
# ===========================================================================


class TestComparisonMetrics:
    """Tests for ComparisonMetrics and compute_comparison_metrics()."""

    def test_approximation_ratio_identical(self):
        """When both solvers find the same objective, ratio == 1.0."""
        metrics = compute_comparison_metrics(
            classical_objectives=[10.0, 10.0, 10.0],
            quantum_objectives=[10.0, 10.0, 10.0],
            classical_runtimes=[5.0, 5.0, 5.0],
            quantum_runtimes=[3.0, 3.0, 3.0],
        )
        assert metrics.approximation_ratio == pytest.approx(1.0)

    def test_approximation_ratio_quantum_better(self):
        """Quantum finds lower objective => ratio > 1.0 (for minimization)."""
        metrics = compute_comparison_metrics(
            classical_objectives=[10.0, 10.0],
            quantum_objectives=[5.0, 5.0],
            classical_runtimes=[5.0, 5.0],
            quantum_runtimes=[3.0, 3.0],
        )
        # ratio = classical_mean / quantum_mean = 10/5 = 2.0
        assert metrics.approximation_ratio == pytest.approx(2.0)

    def test_approximation_ratio_quantum_worse(self):
        """Quantum finds higher objective => ratio < 1.0."""
        metrics = compute_comparison_metrics(
            classical_objectives=[5.0, 5.0],
            quantum_objectives=[10.0, 10.0],
            classical_runtimes=[5.0, 5.0],
            quantum_runtimes=[3.0, 3.0],
        )
        # ratio = 5/10 = 0.5
        assert metrics.approximation_ratio == pytest.approx(0.5)

    def test_runtime_ratio(self):
        """runtime_ratio = quantum_mean / classical_mean."""
        metrics = compute_comparison_metrics(
            classical_objectives=[1.0],
            quantum_objectives=[1.0],
            classical_runtimes=[10.0],
            quantum_runtimes=[2.0],
        )
        # ratio = 2.0 / 10.0 = 0.2
        assert metrics.runtime_ratio == pytest.approx(0.2)

    def test_quantum_wins_pct_all_wins(self):
        """quantum_wins_pct = 1.0 when quantum is always better."""
        metrics = compute_comparison_metrics(
            classical_objectives=[10.0, 10.0, 10.0],
            quantum_objectives=[5.0, 5.0, 5.0],
            classical_runtimes=[1.0, 1.0, 1.0],
            quantum_runtimes=[1.0, 1.0, 1.0],
        )
        assert metrics.quantum_wins_pct == pytest.approx(1.0)

    def test_quantum_wins_pct_no_wins(self):
        """quantum_wins_pct = 0.0 when classical is always better."""
        metrics = compute_comparison_metrics(
            classical_objectives=[1.0, 1.0, 1.0],
            quantum_objectives=[10.0, 10.0, 10.0],
            classical_runtimes=[1.0, 1.0, 1.0],
            quantum_runtimes=[1.0, 1.0, 1.0],
        )
        assert metrics.quantum_wins_pct == pytest.approx(0.0)

    def test_quantum_wins_pct_mixed(self):
        """quantum_wins_pct reflects fraction of wins (minimization: lower is better)."""
        metrics = compute_comparison_metrics(
            classical_objectives=[10.0, 5.0, 8.0, 3.0],
            quantum_objectives=[9.0, 6.0, 7.0, 4.0],
            classical_runtimes=[1.0, 1.0, 1.0, 1.0],
            quantum_runtimes=[1.0, 1.0, 1.0, 1.0],
        )
        # Wins: run0 (9<10), run2 (7<8) => 2/4 = 0.5
        assert metrics.quantum_wins_pct == pytest.approx(0.5)

    def test_mean_and_std_fields(self):
        """Mean and std fields are computed correctly."""
        classical_obj = [10.0, 20.0]
        quantum_obj = [12.0, 18.0]
        classical_rt = [5.0, 7.0]
        quantum_rt = [2.0, 4.0]

        metrics = compute_comparison_metrics(
            classical_objectives=classical_obj,
            quantum_objectives=quantum_obj,
            classical_runtimes=classical_rt,
            quantum_runtimes=quantum_rt,
        )
        assert metrics.mean_objective_classical == pytest.approx(15.0)
        assert metrics.mean_objective_quantum == pytest.approx(15.0)
        assert metrics.std_objective_classical == pytest.approx(np.std(classical_obj, ddof=0), abs=1e-6)
        assert metrics.std_objective_quantum == pytest.approx(np.std(quantum_obj, ddof=0), abs=1e-6)
        assert metrics.mean_runtime_classical_ms == pytest.approx(6.0)
        assert metrics.mean_runtime_quantum_ms == pytest.approx(3.0)

    def test_p_value_with_enough_repetitions(self):
        """p_value is computed when n >= 5."""
        # Clearly different distributions
        classical_obj = [10.0, 11.0, 10.5, 9.5, 10.2]
        quantum_obj = [5.0, 5.5, 4.5, 5.2, 4.8]
        runtimes = [1.0] * 5

        metrics = compute_comparison_metrics(
            classical_objectives=classical_obj,
            quantum_objectives=quantum_obj,
            classical_runtimes=runtimes,
            quantum_runtimes=runtimes,
        )
        assert metrics.p_value is not None
        assert 0.0 <= metrics.p_value <= 1.0
        # Should be significant
        assert metrics.p_value < 0.05

    def test_p_value_none_when_too_few_repetitions(self):
        """p_value is None when n < 5."""
        metrics = compute_comparison_metrics(
            classical_objectives=[10.0, 10.0],
            quantum_objectives=[5.0, 5.0],
            classical_runtimes=[1.0, 1.0],
            quantum_runtimes=[1.0, 1.0],
        )
        assert metrics.p_value is None

    def test_comparison_metrics_dataclass_fields(self):
        """ComparisonMetrics has all expected fields."""
        m = ComparisonMetrics(
            approximation_ratio=1.0,
            runtime_ratio=0.5,
            quantum_wins_pct=0.6,
            mean_objective_classical=10.0,
            mean_objective_quantum=10.0,
            std_objective_classical=1.0,
            std_objective_quantum=1.5,
            mean_runtime_classical_ms=5.0,
            mean_runtime_quantum_ms=2.5,
            p_value=0.03,
        )
        assert m.approximation_ratio == 1.0
        assert m.p_value == 0.03


# ===========================================================================
# 3. BenchmarkRunner.run_comparison()
# ===========================================================================


class TestRunComparison:
    """Tests for BenchmarkRunner.run_comparison()."""

    def test_both_solvers_called(self, small_qubo, tmp_experiments_dir):
        """Both classical and quantum solvers are called."""
        classical_mock = MagicMock(return_value=SolverResult(
            weights=np.zeros(3), objective_value=0.0, runtime_ms=1.0, method="mock_classical",
        ))
        quantum_mock = MagicMock(return_value=SolverResult(
            weights=np.zeros(3), objective_value=0.0, runtime_ms=1.0, method="mock_quantum",
        ))

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        result = runner.run_comparison(
            problem=small_qubo,
            classical_solver=classical_mock,
            quantum_solver=quantum_mock,
            n_repetitions=3,
        )

        assert classical_mock.call_count >= 1
        assert quantum_mock.call_count >= 1
        assert isinstance(result, BenchmarkResult)

    def test_with_real_solvers_on_small_qubo(self, small_qubo, tmp_experiments_dir):
        """Integration: brute_force vs greedy on a small QUBO."""
        from quantum.solvers.classical_solvers import solve_brute_force_qubo, greedy_qubo

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        result = runner.run_comparison(
            problem=small_qubo,
            classical_solver=solve_brute_force_qubo,
            quantum_solver=greedy_qubo,
            n_repetitions=3,
            problem_description="small 3x3 qubo",
        )

        assert isinstance(result, BenchmarkResult)
        assert len(result.classical_results) == 3
        assert len(result.quantum_results) == 3
        # Both should find the same optimum for this small problem
        assert result.comparison["approximation_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_n_repetitions_respected(self, small_qubo, tmp_experiments_dir):
        """n_repetitions controls how many times each solver runs."""
        call_count = {"classical": 0, "quantum": 0}

        def counting_classical(Q):
            call_count["classical"] += 1
            return SolverResult(
                weights=np.zeros(Q.shape[0]), objective_value=0.0,
                runtime_ms=1.0, method="counter_c",
            )

        def counting_quantum(Q):
            call_count["quantum"] += 1
            return SolverResult(
                weights=np.zeros(Q.shape[0]), objective_value=0.0,
                runtime_ms=1.0, method="counter_q",
            )

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        result = runner.run_comparison(
            problem=small_qubo,
            classical_solver=counting_classical,
            quantum_solver=counting_quantum,
            n_repetitions=7,
        )

        # Quantum must be called exactly n_repetitions times
        assert call_count["quantum"] == 7
        # Classical: at least 1 call (deterministic solvers may be called once
        # and replicated, or called n times)
        assert call_count["classical"] >= 1
        assert len(result.quantum_results) == 7
        assert len(result.classical_results) == 7

    def test_results_contain_both_sides(self, small_qubo, tmp_experiments_dir):
        """Result contains classical_results, quantum_results, and comparison."""
        dummy_solver = lambda Q: SolverResult(
            weights=np.zeros(Q.shape[0]), objective_value=1.0,
            runtime_ms=1.0, method="dummy",
        )

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        result = runner.run_comparison(
            problem=small_qubo,
            classical_solver=dummy_solver,
            quantum_solver=dummy_solver,
            n_repetitions=2,
        )

        assert "classical_results" in result.to_dict()
        assert "quantum_results" in result.to_dict()
        assert "comparison" in result.to_dict()

    def test_comparison_metrics_computed(self, small_qubo, tmp_experiments_dir):
        """Comparison dict contains expected metric keys."""
        def solver_a(Q):
            return SolverResult(
                weights=np.zeros(Q.shape[0]), objective_value=5.0,
                runtime_ms=10.0, method="a",
            )

        def solver_b(Q):
            return SolverResult(
                weights=np.zeros(Q.shape[0]), objective_value=10.0,
                runtime_ms=2.0, method="b",
            )

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        result = runner.run_comparison(
            problem=small_qubo,
            classical_solver=solver_a,
            quantum_solver=solver_b,
            n_repetitions=3,
        )

        comp = result.comparison
        assert "approximation_ratio" in comp
        assert "runtime_ratio" in comp
        assert "quantum_wins_pct" in comp
        assert "mean_objective_classical" in comp
        assert "mean_objective_quantum" in comp


# ===========================================================================
# 4. BenchmarkRunner.run_scaling_analysis()
# ===========================================================================


class TestRunScalingAnalysis:
    """Tests for BenchmarkRunner.run_scaling_analysis()."""

    def test_problem_sizes_covered(self, tmp_experiments_dir):
        """Results contain one entry per problem size."""
        def gen(size):
            rng = np.random.default_rng(42)
            Q = rng.random((size, size))
            return Q + Q.T

        def solver(Q):
            n = Q.shape[0]
            return SolverResult(
                weights=np.zeros(n), objective_value=0.0,
                runtime_ms=1.0, method="test",
            )

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        results = runner.run_scaling_analysis(
            problem_generator=gen,
            classical_solver=solver,
            quantum_solver=solver,
            problem_sizes=[2, 3, 4],
            n_repetitions=2,
        )

        assert len(results) == 3

    def test_results_contain_entry_per_size(self, tmp_experiments_dir):
        """Each BenchmarkResult in the list corresponds to a problem size."""
        sizes = [2, 3, 4]

        def gen(size):
            return np.eye(size)

        def solver(Q):
            return SolverResult(
                weights=np.zeros(Q.shape[0]), objective_value=0.0,
                runtime_ms=1.0, method="test",
            )

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        results = runner.run_scaling_analysis(
            problem_generator=gen,
            classical_solver=solver,
            quantum_solver=solver,
            problem_sizes=sizes,
            n_repetitions=2,
        )

        for i, size in enumerate(sizes):
            assert results[i].config.get("problem_size") == size

    def test_runtime_increases_with_size(self, tmp_experiments_dir):
        """Brute force runtime should increase with problem size."""
        from quantum.solvers.classical_solvers import solve_brute_force_qubo, greedy_qubo

        def gen(size):
            rng = np.random.default_rng(123)
            Q = rng.random((size, size))
            return (Q + Q.T) / 2

        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)
        results = runner.run_scaling_analysis(
            problem_generator=gen,
            classical_solver=solve_brute_force_qubo,
            quantum_solver=greedy_qubo,
            problem_sizes=[2, 4, 8],
            n_repetitions=2,
        )

        # Extract mean classical runtime for each size
        runtimes = []
        for r in results:
            mean_rt = r.comparison.get("mean_runtime_classical_ms", 0.0)
            runtimes.append(mean_rt)

        # Runtime for size 8 should be larger than for size 2
        assert runtimes[-1] > runtimes[0]


# ===========================================================================
# 5. BenchmarkRunner.log_experiment()
# ===========================================================================


class TestLogExperiment:
    """Tests for BenchmarkRunner.log_experiment()."""

    def test_creates_json_file(self, tmp_experiments_dir):
        """log_experiment() creates a JSON file in the experiments directory."""
        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)

        result = BenchmarkResult(
            experiment_id="log-test-001",
            timestamp="2026-03-25T12:00:00Z",
            config={"n": 3},
            classical_results=[{"obj": 1.0}],
            quantum_results=[{"obj": 1.1}],
            comparison={"ratio": 0.9},
        )

        path = runner.log_experiment(result)
        assert path.exists()
        assert path.suffix == ".json"
        assert path.parent == Path(tmp_experiments_dir)

    def test_json_contains_required_fields(self, tmp_experiments_dir):
        """Logged JSON contains timestamp, agent, config, results."""
        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)

        result = BenchmarkResult(
            experiment_id="log-test-002",
            timestamp="2026-03-25T12:00:00Z",
            config={"solver": "brute_force"},
            classical_results=[{"obj": 1.0}],
            quantum_results=[{"obj": 1.1}],
            comparison={"ratio": 0.9},
        )

        path = runner.log_experiment(result)
        with open(path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "agent" in data
        assert data["agent"] == "BenchmarkRunner"
        assert "config" in data
        assert "results" in data or "classical_results" in data
        assert "experiment_id" in data

    def test_filename_follows_convention(self, tmp_experiments_dir):
        """Filename matches quantum_benchmark_{timestamp}.json pattern."""
        runner = BenchmarkRunner(experiments_dir=tmp_experiments_dir)

        result = BenchmarkResult(
            experiment_id="log-test-003",
            timestamp="2026-03-25T12:00:00Z",
            config={},
            classical_results=[],
            quantum_results=[],
            comparison={},
        )

        path = runner.log_experiment(result)
        assert path.name.startswith("quantum_benchmark_")
        assert path.name.endswith(".json")


# ===========================================================================
# 6. compute_comparison_metrics() — statistical comparison
# ===========================================================================


class TestComputeComparisonMetrics:
    """Direct tests for compute_comparison_metrics() function."""

    def test_identical_results_ratio_one(self):
        """Identical objectives => approximation_ratio = 1.0."""
        metrics = compute_comparison_metrics(
            classical_objectives=[7.0, 7.0, 7.0],
            quantum_objectives=[7.0, 7.0, 7.0],
            classical_runtimes=[1.0, 1.0, 1.0],
            quantum_runtimes=[1.0, 1.0, 1.0],
        )
        assert metrics.approximation_ratio == pytest.approx(1.0)

    def test_quantum_worse_ratio_below_one(self):
        """Quantum worse (higher obj for minimization) => ratio < 1."""
        metrics = compute_comparison_metrics(
            classical_objectives=[2.0, 2.0, 2.0],
            quantum_objectives=[4.0, 4.0, 4.0],
            classical_runtimes=[1.0, 1.0, 1.0],
            quantum_runtimes=[1.0, 1.0, 1.0],
        )
        # ratio = 2/4 = 0.5
        assert metrics.approximation_ratio == pytest.approx(0.5)

    def test_p_value_computation_enough_reps(self):
        """With n>=5 and different distributions, p_value is computed and significant."""
        rng = np.random.default_rng(42)
        classical = (10.0 + rng.normal(0, 0.5, size=10)).tolist()
        quantum = (5.0 + rng.normal(0, 0.5, size=10)).tolist()
        runtimes = [1.0] * 10

        metrics = compute_comparison_metrics(
            classical_objectives=classical,
            quantum_objectives=quantum,
            classical_runtimes=runtimes,
            quantum_runtimes=runtimes,
        )
        assert metrics.p_value is not None
        assert metrics.p_value < 0.05

    def test_p_value_not_significant_for_same_dist(self):
        """With identical distributions, p_value should not be significant."""
        vals = [5.0, 5.1, 4.9, 5.05, 4.95]
        metrics = compute_comparison_metrics(
            classical_objectives=vals,
            quantum_objectives=vals,
            classical_runtimes=[1.0] * 5,
            quantum_runtimes=[1.0] * 5,
        )
        # With identical values, wilcoxon may return p=1.0 or raise,
        # either way p_value should not indicate significance
        if metrics.p_value is not None:
            assert metrics.p_value >= 0.05


# ===========================================================================
# 7. Scaling analysis utilities
# ===========================================================================


class TestScalingAnalysis:
    """Tests for scaling_analysis.py utility functions."""

    def test_generate_random_qubo_shape(self):
        """generate_random_qubo returns correct shape."""
        Q = generate_random_qubo(5, seed=42)
        assert Q.shape == (5, 5)

    def test_generate_random_qubo_reproducible(self):
        """Same seed produces same QUBO."""
        Q1 = generate_random_qubo(4, seed=123)
        Q2 = generate_random_qubo(4, seed=123)
        np.testing.assert_array_equal(Q1, Q2)

    def test_generate_maxcut_qubo_shape(self):
        """generate_maxcut_qubo returns correct shape."""
        Q = generate_maxcut_qubo(6, edge_prob=0.5, seed=42)
        assert Q.shape == (6, 6)

    def test_generate_maxcut_qubo_is_upper_triangular(self):
        """Max-Cut QUBO should be upper triangular."""
        Q = generate_maxcut_qubo(5, seed=42)
        for i in range(5):
            for j in range(i):
                assert Q[i, j] == 0.0, f"Q[{i},{j}] = {Q[i,j]} should be 0"

    def test_generate_portfolio_qubo_shape(self):
        """generate_portfolio_qubo returns correct shape."""
        n_assets = 3
        n_bits = 3
        Q = generate_portfolio_qubo(n_assets=n_assets, n_bits=n_bits, seed=42)
        expected_size = n_assets * n_bits
        assert Q.shape == (expected_size, expected_size)

    def test_compute_scaling_exponents(self):
        """compute_scaling_exponents fits power law correctly."""
        # O(2^n) scaling: runtime ~ 2^n
        sizes = [2, 4, 6, 8, 10]
        runtimes = [2**s for s in sizes]

        result = compute_scaling_exponents(sizes, runtimes)
        assert "exponent" in result
        assert "r_squared" in result
        # For exponential scaling, the log-log exponent won't be constant
        # but r_squared should still be reasonable for a power-law fit
        assert result["r_squared"] >= 0.0

    def test_compute_scaling_exponents_polynomial(self):
        """For O(n^2) scaling, exponent should be close to 2."""
        sizes = [10, 20, 40, 80, 160]
        runtimes = [s**2 for s in sizes]

        result = compute_scaling_exponents(sizes, runtimes)
        assert result["exponent"] == pytest.approx(2.0, abs=0.1)
        assert result["r_squared"] > 0.99
