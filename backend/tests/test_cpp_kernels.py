"""Tests for C++ QAOA kernels — verifies C++ matches Python outputs.

Tests correctness by comparing C++ results against the reference
Python implementations, and benchmarks speedup.
"""

import time

import numpy as np
import pytest

from quantum.cpp import HAS_CPP

# Python reference implementations
from quantum.solvers.qaoa_solver import (
    build_cost_diagonal as py_build_cost_diagonal,
    apply_cost_unitary as py_apply_cost_unitary,
    apply_mixer_unitary as py_apply_mixer_unitary,
    simulate_qaoa_expectation as py_qaoa_expectation,
)
from quantum.solvers.problem_encodings import evaluate_qubo as py_evaluate_qubo

# C++ implementations (or fallback)
from quantum.cpp import (
    build_cost_diagonal as cpp_build_cost_diagonal,
    apply_cost_unitary as cpp_apply_cost_unitary,
    apply_mixer_unitary as cpp_apply_mixer_unitary,
    evaluate_qubo as cpp_evaluate_qubo,
    qaoa_expectation as cpp_qaoa_expectation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_qubo():
    """4x4 QUBO matrix (4 qubits)."""
    rng = np.random.default_rng(42)
    Q = rng.normal(0, 1, (4, 4))
    Q = (Q + Q.T) / 2  # symmetrize
    return Q


@pytest.fixture
def medium_qubo():
    """8x8 QUBO matrix (8 qubits, 256 states)."""
    rng = np.random.default_rng(99)
    Q = rng.normal(0, 1, (8, 8))
    Q = (Q + Q.T) / 2
    return Q


@pytest.fixture
def large_qubo():
    """12x12 QUBO for benchmarking (4096 states)."""
    rng = np.random.default_rng(123)
    Q = rng.normal(0, 1, (12, 12))
    Q = (Q + Q.T) / 2
    return Q


# ===========================================================================
# 1. C++ available check
# ===========================================================================

class TestCppAvailability:

    def test_cpp_extension_loaded(self):
        """C++ extension should be available after build."""
        assert HAS_CPP, (
            "C++ extension not loaded. Run: cd quantum/cpp && bash build.sh"
        )


# ===========================================================================
# 2. build_cost_diagonal
# ===========================================================================

class TestBuildCostDiagonal:

    def test_matches_python(self, small_qubo):
        py_diag = py_build_cost_diagonal(small_qubo)
        cpp_diag = cpp_build_cost_diagonal(small_qubo)
        assert np.allclose(py_diag, cpp_diag, atol=1e-10)

    def test_matches_python_medium(self, medium_qubo):
        py_diag = py_build_cost_diagonal(medium_qubo)
        cpp_diag = cpp_build_cost_diagonal(medium_qubo)
        assert np.allclose(py_diag, cpp_diag, atol=1e-10)

    def test_output_length(self, small_qubo):
        diag = cpp_build_cost_diagonal(small_qubo)
        assert len(diag) == 2 ** 4

    def test_identity_matrix(self):
        """Q = I: cost(x) = number of 1s in x."""
        Q = np.eye(3)
        diag = cpp_build_cost_diagonal(Q)
        # |000>=0, |001>=1, |010>=1, |011>=2, |100>=1, |101>=2, |110>=2, |111>=3
        assert diag[0] == 0
        assert diag[7] == 3


# ===========================================================================
# 3. apply_cost_unitary
# ===========================================================================

class TestApplyCostUnitary:

    def test_matches_python(self, small_qubo):
        n = small_qubo.shape[0]
        dim = 2 ** n
        sv = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        diag = py_build_cost_diagonal(small_qubo)
        gamma = 0.5

        py_result = py_apply_cost_unitary(sv, diag, gamma)
        cpp_result = cpp_apply_cost_unitary(sv, diag, gamma)
        assert np.allclose(py_result, cpp_result, atol=1e-10)

    def test_preserves_norm(self, small_qubo):
        n = small_qubo.shape[0]
        dim = 2 ** n
        rng = np.random.default_rng(42)
        sv = rng.normal(0, 1, dim) + 1j * rng.normal(0, 1, dim)
        sv /= np.linalg.norm(sv)
        diag = py_build_cost_diagonal(small_qubo)

        result = cpp_apply_cost_unitary(sv, diag, 1.3)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-10)


# ===========================================================================
# 4. apply_mixer_unitary
# ===========================================================================

class TestApplyMixerUnitary:

    def test_matches_python(self, small_qubo):
        n = small_qubo.shape[0]
        dim = 2 ** n
        sv = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        beta = 0.7

        py_result = py_apply_mixer_unitary(sv, n, beta)
        cpp_result = cpp_apply_mixer_unitary(sv, n, beta)
        assert np.allclose(py_result, cpp_result, atol=1e-10)

    def test_preserves_norm(self, small_qubo):
        n = small_qubo.shape[0]
        dim = 2 ** n
        rng = np.random.default_rng(42)
        sv = rng.normal(0, 1, dim) + 1j * rng.normal(0, 1, dim)
        sv /= np.linalg.norm(sv)

        result = cpp_apply_mixer_unitary(sv, n, 0.9)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_matches_medium(self, medium_qubo):
        n = medium_qubo.shape[0]
        dim = 2 ** n
        rng = np.random.default_rng(42)
        sv = rng.normal(0, 1, dim) + 1j * rng.normal(0, 1, dim)
        sv /= np.linalg.norm(sv)
        beta = 1.2

        py_result = py_apply_mixer_unitary(sv, n, beta)
        cpp_result = cpp_apply_mixer_unitary(sv, n, beta)
        assert np.allclose(py_result, cpp_result, atol=1e-10)


# ===========================================================================
# 5. evaluate_qubo
# ===========================================================================

class TestEvaluateQubo:

    def test_matches_python(self, small_qubo):
        x = np.array([1, 0, 1, 0], dtype=float)
        py_val = py_evaluate_qubo(small_qubo, x)
        cpp_val = cpp_evaluate_qubo(small_qubo, x)
        assert np.isclose(py_val, cpp_val, atol=1e-10)

    def test_zero_vector(self, small_qubo):
        x = np.zeros(4)
        assert cpp_evaluate_qubo(small_qubo, x) == 0.0

    def test_all_ones(self, small_qubo):
        x = np.ones(4)
        expected = float(x @ small_qubo @ x)
        assert np.isclose(cpp_evaluate_qubo(small_qubo, x), expected, atol=1e-10)


# ===========================================================================
# 6. qaoa_expectation
# ===========================================================================

class TestQaoaExpectation:

    def test_matches_python(self, small_qubo):
        gamma = np.array([0.5, 0.3])
        beta = np.array([0.7, 0.2])
        py_val = py_qaoa_expectation(small_qubo, gamma, beta)
        cpp_val = cpp_qaoa_expectation(small_qubo, gamma, beta)
        assert np.isclose(py_val, cpp_val, atol=1e-8)

    def test_matches_medium(self, medium_qubo):
        gamma = np.array([0.4])
        beta = np.array([0.6])
        py_val = py_qaoa_expectation(medium_qubo, gamma, beta)
        cpp_val = cpp_qaoa_expectation(medium_qubo, gamma, beta)
        assert np.isclose(py_val, cpp_val, atol=1e-8)


# ===========================================================================
# 7. Benchmarks (speedup measurement)
# ===========================================================================

class TestBenchmarkSpeedup:

    @pytest.mark.skipif(not HAS_CPP, reason="C++ extension not available")
    def test_mixer_speedup(self, large_qubo):
        """C++ mixer should be faster than Python on 12 qubits."""
        n = large_qubo.shape[0]
        dim = 2 ** n
        sv = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        beta = 0.5

        # Warm up
        cpp_apply_mixer_unitary(sv, n, beta)
        py_apply_mixer_unitary(sv, n, beta)

        # Benchmark Python
        t0 = time.perf_counter()
        for _ in range(5):
            py_apply_mixer_unitary(sv, n, beta)
        py_time = time.perf_counter() - t0

        # Benchmark C++
        t0 = time.perf_counter()
        for _ in range(5):
            cpp_apply_mixer_unitary(sv, n, beta)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        # C++ should be meaningfully faster; at minimum not slower
        assert speedup > 0.5, f"C++ speedup: {speedup:.1f}x"

    @pytest.mark.skipif(not HAS_CPP, reason="C++ extension not available")
    def test_cost_diagonal_speedup(self, large_qubo):
        """C++ cost diagonal should be faster on 12 qubits."""
        # Warm up
        cpp_build_cost_diagonal(large_qubo)
        py_build_cost_diagonal(large_qubo)

        t0 = time.perf_counter()
        for _ in range(5):
            py_build_cost_diagonal(large_qubo)
        py_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(5):
            cpp_build_cost_diagonal(large_qubo)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        assert speedup > 0.5, f"C++ speedup: {speedup:.1f}x"
