"""Tests for QAOA solver — written FIRST per TDD.

Tests the numpy-based statevector QAOA simulator that solves QUBO problems
via the Quantum Approximate Optimization Algorithm. Every test here was
written before the implementation exists.
"""

import numpy as np
import pytest
import time

from quantum.solvers.classical_solvers import SolverResult, solve_brute_force_qubo
from quantum.solvers.problem_encodings import (
    portfolio_to_qubo,
    decode_binary_weights,
    evaluate_qubo,
    maxcut_qubo,
)
from quantum.solvers.qaoa_solver import (
    QAOASolver,
    build_cost_diagonal,
    apply_cost_unitary,
    apply_mixer_unitary,
    simulate_qaoa_expectation,
    sample_from_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_2var_qubo():
    """Simple 2-variable QUBO with known optimal.

    Q = [[-1,  2],
         [ 0, -1]]

    Enumerate all 2^2 = 4 solutions (x^T Q x for binary x):
        x=[0,0]: 0
        x=[1,0]: -1
        x=[0,1]: -1
        x=[1,1]: -1 + 2 + -1 = 0

    Minimum = -1 at x=[1,0] or x=[0,1].
    """
    return np.array([[-1.0, 2.0],
                     [0.0, -1.0]])


@pytest.fixture
def triangle_maxcut_qubo():
    """Max-Cut QUBO for a triangle graph (3 nodes, all edges weight 1).

    Using the maxcut_qubo encoding from problem_encodings:
    Q_ii = -sum_j w_ij / 2 = -1.0  (each node has 2 edges)
    Q_ij = w_ij / 2 = 0.5          (for i < j)

    Brute force:
        x=[0,0,0]: 0
        x=[1,0,0]: -1
        x=[0,1,0]: -1
        x=[0,0,1]: -1
        x=[1,1,0]: -1
        x=[1,0,1]: -1
        x=[0,1,1]: -1
        x=[1,1,1]: 0

    Minimum = -1.0 at any partition into 1 vs 2 or 2 vs 1 nodes.
    Max cut value = 2 edges (for triangle, any 1-vs-2 partition cuts 2 edges).
    """
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)
    return maxcut_qubo(adj)


@pytest.fixture
def four_var_qubo():
    """4-variable QUBO for more thorough testing.

    Q = [[-5,  2,  4,  0],
         [ 0, -3,  1,  0],
         [ 0,  0, -8,  5],
         [ 0,  0,  0, -6]]

    (upper triangular form)
    """
    Q = np.array([
        [-5, 2, 4, 0],
        [0, -3, 1, 0],
        [0, 0, -8, 5],
        [0, 0, 0, -6],
    ], dtype=float)
    return Q


@pytest.fixture
def two_asset_portfolio():
    """Simple 2-asset portfolio problem."""
    mu = np.array([0.10, 0.20])
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    return mu, cov


# ===========================================================================
# Tests: QAOASolver.__init__() — configuration
# ===========================================================================

class TestQAOASolverInit:

    def test_default_config_values(self):
        solver = QAOASolver()
        assert solver._config["n_layers"] == 3
        assert solver._config["optimizer"] == "COBYLA"
        assert solver._config["maxiter"] == 500
        assert solver._config["n_shots"] == 4096
        assert solver._config["seed"] is None

    def test_custom_config_overrides(self):
        custom = {
            "n_layers": 5,
            "optimizer": "Nelder-Mead",
            "maxiter": 1000,
            "seed": 42,
        }
        solver = QAOASolver(config=custom)
        assert solver._config["n_layers"] == 5
        assert solver._config["optimizer"] == "Nelder-Mead"
        assert solver._config["maxiter"] == 1000
        assert solver._config["seed"] == 42
        # n_shots should still be default
        assert solver._config["n_shots"] == 4096

    def test_empty_config_uses_defaults(self):
        solver = QAOASolver(config={})
        assert solver._config["n_layers"] == 3
        assert solver._config["optimizer"] == "COBYLA"

    def test_partial_config_merges_with_defaults(self):
        solver = QAOASolver(config={"n_layers": 7})
        assert solver._config["n_layers"] == 7
        assert solver._config["optimizer"] == "COBYLA"
        assert solver._config["maxiter"] == 500


# ===========================================================================
# Tests: build_cost_diagonal
# ===========================================================================

class TestBuildCostDiagonal:

    def test_correct_length(self, simple_2var_qubo):
        diag = build_cost_diagonal(simple_2var_qubo)
        assert len(diag) == 4  # 2^2

    def test_correct_values_2var(self, simple_2var_qubo):
        """Verify each diagonal element matches x^T Q x."""
        Q = simple_2var_qubo
        diag = build_cost_diagonal(Q)
        # Enumerate manually
        for i in range(4):
            x = np.array([(i >> bit) & 1 for bit in range(2)], dtype=float)
            expected = float(x @ Q @ x)
            assert np.isclose(diag[i], expected, atol=1e-12), (
                f"Mismatch at i={i}: diag={diag[i]}, expected={expected}"
            )

    def test_correct_values_3var(self, triangle_maxcut_qubo):
        Q = triangle_maxcut_qubo
        n = Q.shape[0]
        diag = build_cost_diagonal(Q)
        assert len(diag) == 8  # 2^3

        for i in range(8):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=float)
            expected = float(x @ Q @ x)
            assert np.isclose(diag[i], expected, atol=1e-12), (
                f"Mismatch at i={i}: diag={diag[i]}, expected={expected}"
            )

    def test_single_variable(self):
        Q = np.array([[-3.0]])
        diag = build_cost_diagonal(Q)
        assert len(diag) == 2
        assert np.isclose(diag[0], 0.0)   # x=0: 0
        assert np.isclose(diag[1], -3.0)  # x=1: -3


# ===========================================================================
# Tests: apply_cost_unitary
# ===========================================================================

class TestApplyCostUnitary:

    def test_preserves_norm(self, simple_2var_qubo):
        """Cost unitary is diagonal phase rotation — should preserve norm."""
        diag = build_cost_diagonal(simple_2var_qubo)
        n = 4
        sv = np.ones(n, dtype=complex) / np.sqrt(n)
        gamma = 0.5
        result = apply_cost_unitary(sv, diag, gamma)
        assert np.isclose(np.sum(np.abs(result) ** 2), 1.0, atol=1e-12)

    def test_gamma_zero_is_identity(self, simple_2var_qubo):
        """gamma=0 should leave statevector unchanged."""
        diag = build_cost_diagonal(simple_2var_qubo)
        n = 4
        sv = np.array([0.5, 0.5, 0.3, np.sqrt(1 - 0.25 - 0.25 - 0.09)], dtype=complex)
        result = apply_cost_unitary(sv, diag, 0.0)
        np.testing.assert_allclose(result, sv, atol=1e-12)

    def test_applies_correct_phase(self):
        """For a simple diagonal, verify the phase rotation."""
        diag = np.array([1.0, -1.0])
        sv = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
        gamma = np.pi / 4
        result = apply_cost_unitary(sv, diag, gamma)

        expected_0 = sv[0] * np.exp(-1j * gamma * diag[0])
        expected_1 = sv[1] * np.exp(-1j * gamma * diag[1])
        assert np.isclose(result[0], expected_0, atol=1e-12)
        assert np.isclose(result[1], expected_1, atol=1e-12)


# ===========================================================================
# Tests: apply_mixer_unitary
# ===========================================================================

class TestApplyMixerUnitary:

    def test_preserves_norm(self):
        """Mixer unitary should preserve norm."""
        n_qubits = 3
        n = 2 ** n_qubits
        sv = np.ones(n, dtype=complex) / np.sqrt(n)
        beta = 0.7
        result = apply_mixer_unitary(sv, n_qubits, beta)
        assert np.isclose(np.sum(np.abs(result) ** 2), 1.0, atol=1e-12)

    def test_beta_zero_is_identity(self):
        """beta=0 should leave statevector unchanged."""
        n_qubits = 2
        n = 2 ** n_qubits
        sv = np.array([0.5, 0.3, 0.4, np.sqrt(1 - 0.25 - 0.09 - 0.16)], dtype=complex)
        result = apply_mixer_unitary(sv, n_qubits, 0.0)
        np.testing.assert_allclose(result, sv, atol=1e-12)

    def test_single_qubit_rotation(self):
        """For 1 qubit, mixer is exp(-i*beta*X).

        X = [[0, 1], [1, 0]]
        exp(-i*beta*X) = [[cos(beta), -i*sin(beta)],
                          [-i*sin(beta), cos(beta)]]

        Apply to |0> = [1, 0]:
        result = [cos(beta), -i*sin(beta)]
        """
        sv = np.array([1.0, 0.0], dtype=complex)
        beta = np.pi / 6
        result = apply_mixer_unitary(sv, 1, beta)
        expected = np.array([np.cos(beta), -1j * np.sin(beta)])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_beta_pi_half_flips_all(self):
        """beta=pi/2 should flip |0...0> to i^n |1...1>."""
        n_qubits = 2
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0  # |00>
        result = apply_mixer_unitary(sv, n_qubits, np.pi / 2)
        # exp(-i * pi/2 * X) applied to each qubit flips 0->1 with phase -i
        # For 2 qubits: |00> -> (-i)^2 |11> = -1 * |11>
        expected = np.zeros(4, dtype=complex)
        expected[3] = (-1j) ** n_qubits  # |11> = index 3
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ===========================================================================
# Tests: simulate_qaoa_expectation
# ===========================================================================

class TestSimulateQaoaExpectation:

    def test_returns_real_value(self, simple_2var_qubo):
        Q = simple_2var_qubo
        gamma = np.array([0.5])
        beta = np.array([0.3])
        expectation = simulate_qaoa_expectation(Q, gamma, beta)
        assert isinstance(expectation, float)
        # Should be a real number (no imaginary part)
        assert np.isfinite(expectation)

    def test_known_small_qubo(self):
        """For a single-variable QUBO Q=[[-1]], the cost diagonal is [0, -1].
        Starting from |+>, with gamma and beta = 0, expectation = -0.5.
        """
        Q = np.array([[-1.0]])
        gamma = np.array([0.0])
        beta = np.array([0.0])
        expectation = simulate_qaoa_expectation(Q, gamma, beta)
        # |+> = [1/sqrt(2), 1/sqrt(2)], cost = [0, -1]
        # <C> = 0.5 * 0 + 0.5 * (-1) = -0.5
        assert np.isclose(expectation, -0.5, atol=1e-10)

    def test_random_angles_worse_than_optimized(self, simple_2var_qubo):
        """Optimized angles should yield better (lower) expectation than random."""
        Q = simple_2var_qubo
        rng = np.random.default_rng(42)

        # Random angles — average over several
        random_expectations = []
        for _ in range(20):
            gamma = rng.uniform(0, 2 * np.pi, size=1)
            beta = rng.uniform(0, np.pi, size=1)
            exp_val = simulate_qaoa_expectation(Q, gamma, beta)
            random_expectations.append(exp_val)
        avg_random = np.mean(random_expectations)

        # Optimized: use the solver to find good angles
        solver = QAOASolver(config={"n_layers": 1, "maxiter": 200, "seed": 42})
        result = solver.solve(Q)
        # The optimized objective should be at least as good as average random
        assert result.objective_value <= avg_random + 0.1, (
            f"Optimized {result.objective_value} worse than avg random {avg_random}"
        )

    def test_multiple_layers(self, simple_2var_qubo):
        """Test that simulate_qaoa_expectation works with multiple layers."""
        Q = simple_2var_qubo
        p = 3
        gamma = np.array([0.1, 0.2, 0.3])
        beta = np.array([0.4, 0.5, 0.6])
        expectation = simulate_qaoa_expectation(Q, gamma, beta)
        assert isinstance(expectation, float)
        assert np.isfinite(expectation)


# ===========================================================================
# Tests: sample_from_state
# ===========================================================================

class TestSampleFromState:

    def test_samples_from_known_state(self):
        """For a state concentrated on |1>, all samples should be 1."""
        sv = np.array([0.0, 1.0], dtype=complex)
        samples = sample_from_state(sv, n_shots=100, seed=42)
        assert len(samples) == 100
        assert np.all(samples == 1)

    def test_uniform_state_samples_all_bitstrings(self):
        """For |+>^2 (uniform), we should see all 4 bitstrings."""
        sv = np.ones(4, dtype=complex) / 2.0
        samples = sample_from_state(sv, n_shots=10000, seed=42)
        unique = set(samples.tolist())
        assert unique == {0, 1, 2, 3}, f"Expected all 4 bitstrings, got {unique}"

    def test_reproducibility_with_seed(self):
        sv = np.array([0.6, 0.0, 0.8, 0.0], dtype=complex)
        sv = sv / np.linalg.norm(sv)
        s1 = sample_from_state(sv, n_shots=50, seed=123)
        s2 = sample_from_state(sv, n_shots=50, seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_returns_integer_indices(self):
        sv = np.ones(8, dtype=complex) / np.sqrt(8)
        samples = sample_from_state(sv, n_shots=10, seed=0)
        assert samples.dtype in (np.int64, np.int32, np.intp)
        assert np.all(samples >= 0)
        assert np.all(samples < 8)


# ===========================================================================
# Tests: QAOASolver.solve() — main solve method
# ===========================================================================

class TestQAOASolverSolve:

    def test_solve_2var_finds_optimal_or_near_optimal(self, simple_2var_qubo):
        """QAOA should find optimal or near-optimal for a 2-variable QUBO."""
        Q = simple_2var_qubo
        bf = solve_brute_force_qubo(Q)

        solver = QAOASolver(config={"n_layers": 3, "maxiter": 500, "seed": 42})
        result = solver.solve(Q)

        # QAOA should find a solution within 20% of optimal
        # (for such a tiny problem, it should usually find the exact optimum)
        assert result.objective_value <= bf.objective_value * 0.8, (
            f"QAOA found {result.objective_value}, optimal is {bf.objective_value}"
        )

    def test_solve_maxcut_triangle_approximation_ratio(self, triangle_maxcut_qubo):
        """QAOA on Max-Cut triangle should achieve approximation ratio > 0.6."""
        Q = triangle_maxcut_qubo
        bf = solve_brute_force_qubo(Q)
        optimal_val = bf.objective_value  # Should be -1.0

        solver = QAOASolver(config={"n_layers": 3, "maxiter": 500, "seed": 42})
        result = solver.solve(Q)

        # Approximation ratio for minimization: optimal / found
        # Both values are negative, so ratio = qaoa_val / optimal_val
        # A ratio > 0.6 means QAOA found at least 60% of the optimal cut
        if optimal_val < 0:
            approx_ratio = result.objective_value / optimal_val
        else:
            approx_ratio = 1.0  # trivial case

        assert approx_ratio > 0.6, (
            f"QAOA approx ratio = {approx_ratio:.3f} (found {result.objective_value}, "
            f"optimal {optimal_val}), expected > 0.6"
        )

    def test_returns_solver_result_with_proper_fields(self, simple_2var_qubo):
        solver = QAOASolver(config={"n_layers": 1, "seed": 0})
        result = solver.solve(simple_2var_qubo)

        assert isinstance(result, SolverResult)
        assert result.weights is not None
        assert len(result.weights) == simple_2var_qubo.shape[0]
        assert isinstance(result.objective_value, float)
        assert result.method == "qaoa"
        assert result.converged is True or result.converged is False

    def test_runtime_is_recorded(self, simple_2var_qubo):
        solver = QAOASolver(config={"n_layers": 1, "seed": 0})
        result = solver.solve(simple_2var_qubo)
        assert result.runtime_ms > 0, "Runtime should be positive"

    def test_solution_is_binary(self, simple_2var_qubo):
        solver = QAOASolver(config={"n_layers": 2, "seed": 42})
        result = solver.solve(simple_2var_qubo)
        for w in result.weights:
            assert w in (0.0, 1.0), f"Non-binary value in solution: {w}"

    def test_different_layers_p1_vs_p3(self, simple_2var_qubo):
        """More layers should give same or better results on average."""
        Q = simple_2var_qubo
        results_p1 = []
        results_p3 = []
        for seed in range(10):
            solver_p1 = QAOASolver(config={"n_layers": 1, "maxiter": 300, "seed": seed})
            solver_p3 = QAOASolver(config={"n_layers": 3, "maxiter": 300, "seed": seed})
            results_p1.append(solver_p1.solve(Q).objective_value)
            results_p3.append(solver_p3.solve(Q).objective_value)

        avg_p1 = np.mean(results_p1)
        avg_p3 = np.mean(results_p3)
        # p=3 should do at least as well as p=1 on average (with tolerance)
        assert avg_p3 <= avg_p1 + 0.2, (
            f"p=3 (avg={avg_p3:.3f}) should be <= p=1 (avg={avg_p1:.3f})"
        )

    def test_reproducibility_with_fixed_seed(self, simple_2var_qubo):
        solver1 = QAOASolver(config={"n_layers": 2, "seed": 99})
        solver2 = QAOASolver(config={"n_layers": 2, "seed": 99})
        r1 = solver1.solve(simple_2var_qubo)
        r2 = solver2.solve(simple_2var_qubo)

        np.testing.assert_array_equal(r1.weights, r2.weights)
        assert r1.objective_value == r2.objective_value

    def test_solve_4var_qubo(self, four_var_qubo):
        """QAOA should find a reasonable solution for a 4-variable QUBO."""
        Q = four_var_qubo
        bf = solve_brute_force_qubo(Q)

        solver = QAOASolver(config={"n_layers": 3, "maxiter": 500, "seed": 42})
        result = solver.solve(Q)

        # Should at least find a solution no worse than zero (trivial x=0 solution)
        assert result.objective_value <= 0.0, (
            f"QAOA should find negative objective, got {result.objective_value}"
        )

    def test_metadata_contains_optimal_params(self, simple_2var_qubo):
        solver = QAOASolver(config={"n_layers": 2, "seed": 42})
        result = solver.solve(simple_2var_qubo)
        assert "optimal_gamma" in result.metadata
        assert "optimal_beta" in result.metadata
        assert len(result.metadata["optimal_gamma"]) == 2
        assert len(result.metadata["optimal_beta"]) == 2


# ===========================================================================
# Tests: QAOASolver.solve_portfolio() — convenience method
# ===========================================================================

class TestQAOASolverSolvePortfolio:

    def test_returns_valid_weights(self, two_asset_portfolio):
        mu, cov = two_asset_portfolio
        solver = QAOASolver(config={"n_layers": 2, "maxiter": 300, "seed": 42})
        result = solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=0.12,
            max_weight=0.50,
            n_bits=3,
        )
        assert isinstance(result, SolverResult)
        assert result.method == "qaoa_portfolio"

    def test_weights_are_non_negative(self, two_asset_portfolio):
        mu, cov = two_asset_portfolio
        solver = QAOASolver(config={"n_layers": 2, "seed": 42})
        result = solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=0.12,
            max_weight=0.50,
            n_bits=3,
        )
        # Decoded portfolio weights should be >= 0
        decoded_weights = result.metadata.get("portfolio_weights")
        assert decoded_weights is not None, "Expected portfolio_weights in metadata"
        assert np.all(np.array(decoded_weights) >= -1e-10), (
            f"Negative weights found: {decoded_weights}"
        )

    def test_returns_comparison_with_brute_force(self, two_asset_portfolio):
        mu, cov = two_asset_portfolio
        solver = QAOASolver(config={"n_layers": 2, "seed": 42})
        result = solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=0.12,
            max_weight=0.50,
            n_bits=3,
        )
        assert "brute_force_objective" in result.metadata, (
            "Expected brute_force_objective comparison in metadata"
        )
        assert "approximation_ratio" in result.metadata, (
            "Expected approximation_ratio in metadata"
        )

    def test_portfolio_weights_bounded_by_max_weight(self, two_asset_portfolio):
        mu, cov = two_asset_portfolio
        max_w = 0.50
        solver = QAOASolver(config={"n_layers": 2, "seed": 42})
        result = solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=0.12,
            max_weight=max_w,
            n_bits=3,
        )
        decoded = np.array(result.metadata["portfolio_weights"])
        assert np.all(decoded <= max_w + 1e-10), (
            f"Weight exceeds max_weight={max_w}: {decoded}"
        )


# ===========================================================================
# Tests: build_qaoa_circuit (via simulate_qaoa_expectation)
# ===========================================================================

class TestQAOACircuitProperties:
    """These tests verify QAOA circuit properties indirectly through
    the expectation value function, since we use statevector simulation
    rather than explicit circuit objects."""

    def test_circuit_respects_qubit_count(self, simple_2var_qubo):
        """The cost diagonal should have 2^n entries for n-qubit problem."""
        Q = simple_2var_qubo
        diag = build_cost_diagonal(Q)
        assert len(diag) == 2 ** Q.shape[0]

    def test_circuit_with_3_qubits(self, triangle_maxcut_qubo):
        Q = triangle_maxcut_qubo
        diag = build_cost_diagonal(Q)
        assert len(diag) == 2 ** 3  # 3-qubit problem

    def test_gamma_beta_length_matches_layers(self, simple_2var_qubo):
        """Expectation function should work when gamma/beta length matches p."""
        Q = simple_2var_qubo
        for p in [1, 2, 3, 5]:
            gamma = np.zeros(p)
            beta = np.zeros(p)
            val = simulate_qaoa_expectation(Q, gamma, beta)
            assert np.isfinite(val), f"Non-finite expectation for p={p}"

    def test_gamma_beta_length_mismatch_raises(self, simple_2var_qubo):
        """gamma and beta must have the same length."""
        Q = simple_2var_qubo
        gamma = np.array([0.1, 0.2])
        beta = np.array([0.3])
        with pytest.raises(ValueError, match="gamma and beta must have the same length"):
            simulate_qaoa_expectation(Q, gamma, beta)
