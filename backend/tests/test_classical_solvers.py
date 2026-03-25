"""Tests for classical optimization solvers — written FIRST per TDD.

These tests define the contract that classical_solvers.py must satisfy.
Every quantum method will be benchmarked against these solvers, so
correctness here is critical.
"""

import numpy as np
import pytest
import time

from quantum.solvers.classical_solvers import (
    SolverResult,
    solve_markowitz_cvxpy,
    solve_markowitz_scipy,
    solve_brute_force_qubo,
    simulated_annealing_qubo,
    greedy_qubo,
    efficient_frontier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_asset_problem():
    """Standard 3-asset test problem with realistic parameters."""
    expected_returns = np.array([0.10, 0.15, 0.20])
    # Positive-definite covariance matrix
    covariance = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.009],
        [0.002, 0.009, 0.16],
    ])
    target_return = 0.12
    return expected_returns, covariance, target_return


@pytest.fixture
def two_asset_unconstrained():
    """2-asset problem with known analytical solution.

    For 2 assets with no upper-bound constraint:
    minimize  w^T Sigma w
    s.t.      mu^T w = target, sum(w) = 1, w >= 0

    With mu = [0.10, 0.20], Sigma = diag(0.04, 0.16), target = 0.15:
    The solution is w = [0.5, 0.5] (by symmetry of risk-return trade-off).
    Portfolio variance = 0.25*0.04 + 0.25*0.16 = 0.05.
    """
    expected_returns = np.array([0.10, 0.20])
    covariance = np.array([
        [0.04, 0.0],
        [0.0, 0.16],
    ])
    target_return = 0.15
    return expected_returns, covariance, target_return


@pytest.fixture
def small_qubo_4var():
    """4-variable QUBO with known optimal solution.

    Q = [[-5,  2,  4,  0],
         [ 2, -3,  1,  0],
         [ 4,  1, -8,  5],
         [ 0,  0,  5, -6]]

    The objective x^T Q x for binary x can be enumerated (2^4 = 16).
    We precompute the optimal to verify.
    """
    Q = np.array([
        [-5, 2, 4, 0],
        [2, -3, 1, 0],
        [4, 1, -8, 5],
        [0, 0, 5, -6],
    ], dtype=float)
    return Q


@pytest.fixture
def maxcut_qubo():
    """Max-Cut QUBO for a simple triangle graph.

    For Max-Cut, Q_{ij} = -w_{ij} for edges, Q_{ii} = sum of edge weights
    incident on node i. We want to MINIMIZE x^T Q x.

    Triangle with unit weights:
    Q = [[ 2, -1, -1],
         [-1,  2, -1],
         [-1, -1,  2]]

    Optimal cuts: any partition into 1 vs 2 nodes.
    Value of max cut = 2 (2 edges cut).
    Minimum of QUBO = -2 (at optimal cut solutions like [1,0,0], [0,1,1], etc.)

    Actually for Max-Cut QUBO formulation:
    min x^T Q x where Q_ii = degree_i, Q_ij = -2*w_ij for edges.
    Wait, let's use the standard formulation directly.

    Standard Max-Cut QUBO: maximize sum_{(i,j) in E} w_ij * (x_i + x_j - 2*x_i*x_j)
    = maximize sum_{(i,j)} w_ij * x_i + w_ij * x_j - 2*w_ij * x_i * x_j

    As minimization of x^T Q x:
    Q_ii = -sum_{j: (i,j) in E} w_ij
    Q_ij = w_ij  (for i != j, if (i,j) is an edge)

    Triangle (all weights = 1):
    Q = [[-2,  1,  1],
         [ 1, -2,  1],
         [ 1,  1, -2]]

    Brute force (x^T Q x):
    x=[0,0,0]: 0
    x=[1,0,0]: -2
    x=[0,1,0]: -2
    x=[0,0,1]: -2
    x=[1,1,0]: -2
    x=[1,0,1]: -2
    x=[0,1,1]: -2
    x=[1,1,1]: 0

    Minimum = -2 at any single-node or two-node partition.
    """
    Q = np.array([
        [-2, 1, 1],
        [1, -2, 1],
        [1, 1, -2],
    ], dtype=float)
    return Q


# ===========================================================================
# Tests: solve_markowitz_cvxpy
# ===========================================================================

class TestSolveMarkowitzCvxpy:

    def test_weights_sum_to_one(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0)
        assert isinstance(result, SolverResult)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_long_only_constraint(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0, long_only=True)
        assert np.all(result.weights >= -1e-8), (
            f"Long-only violated: {result.weights}"
        )

    def test_max_weight_constraint(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        max_w = 0.5
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=max_w)
        assert np.all(result.weights <= max_w + 1e-6), (
            f"Max weight {max_w} violated: {result.weights}"
        )

    def test_portfolio_return_meets_target(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0)
        portfolio_return = mu @ result.weights
        assert portfolio_return >= target - 1e-6, (
            f"Target return {target} not met: got {portfolio_return}"
        )

    def test_known_analytical_solution(self, two_asset_unconstrained):
        mu, cov, target = two_asset_unconstrained
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0, long_only=True)
        expected_weights = np.array([0.5, 0.5])
        np.testing.assert_allclose(result.weights, expected_weights, atol=1e-4)

    def test_objective_is_positive(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0)
        assert result.objective_value > 0, (
            f"Variance should be positive, got {result.objective_value}"
        )

    def test_result_has_timing_info(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0)
        assert result.runtime_ms >= 0
        assert result.method == "markowitz_cvxpy"

    def test_converged_flag(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_cvxpy(mu, cov, target, max_weight=1.0)
        assert result.converged is True


# ===========================================================================
# Tests: solve_markowitz_scipy
# ===========================================================================

class TestSolveMarkowitzScipy:

    def test_weights_sum_to_one(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0)
        assert isinstance(result, SolverResult)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_long_only_constraint(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0, long_only=True)
        assert np.all(result.weights >= -1e-6), (
            f"Long-only violated: {result.weights}"
        )

    def test_max_weight_constraint(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        max_w = 0.5
        result = solve_markowitz_scipy(mu, cov, target, max_weight=max_w)
        assert np.all(result.weights <= max_w + 1e-6), (
            f"Max weight {max_w} violated: {result.weights}"
        )

    def test_portfolio_return_meets_target(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0)
        portfolio_return = mu @ result.weights
        assert portfolio_return >= target - 1e-6, (
            f"Target return {target} not met: got {portfolio_return}"
        )

    def test_known_analytical_solution(self, two_asset_unconstrained):
        mu, cov, target = two_asset_unconstrained
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0, long_only=True)
        expected_weights = np.array([0.5, 0.5])
        np.testing.assert_allclose(result.weights, expected_weights, atol=1e-3)

    def test_objective_is_positive(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0)
        assert result.objective_value > 0

    def test_results_close_to_cvxpy(self, three_asset_problem):
        """Scipy and CVXPY should agree within tolerance on the same problem."""
        mu, cov, target = three_asset_problem
        max_w = 0.6
        cvxpy_result = solve_markowitz_cvxpy(mu, cov, target, max_weight=max_w)
        scipy_result = solve_markowitz_scipy(mu, cov, target, max_weight=max_w)
        np.testing.assert_allclose(
            cvxpy_result.weights, scipy_result.weights, atol=1e-3,
            err_msg="CVXPY and SciPy solutions diverge",
        )
        assert abs(cvxpy_result.objective_value - scipy_result.objective_value) < 1e-4

    def test_result_metadata(self, three_asset_problem):
        mu, cov, target = three_asset_problem
        result = solve_markowitz_scipy(mu, cov, target, max_weight=1.0)
        assert result.method == "markowitz_scipy"
        assert result.runtime_ms >= 0


# ===========================================================================
# Tests: solve_brute_force_qubo
# ===========================================================================

class TestBruteForceQubo:

    def test_finds_true_minimum_4var(self, small_qubo_4var):
        Q = small_qubo_4var
        result = solve_brute_force_qubo(Q)

        # Verify by manually computing all 2^4 = 16 solutions
        n = Q.shape[0]
        best_val = float("inf")
        best_x = None
        for i in range(2 ** n):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=float)
            val = x @ Q @ x
            if val < best_val:
                best_val = val
                best_x = x.copy()

        assert np.isclose(result.objective_value, best_val, atol=1e-10), (
            f"Brute force returned {result.objective_value}, expected {best_val}"
        )
        # The solution vector should also yield the optimal value
        assert np.isclose(result.weights @ Q @ result.weights, best_val, atol=1e-10)

    def test_maxcut_correct_value(self, maxcut_qubo):
        Q = maxcut_qubo
        result = solve_brute_force_qubo(Q)
        # Minimum should be -2 (max cut of triangle = 2 edges)
        assert np.isclose(result.objective_value, -2.0, atol=1e-10), (
            f"Max-Cut minimum should be -2, got {result.objective_value}"
        )

    def test_returns_solution_and_objective(self, small_qubo_4var):
        result = solve_brute_force_qubo(small_qubo_4var)
        assert isinstance(result, SolverResult)
        assert result.weights is not None
        assert result.objective_value is not None
        assert result.method == "brute_force_qubo"

    def test_solution_is_binary(self, small_qubo_4var):
        result = solve_brute_force_qubo(small_qubo_4var)
        for w in result.weights:
            assert w in (0.0, 1.0), f"Non-binary value in solution: {w}"

    def test_trivial_qubo(self):
        """Single-variable QUBO: Q = [[-1]]. Optimum is x=[1], val=-1."""
        Q = np.array([[-1.0]])
        result = solve_brute_force_qubo(Q)
        assert np.isclose(result.objective_value, -1.0)
        assert np.isclose(result.weights[0], 1.0)


# ===========================================================================
# Tests: simulated_annealing_qubo
# ===========================================================================

class TestSimulatedAnnealingQubo:

    def test_finds_known_optimum_small(self, small_qubo_4var):
        """SA should find the optimum of a small QUBO reliably."""
        Q = small_qubo_4var
        bf_result = solve_brute_force_qubo(Q)

        # Run SA with enough iterations to converge on small instance
        sa_result = simulated_annealing_qubo(
            Q, n_iterations=50000, seed=42,
        )
        assert sa_result.objective_value <= bf_result.objective_value + 1e-6, (
            f"SA found {sa_result.objective_value}, optimal is {bf_result.objective_value}"
        )

    def test_quality_improves_with_iterations(self, small_qubo_4var):
        """More iterations should yield same or better solutions."""
        Q = small_qubo_4var
        results_short = []
        results_long = []
        for seed in range(10):
            short = simulated_annealing_qubo(Q, n_iterations=100, seed=seed)
            long = simulated_annealing_qubo(Q, n_iterations=50000, seed=seed)
            results_short.append(short.objective_value)
            results_long.append(long.objective_value)

        avg_short = np.mean(results_short)
        avg_long = np.mean(results_long)
        assert avg_long <= avg_short + 1e-6, (
            f"More iterations should help: avg_short={avg_short}, avg_long={avg_long}"
        )

    def test_reproducibility_with_seed(self, small_qubo_4var):
        Q = small_qubo_4var
        r1 = simulated_annealing_qubo(Q, n_iterations=5000, seed=123)
        r2 = simulated_annealing_qubo(Q, n_iterations=5000, seed=123)
        np.testing.assert_array_equal(r1.weights, r2.weights)
        assert r1.objective_value == r2.objective_value

    def test_returns_valid_binary_vector(self, small_qubo_4var):
        Q = small_qubo_4var
        result = simulated_annealing_qubo(Q, n_iterations=1000, seed=0)
        assert len(result.weights) == Q.shape[0]
        for w in result.weights:
            assert w in (0.0, 1.0), f"Non-binary value: {w}"

    def test_result_metadata(self, small_qubo_4var):
        result = simulated_annealing_qubo(small_qubo_4var, n_iterations=100, seed=0)
        assert result.method == "simulated_annealing_qubo"
        assert result.runtime_ms >= 0
        assert result.iterations == 100


# ===========================================================================
# Tests: greedy_qubo
# ===========================================================================

class TestGreedyQubo:

    def test_finds_reasonable_solution(self, small_qubo_4var):
        """Greedy should find a decent solution (not necessarily optimal)."""
        Q = small_qubo_4var
        bf_result = solve_brute_force_qubo(Q)
        greedy_result = greedy_qubo(Q)

        # Greedy is not guaranteed optimal, but on small instances should be close
        assert greedy_result.objective_value <= 0, (
            "Greedy should find at least a negative-valued solution on this QUBO"
        )

    def test_returns_valid_binary_vector(self, small_qubo_4var):
        result = greedy_qubo(small_qubo_4var)
        assert len(result.weights) == small_qubo_4var.shape[0]
        for w in result.weights:
            assert w in (0.0, 1.0), f"Non-binary value: {w}"

    def test_bounded_runtime(self, small_qubo_4var):
        """Greedy should complete quickly (not hang)."""
        start = time.perf_counter()
        greedy_qubo(small_qubo_4var)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1000, f"Greedy took too long: {elapsed_ms:.1f} ms"

    def test_result_metadata(self, small_qubo_4var):
        result = greedy_qubo(small_qubo_4var)
        assert result.method == "greedy_qubo"
        assert result.runtime_ms >= 0

    def test_maxcut_greedy(self, maxcut_qubo):
        """Greedy should find optimal or near-optimal on small Max-Cut."""
        Q = maxcut_qubo
        result = greedy_qubo(Q)
        # Optimal is -2; greedy should find it on this tiny instance
        assert result.objective_value <= -1.0, (
            f"Greedy should find a good cut, got {result.objective_value}"
        )


# ===========================================================================
# Tests: efficient_frontier
# ===========================================================================

class TestEfficientFrontier:

    def test_correct_number_of_points(self, three_asset_problem):
        mu, cov, _ = three_asset_problem
        n_points = 15
        frontier = efficient_frontier(mu, cov, n_points=n_points, max_weight=1.0)
        assert len(frontier) == n_points, (
            f"Expected {n_points} frontier points, got {len(frontier)}"
        )

    def test_risk_increases_monotonically(self, three_asset_problem):
        mu, cov, _ = three_asset_problem
        frontier = efficient_frontier(mu, cov, n_points=20, max_weight=1.0)
        risks = [r.objective_value for r in frontier]
        for i in range(1, len(risks)):
            assert risks[i] >= risks[i - 1] - 1e-8, (
                f"Risk not monotonic at point {i}: {risks[i-1]:.6f} > {risks[i]:.6f}"
            )

    def test_all_points_satisfy_constraints(self, three_asset_problem):
        mu, cov, _ = three_asset_problem
        max_w = 0.6
        frontier = efficient_frontier(mu, cov, n_points=10, max_weight=max_w)
        for i, result in enumerate(frontier):
            assert np.isclose(result.weights.sum(), 1.0, atol=1e-6), (
                f"Point {i}: weights sum to {result.weights.sum()}"
            )
            assert np.all(result.weights >= -1e-6), (
                f"Point {i}: negative weight {result.weights}"
            )
            assert np.all(result.weights <= max_w + 1e-6), (
                f"Point {i}: weight exceeds max {max_w}"
            )

    def test_frontier_spans_return_range(self, three_asset_problem):
        """First point should have low return, last should have high return."""
        mu, cov, _ = three_asset_problem
        frontier = efficient_frontier(mu, cov, n_points=10, max_weight=1.0)
        first_return = mu @ frontier[0].weights
        last_return = mu @ frontier[-1].weights
        assert last_return > first_return + 0.01, (
            f"Frontier doesn't span returns: first={first_return:.4f}, last={last_return:.4f}"
        )

    def test_all_results_are_solver_results(self, three_asset_problem):
        mu, cov, _ = three_asset_problem
        frontier = efficient_frontier(mu, cov, n_points=5, max_weight=1.0)
        for result in frontier:
            assert isinstance(result, SolverResult)
            assert result.converged is True
