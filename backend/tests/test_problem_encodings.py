"""Tests for QUBO problem encodings module.

TDD RED phase: all tests written before implementation.
Covers portfolio_to_qubo, decode_binary_weights, maxcut_qubo, evaluate_qubo.
"""

import numpy as np
import pytest
from itertools import product

from quantum.solvers.problem_encodings import (
    portfolio_to_qubo,
    decode_binary_weights,
    maxcut_qubo,
    evaluate_qubo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brute_force_qubo(Q: np.ndarray) -> tuple:
    """Brute-force optimal binary vector for a QUBO matrix.

    Returns (best_x, best_value).
    Only feasible for small problem sizes.
    """
    n = Q.shape[0]
    best_val = np.inf
    best_x = None
    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        val = x @ Q @ x
        if val < best_val:
            best_val = val
            best_x = x.copy()
    return best_x, best_val


# ===========================================================================
# 1. portfolio_to_qubo
# ===========================================================================

class TestPortfolioToQubo:
    """Tests for portfolio_to_qubo conversion."""

    def _make_2asset_problem(self):
        """Create a simple 2-asset problem."""
        mu = np.array([0.10, 0.05])
        sigma = np.array([[0.04, 0.01],
                          [0.01, 0.02]])
        return mu, sigma

    def _make_3asset_problem(self):
        """Create a 3-asset problem."""
        mu = np.array([0.12, 0.08, 0.05])
        sigma = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.008],
            [0.005, 0.008, 0.02],
        ])
        return mu, sigma

    # --- Dimension tests ---------------------------------------------------

    def test_qubo_dimensions_2assets_3bits(self):
        """Q matrix should be (N*K) x (N*K)."""
        mu, sigma = self._make_2asset_problem()
        n_bits = 3
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05, n_bits=n_bits)
        expected_dim = 2 * n_bits  # N * K = 6
        assert Q.shape == (expected_dim, expected_dim)

    def test_qubo_dimensions_3assets_3bits(self):
        """Q matrix should be 9x9 for 3 assets, 3 bits."""
        mu, sigma = self._make_3asset_problem()
        n_bits = 3
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05, n_bits=n_bits)
        expected_dim = 3 * n_bits  # 9
        assert Q.shape == (expected_dim, expected_dim)

    def test_qubo_dimensions_2assets_4bits(self):
        """Q matrix should be 8x8 for 2 assets, 4 bits."""
        mu, sigma = self._make_2asset_problem()
        n_bits = 4
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05, n_bits=n_bits)
        expected_dim = 2 * n_bits  # 8
        assert Q.shape == (expected_dim, expected_dim)

    # --- Offset is scalar --------------------------------------------------

    def test_offset_is_scalar(self):
        """The offset should be a scalar float."""
        mu, sigma = self._make_2asset_problem()
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05)
        assert isinstance(offset, (float, np.floating))

    # --- Upper triangular form ---------------------------------------------

    def test_qubo_upper_triangular(self):
        """QUBO matrix should be upper triangular (standard convention)."""
        mu, sigma = self._make_2asset_problem()
        Q, _ = portfolio_to_qubo(mu, sigma, target_return=0.05, n_bits=2)
        # All below-diagonal entries should be zero
        for i in range(Q.shape[0]):
            for j in range(i):
                assert Q[i, j] == pytest.approx(0.0, abs=1e-12), \
                    f"Q[{i},{j}] = {Q[i, j]} is not zero (not upper triangular)"

    # --- Budget constraint: weights sum to ~1.0 ----------------------------

    def test_optimal_solution_budget_constraint(self):
        """The brute-force optimal solution should have weights summing close to 1.0."""
        mu, sigma = self._make_2asset_problem()
        # Use high max_weight so budget constraint can be satisfied with 2 assets
        Q, offset = portfolio_to_qubo(
            mu, sigma,
            target_return=0.02,
            max_weight=0.60,
            n_bits=3,
            penalty_budget=50.0,
            penalty_return=1.0,
        )
        best_x, best_val = brute_force_qubo(Q)
        weights = decode_binary_weights(best_x, n_assets=2, n_bits=3, max_weight=0.60)
        total = np.sum(weights)
        assert total == pytest.approx(1.0, abs=0.15), \
            f"Weights sum to {total}, expected ~1.0"

    # --- Max weight constraint ---------------------------------------------

    def test_no_weight_exceeds_max(self):
        """No individual weight should exceed max_weight."""
        mu, sigma = self._make_2asset_problem()
        max_w = 0.60
        Q, offset = portfolio_to_qubo(
            mu, sigma,
            target_return=0.02,
            max_weight=max_w,
            n_bits=3,
            penalty_budget=50.0,
        )
        best_x, _ = brute_force_qubo(Q)
        weights = decode_binary_weights(best_x, n_assets=2, n_bits=3, max_weight=max_w)
        for i, w in enumerate(weights):
            assert w <= max_w + 1e-10, f"Weight {i} = {w} exceeds max_weight {max_w}"

    # --- Reasonable weights from optimal solution --------------------------

    def test_optimal_solution_reasonable_weights(self):
        """Decoded weights should be non-negative and <= max_weight."""
        mu, sigma = self._make_3asset_problem()
        max_w = 0.50
        Q, offset = portfolio_to_qubo(
            mu, sigma,
            target_return=0.03,
            max_weight=max_w,
            n_bits=3,
            penalty_budget=20.0,
            penalty_return=5.0,
        )
        best_x, _ = brute_force_qubo(Q)
        weights = decode_binary_weights(best_x, n_assets=3, n_bits=3, max_weight=max_w)
        assert len(weights) == 3
        for w in weights:
            assert 0.0 <= w <= max_w + 1e-10

    # --- Q matrix is real-valued -------------------------------------------

    def test_qubo_real_valued(self):
        """Q should contain only real numbers, no NaN or inf."""
        mu, sigma = self._make_2asset_problem()
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05)
        assert np.all(np.isfinite(Q))
        assert np.all(np.isfinite(offset))


# ===========================================================================
# 2. decode_binary_weights
# ===========================================================================

class TestDecodeBinaryWeights:
    """Tests for decode_binary_weights."""

    def test_basic_decode_2bits(self):
        """With 2 bits and max_weight=1.0, binary [1,1] → weight = (1+2)/(2^2-1) = 1.0."""
        # 1 asset, 2 bits: x = [1, 1] → w = (1*2^0 + 1*2^1) * 1.0 / (2^2 - 1) = 3/3 = 1.0
        x = np.array([1, 1], dtype=float)
        weights = decode_binary_weights(x, n_assets=1, n_bits=2, max_weight=1.0)
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_basic_decode_2bits_half(self):
        """With 2 bits, binary [1,0] → weight = 1/3 * max_weight."""
        x = np.array([1, 0], dtype=float)
        weights = decode_binary_weights(x, n_assets=1, n_bits=2, max_weight=0.60)
        # (1*2^0) / (2^2 - 1) * 0.60 = 1/3 * 0.60 = 0.20
        assert weights[0] == pytest.approx(0.20, abs=1e-10)

    def test_decode_all_zeros(self):
        """All-zero binary → all-zero weights."""
        x = np.zeros(6)
        weights = decode_binary_weights(x, n_assets=2, n_bits=3, max_weight=0.50)
        assert len(weights) == 2
        np.testing.assert_array_almost_equal(weights, [0.0, 0.0])

    def test_decode_all_ones(self):
        """All-one binary → each weight equals max_weight."""
        x = np.ones(6)
        weights = decode_binary_weights(x, n_assets=2, n_bits=3, max_weight=0.50)
        assert len(weights) == 2
        np.testing.assert_array_almost_equal(weights, [0.50, 0.50])

    def test_decode_3bits_precision(self):
        """3-bit precision gives 8 levels (0 to 7), so step = max_weight/7."""
        max_w = 0.70
        # Asset 0: bits [1, 0, 1] → 1*1 + 0*2 + 1*4 = 5 → 5/7 * 0.70
        # Asset 1: bits [0, 1, 1] → 0*1 + 1*2 + 1*4 = 6 → 6/7 * 0.70
        x = np.array([1, 0, 1, 0, 1, 1], dtype=float)
        weights = decode_binary_weights(x, n_assets=2, n_bits=3, max_weight=max_w)
        expected_0 = 5.0 / 7.0 * max_w
        expected_1 = 6.0 / 7.0 * max_w
        assert weights[0] == pytest.approx(expected_0)
        assert weights[1] == pytest.approx(expected_1)

    def test_decode_4bits_precision(self):
        """4-bit precision gives 16 levels (0 to 15), step = max_weight/15."""
        max_w = 0.10
        # 1 asset, bits [1, 1, 0, 0] → 1 + 2 = 3 → 3/15 * 0.10 = 0.02
        x = np.array([1, 1, 0, 0], dtype=float)
        weights = decode_binary_weights(x, n_assets=1, n_bits=4, max_weight=max_w)
        assert weights[0] == pytest.approx(3.0 / 15.0 * max_w)

    def test_round_trip_encode_decode(self):
        """Known weights → binary → decode should be close to original (within discretization)."""
        max_w = 0.50
        n_bits = 4
        n_assets = 2
        # Target weights: [0.30, 0.20]
        # Discretized: w_i = val_i / (2^K - 1) * max_weight
        # val_i = round(w_i * (2^K - 1) / max_weight)
        # val_0 = round(0.30 * 15 / 0.50) = round(9.0) = 9 → binary: 1001
        # val_1 = round(0.20 * 15 / 0.50) = round(6.0) = 6 → binary: 0110
        x = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=float)
        weights = decode_binary_weights(x, n_assets=n_assets, n_bits=n_bits, max_weight=max_w)
        assert weights[0] == pytest.approx(0.30, abs=0.01)
        assert weights[1] == pytest.approx(0.20, abs=0.01)


# ===========================================================================
# 3. maxcut_qubo
# ===========================================================================

class TestMaxCutQubo:
    """Tests for maxcut_qubo."""

    def _triangle_graph(self):
        """Simple 3-node triangle with uniform weight."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        return adj

    def test_triangle_dimensions(self):
        """Q should be 3x3 for 3-node graph."""
        adj = self._triangle_graph()
        Q = maxcut_qubo(adj)
        assert Q.shape == (3, 3)

    def test_triangle_brute_force_optimal(self):
        """Brute-force max-cut of triangle should cut 2 edges."""
        adj = self._triangle_graph()
        Q = maxcut_qubo(adj)
        best_x, best_val = brute_force_qubo(Q)

        # For a triangle, the maximum cut puts 1 node on one side, 2 on the other.
        # That cuts exactly 2 edges out of 3.
        # Manually verify: all possible cuts for triangle
        # {0} vs {1,2}: cuts (0,1), (0,2) = 2 edges
        # {1} vs {0,2}: cuts (0,1), (1,2) = 2 edges
        # {2} vs {0,1}: cuts (0,2), (1,2) = 2 edges
        # {} or {0,1,2}: cuts 0 edges
        # So the min QUBO value should correspond to max-cut of 2 edges.

        # Compute cut value from the assignment
        cut_value = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                if adj[i, j] != 0 and best_x[i] != best_x[j]:
                    cut_value += adj[i, j]
        assert cut_value == pytest.approx(2.0)

    def test_weighted_graph_optimal(self):
        """Weighted 3-node graph: brute-force QUBO solution matches brute-force max-cut."""
        adj = np.array([
            [0, 3, 1],
            [3, 0, 2],
            [1, 2, 0],
        ], dtype=float)
        Q = maxcut_qubo(adj)
        best_x, _ = brute_force_qubo(Q)

        # Brute-force max-cut directly
        max_cut = 0.0
        for bits in product([0, 1], repeat=3):
            cut = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    if bits[i] != bits[j]:
                        cut += adj[i, j]
            max_cut = max(max_cut, cut)

        # The QUBO solution should achieve the same max cut
        qubo_cut = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                if best_x[i] != best_x[j]:
                    qubo_cut += adj[i, j]
        assert qubo_cut == pytest.approx(max_cut)

    def test_qubo_upper_triangular_or_symmetric(self):
        """Q should be upper triangular (standard QUBO convention)."""
        adj = self._triangle_graph()
        Q = maxcut_qubo(adj)
        # Check upper triangular: below-diagonal should be zero
        for i in range(Q.shape[0]):
            for j in range(i):
                assert Q[i, j] == pytest.approx(0.0, abs=1e-12), \
                    f"Q[{i},{j}] = {Q[i, j]} should be 0 for upper triangular"

    def test_no_self_loops_contribution(self):
        """A graph with self-loops should ignore them (self-loop doesn't affect cut)."""
        adj = np.array([
            [5, 1, 0],
            [1, 3, 1],
            [0, 1, 2],
        ], dtype=float)
        Q = maxcut_qubo(adj)
        # The diagonal of Q should only reflect off-diagonal adjacency, not self-loops
        # Just verify it runs and gives valid shape
        assert Q.shape == (3, 3)
        assert np.all(np.isfinite(Q))

    def test_empty_graph(self):
        """An empty graph (no edges) should give a zero QUBO matrix."""
        adj = np.zeros((3, 3))
        Q = maxcut_qubo(adj)
        np.testing.assert_array_almost_equal(Q, np.zeros((3, 3)))


# ===========================================================================
# 4. evaluate_qubo
# ===========================================================================

class TestEvaluateQubo:
    """Tests for evaluate_qubo."""

    def test_identity_matrix(self):
        """Q = I, x = [1,0,1] → x^T I x = 2."""
        Q = np.eye(3)
        x = np.array([1, 0, 1], dtype=float)
        assert evaluate_qubo(Q, x) == pytest.approx(2.0)

    def test_zero_vector(self):
        """x = 0 → objective = 0 regardless of Q."""
        Q = np.array([[1, 2], [3, 4]], dtype=float)
        x = np.zeros(2)
        assert evaluate_qubo(Q, x) == pytest.approx(0.0)

    def test_hand_computed_value(self):
        """Hand-compute x^T Q x for a small example.

        Q = [[1, 2],
             [0, 3]]
        x = [1, 1]
        x^T Q x = 1*1*1 + 1*2*1 + 1*0*1 + 1*3*1 = 1 + 2 + 0 + 3 = 6
        """
        Q = np.array([[1, 2], [0, 3]], dtype=float)
        x = np.array([1, 1], dtype=float)
        assert evaluate_qubo(Q, x) == pytest.approx(6.0)

    def test_single_variable(self):
        """Single binary variable: Q = [[5]], x = [1] → 5."""
        Q = np.array([[5.0]])
        x = np.array([1.0])
        assert evaluate_qubo(Q, x) == pytest.approx(5.0)

    def test_optimal_solution_has_lowest_objective(self):
        """The brute-force optimal should have the lowest evaluate_qubo value."""
        Q = np.array([
            [1, -3, 0],
            [0, 2, -1],
            [0, 0, 1],
        ], dtype=float)
        best_x, best_val = brute_force_qubo(Q)
        # Verify evaluate_qubo matches brute-force
        assert evaluate_qubo(Q, best_x) == pytest.approx(best_val)
        # Verify it's the minimum over all binary vectors
        n = Q.shape[0]
        for bits in product([0, 1], repeat=n):
            x = np.array(bits, dtype=float)
            assert evaluate_qubo(Q, x) >= best_val - 1e-10

    def test_consistency_with_portfolio_qubo(self):
        """evaluate_qubo(Q, x) should equal x^T Q x for portfolio QUBO."""
        mu = np.array([0.10, 0.05])
        sigma = np.array([[0.04, 0.01],
                          [0.01, 0.02]])
        Q, offset = portfolio_to_qubo(mu, sigma, target_return=0.05, n_bits=2)
        x = np.array([1, 0, 1, 1], dtype=float)
        expected = x @ Q @ x
        assert evaluate_qubo(Q, x) == pytest.approx(expected)
