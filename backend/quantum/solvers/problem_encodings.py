"""QUBO problem encodings for quantum optimization.

Converts financial optimization problems into Quadratic Unconstrained
Binary Optimization (QUBO) form suitable for QAOA and quantum annealers.

Weight discretization scheme:
    w_i = sum_{k=0}^{K-1} 2^k * x_{i,k} * max_weight / (2^K - 1)

where K = n_bits, and x_{i,k} in {0, 1}.

QUBO convention: upper triangular form, Q_{ij} with i <= j.
Objective: minimize x^T Q x.
"""

import numpy as np
from typing import Tuple


def portfolio_to_qubo(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
    max_weight: float = 0.10,
    n_bits: int = 3,
    penalty_budget: float = 10.0,
    penalty_return: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """Convert portfolio optimization to QUBO.

    Discretizes weights: w_i = sum_{k=0}^{K-1} 2^k * x_{i,k} * max_weight / (2^K - 1)

    Objective: minimize w^T Sigma w
               + penalty_budget * (sum(w) - 1)^2
               + penalty_return * max(0, target_return - mu^T w)^2

    For the QUBO formulation, the return constraint is relaxed to a
    quadratic penalty (ignoring the max(0,...) for tractability; the
    penalty still drives solutions toward feasibility).

    Args:
        expected_returns: Expected return per asset, shape (N,).
        covariance: Covariance matrix, shape (N, N). Should be Ledoit-Wolf shrunk.
        target_return: Minimum target return.
        max_weight: Maximum weight per asset.
        n_bits: Bits of precision per weight (K).
        penalty_budget: Penalty strength for budget constraint.
        penalty_return: Penalty strength for return constraint.

    Returns:
        (Q, offset) where Q is the upper-triangular QUBO matrix of shape
        (N*K, N*K) and offset is the constant term of the objective.
    """
    n_assets = len(expected_returns)
    total_bits = n_assets * n_bits
    scale = max_weight / (2 ** n_bits - 1)

    # Precompute the power-of-2 coefficients for each binary variable.
    # Variable index for asset i, bit k: idx = i * n_bits + k
    # Coefficient: c_{i,k} = 2^k * scale
    # So w_i = sum_k c_{i,k} * x_{i,k}

    coeffs = np.zeros(total_bits)
    for i in range(n_assets):
        for k in range(n_bits):
            coeffs[i * n_bits + k] = (2 ** k) * scale

    # Build the full (symmetric) Q matrix, then fold into upper triangular.
    Q_full = np.zeros((total_bits, total_bits))

    # --- Term 1: w^T Sigma w ---
    # w^T Sigma w = sum_{i,j} Sigma_{i,j} * w_i * w_j
    #             = sum_{i,j} Sigma_{i,j} * sum_{k} c_{ik} x_{ik} * sum_{l} c_{jl} x_{jl}
    #             = sum_{(i,k),(j,l)} Sigma_{i,j} * c_{ik} * c_{jl} * x_{ik} * x_{jl}
    for i in range(n_assets):
        for j in range(n_assets):
            for ki in range(n_bits):
                for kj in range(n_bits):
                    idx_a = i * n_bits + ki
                    idx_b = j * n_bits + kj
                    Q_full[idx_a, idx_b] += covariance[i, j] * coeffs[idx_a] * coeffs[idx_b]

    # --- Term 2: penalty_budget * (sum(w) - 1)^2 ---
    # (sum(w) - 1)^2 = (sum_a c_a x_a - 1)^2
    #                = sum_{a,b} c_a c_b x_a x_b - 2 sum_a c_a x_a + 1
    # Quadratic part:
    for a in range(total_bits):
        for b in range(total_bits):
            Q_full[a, b] += penalty_budget * coeffs[a] * coeffs[b]
    # Linear part: -2 * c_a on diagonal (x_a^2 = x_a for binary)
    for a in range(total_bits):
        Q_full[a, a] -= 2.0 * penalty_budget * coeffs[a]
    # Constant: penalty_budget * 1
    offset = penalty_budget * 1.0

    # --- Term 3: penalty_return * (target_return - mu^T w)^2 ---
    # (r_t - mu^T w)^2 = r_t^2 - 2 r_t sum_a mu_asset(a) c_a x_a
    #                   + (sum_a mu_asset(a) c_a x_a)^2
    # where mu_asset(a) = expected_returns[a // n_bits]

    mu_coeffs = np.zeros(total_bits)
    for i in range(n_assets):
        for k in range(n_bits):
            mu_coeffs[i * n_bits + k] = expected_returns[i] * coeffs[i * n_bits + k]

    # Quadratic: sum_{a,b} mu_a c_a * mu_b c_b  → mu_coeffs[a] * mu_coeffs[b] (wrong, need separate)
    # Actually: (mu^T w)^2 = (sum_a mu_asset(a) c_a x_a)^2 = sum_{a,b} mu_coeffs[a] mu_coeffs[b] x_a x_b
    for a in range(total_bits):
        for b in range(total_bits):
            Q_full[a, b] += penalty_return * mu_coeffs[a] * mu_coeffs[b]

    # Linear: -2 * r_t * mu_coeffs[a] on diagonal
    for a in range(total_bits):
        Q_full[a, a] -= 2.0 * penalty_return * target_return * mu_coeffs[a]

    # Constant: penalty_return * r_t^2
    offset += penalty_return * target_return ** 2

    # --- Fold into upper triangular ---
    Q = _to_upper_triangular(Q_full)

    return Q, float(offset)


def decode_binary_weights(
    binary_solution: np.ndarray,
    n_assets: int,
    n_bits: int,
    max_weight: float = 0.10,
) -> np.ndarray:
    """Decode binary QUBO solution to continuous portfolio weights.

    Each asset's weight is encoded by n_bits binary variables:
        w_i = sum_{k=0}^{K-1} 2^k * x_{i*K+k} * max_weight / (2^K - 1)

    Args:
        binary_solution: Binary vector of length N * K.
        n_assets: Number of assets (N).
        n_bits: Bits of precision per weight (K).
        max_weight: Maximum weight per asset.

    Returns:
        Weight vector of length N with values in [0, max_weight].
    """
    scale = max_weight / (2 ** n_bits - 1)
    weights = np.zeros(n_assets)
    for i in range(n_assets):
        val = 0.0
        for k in range(n_bits):
            val += (2 ** k) * binary_solution[i * n_bits + k]
        weights[i] = val * scale
    return weights


def maxcut_qubo(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Convert weighted graph to Max-Cut QUBO.

    Max-Cut objective: maximize sum_{(i,j) in E} w_ij * (x_i XOR x_j)
    where x_i in {0, 1} indicates which partition node i belongs to.

    Equivalently, minimize:
        sum_{(i,j)} w_ij * (x_i x_j - x_i/2 - x_j/2 + 1/4)
    ignoring constants, this becomes:
        sum_{(i,j)} w_ij * x_i x_j - sum_i (sum_j w_ij / 2) * x_i

    So Q_ii = -sum_{j!=i} w_ij / 2  (diagonal)
       Q_ij = w_ij / 2               (off-diagonal, i < j)

    The constant offset (sum of w_ij / 4) is dropped since it doesn't
    affect the optimal solution.

    Args:
        adjacency_matrix: Symmetric weighted adjacency matrix, shape (N, N).
            Self-loops (diagonal entries) are ignored.

    Returns:
        Upper-triangular QUBO matrix Q of shape (N, N).
    """
    n = adjacency_matrix.shape[0]
    # Zero out diagonal (ignore self-loops)
    adj = adjacency_matrix.copy()
    np.fill_diagonal(adj, 0.0)

    Q_full = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: -sum_{j!=i} w_ij / 2
                Q_full[i, i] = -np.sum(adj[i, :]) / 2.0
            else:
                # Off-diagonal: w_ij / 2
                Q_full[i, j] = adj[i, j] / 2.0

    return _to_upper_triangular(Q_full)


def evaluate_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate QUBO objective: x^T Q x.

    Args:
        Q: QUBO matrix of shape (n, n).
        x: Binary solution vector of length n.

    Returns:
        Objective value as a scalar float.
    """
    return float(x @ Q @ x)


def _to_upper_triangular(Q_full: np.ndarray) -> np.ndarray:
    """Fold a symmetric/full matrix into upper triangular form.

    For QUBO, x_i * x_j and x_j * x_i contribute identically,
    so we combine them: Q_upper[i,j] = Q_full[i,j] + Q_full[j,i]
    for i < j, and Q_upper[i,i] = Q_full[i,i].

    Args:
        Q_full: Square matrix (possibly symmetric).

    Returns:
        Upper triangular matrix with equivalent QUBO objective.
    """
    n = Q_full.shape[0]
    Q_upper = np.zeros((n, n))
    for i in range(n):
        Q_upper[i, i] = Q_full[i, i]
        for j in range(i + 1, n):
            Q_upper[i, j] = Q_full[i, j] + Q_full[j, i]
    return Q_upper
