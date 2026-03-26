"""Scaling analysis utilities for problem-size experiments."""

from typing import Dict, List, Optional

import numpy as np

from quantum.solvers.problem_encodings import maxcut_qubo, portfolio_to_qubo


def generate_random_qubo(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate random upper-triangular QUBO matrix of size n x n."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    return np.triu(Q)


def generate_maxcut_qubo(
    n: int,
    edge_prob: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Max-Cut QUBO from random Erdos-Renyi graph G(n, edge_prob)."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = rng.uniform(0.5, 2.0)
                adj[i, j] = w
                adj[j, i] = w
    return maxcut_qubo(adj)


def generate_portfolio_qubo(
    n_assets: int,
    n_bits: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate portfolio optimisation QUBO with random returns/covariance."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n_assets)

    A = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + np.eye(n_assets) * 0.01

    target_return = float(np.mean(mu))
    Q, _ = portfolio_to_qubo(
        expected_returns=mu,
        covariance=cov,
        target_return=target_return,
        max_weight=0.5,
        n_bits=n_bits,
    )
    return Q


def compute_scaling_exponents(
    sizes: List[int],
    runtimes: List[float],
) -> Dict[str, float]:
    """Fit log(runtime) vs log(size) to estimate scaling exponent.

    Returns:
        {"exponent": float, "r_squared": float}
    """
    log_s = np.log(np.asarray(sizes, dtype=float))
    log_r = np.log(np.asarray(runtimes, dtype=float))

    coeffs = np.polyfit(log_s, log_r, 1)
    exponent = float(coeffs[0])

    predicted = np.polyval(coeffs, log_s)
    ss_res = float(np.sum((log_r - predicted) ** 2))
    ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"exponent": exponent, "r_squared": r_squared}
