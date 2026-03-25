"""Classical optimization solvers -- baselines for quantum comparison.

Every quantum optimization result is benchmarked against these classical
methods. If the quantum method can't beat these, we report that honestly.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass(frozen=True)
class SolverResult:
    """Standard result from any solver."""
    weights: np.ndarray                # solution vector
    objective_value: float             # objective function value
    runtime_ms: float                  # wall-clock time in milliseconds
    converged: bool = True             # whether solver converged
    iterations: int = 0                # solver iterations
    method: str = ""                   # solver name
    metadata: dict = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


def solve_markowitz_cvxpy(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
    max_weight: float = 0.10,
    long_only: bool = True,
) -> SolverResult:
    """Solve Markowitz mean-variance optimization via CVXPY.

    minimize    w^T Sigma w
    subject to  mu^T w >= target_return
                sum(w) == 1
                0 <= w <= max_weight  (if long_only)
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for solve_markowitz_cvxpy. "
            "Install it with: pip install cvxpy"
        ) from exc

    n = len(expected_returns)
    start = time.perf_counter()

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, covariance))

    constraints = [
        expected_returns @ w >= target_return,
        cp.sum(w) == 1,
    ]

    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= max_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return SolverResult(
            weights=np.zeros(n),
            objective_value=float("inf"),
            runtime_ms=elapsed_ms,
            converged=False,
            method="markowitz_cvxpy",
            metadata={"status": prob.status},
        )

    weights = np.array(w.value).flatten()
    objective_value = float(prob.value)

    return SolverResult(
        weights=weights,
        objective_value=objective_value,
        runtime_ms=elapsed_ms,
        converged=True,
        method="markowitz_cvxpy",
        metadata={"status": prob.status},
    )


def solve_markowitz_scipy(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
    max_weight: float = 0.10,
    long_only: bool = True,
) -> SolverResult:
    """Solve Markowitz via scipy.optimize.minimize (SLSQP).

    Fallback when cvxpy is not available.
    """
    from scipy.optimize import minimize as scipy_minimize

    n = len(expected_returns)
    start = time.perf_counter()

    def objective(w: np.ndarray) -> float:
        return float(w @ covariance @ w)

    def grad_objective(w: np.ndarray) -> np.ndarray:
        return 2.0 * covariance @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: expected_returns @ w - target_return},
    ]

    if long_only:
        bounds = [(0.0, max_weight) for _ in range(n)]
    else:
        bounds = [(None, None) for _ in range(n)]

    w0 = np.ones(n) / n

    result = scipy_minimize(
        objective,
        w0,
        method="SLSQP",
        jac=grad_objective,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    weights = result.x
    obj_val = float(weights @ covariance @ weights)

    return SolverResult(
        weights=weights,
        objective_value=obj_val,
        runtime_ms=elapsed_ms,
        converged=result.success,
        iterations=int(result.nit),
        method="markowitz_scipy",
        metadata={"message": result.message},
    )


def solve_brute_force_qubo(Q: np.ndarray) -> SolverResult:
    """Brute-force enumerate all 2^N solutions for QUBO.

    Only feasible for N <= 20. Returns exact optimal.
    Q is an (N, N) QUBO matrix.
    Objective: x^T Q x for binary x.
    """
    n = Q.shape[0]
    if n > 20:
        raise ValueError(
            f"Brute force is only feasible for N <= 20, got N={n}"
        )

    start = time.perf_counter()

    best_val = float("inf")
    best_x = np.zeros(n, dtype=float)
    total = 2 ** n

    for i in range(total):
        x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=float)
        val = float(x @ Q @ x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SolverResult(
        weights=best_x,
        objective_value=best_val,
        runtime_ms=elapsed_ms,
        converged=True,
        iterations=total,
        method="brute_force_qubo",
    )


def simulated_annealing_qubo(
    Q: np.ndarray,
    n_iterations: int = 10000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    seed: Optional[int] = None,
) -> SolverResult:
    """Simulated annealing for QUBO minimization.

    Uses geometric cooling schedule. At each step, flip a random bit and
    accept or reject based on the Metropolis criterion.
    """
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    # Initialize with a random binary vector
    x = rng.integers(0, 2, size=n).astype(float)
    current_val = float(x @ Q @ x)

    best_x = x.copy()
    best_val = current_val
    temp = initial_temp

    for iteration in range(n_iterations):
        # Pick a random bit to flip
        flip_idx = rng.integers(0, n)

        # Compute delta efficiently:
        # If x_i goes from 0 -> 1: delta = Q[i,i] + 2 * sum_{j!=i} Q[i,j]*x[j]
        # If x_i goes from 1 -> 0: delta = -Q[i,i] - 2 * sum_{j!=i} Q[i,j]*x[j]
        # More precisely, for x^T Q x:
        # new_val - old_val when flipping bit i
        if x[flip_idx] == 0.0:
            # Flipping 0 -> 1
            delta = Q[flip_idx, flip_idx] + 2.0 * np.dot(Q[flip_idx, :], x) - 2.0 * Q[flip_idx, flip_idx] * x[flip_idx]
            # Simplify: since x[flip_idx]=0, dot includes Q[i,i]*0
            # delta = Q[i,i] + 2 * sum_{j} Q[i,j]*x[j]  (x[i]=0 so Q[i,i]*x[i]=0)
            # Actually let me just compute it cleanly
            delta = Q[flip_idx, flip_idx] + 2.0 * np.dot(Q[flip_idx, :], x)
        else:
            # Flipping 1 -> 0
            # new x[i] = 0, old x[i] = 1
            delta = -Q[flip_idx, flip_idx] - 2.0 * (np.dot(Q[flip_idx, :], x) - Q[flip_idx, flip_idx])

        # Metropolis acceptance
        if delta < 0:
            accept = True
        else:
            if temp > 1e-15:
                accept = rng.random() < np.exp(-delta / temp)
            else:
                accept = False

        if accept:
            x[flip_idx] = 1.0 - x[flip_idx]
            current_val += delta

            if current_val < best_val:
                best_x = x.copy()
                best_val = current_val

        # Cool down
        temp *= cooling_rate

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Recompute best_val exactly to avoid floating-point drift
    best_val = float(best_x @ Q @ best_x)

    return SolverResult(
        weights=best_x,
        objective_value=best_val,
        runtime_ms=elapsed_ms,
        converged=True,
        iterations=n_iterations,
        method="simulated_annealing_qubo",
    )


def greedy_qubo(Q: np.ndarray) -> SolverResult:
    """Greedy construction for QUBO minimization.

    Start with all zeros, flip the bit that gives the largest
    objective decrease. Repeat until no improvement.
    """
    n = Q.shape[0]
    start = time.perf_counter()

    x = np.zeros(n, dtype=float)
    current_val = 0.0
    iterations = 0

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_flip = -1

        for i in range(n):
            if x[i] == 0.0:
                # Flipping 0 -> 1
                delta = Q[i, i] + 2.0 * np.dot(Q[i, :], x)
            else:
                # Flipping 1 -> 0
                delta = -Q[i, i] - 2.0 * (np.dot(Q[i, :], x) - Q[i, i])

            if delta < best_delta:
                best_delta = delta
                best_flip = i

        if best_flip >= 0:
            x[best_flip] = 1.0 - x[best_flip]
            current_val += best_delta
            improved = True
            iterations += 1

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Recompute exactly to avoid drift
    current_val = float(x @ Q @ x)

    return SolverResult(
        weights=x,
        objective_value=current_val,
        runtime_ms=elapsed_ms,
        converged=True,
        iterations=iterations,
        method="greedy_qubo",
    )


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    n_points: int = 20,
    max_weight: float = 0.10,
    long_only: bool = True,
) -> List[SolverResult]:
    """Generate efficient frontier points.

    Varies target_return from min achievable to max achievable,
    solves Markowitz at each point.
    """
    n = len(expected_returns)

    # Min achievable return: minimum-variance portfolio return
    # Max achievable return: portfolio concentrated in highest-return asset
    # With max_weight constraint, max return = max_weight * sorted_top returns
    # Simple approach: min = min(mu), max = achievable given constraints

    if long_only:
        # Min achievable: invest in the lowest-return asset (or blend)
        min_return = float(np.min(expected_returns))
        # Max achievable: greedily allocate max_weight to highest-return assets
        sorted_returns = np.sort(expected_returns)[::-1]
        max_return = 0.0
        remaining = 1.0
        for r in sorted_returns:
            alloc = min(max_weight, remaining)
            max_return += alloc * r
            remaining -= alloc
            if remaining <= 1e-10:
                break
        max_return = float(max_return)
    else:
        min_return = float(np.min(expected_returns))
        max_return = float(np.max(expected_returns))

    # Add small buffer to avoid infeasibility at the exact boundary
    buffer = (max_return - min_return) * 0.01
    min_target = min_return + buffer
    max_target = max_return - buffer

    targets = np.linspace(min_target, max_target, n_points)

    results: List[SolverResult] = []
    for target in targets:
        result = solve_markowitz_cvxpy(
            expected_returns,
            covariance,
            target_return=float(target),
            max_weight=max_weight,
            long_only=long_only,
        )
        results.append(result)

    return results
