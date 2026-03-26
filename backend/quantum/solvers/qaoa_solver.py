"""QAOA solver — Quantum Approximate Optimization Algorithm.

Implements QAOA using direct statevector simulation (numpy), with optional
C++ acceleration for hot-path kernels (build_cost_diagonal, apply_mixer_unitary).

The QAOA circuit for a QUBO problem:
    |psi(gamma,beta)> = prod_{p=1}^{P} [U_M(beta_p) * U_C(gamma_p)] |+>^n

where:
    U_C(gamma) = exp(-i*gamma*C) — cost unitary (diagonal in Z basis)
    U_M(beta)  = exp(-i*beta*B)  — mixer unitary (B = sum X_i)
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quantum.solvers.classical_solvers import SolverResult, solve_brute_force_qubo
from quantum.solvers.problem_encodings import (
    decode_binary_weights,
    evaluate_qubo,
    portfolio_to_qubo,
)

logger = logging.getLogger(__name__)

# Try to load C++ accelerated kernels
try:
    from quantum.cpp import (
        build_cost_diagonal as _cpp_build_cost_diagonal,
        apply_cost_unitary as _cpp_apply_cost_unitary,
        apply_mixer_unitary as _cpp_apply_mixer_unitary,
        qaoa_expectation as _cpp_qaoa_expectation,
        HAS_CPP,
    )
    logger.info("QAOA solver: C++ acceleration available")
except ImportError:
    HAS_CPP = False
    logger.info("QAOA solver: using pure Python (C++ not available)")


def build_cost_diagonal(Q: np.ndarray) -> np.ndarray:
    """Build the diagonal of the cost Hamiltonian.

    For each computational basis state |x>, the cost is x^T Q x.

    Args:
        Q: QUBO matrix of shape (n, n).

    Returns:
        Array of length 2^n with cost for each basis state.
    """
    n = Q.shape[0]
    num_states = 1 << n
    diag = np.empty(num_states, dtype=float)
    for i in range(num_states):
        x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=float)
        diag[i] = float(x @ Q @ x)
    return diag


def apply_cost_unitary(
    statevector: np.ndarray,
    cost_diagonal: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply U_C(gamma) = exp(-i*gamma*C) to statevector.

    Since C is diagonal in the computational basis:
        U_C|x> = exp(-i*gamma*C(x))|x>
    """
    return statevector * np.exp(-1j * gamma * cost_diagonal)


def apply_mixer_unitary(
    statevector: np.ndarray,
    n_qubits: int,
    beta: float,
) -> np.ndarray:
    """Apply U_M(beta) = exp(-i*beta*B) where B = sum X_i.

    Since X_i operators on different qubits commute, apply individually:
        exp(-i*beta*X_q) for each qubit q.

    For each qubit q this rotates pairs of amplitudes that differ
    only in bit q:
        |...0_q...> -> cos(beta)|...0_q...> - i*sin(beta)|...1_q...>
        |...1_q...> -> -i*sin(beta)|...0_q...> + cos(beta)|...1_q...>
    """
    sv = statevector.copy()
    c = np.cos(beta)
    s = np.sin(beta)
    for q in range(n_qubits):
        mask = 1 << q
        num_states = len(sv)
        for i in range(num_states):
            if i & mask == 0:
                j = i | mask
                a0 = sv[i]
                a1 = sv[j]
                sv[i] = c * a0 - 1j * s * a1
                sv[j] = -1j * s * a0 + c * a1
    return sv


def simulate_qaoa_expectation(
    Q: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Compute <psi(gamma,beta)|C|psi(gamma,beta)>.

    Uses C++ acceleration when available, falls back to Python.

    Args:
        Q: QUBO matrix (n, n).
        gamma: array of shape (p,) — cost angles.
        beta: array of shape (p,) — mixer angles.

    Returns:
        Real-valued expectation of cost Hamiltonian.

    Raises:
        ValueError: If gamma and beta have different lengths.
    """
    if len(gamma) != len(beta):
        raise ValueError("gamma and beta must have the same length")

    # Use C++ fast path when available
    if HAS_CPP:
        return float(_cpp_qaoa_expectation(
            np.ascontiguousarray(Q, dtype=np.float64),
            np.ascontiguousarray(gamma, dtype=np.float64),
            np.ascontiguousarray(beta, dtype=np.float64),
        ))

    n = Q.shape[0]
    num_states = 1 << n
    p = len(gamma)

    cost_diag = build_cost_diagonal(Q)

    # |+>^n
    sv = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for layer in range(p):
        sv = apply_cost_unitary(sv, cost_diag, gamma[layer])
        sv = apply_mixer_unitary(sv, n, beta[layer])

    probs = np.abs(sv) ** 2
    return float(np.real(probs @ cost_diag))


def sample_from_state(
    statevector: np.ndarray,
    n_shots: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample bitstrings from statevector.

    Args:
        statevector: complex array of length 2^n.
        n_shots: number of measurement shots.
        seed: random seed for reproducibility.

    Returns:
        Integer array of shape (n_shots,) with sampled basis-state indices.
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(statevector) ** 2
    probs = probs / probs.sum()  # normalise for numerical safety
    return rng.choice(len(statevector), size=n_shots, p=probs)


class QAOASolver:
    """QAOA solver with numpy statevector backend.

    Config:
        n_layers (int): Number of QAOA layers p. Default 3.
        optimizer (str): Scipy optimizer. Default "COBYLA".
        maxiter (int): Max optimizer iterations. Default 500.
        n_shots (int): Measurement shots for sampling. Default 4096.
        seed (int|None): Random seed for reproducibility.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_layers": 3,
        "optimizer": "COBYLA",
        "maxiter": 500,
        "n_shots": 4096,
        "seed": None,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}

    def solve(self, Q: np.ndarray) -> SolverResult:
        """Solve QUBO via QAOA.

        1. Build cost function diagonal from Q
        2. Initialise gamma, beta randomly
        3. Optimise via classical optimiser
        4. Sample from optimal state
        5. Return best bitstring found
        """
        t0 = time.perf_counter()

        p = self._config["n_layers"]
        optimizer = self._config["optimizer"]
        maxiter = self._config["maxiter"]
        n_shots = self._config["n_shots"]
        seed = self._config["seed"]

        rng = np.random.default_rng(seed)

        n = Q.shape[0]
        cost_diag = build_cost_diagonal(Q)

        def objective(params: np.ndarray) -> float:
            g = params[:p]
            b = params[p:]
            return simulate_qaoa_expectation(Q, g, b)

        x0 = rng.uniform(0, 2 * np.pi, size=2 * p)
        opt_result = scipy_minimize(
            objective,
            x0,
            method=optimizer,
            options={"maxiter": maxiter},
        )

        optimal_gamma = opt_result.x[:p]
        optimal_beta = opt_result.x[p:]

        # Build final statevector and sample
        num_states = 1 << n
        sv = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)
        for layer in range(p):
            sv = apply_cost_unitary(sv, cost_diag, optimal_gamma[layer])
            sv = apply_mixer_unitary(sv, n, optimal_beta[layer])

        samples = sample_from_state(sv, n_shots, seed=seed)

        # Evaluate each sampled bitstring and pick the best
        best_val = float("inf")
        best_x: Optional[np.ndarray] = None
        for s in np.unique(samples):
            x = np.array([(int(s) >> bit) & 1 for bit in range(n)], dtype=float)
            val = float(x @ Q @ x)
            if val < best_val:
                best_val = val
                best_x = x

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        return SolverResult(
            weights=best_x if best_x is not None else np.zeros(n),
            objective_value=best_val if best_val < float("inf") else 0.0,
            runtime_ms=runtime_ms,
            converged=opt_result.success,
            iterations=int(getattr(opt_result, "nfev", 0)),
            method="qaoa",
            metadata={
                "optimal_gamma": optimal_gamma.tolist(),
                "optimal_beta": optimal_beta.tolist(),
                "n_layers": p,
                "optimizer": optimizer,
                "n_shots": n_shots,
                "opt_fun": float(opt_result.fun),
            },
        )

    def solve_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: float,
        max_weight: float = 0.10,
        n_bits: int = 3,
    ) -> SolverResult:
        """Convenience: convert portfolio problem to QUBO and solve.

        Also runs brute-force for comparison on small problems.
        """
        Q, offset = portfolio_to_qubo(
            expected_returns=expected_returns,
            covariance=covariance,
            target_return=target_return,
            max_weight=max_weight,
            n_bits=n_bits,
        )

        qaoa_result = self.solve(Q)

        n_assets = len(expected_returns)
        portfolio_weights = decode_binary_weights(
            qaoa_result.weights, n_assets, n_bits, max_weight
        )

        # Brute-force comparison for small problems
        n_vars = Q.shape[0]
        bf_objective = None
        approx_ratio = None
        if n_vars <= 20:
            bf = solve_brute_force_qubo(Q)
            bf_objective = bf.objective_value
            if bf.objective_value != 0.0:
                approx_ratio = qaoa_result.objective_value / bf.objective_value
            else:
                approx_ratio = 1.0 if qaoa_result.objective_value == 0.0 else float("inf")

        meta = dict(qaoa_result.metadata)
        meta["portfolio_weights"] = portfolio_weights.tolist()
        meta["n_assets"] = n_assets
        meta["n_bits"] = n_bits
        meta["max_weight"] = max_weight
        meta["qubo_offset"] = float(offset)
        if bf_objective is not None:
            meta["brute_force_objective"] = float(bf_objective)
        if approx_ratio is not None:
            meta["approximation_ratio"] = float(approx_ratio)

        return SolverResult(
            weights=qaoa_result.weights,
            objective_value=qaoa_result.objective_value,
            runtime_ms=qaoa_result.runtime_ms,
            converged=qaoa_result.converged,
            iterations=qaoa_result.iterations,
            method="qaoa_portfolio",
            metadata=meta,
        )
