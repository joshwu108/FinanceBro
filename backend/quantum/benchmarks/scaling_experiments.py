"""Scaling experiments — real financial data + synthetic NP-hard problems.

Runs comprehensive benchmarks across problem sizes and logs results
to /experiments/ for analysis and visualization.

Experiment types:
  1. Portfolio optimization: real stock data, 3→12 assets, QAOA vs classical
  2. Max-Cut: Erdos-Renyi graphs, 4→14 nodes, QAOA vs SA vs greedy
  3. C++ vs Python speedup: mixer unitary across qubit counts
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantum.benchmarks.scaling_analysis import (
    compute_scaling_exponents,
    generate_maxcut_qubo,
    generate_portfolio_qubo,
)
from quantum.solvers.classical_solvers import (
    SolverResult,
    greedy_qubo,
    simulated_annealing_qubo,
    solve_brute_force_qubo,
    solve_markowitz_scipy,
)
from quantum.solvers.problem_encodings import portfolio_to_qubo
from quantum.solvers.qaoa_solver import QAOASolver, build_cost_diagonal, apply_mixer_unitary

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def _log_experiment(name: str, data: Dict[str, Any]) -> Path:
    """Write experiment results as JSON to /experiments/."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{ts}.json"
    path = EXPERIMENTS_DIR / filename

    # Make numpy arrays serializable
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

    logger.info("Experiment logged: %s", path)
    return path


def fetch_real_returns(
    tickers: List[str],
    period: str = "1y",
) -> Optional[pd.DataFrame]:
    """Fetch real daily returns via yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        if len(returns) < 60:
            return None
        return returns
    except Exception as e:
        logger.warning("Failed to fetch real data: %s", e)
        return None


def synthetic_returns(n_assets: int, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily returns as fallback."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for i in range(n_assets):
        mu = 0.0003 + i * 0.0001
        sigma = 0.01 + i * 0.002
        data[f"ASSET_{i}"] = rng.normal(mu, sigma, n_days)
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Experiment 1: Portfolio scaling
# ---------------------------------------------------------------------------

def run_portfolio_scaling(
    asset_counts: Optional[List[int]] = None,
    tickers: Optional[List[str]] = None,
    n_bits: int = 3,
    qaoa_layers: int = 2,
    qaoa_maxiter: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep portfolio sizes: QAOA vs classical Markowitz.

    For each asset count, solves the portfolio optimization problem
    and records runtime, objective value, and approximation ratio.
    """
    if asset_counts is None:
        asset_counts = [3, 4, 5, 6, 8]

    if tickers is None:
        tickers = [
            "AAPL", "MSFT", "GOOG", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
            "UNH", "WMT",
        ]

    # Try real data
    real_returns = fetch_real_returns(tickers[:max(asset_counts)])

    results = []
    for n_assets in asset_counts:
        if real_returns is not None and n_assets <= real_returns.shape[1]:
            returns_df = real_returns.iloc[:, :n_assets]
            data_source = "yfinance"
        else:
            returns_df = synthetic_returns(n_assets, seed=seed)
            data_source = "synthetic"

        mu = np.array(returns_df.mean())
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns_df.values).covariance_
        target_return = float(np.mean(mu))
        max_weight = max(0.50, 1.0 / n_assets + 0.1)

        # Classical
        t0 = time.perf_counter()
        classical = solve_markowitz_scipy(
            expected_returns=mu, covariance=cov,
            target_return=target_return, max_weight=max_weight,
        )
        classical_ms = (time.perf_counter() - t0) * 1000

        # QAOA
        solver = QAOASolver(config={
            "n_layers": qaoa_layers,
            "optimizer": "COBYLA",
            "maxiter": qaoa_maxiter,
            "n_shots": 2048,
            "seed": seed,
        })

        t0 = time.perf_counter()
        Q, offset = portfolio_to_qubo(
            expected_returns=mu, covariance=cov,
            target_return=target_return, max_weight=max_weight, n_bits=n_bits,
        )
        qaoa_result = solver.solve(Q)
        qaoa_ms = (time.perf_counter() - t0) * 1000

        # Brute force for small problems
        n_vars = Q.shape[0]
        bf_obj = None
        approx_ratio = None
        if n_vars <= 18:
            bf = solve_brute_force_qubo(Q)
            bf_obj = bf.objective_value
            if bf_obj != 0:
                approx_ratio = qaoa_result.objective_value / bf_obj

        results.append({
            "n_assets": n_assets,
            "n_qubits": n_vars,
            "data_source": data_source,
            "classical_runtime_ms": classical_ms,
            "classical_objective": classical.objective_value,
            "qaoa_runtime_ms": qaoa_ms,
            "qaoa_objective": qaoa_result.objective_value,
            "brute_force_objective": bf_obj,
            "approximation_ratio": approx_ratio,
            "runtime_ratio": qaoa_ms / max(classical_ms, 0.001),
        })

    # Compute scaling exponents
    qaoa_sizes = [r["n_qubits"] for r in results]
    qaoa_times = [r["qaoa_runtime_ms"] for r in results]
    classical_times = [r["classical_runtime_ms"] for r in results]

    experiment = {
        "experiment": "portfolio_scaling",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "asset_counts": asset_counts,
            "n_bits": n_bits,
            "qaoa_layers": qaoa_layers,
            "qaoa_maxiter": qaoa_maxiter,
        },
        "results": results,
        "scaling": {
            "qaoa": compute_scaling_exponents(qaoa_sizes, qaoa_times),
            "classical": compute_scaling_exponents(
                [r["n_assets"] for r in results], classical_times
            ),
        },
    }

    _log_experiment("portfolio_scaling", experiment)
    return experiment


# ---------------------------------------------------------------------------
# Experiment 2: Max-Cut scaling
# ---------------------------------------------------------------------------

def run_maxcut_scaling(
    node_counts: Optional[List[int]] = None,
    edge_prob: float = 0.5,
    qaoa_layers: int = 2,
    qaoa_maxiter: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep Max-Cut problem sizes: QAOA vs SA vs greedy vs brute-force.

    For each graph size, compares solution quality and runtime.
    """
    if node_counts is None:
        node_counts = [4, 6, 8, 10, 12]

    results = []
    for n in node_counts:
        Q = generate_maxcut_qubo(n, edge_prob=edge_prob, seed=seed + n)

        # Brute force (exact) for small problems
        bf_obj = None
        if n <= 16:
            bf = solve_brute_force_qubo(Q)
            bf_obj = bf.objective_value

        # QAOA
        solver = QAOASolver(config={
            "n_layers": qaoa_layers,
            "optimizer": "COBYLA",
            "maxiter": qaoa_maxiter,
            "n_shots": 2048,
            "seed": seed,
        })
        t0 = time.perf_counter()
        qaoa_result = solver.solve(Q)
        qaoa_ms = (time.perf_counter() - t0) * 1000

        # Simulated annealing
        t0 = time.perf_counter()
        sa_result = simulated_annealing_qubo(Q, n_iterations=5000, seed=seed)
        sa_ms = (time.perf_counter() - t0) * 1000

        # Greedy
        t0 = time.perf_counter()
        greedy_result = greedy_qubo(Q)
        greedy_ms = (time.perf_counter() - t0) * 1000

        entry = {
            "n_nodes": n,
            "brute_force_objective": bf_obj,
            "qaoa_objective": qaoa_result.objective_value,
            "qaoa_runtime_ms": qaoa_ms,
            "sa_objective": sa_result.objective_value,
            "sa_runtime_ms": sa_ms,
            "greedy_objective": greedy_result.objective_value,
            "greedy_runtime_ms": greedy_ms,
        }

        # Approximation ratios (for minimization: ratio < 1 is better)
        if bf_obj is not None and bf_obj != 0:
            entry["qaoa_approx_ratio"] = qaoa_result.objective_value / bf_obj
            entry["sa_approx_ratio"] = sa_result.objective_value / bf_obj
            entry["greedy_approx_ratio"] = greedy_result.objective_value / bf_obj

        results.append(entry)

    experiment = {
        "experiment": "maxcut_scaling",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "node_counts": node_counts,
            "edge_prob": edge_prob,
            "qaoa_layers": qaoa_layers,
            "qaoa_maxiter": qaoa_maxiter,
        },
        "results": results,
        "scaling": {
            "qaoa": compute_scaling_exponents(
                node_counts, [r["qaoa_runtime_ms"] for r in results]
            ),
            "sa": compute_scaling_exponents(
                node_counts, [r["sa_runtime_ms"] for r in results]
            ),
            "greedy": compute_scaling_exponents(
                node_counts, [r["greedy_runtime_ms"] for r in results]
            ),
        },
    }

    _log_experiment("maxcut_scaling", experiment)
    return experiment


# ---------------------------------------------------------------------------
# Experiment 3: C++ vs Python speedup
# ---------------------------------------------------------------------------

def run_cpp_speedup_benchmark(
    qubit_counts: Optional[List[int]] = None,
    n_iterations: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure C++ vs Python speedup across qubit counts."""
    from quantum.cpp import HAS_CPP

    if qubit_counts is None:
        qubit_counts = [4, 6, 8, 10, 12, 14]

    results = []
    for n in qubit_counts:
        dim = 2 ** n
        sv = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
        beta = 0.5

        # Python mixer
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            apply_mixer_unitary(sv, n, beta)
        py_ms = (time.perf_counter() - t0) * 1000 / n_iterations

        cpp_ms = None
        speedup = None
        if HAS_CPP:
            from quantum.cpp import apply_mixer_unitary as cpp_mixer
            t0 = time.perf_counter()
            for _ in range(n_iterations):
                cpp_mixer(sv, n, beta)
            cpp_ms = (time.perf_counter() - t0) * 1000 / n_iterations
            speedup = py_ms / max(cpp_ms, 1e-6)

        results.append({
            "n_qubits": n,
            "dim": dim,
            "python_ms": py_ms,
            "cpp_ms": cpp_ms,
            "speedup": speedup,
        })

    experiment = {
        "experiment": "cpp_speedup",
        "timestamp": datetime.now().isoformat(),
        "has_cpp": HAS_CPP,
        "n_iterations": n_iterations,
        "results": results,
    }

    _log_experiment("cpp_speedup", experiment)
    return experiment


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all_experiments() -> Dict[str, Any]:
    """Run the complete benchmark suite and log results."""
    logger.info("Starting full benchmark suite...")

    portfolio = run_portfolio_scaling()
    logger.info("Portfolio scaling complete: %d data points", len(portfolio["results"]))

    maxcut = run_maxcut_scaling()
    logger.info("Max-Cut scaling complete: %d data points", len(maxcut["results"]))

    cpp = run_cpp_speedup_benchmark()
    logger.info("C++ speedup benchmark complete: %d data points", len(cpp["results"]))

    summary = {
        "portfolio_scaling": portfolio,
        "maxcut_scaling": maxcut,
        "cpp_speedup": cpp,
    }

    _log_experiment("full_benchmark_suite", summary)
    return summary
