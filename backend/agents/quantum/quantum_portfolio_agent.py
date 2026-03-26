"""QuantumPortfolioAgent — Portfolio optimization via QAOA + classical baselines.

Solves the constrained portfolio allocation problem using both classical
(CVXPY mean-variance) and quantum (QAOA on QUBO) approaches, then
rigorously compares runtime, solution quality, and approximation ratio.

Constraints (inherited from PortfolioAgent):
  - Covariance estimated via Ledoit-Wolf shrinkage
  - Long-only by default
  - Max position weight: configurable
  - No look-ahead: weights at time t use data up to t
"""

import logging
import time
from types import MappingProxyType
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from agents.base_agent import BaseAgent
from quantum.solvers.classical_solvers import (
    SolverResult,
    solve_markowitz_cvxpy,
    solve_markowitz_scipy,
    efficient_frontier,
)
from quantum.solvers.problem_encodings import decode_binary_weights, portfolio_to_qubo
from quantum.solvers.qaoa_solver import QAOASolver

logger = logging.getLogger(__name__)


class QuantumPortfolioAgent(BaseAgent):
    """Portfolio optimization via QAOA + classical baselines.

    Methods available:
      - markowitz_cvxpy: Classical convex optimization (baseline)
      - markowitz_scipy: Scipy fallback (baseline)
      - qaoa: Quantum Approximate Optimization Algorithm

    All methods respect long-only constraints, position limits,
    and use Ledoit-Wolf covariance.
    """

    DEFAULT_CONFIG: MappingProxyType = MappingProxyType({
        "methods": ["markowitz_cvxpy", "qaoa"],
        "max_weight": 0.10,
        "covariance_window": 252,
        "qaoa_layers": 3,
        "weight_precision_bits": 3,
        "qaoa_optimizer": "COBYLA",
        "qaoa_maxiter": 500,
        "qaoa_n_shots": 4096,
        "qaoa_seed": None,
        "penalty_budget": 10.0,
        "penalty_return": 5.0,
        "frontier_points": 0,
    })

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ────────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "returns": "pd.DataFrame — daily returns, columns=tickers, index=dates",
            "expected_returns": "(optional) np.ndarray — override return estimates",
            "config": "(optional) dict — runtime config overrides",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "classical_weights": "np.ndarray — Markowitz optimal weights",
            "classical_objective": "float — portfolio variance",
            "classical_runtime_ms": "float",
            "quantum_weights": "(if qaoa in methods) np.ndarray",
            "quantum_objective": "(if qaoa) float",
            "quantum_runtime_ms": "(if qaoa) float",
            "quantum_metadata": "(if qaoa) dict",
            "comparison": "(if both methods) dict with weight_distance, runtime_ratio",
            "efficient_frontier": "(if frontier_points > 0) dict",
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio optimization.

        Runs all configured methods and compares results.
        """
        returns_df: pd.DataFrame = inputs["returns"]
        runtime_config = inputs.get("config", {})
        cfg = {**self._config, **runtime_config}

        methods = cfg["methods"]
        max_weight = cfg["max_weight"]

        # Estimate expected returns and covariance
        mu = np.array(returns_df.mean())
        cov = self._estimate_covariance(returns_df)
        n_assets = len(mu)
        target_return = float(np.mean(mu))

        result: Dict[str, Any] = {"n_assets": n_assets, "tickers": list(returns_df.columns)}

        # ── Classical baseline ──
        if "markowitz_cvxpy" in methods or "markowitz_scipy" in methods:
            c_result = self._run_classical(mu, cov, target_return, max_weight, methods)
            result["classical_weights"] = c_result.weights
            result["classical_objective"] = c_result.objective_value
            result["classical_runtime_ms"] = c_result.runtime_ms

        # ── QAOA ──
        if "qaoa" in methods:
            q_result = self._run_qaoa(mu, cov, target_return, cfg)
            decoded = decode_binary_weights(
                q_result.weights,
                n_assets,
                cfg["weight_precision_bits"],
                max_weight,
            )
            result["quantum_weights"] = decoded
            result["quantum_objective"] = q_result.objective_value
            result["quantum_runtime_ms"] = q_result.runtime_ms
            result["quantum_metadata"] = q_result.metadata

        # ── Comparison ──
        if "classical_weights" in result and "quantum_weights" in result:
            cw = result["classical_weights"]
            qw = result["quantum_weights"]
            result["comparison"] = {
                "weight_distance": float(np.linalg.norm(cw - qw)),
                "runtime_ratio": (
                    result["quantum_runtime_ms"] / result["classical_runtime_ms"]
                    if result["classical_runtime_ms"] > 0
                    else float("inf")
                ),
                "objective_classical": result["classical_objective"],
                "objective_quantum": result["quantum_objective"],
            }

        # ── Efficient frontier ──
        frontier_pts = cfg["frontier_points"]
        if frontier_pts > 0:
            result["efficient_frontier"] = self._build_frontier(
                mu, cov, frontier_pts, max_weight
            )

        self._metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in result.items()
            if k != "efficient_frontier"
        }
        return result

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate inputs and outputs."""
        if "returns" not in inputs:
            raise ValueError("Missing required input: 'returns'")

        returns_df = inputs["returns"]
        if not isinstance(returns_df, pd.DataFrame):
            raise ValueError("'returns' must be a pandas DataFrame")

        if "classical_weights" in outputs:
            w = outputs["classical_weights"]
            if np.any(np.isnan(w)):
                raise ValueError("Classical weights contain NaN")

        if "quantum_weights" in outputs:
            w = outputs["quantum_weights"]
            if np.any(np.isnan(w)):
                raise ValueError("Quantum weights contain NaN")

        return True

    def log_metrics(self) -> None:
        """Log metrics from most recent run."""
        if not self._metrics:
            logger.info("QuantumPortfolioAgent: no metrics to log (run() not called)")
            return

        logger.info("QuantumPortfolioAgent metrics: %s", {
            k: v for k, v in self._metrics.items()
            if k not in ("quantum_metadata",)
        })

    # ── Private methods ────────────────────────────────────────────

    def _estimate_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimation."""
        lw = LedoitWolf().fit(returns_df.values)
        return lw.covariance_

    def _run_classical(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float,
        max_weight: float,
        methods: list,
    ) -> SolverResult:
        """Run classical Markowitz solver."""
        if "markowitz_cvxpy" in methods:
            try:
                return solve_markowitz_cvxpy(
                    expected_returns=mu,
                    covariance=cov,
                    target_return=target_return,
                    max_weight=max_weight,
                )
            except Exception:
                logger.warning("cvxpy solver failed, falling back to scipy")

        return solve_markowitz_scipy(
            expected_returns=mu,
            covariance=cov,
            target_return=target_return,
            max_weight=max_weight,
        )

    def _run_qaoa(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float,
        cfg: Dict[str, Any],
    ) -> SolverResult:
        """Run QAOA portfolio optimization."""
        solver = QAOASolver(config={
            "n_layers": cfg["qaoa_layers"],
            "optimizer": cfg["qaoa_optimizer"],
            "maxiter": cfg["qaoa_maxiter"],
            "n_shots": cfg["qaoa_n_shots"],
            "seed": cfg.get("qaoa_seed"),
        })

        return solver.solve_portfolio(
            expected_returns=mu,
            covariance=cov,
            target_return=target_return,
            max_weight=cfg["max_weight"],
            n_bits=cfg["weight_precision_bits"],
        )

    def _build_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n_points: int,
        max_weight: float,
    ) -> Dict[str, Any]:
        """Generate efficient frontier points."""
        frontier_results = efficient_frontier(
            expected_returns=mu,
            covariance=cov,
            n_points=n_points,
            max_weight=max_weight,
        )
        risks = [float(np.sqrt(r.objective_value)) for r in frontier_results]
        rets = [float(mu @ r.weights) for r in frontier_results]
        return {"risks": risks, "returns": rets}
