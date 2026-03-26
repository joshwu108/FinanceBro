"""Benchmark runner for quantum vs classical solver comparison.

Every quantum experiment is wrapped in this standardised framework
to ensure fair, reproducible, and logged comparisons.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantum.solvers.classical_solvers import SolverResult
from quantum.benchmarks.metrics import ComparisonMetrics, compute_comparison_metrics


def _serialize_solver_result(sr: SolverResult) -> Dict[str, Any]:
    """Convert a SolverResult to a JSON-safe dict."""
    return {
        "weights": sr.weights.tolist() if isinstance(sr.weights, np.ndarray) else list(sr.weights),
        "objective_value": float(sr.objective_value),
        "runtime_ms": float(sr.runtime_ms),
        "converged": bool(sr.converged),
        "iterations": int(sr.iterations),
        "method": str(sr.method),
        "metadata": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in sr.metadata.items()
        } if sr.metadata else {},
    }


@dataclass
class BenchmarkResult:
    """Full benchmark result for one experiment."""

    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    classical_results: List[Dict[str, Any]]
    quantum_results: List[Dict[str, Any]]
    comparison: Dict[str, Any]
    problem_description: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "classical_results": self.classical_results,
            "quantum_results": self.quantum_results,
            "comparison": self.comparison,
            "problem_description": self.problem_description,
            "notes": self.notes,
        }

    @classmethod
    def from_solver_results(
        cls,
        classical: List[SolverResult],
        quantum: List[SolverResult],
        config: Dict[str, Any],
        problem_description: str = "",
    ) -> "BenchmarkResult":
        """Create BenchmarkResult from solver results."""
        c_objs = [r.objective_value for r in classical]
        q_objs = [r.objective_value for r in quantum]
        c_rts = [r.runtime_ms for r in classical]
        q_rts = [r.runtime_ms for r in quantum]

        metrics = compute_comparison_metrics(c_objs, q_objs, c_rts, q_rts)

        return cls(
            experiment_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=config,
            classical_results=[_serialize_solver_result(r) for r in classical],
            quantum_results=[_serialize_solver_result(r) for r in quantum],
            comparison=asdict(metrics),
            problem_description=problem_description,
        )


class BenchmarkRunner:
    """Orchestrates quantum vs classical comparisons."""

    def __init__(self, experiments_dir: str = "experiments") -> None:
        self._experiments_dir = Path(experiments_dir)
        self._experiments_dir.mkdir(parents=True, exist_ok=True)

    def run_comparison(
        self,
        problem: np.ndarray,
        classical_solver: Callable,
        quantum_solver: Callable,
        n_repetitions: int = 10,
        config: Optional[Dict[str, Any]] = None,
        problem_description: str = "",
    ) -> BenchmarkResult:
        """Run both solvers n_repetitions times and compare."""
        cfg = config or {}

        classical_results: List[SolverResult] = []
        quantum_results: List[SolverResult] = []

        # Classical solver may be deterministic — run it n_repetitions times anyway
        # so the caller sees equal-length lists.
        for _ in range(n_repetitions):
            classical_results.append(classical_solver(problem))

        for _ in range(n_repetitions):
            quantum_results.append(quantum_solver(problem))

        result = BenchmarkResult.from_solver_results(
            classical=classical_results,
            quantum=quantum_results,
            config=cfg,
            problem_description=problem_description,
        )
        return result

    def run_scaling_analysis(
        self,
        problem_generator: Callable,
        classical_solver: Callable,
        quantum_solver: Callable,
        problem_sizes: List[int],
        n_repetitions: int = 5,
    ) -> List[BenchmarkResult]:
        """Run comparison at multiple problem sizes."""
        results: List[BenchmarkResult] = []
        for size in problem_sizes:
            Q = problem_generator(size)
            bench = self.run_comparison(
                problem=Q,
                classical_solver=classical_solver,
                quantum_solver=quantum_solver,
                n_repetitions=n_repetitions,
                config={"problem_size": size},
                problem_description=f"Scaling analysis, size={size}",
            )
            results.append(bench)
        return results

    def log_experiment(self, result: BenchmarkResult) -> Path:
        """Save benchmark result to experiments/ directory as JSON."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_benchmark_{ts}_{result.experiment_id}.json"
        path = self._experiments_dir / filename

        payload = result.to_dict()
        payload["agent"] = "BenchmarkRunner"

        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        return path
