"""Metrics for quantum vs classical comparison."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ComparisonMetrics:
    """Comparison between classical and quantum solver results."""

    approximation_ratio: float
    runtime_ratio: float
    quantum_wins_pct: float
    mean_objective_classical: float
    mean_objective_quantum: float
    std_objective_classical: float
    std_objective_quantum: float
    mean_runtime_classical_ms: float
    mean_runtime_quantum_ms: float
    p_value: Optional[float] = None


def compute_comparison_metrics(
    classical_objectives: List[float],
    quantum_objectives: List[float],
    classical_runtimes: List[float],
    quantum_runtimes: List[float],
) -> ComparisonMetrics:
    """Compute comparison metrics from repeated runs.

    For minimisation problems:
    - approximation_ratio = mean(classical_obj) / mean(quantum_obj)
      > 1 means quantum is better (lower objective)
    - quantum_wins_pct = fraction where quantum_obj < classical_obj

    Uses scipy.stats.wilcoxon for p-value if n >= 5 and there is
    non-zero variance in the paired differences.
    """
    c_obj = np.asarray(classical_objectives, dtype=float)
    q_obj = np.asarray(quantum_objectives, dtype=float)
    c_rt = np.asarray(classical_runtimes, dtype=float)
    q_rt = np.asarray(quantum_runtimes, dtype=float)

    mean_c = float(np.mean(c_obj))
    mean_q = float(np.mean(q_obj))
    mean_c_rt = float(np.mean(c_rt))
    mean_q_rt = float(np.mean(q_rt))

    if mean_q != 0.0:
        approx_ratio = mean_c / mean_q
    else:
        approx_ratio = 1.0 if mean_c == 0.0 else float("inf")

    runtime_ratio = mean_q_rt / mean_c_rt if mean_c_rt != 0.0 else float("inf")

    wins = int(np.sum(q_obj < c_obj))
    n = len(c_obj)
    quantum_wins_pct = wins / n if n > 0 else 0.0

    p_value: Optional[float] = None
    if n >= 5:
        diffs = c_obj - q_obj
        if np.any(diffs != 0):
            try:
                from scipy.stats import wilcoxon

                _, p_value = wilcoxon(diffs)
                p_value = float(p_value)
            except Exception:
                p_value = None

    return ComparisonMetrics(
        approximation_ratio=approx_ratio,
        runtime_ratio=runtime_ratio,
        quantum_wins_pct=quantum_wins_pct,
        mean_objective_classical=mean_c,
        mean_objective_quantum=mean_q,
        std_objective_classical=float(np.std(c_obj, ddof=0)),
        std_objective_quantum=float(np.std(q_obj, ddof=0)),
        mean_runtime_classical_ms=mean_c_rt,
        mean_runtime_quantum_ms=mean_q_rt,
        p_value=p_value,
    )
