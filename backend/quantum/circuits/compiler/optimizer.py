"""Multi-pass circuit optimizer.

Orchestrates multiple optimization passes (peephole, depth reduction, etc.)
and reports before/after metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from quantum.circuits.compiler.peephole import Circuit, PeepholeOptimizer


@dataclass
class OptimizationResult:
    """Result of multi-pass circuit optimization."""

    original_gate_count: int
    optimized_gate_count: int
    original_depth: int
    optimized_depth: int
    optimized_circuit: Circuit
    gate_count_reduction_pct: float
    pass_log: List[Dict[str, Any]] = field(default_factory=list)


class CircuitOptimizer:
    """Multi-pass circuit optimizer combining peephole and future strategies."""

    def __init__(self) -> None:
        self._peephole = PeepholeOptimizer()

    def optimize(self, circuit: Circuit) -> OptimizationResult:
        original_gate_count = circuit.gate_count
        original_depth = circuit.depth

        pass_log: List[Dict[str, Any]] = []
        current = circuit

        # Pass 1: Peephole optimization
        optimized = self._peephole.optimize(current)
        pass_log.append({
            "pass": "peephole",
            "gates_before": current.gate_count,
            "gates_after": optimized.gate_count,
            "depth_before": current.depth,
            "depth_after": optimized.depth,
        })
        current = optimized

        # Compute final metrics
        optimized_gate_count = current.gate_count
        optimized_depth = current.depth

        if original_gate_count > 0:
            reduction_pct = (1.0 - optimized_gate_count / original_gate_count) * 100.0
        else:
            reduction_pct = 0.0

        return OptimizationResult(
            original_gate_count=original_gate_count,
            optimized_gate_count=optimized_gate_count,
            original_depth=original_depth,
            optimized_depth=optimized_depth,
            optimized_circuit=current,
            gate_count_reduction_pct=max(reduction_pct, 0.0),
            pass_log=pass_log,
        )
