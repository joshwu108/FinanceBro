"""Peephole optimizer for quantum circuits.

Applies local gate cancellation and fusion rules:
  - Self-inverse cancellation: H*H, X*X, CX*CX, etc. = I
  - Rotation fusion: Rz(a)*Rz(b) = Rz(a+b)
  - Identity removal: Rz(0), Rz(2*pi) = I
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class Gate:
    """Single quantum gate with name, target qubits, and optional parameters."""

    __slots__ = ("name", "qubits", "params")

    def __init__(
        self,
        name: str,
        qubits: List[int],
        params: Optional[List[float]] = None,
    ) -> None:
        self.name = name
        self.qubits = list(qubits)
        self.params = list(params) if params is not None else []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.name == other.name
            and self.qubits == other.qubits
            and self.params == other.params
        )

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self) -> str:
        if self.params:
            return f"Gate({self.name!r}, {self.qubits}, {self.params})"
        return f"Gate({self.name!r}, {self.qubits})"


class Circuit:
    """Quantum circuit as an ordered gate list on a fixed number of qubits."""

    __slots__ = ("n_qubits", "gates")

    def __init__(self, n_qubits: int, gates: List[Gate]) -> None:
        self.n_qubits = n_qubits
        self.gates = list(gates)

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def depth(self) -> int:
        """Circuit depth: longest chain of dependent gates across all qubits."""
        if not self.gates:
            return 0
        qubit_depth = [0] * self.n_qubits
        for gate in self.gates:
            max_dep = max(qubit_depth[q] for q in gate.qubits)
            new_depth = max_dep + 1
            for q in gate.qubits:
                qubit_depth[q] = new_depth
        return max(qubit_depth)

    @property
    def cx_count(self) -> int:
        return sum(1 for g in self.gates if g.name == "cx")


# Gates where G*G = I (no parameters)
_SELF_INVERSE = frozenset({"h", "x", "y", "z", "cx", "cz", "swap"})

# Rotation gates: Rg(a)*Rg(b) = Rg(a+b)
_ROTATION_GATES = frozenset({"rz", "rx", "ry"})


def _is_zero_rotation(angle: float) -> bool:
    """Check if rotation angle is effectively 0 mod 2*pi."""
    mod = angle % (2 * np.pi)
    return np.isclose(mod, 0.0, atol=1e-10) or np.isclose(mod, 2 * np.pi, atol=1e-10)


class PeepholeOptimizer:
    """Local peephole optimizer: cancel adjacent self-inverse gates, fuse rotations."""

    def optimize(self, circuit: Circuit) -> Circuit:
        """Apply peephole rules repeatedly until convergence."""
        gates = list(circuit.gates)
        changed = True
        while changed:
            gates, changed = self._one_pass(gates)
        return Circuit(n_qubits=circuit.n_qubits, gates=gates)

    def _one_pass(self, gates: List[Gate]) -> tuple:
        result: List[Gate] = []
        changed = False
        i = 0

        while i < len(gates):
            g = gates[i]

            # Rule: remove zero/2*pi single rotations
            if (
                g.name in _ROTATION_GATES
                and len(g.params) == 1
                and _is_zero_rotation(g.params[0])
            ):
                changed = True
                i += 1
                continue

            # Look-ahead rules require a next gate
            if i + 1 < len(gates):
                g_next = gates[i + 1]

                # Rule: self-inverse cancellation (parameterless)
                if (
                    g.name in _SELF_INVERSE
                    and g.name == g_next.name
                    and g.qubits == g_next.qubits
                    and not g.params
                    and not g_next.params
                ):
                    changed = True
                    i += 2
                    continue

                # Rule: rotation fusion
                if (
                    g.name in _ROTATION_GATES
                    and g.name == g_next.name
                    and g.qubits == g_next.qubits
                    and len(g.params) == 1
                    and len(g_next.params) == 1
                ):
                    fused_angle = g.params[0] + g_next.params[0]
                    if _is_zero_rotation(fused_angle):
                        # Fused to identity — drop both
                        changed = True
                        i += 2
                        continue
                    else:
                        result.append(Gate(g.name, g.qubits, [fused_angle]))
                        changed = True
                        i += 2
                        continue

            result.append(g)
            i += 1

        return result, changed
