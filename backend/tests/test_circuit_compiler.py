"""Tests for Quantum Circuit Compiler/Optimizer — TDD: tests written first.

Covers gate representation, peephole optimization, depth reduction,
and multi-pass optimization pipeline.
"""

import numpy as np
import pytest

from quantum.circuits.compiler.optimizer import (
    CircuitOptimizer,
    OptimizationResult,
)
from quantum.circuits.compiler.peephole import (
    PeepholeOptimizer,
    Gate,
    Circuit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_circuit():
    """Circuit with two consecutive H gates on same qubit (H*H = I)."""
    return Circuit(n_qubits=1, gates=[
        Gate("h", qubits=[0]),
        Gate("h", qubits=[0]),
    ])


@pytest.fixture
def double_cx_circuit():
    """Two consecutive CX gates on same qubits (CX*CX = I)."""
    return Circuit(n_qubits=2, gates=[
        Gate("cx", qubits=[0, 1]),
        Gate("cx", qubits=[0, 1]),
    ])


@pytest.fixture
def rotation_fusion_circuit():
    """Two Rz rotations on same qubit that should fuse."""
    return Circuit(n_qubits=1, gates=[
        Gate("rz", qubits=[0], params=[0.3]),
        Gate("rz", qubits=[0], params=[0.7]),
    ])


@pytest.fixture
def mixed_circuit():
    """Multi-qubit circuit with optimization opportunities."""
    return Circuit(n_qubits=3, gates=[
        Gate("h", qubits=[0]),
        Gate("cx", qubits=[0, 1]),
        Gate("rz", qubits=[1], params=[0.5]),
        Gate("rz", qubits=[1], params=[0.3]),   # fuse with above
        Gate("h", qubits=[2]),
        Gate("h", qubits=[2]),                   # cancel with above
        Gate("cx", qubits=[1, 2]),
    ])


@pytest.fixture
def no_optimization_circuit():
    """Circuit with no optimization opportunities."""
    return Circuit(n_qubits=2, gates=[
        Gate("h", qubits=[0]),
        Gate("cx", qubits=[0, 1]),
        Gate("rz", qubits=[1], params=[0.5]),
    ])


# ===========================================================================
# 1. Gate and Circuit data structures
# ===========================================================================

class TestGate:

    def test_gate_creation(self):
        g = Gate("h", qubits=[0])
        assert g.name == "h"
        assert g.qubits == [0]
        assert g.params == []

    def test_gate_with_params(self):
        g = Gate("rz", qubits=[0], params=[1.5])
        assert g.params == [1.5]

    def test_two_qubit_gate(self):
        g = Gate("cx", qubits=[0, 1])
        assert len(g.qubits) == 2

    def test_gate_equality(self):
        g1 = Gate("h", qubits=[0])
        g2 = Gate("h", qubits=[0])
        assert g1 == g2

    def test_gate_inequality(self):
        g1 = Gate("h", qubits=[0])
        g2 = Gate("x", qubits=[0])
        assert g1 != g2


class TestCircuit:

    def test_circuit_creation(self):
        c = Circuit(n_qubits=2, gates=[Gate("h", qubits=[0])])
        assert c.n_qubits == 2
        assert len(c.gates) == 1

    def test_empty_circuit(self):
        c = Circuit(n_qubits=1, gates=[])
        assert len(c.gates) == 0

    def test_gate_count(self):
        c = Circuit(n_qubits=2, gates=[
            Gate("h", qubits=[0]),
            Gate("cx", qubits=[0, 1]),
            Gate("rz", qubits=[1], params=[0.5]),
        ])
        assert c.gate_count == 3

    def test_depth_single_qubit(self):
        """Sequential gates on one qubit: depth = gate count."""
        c = Circuit(n_qubits=1, gates=[
            Gate("h", qubits=[0]),
            Gate("rz", qubits=[0], params=[0.5]),
            Gate("h", qubits=[0]),
        ])
        assert c.depth == 3

    def test_depth_parallel_gates(self):
        """Independent gates on different qubits: depth = 1."""
        c = Circuit(n_qubits=3, gates=[
            Gate("h", qubits=[0]),
            Gate("h", qubits=[1]),
            Gate("h", qubits=[2]),
        ])
        assert c.depth == 1

    def test_depth_mixed(self):
        """H(0), CX(0,1), H(2) => depth = 2 (CX after H(0), H(2) parallel)."""
        c = Circuit(n_qubits=3, gates=[
            Gate("h", qubits=[0]),
            Gate("cx", qubits=[0, 1]),
            Gate("h", qubits=[2]),
        ])
        assert c.depth == 2

    def test_cx_count(self):
        c = Circuit(n_qubits=3, gates=[
            Gate("h", qubits=[0]),
            Gate("cx", qubits=[0, 1]),
            Gate("cx", qubits=[1, 2]),
            Gate("rz", qubits=[0], params=[0.5]),
        ])
        assert c.cx_count == 2


# ===========================================================================
# 2. Peephole optimization
# ===========================================================================

class TestPeepholeOptimizer:

    def test_cancel_double_h(self, identity_circuit):
        """H*H = I => remove both gates."""
        opt = PeepholeOptimizer()
        result = opt.optimize(identity_circuit)
        assert result.gate_count == 0

    def test_cancel_double_cx(self, double_cx_circuit):
        """CX*CX = I => remove both gates."""
        opt = PeepholeOptimizer()
        result = opt.optimize(double_cx_circuit)
        assert result.gate_count == 0

    def test_fuse_rz_rotations(self, rotation_fusion_circuit):
        """Rz(a)*Rz(b) = Rz(a+b)."""
        opt = PeepholeOptimizer()
        result = opt.optimize(rotation_fusion_circuit)
        assert result.gate_count == 1
        assert result.gates[0].name == "rz"
        assert np.isclose(result.gates[0].params[0], 1.0, atol=1e-10)

    def test_remove_zero_rotation(self):
        """Rz(0) = I => remove."""
        c = Circuit(n_qubits=1, gates=[Gate("rz", qubits=[0], params=[0.0])])
        opt = PeepholeOptimizer()
        result = opt.optimize(c)
        assert result.gate_count == 0

    def test_remove_2pi_rotation(self):
        """Rz(2*pi) = I => remove."""
        c = Circuit(n_qubits=1, gates=[
            Gate("rz", qubits=[0], params=[2 * np.pi]),
        ])
        opt = PeepholeOptimizer()
        result = opt.optimize(c)
        assert result.gate_count == 0

    def test_mixed_circuit_optimization(self, mixed_circuit):
        """Mixed circuit: fuse Rz, cancel H*H."""
        opt = PeepholeOptimizer()
        result = opt.optimize(mixed_circuit)
        # Original: 7 gates. After: fuse 2 Rz -> 1, cancel H*H -> 0
        # Remaining: H(0), CX(0,1), Rz(1,0.8), CX(1,2) = 4 gates
        assert result.gate_count < mixed_circuit.gate_count

    def test_no_optimization_unchanged(self, no_optimization_circuit):
        """Circuit with no optimization opportunities stays the same."""
        opt = PeepholeOptimizer()
        result = opt.optimize(no_optimization_circuit)
        assert result.gate_count == no_optimization_circuit.gate_count

    def test_cancel_double_x(self):
        """X*X = I."""
        c = Circuit(n_qubits=1, gates=[
            Gate("x", qubits=[0]),
            Gate("x", qubits=[0]),
        ])
        opt = PeepholeOptimizer()
        result = opt.optimize(c)
        assert result.gate_count == 0

    def test_preserves_qubit_count(self, mixed_circuit):
        opt = PeepholeOptimizer()
        result = opt.optimize(mixed_circuit)
        assert result.n_qubits == mixed_circuit.n_qubits


# ===========================================================================
# 3. CircuitOptimizer (multi-pass)
# ===========================================================================

class TestCircuitOptimizer:

    def test_optimize_returns_result(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert isinstance(result, OptimizationResult)

    def test_result_has_metrics(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert hasattr(result, "original_gate_count")
        assert hasattr(result, "optimized_gate_count")
        assert hasattr(result, "original_depth")
        assert hasattr(result, "optimized_depth")
        assert hasattr(result, "optimized_circuit")

    def test_gate_count_improves_or_equal(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert result.optimized_gate_count <= result.original_gate_count

    def test_depth_improves_or_equal(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert result.optimized_depth <= result.original_depth

    def test_no_optimization_preserves_circuit(self, no_optimization_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(no_optimization_circuit)
        assert result.optimized_gate_count == result.original_gate_count

    def test_improvement_percentages(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert hasattr(result, "gate_count_reduction_pct")
        assert result.gate_count_reduction_pct >= 0.0
        assert result.gate_count_reduction_pct <= 100.0

    def test_pass_log_recorded(self, mixed_circuit):
        opt = CircuitOptimizer()
        result = opt.optimize(mixed_circuit)
        assert hasattr(result, "pass_log")
        assert len(result.pass_log) > 0

    def test_empty_circuit(self):
        c = Circuit(n_qubits=1, gates=[])
        opt = CircuitOptimizer()
        result = opt.optimize(c)
        assert result.optimized_gate_count == 0
        assert result.optimized_depth == 0

    def test_large_cancellation_chain(self):
        """Long chain of H*H pairs should all cancel."""
        gates = []
        for _ in range(50):
            gates.append(Gate("h", qubits=[0]))
            gates.append(Gate("h", qubits=[0]))
        c = Circuit(n_qubits=1, gates=gates)
        opt = CircuitOptimizer()
        result = opt.optimize(c)
        assert result.optimized_gate_count == 0
