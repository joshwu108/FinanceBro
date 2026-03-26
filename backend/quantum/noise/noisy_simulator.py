"""Noisy quantum circuit simulator using density matrix formalism.

Simulates circuits with gate noise (depolarizing) and measurement error.
Uses density matrices (rho) instead of statevectors to represent mixed states.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from quantum.circuits.compiler.peephole import Circuit, Gate
from quantum.noise.noise_model import (
    NoiseModel,
    depolarizing_channel,
    readout_error_channel,
)


def _gate_matrix(name: str, params: List[float]) -> np.ndarray:
    """Return the unitary matrix for a single gate."""
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "rz":
        theta = params[0]
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=complex)
    if name == "ry":
        theta = params[0]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rx":
        theta = params[0]
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    raise ValueError(f"Unknown gate: {name}")


def _apply_single_qubit_gate(
    rho: np.ndarray, U: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply single-qubit unitary to density matrix."""
    dim = 1 << n_qubits
    mask = 1 << qubit
    rho_new = np.zeros_like(rho)

    for i in range(dim):
        for j in range(dim):
            if rho[i, j] == 0:
                continue
            i_bit = (i >> qubit) & 1
            j_bit = (j >> qubit) & 1
            for a in range(2):
                for b in range(2):
                    new_i = (i & ~mask) | (a << qubit)
                    new_j = (j & ~mask) | (b << qubit)
                    rho_new[new_i, new_j] += U[a, i_bit] * rho[i, j] * U[b, j_bit].conj()

    return rho_new


def _apply_cnot(rho: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT to density matrix."""
    dim = 1 << n_qubits
    c_mask = 1 << control
    t_mask = 1 << target

    # Build CNOT permutation
    perm = np.arange(dim)
    for i in range(dim):
        if i & c_mask:
            perm[i] = i ^ t_mask

    # Apply: rho' = P rho P^T (permutation is real and self-adjoint)
    rho_new = rho[np.ix_(perm, perm)]
    return rho_new


def _apply_noise_channel(
    rho: np.ndarray,
    kraus_ops: List[np.ndarray],
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply a single-qubit noise channel (Kraus operators) to density matrix."""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += _apply_single_qubit_gate(
            _apply_single_qubit_gate(rho, K, qubit, n_qubits),
            np.eye(2, dtype=complex),
            qubit,
            n_qubits,
        )
    # Actually, for Kraus: rho' = sum_k K_k rho K_k^dag
    # Using the _apply_single_qubit_gate for U rho U^dag:
    # We need: sum_k (K_k rho K_k^dag)
    return result


def _apply_kraus_channel(
    rho: np.ndarray,
    kraus_ops: List[np.ndarray],
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply Kraus channel: rho -> sum_k K_k rho K_k^dag."""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += _apply_single_qubit_gate(rho, K, qubit, n_qubits)
    return result


class NoisySimulator:
    """Density-matrix simulator with configurable noise."""

    def __init__(self, noise_model: Optional[NoiseModel] = None) -> None:
        self.noise_model = noise_model

    def run(
        self,
        circuit: Circuit,
        n_shots: int = 1024,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Simulate circuit and return probability distribution.

        Returns dict mapping bitstring -> probability.
        """
        n = circuit.n_qubits
        dim = 1 << n

        # Initialize |0...0><0...0|
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        # Apply gates with noise
        for gate in circuit.gates:
            rho = self._apply_gate(rho, gate, n)

        # Extract probabilities from diagonal
        probs = np.real(np.diag(rho))
        probs = np.maximum(probs, 0)  # numerical fix

        # Apply readout error
        if self.noise_model and self.noise_model.readout_error > 0:
            probs = self._apply_readout_error(probs, n)

        # Normalize
        total = np.sum(probs)
        if total > 0:
            probs = probs / total

        # Sample shots
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n_shots, probs)

        # Convert to bitstring -> probability
        result = {}
        for i, count in enumerate(counts):
            if count > 0:
                bitstring = format(i, f"0{n}b")
                result[bitstring] = count / n_shots

        return result

    def _apply_gate(self, rho: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
        """Apply a gate and optional noise."""
        if gate.name == "cx":
            rho = _apply_cnot(rho, gate.qubits[0], gate.qubits[1], n_qubits)
            # Two-qubit noise on both qubits
            if self.noise_model and self.noise_model.two_qubit_error > 0:
                kraus = depolarizing_channel(self.noise_model.two_qubit_error)
                for q in gate.qubits:
                    rho = _apply_kraus_channel(rho, kraus, q, n_qubits)
        else:
            U = _gate_matrix(gate.name, gate.params)
            rho = _apply_single_qubit_gate(rho, U, gate.qubits[0], n_qubits)
            # Single-qubit noise
            if self.noise_model and self.noise_model.single_qubit_error > 0:
                kraus = depolarizing_channel(self.noise_model.single_qubit_error)
                rho = _apply_kraus_channel(rho, kraus, gate.qubits[0], n_qubits)

        return rho

    def _apply_readout_error(self, probs: np.ndarray, n_qubits: int) -> np.ndarray:
        """Apply per-qubit readout error to measurement probabilities."""
        confusion = readout_error_channel(self.noise_model.readout_error)
        dim = len(probs)
        new_probs = np.zeros(dim)

        for i in range(dim):
            if probs[i] < 1e-15:
                continue
            # For each qubit, apply confusion matrix independently
            for j in range(dim):
                p = probs[i]
                for q in range(n_qubits):
                    i_bit = (i >> q) & 1
                    j_bit = (j >> q) & 1
                    p *= confusion[i_bit, j_bit]
                new_probs[j] += p

        return new_probs
