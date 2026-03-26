"""Quantum feature maps for encoding classical data into quantum states.

Two encoding strategies:
  - Angle encoding: features become rotation angles (Ry gates)
  - Amplitude encoding: features become statevector amplitudes

The quantum kernel K(x, x') = |<phi(x)|phi(x')>|^2 measures
similarity in Hilbert space.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


def angle_encode(features: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encode a feature vector as rotation angles on n_qubits.

    Applies Ry(theta_i) to qubit i, where theta_i = features[i % len(features)] * pi.
    Starts from |0...0>.

    Returns:
        Complex statevector of length 2^n_qubits, unit norm.
    """
    n = n_qubits
    dim = 1 << n
    sv = np.zeros(dim, dtype=complex)
    sv[0] = 1.0  # |0...0>

    n_features = len(features)

    for q in range(n):
        theta = features[q % n_features] * np.pi
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        mask = 1 << q
        new_sv = np.zeros_like(sv)
        for i in range(dim):
            if i & mask == 0:
                j = i | mask
                new_sv[i] += c * sv[i]
                new_sv[j] += s * sv[i]
                new_sv[i] += -s * sv[j]
                new_sv[j] += c * sv[j]
        sv = new_sv

    return sv


def amplitude_encode(features: np.ndarray) -> np.ndarray:
    """Encode features as amplitudes of a quantum state.

    Pads to nearest power of 2 and normalizes.

    Returns:
        Complex statevector, unit norm.
    """
    n = len(features)
    n_qubits = max(1, math.ceil(math.log2(n))) if n > 1 else 1
    dim = 1 << n_qubits

    padded = np.zeros(dim, dtype=complex)
    padded[:n] = features.astype(complex)

    norm = np.linalg.norm(padded)
    if norm > 1e-15:
        padded /= norm

    return padded


class QuantumFeatureMap:
    """Batch feature encoding and quantum kernel computation."""

    def __init__(
        self,
        n_qubits: int,
        encoding: Literal["angle", "amplitude"] = "angle",
    ) -> None:
        self.n_qubits = n_qubits
        self.encoding = encoding

    def _encode_single(self, x: np.ndarray) -> np.ndarray:
        if self.encoding == "angle":
            return angle_encode(x, self.n_qubits)
        return amplitude_encode(x)

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode each row of X into a quantum state.

        Returns:
            Array of shape (n_samples, 2^n_qubits).
        """
        return np.array([self._encode_single(x) for x in X])

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute quantum kernel: K(i,j) = |<phi(x_i)|phi(x_j)>|^2."""
        states = self.encode_batch(X)
        n = len(states)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                overlap = np.vdot(states[i], states[j])
                K[i, j] = float(np.abs(overlap) ** 2)
        return K
