"""Quantum noise models — Kraus operator representations.

Supported channels:
  - Depolarizing: symmetric noise on all Pauli axes
  - Amplitude damping: energy relaxation (T1 decay)
  - Readout error: classical bit-flip on measurement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def depolarizing_channel(p: float) -> List[np.ndarray]:
    """Single-qubit depolarizing channel Kraus operators.

    With probability p, applies a random Pauli (X, Y, or Z).
    With probability 1-p, identity.

    Returns 4 Kraus operators: sqrt(1-p)*I, sqrt(p/3)*X, sqrt(p/3)*Y, sqrt(p/3)*Z.
    """
    p = max(0.0, min(1.0, p))
    return [
        np.sqrt(1 - p) * _I,
        np.sqrt(p / 3) * _X,
        np.sqrt(p / 3) * _Y,
        np.sqrt(p / 3) * _Z,
    ]


def amplitude_damping_channel(gamma: float) -> List[np.ndarray]:
    """Single-qubit amplitude damping channel.

    Models T1 relaxation: |1> decays to |0> with probability gamma.

    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, sqrt(gamma)], [0, 0]]
    """
    gamma = max(0.0, min(1.0, gamma))
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [k0, k1]


def readout_error_channel(p: float) -> np.ndarray:
    """Readout error confusion matrix.

    p = probability of misreading a qubit.
    Returns 2x2 matrix where M[i,j] = P(measure j | true state i).
    """
    p = max(0.0, min(1.0, p))
    return np.array([
        [1 - p, p],
        [p, 1 - p],
    ])


@dataclass
class NoiseModel:
    """Parameterized noise model for circuit simulation."""

    single_qubit_error: float = 0.0
    two_qubit_error: float = 0.0
    readout_error: float = 0.0

    def scale(self, factor: float) -> NoiseModel:
        """Return a new NoiseModel with scaled error rates (for ZNE)."""
        return NoiseModel(
            single_qubit_error=min(self.single_qubit_error * factor, 1.0),
            two_qubit_error=min(self.two_qubit_error * factor, 1.0),
            readout_error=self.readout_error,  # readout stays fixed
        )
