"""Variational Quantum Regressor — hybrid classical-quantum model.

Architecture:
  1. Encode features via angle encoding into a parameterized quantum circuit
  2. Apply variational (trainable) rotation + entangling layers
  3. Measure expectation value of Z on first qubit as prediction
  4. Optimize parameters via classical optimizer (COBYLA) to minimize MSE

All quantum simulation is numpy-based statevector (no Qiskit dependency).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize


def _apply_ry(sv: np.ndarray, qubit: int, theta: float, n_qubits: int) -> np.ndarray:
    """Apply Ry(theta) to a single qubit in the statevector."""
    dim = 1 << n_qubits
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    mask = 1 << qubit
    new_sv = np.zeros_like(sv)
    for i in range(dim):
        if i & mask == 0:
            j = i | mask
            new_sv[i] += c * sv[i] - s * sv[j]
            new_sv[j] += s * sv[i] + c * sv[j]
    return new_sv


def _apply_rz(sv: np.ndarray, qubit: int, theta: float, n_qubits: int) -> np.ndarray:
    """Apply Rz(theta) to a single qubit."""
    dim = 1 << n_qubits
    mask = 1 << qubit
    new_sv = sv.copy()
    for i in range(dim):
        if i & mask == 0:
            new_sv[i] *= np.exp(-1j * theta / 2)
        else:
            new_sv[i] *= np.exp(1j * theta / 2)
    return new_sv


def _apply_cnot(sv: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT gate."""
    dim = 1 << n_qubits
    c_mask = 1 << control
    t_mask = 1 << target
    new_sv = sv.copy()
    for i in range(dim):
        if (i & c_mask) and not (i & t_mask):
            j = i | t_mask
            new_sv[i], new_sv[j] = sv[j], sv[i]
    return new_sv


def _expectation_z0(sv: np.ndarray, n_qubits: int) -> float:
    """Expectation value of Z on qubit 0."""
    result = 0.0
    for i, amp in enumerate(sv):
        prob = float(np.abs(amp) ** 2)
        if i & 1 == 0:
            result += prob  # |0> eigenvalue +1
        else:
            result -= prob  # |1> eigenvalue -1
    return result


class VariationalQuantumRegressor:
    """Hybrid classical-quantum regressor.

    Circuit structure per sample:
      1. Feature encoding: Ry(x_i * pi) on qubit i
      2. For each layer:
         a. Ry(theta) + Rz(phi) on each qubit (trainable)
         b. CNOT chain: qubit i -> qubit i+1
      3. Measure <Z> on qubit 0 as prediction

    Training minimizes MSE via classical optimizer.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        maxiter: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.maxiter = maxiter
        self.seed = seed
        self._parameters: Optional[np.ndarray] = None
        self._scale_y: float = 1.0
        self._offset_y: float = 0.0

    @property
    def parameters(self) -> Optional[np.ndarray]:
        return self._parameters

    @property
    def n_params(self) -> int:
        """2 parameters per qubit per layer (Ry + Rz)."""
        return 2 * self.n_qubits * self.n_layers

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> float:
        """Run the variational circuit on one sample, return <Z_0>."""
        n = self.n_qubits
        dim = 1 << n

        # Initialize |0...0>
        sv = np.zeros(dim, dtype=complex)
        sv[0] = 1.0

        # Feature encoding
        n_features = len(x)
        for q in range(n):
            theta = x[q % n_features] * np.pi
            sv = _apply_ry(sv, q, theta, n)

        # Variational layers
        idx = 0
        for _ in range(self.n_layers):
            for q in range(n):
                sv = _apply_ry(sv, q, params[idx], n)
                idx += 1
                sv = _apply_rz(sv, q, params[idx], n)
                idx += 1
            # Entangling: CNOT chain
            for q in range(n - 1):
                sv = _apply_cnot(sv, q, q + 1, n)

        return _expectation_z0(sv, n)

    def _predict_raw(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Raw predictions (expectation values in [-1, 1])."""
        return np.array([self._circuit(x, params) for x in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the variational circuit parameters."""
        # Scale targets to [-1, 1] range for Z expectation
        self._offset_y = float(np.mean(y))
        self._scale_y = float(np.std(y)) if np.std(y) > 1e-15 else 1.0

        y_scaled = (y - self._offset_y) / self._scale_y
        # Clip to [-1, 1] since <Z> is bounded
        y_scaled = np.clip(y_scaled, -0.95, 0.95)

        rng = np.random.default_rng(self.seed)
        init_params = rng.uniform(-np.pi, np.pi, self.n_params)

        def cost(params: np.ndarray) -> float:
            preds = self._predict_raw(X, params)
            return float(np.mean((preds - y_scaled) ** 2))

        result = scipy_minimize(
            cost,
            init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.5},
        )

        self._parameters = result.x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained parameters."""
        if self._parameters is None:
            raise RuntimeError("Must call fit() before predict()")

        raw = self._predict_raw(X, self._parameters)
        return raw * self._scale_y + self._offset_y
