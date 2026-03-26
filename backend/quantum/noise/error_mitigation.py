"""Error mitigation techniques for noisy quantum circuits.

Implements:
  - Zero-Noise Extrapolation (ZNE): run at multiple noise levels, extrapolate to zero
  - Measurement error mitigation: invert the readout confusion matrix
  - Combined pipeline: ideal → noisy → mitigated comparison
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from quantum.circuits.compiler.peephole import Circuit
from quantum.noise.noise_model import NoiseModel, readout_error_channel
from quantum.noise.noisy_simulator import NoisySimulator


def zero_noise_extrapolation(
    circuit: Circuit,
    noise_model: NoiseModel,
    n_shots: int = 10000,
    scale_factors: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Zero-Noise Extrapolation (ZNE).

    Run the circuit at multiple noise scale factors and extrapolate
    probabilities to the zero-noise limit using Richardson extrapolation.

    Args:
        circuit: The quantum circuit to simulate.
        noise_model: Base noise model.
        n_shots: Shots per noise level.
        scale_factors: Noise scaling factors (e.g., [1.0, 2.0, 3.0]).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'mitigated' (extrapolated probabilities) and 'noisy_results'.
    """
    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]

    # Collect results at each noise level
    noisy_results: List[Dict[str, float]] = []
    for sf in scale_factors:
        scaled_noise = noise_model.scale(sf)
        sim = NoisySimulator(noise_model=scaled_noise)
        s = seed if seed is not None else None
        probs = sim.run(circuit, n_shots=n_shots, seed=s)
        noisy_results.append(probs)

    # Collect all bitstrings
    all_bitstrings = set()
    for r in noisy_results:
        all_bitstrings.update(r.keys())

    # Richardson extrapolation to zero noise for each bitstring
    mitigated: Dict[str, float] = {}
    for bs in all_bitstrings:
        values = [r.get(bs, 0.0) for r in noisy_results]
        # Linear extrapolation: fit p(c) = a + b*c, extrapolate to c=0
        # Using numpy polyfit
        coeffs = np.polyfit(scale_factors, values, min(len(scale_factors) - 1, 2))
        extrapolated = float(np.polyval(coeffs, 0.0))
        mitigated[bs] = max(extrapolated, 0.0)

    # Normalize
    total = sum(mitigated.values())
    if total > 0:
        mitigated = {k: v / total for k, v in mitigated.items()}

    return {
        "mitigated": mitigated,
        "noisy_results": noisy_results,
        "scale_factors": scale_factors,
    }


def measurement_error_mitigation(
    raw_probs: Dict[str, float],
    readout_matrix: np.ndarray,
    n_qubits: int,
) -> Dict[str, float]:
    """Correct measurement probabilities by inverting the readout confusion matrix.

    For a single qubit, readout_matrix is 2x2.
    For multi-qubit, we build the full tensor-product confusion matrix.

    Args:
        raw_probs: Measured probability distribution.
        readout_matrix: Per-qubit 2x2 confusion matrix.
        n_qubits: Number of qubits.

    Returns:
        Corrected probability distribution.
    """
    dim = 1 << n_qubits

    # Build full confusion matrix as tensor product
    full_confusion = readout_matrix
    for _ in range(n_qubits - 1):
        full_confusion = np.kron(full_confusion, readout_matrix)

    # Convert raw_probs to vector
    raw_vec = np.zeros(dim)
    for bs, p in raw_probs.items():
        idx = int(bs, 2)
        raw_vec[idx] = p

    # Invert: p_true = M^-1 @ p_measured
    try:
        corrected_vec = np.linalg.solve(full_confusion, raw_vec)
    except np.linalg.LinAlgError:
        corrected_vec = raw_vec  # fallback

    # Clip negatives and normalize
    corrected_vec = np.maximum(corrected_vec, 0.0)
    total = np.sum(corrected_vec)
    if total > 0:
        corrected_vec /= total

    # Convert back to dict
    result: Dict[str, float] = {}
    for i in range(dim):
        if corrected_vec[i] > 1e-10:
            result[format(i, f"0{n_qubits}b")] = float(corrected_vec[i])

    return result


def apply_mitigation_pipeline(
    circuit: Circuit,
    noise_model: NoiseModel,
    n_shots: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full comparison: ideal vs noisy vs mitigated.

    Returns dict with 'ideal', 'noisy', and 'mitigated' probability distributions.
    """
    # Ideal (no noise)
    sim_ideal = NoisySimulator(noise_model=None)
    ideal = sim_ideal.run(circuit, n_shots=n_shots, seed=seed)

    # Noisy
    sim_noisy = NoisySimulator(noise_model=noise_model)
    noisy = sim_noisy.run(circuit, n_shots=n_shots, seed=seed)

    # Mitigated via ZNE
    zne_result = zero_noise_extrapolation(
        circuit=circuit,
        noise_model=noise_model,
        n_shots=n_shots,
        scale_factors=[1.0, 2.0, 3.0],
        seed=seed,
    )

    return {
        "ideal": ideal,
        "noisy": noisy,
        "mitigated": zne_result["mitigated"],
        "zne_detail": zne_result,
    }
