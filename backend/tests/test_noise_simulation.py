"""Tests for Noise Simulation & Error Mitigation — TDD: tests written first.

Covers noise models (depolarizing, amplitude damping, readout error),
noisy circuit simulation, and error mitigation techniques
(zero-noise extrapolation, measurement error mitigation).
"""

import numpy as np
import pytest

from quantum.noise.noise_model import (
    NoiseModel,
    depolarizing_channel,
    amplitude_damping_channel,
    readout_error_channel,
)
from quantum.noise.noisy_simulator import NoisySimulator
from quantum.noise.error_mitigation import (
    zero_noise_extrapolation,
    measurement_error_mitigation,
    apply_mitigation_pipeline,
)
from quantum.circuits.compiler.peephole import Gate, Circuit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_circuit():
    """H(0), CX(0,1) — Bell state preparation."""
    return Circuit(n_qubits=2, gates=[
        Gate("h", qubits=[0]),
        Gate("cx", qubits=[0, 1]),
    ])


@pytest.fixture
def single_qubit_circuit():
    """X gate on qubit 0."""
    return Circuit(n_qubits=1, gates=[
        Gate("x", qubits=[0]),
    ])


@pytest.fixture
def multi_gate_circuit():
    """Circuit with several gates for noise accumulation tests."""
    return Circuit(n_qubits=2, gates=[
        Gate("h", qubits=[0]),
        Gate("h", qubits=[1]),
        Gate("cx", qubits=[0, 1]),
        Gate("rz", qubits=[0], params=[0.5]),
        Gate("cx", qubits=[0, 1]),
    ])


@pytest.fixture
def low_noise_model():
    """Noise model with low error rates."""
    return NoiseModel(
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        readout_error=0.01,
    )


@pytest.fixture
def high_noise_model():
    """Noise model with high error rates."""
    return NoiseModel(
        single_qubit_error=0.05,
        two_qubit_error=0.10,
        readout_error=0.05,
    )


# ===========================================================================
# 1. Noise channels (Kraus operators)
# ===========================================================================

class TestDepolarizingChannel:

    def test_returns_kraus_operators(self):
        kraus = depolarizing_channel(0.01)
        assert isinstance(kraus, list)
        assert len(kraus) == 4  # I, X, Y, Z

    def test_kraus_completeness(self):
        """Sum of K_i^dag K_i = I (trace preserving)."""
        kraus = depolarizing_channel(0.05)
        total = sum(k.conj().T @ k for k in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10)

    def test_zero_noise_is_identity(self):
        kraus = depolarizing_channel(0.0)
        # Only the identity operator should have weight 1
        assert np.allclose(kraus[0], np.eye(2), atol=1e-10)

    def test_matrices_are_2x2(self):
        kraus = depolarizing_channel(0.1)
        for k in kraus:
            assert k.shape == (2, 2)


class TestAmplitudeDampingChannel:

    def test_returns_kraus_operators(self):
        kraus = amplitude_damping_channel(0.01)
        assert isinstance(kraus, list)
        assert len(kraus) == 2

    def test_kraus_completeness(self):
        kraus = amplitude_damping_channel(0.05)
        total = sum(k.conj().T @ k for k in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-10)

    def test_zero_damping_is_identity(self):
        kraus = amplitude_damping_channel(0.0)
        assert np.allclose(kraus[0], np.eye(2), atol=1e-10)


class TestReadoutErrorChannel:

    def test_returns_matrix(self):
        mat = readout_error_channel(0.01)
        assert mat.shape == (2, 2)

    def test_rows_sum_to_one(self):
        """Each row is a probability distribution."""
        mat = readout_error_channel(0.05)
        assert np.allclose(np.sum(mat, axis=1), 1.0)

    def test_zero_error_is_identity(self):
        mat = readout_error_channel(0.0)
        assert np.allclose(mat, np.eye(2))


# ===========================================================================
# 2. Noise model
# ===========================================================================

class TestNoiseModel:

    def test_creation(self):
        nm = NoiseModel(single_qubit_error=0.01, two_qubit_error=0.02)
        assert nm.single_qubit_error == 0.01
        assert nm.two_qubit_error == 0.02

    def test_default_readout(self):
        nm = NoiseModel(single_qubit_error=0.01)
        assert nm.readout_error == 0.0

    def test_scale_noise(self):
        """Scaling noise rates for ZNE."""
        nm = NoiseModel(single_qubit_error=0.01, two_qubit_error=0.02, readout_error=0.01)
        scaled = nm.scale(2.0)
        assert np.isclose(scaled.single_qubit_error, 0.02)
        assert np.isclose(scaled.two_qubit_error, 0.04)


# ===========================================================================
# 3. Noisy simulator
# ===========================================================================

class TestNoisySimulator:

    def test_ideal_x_gate(self, single_qubit_circuit):
        """Without noise, X|0> should give |1> with probability 1."""
        sim = NoisySimulator(noise_model=None)
        probs = sim.run(single_qubit_circuit, n_shots=1000)
        assert probs.get("1", 0) > 0.99

    def test_ideal_bell_state(self, simple_circuit):
        """Without noise, Bell state gives |00> and |11> each ~50%."""
        sim = NoisySimulator(noise_model=None)
        probs = sim.run(simple_circuit, n_shots=10000)
        assert abs(probs.get("00", 0) - 0.5) < 0.05
        assert abs(probs.get("11", 0) - 0.5) < 0.05

    def test_noise_degrades_fidelity(self, single_qubit_circuit, high_noise_model):
        """High noise should reduce |1> probability below ideal."""
        sim = NoisySimulator(noise_model=high_noise_model)
        probs = sim.run(single_qubit_circuit, n_shots=10000)
        # With noise, P(|1>) should be less than 1.0
        assert probs.get("1", 0) < 0.99

    def test_more_noise_more_degradation(self, simple_circuit, low_noise_model, high_noise_model):
        """Higher noise should give worse fidelity."""
        sim_low = NoisySimulator(noise_model=low_noise_model)
        sim_high = NoisySimulator(noise_model=high_noise_model)

        probs_low = sim_low.run(simple_circuit, n_shots=10000, seed=42)
        probs_high = sim_high.run(simple_circuit, n_shots=10000, seed=42)

        # Bell state: ideally |00> + |11> each 50%
        # Fidelity: P(00) + P(11)
        fid_low = probs_low.get("00", 0) + probs_low.get("11", 0)
        fid_high = probs_high.get("00", 0) + probs_high.get("11", 0)
        assert fid_low > fid_high

    def test_returns_probability_dict(self, simple_circuit):
        sim = NoisySimulator(noise_model=None)
        probs = sim.run(simple_circuit, n_shots=1000)
        assert isinstance(probs, dict)
        assert np.isclose(sum(probs.values()), 1.0, atol=0.01)

    def test_deterministic_with_seed(self, simple_circuit, low_noise_model):
        sim = NoisySimulator(noise_model=low_noise_model)
        p1 = sim.run(simple_circuit, n_shots=1000, seed=42)
        p2 = sim.run(simple_circuit, n_shots=1000, seed=42)
        assert p1 == p2


# ===========================================================================
# 4. Error mitigation: Zero-noise extrapolation
# ===========================================================================

class TestZeroNoiseExtrapolation:

    def test_returns_dict(self, simple_circuit, low_noise_model):
        result = zero_noise_extrapolation(
            circuit=simple_circuit,
            noise_model=low_noise_model,
            n_shots=5000,
            scale_factors=[1.0, 2.0, 3.0],
            seed=42,
        )
        assert isinstance(result, dict)
        assert "mitigated" in result
        assert "noisy_results" in result

    def test_mitigated_closer_to_ideal(self, single_qubit_circuit):
        """ZNE result should be closer to ideal than raw noisy result."""
        noise = NoiseModel(single_qubit_error=0.03, two_qubit_error=0.05, readout_error=0.02)

        # Ideal: P(1) = 1.0
        ideal_p1 = 1.0

        result = zero_noise_extrapolation(
            circuit=single_qubit_circuit,
            noise_model=noise,
            n_shots=10000,
            scale_factors=[1.0, 2.0, 3.0],
            seed=42,
        )

        noisy_p1 = result["noisy_results"][0].get("1", 0)
        mitigated_p1 = result["mitigated"].get("1", 0)

        noisy_error = abs(noisy_p1 - ideal_p1)
        mitigated_error = abs(mitigated_p1 - ideal_p1)

        # Mitigated should be at least as good as noisy (with tolerance)
        assert mitigated_error <= noisy_error + 0.05

    def test_returns_all_scale_factors(self, simple_circuit, low_noise_model):
        scales = [1.0, 1.5, 2.0]
        result = zero_noise_extrapolation(
            circuit=simple_circuit,
            noise_model=low_noise_model,
            n_shots=1000,
            scale_factors=scales,
            seed=42,
        )
        assert len(result["noisy_results"]) == len(scales)


# ===========================================================================
# 5. Error mitigation: Measurement error
# ===========================================================================

class TestMeasurementErrorMitigation:

    def test_returns_corrected_probs(self):
        """Corrected probabilities should be valid distribution."""
        raw_probs = {"0": 0.45, "1": 0.55}
        readout_mat = np.array([[0.95, 0.05], [0.03, 0.97]])
        corrected = measurement_error_mitigation(raw_probs, readout_mat, n_qubits=1)
        assert isinstance(corrected, dict)
        assert np.isclose(sum(corrected.values()), 1.0, atol=0.01)

    def test_identity_readout_unchanged(self):
        """Perfect readout should return same distribution."""
        raw_probs = {"0": 0.3, "1": 0.7}
        readout_mat = np.eye(2)
        corrected = measurement_error_mitigation(raw_probs, readout_mat, n_qubits=1)
        assert np.isclose(corrected.get("0", 0), 0.3, atol=0.01)
        assert np.isclose(corrected.get("1", 0), 0.7, atol=0.01)


# ===========================================================================
# 6. Full mitigation pipeline
# ===========================================================================

class TestMitigationPipeline:

    def test_pipeline_returns_all_stages(self, single_qubit_circuit):
        noise = NoiseModel(single_qubit_error=0.02, two_qubit_error=0.05, readout_error=0.02)
        result = apply_mitigation_pipeline(
            circuit=single_qubit_circuit,
            noise_model=noise,
            n_shots=5000,
            seed=42,
        )
        assert "ideal" in result
        assert "noisy" in result
        assert "mitigated" in result

    def test_ideal_is_noiseless(self, single_qubit_circuit):
        noise = NoiseModel(single_qubit_error=0.02, two_qubit_error=0.05, readout_error=0.02)
        result = apply_mitigation_pipeline(
            circuit=single_qubit_circuit,
            noise_model=noise,
            n_shots=5000,
            seed=42,
        )
        # X|0> = |1>
        assert result["ideal"].get("1", 0) > 0.99

    def test_noisy_differs_from_ideal(self, single_qubit_circuit):
        noise = NoiseModel(single_qubit_error=0.05, two_qubit_error=0.10, readout_error=0.05)
        result = apply_mitigation_pipeline(
            circuit=single_qubit_circuit,
            noise_model=noise,
            n_shots=10000,
            seed=42,
        )
        # Noisy should differ from ideal
        assert result["noisy"].get("1", 0) < 0.99
