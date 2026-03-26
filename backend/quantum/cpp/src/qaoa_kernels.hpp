/**
 * @file qaoa_kernels.hpp
 * @brief High-performance QAOA statevector simulation kernels.
 *
 * These functions are the hot path in QAOA optimization:
 *   - build_cost_diagonal: O(2^n * n) — builds cost Hamiltonian diagonal
 *   - apply_mixer_unitary: O(2^n * n) — applies exp(-i*beta*sum(X_i))
 *   - evaluate_qubo: O(n^2) — evaluates x^T Q x for one bitstring
 *
 * All functions operate on raw C arrays for maximum performance.
 * Memory is managed by the caller (Python via pybind11).
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace qaoa {

/**
 * Build the cost diagonal: for each basis state |i>, compute x^T Q x
 * where x is the binary representation of i.
 *
 * @param Q       Flattened row-major QUBO matrix (n x n)
 * @param n       Number of qubits (variables)
 * @param out     Output array of size 2^n (caller-allocated)
 */
inline void build_cost_diagonal(
    const double* Q,
    int n,
    double* out
) {
    const uint64_t num_states = 1ULL << n;

    for (uint64_t i = 0; i < num_states; ++i) {
        double val = 0.0;
        for (int r = 0; r < n; ++r) {
            if (!((i >> r) & 1)) continue;
            for (int c = 0; c < n; ++c) {
                if ((i >> c) & 1) {
                    val += Q[r * n + c];
                }
            }
        }
        out[i] = val;
    }
}

/**
 * Apply cost unitary: sv[i] *= exp(-i * gamma * cost_diag[i])
 * In-place operation on the statevector.
 *
 * @param sv_real  Real parts of statevector (length 2^n)
 * @param sv_imag  Imaginary parts of statevector (length 2^n)
 * @param cost_diag Cost diagonal (length 2^n)
 * @param gamma    Cost angle
 * @param dim      Hilbert space dimension (2^n)
 */
inline void apply_cost_unitary(
    double* sv_real,
    double* sv_imag,
    const double* cost_diag,
    double gamma,
    uint64_t dim
) {
    for (uint64_t i = 0; i < dim; ++i) {
        const double angle = -gamma * cost_diag[i];
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double re = sv_real[i];
        const double im = sv_imag[i];
        sv_real[i] = c * re - s * im;
        sv_imag[i] = s * re + c * im;
    }
}

/**
 * Apply mixer unitary: exp(-i * beta * sum(X_q)) to statevector.
 * Since X_q commute across qubits, applies Rx(2*beta) to each qubit
 * sequentially, operating on pairs of amplitudes differing in bit q.
 *
 * @param sv_real  Real parts (length 2^n), modified in-place
 * @param sv_imag  Imaginary parts (length 2^n), modified in-place
 * @param n_qubits Number of qubits
 * @param beta     Mixer angle
 */
inline void apply_mixer_unitary(
    double* sv_real,
    double* sv_imag,
    int n_qubits,
    double beta
) {
    const double c = std::cos(beta);
    const double s = std::sin(beta);
    const uint64_t dim = 1ULL << n_qubits;

    for (int q = 0; q < n_qubits; ++q) {
        const uint64_t mask = 1ULL << q;
        for (uint64_t i = 0; i < dim; ++i) {
            if (i & mask) continue;  // only process |...0_q...> states
            const uint64_t j = i | mask;

            // sv[i] = cos(beta)*sv[i] - i*sin(beta)*sv[j]
            // sv[j] = -i*sin(beta)*sv[i] + cos(beta)*sv[j]
            const double ri = sv_real[i], ii = sv_imag[i];
            const double rj = sv_real[j], ij = sv_imag[j];

            // new_i = c*(ri+i*ii) + (-i*s)*(rj+i*ij)
            //       = c*ri + s*ij + i*(c*ii - s*rj)
            sv_real[i] = c * ri + s * ij;
            sv_imag[i] = c * ii - s * rj;

            // new_j = (-i*s)*(ri+i*ii) + c*(rj+i*ij)
            //       = s*ii + c*rj + i*(-s*ri + c*ij)
            sv_real[j] = s * ii + c * rj;
            sv_imag[j] = -s * ri + c * ij;
        }
    }
}

/**
 * Evaluate QUBO objective x^T Q x for a single binary vector.
 *
 * @param Q  Flattened row-major QUBO matrix (n x n)
 * @param x  Binary vector of length n (0.0 or 1.0)
 * @param n  Number of variables
 * @return   Objective value
 */
inline double evaluate_qubo(
    const double* Q,
    const double* x,
    int n
) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.5) continue;
        for (int j = 0; j < n; ++j) {
            if (x[j] < 0.5) continue;
            val += Q[i * n + j];
        }
    }
    return val;
}

/**
 * Compute full QAOA expectation value: <psi(gamma,beta)|C|psi(gamma,beta)>.
 * Combines build_cost_diagonal + layer application + expectation.
 *
 * @param Q       Flattened QUBO matrix (n x n)
 * @param n       Number of qubits
 * @param gamma   Cost angles (length p)
 * @param beta    Mixer angles (length p)
 * @param p       Number of QAOA layers
 * @return        Expectation value of cost Hamiltonian
 */
inline double qaoa_expectation(
    const double* Q,
    int n,
    const double* gamma,
    const double* beta,
    int p
) {
    const uint64_t dim = 1ULL << n;

    // Build cost diagonal
    std::vector<double> cost_diag(dim);
    build_cost_diagonal(Q, n, cost_diag.data());

    // Initialize |+>^n
    const double init_amp = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<double> sv_real(dim, init_amp);
    std::vector<double> sv_imag(dim, 0.0);

    // Apply QAOA layers
    for (int layer = 0; layer < p; ++layer) {
        apply_cost_unitary(sv_real.data(), sv_imag.data(),
                           cost_diag.data(), gamma[layer], dim);
        apply_mixer_unitary(sv_real.data(), sv_imag.data(),
                            n, beta[layer]);
    }

    // Compute expectation: sum |alpha_i|^2 * cost_diag[i]
    double expectation = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        const double prob = sv_real[i] * sv_real[i] + sv_imag[i] * sv_imag[i];
        expectation += prob * cost_diag[i];
    }

    return expectation;
}

}  // namespace qaoa
