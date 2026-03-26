/**
 * @file bindings.cpp
 * @brief pybind11 bindings for QAOA C++ kernels.
 *
 * Exposes the hot-path QAOA functions to Python via numpy arrays.
 * Uses raw data pointers from py::array_t for compatibility with pybind11 3.x.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <complex>
#include <stdexcept>
#include <vector>

#include "qaoa_kernels.hpp"

namespace py = pybind11;

/**
 * Python wrapper: build_cost_diagonal(Q) -> np.ndarray
 */
py::array_t<double> py_build_cost_diagonal(py::array_t<double, py::array::c_style> Q_arr) {
    py::buffer_info Q_buf = Q_arr.request();
    if (Q_buf.ndim != 2 || Q_buf.shape[0] != Q_buf.shape[1]) {
        throw std::invalid_argument("Q must be a square 2-D matrix");
    }
    int n = static_cast<int>(Q_buf.shape[0]);
    if (n > 25) {
        throw std::invalid_argument("n > 25 would require too much memory");
    }

    uint64_t dim = 1ULL << n;
    auto result = py::array_t<double>(dim);
    py::buffer_info res_buf = result.request();

    qaoa::build_cost_diagonal(
        static_cast<const double*>(Q_buf.ptr),
        n,
        static_cast<double*>(res_buf.ptr)
    );

    return result;
}

/**
 * Python wrapper: apply_cost_unitary(sv, cost_diag, gamma) -> np.ndarray
 */
py::array_t<std::complex<double>> py_apply_cost_unitary(
    py::array_t<std::complex<double>, py::array::c_style> sv_arr,
    py::array_t<double, py::array::c_style> cost_diag_arr,
    double gamma
) {
    py::buffer_info sv_buf = sv_arr.request();
    py::buffer_info cd_buf = cost_diag_arr.request();
    uint64_t dim = static_cast<uint64_t>(sv_buf.shape[0]);

    if (static_cast<uint64_t>(cd_buf.shape[0]) != dim) {
        throw std::invalid_argument("sv and cost_diag must have the same length");
    }

    auto* sv_ptr = static_cast<const std::complex<double>*>(sv_buf.ptr);
    auto* cd_ptr = static_cast<const double*>(cd_buf.ptr);

    // Split into real/imag for the kernel
    std::vector<double> sv_real(dim), sv_imag(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        sv_real[i] = sv_ptr[i].real();
        sv_imag[i] = sv_ptr[i].imag();
    }

    qaoa::apply_cost_unitary(sv_real.data(), sv_imag.data(), cd_ptr, gamma, dim);

    // Pack back
    auto result = py::array_t<std::complex<double>>(dim);
    auto* out_ptr = static_cast<std::complex<double>*>(result.request().ptr);
    for (uint64_t i = 0; i < dim; ++i) {
        out_ptr[i] = std::complex<double>(sv_real[i], sv_imag[i]);
    }
    return result;
}

/**
 * Python wrapper: apply_mixer_unitary(sv, n_qubits, beta) -> np.ndarray
 */
py::array_t<std::complex<double>> py_apply_mixer_unitary(
    py::array_t<std::complex<double>, py::array::c_style> sv_arr,
    int n_qubits,
    double beta
) {
    py::buffer_info sv_buf = sv_arr.request();
    uint64_t dim = static_cast<uint64_t>(sv_buf.shape[0]);

    if (dim != (1ULL << n_qubits)) {
        throw std::invalid_argument("sv length must be 2^n_qubits");
    }

    auto* sv_ptr = static_cast<const std::complex<double>*>(sv_buf.ptr);

    std::vector<double> sv_real(dim), sv_imag(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        sv_real[i] = sv_ptr[i].real();
        sv_imag[i] = sv_ptr[i].imag();
    }

    qaoa::apply_mixer_unitary(sv_real.data(), sv_imag.data(), n_qubits, beta);

    auto result = py::array_t<std::complex<double>>(dim);
    auto* out_ptr = static_cast<std::complex<double>*>(result.request().ptr);
    for (uint64_t i = 0; i < dim; ++i) {
        out_ptr[i] = std::complex<double>(sv_real[i], sv_imag[i]);
    }
    return result;
}

/**
 * Python wrapper: evaluate_qubo(Q, x) -> float
 */
double py_evaluate_qubo(
    py::array_t<double, py::array::c_style> Q_arr,
    py::array_t<double, py::array::c_style> x_arr
) {
    py::buffer_info Q_buf = Q_arr.request();
    py::buffer_info x_buf = x_arr.request();
    int n = static_cast<int>(Q_buf.shape[0]);

    if (Q_buf.ndim != 2 || Q_buf.shape[1] != n || x_buf.shape[0] != n) {
        throw std::invalid_argument("Dimension mismatch: Q must be (n,n) and x must be (n,)");
    }

    return qaoa::evaluate_qubo(
        static_cast<const double*>(Q_buf.ptr),
        static_cast<const double*>(x_buf.ptr),
        n
    );
}

/**
 * Python wrapper: qaoa_expectation(Q, gamma, beta) -> float
 */
double py_qaoa_expectation(
    py::array_t<double, py::array::c_style> Q_arr,
    py::array_t<double, py::array::c_style> gamma_arr,
    py::array_t<double, py::array::c_style> beta_arr
) {
    py::buffer_info Q_buf = Q_arr.request();
    py::buffer_info g_buf = gamma_arr.request();
    py::buffer_info b_buf = beta_arr.request();

    int n = static_cast<int>(Q_buf.shape[0]);
    int p = static_cast<int>(g_buf.shape[0]);

    if (Q_buf.ndim != 2 || Q_buf.shape[1] != n) {
        throw std::invalid_argument("Q must be a square 2-D matrix");
    }
    if (b_buf.shape[0] != p) {
        throw std::invalid_argument("gamma and beta must have the same length");
    }
    if (n > 25) {
        throw std::invalid_argument("n > 25 would require too much memory");
    }

    return qaoa::qaoa_expectation(
        static_cast<const double*>(Q_buf.ptr),
        n,
        static_cast<const double*>(g_buf.ptr),
        static_cast<const double*>(b_buf.ptr),
        p
    );
}


PYBIND11_MODULE(qaoa_cpp, m) {
    m.doc() = "C++ accelerated QAOA kernels for statevector simulation";

    m.def("build_cost_diagonal", &py_build_cost_diagonal,
          py::arg("Q"),
          "Build cost Hamiltonian diagonal from QUBO matrix Q (n x n). "
          "Returns array of length 2^n.");

    m.def("apply_cost_unitary", &py_apply_cost_unitary,
          py::arg("sv"), py::arg("cost_diag"), py::arg("gamma"),
          "Apply cost unitary exp(-i*gamma*C) to statevector.");

    m.def("apply_mixer_unitary", &py_apply_mixer_unitary,
          py::arg("sv"), py::arg("n_qubits"), py::arg("beta"),
          "Apply mixer unitary exp(-i*beta*sum(X_i)) to statevector.");

    m.def("evaluate_qubo", &py_evaluate_qubo,
          py::arg("Q"), py::arg("x"),
          "Evaluate QUBO objective x^T Q x.");

    m.def("qaoa_expectation", &py_qaoa_expectation,
          py::arg("Q"), py::arg("gamma"), py::arg("beta"),
          "Compute full QAOA expectation value <psi|C|psi>.");
}
