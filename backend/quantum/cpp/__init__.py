"""C++ accelerated QAOA kernels with Python fallback.

Usage:
    from quantum.cpp import (
        build_cost_diagonal,
        apply_cost_unitary,
        apply_mixer_unitary,
        evaluate_qubo,
        qaoa_expectation,
        HAS_CPP,
    )

If the C++ extension is available, these are the compiled versions.
Otherwise, they fall back to pure Python/numpy implementations.
"""

import sys
import os

# Try to import the compiled C++ extension
_cpp_dir = os.path.dirname(__file__)
if _cpp_dir not in sys.path:
    sys.path.insert(0, _cpp_dir)

try:
    from qaoa_cpp import (
        build_cost_diagonal,
        apply_cost_unitary,
        apply_mixer_unitary,
        evaluate_qubo,
        qaoa_expectation,
    )
    HAS_CPP = True
except ImportError:
    # Fall back to Python implementations
    from quantum.solvers.qaoa_solver import (
        build_cost_diagonal,
        apply_cost_unitary,
        apply_mixer_unitary,
        simulate_qaoa_expectation as qaoa_expectation,
    )
    from quantum.solvers.problem_encodings import evaluate_qubo
    HAS_CPP = False
