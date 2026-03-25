# Quantum vs Classical Optimization for Portfolio Allocation Under Constraints

## Comprehensive Implementation Specification & Guide

**Version**: 1.0
**Author**: Quantitative Engineering Team
**Date**: 2026-03-25
**Status**: DRAFT — Awaiting confirmation before implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Feature 1: Quantum-Inspired Portfolio Optimization](#3-feature-1-quantum-inspired-portfolio-optimization)
4. [Feature 2: Quantum Monte Carlo Simulator](#4-feature-2-quantum-monte-carlo-simulator)
5. [Feature 3: Quantum Circuit Compiler/Optimizer](#5-feature-3-quantum-circuit-compileroptimizer)
6. [Feature 4: Quantum Machine Learning for Time Series](#6-feature-4-quantum-machine-learning-for-time-series)
7. [Feature 5: QAOA vs Classical Heuristics for NP-Hard Problems](#7-feature-5-qaoa-vs-classical-heuristics-for-np-hard-problems)
8. [Feature 6: Noise Simulation & Error Mitigation](#8-feature-6-noise-simulation--error-mitigation)
9. [Feature 7: Quantum Backtesting Engine](#9-feature-7-quantum-backtesting-engine)
10. [C++ Performance Layer](#10-c-performance-layer)
11. [Dependency Map & New Libraries](#11-dependency-map--new-libraries)
12. [Project Structure](#12-project-structure)
13. [Phased Implementation Plan](#13-phased-implementation-plan)
14. [Benchmarking & Evaluation Framework](#14-benchmarking--evaluation-framework)
15. [Hard Constraints & Integration Rules](#15-hard-constraints--integration-rules)
16. [Risk Assessment](#16-risk-assessment)
17. [API Endpoints](#17-api-endpoints)
18. [Frontend Visualization Requirements](#18-frontend-visualization-requirements)

---

## 1. Executive Summary

This spec extends FinanceBro's quantitative pipeline with quantum computing capabilities across seven integrated features. The core thesis: **rigorously benchmark quantum-inspired methods against classical baselines on real financial data, showing exactly where quantum approaches provide advantage — and where they don't.**

### What Makes This Stand Out

- Not a toy demo — integrated into a production-grade quant pipeline with walk-forward validation
- Every quantum method benchmarked against classical baselines with statistical significance tests
- C++ performance layer for latency-critical components (matrix ops, Monte Carlo sampling)
- Honest reporting: we show when quantum methods lose, not just when they win
- Noise-aware: all quantum results include ideal, noisy, and error-mitigated variants

### Integration Philosophy

All quantum features plug into the existing agent pipeline via the `BaseAgent` contract. Quantum agents are **alternatives to**, not replacements for, classical agents. The orchestrator selects the method; the benchmarking framework compares them.

```
Existing Pipeline:
DataAgent → FeatureAgent → ModelAgent → WalkForwardAgent →
BacktestAgent → OverfittingAgent → RiskAgent → PortfolioAgent → StatsAgent

Extended Pipeline:
DataAgent → FeatureAgent → [ModelAgent | QuantumMLAgent] → WalkForwardAgent →
[BacktestAgent | QuantumBacktestAgent] → OverfittingAgent → RiskAgent →
[PortfolioAgent | QuantumPortfolioAgent] → StatsAgent
                                            ↑
                              QuantumMonteCarloAgent (option pricing, side-pipeline)
                              NoiseSimulationAgent (wraps any quantum agent)
```

---

## 2. Architecture Overview

### 2.1 Directory Structure

```
backend/
├── agents/
│   ├── quantum/                          # All quantum agent implementations
│   │   ├── __init__.py
│   │   ├── quantum_portfolio_agent.py    # Feature 1: QAOA portfolio optimization
│   │   ├── quantum_montecarlo_agent.py   # Feature 2: Quantum amplitude estimation
│   │   ├── quantum_ml_agent.py           # Feature 4: Hybrid QML for time series
│   │   ├── quantum_backtest_agent.py     # Feature 7: Quantum backtesting engine
│   │   └── noise_simulation_agent.py     # Feature 6: Noise wrapper agent
│   └── (existing agents unchanged)
├── quantum/                              # Quantum infrastructure layer
│   ├── __init__.py
│   ├── circuits/                         # Feature 3 & 5: Circuit building + optimization
│   │   ├── __init__.py
│   │   ├── portfolio_circuit.py          # QAOA circuit for portfolio optimization
│   │   ├── amplitude_estimation.py       # QAE circuits for Monte Carlo
│   │   ├── variational.py               # Variational ansatz (VQE/QAOA layers)
│   │   ├── qml_circuits.py              # Parameterized circuits for QML
│   │   └── compiler/                     # Feature 3: Circuit compiler
│   │       ├── __init__.py
│   │       ├── optimizer.py              # Gate reduction, depth minimization
│   │       ├── transpiler.py             # Hardware-aware transpilation
│   │       ├── peephole.py               # Peephole optimization passes
│   │       └── hardware_constraints.py   # IBM backend topology/gate sets
│   ├── noise/                            # Feature 6: Noise simulation
│   │   ├── __init__.py
│   │   ├── noise_models.py              # Decoherence, gate errors, readout errors
│   │   ├── error_mitigation.py          # ZNE, PEC, M3 readout mitigation
│   │   └── noise_benchmarks.py          # Ideal vs noisy vs mitigated comparison
│   ├── solvers/                          # Feature 5: Classical + quantum solvers
│   │   ├── __init__.py
│   │   ├── qaoa_solver.py               # QAOA for combinatorial optimization
│   │   ├── vqe_solver.py                # VQE for continuous optimization
│   │   ├── classical_solvers.py         # SA, greedy, branch-and-bound baselines
│   │   └── problem_encodings.py         # QUBO/Ising formulations
│   └── benchmarks/                       # Cross-cutting benchmark framework
│       ├── __init__.py
│       ├── benchmark_runner.py           # Orchestrates quantum vs classical comparison
│       ├── metrics.py                    # Runtime, solution quality, approximation ratio
│       └── scaling_analysis.py           # Problem-size scaling experiments
├── cpp/                                  # C++ performance layer
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── matrix_ops.hpp               # Fast covariance, Cholesky, eigendecomp
│   │   ├── monte_carlo.hpp              # High-perf MC sampling engine
│   │   ├── portfolio_optimizer.hpp      # Convex optimization (quadratic programming)
│   │   └── circuit_simulator.hpp        # Statevector simulation backend
│   ├── src/
│   │   ├── matrix_ops.cpp
│   │   ├── monte_carlo.cpp
│   │   ├── portfolio_optimizer.cpp
│   │   └── circuit_simulator.cpp
│   ├── bindings/
│   │   └── pybind_module.cpp            # pybind11 Python bindings
│   └── tests/
│       ├── test_matrix_ops.cpp
│       ├── test_monte_carlo.cpp
│       └── test_portfolio_optimizer.cpp
├── specs/
│   └── quantum_implementation_spec.md   # This document
└── tests/
    ├── test_quantum_portfolio_agent.py
    ├── test_quantum_montecarlo_agent.py
    ├── test_quantum_ml_agent.py
    ├── test_quantum_backtest_agent.py
    ├── test_noise_simulation_agent.py
    ├── test_circuit_compiler.py
    ├── test_qaoa_solver.py
    └── test_cpp_bindings.py
```

### 2.2 Agent Hierarchy

All quantum agents extend `BaseAgent` and follow the existing contract:

```python
class QuantumPortfolioAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool: ...
    def log_metrics(self) -> None: ...

    @property
    def input_schema(self) -> Dict[str, Any]: ...

    @property
    def output_schema(self) -> Dict[str, Any]: ...
```

### 2.3 Classical-First Principle

Every quantum method MUST have a classical baseline computed first. The quantum result is always presented **relative to** the classical result. If the quantum method is slower or worse, report that honestly.

---

## 3. Feature 1: Quantum-Inspired Portfolio Optimization

### 3.1 Overview

Solve the constrained portfolio allocation problem using both classical and quantum approaches, then rigorously compare runtime, solution quality, and scaling behavior.

### 3.2 Problem Formulation

**Classical Markowitz (Quadratic Program):**
```
minimize    w^T Σ w                    (portfolio variance)
subject to  w^T μ >= r_target          (minimum return)
            Σ w_i = 1                  (fully invested)
            0 <= w_i <= w_max          (position limits)
            w_i = 0 for excluded       (sector constraints)
```

**QUBO Reformulation for QAOA:**

Discretize weights into binary variables. For N assets with K bits of precision per weight:

```
w_i = Σ_{k=0}^{K-1} 2^k * x_{i,k} / (2^K - 1)

Minimize: x^T Q x
where Q encodes:
  - Portfolio variance (quadratic term)
  - Return constraint (penalty term, λ_return)
  - Budget constraint (penalty term, λ_budget)
  - Position limit constraint (penalty term, λ_position)
```

**QAOA Circuit:**
```
|ψ(γ, β)> = Π_{p=1}^{P} [U_M(β_p) U_C(γ_p)] |+>^n

U_C(γ) = exp(-iγ C)    — cost Hamiltonian (encodes QUBO)
U_M(β) = exp(-iβ B)    — mixer Hamiltonian (Σ X_i)
```

Optimize (γ, β) via classical optimizer (COBYLA or L-BFGS-B).

### 3.3 Implementation Details

**File: `agents/quantum/quantum_portfolio_agent.py`**

```python
class QuantumPortfolioAgent(BaseAgent):
    """Portfolio optimization via QAOA + classical baselines.

    Methods:
      - markowitz_cvxpy: Classical convex optimization (baseline)
      - qaoa: Quantum Approximate Optimization Algorithm
      - vqe: Variational Quantum Eigensolver
      - hybrid: QAOA warm-started from classical solution

    All methods respect:
      - Long-only constraints
      - Position limits (max_weight)
      - Ledoit-Wolf covariance (no raw sample)
      - No look-ahead (weights at t use data up to t)
    """

    DEFAULT_CONFIG = {
        "methods": ["markowitz_cvxpy", "qaoa"],  # run both for comparison
        "qaoa_layers": 3,                         # QAOA circuit depth (p)
        "weight_precision_bits": 3,               # bits per weight (2^3 = 8 levels)
        "max_weight": 0.10,
        "covariance_window": 252,
        "optimizer": "COBYLA",
        "optimizer_maxiter": 500,
        "n_shots": 4096,
        "backend": "aer_simulator",               # or "ibm_brisbane" for real HW
        "transaction_cost_bps": 5,
        "initial_capital": 100_000.0,
        "rebalance_frequency": 21,
    }
```

**Input Schema:**
```python
{
    "returns": "pd.DataFrame — daily returns (columns=tickers, index=dates)",
    "expected_returns": "(optional) pd.Series — return forecasts per asset",
    "config": "(optional) dict overriding DEFAULT_CONFIG",
}
```

**Output Schema:**
```python
{
    "classical_weights": "pd.DataFrame — Markowitz optimal weights per rebalance date",
    "quantum_weights": "pd.DataFrame — QAOA optimal weights per rebalance date",
    "efficient_frontier": {
        "classical": "dict — returns, risks, sharpes along frontier",
        "quantum": "dict — same structure, QAOA frontier points",
    },
    "comparison_metrics": {
        "runtime_classical_ms": "float",
        "runtime_quantum_ms": "float",
        "objective_classical": "float — portfolio variance achieved",
        "objective_quantum": "float",
        "weight_distance": "float — L2 norm between classical/quantum weights",
        "approximation_ratio": "float — quantum_obj / classical_obj",
    },
    "portfolio_returns_classical": "pd.Series — daily returns using classical weights",
    "portfolio_returns_quantum": "pd.Series — daily returns using quantum weights",
    "equity_curves": {
        "classical": "pd.Series",
        "quantum": "pd.Series",
    },
}
```

### 3.4 Classical Baseline: CVXPY Solver

```python
import cvxpy as cp

def solve_markowitz(mu, Sigma, r_target, w_max):
    """Solve Markowitz mean-variance optimization.

    Args:
        mu: Expected returns vector (N,)
        Sigma: Covariance matrix (N, N) — Ledoit-Wolf shrunk
        r_target: Minimum target return
        w_max: Maximum weight per asset

    Returns:
        Optimal weight vector (N,)
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [
        w >= 0,                    # long-only
        w <= w_max,                # position limits
        cp.sum(w) == 1,            # fully invested
        mu @ w >= r_target,        # return target
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    return w.value
```

### 3.5 QAOA Implementation

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def build_portfolio_qubo(mu, Sigma, r_target, w_max, n_bits):
    """Convert portfolio optimization to QUBO.

    Discretize weights: w_i = sum_{k} 2^k * x_{ik} / (2^n_bits - 1)
    Encode objective + constraints as quadratic penalty terms.
    """
    qp = QuadraticProgram()
    n_assets = len(mu)

    # Binary variables: x_{i,k} for asset i, bit k
    for i in range(n_assets):
        for k in range(n_bits):
            qp.binary_var(f"x_{i}_{k}")

    # Build objective: minimize w^T Sigma w
    # + penalty * (sum(w) - 1)^2
    # + penalty * max(0, r_target - mu^T w)^2
    # ... (full QUBO construction)

    return QuadraticProgramToQubo().convert(qp)


def solve_qaoa(qubo, p_layers, optimizer, n_shots, backend):
    """Run QAOA on the QUBO formulation.

    Args:
        qubo: QuadraticProgram in QUBO form
        p_layers: Number of QAOA layers
        optimizer: Classical optimizer name
        n_shots: Measurement shots
        backend: Qiskit backend

    Returns:
        (optimal_weights, objective_value, runtime_ms, circuit_depth)
    """
    qaoa = QAOA(
        reps=p_layers,
        optimizer=optimizer,
        initial_point=None,  # random init
    )
    result = qaoa.compute_minimum_eigenvalue(qubo.to_ising()[0])
    # Decode binary solution back to continuous weights
    weights = decode_binary_weights(result.best_measurement, n_assets, n_bits)
    return weights
```

### 3.6 Efficient Frontier Visualization

Generate 20–50 frontier points by varying `r_target` from min(mu) to max(mu):

```
For each r_target:
    1. Solve classical Markowitz → (risk_c, return_c)
    2. Solve QAOA → (risk_q, return_q)
    3. Record runtime, approximation ratio

Plot:
    - X-axis: Portfolio risk (std dev)
    - Y-axis: Portfolio return
    - Blue line: Classical efficient frontier
    - Red dots: QAOA frontier points
    - Highlight: Points where QAOA finds better solution
    - Inset: Runtime comparison bar chart
```

### 3.7 Scaling Analysis

Test with N = {3, 5, 7, 10, 15, 20} assets:

| Metric | How to Measure |
|--------|---------------|
| Classical runtime | Wall-clock time for CVXPY solve |
| QAOA runtime | Circuit construction + simulation + optimization loops |
| Qubit count | N * n_bits |
| Circuit depth | After transpilation to IBM basis gates |
| Approximation ratio | quantum_variance / classical_variance |
| Weight accuracy | L2 distance to classical optimal |

---

## 4. Feature 2: Quantum Monte Carlo Simulator

### 4.1 Overview

Price European options using Classical Monte Carlo, then compare against Quantum Amplitude Estimation (QAE) which theoretically provides quadratic speedup (O(1/ε) vs O(1/ε²) samples for precision ε).

### 4.2 Option Pricing Problem

**Black-Scholes European Call:**
```
C = E[max(S_T - K, 0) * exp(-rT)]

where S_T = S_0 * exp((r - σ²/2)T + σ√T * Z),  Z ~ N(0,1)
```

**Classical Monte Carlo:**
```python
def classical_mc_option_price(S0, K, r, sigma, T, n_paths):
    """Standard Monte Carlo option pricing.

    Returns: (price, std_error, confidence_interval)
    """
    Z = np.random.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0) * np.exp(-r * T)
    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_paths)
    return price, std_error, (price - 1.96*std_error, price + 1.96*std_error)
```

**Quantum Amplitude Estimation:**

Encode the payoff distribution into a quantum state, then use amplitude estimation to extract the expected value with quadratic speedup.

```
1. State preparation: |ψ> encodes P(S_T) via log-normal loading
2. Payoff operator: A|ψ> = sqrt(payoff)|good> + sqrt(1-payoff)|bad>
3. Amplitude estimation: estimate a = <ψ|A†A|ψ> = E[payoff]
```

### 4.3 Implementation

**File: `agents/quantum/quantum_montecarlo_agent.py`**

```python
class QuantumMonteCarloAgent(BaseAgent):
    """Quantum amplitude estimation for option pricing.

    Compares:
      - Classical MC (baseline, with antithetic + control variates)
      - Iterative QAE (Suzuki et al. 2020)
      - Maximum Likelihood AE (MLAE)
      - Classical with C++ accelerated sampling

    Reports: price, std error, runtime, convergence rate.
    """

    DEFAULT_CONFIG = {
        "methods": ["classical_mc", "classical_mc_cpp", "iqae"],
        "option_type": "european_call",
        "n_classical_paths": [1000, 10000, 100000, 1000000],
        "n_qubits_price": 5,        # qubits for price discretization
        "n_qubits_estimation": 8,    # qubits for amplitude estimation
        "confidence_level": 0.05,    # 95% CI
        "variance_reduction": ["antithetic", "control_variate"],
    }
```

**Input Schema:**
```python
{
    "spot_price": "float — current underlying price S_0",
    "strike_price": "float — option strike K",
    "risk_free_rate": "float — annualized risk-free rate r",
    "volatility": "float — annualized volatility σ",
    "time_to_expiry": "float — time to expiry in years T",
    "option_type": "(optional) 'european_call' | 'european_put' | 'asian'",
    "config": "(optional) dict overriding DEFAULT_CONFIG",
}
```

**Output Schema:**
```python
{
    "black_scholes_price": "float — analytical BS price (ground truth)",
    "classical_mc": {
        "price": "float",
        "std_error": "float",
        "ci_95": "[float, float]",
        "runtime_ms": "float",
        "n_paths": "int",
    },
    "classical_mc_cpp": {
        # same structure, C++ accelerated
    },
    "quantum_ae": {
        "price": "float",
        "std_error": "float",
        "ci_95": "[float, float]",
        "runtime_ms": "float",
        "n_oracle_calls": "int",
        "circuit_depth": "int",
    },
    "convergence_analysis": {
        "classical_errors_by_n": "list[dict] — error vs sample count",
        "quantum_errors_by_n": "list[dict] — error vs oracle calls",
        "scaling_exponent_classical": "float — should be ~-0.5",
        "scaling_exponent_quantum": "float — should be ~-1.0 (quadratic speedup)",
    },
    "variance_reduction_impact": {
        "naive_std_error": "float",
        "antithetic_std_error": "float",
        "control_variate_std_error": "float",
        "reduction_factor": "float",
    },
}
```

### 4.4 Quantum Circuit for Log-Normal Loading

```python
from qiskit.circuit.library import LogNormalDistribution, LinearAmplitudeFunction

def build_qae_circuit(S0, K, r, sigma, T, n_qubits_price, n_qubits_est):
    """Build quantum amplitude estimation circuit for European call.

    Steps:
        1. Prepare log-normal distribution on n_qubits_price qubits
        2. Apply payoff function as controlled rotation
        3. Wrap in amplitude estimation (Grover iterations)
    """
    # Parameters for log-normal
    mu_ln = (r - 0.5 * sigma**2) * T + np.log(S0)
    sigma_ln = sigma * np.sqrt(T)

    # Price bounds (truncate distribution at ~4σ)
    low = np.exp(mu_ln - 4 * sigma_ln)
    high = np.exp(mu_ln + 4 * sigma_ln)

    # Uncertainty model: load P(S_T) into quantum state
    uncertainty_model = LogNormalDistribution(
        n_qubits_price,
        mu=mu_ln,
        sigma=sigma_ln**2,
        bounds=(low, high),
    )

    # Payoff function: max(S - K, 0) mapped to rotation angle
    breakpoints = [low, K]
    slopes = [0, 1]      # 0 below strike, 1 above
    offsets = [0, -K]
    f_min, f_max = 0, high - K

    payoff = LinearAmplitudeFunction(
        n_qubits_price,
        slopes, offsets, domain=(low, high),
        image=(f_min, f_max),
        breakpoints=breakpoints,
    )

    # Compose: uncertainty → payoff
    qc = payoff.compose(uncertainty_model, front=True)
    return qc, (f_min, f_max)
```

### 4.5 Convergence Benchmark

Run classical MC with N = {10², 10³, 10⁴, 10⁵, 10⁶} paths and QAE with M = {2¹, 2², ..., 2⁸} Grover iterations. Plot:

```
X-axis: log(computational cost)  — paths for MC, oracle calls for QAE
Y-axis: log(estimation error)    — |price - BS_analytical|

Expected:
  Classical MC: slope ≈ -0.5  (standard MC convergence)
  Quantum AE:   slope ≈ -1.0  (quadratic speedup)
```

---

## 5. Feature 3: Quantum Circuit Compiler/Optimizer

### 5.1 Overview

Build a custom circuit optimization pipeline that takes arbitrary quantum circuits and minimizes gate count and depth, targeting IBM hardware constraints. This is a systems/compiler problem that demonstrates deep understanding of quantum computing infrastructure.

### 5.2 IBM Hardware Constraints

**Target Backend: IBM Brisbane (127 qubits, Eagle r3)**

```python
HARDWARE_CONSTRAINTS = {
    "basis_gates": ["cx", "id", "rz", "sx", "x"],
    "coupling_map": [...],  # Heavy-hex topology
    "gate_errors": {
        "cx": 0.01,    # ~1% two-qubit gate error
        "sx": 0.001,   # ~0.1% single-qubit gate error
    },
    "t1_times_us": 200,     # T1 relaxation
    "t2_times_us": 150,     # T2 dephasing
    "readout_error": 0.02,  # ~2% measurement error
    "max_circuit_depth": 100,  # practical depth limit before decoherence
}
```

### 5.3 Optimization Passes

**File: `quantum/circuits/compiler/optimizer.py`**

```python
class CircuitOptimizer:
    """Multi-pass quantum circuit optimizer.

    Optimization passes (applied in order):
        1. Peephole optimization — local gate cancellation/fusion
        2. Commutation analysis — reorder commuting gates to enable cancellation
        3. Template matching — replace known subcircuits with shorter equivalents
        4. Routing — map logical to physical qubits (minimize SWAP gates)
        5. Gate synthesis — decompose arbitrary unitaries to basis gates
        6. Depth reduction — parallelize independent gates
    """

    def optimize(self, circuit, hardware=None, passes=None):
        """Optimize circuit with configurable pass pipeline.

        Args:
            circuit: QuantumCircuit to optimize
            hardware: HardwareConstraints (optional, for routing)
            passes: list of pass names (optional, default=all)

        Returns:
            OptimizationResult with:
              - optimized_circuit
              - metrics_before: {gate_count, depth, cx_count, ...}
              - metrics_after: {gate_count, depth, cx_count, ...}
              - improvement_pct: {gate_count, depth, cx_count, ...}
              - pass_log: list of applied passes with per-pass metrics
        """
```

### 5.4 Peephole Optimization Rules

```python
PEEPHOLE_RULES = [
    # Identity cancellation
    ("cx", "cx") → Identity,               # CNOT self-inverse
    ("h", "h") → Identity,                  # Hadamard self-inverse
    ("x", "x") → Identity,                  # Pauli-X self-inverse
    ("z", "z") → Identity,                  # Pauli-Z self-inverse

    # Rotation fusion
    ("rz(θ₁)", "rz(θ₂)") → "rz(θ₁+θ₂)",  # Combine Z-rotations
    ("rx(θ₁)", "rx(θ₂)") → "rx(θ₁+θ₂)",

    # Gate decomposition shortcuts
    ("h", "z", "h") → "x",                  # HZH = X
    ("h", "x", "h") → "z",                  # HXH = Z

    # CNOT reduction
    ("cx(a,b)", "cx(b,a)", "cx(a,b)") → "swap(a,b)",

    # Remove rotations by 0 or 2π
    "rz(0)" → Identity,
    "rz(2π)" → Identity,
]
```

### 5.5 Benchmarking Circuit Optimization

For each quantum feature (portfolio QAOA, QAE, QML), measure:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total gate count | — | — | X% |
| CX gate count | — | — | X% (most important) |
| Circuit depth | — | — | X% |
| Estimated fidelity | — | — | X% |
| Transpilation time | — | — | — |

Generate visual circuit diagrams (before/after) using `qiskit.visualization.circuit_drawer`.

---

## 6. Feature 4: Quantum Machine Learning for Time Series

### 6.1 Overview

Build a hybrid classical-quantum model for financial time series prediction. The classical layers extract features; the quantum layer provides an expressive kernel or variational model. Compare rigorously against LSTM/transformer baselines.

### 6.2 Architecture: Hybrid QML Model

```
Input (returns, features)
    ↓
Classical Encoder (Dense layers, 30 features → 8 features)
    ↓
Quantum Layer (8-qubit parameterized circuit, depth=4)
    ↓
Measurement (Pauli-Z expectations on all qubits)
    ↓
Classical Decoder (Dense layer, 8 → 3 classes: up/flat/down)
    ↓
Output (prediction probabilities)
```

### 6.3 Parameterized Quantum Circuit

**File: `quantum/circuits/qml_circuits.py`**

```python
def build_variational_classifier(n_qubits, n_layers):
    """Hardware-efficient ansatz for classification.

    Structure per layer:
        1. Ry(θ) rotation on each qubit (data encoding or trainable)
        2. Rz(φ) rotation on each qubit (trainable)
        3. Entangling CX gates (nearest-neighbor, circular)

    Total parameters: n_qubits * 2 * n_layers
    """
    qc = QuantumCircuit(n_qubits)
    params = []

    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            theta = Parameter(f"θ_{layer}_{q}")
            phi = Parameter(f"φ_{layer}_{q}")
            qc.ry(theta, q)
            qc.rz(phi, q)
            params.extend([theta, phi])

        # Entangling layer
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)  # circular

    return qc, params
```

### 6.4 Data Encoding Strategy

**Angle encoding** — map classical features to qubit rotation angles:

```python
def encode_features(circuit, features, n_qubits):
    """Encode classical features as rotation angles.

    Uses re-uploading strategy (Pérez-Salinas et al. 2020):
    interleave data encoding with trainable layers for expressivity.

    features: array of shape (n_qubits,) normalized to [0, 2π]
    """
    for q in range(n_qubits):
        circuit.rx(features[q], q)
```

### 6.5 Training Pipeline

```python
class QuantumMLAgent(BaseAgent):
    """Hybrid classical-quantum model for financial time series.

    Predicts: {up, flat, down} classification for next-day returns.

    Models compared:
      - logistic_regression (baseline from existing ModelAgent)
      - random_forest (baseline from existing ModelAgent)
      - lstm (classical deep learning baseline)
      - hybrid_qml (classical encoder + quantum variational circuit)

    Training:
      - Walk-forward validation ONLY (no random splits)
      - Expanding window: train on [0, t], validate on [t, t+k]
      - Early stopping on validation loss
      - Parameter-shift rule for quantum gradient computation
    """

    DEFAULT_CONFIG = {
        "n_qubits": 8,
        "n_layers": 4,
        "learning_rate": 0.01,
        "n_epochs": 100,
        "batch_size": 32,
        "early_stopping_patience": 10,
        "n_shots": 1024,
        "optimizer": "adam",       # classical optimizer for hybrid training
        "feature_dim": 30,         # input from FeatureAgent
        "encoded_dim": 8,          # compressed before quantum layer
        "targets": ["direction"],  # up/flat/down classification
    }
```

**Output Schema:**
```python
{
    "models": {
        "logistic_regression": {"accuracy": ..., "f1": ..., "sharpe_backtest": ...},
        "random_forest": {"accuracy": ..., "f1": ..., "sharpe_backtest": ...},
        "lstm": {"accuracy": ..., "f1": ..., "sharpe_backtest": ...},
        "hybrid_qml": {"accuracy": ..., "f1": ..., "sharpe_backtest": ...},
    },
    "quantum_specific_metrics": {
        "trainable_params_quantum": "int",
        "trainable_params_lstm": "int",
        "training_time_quantum_s": "float",
        "training_time_lstm_s": "float",
        "expressibility": "float — circuit expressibility metric",
        "entanglement_capability": "float — Meyer-Wallach measure",
    },
    "walk_forward_results": "list[dict] — per-fold metrics for all models",
    "feature_importance_quantum": "dict — Pauli-Z expectation sensitivity",
}
```

### 6.6 Honest Comparison Protocol

1. Match parameter count: quantum model should have comparable trainable params to classical
2. Same walk-forward splits for ALL models
3. Same feature set input (from FeatureAgent)
4. Report mean ± std across folds, not cherry-picked best fold
5. Include training time — quantum will likely be slower
6. Show where quantum features help: high-entanglement regimes, non-linear boundaries
7. Show where they don't: simple linear trends, low-feature settings

---

## 7. Feature 5: QAOA vs Classical Heuristics for NP-Hard Problems

### 7.1 Overview

Implement QAOA and classical heuristics for combinatorial optimization problems relevant to finance (portfolio selection as Max-Cut, asset clustering as graph partitioning, trade scheduling as TSP variant).

### 7.2 Problems

**Problem 1: Max-Cut (Portfolio Partitioning)**

Given a graph G = (V, E) with edge weights (asset correlations), partition vertices into two sets to maximize the weight of cut edges. Financial interpretation: divide assets into two groups with maximum between-group diversification.

```
Hamiltonian: C = Σ_{(i,j)∈E} w_{ij} * (1 - Z_i Z_j) / 2
```

**Problem 2: Minimum Vertex Cover (Essential Asset Selection)**

Select the minimum set of assets that "covers" all correlation relationships above a threshold.

**Problem 3: Weighted Max-k-Cut (Multi-Sector Allocation)**

Partition assets into k sectors to maximize inter-sector diversification. Generalizes Max-Cut to k > 2 groups.

### 7.3 Solvers to Compare

| Solver | Type | Implementation |
|--------|------|---------------|
| QAOA (p=1..5) | Quantum | Qiskit QAOA with COBYLA |
| VQE | Quantum | Variational eigensolver |
| Brute force | Classical (exact) | Enumerate all 2^N (small N only) |
| Simulated annealing | Classical (heuristic) | Custom with geometric cooling |
| Greedy | Classical (heuristic) | Deterministic greedy construction |
| Goemans-Williamson | Classical (approx) | SDP relaxation + rounding |
| Branch and bound | Classical (exact) | CPLEX/Gurobi via Python interface |

### 7.4 Scaling Experiments

```
For N in {4, 6, 8, 10, 12, 14, 16, 20}:
    1. Generate random Erdős-Rényi graph G(N, 0.5) with random weights
    2. Also generate correlation graph from real stock data (N stocks)
    3. For each solver:
        - Record: solution value, runtime, approximation ratio
    4. Repeat 20 times per (N, solver) for statistical significance

Plot:
    - X: problem size N
    - Y1: approximation ratio (solution / optimal)
    - Y2: runtime (log scale)
    - Lines: one per solver, with confidence bands
```

### 7.5 Financial Relevance

Show that these abstract combinatorial problems map directly to portfolio construction:

```
Asset Correlation Graph → Max-Cut → Diversified long/short partition
Stock Universe → Vertex Cover → Core holdings selection
Multi-sector allocation → Max-k-Cut → Sector-balanced portfolio
Trade execution scheduling → TSP variant → Optimal order routing
```

---

## 8. Feature 6: Noise Simulation & Error Mitigation

### 8.1 Overview

All previous features assume ideal quantum execution. This feature adds realistic quantum noise and demonstrates error mitigation techniques, showing how noise degrades financial outputs and how mitigation recovers accuracy.

### 8.2 Noise Models

**File: `quantum/noise/noise_models.py`**

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

def build_realistic_noise_model(backend_name="ibm_brisbane"):
    """Build noise model from real IBM hardware calibration data.

    Includes:
        - Depolarizing error on 1-qubit gates (p ~ 0.001)
        - Depolarizing error on 2-qubit gates (p ~ 0.01)
        - Thermal relaxation (T1 ~ 200μs, T2 ~ 150μs)
        - Readout errors (~2% bit-flip on measurement)
    """

def build_parameterized_noise_model(p1=0.001, p2=0.01, readout=0.02):
    """Build noise model with configurable error rates.

    Useful for scaling analysis: vary noise level and observe impact.
    """
```

### 8.3 Error Mitigation Techniques

| Technique | Description | Overhead |
|-----------|-------------|----------|
| **Zero-Noise Extrapolation (ZNE)** | Run at multiple noise levels, extrapolate to zero noise | 3-5x shots |
| **Probabilistic Error Cancellation (PEC)** | Inverse noise channel via quasi-probability decomposition | Exponential in circuit size |
| **M3 Readout Mitigation** | Matrix-free measurement mitigation | Minimal overhead |
| **Twirled Readout Error Extinction (TREX)** | Randomized readout error correction | 2x shots |

**File: `quantum/noise/error_mitigation.py`**

```python
class ErrorMitigator:
    """Apply error mitigation to noisy quantum results.

    Supports:
        - zne: Zero-Noise Extrapolation (Richardson, linear, polynomial)
        - pec: Probabilistic Error Cancellation
        - m3: Matrix-free measurement mitigation
        - trex: Twirled Readout Error Extinction

    Usage:
        mitigator = ErrorMitigator(technique="zne", scale_factors=[1, 2, 3])
        mitigated_result = mitigator.apply(noisy_results, circuit, backend)
    """
```

### 8.4 Impact Analysis

For each quantum feature, run three variants:

```
1. Ideal simulation (statevector, no noise)
2. Noisy simulation (realistic IBM noise model)
3. Noisy + mitigated (noise model + ZNE/M3)

Compare:
    - Portfolio optimization: weight accuracy vs classical, objective value
    - Monte Carlo pricing: pricing error vs analytical
    - QML: classification accuracy degradation
    - QAOA: approximation ratio degradation
```

**Output format per experiment:**
```python
{
    "ideal": {"metric": value, "runtime_ms": ...},
    "noisy": {"metric": value, "runtime_ms": ..., "degradation_pct": ...},
    "mitigated": {"metric": value, "runtime_ms": ..., "recovery_pct": ...},
    "noise_parameters": {"p1": ..., "p2": ..., "readout": ...},
    "mitigation_overhead_factor": float,  # mitigated_shots / base_shots
}
```

---

## 9. Feature 7: Quantum Backtesting Engine

### 9.1 Overview

Extend the existing event-driven `BacktestAgent` to support pluggable quantum subroutines at portfolio rebalancing and risk assessment points. This is the integration layer that ties all quantum features into the trading simulation.

### 9.2 Architecture

```
QuantumBacktestAgent extends BacktestAgent:

For each timestep t:
    1. Receive signal from ModelAgent (or QuantumMLAgent)
    2. At rebalance points:
        a. Call PortfolioAgent OR QuantumPortfolioAgent for weight optimization
        b. Optionally call QuantumMonteCarloAgent for options hedging
    3. Apply RiskAgent constraints
    4. Execute trades with costs/slippage
    5. Log quantum-specific metrics (circuit depth, shots, runtime)
```

**File: `agents/quantum/quantum_backtest_agent.py`**

```python
class QuantumBacktestAgent(BaseAgent):
    """Event-driven backtesting with pluggable quantum subroutines.

    Extends BacktestAgent execution model:
      - Signal at bar i executes at open of bar i+1 (no look-ahead)
      - Transaction costs + slippage applied
      - Quantum subroutines called at rebalance points

    Quantum integration points:
      - portfolio_optimizer: "classical" | "qaoa" | "vqe" | "hybrid"
      - risk_sampler: "classical_mc" | "quantum_ae"
      - signal_model: "logistic" | "random_forest" | "hybrid_qml"

    All quantum calls are timed and logged for performance comparison.
    """

    DEFAULT_CONFIG = {
        # Inherit BacktestAgent defaults
        "initial_capital": 100_000.0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,

        # Quantum configuration
        "portfolio_optimizer": "classical",   # or "qaoa", "vqe", "hybrid"
        "risk_sampler": "classical_mc",       # or "quantum_ae"
        "signal_model": "logistic",           # or "hybrid_qml"
        "noise_model": None,                  # None = ideal, "ibm_brisbane" = realistic
        "error_mitigation": None,             # None, "zne", "m3"

        # Comparison mode: run BOTH classical and quantum, compare
        "comparison_mode": True,
    }
```

**Output Schema:**
```python
{
    "classical_backtest": {
        # Standard BacktestAgent output
        "equity_curve": "pd.Series",
        "trade_log": "list[dict]",
        "metrics": {"sharpe": ..., "max_drawdown": ..., "total_return": ...},
    },
    "quantum_backtest": {
        # Same structure with quantum subroutines
        "equity_curve": "pd.Series",
        "trade_log": "list[dict]",
        "metrics": {"sharpe": ..., "max_drawdown": ..., "total_return": ...},
    },
    "quantum_overhead": {
        "total_quantum_runtime_ms": "float",
        "total_classical_runtime_ms": "float",
        "n_quantum_calls": "int",
        "avg_circuit_depth": "float",
        "avg_shots_per_call": "float",
    },
    "comparison": {
        "sharpe_difference": "float — quantum - classical",
        "return_difference": "float",
        "drawdown_difference": "float",
        "runtime_ratio": "float — quantum / classical",
        "statistical_significance": "float — p-value from bootstrap test",
    },
}
```

### 9.3 Walk-Forward Integration

The quantum backtesting engine uses the same walk-forward validation as `WalkForwardAgent`:

```
For each fold [train_start, train_end, test_start, test_end]:
    1. Train models on [train_start, train_end]
    2. Generate signals on [test_start, test_end]
    3. Run classical backtest
    4. Run quantum backtest (same signals, quantum optimization)
    5. Compare out-of-sample performance
    6. Log to /experiments/
```

---

## 10. C++ Performance Layer

### 10.1 Overview

Performance-critical numerical routines implemented in C++ with pybind11 bindings. Provides 10-100x speedup for hot paths in Monte Carlo simulation, matrix operations, and optimization.

### 10.2 Components

#### 10.2.1 Matrix Operations (`cpp/src/matrix_ops.cpp`)

```cpp
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace financebro {

/**
 * Ledoit-Wolf shrinkage covariance estimation.
 * O(N²T) where N = assets, T = observations.
 * ~50x faster than sklearn for large N.
 */
Eigen::MatrixXd ledoit_wolf_shrinkage(
    const Eigen::MatrixXd& returns  // T x N matrix
);

/**
 * Cholesky decomposition with positive-definite fix.
 * Used for: correlated random variate generation in MC.
 */
Eigen::MatrixXd cholesky_with_fix(
    const Eigen::MatrixXd& covariance
);

/**
 * Fast eigendecomposition for minimum-variance portfolio.
 * Returns only the smallest k eigenvalues/vectors.
 */
std::pair<Eigen::VectorXd, Eigen::MatrixXd> partial_eigen(
    const Eigen::MatrixXd& covariance,
    int k
);

}  // namespace financebro
```

#### 10.2.2 Monte Carlo Engine (`cpp/src/monte_carlo.cpp`)

```cpp
#include <random>
#include <vector>
#include <omp.h>  // OpenMP for parallel paths

namespace financebro {

/**
 * High-performance Monte Carlo option pricer.
 * Features:
 *   - Parallel path generation (OpenMP)
 *   - Antithetic variates (halves variance for free)
 *   - Control variates (geometric average Asian as control)
 *   - SIMD-friendly memory layout
 *
 * Performance: ~1M paths in <10ms on modern hardware.
 */
struct MCResult {
    double price;
    double std_error;
    double ci_lower;
    double ci_upper;
    int64_t n_paths;
    double runtime_us;
};

MCResult price_european_call(
    double S0, double K, double r, double sigma, double T,
    int64_t n_paths,
    bool antithetic = true,
    int n_threads = 0  // 0 = auto-detect
);

MCResult price_european_put(
    double S0, double K, double r, double sigma, double T,
    int64_t n_paths,
    bool antithetic = true,
    int n_threads = 0
);

/**
 * Correlated multi-asset MC for portfolio VaR.
 * Generates correlated paths via Cholesky decomposition.
 */
struct PortfolioVaRResult {
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double runtime_us;
};

PortfolioVaRResult compute_portfolio_var(
    const Eigen::VectorXd& weights,
    const Eigen::VectorXd& expected_returns,
    const Eigen::MatrixXd& covariance,
    double horizon_days,
    int64_t n_scenarios,
    int n_threads = 0
);

}  // namespace financebro
```

#### 10.2.3 Portfolio Optimizer (`cpp/src/portfolio_optimizer.cpp`)

```cpp
namespace financebro {

/**
 * Active-set quadratic programming solver for Markowitz.
 * Solves: min 0.5 * w^T Q w + c^T w  s.t.  Ax <= b, Cx = d, lb <= w <= ub
 *
 * Faster than OSQP for small-medium problems (N < 100 assets).
 * ~20x faster than cvxpy Python overhead for N < 50.
 */
struct QPResult {
    Eigen::VectorXd weights;
    double objective_value;
    int iterations;
    double runtime_us;
    bool converged;
};

QPResult solve_markowitz_qp(
    const Eigen::MatrixXd& covariance,  // N x N
    const Eigen::VectorXd& expected_returns,  // N
    double target_return,
    double max_weight,
    bool long_only = true
);

/**
 * Generate efficient frontier points.
 * Solves QP for each target return in [min_ret, max_ret].
 */
std::vector<QPResult> efficient_frontier(
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& expected_returns,
    int n_points,
    double max_weight,
    bool long_only = true
);

}  // namespace financebro
```

#### 10.2.4 Statevector Simulator (`cpp/src/circuit_simulator.cpp`)

```cpp
#include <complex>
#include <vector>

namespace financebro {

using Complex = std::complex<double>;
using StateVector = std::vector<Complex>;

/**
 * Lightweight statevector simulator for small circuits (< 20 qubits).
 * Used as a fast backend for QAOA parameter optimization loops
 * where Qiskit Aer overhead dominates.
 *
 * Supports: H, X, Y, Z, Rx, Ry, Rz, CX, CZ, SWAP
 * Performance: ~5x faster than Aer for < 16 qubits (avoids Python overhead)
 */
class CircuitSimulator {
public:
    explicit CircuitSimulator(int n_qubits);

    void apply_h(int qubit);
    void apply_x(int qubit);
    void apply_rx(int qubit, double theta);
    void apply_ry(int qubit, double theta);
    void apply_rz(int qubit, double theta);
    void apply_cx(int control, int target);
    void apply_cz(int control, int target);

    // Measure expectation value of Pauli-Z on target qubit
    double expectation_z(int qubit) const;

    // Measure all qubits, sample n_shots bitstrings
    std::vector<int64_t> sample(int n_shots) const;

    // Get full statevector (for debugging/validation)
    const StateVector& statevector() const;

    void reset();

private:
    int n_qubits_;
    StateVector state_;

    void apply_single_qubit_gate(int qubit, const Complex gate[2][2]);
    void apply_controlled_gate(int control, int target, const Complex gate[2][2]);
};

}  // namespace financebro
```

### 10.3 Build System

**File: `cpp/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.16)
project(financebro_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Dependencies
find_package(Eigen3 3.4 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

# Core library
add_library(financebro_core STATIC
    src/matrix_ops.cpp
    src/monte_carlo.cpp
    src/portfolio_optimizer.cpp
    src/circuit_simulator.cpp
)
target_link_libraries(financebro_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(financebro_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python bindings
pybind11_add_module(financebro_native bindings/pybind_module.cpp)
target_link_libraries(financebro_native PRIVATE financebro_core)

# C++ tests
enable_testing()
find_package(GTest REQUIRED)

add_executable(test_matrix_ops tests/test_matrix_ops.cpp)
target_link_libraries(test_matrix_ops financebro_core GTest::gtest_main)
add_test(NAME MatrixOps COMMAND test_matrix_ops)

add_executable(test_monte_carlo tests/test_monte_carlo.cpp)
target_link_libraries(test_monte_carlo financebro_core GTest::gtest_main)
add_test(NAME MonteCarlo COMMAND test_monte_carlo)

add_executable(test_portfolio tests/test_portfolio_optimizer.cpp)
target_link_libraries(test_portfolio financebro_core GTest::gtest_main)
add_test(NAME PortfolioOptimizer COMMAND test_portfolio)
```

### 10.4 Python Bindings

**File: `cpp/bindings/pybind_module.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "matrix_ops.hpp"
#include "monte_carlo.hpp"
#include "portfolio_optimizer.hpp"
#include "circuit_simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(financebro_native, m) {
    m.doc() = "FinanceBro C++ performance library";

    // Matrix operations
    m.def("ledoit_wolf_shrinkage", &financebro::ledoit_wolf_shrinkage);
    m.def("cholesky_with_fix", &financebro::cholesky_with_fix);
    m.def("partial_eigen", &financebro::partial_eigen);

    // Monte Carlo
    py::class_<financebro::MCResult>(m, "MCResult")
        .def_readonly("price", &financebro::MCResult::price)
        .def_readonly("std_error", &financebro::MCResult::std_error)
        .def_readonly("ci_lower", &financebro::MCResult::ci_lower)
        .def_readonly("ci_upper", &financebro::MCResult::ci_upper)
        .def_readonly("runtime_us", &financebro::MCResult::runtime_us);

    m.def("price_european_call", &financebro::price_european_call,
        py::arg("S0"), py::arg("K"), py::arg("r"),
        py::arg("sigma"), py::arg("T"), py::arg("n_paths"),
        py::arg("antithetic") = true, py::arg("n_threads") = 0);

    // Portfolio optimizer
    py::class_<financebro::QPResult>(m, "QPResult")
        .def_readonly("weights", &financebro::QPResult::weights)
        .def_readonly("objective_value", &financebro::QPResult::objective_value)
        .def_readonly("runtime_us", &financebro::QPResult::runtime_us)
        .def_readonly("converged", &financebro::QPResult::converged);

    m.def("solve_markowitz_qp", &financebro::solve_markowitz_qp);
    m.def("efficient_frontier", &financebro::efficient_frontier);

    // Circuit simulator
    py::class_<financebro::CircuitSimulator>(m, "CircuitSimulator")
        .def(py::init<int>())
        .def("apply_h", &financebro::CircuitSimulator::apply_h)
        .def("apply_rx", &financebro::CircuitSimulator::apply_rx)
        .def("apply_ry", &financebro::CircuitSimulator::apply_ry)
        .def("apply_rz", &financebro::CircuitSimulator::apply_rz)
        .def("apply_cx", &financebro::CircuitSimulator::apply_cx)
        .def("expectation_z", &financebro::CircuitSimulator::expectation_z)
        .def("sample", &financebro::CircuitSimulator::sample)
        .def("reset", &financebro::CircuitSimulator::reset);
}
```

### 10.5 Graceful Fallback

All C++ bindings are optional. Python code MUST work without them:

```python
try:
    from financebro_native import solve_markowitz_qp, price_european_call
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    # Fall back to pure Python / NumPy implementations

def solve_portfolio(mu, Sigma, r_target, w_max):
    if HAS_CPP:
        return solve_markowitz_qp(Sigma, mu, r_target, w_max)
    else:
        return _solve_markowitz_cvxpy(mu, Sigma, r_target, w_max)
```

---

## 11. Dependency Map & New Libraries

### 11.1 Python Dependencies (add to requirements.txt)

```
# Quantum Computing
qiskit>=1.0                          # Core quantum circuits + transpiler
qiskit-aer>=0.14                     # Statevector + noise simulation
qiskit-algorithms>=0.3               # QAOA, VQE, amplitude estimation
qiskit-optimization>=0.6             # QuadraticProgram, QUBO conversion
qiskit-machine-learning>=0.7         # Quantum neural networks, classifiers
qiskit-ibm-runtime>=0.20             # IBM hardware access (optional)

# Classical Optimization
cvxpy>=1.4                           # Convex optimization (Markowitz baseline)

# ML Comparison
torch>=2.0                           # LSTM baseline
pennylane>=0.35                      # Alternative QML framework (optional)
pennylane-qiskit>=0.35               # PennyLane-Qiskit bridge (optional)

# C++ Build
pybind11>=2.11                       # Python-C++ bindings
cmake>=3.16                          # Build system (pip install cmake)

# Visualization
plotly>=5.18                         # Interactive plots for frontiers
```

### 11.2 C++ Dependencies

```
# System packages (install via Homebrew on macOS)
brew install eigen
brew install libomp       # OpenMP support on macOS
brew install googletest   # C++ unit tests

# Python packages (pip)
pip install pybind11 cmake
```

### 11.3 Optional (for real quantum hardware)

```
# IBM Quantum credentials (set in .env, NEVER in code)
IBMQ_TOKEN=<your-token>
IBMQ_BACKEND=ibm_brisbane
```

---

## 12. Project Structure (Full)

After implementation, the new files added to the project:

```
backend/
├── agents/
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── quantum_portfolio_agent.py      # ~500 lines
│   │   ├── quantum_montecarlo_agent.py     # ~400 lines
│   │   ├── quantum_ml_agent.py             # ~600 lines
│   │   ├── quantum_backtest_agent.py       # ~450 lines
│   │   └── noise_simulation_agent.py       # ~300 lines
│   └── ...
├── quantum/
│   ├── __init__.py
│   ├── circuits/
│   │   ├── __init__.py
│   │   ├── portfolio_circuit.py            # ~200 lines
│   │   ├── amplitude_estimation.py         # ~250 lines
│   │   ├── variational.py                  # ~150 lines
│   │   ├── qml_circuits.py                 # ~200 lines
│   │   └── compiler/
│   │       ├── __init__.py
│   │       ├── optimizer.py                # ~400 lines
│   │       ├── transpiler.py               # ~300 lines
│   │       ├── peephole.py                 # ~250 lines
│   │       └── hardware_constraints.py     # ~100 lines
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── noise_models.py                 # ~200 lines
│   │   ├── error_mitigation.py             # ~350 lines
│   │   └── noise_benchmarks.py             # ~200 lines
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── qaoa_solver.py                  # ~300 lines
│   │   ├── vqe_solver.py                   # ~250 lines
│   │   ├── classical_solvers.py            # ~250 lines
│   │   └── problem_encodings.py            # ~200 lines
│   └── benchmarks/
│       ├── __init__.py
│       ├── benchmark_runner.py             # ~300 lines
│       ├── metrics.py                      # ~150 lines
│       └── scaling_analysis.py             # ~200 lines
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── matrix_ops.hpp
│   │   ├── monte_carlo.hpp
│   │   ├── portfolio_optimizer.hpp
│   │   └── circuit_simulator.hpp
│   ├── src/
│   │   ├── matrix_ops.cpp                  # ~300 lines
│   │   ├── monte_carlo.cpp                 # ~350 lines
│   │   ├── portfolio_optimizer.cpp         # ~400 lines
│   │   └── circuit_simulator.cpp           # ~500 lines
│   ├── bindings/
│   │   └── pybind_module.cpp               # ~150 lines
│   └── tests/
│       ├── test_matrix_ops.cpp             # ~200 lines
│       ├── test_monte_carlo.cpp            # ~200 lines
│       └── test_portfolio_optimizer.cpp    # ~200 lines
├── specs/
│   └── quantum_implementation_spec.md      # This document
└── tests/
    ├── test_quantum_portfolio_agent.py     # ~500 lines
    ├── test_quantum_montecarlo_agent.py    # ~400 lines
    ├── test_quantum_ml_agent.py            # ~450 lines
    ├── test_quantum_backtest_agent.py      # ~500 lines
    ├── test_noise_simulation_agent.py      # ~350 lines
    ├── test_circuit_compiler.py            # ~400 lines
    ├── test_qaoa_solver.py                 # ~350 lines
    └── test_cpp_bindings.py                # ~200 lines

Estimated total new code: ~10,500 lines Python + ~1,800 lines C++
```

---

## 13. Phased Implementation Plan

### Phase 0: Foundation (Week 1)

**Goal**: Set up quantum infrastructure, install dependencies, verify toolchain.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 0.1 | Add quantum dependencies to requirements.txt | Updated requirements.txt | — |
| 0.2 | Create `quantum/` package structure with `__init__.py` files | Directory skeleton | ~50 |
| 0.3 | Create `agents/quantum/` package | Directory skeleton | ~20 |
| 0.4 | Set up C++ build system (CMakeLists.txt, pybind11 config) | Working `pip install -e .` | ~100 |
| 0.5 | Verify Qiskit installation with hello-world QAOA | Passing smoke test | ~50 |
| 0.6 | Verify C++ build + Python binding works | Passing binding test | ~100 |

**Blocked by**: Nothing — can start immediately.

---

### Phase 1: Classical Baselines + C++ Core (Week 2-3)

**Goal**: Implement classical solvers first. These are the baselines everything quantum is compared against.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 1.1 | Implement `classical_solvers.py` (SA, greedy, G-W) | Passing tests | ~250 |
| 1.2 | Implement `cvxpy` Markowitz solver in QuantumPortfolioAgent | Passing tests, verified against existing PortfolioAgent | ~200 |
| 1.3 | Implement classical Monte Carlo pricer (Python) | Verified against Black-Scholes analytical | ~150 |
| 1.4 | Implement C++ matrix_ops (Ledoit-Wolf, Cholesky) | C++ tests passing, Python bindings working | ~400 |
| 1.5 | Implement C++ monte_carlo engine | 1M paths < 10ms benchmark | ~350 |
| 1.6 | Implement C++ portfolio_optimizer (QP solver) | Verified against cvxpy results | ~400 |
| 1.7 | Implement Python fallbacks for all C++ functions | Tests pass with `HAS_CPP=False` | ~200 |
| 1.8 | Benchmark C++ vs Python for all numerical ops | Benchmark report in experiments/ | — |

**Key validation**: Classical baselines must match existing PortfolioAgent results (within numerical tolerance) before proceeding.

---

### Phase 2: Quantum Portfolio Optimization — Feature 1 (Week 3-4)

**Goal**: QAOA portfolio optimization with rigorous comparison to classical.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 2.1 | Implement `problem_encodings.py` (QUBO formulation) | Unit tests for encoding correctness | ~200 |
| 2.2 | Implement `portfolio_circuit.py` (QAOA circuit builder) | Circuit generates, runs on simulator | ~200 |
| 2.3 | Implement `qaoa_solver.py` (QAOA optimization loop) | Solves small portfolios (3-5 assets) | ~300 |
| 2.4 | Implement `quantum_portfolio_agent.py` (full agent) | BaseAgent contract satisfied | ~500 |
| 2.5 | Implement efficient frontier generation (classical + quantum) | Side-by-side frontier comparison | ~150 |
| 2.6 | Scaling analysis: N = {3, 5, 7, 10} assets | Plots + metrics logged to experiments/ | ~100 |
| 2.7 | Write comprehensive tests | 80%+ coverage | ~500 |

---

### Phase 3: Quantum Monte Carlo — Feature 2 (Week 4-5)

**Goal**: Amplitude estimation for option pricing, convergence analysis.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 3.1 | Implement `amplitude_estimation.py` (QAE circuits) | Log-normal loading verified | ~250 |
| 3.2 | Implement `quantum_montecarlo_agent.py` | Prices match BS ± tolerance | ~400 |
| 3.3 | Convergence analysis: classical MC vs QAE | Scaling exponent plot | ~150 |
| 3.4 | Integration with C++ MC engine for comparison | Runtime benchmarks | ~100 |
| 3.5 | Write tests | 80%+ coverage | ~400 |

---

### Phase 4: Circuit Compiler — Feature 3 (Week 5-6)

**Goal**: Custom circuit optimization pipeline.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 4.1 | Implement `peephole.py` (local gate optimization) | Cancellation rules working | ~250 |
| 4.2 | Implement `optimizer.py` (multi-pass pipeline) | Before/after metrics for test circuits | ~400 |
| 4.3 | Implement `transpiler.py` (hardware-aware routing) | IBM basis gate decomposition | ~300 |
| 4.4 | Implement `hardware_constraints.py` (IBM topologies) | Backend configs loaded correctly | ~100 |
| 4.5 | Apply optimizer to Phase 2-3 circuits | Gate/depth reduction measured | — |
| 4.6 | Implement C++ `circuit_simulator.cpp` | 5x faster than Aer for <16 qubits | ~500 |
| 4.7 | Write tests | 80%+ coverage | ~400 |

---

### Phase 5: Quantum ML — Feature 4 (Week 6-7)

**Goal**: Hybrid quantum-classical model for time series, honest comparison.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 5.1 | Implement `qml_circuits.py` (variational ansatz) | Parameterized circuits build correctly | ~200 |
| 5.2 | Implement `variational.py` (encoding + training) | Gradient computation via parameter-shift | ~150 |
| 5.3 | Implement `quantum_ml_agent.py` (full hybrid model) | Walk-forward training works | ~600 |
| 5.4 | Implement LSTM baseline for comparison | Same features, same splits | ~200 |
| 5.5 | Walk-forward evaluation: all 4 models | Per-fold metrics for all models | ~150 |
| 5.6 | Write tests | 80%+ coverage | ~450 |

---

### Phase 6: QAOA vs Classical — Feature 5 (Week 7-8)

**Goal**: NP-hard problem benchmarks with financial relevance.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 6.1 | Implement Max-Cut, Vertex Cover, Max-k-Cut encodings | QUBO formulations correct | ~200 |
| 6.2 | Financial interpretation: correlation graph → Max-Cut | Real stock data graphs | ~100 |
| 6.3 | Scaling experiments N = {4..20} | Approximation ratio + runtime plots | ~300 |
| 6.4 | Implement `vqe_solver.py` as additional comparison | VQE results included | ~250 |
| 6.5 | Write tests | 80%+ coverage | ~350 |

---

### Phase 7: Noise & Error Mitigation — Feature 6 (Week 8-9)

**Goal**: Realistic noise analysis, show impact on financial outputs.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 7.1 | Implement `noise_models.py` (IBM calibration) | Noise model matches real hardware | ~200 |
| 7.2 | Implement `error_mitigation.py` (ZNE, M3, TREX) | Mitigation improves results | ~350 |
| 7.3 | Implement `noise_simulation_agent.py` (wrapper) | Wraps any quantum agent with noise | ~300 |
| 7.4 | Run all Phase 2-6 experiments with noise | Ideal/noisy/mitigated comparison | ~200 |
| 7.5 | Noise scaling: vary error rate, observe degradation | Degradation curves | ~100 |
| 7.6 | Write tests | 80%+ coverage | ~350 |

---

### Phase 8: Quantum Backtesting Engine — Feature 7 (Week 9-10)

**Goal**: Full integration, pluggable quantum subroutines in backtest.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 8.1 | Implement `quantum_backtest_agent.py` | Event-driven with quantum calls at rebalance | ~450 |
| 8.2 | Integration test: full pipeline with quantum portfolio | End-to-end passing | ~200 |
| 8.3 | Comparison mode: classical vs quantum backtest | Statistical significance test | ~150 |
| 8.4 | Full walk-forward: 7 stocks, 3-year history | Logged experiment with all metrics | — |
| 8.5 | Write tests | 80%+ coverage | ~500 |

---

### Phase 9: Reporting & Visualization (Week 10-11)

**Goal**: Publication-quality results, interactive dashboards.

| Step | Task | Deliverable | Est. LOC |
|------|------|------------|----------|
| 9.1 | Efficient frontier visualization (Plotly) | Interactive frontier comparison | ~200 |
| 9.2 | MC convergence plots | Classical vs quantum scaling | ~150 |
| 9.3 | Circuit optimization before/after diagrams | Circuit drawings | ~100 |
| 9.4 | Noise impact heatmaps | Noise rate vs metric degradation | ~150 |
| 9.5 | Scaling analysis summary plots | All NP-hard problem comparisons | ~150 |
| 9.6 | API endpoints for all quantum features | FastAPI routes | ~300 |
| 9.7 | Frontend dashboard components (if time) | React charts for quantum metrics | ~500 |

---

### Phase 10: Documentation & Polish (Week 11-12)

| Step | Task | Deliverable |
|------|------|------------|
| 10.1 | Jupyter notebook: "Quantum Portfolio Optimization" | Runnable, self-contained |
| 10.2 | Jupyter notebook: "Quantum vs Classical Benchmarks" | All scaling results |
| 10.3 | Update CLAUDE.md with quantum pipeline docs | — |
| 10.4 | Update Agents.MD with quantum agent contracts | — |
| 10.5 | Final experiment log: comprehensive benchmark | experiments/ JSON |

---

## 14. Benchmarking & Evaluation Framework

### 14.1 Benchmark Runner

**File: `quantum/benchmarks/benchmark_runner.py`**

Every quantum experiment is wrapped in a standard benchmarking harness:

```python
class BenchmarkRunner:
    """Orchestrates quantum vs classical comparisons.

    For each experiment:
        1. Run classical solver → record (result, runtime, metrics)
        2. Run quantum solver → record (result, runtime, metrics)
        3. Compute comparison metrics
        4. Run statistical significance test (bootstrap)
        5. Log everything to /experiments/

    Results are NEVER cherry-picked. All runs logged.
    """

    def run_comparison(self, problem, classical_solver, quantum_solver,
                       n_repetitions=20, **kwargs):
        results = {
            "classical": [],
            "quantum": [],
            "comparison": {},
        }
        for i in range(n_repetitions):
            c_result = classical_solver.solve(problem)
            q_result = quantum_solver.solve(problem)
            results["classical"].append(c_result)
            results["quantum"].append(q_result)

        results["comparison"] = {
            "mean_approximation_ratio": ...,
            "runtime_ratio": ...,
            "p_value_quality_difference": ...,  # paired t-test or bootstrap
            "quantum_wins_pct": ...,            # % of runs where quantum was better
        }
        self._log_experiment(results)
        return results
```

### 14.2 Metrics Tracked

| Category | Metric | Description |
|----------|--------|-------------|
| **Quality** | Objective value | Solution quality (variance, price, etc.) |
| | Approximation ratio | quantum_solution / optimal_solution |
| | Weight distance (L2) | For portfolio: ||w_q - w_c||_2 |
| | Sharpe ratio | For backtests: risk-adjusted return |
| **Runtime** | Wall-clock time (ms) | End-to-end solve time |
| | Circuit construction time | Time to build quantum circuit |
| | Optimization iterations | Classical optimizer iterations for QAOA |
| **Quantum-specific** | Qubit count | Number of qubits required |
| | Circuit depth | After transpilation |
| | CX gate count | Two-qubit gate count (dominant error source) |
| | Shot count | Measurement repetitions |
| | Estimated fidelity | Product of gate fidelities |
| **Statistical** | Standard error | Of the quantum result |
| | p-value | Significance of quality difference |
| | Bootstrap CI | 95% confidence interval |

### 14.3 Experiment Logging

All quantum experiments log to `/experiments/` following the existing format:

```json
{
    "experiment_id": "quantum_portfolio_20260325_143022",
    "agent": "QuantumPortfolioAgent",
    "timestamp": "2026-03-25T14:30:22Z",
    "config": { ... },
    "results": {
        "classical": { ... },
        "quantum": { ... },
        "comparison": { ... }
    },
    "noise_model": "ibm_brisbane" | null,
    "error_mitigation": "zne" | null,
    "out_of_sample": true,
    "notes": "5-asset portfolio, QAOA p=3, 4096 shots"
}
```

---

## 15. Hard Constraints & Integration Rules

### 15.1 Inherited from FinanceBro Core

All quantum agents MUST respect:

1. **No look-ahead bias**: Quantum optimization at time t uses only data up to t
2. **Time-series validation**: Walk-forward only, no random splits
3. **Realistic costs**: All backtest results include transaction costs and slippage
4. **Experiment tracking**: Every quantum run logged, no silent experiments
5. **BaseAgent contract**: All agents implement run/validate/log_metrics/schemas

### 15.2 Quantum-Specific Constraints

6. **Classical baseline FIRST**: Every quantum result must include classical comparison
7. **Honest reporting**: If quantum is worse, say so. No cherry-picking.
8. **Noise awareness**: Always report whether results are ideal/noisy/mitigated
9. **Reproducibility**: All experiments seeded (both classical random and quantum shots)
10. **Graceful degradation**: All quantum features work in simulation; real hardware is optional
11. **C++ fallback**: All C++ functions have pure-Python fallbacks

### 15.3 Security

- IBMQ credentials stored in `.env` only, NEVER in code or committed to git
- `.env` is in `.gitignore`
- All API keys validated at startup, not at call time

---

## 16. Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Qiskit API breaking changes (v1.0 transition) | HIGH | MEDIUM | Pin versions, isolate quantum code behind interfaces |
| QAOA doesn't beat classical for small N | HIGH | HIGH | Expected — document honestly, show scaling trends |
| C++ build fails on different platforms | MEDIUM | MEDIUM | Docker build environment, CI matrix |
| Circuit depth exceeds hardware limits | MEDIUM | HIGH | Compiler optimization (Feature 3), limit problem size |
| QML training too slow | MEDIUM | HIGH | Reduce circuit depth, use PennyLane lightning.qubit |
| Noise completely destroys quantum advantage | LOW | MEDIUM | Error mitigation (Feature 6), expected result to report |
| IBM hardware access quota limits | LOW | LOW | Use Aer simulator for all benchmarks, hardware is bonus |

---

## 17. API Endpoints

New FastAPI routes to expose quantum features:

```python
# backend/api/routes/quantum.py

@router.post("/quantum/portfolio/optimize")
async def optimize_portfolio_quantum(request: PortfolioOptRequest):
    """Run classical + quantum portfolio optimization, return comparison."""

@router.post("/quantum/montecarlo/price")
async def price_option_quantum(request: OptionPriceRequest):
    """Price option via classical MC + quantum AE, return comparison."""

@router.post("/quantum/circuit/optimize")
async def optimize_circuit(request: CircuitOptRequest):
    """Optimize a quantum circuit, return before/after metrics."""

@router.post("/quantum/backtest/run")
async def run_quantum_backtest(request: BacktestRequest):
    """Run backtest with quantum subroutines, compare to classical."""

@router.get("/quantum/benchmarks/{experiment_id}")
async def get_benchmark_results(experiment_id: str):
    """Retrieve logged quantum benchmark results."""

@router.get("/quantum/benchmarks/latest")
async def get_latest_benchmarks():
    """Get most recent quantum vs classical comparison results."""
```

---

## 18. Frontend Visualization Requirements

### 18.1 Efficient Frontier Chart
- Interactive Plotly scatter + line chart
- Classical frontier (blue line) vs quantum points (red dots)
- Hover: show weights, Sharpe, runtime for each point
- Slider: vary QAOA layers (p=1..5) to show convergence

### 18.2 Monte Carlo Convergence
- Log-log plot: computational cost vs estimation error
- Two lines: classical MC (slope -0.5) vs QAE (slope -1.0)
- Confidence bands from multiple runs

### 18.3 Circuit Optimization Dashboard
- Before/after circuit diagrams (rendered as images)
- Bar chart: gate count, depth, CX count reduction
- Table: per-pass optimization metrics

### 18.4 Noise Impact Heatmap
- X-axis: noise parameter (gate error rate)
- Y-axis: metric (Sharpe, pricing error, approx ratio)
- Color: degradation from ideal
- Three rows: ideal / noisy / mitigated

### 18.5 Scaling Analysis
- Multi-line plot: problem size vs runtime/quality
- One line per solver (QAOA, SA, greedy, exact)
- Log scale for runtime axis

---

## Appendix A: Mathematical Background

### A.1 QAOA Derivation

The Quantum Approximate Optimization Algorithm works by:

1. **Encoding** the cost function C into a diagonal Hamiltonian H_C
2. **Preparing** the uniform superposition |+>^n
3. **Alternating** between cost unitary exp(-iγH_C) and mixer unitary exp(-iβH_M)
4. **Measuring** in the computational basis
5. **Optimizing** (γ, β) classically to minimize <C>

For p layers, the approximation ratio improves monotonically with p.
At p → ∞, QAOA recovers the exact solution (adiabatic limit).

For Max-Cut on 3-regular graphs, QAOA p=1 guarantees approximation ratio ≥ 0.6924
(Farhi et al. 2014), vs classical greedy ≈ 0.5.

### A.2 Quantum Amplitude Estimation

Given an operator A such that A|0> = sin(θ)|good> + cos(θ)|bad>,
QAE estimates θ (and hence sin²(θ) = probability of good outcome) using
O(1/ε) applications of A, vs O(1/ε²) for classical sampling.

For option pricing, A encodes the payoff function and |good> corresponds to
the payoff being positive.

### A.3 Portfolio QUBO Encoding

For N assets with K-bit weight precision, the QUBO has N*K binary variables.
The number of qubits = N*K. For 10 assets with 3-bit precision: 30 qubits.

The QUBO matrix Q is constructed as:
```
Q = λ_var * Q_variance + λ_ret * Q_return + λ_budget * Q_budget
```

where penalty strengths λ must be tuned carefully (too low = constraints violated,
too high = cost landscape flattened).

---

## Appendix B: Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB (for large statevector sims) |
| CPU | 4 cores | 8+ cores (C++ OpenMP parallelism) |
| GPU | Not required | NVIDIA (for PyTorch LSTM baseline) |
| Disk | 5 GB free | 10 GB (cached data + models) |
| Python | 3.9+ | 3.11 (best Qiskit compatibility) |
| C++ compiler | GCC 9+ / Clang 12+ | GCC 12+ (C++17, AVX2) |
| CMake | 3.16+ | 3.25+ |

---

## Decision: WAITING FOR CONFIRMATION

**This plan covers all 7 quantum features + C++ performance layer + benchmarking framework.**

Before proceeding to implementation:

1. **Confirm scope**: All 7 features, or prioritize a subset?
2. **Confirm phase order**: Phase 1 (classical baselines + C++) → Phase 2 (QAOA portfolio) first?
3. **C++ priority**: Build C++ layer first, or defer to later phases?
4. **Hardware access**: Do you have IBM Quantum credentials, or simulation-only?
5. **Frontend**: Include frontend visualization, or backend-only for now?

**Recommended minimum viable path**: Phases 0 → 1 → 2 → 3 → 7 gives you the "Quantum vs Classical Portfolio Optimization" project that gets brought up in interviews, with Monte Carlo as a strong second showcase and the quantum backtesting engine as the integration capstone.

**WAITING FOR CONFIRMATION**: Proceed with this plan? (yes / modify / prioritize subset)
