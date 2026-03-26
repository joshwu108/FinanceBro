"""Quantum Monte Carlo Agent — Black-Scholes, Classical MC, Quantum AE.

Compares classical Monte Carlo (O(1/sqrt(N)) convergence) against
quantum amplitude estimation (O(1/M) convergence) for European option
pricing, with Black-Scholes analytical as ground truth.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analytical pricing
# ---------------------------------------------------------------------------

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European call price.

    C = S*N(d1) - K*exp(-rT)*N(d2)
    """
    if sigma < 1e-10:
        return max(S - K * np.exp(-r * T), 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European put via put-call parity: P = C - S + K*exp(-rT)."""
    call = black_scholes_call(S, K, r, sigma, T)
    return float(call - S + K * np.exp(-r * T))


# ---------------------------------------------------------------------------
# Classical Monte Carlo
# ---------------------------------------------------------------------------

def classical_mc_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int,
    option_type: str,
    antithetic: bool = False,
) -> Dict[str, Any]:
    """Classical Monte Carlo option pricing.

    GBM: S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)

    With antithetic variates, pair each Z with -Z for variance reduction.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    Z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma ** 2) * T
    vol_term = sigma * np.sqrt(T)

    if antithetic:
        S_T = S0 * np.exp(drift + vol_term * Z)
        S_T_anti = S0 * np.exp(drift + vol_term * (-Z))

        if option_type == "call":
            payoffs = 0.5 * (np.maximum(S_T - K, 0) + np.maximum(S_T_anti - K, 0))
        else:
            payoffs = 0.5 * (np.maximum(K - S_T, 0) + np.maximum(K - S_T_anti, 0))
    else:
        S_T = S0 * np.exp(drift + vol_term * Z)
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "price": price,
        "std_error": std_error,
        "ci_lower": price - 1.96 * std_error,
        "ci_upper": price + 1.96 * std_error,
        "runtime_ms": elapsed_ms,
        "n_paths": n_paths,
    }


# ---------------------------------------------------------------------------
# Quantum amplitude estimation (simulated)
# ---------------------------------------------------------------------------

def quantum_ae_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_qubits: int,
    n_estimation_qubits: int,
) -> Dict[str, Any]:
    """Simulated quantum amplitude estimation for option pricing.

    Key property: QAE convergence is O(1/M) where M = 2^n_estimation_qubits,
    compared to classical MC's O(1/sqrt(N)).  This is a quadratic speedup.

    We simulate the QAE circuit classically: discretize the log-normal
    distribution on 2^n_qubits grid points and estimate the expected payoff
    with noise proportional to 1/M.
    """
    start = time.perf_counter()

    n_price_points = 2 ** n_qubits
    M = 2 ** n_estimation_qubits  # effective Grover iterations

    # Log-normal parameters for terminal price
    mu_log = np.log(S0) + (r - 0.5 * sigma ** 2) * T
    sigma_log = sigma * np.sqrt(T)

    # Discretize on grid
    n_std = 4
    log_prices = np.linspace(
        mu_log - n_std * sigma_log,
        mu_log + n_std * sigma_log,
        n_price_points,
    )
    prices = np.exp(log_prices)

    # Probability weights (discretized log-normal)
    probs = norm.pdf(log_prices, loc=mu_log, scale=sigma_log)
    probs = probs / np.sum(probs)

    # Call payoff
    payoffs = np.maximum(prices - K, 0)
    expected_payoff = float(np.sum(probs * payoffs))

    # Simulate QAE estimation error: scales as O(1/M)
    rng = np.random.default_rng(42 + n_estimation_qubits)
    noise_scale = max(expected_payoff, 0.01) / (M + 1)
    estimation_noise = rng.normal(0, noise_scale)

    estimated_payoff = max(expected_payoff + estimation_noise, 0.0)
    price = float(np.exp(-r * T) * estimated_payoff)

    # Circuit depth: QPE requires O(2^n_est) controlled-U applications
    circuit_depth = int(n_qubits * 3 + n_estimation_qubits * (2 ** n_estimation_qubits))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "price": price,
        "runtime_ms": elapsed_ms,
        "n_qubits": n_qubits + n_estimation_qubits,
        "circuit_depth": circuit_depth,
    }


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_analysis(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    classical_path_counts: List[int],
    quantum_estimation_qubits: List[int],
    n_qubits_price: int,
) -> Dict[str, Any]:
    """Compare convergence scaling: classical MC vs quantum AE."""
    classical_results = []
    for n_paths in classical_path_counts:
        result = classical_mc_price(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            n_paths=n_paths, seed=42, option_type="call",
        )
        classical_results.append(result)

    quantum_results = []
    for n_est in quantum_estimation_qubits:
        result = quantum_ae_price(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            n_qubits=n_qubits_price, n_estimation_qubits=n_est,
        )
        quantum_results.append(result)

    return {
        "classical": classical_results,
        "quantum": quantum_results,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QuantumMonteCarloAgent(BaseAgent):
    """Option pricing agent: BS analytical + classical MC + quantum AE."""

    DEFAULT_CONFIG = {
        "methods": ["classical_mc", "quantum_ae"],
        "n_classical_paths": 100_000,
        "n_qubits_price": 4,
        "n_estimation_qubits": 4,
        "seed": 42,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "spot_price": "float — current underlying price",
            "strike_price": "float — option strike price",
            "risk_free_rate": "float — annualized risk-free rate",
            "volatility": "float — annualized volatility",
            "time_to_expiry": "float — years to expiry",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "black_scholes_price": "float — analytical BS call price",
            "classical_mc": "(if configured) dict with price, std_error, etc.",
            "quantum_ae": "(if configured) dict with price, circuit_depth, etc.",
            "comparison": "(if both methods) dict with error comparison",
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        S = inputs["spot_price"]
        K = inputs["strike_price"]
        r = inputs["risk_free_rate"]
        sigma = inputs["volatility"]
        T = inputs["time_to_expiry"]
        cfg = self._config
        methods = cfg["methods"]

        result: Dict[str, Any] = {}

        # Always compute BS analytical price
        bs_price = black_scholes_call(S, K, r, sigma, T)
        result["black_scholes_price"] = bs_price

        if "classical_mc" in methods:
            result["classical_mc"] = classical_mc_price(
                S0=S, K=K, r=r, sigma=sigma, T=T,
                n_paths=cfg.get("n_classical_paths", 100_000),
                seed=cfg.get("seed", 42),
                option_type="call",
            )

        if "quantum_ae" in methods:
            result["quantum_ae"] = quantum_ae_price(
                S0=S, K=K, r=r, sigma=sigma, T=T,
                n_qubits=cfg.get("n_qubits_price", 4),
                n_estimation_qubits=cfg.get("n_estimation_qubits", 4),
            )

        if "classical_mc" in result and "quantum_ae" in result:
            result["comparison"] = {
                "bs_price": bs_price,
                "mc_price": result["classical_mc"]["price"],
                "qae_price": result["quantum_ae"]["price"],
                "mc_error": abs(result["classical_mc"]["price"] - bs_price),
                "qae_error": abs(result["quantum_ae"]["price"] - bs_price),
            }

        self._metrics = result
        return result

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        if "spot_price" not in inputs:
            raise ValueError("Missing required input: 'spot_price'")
        return True

    def log_metrics(self) -> None:
        if not self._metrics:
            logger.info("QuantumMonteCarloAgent: no metrics (run() not called)")
            return
        logger.info("QuantumMonteCarloAgent metrics: %s", self._metrics)
