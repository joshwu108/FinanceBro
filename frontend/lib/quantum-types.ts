// ── Quantum Portfolio ─────────────────────────────────────────────────────────

export interface QuantumPortfolioResponse {
  n_assets: number
  tickers: string[]
  classical_weights: number[]
  classical_objective: number
  classical_runtime_ms: number
  quantum_weights?: number[]
  quantum_objective?: number
  quantum_runtime_ms?: number
  quantum_metadata?: Record<string, any>
  comparison?: {
    weight_distance: number
    runtime_ratio: number
    objective_classical: number
    objective_quantum: number
  }
  efficient_frontier?: {
    risks: number[]
    returns: number[]
  }
}

export interface FrontierResponse {
  tickers: string[]
  efficient_frontier: {
    risks: number[]
    returns: number[]
  }
  classical_weights: number[]
  classical_objective: number
}

// ── Quantum Option Pricing ───────────────────────────────────────────────────

export interface MCResult {
  price: number
  std_error: number
  ci_lower: number
  ci_upper: number
  runtime_ms: number
  n_paths: number
}

export interface QAEResult {
  price: number
  runtime_ms: number
  n_qubits: number
  circuit_depth: number
}

export interface OptionPricingResponse {
  black_scholes_price: number
  classical_mc?: MCResult
  quantum_ae?: QAEResult
  comparison?: {
    bs_price: number
    mc_price: number
    qae_price: number
    mc_error: number
    qae_error: number
  }
}

export interface ConvergencePoint {
  price: number
  runtime_ms: number
  error: number
  n_paths?: number
  n_qubits?: number
  circuit_depth?: number
}

export interface ConvergenceResponse {
  bs_price: number
  classical: ConvergencePoint[]
  quantum: ConvergencePoint[]
}

// ── Quantum ML ───────────────────────────────────────────────────────────────

export interface MethodResult {
  mse: number
  runtime_ms: number
  n_test: number
  n_params?: number
}

export interface QuantumMLResponse {
  ticker: string
  n_data_points: number
  n_samples: number
  n_lags: number
  rolling_mean?: MethodResult
  linear?: MethodResult
  vqr?: MethodResult
  comparison?: {
    best_method: string
    best_mse: number
    all_mse: Record<string, number>
  }
}

// ── Quantum Backtest ─────────────────────────────────────────────────────────

export interface BacktestModeResult {
  portfolio_values: number[]
  metrics: {
    total_return: number
    annualized_return: number
    sharpe_ratio: number
    max_drawdown: number
    volatility: number
  }
  total_transaction_costs: number
  optimizer_name: string
  n_trades: number
  n_rebalances: number
  weight_history: number[][]
}

export interface QuantumBacktestResponse {
  tickers: string[]
  n_days: number
  classical?: BacktestModeResult
  qaoa_ideal?: BacktestModeResult
  qaoa_noisy?: BacktestModeResult
  qaoa_mitigated?: BacktestModeResult
  summary?: Record<string, {
    total_return: number
    sharpe_ratio: number
    max_drawdown: number
    transaction_costs: number
  }>
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

export interface PortfolioScalingPoint {
  n_assets: number
  n_qubits: number
  data_source: string
  classical_runtime_ms: number
  classical_objective: number
  qaoa_runtime_ms: number
  qaoa_objective: number
  brute_force_objective: number | null
  approximation_ratio: number | null
  runtime_ratio: number
}

export interface ScalingExponents {
  polynomial_degree?: number
  polynomial_exponent?: number
  exponential_base?: number
}

export interface PortfolioScalingResponse {
  experiment: string
  timestamp: string
  config: Record<string, any>
  results: PortfolioScalingPoint[]
  scaling: {
    qaoa: ScalingExponents
    classical: ScalingExponents
  }
}

export interface MaxCutScalingPoint {
  n_nodes: number
  brute_force_objective: number | null
  qaoa_objective: number
  qaoa_runtime_ms: number
  sa_objective: number
  sa_runtime_ms: number
  greedy_objective: number
  greedy_runtime_ms: number
  qaoa_approx_ratio?: number
  sa_approx_ratio?: number
  greedy_approx_ratio?: number
}

export interface MaxCutScalingResponse {
  experiment: string
  timestamp: string
  config: Record<string, any>
  results: MaxCutScalingPoint[]
  scaling: {
    qaoa: ScalingExponents
    sa: ScalingExponents
    greedy: ScalingExponents
  }
}

export interface CppSpeedupPoint {
  n_qubits: number
  dim: number
  python_ms: number
  cpp_ms: number | null
  speedup: number | null
}

export interface CppSpeedupResponse {
  experiment: string
  timestamp: string
  has_cpp: boolean
  n_iterations: number
  results: CppSpeedupPoint[]
}
