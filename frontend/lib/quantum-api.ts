import type {
  QuantumPortfolioResponse,
  FrontierResponse,
  OptionPricingResponse,
  ConvergenceResponse,
  QuantumMLResponse,
  QuantumBacktestResponse,
  PortfolioScalingResponse,
  MaxCutScalingResponse,
  CppSpeedupResponse,
} from "./quantum-types"

const API_BASE = "http://localhost:8000"

async function post<T>(path: string, body: Record<string, any>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let msg = "Unknown error"
    try {
      const json = await res.json()
      msg = json?.detail ?? JSON.stringify(json)
    } catch {
      msg = await res.text().catch(() => "Unknown error")
    }
    throw new Error(`${path} failed: ${res.status} ${msg}`)
  }
  return res.json()
}

// ── Quantum Portfolio ─────────────────────────────────────────────────────────

export function runQuantumPortfolio(params: {
  tickers: string[]
  max_weight?: number
  qaoa_layers?: number
  frontier_points?: number
}): Promise<QuantumPortfolioResponse> {
  return post("/api/quantum/portfolio", params)
}

export function runEfficientFrontier(params: {
  tickers: string[]
  max_weight?: number
  n_points?: number
}): Promise<FrontierResponse> {
  return post("/api/quantum/portfolio/frontier", params)
}

// ── Option Pricing ───────────────────────────────────────────────────────────

export function priceOption(params: {
  spot_price: number
  strike_price: number
  risk_free_rate?: number
  volatility: number
  time_to_expiry: number
  n_classical_paths?: number
  n_qubits_price?: number
  n_estimation_qubits?: number
}): Promise<OptionPricingResponse> {
  return post("/api/quantum/options/price", params)
}

export function runConvergenceAnalysis(params?: {
  spot_price?: number
  strike_price?: number
  volatility?: number
  time_to_expiry?: number
}): Promise<ConvergenceResponse> {
  return post("/api/quantum/options/convergence", params ?? {})
}

// ── Quantum ML ───────────────────────────────────────────────────────────────

export function runQuantumML(params: {
  ticker: string
  n_lags?: number
  n_qubits?: number
  n_layers?: number
  maxiter?: number
  methods?: string[]
}): Promise<QuantumMLResponse> {
  return post("/api/quantum/ml/predict", params)
}

// ── Quantum Backtest ─────────────────────────────────────────────────────────

export function runQuantumBacktest(params: {
  tickers: string[]
  initial_capital?: number
  transaction_cost_bps?: number
  slippage_bps?: number
  rebalance_frequency?: number
  max_weight?: number
  qaoa_layers?: number
  single_qubit_error?: number
  two_qubit_error?: number
  readout_error?: number
}): Promise<QuantumBacktestResponse> {
  return post("/api/quantum/backtest", params)
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

export function runPortfolioScaling(params?: {
  asset_counts?: number[]
  n_bits?: number
  qaoa_layers?: number
}): Promise<PortfolioScalingResponse> {
  return post("/api/quantum/benchmarks/portfolio-scaling", params ?? {})
}

export function runMaxCutScaling(params?: {
  node_counts?: number[]
  qaoa_layers?: number
}): Promise<MaxCutScalingResponse> {
  return post("/api/quantum/benchmarks/maxcut-scaling", params ?? {})
}

export function runCppSpeedup(params?: {
  qubit_counts?: number[]
  n_iterations?: number
}): Promise<CppSpeedupResponse> {
  return post("/api/quantum/benchmarks/cpp-speedup", params ?? {})
}
