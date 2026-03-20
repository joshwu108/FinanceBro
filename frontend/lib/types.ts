// Pipeline config sent to backend
export interface PipelineConfig {
  symbols: string[]
  start_date: string
  end_date: string
  model_type: string
  transaction_costs_bps: number
  slippage_bps: number
  max_position_size: number
  benchmark: string
}

// OHLCV data point
export interface OHLCVBar {
  date: string // ISO date
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// Trade from backtest
export interface Trade {
  date: string
  action: string // "buy" | "sell" | "liquidate"
  price: number
  shares: number
  cost: number
  slippage: number
  portfolio_value: number
}

// Performance summary from BacktestAgent
export interface PerformanceSummary {
  sharpe: number
  sortino: number
  calmar: number
  max_drawdown: number
  total_return: number
  cumulative_return: number
  win_rate: number
  total_trades: number
  turnover: number
  benchmark_comparison?: {
    benchmark_sharpe: number
    benchmark_return: number
    excess_return: number
    information_ratio: number
  }
}

// Per-symbol model results
export interface ModelResults {
  model_type: string
  train_metrics: Record<string, number>
  test_metrics: Record<string, number>
  train_test_gap: Record<string, number>
  feature_importances: Record<string, number>
  split_info: {
    train_size: number
    test_size: number
    train_start: string
    train_end: string
    test_start: string
    test_end: string
  }
}

// Walk-forward results
export interface WalkForwardResults {
  n_folds: number
  aggregated_metrics: {
    mean_sharpe: number
    std_sharpe: number
    mean_max_drawdown: number
    mean_total_return: number
    mean_win_rate: number
  }
  fold_results: FoldResult[]
}

export interface FoldResult {
  fold_index: number
  split_info: {
    train_size: number
    test_size: number
    train_start: string
    train_end: string
    test_start: string
    test_end: string
  }
  model_metrics: {
    train: Record<string, number>
    test: Record<string, number>
  }
  backtest_metrics: PerformanceSummary
  model_type: string
}

// Overfitting analysis
export interface OverfittingResults {
  overfitting_score: number
  warnings: string[]
  failure_modes: string[]
  recommendations: string[]
  diagnostics: Record<string, number>
}

// Risk metrics
export interface RiskResults {
  risk_metrics: {
    var_95: number
    var_99: number
    cvar_95: number
    max_position_exposure: number
    var_breaches: number
  }
  mean_position_size: number
}

// Equity curve data point
export interface EquityCurvePoint {
  date: string
  value: number
}

// Per-symbol pipeline output
export interface SymbolResult {
  feature_count: number
  feature_metadata: Record<string, any>
  model: ModelResults
  walk_forward: WalkForwardResults
  backtest: {
    performance_summary: PerformanceSummary
    trade_count: number
    equity_start: number
    equity_end: number
    equity_curve: EquityCurvePoint[]
    trade_log: Trade[]
    ohlcv: OHLCVBar[]
  }
  overfitting: OverfittingResults
  risk: RiskResults
}

// Portfolio results (multi-symbol only)
export interface PortfolioResults {
  portfolio_metrics: {
    annualized_return: number
    annualized_volatility: number
    sharpe_ratio: number
    max_drawdown: number
    diversification_ratio: number
  }
  n_assets: number
  equity_curve?: EquityCurvePoint[]
  weights?: Record<string, number[]>
}

// Stats results
export interface StatsResults {
  metrics: {
    sharpe: number
    annualized_volatility: number
    max_drawdown: number
    annualized_return: number
  }
  bootstrap: {
    sharpe_ci_lower: number
    sharpe_ci_upper: number
    bootstrap_std: number
  }
  hypothesis_test: {
    p_value: number
    is_significant: boolean
  }
  multiple_testing: {
    bonferroni_threshold: number
    bh_threshold: number
    adjusted_p: number
  }
  benchmark?: Record<string, number> | null
}

// Data quality report
export interface DataQualityReport {
  survivorship_bias_warnings: string[]
  symbols_requested: number
  symbols_received: number
  per_symbol: Record<
    string,
    {
      rows: number
      date_range: string
      missing_pct: number
      anomalies: number
    }
  >
}

// Full pipeline response
export interface PipelineResponse {
  run_id: string
  timestamp: string
  config: PipelineConfig & { n_folds: number }
  data_quality: DataQualityReport
  per_symbol: Record<string, SymbolResult>
  portfolio: PortfolioResults | null
  stats: StatsResults
}

// Experiment summary (from experiments endpoint)
export interface ExperimentSummary {
  experiment_id: string
  date: string
  symbols: string[]
  model: string
  metrics: {
    sharpe: number
    max_drawdown: number
    total_return: number
    win_rate: number
  }
  overfitting_score: number
  statistical_significance: {
    is_significant: boolean
    sharpe_p_value: number
  }
}

// Log entry for bottom panel
export interface LogEntry {
  id: number
  timestamp: string
  agent: string
  message: string
  level: "info" | "warn" | "error" | "success" | "debug"
}

// Predict response
export interface PredictResponse {
  symbol: string
  prediction: number
  signal: string
  timestamp: string
  model_type: string | null
}
