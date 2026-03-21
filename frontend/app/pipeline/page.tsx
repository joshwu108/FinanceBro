"use client"

import Link from "next/link"
import { Navbar } from "@/components/site/navbar"
import { Footer } from "@/components/site/footer"
import {
  ArrowRight,
  Database,
  Layers,
  Brain,
  BarChart3,
  TestTube,
  AlertTriangle,
  Shield,
  PieChart,
  Calculator,
  Terminal,
} from "lucide-react"

const AGENTS = [
  {
    id: 1,
    name: "DataAgent",
    icon: Database,
    color: "#4CC9F0",
    status: "wraps data_collector.py",
    responsibilities: [
      "Load historical market data (price, volume)",
      "Integrate factor and alternative datasets",
      "Ensure clean, aligned time-series",
    ],
    constraints: ["No survivorship bias", "No forward-filled labels", "Validate time alignment"],
    output: "Cleaned DataFrame indexed by timestamp",
  },
  {
    id: 2,
    name: "FeatureAgent",
    icon: Layers,
    color: "#00FF9C",
    status: "wraps feature_engineering.py",
    responsibilities: [
      "Generate predictive features using ONLY past data",
      "Returns, rolling volatility, momentum signals",
      "Factor exposures and technical indicators",
    ],
    constraints: ["All features use rolling windows", "No future leakage", "No shift(-k) allowed"],
    output: "Feature matrix X(t), target y(t+k)",
  },
  {
    id: 3,
    name: "ModelAgent",
    icon: Brain,
    color: "#FFD60A",
    status: "wraps ml_models.py",
    responsibilities: [
      "Train ML models (logistic baseline, XGBoost, LSTM)",
      "Proper early stopping on validation set",
      "Feature importance tracking",
    ],
    constraints: ["Baseline model required", "No debug prints", "Early stopping mandatory"],
    output: "Trained model, metrics, feature importance",
  },
  {
    id: 4,
    name: "WalkForwardAgent",
    icon: BarChart3,
    color: "#4CC9F0",
    status: "build new",
    responsibilities: [
      "Expanding window cross-validation",
      "Strict temporal ordering of folds",
      "Per-fold and aggregate metric reporting",
    ],
    constraints: ["No random splits", "Time-ordered only", "Minimum train window enforced"],
    output: "Fold-level metrics, aggregate performance",
  },
  {
    id: 5,
    name: "BacktestAgent",
    icon: TestTube,
    color: "#00FF9C",
    status: "build new",
    responsibilities: [
      "Event-driven simulation (step through time)",
      "Apply transaction costs (5-10 bps)",
      "Model slippage and market impact",
    ],
    constraints: ["No vectorized shortcuts that leak", "Only use information at each timestep", "Benchmark against SPY buy-and-hold"],
    output: "Equity curve, trade log, performance summary",
  },
  {
    id: 6,
    name: "OverfittingAgent",
    icon: AlertTriangle,
    color: "#FF4D4D",
    status: "build new",
    responsibilities: [
      "Probability of Backtest Overfitting (PBO)",
      "Train/test performance gap analysis",
      "Parameter sensitivity testing",
    ],
    constraints: ["Flag Sharpe > 2.0 for review", "Require multiple metrics", "Report confidence intervals"],
    output: "Overfitting probability, sensitivity report",
  },
  {
    id: 7,
    name: "RiskAgent",
    icon: Shield,
    color: "#FFD60A",
    status: "build new",
    responsibilities: [
      "Kelly criterion position sizing",
      "VaR and CVaR calculation",
      "Maximum drawdown constraints",
    ],
    constraints: ["Position limits enforced", "Correlation-aware sizing", "Regime-dependent adjustments"],
    output: "Position sizes, risk metrics, limit breaches",
  },
  {
    id: 8,
    name: "PortfolioAgent",
    icon: PieChart,
    color: "#4CC9F0",
    status: "wraps portfolio.py (router)",
    responsibilities: [
      "Markowitz mean-variance optimization",
      "Risk parity allocation",
      "Covariance shrinkage (Ledoit-Wolf)",
    ],
    constraints: ["Rebalancing costs included", "Turnover constraints", "No unconstrained optimization"],
    output: "Allocation weights, rebalance schedule",
  },
  {
    id: 9,
    name: "StatsAgent",
    icon: Calculator,
    color: "#00FF9C",
    status: "build new",
    responsibilities: [
      "Bootstrap confidence intervals for Sharpe",
      "Hypothesis testing (strategy vs. benchmark)",
      "Multiple testing correction (Bonferroni/FDR)",
    ],
    constraints: ["Report p-values and confidence intervals", "Correct for data snooping", "Disclose degradation"],
    output: "Statistical significance report, confidence bounds",
  },
]

export default function PipelinePage() {
  return (
    <div className="min-h-screen bg-[#0B0F14] text-[#E6EDF3] flex flex-col">
      <Navbar />

      {/* Header */}
      <section className="px-6 py-20 border-b border-[#1E2A38]">
        <div className="max-w-4xl mx-auto">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
            Pipeline
          </span>
          <h1 className="mt-3 text-3xl md:text-4xl font-bold">
            Agent <span className="text-[#00FF9C]">pipeline</span> architecture
          </h1>
          <p className="mt-4 text-sm text-[#8B949E] max-w-2xl leading-relaxed">
            The FinanceBro pipeline chains 9 specialized agents in strict order.
            Each agent implements the BaseAgent interface, validates its inputs and outputs,
            and logs all metrics. Data flows forward only — no agent can access outputs
            from a downstream stage.
          </p>
        </div>
      </section>

      {/* Flow Diagram */}
      <section className="px-6 py-12 border-b border-[#1E2A38] bg-[#11161C]/30">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-wrap items-center justify-center gap-2">
            {AGENTS.map((agent, i) => (
              <div key={agent.name} className="flex items-center gap-2">
                <span
                  className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded-sm border"
                  style={{
                    color: agent.color,
                    borderColor: `${agent.color}40`,
                    backgroundColor: `${agent.color}08`,
                  }}
                >
                  {agent.name}
                </span>
                {i < AGENTS.length - 1 && (
                  <ArrowRight className="w-3 h-3 text-[#1E2A38] shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Agent Details */}
      <section className="px-6 py-20">
        <div className="max-w-5xl mx-auto flex flex-col gap-8">
          {AGENTS.map((agent) => {
            const Icon = agent.icon
            return (
              <div
                key={agent.name}
                className="p-6 bg-[#11161C] border border-[#1E2A38] rounded-sm hover:border-[#1E2A38] transition-colors"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-8 h-8 rounded-sm flex items-center justify-center"
                      style={{ backgroundColor: `${agent.color}10` }}
                    >
                      <Icon className="w-4 h-4" style={{ color: agent.color }} />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-[#E6EDF3]">
                        <span className="text-[#8B949E] mr-2">
                          {String(agent.id).padStart(2, "0")}
                        </span>
                        {agent.name}
                      </h3>
                    </div>
                  </div>
                  <span className="text-[10px] px-2 py-0.5 bg-[#1A2130] text-[#8B949E] rounded-sm">
                    {agent.status}
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-[#4CC9F0] block mb-2">
                      Responsibilities
                    </span>
                    <ul className="flex flex-col gap-1.5">
                      {agent.responsibilities.map((r) => (
                        <li key={r} className="text-xs text-[#8B949E] flex items-start gap-2">
                          <span className="text-[#4CC9F0] mt-0.5 shrink-0">-</span>
                          {r}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-[#FFD60A] block mb-2">
                      Constraints
                    </span>
                    <ul className="flex flex-col gap-1.5">
                      {agent.constraints.map((c) => (
                        <li key={c} className="text-xs text-[#8B949E] flex items-start gap-2">
                          <span className="text-[#FFD60A] mt-0.5 shrink-0">!</span>
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-[#00FF9C] block mb-2">
                      Output
                    </span>
                    <p className="text-xs text-[#8B949E]">{agent.output}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-16 border-t border-[#1E2A38]">
        <div className="max-w-2xl mx-auto text-center flex flex-col items-center gap-5">
          <p className="text-sm text-[#8B949E]">
            See the pipeline in action — run a backtest in the terminal.
          </p>
          <Link
            href="/terminal"
            className="flex items-center gap-2 px-6 py-2.5 bg-[#00FF9C] text-[#0B0F14] text-xs font-bold uppercase tracking-wider rounded-sm hover:bg-[#00FF9C]/90 transition-colors"
          >
            <Terminal className="w-4 h-4" />
            Launch Terminal
          </Link>
        </div>
      </section>

      <Footer />
    </div>
  )
}
