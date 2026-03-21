"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import {
  Terminal,
  ArrowRight,
  BarChart3,
  Shield,
  Brain,
  TrendingUp,
  Database,
  GitBranch,
  Activity,
  ChevronRight,
} from "lucide-react"
import { Navbar } from "@/components/site/navbar"
import { Footer } from "@/components/site/footer"

const PIPELINE_STAGES = [
  { name: "DataAgent", desc: "Clean, aligned time-series" },
  { name: "FeatureAgent", desc: "Rolling window features" },
  { name: "ModelAgent", desc: "ML model training" },
  { name: "WalkForwardAgent", desc: "Temporal cross-validation" },
  { name: "BacktestAgent", desc: "Event-driven simulation" },
  { name: "OverfittingAgent", desc: "PBO & sensitivity tests" },
  { name: "RiskAgent", desc: "VaR, CVaR, position limits" },
  { name: "PortfolioAgent", desc: "Allocation & rebalancing" },
  { name: "StatsAgent", desc: "Hypothesis testing" },
]

const FEATURES = [
  {
    icon: BarChart3,
    title: "Walk-Forward Validation",
    desc: "No random splits. Expanding window cross-validation preserves temporal ordering across all experiments.",
  },
  {
    icon: Shield,
    title: "Risk Management",
    desc: "Kelly criterion sizing, VaR/CVaR limits, and maximum drawdown constraints before any capital deployment.",
  },
  {
    icon: Brain,
    title: "ML Pipeline",
    desc: "Logistic regression baselines through XGBoost and LSTM, with proper early stopping and feature importance.",
  },
  {
    icon: TrendingUp,
    title: "Realistic Backtesting",
    desc: "Event-driven simulation with 5-10 bps transaction costs, slippage modeling, and no look-ahead bias.",
  },
  {
    icon: Database,
    title: "Experiment Tracking",
    desc: "Every pipeline run is logged. No silent experiments. Full reproducibility with in-sample and out-of-sample labels.",
  },
  {
    icon: GitBranch,
    title: "Modular Agents",
    desc: "Separation of concerns across 9+ agents. Each validates its own inputs and outputs against strict schemas.",
  },
]

function TypingText({ text, className }: { text: string; className?: string }) {
  const [displayed, setDisplayed] = useState("")
  const [done, setDone] = useState(false)

  useEffect(() => {
    let i = 0
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1))
        i++
      } else {
        setDone(true)
        clearInterval(interval)
      }
    }, 40)
    return () => clearInterval(interval)
  }, [text])

  return (
    <span className={className}>
      {displayed}
      {!done && <span className="animate-pulse text-[#00FF9C]">_</span>}
    </span>
  )
}

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0B0F14] text-[#E6EDF3] flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative flex-1 flex items-center justify-center px-6 py-24 md:py-32">
        {/* Grid background */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(#4CC9F0 1px, transparent 1px), linear-gradient(90deg, #4CC9F0 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        <div className="relative max-w-4xl mx-auto text-center flex flex-col items-center gap-8">
          {/* Terminal prompt */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-[#11161C] border border-[#1E2A38] rounded-sm text-[11px] text-[#8B949E]">
            <Activity className="w-3 h-3 text-[#00FF9C]" />
            <span>quantitative research platform</span>
          </div>

          {/* Heading */}
          <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold leading-tight tracking-tight">
            <span className="text-[#E6EDF3]">Backtest with </span>
            <span className="text-[#00FF9C]">statistical rigor</span>
            <span className="text-[#E6EDF3]">,</span>
            <br />
            <span className="text-[#E6EDF3]">not </span>
            <span className="text-[#FF4D4D]">hindsight bias</span>
          </h1>

          {/* Subheading */}
          <p className="text-sm md:text-base text-[#8B949E] max-w-2xl leading-relaxed">
            FinanceBro is an agent-driven quant platform that enforces walk-forward validation,
            realistic transaction costs, and overfitting detection at every stage of the pipeline.
          </p>

          {/* Terminal snippet */}
          <div className="w-full max-w-lg bg-[#11161C] border border-[#1E2A38] rounded-sm overflow-hidden text-left">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1E2A38]">
              <div className="w-2 h-2 rounded-full bg-[#FF4D4D]" />
              <div className="w-2 h-2 rounded-full bg-[#FFD60A]" />
              <div className="w-2 h-2 rounded-full bg-[#00FF9C]" />
              <span className="ml-2 text-[10px] text-[#8B949E]">financebro-terminal</span>
            </div>
            <div className="p-4 text-xs leading-relaxed">
              <div className="text-[#8B949E]">
                <span className="text-[#4CC9F0]">$</span>{" "}
                <TypingText text="python -m orchestrator.run --symbols AAPL,SPY --validation walk-forward" />
              </div>
              <div className="mt-3 text-[#8B949E] opacity-60">
                <div>[DataAgent] <span className="text-[#00FF9C]">OK</span> — 2,516 bars loaded, no survivorship gaps</div>
                <div>[FeatureAgent] <span className="text-[#00FF9C]">OK</span> — 47 features, 0 look-ahead violations</div>
                <div>[ModelAgent] <span className="text-[#00FF9C]">OK</span> — XGBoost AUC 0.58 (OOS)</div>
                <div>[BacktestAgent] <span className="text-[#FFD60A]">WARN</span> — Sharpe 1.82 — verify with PBO</div>
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="flex items-center gap-4">
            <Link
              href="/terminal"
              className="flex items-center gap-2 px-6 py-2.5 bg-[#00FF9C] text-[#0B0F14] text-xs font-bold uppercase tracking-wider rounded-sm hover:bg-[#00FF9C]/90 transition-colors"
            >
              <Terminal className="w-4 h-4" />
              Launch Terminal
            </Link>
            <Link
              href="/pipeline"
              className="flex items-center gap-2 px-6 py-2.5 border border-[#1E2A38] text-[#8B949E] text-xs font-medium uppercase tracking-wider rounded-sm hover:border-[#4CC9F0] hover:text-[#E6EDF3] transition-colors"
            >
              View Pipeline
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-20 border-t border-[#1E2A38]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
              Capabilities
            </span>
            <h2 className="mt-3 text-2xl md:text-3xl font-bold text-[#E6EDF3]">
              Built for quants, not gamblers
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="p-6 bg-[#11161C] border border-[#1E2A38] rounded-sm hover:border-[#4CC9F0]/30 transition-colors group"
              >
                <Icon className="w-5 h-5 text-[#4CC9F0] mb-4 group-hover:text-[#00FF9C] transition-colors" />
                <h3 className="text-sm font-semibold text-[#E6EDF3] mb-2">{title}</h3>
                <p className="text-xs text-[#8B949E] leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="px-6 py-20 border-t border-[#1E2A38]">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
              Architecture
            </span>
            <h2 className="mt-3 text-2xl md:text-3xl font-bold text-[#E6EDF3]">
              9-stage agent pipeline
            </h2>
            <p className="mt-3 text-xs text-[#8B949E] max-w-lg mx-auto">
              Each agent validates inputs, processes data, and passes structured outputs to the next stage.
              No agent may bypass the BaseAgent interface contract.
            </p>
          </div>

          <div className="flex flex-col gap-1">
            {PIPELINE_STAGES.map((stage, i) => (
              <div key={stage.name} className="flex items-center gap-3">
                <span className="w-5 text-right text-[10px] text-[#8B949E] tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="flex-1 flex items-center justify-between px-4 py-2.5 bg-[#11161C] border border-[#1E2A38] rounded-sm hover:border-[#4CC9F0]/30 transition-colors">
                  <span className="text-xs font-semibold text-[#E6EDF3]">{stage.name}</span>
                  <span className="text-[11px] text-[#8B949E]">{stage.desc}</span>
                </div>
                {i < PIPELINE_STAGES.length - 1 && (
                  <ChevronRight className="w-3 h-3 text-[#1E2A38] shrink-0" />
                )}
                {i === PIPELINE_STAGES.length - 1 && <div className="w-3" />}
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <Link
              href="/pipeline"
              className="inline-flex items-center gap-2 text-xs text-[#4CC9F0] hover:text-[#00FF9C] transition-colors"
            >
              Full pipeline documentation
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-20 border-t border-[#1E2A38]">
        <div className="max-w-2xl mx-auto text-center flex flex-col items-center gap-6">
          <h2 className="text-xl md:text-2xl font-bold text-[#E6EDF3]">
            Ready to run your first backtest?
          </h2>
          <p className="text-xs text-[#8B949E]">
            Launch the terminal, configure your universe, and let the agents validate every assumption.
          </p>
          <Link
            href="/terminal"
            className="flex items-center gap-2 px-8 py-3 bg-[#00FF9C] text-[#0B0F14] text-xs font-bold uppercase tracking-wider rounded-sm hover:bg-[#00FF9C]/90 transition-colors"
          >
            <Terminal className="w-4 h-4" />
            Open Terminal
          </Link>
        </div>
      </section>

      <Footer />
    </div>
  )
}
