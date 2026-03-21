"use client"

import { Navbar } from "@/components/site/navbar"
import { Footer } from "@/components/site/footer"
import {
  Shield,
  Target,
  AlertTriangle,
  BookOpen,
  Users,
  Microscope,
} from "lucide-react"

const PRINCIPLES = [
  {
    icon: AlertTriangle,
    title: "No Look-Ahead Bias",
    desc: "Every feature, signal, and target is constructed using only data available at time t. No future data leaks into the pipeline — ever.",
  },
  {
    icon: Target,
    title: "Out-of-Sample Only",
    desc: "All reported performance metrics are out-of-sample. In-sample results are logged but never used for decision-making.",
  },
  {
    icon: Shield,
    title: "Risk Before Returns",
    desc: "Position sizing is derived from Kelly criterion with VaR/CVaR constraints. Risk management gates every trade before execution.",
  },
  {
    icon: Microscope,
    title: "Overfitting Detection",
    desc: "Probability of Backtest Overfitting (PBO), parameter sensitivity analysis, and train/test performance gap tracking at every run.",
  },
  {
    icon: BookOpen,
    title: "Full Experiment Logging",
    desc: "Every pipeline execution is logged to /experiments/ with config, metrics, and timestamps. No silent experiments. No unlogged results.",
  },
  {
    icon: Users,
    title: "Critic Mindset",
    desc: "The system acts as both builder and reviewer. If a Sharpe ratio looks too good, it flags it. Unrealistic assumptions are surfaced, not hidden.",
  },
]

const STACK = [
  { category: "Backend", items: ["FastAPI", "Python 3.9+", "PostgreSQL", "SQLAlchemy", "Redis", "Celery"] },
  { category: "ML / Stats", items: ["scikit-learn", "XGBoost", "PyTorch (LSTM)", "FinBERT", "NumPy", "Pandas"] },
  { category: "Frontend", items: ["Next.js 16", "React 19", "TypeScript", "Tailwind CSS", "Zustand", "lightweight-charts"] },
  { category: "Data", items: ["yfinance", "Alpha Vantage", "Alpaca (live)", "Socket.io", "WebSockets"] },
]

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-[#0B0F14] text-[#E6EDF3] flex flex-col">
      <Navbar />

      {/* Header */}
      <section className="px-6 py-20 border-b border-[#1E2A38]">
        <div className="max-w-4xl mx-auto">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
            About
          </span>
          <h1 className="mt-3 text-3xl md:text-4xl font-bold">
            Quantitative research,{" "}
            <span className="text-[#00FF9C]">done right</span>
          </h1>
          <p className="mt-4 text-sm text-[#8B949E] max-w-2xl leading-relaxed">
            FinanceBro is a modular, agent-driven quantitative research and trading system.
            It was designed from the ground up to enforce statistical rigor at every stage —
            from data ingestion through portfolio construction. The platform prioritizes
            validation methodology over prediction accuracy, and risk management over returns.
          </p>
        </div>
      </section>

      {/* Principles */}
      <section className="px-6 py-20 border-b border-[#1E2A38]">
        <div className="max-w-6xl mx-auto">
          <div className="mb-12">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
              Core Principles
            </span>
            <h2 className="mt-3 text-2xl font-bold">What we enforce</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {PRINCIPLES.map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="p-6 bg-[#11161C] border border-[#1E2A38] rounded-sm"
              >
                <Icon className="w-5 h-5 text-[#4CC9F0] mb-4" />
                <h3 className="text-sm font-semibold text-[#E6EDF3] mb-2">{title}</h3>
                <p className="text-xs text-[#8B949E] leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="mb-12">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#4CC9F0]">
              Technology
            </span>
            <h2 className="mt-3 text-2xl font-bold">Stack</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {STACK.map(({ category, items }) => (
              <div key={category}>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#E6EDF3] mb-3">
                  {category}
                </h3>
                <div className="flex flex-wrap gap-2">
                  {items.map((item) => (
                    <span
                      key={item}
                      className="px-3 py-1 text-[11px] text-[#8B949E] bg-[#11161C] border border-[#1E2A38] rounded-sm"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}
