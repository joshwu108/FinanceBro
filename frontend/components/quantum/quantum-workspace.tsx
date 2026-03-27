"use client"

import { useState } from "react"
import { QuantumPortfolioPanel } from "./quantum-portfolio-panel"
import { OptionPricer } from "./option-pricer"
import { QuantumMLPanel } from "./quantum-ml-panel"
import { QuantumBacktestPanel } from "./quantum-backtest-panel"
import { BenchmarkDashboard } from "./benchmark-dashboard"

const QUANTUM_TABS = [
  { id: "portfolio" as const, label: "Portfolio" },
  { id: "options" as const, label: "Options" },
  { id: "ml" as const, label: "ML" },
  { id: "backtest" as const, label: "Backtest" },
  { id: "benchmarks" as const, label: "Benchmarks" },
] as const

type QuantumTab = (typeof QUANTUM_TABS)[number]["id"]

export function QuantumWorkspace() {
  const [activeTab, setActiveTab] = useState<QuantumTab>("portfolio")

  return (
    <div className="h-full flex flex-col">
      {/* Sub-tab bar */}
      <div className="flex items-center h-7 border-b border-[#1E2A38] bg-[#0B0F14] shrink-0">
        {QUANTUM_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 h-full text-[9px] uppercase tracking-wider border-r border-[#1E2A38] transition-colors ${
              activeTab === tab.id
                ? "bg-[#11161C] text-[#4CC9F0] font-semibold"
                : "text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#11161C]/50"
            }`}
          >
            {tab.label}
          </button>
        ))}
        <div className="flex-1" />
        <span className="px-3 text-[8px] text-[#8B949E] uppercase tracking-widest">Quantum Lab</span>
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "portfolio" && <QuantumPortfolioPanel />}
        {activeTab === "options" && <OptionPricer />}
        {activeTab === "ml" && <QuantumMLPanel />}
        {activeTab === "backtest" && <QuantumBacktestPanel />}
        {activeTab === "benchmarks" && <BenchmarkDashboard />}
      </div>
    </div>
  )
}
