"use client"

import { useState } from "react"
import { Play, Database, FlaskConical, Brain, ChevronDown } from "lucide-react"
import { useAppStore } from "@/lib/store"
import type { PipelineConfig } from "@/lib/types"

// ── Shared UI primitives ─────────────────────────────────────────────────────

function SectionHeader({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-1.5 mb-2">
      <Icon className="w-3 h-3 text-[#4CC9F0]" />
      <span className="text-[9px] font-bold uppercase tracking-widest text-[#4CC9F0]">{label}</span>
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="text-[10px] text-[#8B949E] mb-1 block">{children}</span>
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] appearance-none cursor-pointer hover:border-[#4CC9F0] transition-colors focus:outline-none focus:border-[#4CC9F0]"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-[#8B949E] pointer-events-none" />
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center justify-between cursor-pointer">
      <span className="text-[11px] text-[#8B949E]">{label}</span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-8 h-4 rounded-full transition-colors ${checked ? "bg-[#00FF9C]" : "bg-[#1E2A38]"}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-[#0B0F14] transition-transform ${checked ? "translate-x-4" : ""}`}
        />
      </button>
    </label>
  )
}

// ── Constants ────────────────────────────────────────────────────────────────

const MODEL_OPTIONS = [
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "random_forest", label: "Random Forest" },
]

// ── Main Component ───────────────────────────────────────────────────────────

export function LeftSidebar() {
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const running = useAppStore((s) => s.running)
  const runPipeline = useAppStore((s) => s.runPipeline)

  const [dateFrom, setDateFrom] = useState("2020-01-01")
  const [dateTo, setDateTo] = useState("2023-01-01")
  const [txCosts, setTxCosts] = useState(true)
  const [slippage, setSlippage] = useState(true)
  const [modelType, setModelType] = useState("logistic_regression")
  const [maxPositionSize, setMaxPositionSize] = useState(0.1)

  function handleRunPipeline() {
    const config: PipelineConfig = {
      symbols: [activeSymbol],
      start_date: dateFrom,
      end_date: dateTo,
      model_type: modelType,
      transaction_costs_bps: txCosts ? 5.0 : 0.0,
      slippage_bps: slippage ? 2.0 : 0.0,
      max_position_size: maxPositionSize,
      benchmark: "SPY",
    }
    runPipeline(config)
  }

  return (
    <aside className="h-full bg-[#11161C] border-r border-[#1E2A38] flex flex-col gap-0 overflow-y-auto">
      {/* A. Data Controls */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={Database} label="Data Controls" />

        <Label>Active Symbol</Label>
        <div className="flex items-center gap-1 bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 mb-2">
          <span className="text-[11px] font-bold text-[#4CC9F0]">{activeSymbol}</span>
        </div>

        <Label>Date Range</Label>
        <div className="flex flex-col gap-1">
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="w-full bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] focus:outline-none focus:border-[#4CC9F0] transition-colors"
          />
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="w-full bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] focus:outline-none focus:border-[#4CC9F0] transition-colors"
          />
        </div>
      </div>

      {/* B. Experiment Controls */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={FlaskConical} label="Experiment" />

        <button
          onClick={handleRunPipeline}
          disabled={running}
          className="w-full flex items-center justify-center gap-2 bg-[#00FF9C] text-[#0B0F14] rounded-sm py-1.5 text-[11px] font-bold uppercase tracking-wider hover:bg-[#00CC7A] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {running ? (
            <>
              <span className="w-3 h-3 border-2 border-[#0B0F14] border-t-transparent rounded-full animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-3 h-3" />
              Run Pipeline
            </>
          )}
        </button>

        <div className="flex flex-col gap-2 mt-3">
          <Toggle label="Transaction Costs" checked={txCosts} onChange={setTxCosts} />
          <Toggle label="Slippage" checked={slippage} onChange={setSlippage} />
        </div>
      </div>

      {/* C. Model / Signal Settings */}
      <div className="p-3">
        <SectionHeader icon={Brain} label="Model / Signals" />

        <Label>Model Type</Label>
        <Select value={modelType} onChange={setModelType} options={MODEL_OPTIONS} />

        <div className="mt-3">
          <div className="flex items-center justify-between mb-1">
            <Label>Max Position Size</Label>
            <span className="text-[11px] text-[#4CC9F0] font-bold tabular-nums">{(maxPositionSize * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min={0.01}
            max={1}
            step={0.01}
            value={maxPositionSize}
            onChange={(e) => setMaxPositionSize(Number(e.target.value))}
            className="w-full accent-[#4CC9F0] h-1 cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-[#8B949E] mt-0.5">
            <span>1%</span>
            <span>Position limit</span>
            <span>100%</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
