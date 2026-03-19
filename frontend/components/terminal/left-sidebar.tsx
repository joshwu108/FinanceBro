"use client"

import { useState } from "react"
import { Play, Database, FlaskConical, Brain, ChevronDown, CheckCircle2, AlertCircle, XCircle } from "lucide-react"

interface LeftSidebarProps {
  onRunPipeline: () => void
  running: boolean
}

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

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] appearance-none cursor-pointer hover:border-[#4CC9F0] transition-colors focus:outline-none focus:border-[#4CC9F0]"
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
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

const EXPERIMENT_CONFIGS = [
  "exp_001_momentum",
  "exp_002_mean_rev",
  "exp_003_ml_hybrid",
  "exp_004_stat_arb",
]

const MODEL_TYPES = [
  "LightGBM Classifier",
  "XGBoost Regressor",
  "LSTM (Sequential)",
  "Linear Factor Model",
  "Random Forest",
]

const DATA_STATUS = [
  { symbol: "SPY", status: "loaded" as const },
  { symbol: "QQQ", status: "cached" as const },
  { symbol: "AAPL", status: "loaded" as const },
]

const statusIcon = {
  loaded: <CheckCircle2 className="w-3 h-3 text-[#00FF9C]" />,
  cached: <AlertCircle className="w-3 h-3 text-[#FFD60A]" />,
  missing: <XCircle className="w-3 h-3 text-[#FF4D4D]" />,
}

const statusLabel = {
  loaded: "text-[#00FF9C]",
  cached: "text-[#FFD60A]",
  missing: "text-[#FF4D4D]",
}

export function LeftSidebar({ onRunPipeline, running }: LeftSidebarProps) {
  const [dateFrom, setDateFrom] = useState("2022-01-01")
  const [dateTo, setDateTo] = useState("2024-12-31")
  const [expConfig, setExpConfig] = useState(EXPERIMENT_CONFIGS[0])
  const [txCosts, setTxCosts] = useState(true)
  const [slippage, setSlippage] = useState(false)
  const [modelType, setModelType] = useState(MODEL_TYPES[0])
  const [threshold, setThreshold] = useState(0.5)

  return (
    <aside className="w-52 bg-[#11161C] border-r border-[#1E2A38] flex flex-col gap-0 shrink-0 overflow-y-auto">
      {/* A. Data Controls */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={Database} label="Data Controls" />

        <Label>Symbols</Label>
        <div className="flex flex-wrap gap-1 mb-2">
          {DATA_STATUS.map(({ symbol, status }) => (
            <div
              key={symbol}
              className="flex items-center gap-1 bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-1.5 py-0.5"
            >
              {statusIcon[status]}
              <span className={`text-[10px] font-bold ${statusLabel[status]}`}>{symbol}</span>
            </div>
          ))}
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

        <Label>Config</Label>
        <Select value={expConfig} onChange={setExpConfig} options={EXPERIMENT_CONFIGS} />

        <button
          onClick={onRunPipeline}
          disabled={running}
          className="w-full mt-3 flex items-center justify-center gap-2 bg-[#00FF9C] text-[#0B0F14] rounded-sm py-1.5 text-[11px] font-bold uppercase tracking-wider hover:bg-[#00CC7A] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
        <Select value={modelType} onChange={setModelType} options={MODEL_TYPES} />

        <div className="mt-3">
          <div className="flex items-center justify-between mb-1">
            <Label>Threshold</Label>
            <span className="text-[11px] text-[#4CC9F0] font-bold tabular-nums">{threshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-full accent-[#4CC9F0] h-1 cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-[#8B949E] mt-0.5">
            <span>0.0</span>
            <span>Decision boundary</span>
            <span>1.0</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
