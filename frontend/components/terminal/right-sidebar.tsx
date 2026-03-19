"use client"

import { AlertTriangle, BarChart3, ShieldAlert, Tag } from "lucide-react"

function SectionHeader({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-1.5 mb-2">
      <Icon className="w-3 h-3 text-[#4CC9F0]" />
      <span className="text-[9px] font-bold uppercase tracking-widest text-[#4CC9F0]">{label}</span>
    </div>
  )
}

interface MetricRowProps {
  label: string
  value: string
  color?: "green" | "red" | "yellow" | "blue" | "white"
  bar?: number // 0-1
}

function MetricRow({ label, value, color = "white", bar }: MetricRowProps) {
  const colorClass = {
    green: "text-[#00FF9C]",
    red: "text-[#FF4D4D]",
    yellow: "text-[#FFD60A]",
    blue: "text-[#4CC9F0]",
    white: "text-[#E6EDF3]",
  }[color]

  const barColor = {
    green: "bg-[#00FF9C]",
    red: "bg-[#FF4D4D]",
    yellow: "bg-[#FFD60A]",
    blue: "bg-[#4CC9F0]",
    white: "bg-[#E6EDF3]",
  }[color]

  return (
    <div className="flex flex-col gap-0.5 py-1.5 border-b border-[#1E2A38] last:border-0">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#8B949E]">{label}</span>
        <span className={`text-[11px] font-bold tabular-nums ${colorClass}`}>{value}</span>
      </div>
      {bar !== undefined && (
        <div className="w-full h-0.5 bg-[#1E2A38] rounded-full">
          <div
            className={`h-full rounded-full ${barColor}`}
            style={{ width: `${Math.min(100, Math.max(0, bar * 100))}%` }}
          />
        </div>
      )}
    </div>
  )
}

interface RightSidebarProps {
  mode: "research" | "live"
}

export function RightSidebar({ mode }: RightSidebarProps) {
  return (
    <aside className="w-52 bg-[#11161C] border-l border-[#1E2A38] flex flex-col shrink-0 overflow-y-auto">
      {/* A. Performance Metrics */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={BarChart3} label="Performance" />
        <MetricRow label="Sharpe Ratio" value="1.84" color="green" bar={0.72} />
        <MetricRow label="Sortino Ratio" value="2.31" color="green" bar={0.78} />
        <MetricRow label="Max Drawdown" value="-12.4%" color="red" bar={0.42} />
        <MetricRow label="Total Return" value="+34.7%" color="green" bar={0.65} />
        <MetricRow label="Win Rate" value="58.2%" color="blue" bar={0.58} />
        <MetricRow label="Profit Factor" value="1.63" color="green" bar={0.55} />
      </div>

      {/* B. Risk Panel */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={ShieldAlert} label="Risk Panel" />
        <MetricRow label="Volatility (Ann.)" value="18.3%" color="yellow" bar={0.35} />
        <MetricRow label="Beta (vs SPY)" value="0.72" color="blue" bar={0.72} />
        <MetricRow label="Exposure (Gross)" value="87.5%" color="white" bar={0.875} />
        <MetricRow label="Turnover (Daily)" value="4.2%" color="white" bar={0.12} />
        <MetricRow label="VaR (95%, 1D)" value="-1.8%" color="yellow" bar={0.25} />

        {/* Warning flags */}
        <div className="mt-2 space-y-1">
          <div className="flex items-center gap-1.5 bg-[#FFD60A]/10 border border-[#FFD60A]/30 rounded-sm px-2 py-1">
            <AlertTriangle className="w-3 h-3 text-[#FFD60A] shrink-0" />
            <span className="text-[9px] text-[#FFD60A]">Elevated drawdown risk</span>
          </div>
          <div className="flex items-center gap-1.5 bg-[#FF4D4D]/10 border border-[#FF4D4D]/30 rounded-sm px-2 py-1">
            <AlertTriangle className="w-3 h-3 text-[#FF4D4D] shrink-0" />
            <span className="text-[9px] text-[#FF4D4D]">High concentration — NVDA</span>
          </div>
        </div>
      </div>

      {/* C. Experiment Metadata */}
      <div className="p-3">
        <SectionHeader icon={Tag} label="Experiment" />
        <div className="space-y-1.5">
          {[
            { label: "Exp ID", value: "EXP-20240318-004" },
            { label: "Timestamp", value: "2024-03-18 09:32" },
            { label: "Dataset Ver", value: "v2.4.1" },
            { label: "Model Ver", value: "lgbm-v3.2" },
            { label: "Lookback", value: "60 bars" },
            { label: "Features", value: "47 (PCA 12)" },
          ].map(({ label, value }) => (
            <div key={label} className="flex flex-col">
              <span className="text-[9px] text-[#8B949E] uppercase tracking-wider">{label}</span>
              <span className="text-[10px] text-[#E6EDF3] font-mono">{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Live Mode: Signal Box */}
      {mode === "live" && (
        <div className="p-3 border-t border-[#1E2A38] bg-[#0B0F14]">
          <div className="text-[9px] font-bold uppercase tracking-widest text-[#00FF9C] mb-2">Live Signal</div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex flex-col">
              <span className="text-[9px] text-[#8B949E]">Prediction</span>
              <span className="text-lg font-bold text-[#00FF9C] tabular-nums">0.742</span>
            </div>
            <div className="bg-[#00FF9C] text-[#0B0F14] px-3 py-1.5 rounded-sm">
              <span className="text-sm font-bold tracking-wider">BUY</span>
            </div>
          </div>
          <div className="flex items-center justify-between text-[9px] text-[#8B949E]">
            <span>Updated: 09:32:14</span>
            <span className="text-[#4CC9F0]">Latency: 3.2ms</span>
          </div>
        </div>
      )}
    </aside>
  )
}
