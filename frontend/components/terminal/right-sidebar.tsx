"use client"

import { useEffect, useState } from "react"
import { AlertTriangle, BarChart3, ShieldAlert, Tag, Atom } from "lucide-react"
import { useAppStore } from "@/lib/store"

// ── Shared UI primitives ─────────────────────────────────────────────────────

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

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtPct(v: number | undefined | null): string {
  if (v == null) return "--"
  return `${(v * 100).toFixed(1)}%`
}

function fmtNum(v: number | undefined | null, decimals = 2): string {
  if (v == null) return "--"
  return v.toFixed(decimals)
}

function fmtAgo(msAgo: number): string {
  const s = Math.max(0, Math.floor(msAgo / 1000))
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  return `${h}h ago`
}

function sharpeColor(v: number | undefined | null): "green" | "red" | "yellow" | "white" {
  if (v == null) return "white"
  if (v >= 1.0) return "green"
  if (v >= 0) return "yellow"
  return "red"
}

function overfittingBadge(score: number): { bg: string; text: string; border: string; label: string } {
  if (score < 0.3) return { bg: "bg-[#00FF9C]/10", text: "text-[#00FF9C]", border: "border-[#00FF9C]/30", label: "Low" }
  if (score <= 0.6) return { bg: "bg-[#FFD60A]/10", text: "text-[#FFD60A]", border: "border-[#FFD60A]/30", label: "Medium" }
  return { bg: "bg-[#FF4D4D]/10", text: "text-[#FF4D4D]", border: "border-[#FF4D4D]/30", label: "High" }
}

// ── Main Component ───────────────────────────────────────────────────────────

export function RightSidebar() {
  const nowMs = useNowMs()

  const mode = useAppStore((s) => s.mode)
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const activeSymbolResult = useAppStore((s) => s.activeSymbolResult)
  const pipelineResult = useAppStore((s) => s.pipelineResult)

  const symbolResult = activeSymbolResult()
  const perf = symbolResult?.backtest?.performance_summary ?? null
  const risk = symbolResult?.risk?.risk_metrics ?? null
  const overfitting = symbolResult?.overfitting ?? null
  const stats = pipelineResult?.stats ?? null
  const tick = useAppStore((s) => s.liveTickBySymbol[activeSymbol])
  const quote = useAppStore((s) => s.liveQuoteBySymbol[activeSymbol])
  const lastUpdatedAt = useAppStore((s) => s.wsLastUpdatedAtBySymbol[activeSymbol])

  return (
    <aside className="h-full bg-[#11161C] border-l border-[#1E2A38] flex flex-col overflow-y-auto">
      {/* A. Performance Metrics */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={BarChart3} label="Performance" />
        <MetricRow
          label="Sharpe Ratio"
          value={fmtNum(perf?.sharpe)}
          color={sharpeColor(perf?.sharpe)}
          bar={perf?.sharpe != null ? Math.min(1, Math.abs(perf.sharpe) / 3) : undefined}
        />
        <MetricRow
          label="Sortino Ratio"
          value={fmtNum(perf?.sortino)}
          color={sharpeColor(perf?.sortino)}
          bar={perf?.sortino != null ? Math.min(1, Math.abs(perf.sortino) / 3) : undefined}
        />
        <MetricRow
          label="Max Drawdown"
          value={fmtPct(perf?.max_drawdown)}
          color="red"
          bar={perf?.max_drawdown != null ? Math.abs(perf.max_drawdown) : undefined}
        />
        <MetricRow
          label="Total Return"
          value={fmtPct(perf?.total_return)}
          color={perf?.total_return != null && perf.total_return >= 0 ? "green" : "red"}
          bar={perf?.total_return != null ? Math.min(1, Math.abs(perf.total_return)) : undefined}
        />
        <MetricRow
          label="Win Rate"
          value={fmtPct(perf?.win_rate)}
          color="blue"
          bar={perf?.win_rate != null ? perf.win_rate : undefined}
        />
        <MetricRow
          label="Total Trades"
          value={perf?.total_trades != null ? String(perf.total_trades) : "--"}
          color="white"
        />
      </div>

      {/* B. Risk Panel */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={ShieldAlert} label="Risk Panel" />
        <MetricRow
          label="VaR (95%, 1D)"
          value={fmtPct(risk?.var_95)}
          color="yellow"
          bar={risk?.var_95 != null ? Math.abs(risk.var_95) : undefined}
        />
        <MetricRow
          label="VaR (99%, 1D)"
          value={fmtPct(risk?.var_99)}
          color="yellow"
          bar={risk?.var_99 != null ? Math.abs(risk.var_99) : undefined}
        />
        <MetricRow
          label="CVaR (95%)"
          value={fmtPct(risk?.cvar_95)}
          color="red"
          bar={risk?.cvar_95 != null ? Math.abs(risk.cvar_95) : undefined}
        />
        <MetricRow
          label="Max Exposure"
          value={fmtPct(risk?.max_position_exposure)}
          color="white"
          bar={risk?.max_position_exposure != null ? risk.max_position_exposure : undefined}
        />
        <MetricRow
          label="VaR Breaches"
          value={risk?.var_breaches != null ? String(risk.var_breaches) : "--"}
          color={risk?.var_breaches != null && risk.var_breaches > 0 ? "red" : "white"}
        />

        {/* Overfitting warning */}
        {overfitting != null && overfitting.overfitting_score > 0.5 && (
          <div className="mt-2 space-y-1">
            <div className="flex items-center gap-1.5 bg-[#FFD60A]/10 border border-[#FFD60A]/30 rounded-sm px-2 py-1">
              <AlertTriangle className="w-3 h-3 text-[#FFD60A] shrink-0" />
              <span className="text-[9px] text-[#FFD60A]">Elevated overfitting risk</span>
            </div>
          </div>
        )}
      </div>

      {/* C. Overfitting & Significance */}
      <div className="p-3 border-b border-[#1E2A38]">
        <SectionHeader icon={Tag} label="Validation" />
        <div className="space-y-2">
          {/* Overfitting score */}
          <div className="flex flex-col gap-1">
            <span className="text-[9px] text-[#8B949E] uppercase tracking-wider">Overfitting Score</span>
            {overfitting != null ? (
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono font-bold text-[#E6EDF3]">{fmtNum(overfitting.overfitting_score)}</span>
                {(() => {
                  const badge = overfittingBadge(overfitting.overfitting_score)
                  return (
                    <span className={`${badge.bg} ${badge.text} border ${badge.border} text-[9px] px-1.5 py-0.5 rounded-sm`}>
                      {badge.label}
                    </span>
                  )
                })()}
              </div>
            ) : (
              <span className="text-[11px] text-[#8B949E] font-mono">--</span>
            )}
          </div>

          {/* Statistical significance */}
          <div className="flex flex-col gap-1">
            <span className="text-[9px] text-[#8B949E] uppercase tracking-wider">Statistical Significance</span>
            {stats?.hypothesis_test != null ? (
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono font-bold text-[#E6EDF3]">
                  p={fmtNum(stats.hypothesis_test.p_value, 4)}
                </span>
                <span
                  className={`text-[9px] px-1.5 py-0.5 rounded-sm border ${
                    stats.hypothesis_test.is_significant
                      ? "bg-[#00FF9C]/10 text-[#00FF9C] border-[#00FF9C]/30"
                      : "bg-[#FF4D4D]/10 text-[#FF4D4D] border-[#FF4D4D]/30"
                  }`}
                >
                  {stats.hypothesis_test.is_significant ? "Significant" : "Not Significant"}
                </span>
              </div>
            ) : (
              <span className="text-[11px] text-[#8B949E] font-mono">--</span>
            )}
          </div>

          {/* Walk-forward summary */}
          {symbolResult?.walk_forward?.aggregated_metrics != null && (
            <div className="flex flex-col gap-1">
              <span className="text-[9px] text-[#8B949E] uppercase tracking-wider">Walk-Forward Sharpe</span>
              <span className="text-[11px] font-mono font-bold text-[#E6EDF3]">
                {fmtNum(symbolResult.walk_forward.aggregated_metrics.mean_sharpe)} +/- {fmtNum(symbolResult.walk_forward.aggregated_metrics.std_sharpe)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Quantum Metrics (visible when quantum results exist) */}
      <QuantumMetrics />

      {/* Live Mode: Signal Box */}
      {mode === "live" && (
        <div className="p-3 border-t border-[#1E2A38] bg-[#0B0F14]">
          <div className="text-[9px] font-bold uppercase tracking-widest text-[#4CC9F0] mb-2">
            Live Market
          </div>

          <MetricRow
            label="Last Price"
            value={fmtNum(tick?.price ?? quote?.askPrice ?? quote?.bidPrice, 2)}
            color="green"
          />
          <MetricRow
            label="Bid / Ask Spread"
            value={
              quote ? fmtNum(quote.askPrice - quote.bidPrice, 4) : "--"
            }
            color={quote ? "yellow" : "white"}
          />
          <MetricRow
            label="Trade Size"
            value={tick?.size != null ? fmtNum(tick.size, 2) : "--"}
            color="white"
          />
          <MetricRow
            label="Last Updated"
            value={
              lastUpdatedAt != null ? fmtAgo(nowMs - lastUpdatedAt) : "--"
            }
            color={lastUpdatedAt != null ? "blue" : "white"}
          />
        </div>
      )}
    </aside>
  )
}

function QuantumMetrics() {
  const portfolioResult = useAppStore((s) => s.quantumPortfolioResult)
  const backtestResult = useAppStore((s) => s.quantumBacktestResult)

  if (!portfolioResult && !backtestResult) return null

  return (
    <div className="p-3 border-b border-[#1E2A38]">
      <SectionHeader icon={Atom} label="Quantum" />
      {portfolioResult?.comparison && (
        <>
          <MetricRow
            label="Weight Distance"
            value={fmtNum(portfolioResult.comparison.weight_distance, 4)}
            color="blue"
          />
          <MetricRow
            label="Runtime Ratio"
            value={`${fmtNum(portfolioResult.comparison.runtime_ratio, 1)}x`}
            color="yellow"
          />
        </>
      )}
      {backtestResult?.summary?.classical && backtestResult?.summary?.qaoa_mitigated && (
        <>
          <MetricRow
            label="Classical Sharpe"
            value={fmtNum(backtestResult.summary.classical.sharpe_ratio)}
            color="blue"
          />
          <MetricRow
            label="Mitigated Sharpe"
            value={fmtNum(backtestResult.summary.qaoa_mitigated.sharpe_ratio)}
            color="green"
          />
        </>
      )}
    </div>
  )
}

function useNowMs() {
  const [now, setNow] = useState(() => Date.now())
  const mode = useAppStore((s) => s.mode)

  useEffect(() => {
    if (mode !== "live") return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [mode])

  return now
}
