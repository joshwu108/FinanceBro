"use client"

import { useMemo } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"
import { Briefcase } from "lucide-react"
import type { PortfolioResults, EquityCurvePoint } from "@/lib/types"

// ── Types ────────────────────────────────────────────────────────────────────

interface SymbolBacktestSummary {
  backtest: {
    performance_summary: {
      sharpe: number
      max_drawdown: number
      total_return: number
      win_rate: number
    }
  }
}

interface PortfolioViewProps {
  portfolio: PortfolioResults | null
  symbolResults: Record<string, SymbolBacktestSummary>
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`
}

function formatNum(v: number, decimals = 2): string {
  return v.toFixed(decimals)
}

function valueColor(v: number): string {
  if (v > 0) return "#00FF9C"
  if (v < 0) return "#FF4D4D"
  return "#E6EDF3"
}

// ── Sub-components ───────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string
  value: string
  color: string
}

function MetricCard({ label, value, color }: MetricCardProps) {
  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-4 py-3 flex flex-col gap-1 min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-[#8B949E]">
        {label}
      </span>
      <span
        className="text-[18px] font-bold tabular-nums font-mono"
        style={{ color }}
      >
        {value}
      </span>
    </div>
  )
}

function EquityCurveTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: { value: number; color: string }[]
  label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <div className="text-[#8B949E] mb-1">{label}</div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-[#00FF9C] inline-block" />
        <span className="text-[#8B949E]">Value:</span>
        <span className="text-[#00FF9C]">{payload[0].value.toFixed(2)}</span>
      </div>
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

export function PortfolioView({ portfolio, symbolResults }: PortfolioViewProps) {
  const symbols = useMemo(() => Object.keys(symbolResults), [symbolResults])

  if (!portfolio) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-[#8B949E]">
        <Briefcase className="w-8 h-8 opacity-40" />
        <span className="text-[12px]">
          Run pipeline with multiple symbols to see portfolio analysis
        </span>
      </div>
    )
  }

  const metrics = portfolio.portfolio_metrics
  const equityCurve = portfolio.equity_curve ?? []

  // Compute the starting value for reference line
  const startValue = equityCurve.length > 0 ? equityCurve[0].value : 0

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* ── Metrics Cards Row ─────────────────────────────────────── */}
      <div className="p-3 border-b border-[#1E2A38]">
        <div className="flex items-center gap-1.5 mb-3">
          <Briefcase className="w-3.5 h-3.5 text-[#4CC9F0]" />
          <span className="text-[10px] font-bold uppercase tracking-widest text-[#4CC9F0]">
            Portfolio Metrics
          </span>
        </div>
        <div className="grid grid-cols-5 gap-2">
          <MetricCard
            label="Annualized Return"
            value={formatPct(metrics.annualized_return)}
            color={valueColor(metrics.annualized_return)}
          />
          <MetricCard
            label="Annualized Vol"
            value={formatPct(metrics.annualized_volatility)}
            color="#E6EDF3"
          />
          <MetricCard
            label="Sharpe Ratio"
            value={formatNum(metrics.sharpe_ratio)}
            color={valueColor(metrics.sharpe_ratio)}
          />
          <MetricCard
            label="Max Drawdown"
            value={formatPct(metrics.max_drawdown)}
            color="#FF4D4D"
          />
          <MetricCard
            label="Diversification"
            value={formatNum(metrics.diversification_ratio)}
            color="#4CC9F0"
          />
        </div>
      </div>

      {/* ── Per-Asset Performance Table ────────────────────────────── */}
      <div className="p-3 border-b border-[#1E2A38]">
        <div className="flex items-center gap-1.5 mb-3">
          <span className="text-[10px] font-bold uppercase tracking-widest text-[#4CC9F0]">
            Per-Asset Performance
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-[#1E2A38]">
                {["Symbol", "Sharpe", "Max DD", "Return", "Win Rate"].map(
                  (header) => (
                    <th
                      key={header}
                      className="text-[9px] uppercase tracking-wider text-[#8B949E] font-medium py-2 px-3"
                    >
                      {header}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {symbols.map((symbol) => {
                const perf = symbolResults[symbol].backtest.performance_summary
                return (
                  <tr
                    key={symbol}
                    className="border-b border-[#1E2A38] last:border-0 hover:bg-[#1A2130] transition-colors"
                  >
                    <td className="text-[11px] font-mono font-bold text-[#E6EDF3] py-2 px-3">
                      {symbol}
                    </td>
                    <td
                      className="text-[11px] font-mono tabular-nums py-2 px-3"
                      style={{ color: valueColor(perf.sharpe) }}
                    >
                      {formatNum(perf.sharpe)}
                    </td>
                    <td className="text-[11px] font-mono tabular-nums py-2 px-3 text-[#FF4D4D]">
                      {formatPct(perf.max_drawdown)}
                    </td>
                    <td
                      className="text-[11px] font-mono tabular-nums py-2 px-3"
                      style={{ color: valueColor(perf.total_return) }}
                    >
                      {formatPct(perf.total_return)}
                    </td>
                    <td className="text-[11px] font-mono tabular-nums py-2 px-3 text-[#4CC9F0]">
                      {formatPct(perf.win_rate)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Portfolio Equity Curve ─────────────────────────────────── */}
      {equityCurve.length > 0 && (
        <div className="flex-1 flex flex-col min-h-[200px] p-3">
          <div className="flex items-center gap-1.5 mb-3">
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#4CC9F0]">
              Portfolio Equity Curve
            </span>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={equityCurve}
                margin={{ top: 8, right: 8, bottom: 0, left: 44 }}
              >
                <CartesianGrid
                  strokeDasharray="2 2"
                  stroke="#1E2A38"
                  vertical={false}
                />
                <XAxis
                  dataKey="date"
                  tick={{
                    fill: "#8B949E",
                    fontSize: 9,
                    fontFamily: "monospace",
                  }}
                  tickLine={false}
                  axisLine={{ stroke: "#1E2A38" }}
                  interval={Math.max(
                    0,
                    Math.floor(equityCurve.length / 8) - 1
                  )}
                />
                <YAxis
                  orientation="right"
                  tick={{
                    fill: "#8B949E",
                    fontSize: 9,
                    fontFamily: "monospace",
                  }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => v.toFixed(0)}
                  width={48}
                />
                <Tooltip content={<EquityCurveTooltip />} />
                {startValue > 0 && (
                  <ReferenceLine
                    y={startValue}
                    stroke="#1E2A38"
                    strokeDasharray="4 2"
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#00FF9C"
                  dot={false}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
