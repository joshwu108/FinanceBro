"use client"

import { useState } from "react"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts"
import { useAppStore } from "@/lib/store"

const MODE_COLORS: Record<string, string> = {
  classical: "#4CC9F0",
  qaoa_ideal: "#00FF9C",
  qaoa_noisy: "#FF4D4D",
  qaoa_mitigated: "#FFD60A",
}

const MODE_LABELS: Record<string, string> = {
  classical: "Classical (Markowitz)",
  qaoa_ideal: "QAOA (Ideal)",
  qaoa_noisy: "QAOA (Noisy)",
  qaoa_mitigated: "QAOA (Mitigated)",
}

export function QuantumBacktestPanel() {
  const quantumRunning = useAppStore((s) => s.quantumRunning)
  const result = useAppStore((s) => s.quantumBacktestResult)
  const runQuantumBacktest = useAppStore((s) => s.runQuantumBacktest)

  const [tickers, setTickers] = useState("AAPL,MSFT,GOOG")
  const [singleError, setSingleError] = useState(0.01)
  const [twoError, setTwoError] = useState(0.02)
  const [readoutError, setReadoutError] = useState(0.01)
  const [rebalFreq, setRebalFreq] = useState(21)
  const [maxWeight, setMaxWeight] = useState(0.5)

  function handleRun() {
    const tickerList = tickers.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean)
    if (tickerList.length < 2) return
    runQuantumBacktest(tickerList, {
      single_qubit_error: singleError,
      two_qubit_error: twoError,
      readout_error: readoutError,
      rebalance_frequency: rebalFreq,
      max_weight: maxWeight,
    })
  }

  // Build equity curve overlay data
  const modes = ["classical", "qaoa_ideal", "qaoa_noisy", "qaoa_mitigated"] as const
  const maxLen = Math.max(
    ...modes.map((m) => result?.[m]?.portfolio_values?.length ?? 0)
  )

  const equityData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { day: i }
    for (const mode of modes) {
      const vals = result?.[mode]?.portfolio_values
      if (vals && i < vals.length) {
        point[mode] = vals[i]
      }
    }
    return point
  })

  return (
    <div className="h-full flex flex-col overflow-auto">
      {/* Config */}
      <div className="p-3 border-b border-[#1E2A38] flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Tickers</label>
          <input
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-48 focus:outline-none focus:border-[#4CC9F0]"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">1Q Err</label>
          <input type="number" min={0} max={0.2} step={0.005} value={singleError} onChange={(e) => setSingleError(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-20 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">2Q Err</label>
          <input type="number" min={0} max={0.3} step={0.005} value={twoError} onChange={(e) => setTwoError(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-20 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Readout Err</label>
          <input type="number" min={0} max={0.2} step={0.005} value={readoutError} onChange={(e) => setReadoutError(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-20 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Rebal (days)</label>
          <input type="number" min={5} max={63} value={rebalFreq} onChange={(e) => setRebalFreq(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-16 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <button
          onClick={handleRun}
          disabled={quantumRunning}
          className="bg-[#4CC9F0] text-[#0B0F14] rounded-sm px-4 py-1 text-[11px] font-bold uppercase tracking-wider hover:bg-[#3AB8DF] transition-colors disabled:opacity-50"
        >
          {quantumRunning ? "Running..." : "Backtest"}
        </button>
      </div>

      {!result ? (
        <div className="flex-1 flex items-center justify-center text-[#8B949E] text-sm">
          Compare classical vs ideal/noisy/mitigated QAOA backtests
        </div>
      ) : (
        <div className="flex-1 flex flex-col gap-2 p-2 overflow-auto">
          {/* 4-mode equity curve */}
          <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3 flex-1 min-h-[280px]">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              Equity Curves — Noise-Aware Comparison
            </div>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={equityData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                <XAxis dataKey="day" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} label={{ value: "Trading Day", position: "bottom", fill: "#8B949E", fontSize: 9 }} />
                <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => `$${v.toFixed(0)}`} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {modes.map((mode) =>
                  result?.[mode] ? (
                    <Line key={mode} dataKey={mode} stroke={MODE_COLORS[mode]} name={MODE_LABELS[mode]} dot={false} strokeWidth={1.5} />
                  ) : null
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary metrics table */}
          {result.summary && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Performance Summary
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="border-b border-[#1E2A38]">
                      <th className="text-left py-1.5 text-[#8B949E] font-normal">Strategy</th>
                      <th className="text-right py-1.5 text-[#8B949E] font-normal">Return</th>
                      <th className="text-right py-1.5 text-[#8B949E] font-normal">Sharpe</th>
                      <th className="text-right py-1.5 text-[#8B949E] font-normal">Max DD</th>
                      <th className="text-right py-1.5 text-[#8B949E] font-normal">Tx Costs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.summary).map(([mode, metrics]) => (
                      <tr key={mode} className="border-b border-[#1E2A38]/50">
                        <td className="py-1.5 font-bold" style={{ color: MODE_COLORS[mode] ?? "#E6EDF3" }}>
                          {MODE_LABELS[mode] ?? mode}
                        </td>
                        <td className={`text-right py-1.5 font-mono ${metrics.total_return >= 0 ? "text-[#00FF9C]" : "text-[#FF4D4D]"}`}>
                          {(metrics.total_return * 100).toFixed(2)}%
                        </td>
                        <td className="text-right py-1.5 font-mono text-[#E6EDF3]">
                          {metrics.sharpe_ratio.toFixed(3)}
                        </td>
                        <td className="text-right py-1.5 font-mono text-[#FF4D4D]">
                          {(metrics.max_drawdown * 100).toFixed(2)}%
                        </td>
                        <td className="text-right py-1.5 font-mono text-[#8B949E]">
                          ${metrics.transaction_costs.toFixed(0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
