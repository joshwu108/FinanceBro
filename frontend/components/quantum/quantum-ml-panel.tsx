"use client"

import { useState } from "react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from "recharts"
import { useAppStore } from "@/lib/store"

const METHOD_COLORS: Record<string, string> = {
  rolling_mean: "#8B949E",
  linear: "#4CC9F0",
  vqr: "#00FF9C",
}

const METHOD_LABELS: Record<string, string> = {
  rolling_mean: "Rolling Mean",
  linear: "Linear Regression",
  vqr: "Variational Quantum (VQR)",
}

export function QuantumMLPanel() {
  const quantumRunning = useAppStore((s) => s.quantumRunning)
  const result = useAppStore((s) => s.quantumMLResult)
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const runQuantumML = useAppStore((s) => s.runQuantumML)

  const [ticker, setTicker] = useState(activeSymbol)
  const [nLags, setNLags] = useState(5)
  const [nQubits, setNQubits] = useState(4)
  const [nLayers, setNLayers] = useState(2)
  const [maxiter, setMaxiter] = useState(100)

  function handleRun() {
    runQuantumML(ticker.trim().toUpperCase(), {
      n_lags: nLags,
      n_qubits: nQubits,
      n_layers: nLayers,
      maxiter,
    })
  }

  // MSE comparison data
  const mseData = result?.comparison?.all_mse
    ? Object.entries(result.comparison.all_mse).map(([method, mse]) => ({
        method: METHOD_LABELS[method] ?? method,
        mse,
        fill: METHOD_COLORS[method] ?? "#8B949E",
        isBest: method === result.comparison?.best_method,
      }))
    : []

  // Runtime comparison data
  const runtimeData = ["rolling_mean", "linear", "vqr"]
    .filter((m) => result?.[m as keyof typeof result])
    .map((m) => {
      const r = result![m as keyof typeof result] as any
      return {
        method: METHOD_LABELS[m] ?? m,
        runtime_ms: r.runtime_ms,
        fill: METHOD_COLORS[m] ?? "#8B949E",
      }
    })

  return (
    <div className="h-full flex flex-col overflow-auto">
      {/* Config */}
      <div className="p-3 border-b border-[#1E2A38] flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Ticker</label>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-24 focus:outline-none focus:border-[#4CC9F0]"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Lags</label>
          <input type="number" min={2} max={20} value={nLags} onChange={(e) => setNLags(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-16 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Qubits</label>
          <input type="number" min={2} max={8} value={nQubits} onChange={(e) => setNQubits(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-16 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Layers</label>
          <input type="number" min={1} max={5} value={nLayers} onChange={(e) => setNLayers(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-16 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Max Iter</label>
          <input type="number" min={20} max={500} step={10} value={maxiter} onChange={(e) => setMaxiter(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-20 focus:outline-none focus:border-[#4CC9F0]" />
        </div>
        <button
          onClick={handleRun}
          disabled={quantumRunning}
          className="bg-[#4CC9F0] text-[#0B0F14] rounded-sm px-4 py-1 text-[11px] font-bold uppercase tracking-wider hover:bg-[#3AB8DF] transition-colors disabled:opacity-50"
        >
          {quantumRunning ? "Training..." : "Run ML"}
        </button>
      </div>

      {!result ? (
        <div className="flex-1 flex items-center justify-center text-[#8B949E] text-sm">
          Compare VQR (variational quantum regressor) against classical baselines
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 gap-2 p-2 overflow-auto">
          {/* MSE comparison */}
          <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              MSE Comparison (lower = better)
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={mseData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                <XAxis dataKey="method" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} tickFormatter={(v: number) => v.toExponential(1)} />
                <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => v.toExponential(4)} />
                <Bar dataKey="mse" radius={[3, 3, 0, 0]}>
                  {mseData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} strokeWidth={entry.isBest ? 2 : 0} stroke={entry.isBest ? "#FFD60A" : "none"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {result.comparison && (
              <div className="mt-2 text-center">
                <span className="text-[9px] text-[#FFD60A] font-bold">
                  Best: {METHOD_LABELS[result.comparison.best_method] ?? result.comparison.best_method}
                </span>
              </div>
            )}
          </div>

          {/* Runtime comparison */}
          <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              Runtime Comparison
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={runtimeData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                <XAxis dataKey="method" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => `${v.toFixed(2)} ms`} />
                <Bar dataKey="runtime_ms" name="Runtime (ms)" radius={[3, 3, 0, 0]}>
                  {runtimeData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Method details */}
          <div className="col-span-2 bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              Method Details
            </div>
            <div className="grid grid-cols-3 gap-3">
              {(["rolling_mean", "linear", "vqr"] as const).map((m) => {
                const r = result[m]
                if (!r) return null
                return (
                  <div key={m} className="bg-[#11161C] border border-[#1E2A38] rounded-sm p-2">
                    <div className="text-[9px] font-bold uppercase tracking-wider mb-1" style={{ color: METHOD_COLORS[m] }}>
                      {METHOD_LABELS[m]}
                    </div>
                    <div className="space-y-1 text-[10px]">
                      <div className="flex justify-between"><span className="text-[#8B949E]">MSE</span><span className="text-[#E6EDF3] font-mono">{r.mse.toExponential(4)}</span></div>
                      <div className="flex justify-between"><span className="text-[#8B949E]">Runtime</span><span className="text-[#E6EDF3] font-mono">{r.runtime_ms.toFixed(2)} ms</span></div>
                      <div className="flex justify-between"><span className="text-[#8B949E]">Test Samples</span><span className="text-[#E6EDF3] font-mono">{r.n_test}</span></div>
                      {r.n_params !== undefined && (
                        <div className="flex justify-between"><span className="text-[#8B949E]">Parameters</span><span className="text-[#E6EDF3] font-mono">{r.n_params}</span></div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
