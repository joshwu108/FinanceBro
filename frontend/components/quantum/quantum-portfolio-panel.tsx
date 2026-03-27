"use client"

import { useState } from "react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ScatterChart, Scatter,
  LineChart, Line, Legend,
} from "recharts"
import { useAppStore } from "@/lib/store"

const TICKER_PRESETS = [
  ["AAPL", "MSFT", "GOOG", "AMZN"],
  ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"],
  ["JPM", "GS", "MS", "BAC", "C"],
]

export function QuantumPortfolioPanel() {
  const quantumRunning = useAppStore((s) => s.quantumRunning)
  const result = useAppStore((s) => s.quantumPortfolioResult)
  const runQuantumPortfolio = useAppStore((s) => s.runQuantumPortfolio)

  const [tickers, setTickers] = useState("AAPL,MSFT,GOOG,AMZN")
  const [maxWeight, setMaxWeight] = useState(0.4)
  const [qaoaLayers, setQaoaLayers] = useState(2)
  const [frontierPoints, setFrontierPoints] = useState(15)

  function handleRun() {
    const tickerList = tickers.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean)
    if (tickerList.length < 2) return
    runQuantumPortfolio(tickerList, {
      max_weight: maxWeight,
      qaoa_layers: qaoaLayers,
      frontier_points: frontierPoints,
    })
  }

  // Build weight comparison data
  const weightData = result?.tickers?.map((ticker, i) => ({
    ticker,
    classical: result.classical_weights?.[i] ?? 0,
    quantum: result.quantum_weights?.[i] ?? 0,
  })) ?? []

  // Build frontier data
  const frontierData = result?.efficient_frontier
    ? result.efficient_frontier.risks.map((risk, i) => ({
        risk: risk * 100,
        return: result.efficient_frontier!.returns[i] * 100,
      }))
    : []

  return (
    <div className="h-full flex flex-col overflow-auto">
      {/* Config */}
      <div className="p-3 border-b border-[#1E2A38] flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Tickers</label>
          <input
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-56 focus:outline-none focus:border-[#4CC9F0]"
            placeholder="AAPL,MSFT,GOOG"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">Max Weight</label>
          <input
            type="number" min={0.1} max={1} step={0.05}
            value={maxWeight}
            onChange={(e) => setMaxWeight(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-20 focus:outline-none focus:border-[#4CC9F0]"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">QAOA Layers</label>
          <input
            type="number" min={1} max={5} step={1}
            value={qaoaLayers}
            onChange={(e) => setQaoaLayers(Number(e.target.value))}
            className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] w-16 focus:outline-none focus:border-[#4CC9F0]"
          />
        </div>
        <div className="flex gap-1">
          {TICKER_PRESETS.map((preset, i) => (
            <button
              key={i}
              onClick={() => setTickers(preset.join(","))}
              className="text-[9px] text-[#8B949E] hover:text-[#4CC9F0] border border-[#1E2A38] rounded-sm px-1.5 py-1 transition-colors"
            >
              {preset.length}A
            </button>
          ))}
        </div>
        <button
          onClick={handleRun}
          disabled={quantumRunning}
          className="bg-[#4CC9F0] text-[#0B0F14] rounded-sm px-4 py-1 text-[11px] font-bold uppercase tracking-wider hover:bg-[#3AB8DF] transition-colors disabled:opacity-50"
        >
          {quantumRunning ? "Running..." : "Optimize"}
        </button>
      </div>

      {!result ? (
        <div className="flex-1 flex items-center justify-center text-[#8B949E] text-sm">
          Configure tickers and run QAOA vs Markowitz comparison
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 gap-2 p-2 overflow-auto">
          {/* Weight comparison bar chart */}
          <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              Portfolio Weights: Classical vs QAOA
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={weightData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                <XAxis dataKey="ticker" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }}
                  formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="classical" fill="#4CC9F0" name="Markowitz" radius={[2, 2, 0, 0]} />
                <Bar dataKey="quantum" fill="#00FF9C" name="QAOA" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Comparison metrics */}
          <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
            <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
              Comparison Metrics
            </div>
            <div className="space-y-2">
              <MetricRow label="Assets" value={String(result.n_assets)} />
              <MetricRow label="Classical Objective" value={result.classical_objective?.toFixed(6) ?? "--"} color="blue" />
              <MetricRow label="Quantum Objective" value={result.quantum_objective?.toFixed(6) ?? "--"} color="green" />
              <MetricRow label="Weight Distance" value={result.comparison?.weight_distance?.toFixed(4) ?? "--"} />
              <MetricRow label="Classical Time" value={`${result.classical_runtime_ms?.toFixed(1) ?? "--"} ms`} color="blue" />
              <MetricRow label="QAOA Time" value={`${result.quantum_runtime_ms?.toFixed(1) ?? "--"} ms`} color="green" />
              <MetricRow label="Runtime Ratio" value={`${result.comparison?.runtime_ratio?.toFixed(1) ?? "--"}x`} color="yellow" />
            </div>
          </div>

          {/* Efficient frontier */}
          {frontierData.length > 0 && (
            <div className="col-span-2 bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Efficient Frontier
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis
                    dataKey="risk" type="number" name="Risk (%)"
                    tick={{ fill: "#8B949E", fontSize: 9 }}
                    axisLine={{ stroke: "#1E2A38" }}
                    label={{ value: "Risk (%)", position: "bottom", fill: "#8B949E", fontSize: 9, dy: 8 }}
                  />
                  <YAxis
                    dataKey="return" type="number" name="Return (%)"
                    tick={{ fill: "#8B949E", fontSize: 9 }}
                    axisLine={{ stroke: "#1E2A38" }}
                    label={{ value: "Return (%)", angle: -90, position: "insideLeft", fill: "#8B949E", fontSize: 9 }}
                  />
                  <Tooltip
                    contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }}
                    formatter={(v: number) => `${v.toFixed(3)}%`}
                  />
                  <Scatter data={frontierData} fill="#4CC9F0" line={{ stroke: "#4CC9F0", strokeWidth: 1 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MetricRow({ label, value, color = "white" }: { label: string; value: string; color?: string }) {
  const cls = {
    white: "text-[#E6EDF3]",
    green: "text-[#00FF9C]",
    blue: "text-[#4CC9F0]",
    yellow: "text-[#FFD60A]",
    red: "text-[#FF4D4D]",
  }[color] ?? "text-[#E6EDF3]"

  return (
    <div className="flex justify-between items-center border-b border-[#1E2A38] py-1.5">
      <span className="text-[10px] text-[#8B949E]">{label}</span>
      <span className={`text-[11px] font-bold font-mono ${cls}`}>{value}</span>
    </div>
  )
}
