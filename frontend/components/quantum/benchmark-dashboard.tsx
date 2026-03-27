"use client"

import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell,
} from "recharts"
import { useAppStore } from "@/lib/store"

export function BenchmarkDashboard() {
  const quantumRunning = useAppStore((s) => s.quantumRunning)
  const portfolioScaling = useAppStore((s) => s.portfolioScalingResult)
  const maxCutScaling = useAppStore((s) => s.maxCutScalingResult)
  const cppSpeedup = useAppStore((s) => s.cppSpeedupResult)
  const runPortfolioScaling = useAppStore((s) => s.runPortfolioScaling)
  const runMaxCutScaling = useAppStore((s) => s.runMaxCutScaling)
  const runCppSpeedup = useAppStore((s) => s.runCppSpeedup)

  const hasAny = portfolioScaling || maxCutScaling || cppSpeedup

  return (
    <div className="h-full flex flex-col overflow-auto">
      {/* Run buttons */}
      <div className="p-3 border-b border-[#1E2A38] flex items-center gap-3">
        <span className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest">Benchmarks</span>
        <button
          onClick={runPortfolioScaling} disabled={quantumRunning}
          className="bg-[#4CC9F0] text-[#0B0F14] rounded-sm px-3 py-1 text-[10px] font-bold uppercase hover:bg-[#3AB8DF] transition-colors disabled:opacity-50"
        >
          Portfolio Scaling
        </button>
        <button
          onClick={runMaxCutScaling} disabled={quantumRunning}
          className="bg-[#00FF9C] text-[#0B0F14] rounded-sm px-3 py-1 text-[10px] font-bold uppercase hover:bg-[#00CC7A] transition-colors disabled:opacity-50"
        >
          Max-Cut Scaling
        </button>
        <button
          onClick={runCppSpeedup} disabled={quantumRunning}
          className="bg-[#FFD60A] text-[#0B0F14] rounded-sm px-3 py-1 text-[10px] font-bold uppercase hover:bg-[#E6C009] transition-colors disabled:opacity-50"
        >
          C++ Speedup
        </button>
        {quantumRunning && (
          <span className="text-[10px] text-[#FFD60A] animate-pulse">Running benchmark...</span>
        )}
      </div>

      {!hasAny ? (
        <div className="flex-1 flex items-center justify-center text-[#8B949E] text-sm">
          Run scaling experiments to measure QAOA vs classical performance
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 gap-2 p-2 overflow-auto">
          {/* Portfolio scaling: runtime */}
          {portfolioScaling && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Portfolio Scaling — Runtime
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={portfolioScaling.results} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="n_assets" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} label={{ value: "Assets", position: "bottom", fill: "#8B949E", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#8B949E", fontSize: 9 }} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => `${v.toFixed(2)} ms`} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line dataKey="classical_runtime_ms" stroke="#4CC9F0" name="Classical" dot strokeWidth={2} />
                  <Line dataKey="qaoa_runtime_ms" stroke="#00FF9C" name="QAOA" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Portfolio scaling: approximation ratio */}
          {portfolioScaling && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                QAOA Approximation Ratio
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={portfolioScaling.results.filter((r) => r.approximation_ratio != null)} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="n_assets" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} domain={[0, 2]} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} />
                  <Bar dataKey="approximation_ratio" name="Approx Ratio" fill="#00FF9C" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Max-Cut scaling */}
          {maxCutScaling && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Max-Cut Scaling — Runtime
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={maxCutScaling.results} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="n_nodes" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} label={{ value: "Nodes", position: "bottom", fill: "#8B949E", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => `${v.toFixed(2)} ms`} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line dataKey="qaoa_runtime_ms" stroke="#00FF9C" name="QAOA" dot strokeWidth={2} />
                  <Line dataKey="sa_runtime_ms" stroke="#4CC9F0" name="Simulated Annealing" dot strokeWidth={2} />
                  <Line dataKey="greedy_runtime_ms" stroke="#8B949E" name="Greedy" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Max-Cut approximation ratios */}
          {maxCutScaling && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Max-Cut Approximation Ratios
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={maxCutScaling.results.filter((r) => r.qaoa_approx_ratio != null)} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="n_nodes" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} domain={[0, 2]} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line dataKey="qaoa_approx_ratio" stroke="#00FF9C" name="QAOA" dot strokeWidth={2} />
                  <Line dataKey="sa_approx_ratio" stroke="#4CC9F0" name="SA" dot strokeWidth={2} />
                  <Line dataKey="greedy_approx_ratio" stroke="#8B949E" name="Greedy" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* C++ speedup */}
          {cppSpeedup && (
            <div className="col-span-2 bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                C++ vs Python Speedup {!cppSpeedup.has_cpp && "(C++ not available — showing Python only)"}
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={cppSpeedup.results} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="n_qubits" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} label={{ value: "Qubits", position: "bottom", fill: "#8B949E", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} formatter={(v: number) => `${v.toFixed(3)} ms`} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="python_ms" name="Python" fill="#4CC9F0" radius={[3, 3, 0, 0]} />
                  {cppSpeedup.has_cpp && <Bar dataKey="cpp_ms" name="C++" fill="#00FF9C" radius={[3, 3, 0, 0]} />}
                </BarChart>
              </ResponsiveContainer>
              {cppSpeedup.has_cpp && (
                <div className="mt-2 flex gap-4 justify-center">
                  {cppSpeedup.results.filter((r) => r.speedup != null).map((r) => (
                    <div key={r.n_qubits} className="text-center">
                      <div className="text-[8px] text-[#8B949E]">{r.n_qubits}q</div>
                      <div className="text-[11px] font-bold text-[#FFD60A]">{r.speedup!.toFixed(1)}x</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
