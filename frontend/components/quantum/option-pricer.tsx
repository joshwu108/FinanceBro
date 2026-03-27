"use client"

import { useState } from "react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend, Cell,
} from "recharts"
import { useAppStore } from "@/lib/store"

export function OptionPricer() {
  const quantumRunning = useAppStore((s) => s.quantumRunning)
  const result = useAppStore((s) => s.optionPricingResult)
  const convergence = useAppStore((s) => s.convergenceResult)
  const runOptionPricing = useAppStore((s) => s.runOptionPricing)
  const runConvergence = useAppStore((s) => s.runConvergence)

  const [spot, setSpot] = useState(100)
  const [strike, setStrike] = useState(105)
  const [rate, setRate] = useState(0.05)
  const [vol, setVol] = useState(0.2)
  const [ttm, setTtm] = useState(1.0)

  function handlePrice() {
    runOptionPricing({
      spot_price: spot,
      strike_price: strike,
      risk_free_rate: rate,
      volatility: vol,
      time_to_expiry: ttm,
    })
  }

  // Price comparison data
  const priceData = result ? [
    { method: "Black-Scholes", price: result.black_scholes_price, error: 0 },
    ...(result.classical_mc ? [{
      method: "Classical MC",
      price: result.classical_mc.price,
      error: result.comparison?.mc_error ?? 0,
    }] : []),
    ...(result.quantum_ae ? [{
      method: "Quantum AE",
      price: result.quantum_ae.price,
      error: result.comparison?.qae_error ?? 0,
    }] : []),
  ] : []

  // Convergence data
  const convergenceData = convergence ? [
    ...convergence.classical.map((c, i) => ({
      index: i,
      classical_error: c.error,
      label: `${(c.n_paths ?? 0).toLocaleString()} paths`,
    })),
  ] : []

  return (
    <div className="h-full flex flex-col overflow-auto">
      {/* Input controls */}
      <div className="p-3 border-b border-[#1E2A38] flex flex-wrap items-end gap-3">
        <SliderInput label="Spot (S)" value={spot} min={10} max={500} step={1} onChange={setSpot} width="w-20" />
        <SliderInput label="Strike (K)" value={strike} min={10} max={500} step={1} onChange={setStrike} width="w-20" />
        <SliderInput label="Rate (r)" value={rate} min={0} max={0.3} step={0.005} onChange={setRate} width="w-20" />
        <SliderInput label="Vol (σ)" value={vol} min={0.05} max={2.0} step={0.01} onChange={setVol} width="w-20" />
        <SliderInput label="TTM (T)" value={ttm} min={0.1} max={5.0} step={0.1} onChange={setTtm} width="w-20" />
        <button
          onClick={handlePrice}
          disabled={quantumRunning}
          className="bg-[#4CC9F0] text-[#0B0F14] rounded-sm px-4 py-1 text-[11px] font-bold uppercase tracking-wider hover:bg-[#3AB8DF] transition-colors disabled:opacity-50"
        >
          {quantumRunning ? "Pricing..." : "Price"}
        </button>
        <button
          onClick={runConvergence}
          disabled={quantumRunning}
          className="bg-[#00FF9C] text-[#0B0F14] rounded-sm px-4 py-1 text-[11px] font-bold uppercase tracking-wider hover:bg-[#00CC7A] transition-colors disabled:opacity-50"
        >
          Convergence
        </button>
      </div>

      {!result && !convergence ? (
        <div className="flex-1 flex items-center justify-center text-[#8B949E] text-sm">
          Set option parameters and compare BS vs MC vs Quantum AE pricing
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-2 gap-2 p-2 overflow-auto">
          {/* Price cards */}
          {result && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-3">
                Option Prices
              </div>
              <div className="grid grid-cols-3 gap-3">
                <PriceCard
                  label="Black-Scholes"
                  price={result.black_scholes_price}
                  color="#E6EDF3"
                  sublabel="Analytical"
                />
                {result.classical_mc && (
                  <PriceCard
                    label="Classical MC"
                    price={result.classical_mc.price}
                    color="#4CC9F0"
                    sublabel={`±${result.classical_mc.std_error.toFixed(4)}`}
                    runtime={result.classical_mc.runtime_ms}
                  />
                )}
                {result.quantum_ae && (
                  <PriceCard
                    label="Quantum AE"
                    price={result.quantum_ae.price}
                    color="#00FF9C"
                    sublabel={`${result.quantum_ae.n_qubits} qubits`}
                    runtime={result.quantum_ae.runtime_ms}
                  />
                )}
              </div>
            </div>
          )}

          {/* Error comparison */}
          {result?.comparison && (
            <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Error vs Black-Scholes
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={priceData.filter((d) => d.error > 0)} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis dataKey="method" tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <YAxis tick={{ fill: "#8B949E", fontSize: 9 }} axisLine={{ stroke: "#1E2A38" }} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} />
                  <Bar dataKey="error" name="Absolute Error" radius={[3, 3, 0, 0]}>
                    {priceData.filter((d) => d.error > 0).map((entry, i) => (
                      <Cell key={i} fill={i === 0 ? "#4CC9F0" : "#00FF9C"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Convergence chart */}
          {convergence && (
            <div className="col-span-2 bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-3">
              <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-2">
                Convergence: Classical MC vs Quantum AE
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid stroke="#1E2A38" strokeDasharray="2 2" />
                  <XAxis
                    dataKey="index" type="number"
                    tick={{ fill: "#8B949E", fontSize: 9 }}
                    axisLine={{ stroke: "#1E2A38" }}
                    label={{ value: "Sample index", position: "bottom", fill: "#8B949E", fontSize: 9 }}
                  />
                  <YAxis
                    tick={{ fill: "#8B949E", fontSize: 9 }}
                    axisLine={{ stroke: "#1E2A38" }}
                    label={{ value: "Error", angle: -90, position: "insideLeft", fill: "#8B949E", fontSize: 9 }}
                  />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2A38", fontSize: 10 }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line
                    data={convergence.classical.map((c, i) => ({ index: i, error: c.error }))}
                    dataKey="error" stroke="#4CC9F0" name="Classical MC" dot={true} strokeWidth={2}
                  />
                  <Line
                    data={convergence.quantum.map((q, i) => ({ index: i, error: q.error }))}
                    dataKey="error" stroke="#00FF9C" name="Quantum AE" dot={true} strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-[9px] text-[#8B949E] mt-1 text-center">
                Classical MC: O(1/√N) convergence | Quantum AE: O(1/M) convergence — quadratic speedup
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SliderInput({ label, value, min, max, step, onChange, width }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; width: string
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[9px] text-[#8B949E] uppercase tracking-wider">{label}</label>
      <input
        type="number" min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-2 py-1 text-[11px] text-[#E6EDF3] ${width} focus:outline-none focus:border-[#4CC9F0]`}
      />
    </div>
  )
}

function PriceCard({ label, price, color, sublabel, runtime }: {
  label: string; price: number; color: string; sublabel: string; runtime?: number
}) {
  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm p-2 text-center">
      <div className="text-[8px] text-[#8B949E] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-lg font-black font-mono" style={{ color }}>${price.toFixed(4)}</div>
      <div className="text-[9px] text-[#8B949E] mt-0.5">{sublabel}</div>
      {runtime !== undefined && (
        <div className="text-[8px] text-[#8B949E] mt-0.5">{runtime.toFixed(2)} ms</div>
      )}
    </div>
  )
}
