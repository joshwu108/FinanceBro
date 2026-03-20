"use client"

import { useState, useMemo } from "react"
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, LineChart,
} from "recharts"
import { TrendingUp, BarChart2, Layers } from "lucide-react"

// ── Seeded PRNG for deterministic fake data ──────────────────────────────────
function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function generateOHLC(symbol: string, days = 90) {
  const rand = mulberry32(symbol.charCodeAt(0) * 137 + symbol.length * 31)
  const data: {
    date: string; open: number; high: number; low: number; close: number
    volume: number; signal: "buy" | "sell" | null; signalStrength: number
    ma20: number; ma50: number
  }[] = []

  let price = 150 + rand() * 200
  const startDate = new Date("2024-01-01")

  for (let i = 0; i < days; i++) {
    const d = new Date(startDate)
    d.setDate(d.getDate() + i)
    const dateStr = d.toLocaleDateString("en-US", { month: "short", day: "2-digit" })

    const change = (rand() - 0.48) * price * 0.025
    const open = price
    const close = price + change
    const high = Math.max(open, close) + rand() * price * 0.012
    const low = Math.min(open, close) - rand() * price * 0.012
    const volume = Math.floor((0.5 + rand()) * 12_000_000)
    const signalType = rand() < 0.08 ? "buy" : rand() < 0.08 ? "sell" : null
    const signalStrength = rand()

    data.push({ date: dateStr, open, high, low, close, volume, signal: signalType, signalStrength, ma20: 0, ma50: 0 })
    price = close
  }

  // compute MAs
  for (let i = 0; i < data.length; i++) {
    if (i >= 19) data[i].ma20 = data.slice(i - 19, i + 1).reduce((s, d) => s + d.close, 0) / 20
    if (i >= 49) data[i].ma50 = data.slice(i - 49, i + 1).reduce((s, d) => s + d.close, 0) / 50
  }

  return data
}

function generateEquity(days = 90) {
  const rand = mulberry32(42)
  const spyRand = mulberry32(99)
  const equity: { date: string; strategy: number; benchmark: number }[] = []
  let strat = 100, bench = 100
  const startDate = new Date("2024-01-01")

  for (let i = 0; i < days; i++) {
    const d = new Date(startDate)
    d.setDate(d.getDate() + i)
    strat *= 1 + (rand() - 0.47) * 0.02
    bench *= 1 + (spyRand() - 0.49) * 0.015
    equity.push({
      date: d.toLocaleDateString("en-US", { month: "short", day: "2-digit" }),
      strategy: +strat.toFixed(2),
      benchmark: +bench.toFixed(2),
    })
  }
  return equity
}

// Custom candlestick bar
function CandleBar(props: {
  x?: number; y?: number; width?: number; height?: number; payload?: {
    open: number; high: number; low: number; close: number
  }
}) {
  const { x = 0, y = 0, width = 0, payload } = props
  if (!payload) return null
  const { open, high, low, close } = payload
  const bullish = close >= open
  const color = bullish ? "#00FF9C" : "#FF4D4D"
  const barWidth = Math.max(3, width - 2)

  // We get chart height via the parent, so we use normalized coords from the chart
  // Recharts passes data coordinates through the chart's y-scale but for composed chart
  // we lean on the chart container rendering. We'll render custom shapes via CustomBarShape.
  return (
    <g>
      <rect x={x - barWidth / 2 + width / 2} y={y} width={barWidth} height={Math.abs(props.height ?? 0)} fill={color} />
    </g>
  )
}

function ChartPanelHeader({ title, controls }: { title: string; controls?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-[#1E2A38] bg-[#0B0F14]">
      <span className="text-[10px] font-bold uppercase tracking-wider text-[#8B949E]">{title}</span>
      {controls}
    </div>
  )
}

// ── Custom Candlestick shape for ComposedChart ────────────────────────────────
interface CandleProps {
  x?: number
  y?: number
  width?: number
  height?: number
  payload?: { open: number; high: number; low: number; close: number; volume: number; signal: "buy" | "sell" | null }
  // recharts internal scale helpers
  yAxis?: { scale: (v: number) => number }
}

function CandleShape(props: CandleProps) {
  const { x = 0, width = 0, payload, yAxis } = props
  if (!payload || !yAxis?.scale) return null

  const { open, high, low, close } = payload
  const scale = yAxis.scale
  const y_open = scale(open)
  const y_close = scale(close)
  const y_high = scale(high)
  const y_low = scale(low)

  const bullish = close >= open
  const color = bullish ? "#00FF9C" : "#FF4D4D"
  const barW = Math.max(3, width * 0.6)
  const cx = x + width / 2
  const bodyTop = Math.min(y_open, y_close)
  const bodyH = Math.max(1, Math.abs(y_close - y_open))

  return (
    <g>
      {/* Wick */}
      <line x1={cx} y1={y_high} x2={cx} y2={y_low} stroke={color} strokeWidth={0.8} />
      {/* Body */}
      <rect x={cx - barW / 2} y={bodyTop} width={barW} height={bodyH} fill={color} opacity={0.9} />
    </g>
  )
}

// Custom dot for buy/sell signals
interface SignalDotProps {
  cx?: number
  cy?: number
  payload?: { signal: "buy" | "sell" | null; close: number }
  value?: number
}

function SignalDot({ cx = 0, cy = 0, payload }: SignalDotProps) {
  if (!payload?.signal) return null
  const isBuy = payload.signal === "buy"
  const color = isBuy ? "#00FF9C" : "#FF4D4D"
  const label = isBuy ? "B" : "S"
  return (
    <g>
      <circle cx={cx} cy={cy} r={6} fill={color} opacity={0.85} />
      <text x={cx} y={cy + 4} textAnchor="middle" fill="#0B0F14" fontSize={7} fontWeight="bold">{label}</text>
    </g>
  )
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────
function CandleTooltip({ active, payload, label }: { active?: boolean; payload?: { payload: { open: number; high: number; low: number; close: number; volume: number } }[]; label?: string }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const bullish = d.close >= d.open
  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <div className="text-[#8B949E] mb-1">{label}</div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
        <span className="text-[#8B949E]">O</span><span className="text-[#E6EDF3]">{d.open.toFixed(2)}</span>
        <span className="text-[#8B949E]">H</span><span className="text-[#00FF9C]">{d.high.toFixed(2)}</span>
        <span className="text-[#8B949E]">L</span><span className="text-[#FF4D4D]">{d.low.toFixed(2)}</span>
        <span className={`text-[#8B949E]`}>C</span><span className={bullish ? "text-[#00FF9C]" : "text-[#FF4D4D]"}>{d.close.toFixed(2)}</span>
        <span className="text-[#8B949E]">Vol</span><span className="text-[#4CC9F0]">{(d.volume / 1e6).toFixed(2)}M</span>
      </div>
    </div>
  )
}

function EquityTooltip({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <div className="text-[#8B949E] mb-1">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-[#8B949E]">{p.name}:</span>
          <span style={{ color: p.color }}>{p.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Main Panel ────────────────────────────────────────────────────────────────
interface MainPanelProps {
  symbol: string
  showMA: boolean
  onToggleMA: () => void
  showVolBands: boolean
  onToggleVolBands: () => void
}

export function MainPanel({ symbol, showMA, onToggleMA, showVolBands, onToggleVolBands }: MainPanelProps) {
  const ohlc = useMemo(() => generateOHLC(symbol), [symbol])
  const equity = useMemo(() => generateEquity(), [])

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
      {/* Row 1: Candlestick + signals */}
      <div className="flex-[3] flex flex-col min-h-0 border-b border-[#1E2A38]">
        <ChartPanelHeader
          title={`${symbol} — Price / Signals`}
          controls={
            <div className="flex items-center gap-2">
              <button
                onClick={onToggleMA}
                className={`flex items-center gap-1 px-2 py-0.5 text-[9px] uppercase tracking-wider border rounded-sm transition-colors ${showMA ? "border-[#4CC9F0] text-[#4CC9F0] bg-[#4CC9F0]/10" : "border-[#1E2A38] text-[#8B949E] hover:border-[#4CC9F0]"}`}
              >
                <Layers className="w-2.5 h-2.5" /> MA
              </button>
              <button
                onClick={onToggleVolBands}
                className={`flex items-center gap-1 px-2 py-0.5 text-[9px] uppercase tracking-wider border rounded-sm transition-colors ${showVolBands ? "border-[#FFD60A] text-[#FFD60A] bg-[#FFD60A]/10" : "border-[#1E2A38] text-[#8B949E] hover:border-[#FFD60A]"}`}
              >
                <BarChart2 className="w-2.5 h-2.5" /> Vol Bands
              </button>
              <div className="flex items-center gap-2 text-[9px] text-[#8B949E]">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#00FF9C] inline-block" /> BUY</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#FF4D4D] inline-block" /> SELL</span>
              </div>
            </div>
          }
        />
        <div className="flex-1 min-h-0 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={ohlc} margin={{ top: 8, right: 8, bottom: 0, left: 44 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="#1E2A38" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
                tickLine={false}
                axisLine={{ stroke: "#1E2A38" }}
                interval={Math.floor(ohlc.length / 8)}
              />
              <YAxis
                orientation="right"
                tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                width={48}
              />
              <Tooltip content={<CandleTooltip />} />
              {/* Candles via custom bar shape */}
              <Bar dataKey="high" shape={<CandleShape />} isAnimationActive={false} />
              {showMA && (
                <>
                  <Line dataKey="ma20" stroke="#4CC9F0" dot={false} strokeWidth={1} name="MA20" isAnimationActive={false} strokeOpacity={0.8} />
                  <Line dataKey="ma50" stroke="#FFD60A" dot={false} strokeWidth={1} name="MA50" isAnimationActive={false} strokeOpacity={0.8} />
                </>
              )}
              {/* Signal dots rendered on close line */}
              <Line
                dataKey="close"
                stroke="transparent"
                dot={<SignalDot />}
                activeDot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 2: Equity curve */}
      <div className="flex-[2] flex flex-col min-h-0">
        <ChartPanelHeader
          title="Equity Curve vs Benchmark"
          controls={
            <div className="flex items-center gap-3 text-[9px] text-[#8B949E]">
              <span className="flex items-center gap-1"><span className="w-5 border-t-2 border-[#00FF9C] inline-block" /> Strategy</span>
              <span className="flex items-center gap-1"><span className="w-5 border-t-2 border-dashed border-[#4CC9F0] inline-block" /> SPY B&H</span>
              <TrendingUp className="w-3 h-3" />
            </div>
          }
        />
        <div className="flex-1 min-h-0 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={equity} margin={{ top: 8, right: 8, bottom: 0, left: 44 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="#1E2A38" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
                tickLine={false}
                axisLine={{ stroke: "#1E2A38" }}
                interval={Math.floor(equity.length / 8)}
              />
              <YAxis
                orientation="right"
                tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `${v.toFixed(0)}`}
                width={40}
              />
              <Tooltip content={<EquityTooltip />} />
              <ReferenceLine y={100} stroke="#1E2A38" strokeDasharray="4 2" />
              <Line
                type="monotone"
                dataKey="strategy"
                stroke="#00FF9C"
                dot={false}
                strokeWidth={1.5}
                name="Strategy"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="#4CC9F0"
                dot={false}
                strokeWidth={1}
                strokeDasharray="4 2"
                name="SPY B&H"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
