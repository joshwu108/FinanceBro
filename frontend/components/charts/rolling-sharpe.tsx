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
  type TooltipProps,
} from "recharts"
import type { EquityCurvePoint } from "@/lib/types"

interface RollingSharpeProps {
  equityCurve: EquityCurvePoint[]
  window?: number
  height?: number
}

interface SharpePoint {
  date: string
  sharpe: number | null
}

function computeRollingSharpe(
  equityCurve: readonly EquityCurvePoint[],
  window: number
): readonly SharpePoint[] {
  if (equityCurve.length < 2) return []

  // Compute daily returns
  const dailyReturns: { date: string; ret: number }[] = []
  for (let i = 1; i < equityCurve.length; i++) {
    const prev = equityCurve[i - 1].value
    const curr = equityCurve[i].value
    const ret = prev !== 0 ? (curr - prev) / prev : 0
    dailyReturns.push({ date: equityCurve[i].date, ret })
  }

  // Compute rolling Sharpe
  const annualizationFactor = Math.sqrt(252)

  return dailyReturns.map((point, i) => {
    if (i < window - 1) {
      return { date: point.date, sharpe: null }
    }

    const windowReturns = dailyReturns.slice(i - window + 1, i + 1).map((d) => d.ret)
    const mean = windowReturns.reduce((sum, v) => sum + v, 0) / windowReturns.length

    const variance =
      windowReturns.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (windowReturns.length - 1)
    const std = Math.sqrt(variance)

    const sharpe = std !== 0 ? (mean / std) * annualizationFactor : 0

    return { date: point.date, sharpe }
  })
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null

  const sharpe = payload[0]?.value
  if (sharpe == null) return null

  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <p className="text-[#8B949E] mb-1">{label}</p>
      <p className={Number(sharpe) >= 0 ? "text-[#00FF9C]" : "text-[#FF4D4D]"}>
        Sharpe: {Number(sharpe).toFixed(3)}
      </p>
    </div>
  )
}

export function RollingSharpe({
  equityCurve,
  window = 63,
  height = 300,
}: RollingSharpeProps) {
  const sharpeData = useMemo(
    () => computeRollingSharpe(equityCurve ?? [], window),
    [equityCurve, window]
  )

  // Filter to points with valid sharpe values for display
  const validData = useMemo(
    () => sharpeData.filter((d): d is { date: string; sharpe: number } => d.sharpe !== null),
    [sharpeData]
  )

  if (validData.length === 0) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height }}
      >
        <span className="text-[#8B949E] text-sm">No data available</span>
      </div>
    )
  }

  const allSharpes = validData.map((d) => d.sharpe)
  const minVal = Math.min(...allSharpes)
  const maxVal = Math.max(...allSharpes)
  const range = maxVal - minVal
  const padding = Math.max(range * 0.1, 0.2)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={validData}>
        <CartesianGrid
          stroke="#1E2A38"
          strokeDasharray="2 2"
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
          minTickGap={40}
        />
        <YAxis
          domain={[minVal - padding, maxVal + padding]}
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
          tickFormatter={(value: number) => value.toFixed(1)}
          width={40}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine
          y={0}
          stroke="#8B949E"
          strokeDasharray="4 4"
          strokeWidth={1}
        />
        <Line
          type="monotone"
          dataKey="sharpe"
          stroke="#4CC9F0"
          strokeWidth={1.5}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
