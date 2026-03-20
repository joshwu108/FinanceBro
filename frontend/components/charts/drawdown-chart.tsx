"use client"

import { useMemo } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  type TooltipProps,
} from "recharts"
import type { EquityCurvePoint } from "@/lib/types"

interface DrawdownChartProps {
  equityCurve: EquityCurvePoint[]
  height?: number
}

interface DrawdownPoint {
  date: string
  drawdown: number
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null

  const drawdown = payload[0]?.value as number

  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <p className="text-[#8B949E] mb-1">{label}</p>
      <p className="text-[#FF4D4D]">{drawdown.toFixed(2)}%</p>
    </div>
  )
}

export function DrawdownChart({ equityCurve, height = 300 }: DrawdownChartProps) {
  const drawdownData = useMemo((): readonly DrawdownPoint[] => {
    if (!equityCurve || equityCurve.length === 0) return []

    let peak = -Infinity
    return equityCurve.map((point) => {
      if (point.value > peak) {
        peak = point.value
      }
      const drawdown = ((point.value / peak) - 1) * 100
      return { date: point.date, drawdown }
    })
  }, [equityCurve])

  if (drawdownData.length === 0) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height }}
      >
        <span className="text-[#8B949E] text-sm">No data available</span>
      </div>
    )
  }

  const minDrawdown = Math.min(...drawdownData.map((d) => d.drawdown))
  const yMin = Math.floor(minDrawdown * 1.1)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={drawdownData as DrawdownPoint[]}>
        <defs>
          <linearGradient id="drawdownFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FF4D4D" stopOpacity={0.2} />
            <stop offset="100%" stopColor="#FF4D4D" stopOpacity={0.05} />
          </linearGradient>
        </defs>
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
          domain={[yMin, 0]}
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
          tickFormatter={(value: number) => `${value.toFixed(0)}%`}
          width={50}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#FF4D4D"
          strokeWidth={1.5}
          fill="url(#drawdownFill)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
