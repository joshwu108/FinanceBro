"use client"

import { useMemo } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  type TooltipProps,
} from "recharts"

interface FeatureImportanceProps {
  importances: Record<string, number>
  maxFeatures?: number
  height?: number
}

interface FeatureRow {
  name: string
  fullName: string
  importance: number
  rank: number
}

function truncateName(name: string, maxLen: number): string {
  if (name.length <= maxLen) return name
  return name.slice(0, maxLen - 1) + "\u2026"
}

function interpolateColor(rank: number, total: number): string {
  if (total <= 1) return "#4CC9F0"
  const t = rank / (total - 1)
  // Interpolate from blue (#4CC9F0) to green (#00FF9C)
  const r = Math.round(0x4C + (0x00 - 0x4C) * t)
  const g = Math.round(0xC9 + (0xFF - 0xC9) * t)
  const b = Math.round(0xF0 + (0x9C - 0xF0) * t)
  return `rgb(${r}, ${g}, ${b})`
}

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null

  const data = payload[0]?.payload as FeatureRow | undefined
  if (!data) return null

  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <p className="text-[#E6EDF3] mb-1">{data.fullName}</p>
      <p className="text-[#4CC9F0]">{data.importance.toFixed(4)}</p>
    </div>
  )
}

export function FeatureImportance({
  importances,
  maxFeatures = 15,
  height = 400,
}: FeatureImportanceProps) {
  const data = useMemo((): readonly FeatureRow[] => {
    if (!importances || Object.keys(importances).length === 0) return []

    const sorted = Object.entries(importances)
      .map(([name, importance]) => ({ name, importance }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, maxFeatures)

    return sorted.map((item, idx) => ({
      name: truncateName(item.name, 20),
      fullName: item.name,
      importance: item.importance,
      rank: idx,
    }))
  }, [importances, maxFeatures])

  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height }}
      >
        <span className="text-[#8B949E] text-sm">No data available</span>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data as FeatureRow[]}
        layout="vertical"
        margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
      >
        <CartesianGrid
          stroke="#1E2A38"
          strokeDasharray="2 2"
          vertical={false}
        />
        <XAxis
          type="number"
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={130}
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(78, 201, 240, 0.08)" }} />
        <Bar dataKey="importance" radius={[0, 3, 3, 0]} isAnimationActive={false}>
          {(data as FeatureRow[]).map((entry) => (
            <Cell
              key={entry.fullName}
              fill={interpolateColor(entry.rank, data.length)}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
