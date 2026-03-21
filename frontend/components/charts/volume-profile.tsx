"use client"

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  type TooltipProps,
} from "recharts"
import type { OHLCVBar } from "@/lib/types"

interface VolumeProfileProps {
  data: OHLCVBar[]
  height?: number
}

interface VolumeBarDatum {
  date: string
  volume: number
  color: string
}

function formatVolume(value: number): string {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`
  }
  return value.toString()
}

function formatData(data: OHLCVBar[]): VolumeBarDatum[] {
  return data.map((bar) => ({
    date: bar.date.slice(0, 10),
    volume: bar.volume,
    color: bar.close >= bar.open ? "#00FF9C" : "#FF4D4D",
  }))
}

function CustomTooltip({
  active,
  payload,
  label,
}: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null

  const volume = payload[0].value as number

  return (
    <div
      style={{
        backgroundColor: "#11161C",
        border: "1px solid #1E2A38",
        borderRadius: 4,
        padding: "8px 12px",
        fontFamily: "monospace",
        fontSize: 12,
      }}
    >
      <p style={{ color: "#E6EDF3", margin: 0 }}>{label}</p>
      <p style={{ color: "#8B949E", margin: "4px 0 0 0" }}>
        Vol: {formatVolume(volume)}
      </p>
    </div>
  )
}

export function VolumeProfile({ data, height = 200 }: VolumeProfileProps) {
  if (!data || data.length === 0) {
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0B0F14",
          border: "1px solid #1E2A38",
          borderRadius: 4,
          color: "#8B949E",
          fontFamily: "monospace",
          fontSize: 14,
        }}
      >
        No volume data available
      </div>
    )
  }

  const chartData = formatData(data)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
      >
        <XAxis
          dataKey="date"
          tick={{ fill: "#8B949E", fontSize: 10, fontFamily: "monospace" }}
          tickLine={{ stroke: "#1E2A38" }}
          axisLine={{ stroke: "#1E2A38" }}
          minTickGap={40}
        />
        <YAxis
          tickFormatter={formatVolume}
          tick={{ fill: "#8B949E", fontSize: 10, fontFamily: "monospace" }}
          tickLine={{ stroke: "#1E2A38" }}
          axisLine={{ stroke: "#1E2A38" }}
          width={60}
        />
        <Tooltip
          content={<CustomTooltip />}
          cursor={{ fill: "rgba(30, 42, 56, 0.4)" }}
        />
        <Bar dataKey="volume" radius={[1, 1, 0, 0]}>
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
