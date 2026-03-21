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
  ReferenceLine,
  type TooltipProps,
} from "recharts"

interface ReturnsDistributionProps {
  returns: number[]
  bins?: number
  height?: number
}

interface HistogramBin {
  binLabel: string
  binStart: number
  binEnd: number
  count: number
}

function computeHistogram(returns: readonly number[], numBins: number): readonly HistogramBin[] {
  if (returns.length === 0) return []

  const sorted = [...returns].sort((a, b) => a - b)
  const min = sorted[0]
  const max = sorted[sorted.length - 1]

  if (min === max) {
    return [{
      binLabel: `${(min * 100).toFixed(2)}%`,
      binStart: min,
      binEnd: min,
      count: returns.length,
    }]
  }

  const binWidth = (max - min) / numBins
  const bins: HistogramBin[] = Array.from({ length: numBins }, (_, i) => {
    const binStart = min + i * binWidth
    const binEnd = min + (i + 1) * binWidth
    const midpoint = (binStart + binEnd) / 2
    return {
      binLabel: `${(midpoint * 100).toFixed(2)}%`,
      binStart,
      binEnd,
      count: 0,
    }
  })

  for (const r of returns) {
    let idx = Math.floor((r - min) / binWidth)
    if (idx >= numBins) idx = numBins - 1
    if (idx < 0) idx = 0
    bins[idx] = { ...bins[idx], count: bins[idx].count + 1 }
  }

  return bins
}

function computeMean(arr: readonly number[]): number {
  if (arr.length === 0) return 0
  return arr.reduce((sum, v) => sum + v, 0) / arr.length
}

function computeStdDev(arr: readonly number[], mean: number): number {
  if (arr.length < 2) return 0
  const variance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (arr.length - 1)
  return Math.sqrt(variance)
}

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null

  const data = payload[0]?.payload as HistogramBin | undefined
  if (!data) return null

  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <p className="text-[#8B949E] mb-1">
        {(data.binStart * 100).toFixed(2)}% to {(data.binEnd * 100).toFixed(2)}%
      </p>
      <p className="text-[#4CC9F0]">Count: {data.count}</p>
    </div>
  )
}

export function ReturnsDistribution({
  returns,
  bins = 50,
  height = 300,
}: ReturnsDistributionProps) {
  const histogramData = useMemo(() => computeHistogram(returns ?? [], bins), [returns, bins])

  const stats = useMemo(() => {
    if (!returns || returns.length === 0) return null
    const mean = computeMean(returns)
    const std = computeStdDev(returns, mean)
    return { mean, std }
  }, [returns])

  if (histogramData.length === 0) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height }}
      >
        <span className="text-[#8B949E] text-sm">No data available</span>
      </div>
    )
  }

  // Find the bin closest to 0 for the reference line
  const zeroBinIndex = histogramData.findIndex(
    (bin) => bin.binStart <= 0 && bin.binEnd >= 0
  )
  const zeroLabel = zeroBinIndex >= 0 ? histogramData[zeroBinIndex].binLabel : "0.00%"

  return (
    <div style={{ position: "relative" }}>
      {stats && (
        <div className="absolute top-2 right-4 text-[10px] font-mono text-[#8B949E] z-10 bg-[#0B0F14]/80 px-2 py-1 rounded border border-[#1E2A38]">
          <span className="text-[#E6EDF3]">
            {"\u03BC"} = {(stats.mean * 100).toFixed(3)}%
          </span>
          <span className="mx-2 text-[#1E2A38]">|</span>
          <span className="text-[#E6EDF3]">
            {"\u03C3"} = {(stats.std * 100).toFixed(3)}%
          </span>
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={histogramData as HistogramBin[]}>
          <CartesianGrid
            stroke="#1E2A38"
            strokeDasharray="2 2"
            vertical={false}
          />
          <XAxis
            dataKey="binLabel"
            tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
            axisLine={{ stroke: "#1E2A38" }}
            tickLine={false}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
            axisLine={{ stroke: "#1E2A38" }}
            tickLine={false}
            allowDecimals={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(78, 201, 240, 0.08)" }} />
          <ReferenceLine x={zeroLabel} stroke="#8B949E" strokeDasharray="4 4" />
          <Bar
            dataKey="count"
            fill="#4CC9F0"
            fillOpacity={0.6}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
