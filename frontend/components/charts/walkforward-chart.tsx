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
  Cell,
  type TooltipProps,
} from "recharts"

interface WalkForwardChartProps {
  foldResults: {
    fold_index: number
    backtest_metrics: {
      sharpe: number
      max_drawdown: number
      total_return: number
      win_rate: number
    }
  }[]
  height?: number
}

interface FoldRow {
  label: string
  fold_index: number
  sharpe: number
  max_drawdown: number
  total_return: number
  win_rate: number
}

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null

  const data = payload[0]?.payload as FoldRow | undefined
  if (!data) return null

  return (
    <div className="bg-[#11161C] border border-[#1E2A38] rounded-sm px-3 py-2 text-[10px] font-mono shadow-xl">
      <p className="text-[#E6EDF3] mb-1 font-semibold">{data.label}</p>
      <div className="space-y-0.5">
        <p>
          <span className="text-[#8B949E]">Sharpe: </span>
          <span className={data.sharpe >= 0 ? "text-[#00FF9C]" : "text-[#FF4D4D]"}>
            {data.sharpe.toFixed(3)}
          </span>
        </p>
        <p>
          <span className="text-[#8B949E]">Drawdown: </span>
          <span className="text-[#FF4D4D]">{(data.max_drawdown * 100).toFixed(1)}%</span>
        </p>
        <p>
          <span className="text-[#8B949E]">Return: </span>
          <span className={data.total_return >= 0 ? "text-[#00FF9C]" : "text-[#FF4D4D]"}>
            {(data.total_return * 100).toFixed(1)}%
          </span>
        </p>
        <p>
          <span className="text-[#8B949E]">Win Rate: </span>
          <span className="text-[#E6EDF3]">{(data.win_rate * 100).toFixed(1)}%</span>
        </p>
      </div>
    </div>
  )
}

export function WalkForwardChart({ foldResults, height = 300 }: WalkForwardChartProps) {
  const { data, meanSharpe } = useMemo(() => {
    if (!foldResults || foldResults.length === 0) {
      return { data: [] as readonly FoldRow[], meanSharpe: 0 }
    }

    const rows: FoldRow[] = foldResults.map((fold) => ({
      label: `Fold ${fold.fold_index + 1}`,
      fold_index: fold.fold_index,
      sharpe: fold.backtest_metrics.sharpe,
      max_drawdown: fold.backtest_metrics.max_drawdown,
      total_return: fold.backtest_metrics.total_return,
      win_rate: fold.backtest_metrics.win_rate,
    }))

    const avg = rows.reduce((sum, r) => sum + r.sharpe, 0) / rows.length

    return { data: rows as readonly FoldRow[], meanSharpe: avg }
  }, [foldResults])

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

  const allSharpes = (data as readonly FoldRow[]).map((d) => d.sharpe)
  const minSharpe = Math.min(...allSharpes, 0)
  const maxSharpe = Math.max(...allSharpes, 0)
  const padding = Math.max(Math.abs(maxSharpe - minSharpe) * 0.15, 0.1)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data as FoldRow[]}>
        <CartesianGrid
          stroke="#1E2A38"
          strokeDasharray="2 2"
          vertical={false}
        />
        <XAxis
          dataKey="label"
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
        />
        <YAxis
          domain={[minSharpe - padding, maxSharpe + padding]}
          tick={{ fill: "#8B949E", fontSize: 9, fontFamily: "monospace" }}
          axisLine={{ stroke: "#1E2A38" }}
          tickLine={false}
          tickFormatter={(value: number) => value.toFixed(2)}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(78, 201, 240, 0.08)" }} />
        <ReferenceLine y={0} stroke="#8B949E" strokeWidth={1} />
        <ReferenceLine
          y={meanSharpe}
          stroke="#FFD60A"
          strokeDasharray="6 3"
          strokeWidth={1.5}
          label={{
            value: `Mean: ${meanSharpe.toFixed(2)}`,
            position: "right",
            fill: "#FFD60A",
            fontSize: 9,
            fontFamily: "monospace",
          }}
        />
        <Bar dataKey="sharpe" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {(data as FoldRow[]).map((entry) => (
            <Cell
              key={entry.label}
              fill={entry.sharpe >= 0 ? "#00FF9C" : "#FF4D4D"}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
