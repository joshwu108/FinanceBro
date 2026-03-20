"use client"

import { useEffect, useRef, useCallback } from "react"
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type Time,
} from "lightweight-charts"
import type { EquityCurvePoint } from "@/lib/types"

interface EquityCurveProps {
  equityCurve: EquityCurvePoint[]
  benchmarkCurve?: EquityCurvePoint[]
  height?: number
}

function formatCurveData(points: EquityCurvePoint[]): LineData<Time>[] {
  return points.map((point) => ({
    time: point.date.slice(0, 10) as unknown as Time,
    value: point.value,
  }))
}

export function EquityCurve({
  equityCurve,
  benchmarkCurve,
  height = 350,
}: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  const createChartInstance = useCallback(() => {
    const container = containerRef.current
    if (!container || equityCurve.length === 0) return

    // Clean up existing chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(container, {
      width: container.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#0B0F14" },
        textColor: "#8B949E",
        fontFamily: "monospace",
      },
      grid: {
        vertLines: { color: "#1E2A38" },
        horzLines: { color: "#1E2A38" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1E2A38" },
      timeScale: { borderColor: "#1E2A38" },
    })

    chartRef.current = chart

    // Strategy equity line (green, solid, 2px)
    const strategySeries: ISeriesApi<"Line"> = chart.addLineSeries({
      color: "#00FF9C",
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: true,
      title: "Strategy",
    })

    const strategyData = formatCurveData(equityCurve)
    strategySeries.setData(strategyData)

    // Reference line at starting value
    if (strategyData.length > 0) {
      const startingValue = strategyData[0].value
      strategySeries.createPriceLine({
        price: startingValue,
        color: "#8B949E",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "Start",
      })
    }

    // Benchmark line (blue, dashed, 1px) — only if provided
    if (benchmarkCurve && benchmarkCurve.length > 0) {
      const benchmarkSeries: ISeriesApi<"Line"> = chart.addLineSeries({
        color: "#4CC9F0",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: true,
        title: "Benchmark",
      })

      benchmarkSeries.setData(formatCurveData(benchmarkCurve))
    }

    // Fit content
    chart.timeScale().fitContent()

    // Resize observer
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect()
    }

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect
        if (chartRef.current && width > 0) {
          chartRef.current.applyOptions({ width })
        }
      }
    })

    resizeObserver.observe(container)
    resizeObserverRef.current = resizeObserver
  }, [equityCurve, benchmarkCurve, height])

  useEffect(() => {
    createChartInstance()

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
        resizeObserverRef.current = null
      }
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [createChartInstance])

  if (!equityCurve || equityCurve.length === 0) {
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
        No equity curve data available
      </div>
    )
  }

  return <div ref={containerRef} style={{ width: "100%", height }} />
}
