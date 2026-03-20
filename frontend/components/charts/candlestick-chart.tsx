"use client"

import { useEffect, useRef, useCallback } from "react"
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
  type CandlestickData,
  type HistogramData,
  type LineData,
} from "lightweight-charts"
import type { OHLCVBar, Trade } from "@/lib/types"

interface CandlestickChartProps {
  data: OHLCVBar[]
  trades?: Trade[]
  showMA20?: boolean
  showMA50?: boolean
  showVolume?: boolean
  height?: number
}

function computeSMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = []
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      result.push(null)
    } else {
      let sum = 0
      for (let j = i - period + 1; j <= i; j++) {
        sum += closes[j]
      }
      result.push(sum / period)
    }
  }
  return result
}

function formatOHLCVForChart(data: OHLCVBar[]): CandlestickData<Time>[] {
  return data.map((bar) => ({
    time: bar.date.slice(0, 10) as unknown as Time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  }))
}

function formatVolumeForChart(
  data: OHLCVBar[]
): HistogramData<Time>[] {
  return data.map((bar) => ({
    time: bar.date.slice(0, 10) as unknown as Time,
    value: bar.volume,
    color:
      bar.close >= bar.open
        ? "rgba(0, 255, 156, 0.3)"
        : "rgba(255, 77, 77, 0.3)",
  }))
}

function formatSMAForChart(
  dates: string[],
  smaValues: (number | null)[]
): LineData<Time>[] {
  const result: LineData<Time>[] = []
  for (let i = 0; i < dates.length; i++) {
    const val = smaValues[i]
    if (val !== null) {
      result.push({
        time: dates[i].slice(0, 10) as unknown as Time,
        value: val,
      })
    }
  }
  return result
}

function buildTradeMarkers(trades: Trade[]): SeriesMarker<Time>[] {
  const sorted = [...trades].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  )

  return sorted.map((trade) => {
    const isBuy = trade.action.toLowerCase() === "buy"
    return {
      time: trade.date.slice(0, 10) as unknown as Time,
      position: isBuy ? ("belowBar" as const) : ("aboveBar" as const),
      shape: isBuy ? ("arrowUp" as const) : ("arrowDown" as const),
      color: isBuy ? "#00FF9C" : "#FF4D4D",
      text: isBuy
        ? `Buy ${trade.shares}@${trade.price.toFixed(2)}`
        : `${trade.action} ${trade.shares}@${trade.price.toFixed(2)}`,
    }
  })
}

export function CandlestickChart({
  data,
  trades,
  showMA20 = true,
  showMA50 = true,
  showVolume = true,
  height = 500,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  const createChartInstance = useCallback(() => {
    const container = containerRef.current
    if (!container || data.length === 0) return

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

    // Candlestick series
    const candlestickSeries: ISeriesApi<"Candlestick"> =
      chart.addCandlestickSeries({
        upColor: "#00FF9C",
        downColor: "#FF4D4D",
        borderUpColor: "#00FF9C",
        borderDownColor: "#FF4D4D",
        wickUpColor: "#00FF9C",
        wickDownColor: "#FF4D4D",
      })

    const candleData = formatOHLCVForChart(data)
    candlestickSeries.setData(candleData)

    // Volume histogram
    if (showVolume) {
      const volumeSeries: ISeriesApi<"Histogram"> =
        chart.addHistogramSeries({
          priceFormat: { type: "volume" },
          priceScaleId: "volume",
        })

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })

      volumeSeries.setData(formatVolumeForChart(data))
    }

    // Moving averages
    const closes = data.map((bar) => bar.close)
    const dates = data.map((bar) => bar.date)

    if (showMA20) {
      const sma20Values = computeSMA(closes, 20)
      const ma20Series: ISeriesApi<"Line"> = chart.addLineSeries({
        color: "#4CC9F0",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
      ma20Series.setData(formatSMAForChart(dates, sma20Values))
    }

    if (showMA50) {
      const sma50Values = computeSMA(closes, 50)
      const ma50Series: ISeriesApi<"Line"> = chart.addLineSeries({
        color: "#FFD60A",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
      ma50Series.setData(formatSMAForChart(dates, sma50Values))
    }

    // Trade markers
    if (trades && trades.length > 0) {
      const markers = buildTradeMarkers(trades)
      candlestickSeries.setMarkers(markers)
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
  }, [data, trades, showMA20, showMA50, showVolume, height])

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
        No price data available
      </div>
    )
  }

  return <div ref={containerRef} style={{ width: "100%", height }} />
}
