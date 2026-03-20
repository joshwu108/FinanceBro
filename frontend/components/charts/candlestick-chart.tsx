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

function toChartTime(dateStr: string): Time {
  // lightweight-charts expects a time in seconds or a business-day string.
  // For minute bars we use Unix time (UTC) to avoid collapsing multiple candles per day.
  // Alpaca timestamps can include 6-9 fractional digits; JS Date parsing is most
  // reliable with millisecond precision.
  let fixed = dateStr
  const zMatch = fixed.match(/^(.*T\d{2}:\d{2}:\d{2})\.(\d+)(Z)$/)
  if (zMatch) {
    const prefix = zMatch[1]
    const frac = zMatch[2].slice(0, 3).padEnd(3, "0")
    fixed = `${prefix}.${frac}Z`
  } else {
    const offMatch = fixed.match(/^(.*T\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:\d{2})$/)
    if (offMatch) {
      const prefix = offMatch[1]
      const frac = offMatch[2].slice(0, 3).padEnd(3, "0")
      fixed = `${prefix}.${frac}${offMatch[3]}`
    }
  }

  const ms = new Date(fixed).getTime()
  return Math.floor(ms / 1000) as unknown as Time
}

function formatCandleForBar(bar: OHLCVBar): CandlestickData<Time> {
  return {
    time: toChartTime(bar.date),
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
  }
}

function formatOHLCVForChart(data: OHLCVBar[]): CandlestickData<Time>[] {
  return data.map(formatCandleForBar)
}

function formatVolumeForChart(
  data: OHLCVBar[]
): HistogramData<Time>[] {
  return data.map((bar) => {
    const color = bar.close >= bar.open ? "rgba(0, 255, 156, 0.3)" : "rgba(255, 77, 77, 0.3)"
    return {
      time: toChartTime(bar.date),
      value: bar.volume,
      color,
    }
  })
}

function formatSMAForChart(
  data: OHLCVBar[],
  smaValues: (number | null)[]
): LineData<Time>[] {
  const result: LineData<Time>[] = []
  for (let i = 0; i < data.length; i++) {
    const val = smaValues[i]
    if (val !== null) {
      result.push({
        time: toChartTime(data[i].date),
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
      time: toChartTime(trade.date),
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
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null)
  const ma20SeriesRef = useRef<ISeriesApi<"Line"> | null>(null)
  const ma50SeriesRef = useRef<ISeriesApi<"Line"> | null>(null)

  const lastCandleTimeRef = useRef<Time | null>(null)
  const initializedRef = useRef(false)

  const createChartInstance = useCallback(() => {
    const container = containerRef.current
    if (!container) return

    // Clean up existing chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chartHeight = container.clientHeight || 400

    const chart = createChart(container, {
      width: container.clientWidth,
      height: chartHeight,
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
    candlestickSeriesRef.current = chart.addCandlestickSeries({
        upColor: "#00FF9C",
        downColor: "#FF4D4D",
        borderUpColor: "#00FF9C",
        borderDownColor: "#FF4D4D",
        wickUpColor: "#00FF9C",
        wickDownColor: "#FF4D4D",
      })

    // Volume histogram
    if (showVolume) {
      volumeSeriesRef.current = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      })

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
    }

    // Moving averages
    if (showMA20) {
      ma20SeriesRef.current = chart.addLineSeries({
        color: "#4CC9F0",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
    }

    if (showMA50) {
      ma50SeriesRef.current = chart.addLineSeries({
        color: "#FFD60A",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
    }

    // Trade markers
    if (trades && trades.length > 0 && candlestickSeriesRef.current) {
      candlestickSeriesRef.current.setMarkers(buildTradeMarkers(trades))
    }

    // Fit content
    chart.timeScale().fitContent()

    initializedRef.current = false
    lastCandleTimeRef.current = null

    // Resize observer
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect()
    }

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height: h } = entry.contentRect
        if (chartRef.current && width > 0 && h > 0) {
          chartRef.current.applyOptions({ width, height: h })
        }
      }
    })

    resizeObserver.observe(container)
    resizeObserverRef.current = resizeObserver
  }, [showMA20, showMA50, showVolume])

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

  // Streaming-friendly updates:
  // - Candles: update only the last candle (O(1) in the common path).
  // - Volume: update only the last bar.
  // - MAs: recompute and setData (acceptable with throttling + capped bar count).
  useEffect(() => {
    if (!candlestickSeriesRef.current || !chartRef.current) return
    if (!data || data.length === 0) return

    const series = candlestickSeriesRef.current
    const lastBar = data[data.length - 1]
    const lastCandle = formatCandleForBar(lastBar)

    if (!initializedRef.current) {
      series.setData(formatOHLCVForChart(data))
      lastCandleTimeRef.current = lastCandle.time
      initializedRef.current = true

      if (showVolume && volumeSeriesRef.current) {
        volumeSeriesRef.current.setData(formatVolumeForChart(data))
      }

      if (showMA20 && ma20SeriesRef.current) {
        const closes = data.map((bar) => bar.close)
        const sma20Values = computeSMA(closes, 20)
        ma20SeriesRef.current.setData(formatSMAForChart(data, sma20Values))
      }
      if (showMA50 && ma50SeriesRef.current) {
        const closes = data.map((bar) => bar.close)
        const sma50Values = computeSMA(closes, 50)
        ma50SeriesRef.current.setData(formatSMAForChart(data, sma50Values))
      }

      chartRef.current.timeScale().fitContent()
      return
    }

    const prevTime = lastCandleTimeRef.current
    const nextTime = lastCandle.time

    // If the incoming data is out of order, resync to avoid chart corruption.
    const nextMs = typeof nextTime === "number" ? nextTime : null
    const prevMs = prevTime && typeof prevTime === "number" ? prevTime : null
    if (prevMs !== null && nextMs !== null && nextMs < prevMs) {
      series.setData(formatOHLCVForChart(data))
      lastCandleTimeRef.current = nextTime
      if (showVolume && volumeSeriesRef.current) {
        volumeSeriesRef.current.setData(formatVolumeForChart(data))
      }

      if (showMA20 && ma20SeriesRef.current) {
        const closes = data.map((bar) => bar.close)
        const sma20Values = computeSMA(closes, 20)
        ma20SeriesRef.current.setData(formatSMAForChart(data, sma20Values))
      }
      if (showMA50 && ma50SeriesRef.current) {
        const closes = data.map((bar) => bar.close)
        const sma50Values = computeSMA(closes, 50)
        ma50SeriesRef.current.setData(formatSMAForChart(data, sma50Values))
      }
      chartRef.current?.timeScale().fitContent()
    } else if (prevTime === nextTime) {
      series.update(lastCandle)
    } else {
      series.update(lastCandle)
      lastCandleTimeRef.current = nextTime
    }

    if (showVolume && volumeSeriesRef.current) {
      const lastVolume = formatVolumeForChart([lastBar])[0]
      volumeSeriesRef.current.update(lastVolume)
    }

    // Update MAs to reflect new candle.
    const closes = data.map((bar) => bar.close)
    if (showMA20 && ma20SeriesRef.current) {
      const sma20Values = computeSMA(closes, 20)
      ma20SeriesRef.current.setData(formatSMAForChart(data, sma20Values))
    }
    if (showMA50 && ma50SeriesRef.current) {
      const sma50Values = computeSMA(closes, 50)
      ma50SeriesRef.current.setData(formatSMAForChart(data, sma50Values))
    }
  }, [data, showMA20, showMA50, showVolume])

  useEffect(() => {
    if (!candlestickSeriesRef.current) return
    if (!trades || trades.length === 0) {
      candlestickSeriesRef.current.setMarkers([])
      return
    }
    candlestickSeriesRef.current.setMarkers(buildTradeMarkers(trades))
  }, [trades])

  if (!data || data.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0B0F14] text-[#8B949E] font-mono text-sm">
        No price data available. Run the pipeline or select a symbol.
      </div>
    )
  }

  return <div ref={containerRef} className="w-full h-full" />
}
