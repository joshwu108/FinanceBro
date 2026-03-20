"use client"

import { useEffect, useRef } from "react"
import type { MutableRefObject } from "react"
import { useAppStore } from "@/lib/store"
import type { OHLCVBar, WSMessage } from "@/lib/types"

const WS_BASE = "ws://localhost:8000"

function scheduleFlush(
  fn: () => void,
  delayMs: number,
  ref: MutableRefObject<number | null>
) {
  if (ref.current != null) return
  ref.current = window.setTimeout(() => {
    ref.current = null
    fn()
  }, delayMs)
}

export function useAlpacaStream(symbol: string, enabled: boolean) {
  const setWsStatus = useAppStore((s) => s.setWsStatus)
  const setLiveTick = useAppStore((s) => s.setLiveTick)
  const setLiveQuote = useAppStore((s) => s.setLiveQuote)
  const upsertLiveBar = useAppStore((s) => s.upsertLiveBar)
  const setLiveMinuteBars = useAppStore((s) => s.setLiveMinuteBars)

  const wsRef = useRef<WebSocket | null>(null)
  const cancelledRef = useRef(false)

  const pendingBarRef = useRef<OHLCVBar | null>(null)
  const flushTimeoutRef = useRef<number | null>(null)

  useEffect(() => {
    cancelledRef.current = false

    // When disabled, make sure we disconnect and don't keep pending timers.
    if (!enabled) {
      setWsStatus("DISCONNECTED")
      wsRef.current?.close()
      wsRef.current = null
      pendingBarRef.current = null
      if (flushTimeoutRef.current != null) {
        window.clearTimeout(flushTimeoutRef.current)
        flushTimeoutRef.current = null
      }
      return
    }

    const sym = symbol.trim().toUpperCase()
    if (!sym) return

    let backoffMs = 1000

    const flushBars = () => {
      const bar = pendingBarRef.current
      pendingBarRef.current = null
      if (!bar) return
      upsertLiveBar(sym, bar)
    }

    const connect = () => {
      if (cancelledRef.current) return

      setWsStatus("CONNECTING")

      const ws = new WebSocket(`${WS_BASE}/ws/live/${encodeURIComponent(sym)}`)
      wsRef.current = ws

      ws.onopen = () => {
        if (cancelledRef.current) return
        backoffMs = 1000
        setWsStatus("CONNECTED")
      }

      ws.onmessage = (evt) => {
        if (cancelledRef.current) return
        try {
          const parsed = JSON.parse(evt.data) as WSMessage
          if (!parsed || typeof parsed !== "object") return

          if (parsed.type === "ping") {
            ws.send(JSON.stringify({ type: "pong" }))
            return
          }

          if (parsed.type === "bar") {
            const incoming = parsed.bar
            const bar: OHLCVBar = {
              date: incoming.date,
              open: incoming.open,
              high: incoming.high,
              low: incoming.low,
              close: incoming.close,
              volume: incoming.volume,
            }
            pendingBarRef.current = bar
            scheduleFlush(
              flushBars,
              250,
              flushTimeoutRef
            )
            return
          }

          if (parsed.type === "trade") {
            // Server envelope uses top-level `symbol` and trade payload `{date, price, size}`
            setLiveTick(sym, parsed.trade)
            return
          }

          if (parsed.type === "quote") {
            setLiveQuote(sym, parsed.quote)
            return
          }
        } catch {
          // Ignore malformed websocket payloads.
        }
      }

      ws.onerror = () => {
        if (cancelledRef.current) return
        setWsStatus("ERROR")
      }

      ws.onclose = () => {
        if (cancelledRef.current) return
        setWsStatus("DISCONNECTED")
        if (flushTimeoutRef.current != null) {
          window.clearTimeout(flushTimeoutRef.current)
          flushTimeoutRef.current = null
        }
        const delay = backoffMs
        backoffMs = Math.min(30000, backoffMs * 2)
        window.setTimeout(() => connect(), delay)
      }
    }

    // Start with empty live state; snapshot loading happens separately.
    setLiveMinuteBars(sym, [])

    connect()

    return () => {
      cancelledRef.current = true
      wsRef.current?.close()
      wsRef.current = null
      pendingBarRef.current = null
      if (flushTimeoutRef.current != null) {
        window.clearTimeout(flushTimeoutRef.current)
        flushTimeoutRef.current = null
      }
    }
  }, [symbol, enabled, setLiveMinuteBars, setLiveQuote, setLiveTick, setWsStatus, upsertLiveBar])
}

