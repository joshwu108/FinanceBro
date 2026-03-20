"use client"

import { useState, useEffect } from "react"
import { ChevronDown, Activity, Wifi, WifiOff, Clock } from "lucide-react"
import { useAppStore } from "@/lib/store"

const SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "BTC-USD", "ETH-USD"]

type Status = "LIVE" | "BACKTEST" | "CONNECTING" | "DISCONNECTED" | "ERROR"

export function TopNav() {
  const mode = useAppStore((s) => s.mode)
  const setMode = useAppStore((s) => s.setMode)
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const setActiveSymbol = useAppStore((s) => s.setActiveSymbol)
  const running = useAppStore((s) => s.running)
  const wsStatus = useAppStore((s) => s.wsStatus)

  const [symbolOpen, setSymbolOpen] = useState(false)
  const [now, setNow] = useState<Date | null>(null)

  useEffect(() => {
    setNow(new Date())
    const interval = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(interval)
  }, [])

  const status: Status =
    mode === "live"
      ? wsStatus === "CONNECTED"
        ? "LIVE"
        : wsStatus === "CONNECTING"
          ? "CONNECTING"
          : wsStatus === "ERROR"
            ? "ERROR"
            : "DISCONNECTED"
      : running
        ? "BACKTEST"
        : "DISCONNECTED"

  const statusColors: Record<Status, string> = {
    LIVE: "text-[#00FF9C]",
    BACKTEST: "text-[#FFD60A]",
    CONNECTING: "text-[#FFD60A]",
    DISCONNECTED: "text-[#FF4D4D]",
    ERROR: "text-[#FF4D4D]",
  }

  const statusDot: Record<Status, string> = {
    LIVE: "bg-[#00FF9C] animate-pulse",
    BACKTEST: "bg-[#FFD60A]",
    CONNECTING: "bg-[#FFD60A] animate-pulse",
    DISCONNECTED: "bg-[#FF4D4D]",
    ERROR: "bg-[#FF4D4D]",
  }

  return (
    <header className="h-10 bg-[#11161C] border-b border-[#1E2A38] flex items-center px-3 gap-4 shrink-0 z-50">
      {/* Brand */}
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-[#00FF9C]" />
        <span className="text-[#E6EDF3] text-xs font-bold tracking-widest uppercase">
          FinanceBro<span className="text-[#00FF9C]">Terminal</span>
        </span>
      </div>

      <div className="w-px h-5 bg-[#1E2A38]" />

      {/* Mode toggle */}
      <div className="flex items-center bg-[#0B0F14] border border-[#1E2A38] rounded-sm overflow-hidden">
        <button
          onClick={() => setMode("research")}
          className={`px-3 py-1 text-[10px] uppercase tracking-wider font-semibold transition-colors ${
            mode === "research"
              ? "bg-[#4CC9F0] text-[#0B0F14]"
              : "text-[#8B949E] hover:text-[#E6EDF3]"
          }`}
        >
          Research
        </button>
        <button
          onClick={() => setMode("live")}
          className={`px-3 py-1 text-[10px] uppercase tracking-wider font-semibold transition-colors ${
            mode === "live"
              ? "bg-[#00FF9C] text-[#0B0F14]"
              : "text-[#8B949E] hover:text-[#E6EDF3]"
          }`}
        >
          Live
        </button>
      </div>

      <div className="w-px h-5 bg-[#1E2A38]" />

      {/* Symbol selector */}
      <div className="relative">
        <button
          onClick={() => setSymbolOpen(!symbolOpen)}
          className="flex items-center gap-2 bg-[#0B0F14] border border-[#1E2A38] rounded-sm px-3 py-1 text-xs text-[#E6EDF3] hover:border-[#4CC9F0] transition-colors"
        >
          <span className="text-[#4CC9F0] font-bold">{activeSymbol}</span>
          <ChevronDown className="w-3 h-3 text-[#8B949E]" />
        </button>
        {symbolOpen && (
          <div className="absolute top-full left-0 mt-1 bg-[#11161C] border border-[#1E2A38] rounded-sm shadow-xl z-50 min-w-[100px]">
            {SYMBOLS.map((sym) => (
              <button
                key={sym}
                onClick={() => { setActiveSymbol(sym); setSymbolOpen(false) }}
                className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-[#1A2130] transition-colors ${
                  sym === activeSymbol ? "text-[#4CC9F0]" : "text-[#E6EDF3]"
                }`}
              >
                {sym}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1" />

      {/* Clock */}
      <div className="flex items-center gap-1.5 text-[10px] text-[#8B949E] tabular-nums">
        <Clock className="w-3 h-3" />
        <span className="min-w-[140px]">
          {now
            ? `${now.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })} ${now.toLocaleTimeString("en-US", { hour12: false })}`
            : "Loading..."}
        </span>
      </div>

      <div className="w-px h-5 bg-[#1E2A38]" />

      {/* Status */}
      <div className="flex items-center gap-2">
        <div className={`w-1.5 h-1.5 rounded-full ${statusDot[status]}`} />
        <span className={`text-[10px] font-bold tracking-widest uppercase ${statusColors[status]}`}>
          {status}
        </span>
        {status === "LIVE" ? (
          <Wifi className="w-3 h-3 text-[#00FF9C]" />
        ) : status === "CONNECTING" ? (
          <Wifi className="w-3 h-3 text-[#FFD60A]" />
        ) : (
          <WifiOff className="w-3 h-3 text-[#8B949E]" />
        )}
      </div>
    </header>
  )
}
