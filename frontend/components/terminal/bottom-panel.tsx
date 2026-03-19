"use client"

import { useEffect, useRef, useState } from "react"
import { Terminal, Bot, AlertTriangle, ChevronUp, ChevronDown } from "lucide-react"

type LogLevel = "info" | "warn" | "error" | "success" | "debug"
type Tab = "pipeline" | "agent" | "errors"

interface LogEntry {
  id: number
  timestamp: string
  agent: string
  message: string
  level: LogLevel
}

const SEED_PIPELINE_LOGS: LogEntry[] = [
  { id: 1, timestamp: "09:32:01.003", agent: "Orchestrator", message: "Pipeline initialized — config: exp_001_momentum", level: "info" },
  { id: 2, timestamp: "09:32:01.112", agent: "DataAgent", message: "Fetching OHLCV for SPY [2022-01-01 → 2024-12-31]", level: "info" },
  { id: 3, timestamp: "09:32:01.548", agent: "DataAgent", message: "Cache hit for QQQ (v2.4.1) — loading from store", level: "success" },
  { id: 4, timestamp: "09:32:01.771", agent: "DataAgent", message: "AAPL data loaded. Rows: 756 | Missing: 0", level: "success" },
  { id: 5, timestamp: "09:32:02.104", agent: "FeatureAgent", message: "Computing 47 features (PCA dim: 12)", level: "info" },
  { id: 6, timestamp: "09:32:02.832", agent: "FeatureAgent", message: "WARN: Feature 'earnings_surprise' has 3.2% NaN fill rate", level: "warn" },
  { id: 7, timestamp: "09:32:03.210", agent: "FeatureAgent", message: "Feature matrix ready — shape: (756, 47)", level: "success" },
  { id: 8, timestamp: "09:32:03.541", agent: "ModelAgent", message: "Loading LightGBM v3.2 checkpoint", level: "info" },
  { id: 9, timestamp: "09:32:03.884", agent: "ModelAgent", message: "Model loaded. Generating signals...", level: "info" },
  { id: 10, timestamp: "09:32:04.423", agent: "BacktestAgent", message: "Starting backtest — tx_costs=ON, slippage=OFF", level: "info" },
  { id: 11, timestamp: "09:32:05.001", agent: "BacktestAgent", message: "Backtest complete — 756 bars, 94 trades, 58.2% win rate", level: "success" },
  { id: 12, timestamp: "09:32:05.104", agent: "Orchestrator", message: "Pipeline complete in 4.10s", level: "success" },
]

const SEED_AGENT_LOGS: LogEntry[] = [
  { id: 1, timestamp: "09:32:01.548", agent: "DataAgent", message: "[INIT] Connecting to data provider...", level: "debug" },
  { id: 2, timestamp: "09:32:01.602", agent: "DataAgent", message: "[CACHE] SPY: stale (>24h) — refetching", level: "warn" },
  { id: 3, timestamp: "09:32:01.748", agent: "DataAgent", message: "[FETCH] SPY — 252 bars received via WebSocket", level: "success" },
  { id: 4, timestamp: "09:32:02.104", agent: "FeatureAgent", message: "[COMPUTE] RSI(14), MACD(12,26,9), ATR(14)...", level: "debug" },
  { id: 5, timestamp: "09:32:02.512", agent: "FeatureAgent", message: "[COMPUTE] Momentum(5,20,60) done", level: "debug" },
  { id: 6, timestamp: "09:32:02.832", agent: "FeatureAgent", message: "[WARN] earnings_surprise fill rate exceeds 3%", level: "warn" },
  { id: 7, timestamp: "09:32:03.541", agent: "ModelAgent", message: "[LOAD] lgbm_momentum_v3.2.pkl — 847 trees, 12 features", level: "info" },
  { id: 8, timestamp: "09:32:03.901", agent: "ModelAgent", message: "[INFER] Batch predict: 756 samples", level: "info" },
  { id: 9, timestamp: "09:32:04.002", agent: "ModelAgent", message: "[SIGNAL] Generated 94 signals (58 BUY / 36 SELL)", level: "success" },
  { id: 10, timestamp: "09:32:04.423", agent: "BacktestAgent", message: "[EXEC] Simulating order fills with 0.01% commission", level: "info" },
  { id: 11, timestamp: "09:32:04.788", agent: "BacktestAgent", message: "[PERF] Sharpe=1.84 | DD=-12.4% | Return=+34.7%", level: "success" },
]

const SEED_ERROR_LOGS: LogEntry[] = [
  { id: 1, timestamp: "09:32:02.832", agent: "FeatureAgent", message: "WARN: earnings_surprise NaN fill 3.2% > threshold 3.0%", level: "warn" },
  { id: 2, timestamp: "09:32:01.602", agent: "DataAgent", message: "WARN: SPY cache stale (last update: 26h ago)", level: "warn" },
  { id: 3, timestamp: "09:15:44.001", agent: "DataAgent", message: "ERROR: ETH-USD feed timeout after 5000ms — retrying (1/3)", level: "error" },
  { id: 4, timestamp: "09:15:45.204", agent: "DataAgent", message: "ERROR: ETH-USD retry 2/3 failed — partial data loaded", level: "error" },
  { id: 5, timestamp: "09:15:46.882", agent: "DataAgent", message: "WARN: ETH-USD using stale cache from 2024-03-17", level: "warn" },
]

const levelColor: Record<LogLevel, string> = {
  info: "text-[#8B949E]",
  debug: "text-[#8B949E]",
  warn: "text-[#FFD60A]",
  error: "text-[#FF4D4D]",
  success: "text-[#00FF9C]",
}

const levelPrefix: Record<LogLevel, string> = {
  info: "INFO ",
  debug: "DEBUG",
  warn: "WARN ",
  error: "ERROR",
  success: "OK   ",
}

const agentColor: Record<string, string> = {
  Orchestrator: "text-[#4CC9F0]",
  DataAgent: "text-[#FFD60A]",
  FeatureAgent: "text-[#00FF9C]",
  ModelAgent: "text-[#4CC9F0]",
  BacktestAgent: "text-[#FF4D4D]",
}

export function BottomPanel({ running }: { running: boolean }) {
  const [activeTab, setActiveTab] = useState<Tab>("pipeline")
  const [collapsed, setCollapsed] = useState(false)
  const [pipelineLogs, setPipelineLogs] = useState<LogEntry[]>(SEED_PIPELINE_LOGS)
  const scrollRef = useRef<HTMLDivElement>(null)

  const tabs: { id: Tab; label: string; icon: React.ElementType; count: number }[] = [
    { id: "pipeline", label: "Pipeline Logs", icon: Terminal, count: pipelineLogs.length },
    { id: "agent", label: "Agent Logs", icon: Bot, count: SEED_AGENT_LOGS.length },
    { id: "errors", label: "Errors / Warnings", icon: AlertTriangle, count: SEED_ERROR_LOGS.filter((l) => l.level === "error").length },
  ]

  const logs = activeTab === "pipeline" ? pipelineLogs : activeTab === "agent" ? SEED_AGENT_LOGS : SEED_ERROR_LOGS

  useEffect(() => {
    if (running) {
      const newLog: LogEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 }),
        agent: "Orchestrator",
        message: "Pipeline triggered by user — initializing...",
        level: "info",
      }
      setPipelineLogs((prev) => [...prev, newLog])
    }
  }, [running])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className={`bg-[#0B0F14] border-t border-[#1E2A38] flex flex-col shrink-0 transition-all ${collapsed ? "h-8" : "h-40"}`}>
      {/* Tab bar */}
      <div className="flex items-center h-8 border-b border-[#1E2A38] shrink-0">
        {tabs.map((tab) => {
          const Icon = tab.icon
          const isError = tab.id === "errors"
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 h-full text-[10px] uppercase tracking-wider border-r border-[#1E2A38] transition-colors ${
                activeTab === tab.id
                  ? "bg-[#11161C] text-[#E6EDF3]"
                  : "text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#11161C]/50"
              }`}
            >
              <Icon className={`w-3 h-3 ${isError && tab.count > 0 ? "text-[#FF4D4D]" : ""}`} />
              <span>{tab.label}</span>
              <span className={`text-[9px] px-1 rounded-sm ${
                isError && tab.count > 0
                  ? "bg-[#FF4D4D]/20 text-[#FF4D4D]"
                  : "bg-[#1E2A38] text-[#8B949E]"
              }`}>
                {tab.count}
              </span>
            </button>
          )
        })}
        <div className="flex-1" />
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="px-3 h-full text-[#8B949E] hover:text-[#E6EDF3] border-l border-[#1E2A38] transition-colors"
          aria-label={collapsed ? "Expand logs" : "Collapse logs"}
        >
          {collapsed ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>

      {/* Log content */}
      {!collapsed && (
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-1.5 space-y-0.5">
          {logs.map((entry) => (
            <div key={entry.id} className="flex items-start gap-3 text-[10px] font-mono leading-5">
              <span className="text-[#8B949E] tabular-nums shrink-0">{entry.timestamp}</span>
              <span className={`shrink-0 tabular-nums font-bold ${levelColor[entry.level]}`}>
                {levelPrefix[entry.level]}
              </span>
              <span className={`shrink-0 font-semibold ${agentColor[entry.agent] ?? "text-[#8B949E]"}`}>
                [{entry.agent}]
              </span>
              <span className={levelColor[entry.level]}>{entry.message}</span>
            </div>
          ))}
          {running && (
            <div className="flex items-center gap-2 text-[10px] font-mono text-[#4CC9F0]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#4CC9F0] animate-pulse" />
              Processing...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
