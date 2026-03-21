"use client"

import { useEffect, useRef, useState, useMemo } from "react"
import { Terminal, AlertTriangle, ChevronUp, ChevronDown } from "lucide-react"
import { useAppStore } from "@/lib/store"

type Tab = "pipeline" | "errors"
type LogLevel = "info" | "warn" | "error" | "success" | "debug"

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

const AGENT_COLORS: Record<string, string> = {
  orchestrator: "text-[#4CC9F0]",
  data: "text-[#FFD60A]",
  features: "text-[#00FF9C]",
  model: "text-[#4CC9F0]",
  backtest: "text-[#FF4D4D]",
  experiments: "text-[#FFD60A]",
}

export function BottomPanel() {
  const logs = useAppStore((s) => s.logs)
  const running = useAppStore((s) => s.running)

  const [activeTab, setActiveTab] = useState<Tab>("pipeline")
  const [collapsed, setCollapsed] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const errorLogs = useMemo(
    () => logs.filter((l) => l.level === "error" || l.level === "warn"),
    [logs]
  )

  const visibleLogs = activeTab === "pipeline" ? logs : errorLogs

  const tabs: { id: Tab; label: string; icon: React.ElementType; count: number }[] = [
    { id: "pipeline", label: "Pipeline Logs", icon: Terminal, count: logs.length },
    { id: "errors", label: "Errors / Warnings", icon: AlertTriangle, count: errorLogs.filter((l) => l.level === "error").length },
  ]

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [visibleLogs])

  return (
    <div className={`h-full bg-[#0B0F14] border-t border-[#1E2A38] flex flex-col`}>
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
          {visibleLogs.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[10px] text-[#8B949E] font-mono">
              {activeTab === "pipeline" ? "No logs yet. Run the pipeline to see output." : "No errors or warnings."}
            </div>
          ) : (
            visibleLogs.map((entry) => (
              <div key={entry.id} className="flex items-start gap-3 text-[10px] font-mono leading-5">
                <span className="text-[#8B949E] tabular-nums shrink-0">
                  {entry.timestamp.includes("T") ? new Date(entry.timestamp).toLocaleTimeString("en-US", { hour12: false }) : entry.timestamp}
                </span>
                <span className={`shrink-0 tabular-nums font-bold ${levelColor[entry.level]}`}>
                  {levelPrefix[entry.level]}
                </span>
                <span className={`shrink-0 font-semibold ${AGENT_COLORS[entry.agent] ?? "text-[#8B949E]"}`}>
                  [{entry.agent}]
                </span>
                <span className={levelColor[entry.level]}>{entry.message}</span>
              </div>
            ))
          )}
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
