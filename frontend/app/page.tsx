"use client"

import { useState } from "react"
import { TopNav } from "@/components/terminal/top-nav"
import { LeftSidebar } from "@/components/terminal/left-sidebar"
import { MainPanel } from "@/components/terminal/main-panel"
import { RightSidebar } from "@/components/terminal/right-sidebar"
import { BottomPanel } from "@/components/terminal/bottom-panel"
import { runPipeline } from "@/lib/api"
import type { PipelineConfig } from "@/lib/types"

type Mode = "research" | "live"

export default function TerminalPage() {
  const [mode, setMode] = useState<Mode>("research")
  const [symbol, setSymbol] = useState("SPY")
  const [running, setRunning] = useState(false)
  const [showMA, setShowMA] = useState(true)
  const [showVolBands, setShowVolBands] = useState(false)

  const status = mode === "live" ? "LIVE" : running ? "BACKTEST" : "IDLE"

  async function handleRunPipeline() {
    setRunning(true)
    try {
      const config: PipelineConfig = {
        symbols: [symbol],
        start_date: "2020-01-01",
        end_date: "2023-01-01",
        model_type: "random_forest",
        transaction_costs_bps: 5.0,
        slippage_bps: 2.0,
        max_position_size: 0.1,
        benchmark: "SPY"
      }
      
      const response = await runPipeline(config)
      console.log("Pipeline Results:", response)
      alert("Pipeline completed successfully! Check browser console for full output.")
    } catch (error) {
      console.error("Pipeline failed:", error)
      alert("Failed to run pipeline. Is the backend running on port 8000?")
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0B0F14] text-[#E6EDF3] overflow-hidden">
      {/* Top Nav */}
      <TopNav
        mode={mode}
        onModeChange={setMode}
        activeSymbol={symbol}
        onSymbolChange={setSymbol}
        status={status as "LIVE" | "BACKTEST" | "IDLE"}
      />

      {/* Main body: sidebars + main panel */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left sidebar */}
        <LeftSidebar onRunPipeline={handleRunPipeline} running={running} />

        {/* Center charts */}
        <MainPanel
          symbol={symbol}
          showMA={showMA}
          onToggleMA={() => setShowMA((v) => !v)}
          showVolBands={showVolBands}
          onToggleVolBands={() => setShowVolBands((v) => !v)}
        />

        {/* Right sidebar */}
        <RightSidebar mode={mode} />
      </div>

      {/* Bottom logs panel */}
      <BottomPanel running={running} />
    </div>
  )
}
