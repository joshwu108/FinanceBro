import { create } from "zustand"
import type {
  PipelineResponse,
  PipelineConfig,
  LogEntry,
  ExperimentSummary,
  OHLCVBar,
  SymbolResult,
} from "./types"
import * as api from "./api"

type Mode = "research" | "live"
type Status = "IDLE" | "BACKTEST" | "LIVE"
type ChartTab = "price" | "equity" | "analytics" | "portfolio" | "experiments" | "trades"

interface AppState {
  // UI state
  mode: Mode
  setMode: (mode: Mode) => void
  activeSymbol: string
  setActiveSymbol: (symbol: string) => void
  chartTab: ChartTab
  setChartTab: (tab: ChartTab) => void

  // Pipeline state
  status: Status
  running: boolean
  pipelineResult: PipelineResponse | null

  // Data
  ohlcvData: Record<string, OHLCVBar[]>
  experiments: ExperimentSummary[]

  // Logs
  logs: LogEntry[]
  addLog: (entry: Omit<LogEntry, "id">) => void
  clearLogs: () => void

  // Actions
  runPipeline: (config: PipelineConfig) => Promise<void>
  fetchExperiments: () => Promise<void>
  fetchOHLCV: (
    symbol: string,
    startDate?: string,
    endDate?: string
  ) => Promise<void>

  // Computed helpers
  activeSymbolResult: () => SymbolResult | null
}

let nextLogId = 1

function createLogEntry(
  agent: string,
  message: string,
  level: LogEntry["level"]
): Omit<LogEntry, "id"> {
  return {
    timestamp: new Date().toISOString(),
    agent,
    message,
    level,
  }
}

export const useAppStore = create<AppState>((set, get) => ({
  // UI state
  mode: "research",
  setMode: (mode) => set({ mode }),
  activeSymbol: "AAPL",
  setActiveSymbol: (symbol) => set({ activeSymbol: symbol }),
  chartTab: "price",
  setChartTab: (tab) => set({ chartTab: tab }),

  // Pipeline state
  status: "IDLE",
  running: false,
  pipelineResult: null,

  // Data
  ohlcvData: {},
  experiments: [],

  // Logs
  logs: [],
  addLog: (entry) =>
    set((state) => ({
      logs: [...state.logs, { ...entry, id: nextLogId++ }],
    })),
  clearLogs: () => set({ logs: [] }),

  // Actions
  runPipeline: async (config) => {
    const { addLog } = get()

    set({ status: "BACKTEST", running: true })
    addLog(
      createLogEntry(
        "orchestrator",
        `Starting pipeline for ${config.symbols.join(", ")}`,
        "info"
      )
    )

    try {
      const result = await api.runPipeline(config)

      // Cache OHLCV data from backtest results
      const updatedOhlcv = { ...get().ohlcvData }
      for (const [symbol, symbolResult] of Object.entries(result.per_symbol)) {
        if (symbolResult.backtest?.ohlcv) {
          updatedOhlcv[symbol] = symbolResult.backtest.ohlcv
        }
      }

      set({
        pipelineResult: result,
        ohlcvData: updatedOhlcv,
        activeSymbol: config.symbols[0] ?? get().activeSymbol,
        status: "IDLE",
        running: false,
      })

      addLog(
        createLogEntry(
          "orchestrator",
          `Pipeline complete: run_id=${result.run_id}`,
          "success"
        )
      )
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown pipeline error"
      console.error("Pipeline error:", error)

      set({ status: "IDLE", running: false })
      addLog(createLogEntry("orchestrator", `Pipeline failed: ${message}`, "error"))
    }
  },

  fetchExperiments: async () => {
    const { addLog } = get()
    try {
      const experiments = await api.fetchExperiments()
      set({ experiments })
      addLog(
        createLogEntry(
          "experiments",
          `Loaded ${experiments.length} experiments`,
          "info"
        )
      )
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error"
      console.error("Failed to fetch experiments:", error)
      addLog(
        createLogEntry(
          "experiments",
          `Failed to load experiments: ${message}`,
          "error"
        )
      )
    }
  },

  fetchOHLCV: async (symbol, startDate?, endDate?) => {
    const { addLog } = get()
    try {
      const data = await api.fetchOHLCV(symbol, startDate, endDate)
      set((state) => ({
        ohlcvData: { ...state.ohlcvData, [symbol]: data },
      }))
      addLog(
        createLogEntry(
          "data",
          `Loaded ${data.length} bars for ${symbol}`,
          "info"
        )
      )
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error"
      console.error(`Failed to fetch OHLCV for ${symbol}:`, error)
      addLog(
        createLogEntry(
          "data",
          `Failed to load OHLCV for ${symbol}: ${message}`,
          "error"
        )
      )
    }
  },

  // Computed helpers
  activeSymbolResult: () => {
    const { pipelineResult, activeSymbol } = get()
    if (!pipelineResult?.per_symbol) return null
    return pipelineResult.per_symbol[activeSymbol] ?? null
  },
}))
