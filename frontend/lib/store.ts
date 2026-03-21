import { create } from "zustand"
import type {
  PipelineResponse,
  PipelineConfig,
  LogEntry,
  ExperimentSummary,
  OHLCVBar,
  SymbolResult,
  LiveTick,
  LiveQuote,
  WSStatus,
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

  // Live data (Alpaca stream)
  wsStatus: WSStatus
  wsError: string | null
  wsLastUpdatedAtBySymbol: Record<string, number | null>
  liveTickBySymbol: Record<string, LiveTick | null>
  liveQuoteBySymbol: Record<string, LiveQuote | null>
  liveMinuteBars: Record<string, OHLCVBar[]>
  setWsStatus: (status: WSStatus, error?: string | null) => void
  setLiveMinuteBars: (symbol: string, bars: OHLCVBar[]) => void
  upsertLiveBar: (symbol: string, bar: OHLCVBar) => void
  setLiveTick: (symbol: string, tick: LiveTick | null) => void
  setLiveQuote: (symbol: string, quote: LiveQuote | null) => void

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

  // Live data
  wsStatus: "DISCONNECTED",
  wsError: null,
  wsLastUpdatedAtBySymbol: {},
  liveTickBySymbol: {},
  liveQuoteBySymbol: {},
  liveMinuteBars: {},
  setWsStatus: (status, error = null) => set({ wsStatus: status, wsError: error }),
  setLiveMinuteBars: (symbol, bars) =>
    set((state) => {
      const sym = symbol.trim().toUpperCase()
      const capped = (bars ?? []).slice(-500)
      return {
        liveMinuteBars: { ...state.liveMinuteBars, [sym]: capped },
        wsLastUpdatedAtBySymbol: { ...state.wsLastUpdatedAtBySymbol, [sym]: Date.now() },
      }
    }),
  upsertLiveBar: (symbol, bar) =>
    set((state) => {
      const sym = symbol.trim().toUpperCase()
      const existing = state.liveMinuteBars[sym] ?? []

      const barT = Date.parse(bar.date)
      let next = existing

      if (existing.length === 0) {
        next = [bar]
      } else {
        const last = existing[existing.length - 1]
        const lastT = Date.parse(last.date)
        if (barT === lastT) {
          next = [...existing.slice(0, -1), bar]
        } else if (barT > lastT) {
          next = [...existing, bar]
        } else {
          const idx = existing.findIndex((b) => Date.parse(b.date) === barT)
          if (idx >= 0) {
            next = [...existing]
            next[idx] = bar
          } else {
            next = [...existing, bar].sort(
              (a, b) => Date.parse(a.date) - Date.parse(b.date)
            )
          }
        }
      }

      const capped = next.slice(-500)
      return {
        liveMinuteBars: { ...state.liveMinuteBars, [sym]: capped },
        wsLastUpdatedAtBySymbol: {
          ...state.wsLastUpdatedAtBySymbol,
          [sym]: Date.now(),
        },
      }
    }),
  setLiveTick: (symbol, tick) =>
    set((state) => {
      const sym = symbol.trim().toUpperCase()
      return { liveTickBySymbol: { ...state.liveTickBySymbol, [sym]: tick } }
    }),
  setLiveQuote: (symbol, quote) =>
    set((state) => {
      const sym = symbol.trim().toUpperCase()
      return { liveQuoteBySymbol: { ...state.liveQuoteBySymbol, [sym]: quote } }
    }),

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
      if (result.per_symbol) {
        for (const [symbol, symbolResult] of Object.entries(result.per_symbol)) {
          if (symbolResult.backtest?.ohlcv) {
            updatedOhlcv[symbol] = symbolResult.backtest.ohlcv
          }
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
