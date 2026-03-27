"use client"

import { useEffect } from "react"
import { useAppStore } from "@/lib/store"
import { CandlestickChart } from "@/components/charts/candlestick-chart"
import { EquityCurve } from "@/components/charts/equity-curve"
import { DrawdownChart } from "@/components/charts/drawdown-chart"
import { FeatureImportance } from "@/components/charts/feature-importance"
import { ReturnsDistribution } from "@/components/charts/returns-distribution"
import { WalkForwardChart } from "@/components/charts/walkforward-chart"
import { RollingSharpe } from "@/components/charts/rolling-sharpe"
import { PortfolioView } from "@/components/workspace/portfolio-view"
import { ExperimentBrowser } from "@/components/workspace/experiment-browser"
import { TradeLogTable } from "@/components/workspace/trade-log-table"
import { useAlpacaStream } from "@/lib/websocket"
import { PredictionPanel } from "@/components/workspace/prediction-panel"
import { QuantumWorkspace } from "@/components/quantum/quantum-workspace"
import type { EquityCurvePoint, PipelineResponse, PerformanceSummary } from "@/lib/types"
import * as api from "@/lib/api"

// ── Helpers ──────────────────────────────────────────────────────────────────

function computeReturnsFromEquity(curve: EquityCurvePoint[]): number[] {
  if (curve.length < 2) return []
  return curve.slice(1).map((point, i) => (point.value - curve[i].value) / curve[i].value)
}

function buildSymbolResultsForPortfolio(
  result: PipelineResponse | null
): Record<string, { backtest: { performance_summary: PerformanceSummary } }> {
  if (!result?.per_symbol) return {}
  const out: Record<string, { backtest: { performance_summary: PerformanceSummary } }> = {}
  for (const [sym, data] of Object.entries(result.per_symbol)) {
    out[sym] = { backtest: { performance_summary: data.backtest.performance_summary } }
  }
  return out
}

// ── Tab definitions ──────────────────────────────────────────────────────────

const TABS = [
  { id: "price" as const, label: "Price" },
  { id: "equity" as const, label: "Equity" },
  { id: "analytics" as const, label: "Analytics" },
  { id: "portfolio" as const, label: "Portfolio" },
  { id: "experiments" as const, label: "Experiments" },
  { id: "trades" as const, label: "Trades" },
  { id: "predict" as const, label: "Predict" },
  { id: "quantum" as const, label: "Quantum" },
] as const

// ── Main Component ───────────────────────────────────────────────────────────

export function ChartWorkspace() {
  const mode = useAppStore((s) => s.mode)
  const chartTab = useAppStore((s) => s.chartTab)
  const setChartTab = useAppStore((s) => s.setChartTab)
  const activeSymbolResult = useAppStore((s) => s.activeSymbolResult)
  const ohlcvData = useAppStore((s) => s.ohlcvData)
  const liveMinuteBars = useAppStore((s) => s.liveMinuteBars)
  const setLiveMinuteBars = useAppStore((s) => s.setLiveMinuteBars)
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const pipelineResult = useAppStore((s) => s.pipelineResult)
  const experiments = useAppStore((s) => s.experiments)
  const fetchExperiments = useAppStore((s) => s.fetchExperiments)
  const liveSignalBySymbol = useAppStore((s) => s.liveSignalBySymbol)

  useAlpacaStream(activeSymbol, mode === "live")

  useEffect(() => {
    if (mode !== "live") return;
    const existingBars = ohlcvData[activeSymbol] || [];
    if (existingBars.length === 0) {
      api.fetchMarketSnapshot(activeSymbol).then((bars) => {
        useAppStore.setState((state) => ({
          ohlcvData: { ...state.ohlcvData, [activeSymbol]: bars }
        }))
      })
    }
  }, [mode, activeSymbol, ohlcvData])

  const symbolResult = activeSymbolResult()

  const livePriceData =
    liveMinuteBars[activeSymbol] ?? []

  const priceData =
    mode === "live"
      ? ohlcvData[activeSymbol] || []
      : symbolResult?.backtest?.ohlcv ?? ohlcvData[activeSymbol] ?? []

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Tab bar */}
      <div className="flex items-center h-8 border-b border-[#1E2A38] bg-[#11161C] shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setChartTab(tab.id)}
            className={`px-3 h-full text-[10px] uppercase tracking-wider border-r border-[#1E2A38] transition-colors ${
              chartTab === tab.id
                ? "bg-[#0B0F14] text-[#E6EDF3] font-semibold"
                : "text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#11161C]/50"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {chartTab === "price" && (
          <div className="w-full h-full relative">
            <CandlestickChart
              data={priceData}
              trades={mode === "live" ? [] : symbolResult?.backtest?.trade_log}
              showMA20
              showMA50
              showVolume
            />
            
            {/* Live Agent Signal Indicator */}
            {mode === "live" && liveSignalBySymbol[activeSymbol] && (
              <div className="absolute top-4 right-4 z-10 bg-[#0B0F14]/90 border border-[#1E2A38] p-3 rounded-md backdrop-blur-sm shadow-xl min-w-[140px]">
                <div className="text-[10px] text-[#8B949E] uppercase tracking-widest mb-1.5 font-bold">
                  Live Agent Signal
                </div>
                <div className="flex items-center gap-3">
                  <div className={`text-xl font-black tracking-tighter ${
                    liveSignalBySymbol[activeSymbol]?.signal === "LONG" ? "text-[#00C076]" :
                    liveSignalBySymbol[activeSymbol]?.signal === "SHORT" ? "text-[#FF4D4D]" :
                    "text-[#8B949E]"
                  }`}>
                    {liveSignalBySymbol[activeSymbol]?.signal}
                  </div>
                  {liveSignalBySymbol[activeSymbol]?.confidence !== undefined && (
                    <div className="flex flex-col">
                      <div className="text-[8px] text-[#8B949E] uppercase leading-none mb-0.5">Confidence</div>
                      <div className="text-xs font-mono text-[#E6EDF3]">
                        {(liveSignalBySymbol[activeSymbol]?.confidence! * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>
                <div className="mt-2 h-1 w-full bg-[#1E2A38] rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      liveSignalBySymbol[activeSymbol]?.signal === "LONG" ? "bg-[#00C076]" :
                      liveSignalBySymbol[activeSymbol]?.signal === "SHORT" ? "bg-[#FF4D4D]" :
                      "bg-[#8B949E]"
                    }`}
                    style={{ width: `${(liveSignalBySymbol[activeSymbol]?.confidence ?? 0.5) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {chartTab === "equity" && (
          <div className="w-full h-full">
            <EquityCurve
              equityCurve={symbolResult?.backtest?.equity_curve ?? []}
            />
          </div>
        )}

        {chartTab === "analytics" && (
          <div className="grid grid-cols-2 gap-2 p-2 overflow-auto h-full">
            <DrawdownChart
              equityCurve={symbolResult?.backtest?.equity_curve ?? []}
              height={200}
            />
            <FeatureImportance
              importances={symbolResult?.model?.feature_importances ?? {}}
              height={200}
            />
            <WalkForwardChart
              foldResults={symbolResult?.walk_forward?.fold_results ?? []}
              height={200}
            />
            <RollingSharpe
              equityCurve={symbolResult?.backtest?.equity_curve ?? []}
              height={200}
            />
            <ReturnsDistribution
              returns={computeReturnsFromEquity(symbolResult?.backtest?.equity_curve ?? [])}
              height={200}
            />
          </div>
        )}

        {chartTab === "portfolio" && (
          <PortfolioView
            portfolio={pipelineResult?.portfolio ?? null}
            symbolResults={buildSymbolResultsForPortfolio(pipelineResult)}
          />
        )}

        {chartTab === "experiments" && (
          <ExperimentBrowser
            experiments={experiments}
            onRefresh={fetchExperiments}
          />
        )}

        {chartTab === "trades" && (
          <TradeLogTable trades={symbolResult?.backtest?.trade_log ?? []} />
        )}

        {chartTab === "predict" && (
          <PredictionPanel />
        )}

        {chartTab === "quantum" && (
          <QuantumWorkspace />
        )}
      </div>
    </div>
  )
}
