"use client"

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
import type { EquityCurvePoint, PipelineResponse, PerformanceSummary } from "@/lib/types"

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
] as const

// ── Main Component ───────────────────────────────────────────────────────────

export function ChartWorkspace() {
  const chartTab = useAppStore((s) => s.chartTab)
  const setChartTab = useAppStore((s) => s.setChartTab)
  const activeSymbolResult = useAppStore((s) => s.activeSymbolResult)
  const ohlcvData = useAppStore((s) => s.ohlcvData)
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const pipelineResult = useAppStore((s) => s.pipelineResult)
  const experiments = useAppStore((s) => s.experiments)
  const fetchExperiments = useAppStore((s) => s.fetchExperiments)

  const symbolResult = activeSymbolResult()

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
      <div className="flex-1 min-h-0 overflow-auto">
        {chartTab === "price" && (
          <CandlestickChart
            data={symbolResult?.backtest?.ohlcv ?? ohlcvData[activeSymbol] ?? []}
            trades={symbolResult?.backtest?.trade_log}
            showMA20
            showMA50
            showVolume
          />
        )}

        {chartTab === "equity" && (
          <EquityCurve
            equityCurve={symbolResult?.backtest?.equity_curve ?? []}
          />
        )}

        {chartTab === "analytics" && (
          <div className="grid grid-cols-2 gap-2 p-2">
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
      </div>
    </div>
  )
}
