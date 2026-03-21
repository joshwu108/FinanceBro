import type {
  PipelineConfig,
  PipelineResponse,
  OHLCVBar,
  ExperimentSummary,
  PredictResponse,
} from "./types"

const API_BASE = "http://localhost:8000"

export async function runPipeline(
  config: PipelineConfig
): Promise<PipelineResponse> {
  const response = await fetch(`${API_BASE}/run_pipeline`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  })
  if (!response.ok) {
    let message = "Unknown error"
    try {
      const body = await response.json()
      const detail = body?.detail
      if (typeof detail === "string") message = detail
      else if (detail?.message) message = detail.message
      else message = JSON.stringify(detail ?? body)
    } catch {
      const errorText = await response.text().catch(() => "Unknown error")
      message = errorText
    }
    throw new Error(`Failed to run pipeline: ${response.status} ${message}`)
  }
  // Backend wraps in { status: "success", data: {...} }
  const json = await response.json()
  return json.data ?? json
}

export async function fetchOHLCV(
  symbol: string,
  startDate?: string,
  endDate?: string
): Promise<OHLCVBar[]> {
  const params = new URLSearchParams({ symbol })
  if (startDate) params.set("start_date", startDate)
  if (endDate) params.set("end_date", endDate)

  const response = await fetch(`${API_BASE}/api/data/${symbol}?${params}`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(
      `Failed to fetch OHLCV for ${symbol}: ${response.status} ${errorText}`
    )
  }
  // Backend wraps in { symbol, data: [...] }
  const json = await response.json()
  return json.data ?? json
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${API_BASE}/api/experiments`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(
      `Failed to fetch experiments: ${response.status} ${errorText}`
    )
  }
  // Backend wraps in { experiments: [...] }
  const json = await response.json()
  return json.experiments ?? json
}

export async function fetchExperiment(runId: string): Promise<any> {
  const response = await fetch(`${API_BASE}/api/experiments/${runId}`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(
      `Failed to fetch experiment ${runId}: ${response.status} ${errorText}`
    )
  }
  return response.json()
}

export async function predict(
  symbol: string,
  refreshData?: boolean
): Promise<PredictResponse> {
  const params = new URLSearchParams({ symbol })
  if (refreshData) params.set("refresh_data", "true")

  const response = await fetch(`${API_BASE}/api/predict?${params}`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(
      `Failed to predict for ${symbol}: ${response.status} ${errorText}`
    )
  }
  return response.json()
}

export async function fetchMarketSnapshot(symbol: string): Promise<OHLCVBar[]> {
  const sym = symbol.trim().toUpperCase()
  const response = await fetch(`${API_BASE}/api/market/snapshot/${sym}`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(`Failed to fetch market snapshot for ${sym}: ${response.status} ${errorText}`)
  }
  const json = await response.json()
  const bars = json?.bars ?? []
  return bars.map((b: any) => ({
    date: b.date,
    open: Number(b.open),
    high: Number(b.high),
    low: Number(b.low),
    close: Number(b.close),
    volume: Number(b.volume),
  }))
}
