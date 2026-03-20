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
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(`Failed to run pipeline: ${response.status} ${errorText}`)
  }
  return response.json()
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
  return response.json()
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${API_BASE}/api/experiments`)
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error")
    throw new Error(
      `Failed to fetch experiments: ${response.status} ${errorText}`
    )
  }
  return response.json()
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
