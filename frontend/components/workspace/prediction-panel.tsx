"use client"

import { useState } from "react"
import { useAppStore } from "@/lib/store"
import * as api from "@/lib/api"
import type { PredictResponse } from "@/lib/types"

export function PredictionPanel() {
  const activeSymbol = useAppStore((s) => s.activeSymbol)
  const addLog = useAppStore((s) => s.addLog)

  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handlePredict() {
    setLoading(true)
    setError(null)
    try {
      const res = await api.predict(activeSymbol, true)
      setResult(res)
      addLog({
        timestamp: new Date().toISOString(),
        agent: "predict",
        message: `Prediction for ${activeSymbol}: ${res.signal} (${res.model_type ?? "unknown model"})`,
        level: "success",
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error"
      setError(msg)
      addLog({
        timestamp: new Date().toISOString(),
        agent: "predict",
        message: `Prediction failed for ${activeSymbol}: ${msg}`,
        level: "error",
      })
    } finally {
      setLoading(false)
    }
  }

  const signalColor = result?.signal === "LONG"
    ? "text-[#00FF9C]"
    : result?.signal === "SHORT"
      ? "text-[#FF4D4D]"
      : "text-[#8B949E]"

  return (
    <div className="h-full flex flex-col items-center justify-center gap-6 p-6">
      <div className="bg-[#0B0F14] border border-[#1E2A38] rounded-sm p-6 w-full max-w-md">
        <div className="text-[9px] text-[#4CC9F0] font-bold uppercase tracking-widest mb-4 text-center">
          Model Prediction
        </div>

        <div className="text-center mb-4">
          <span className="text-2xl font-black text-[#4CC9F0]">{activeSymbol}</span>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full bg-[#4CC9F0] text-[#0B0F14] rounded-sm py-2 text-[11px] font-bold uppercase tracking-wider hover:bg-[#3AB8DF] transition-colors disabled:opacity-50 mb-4"
        >
          {loading ? "Predicting..." : "Run Prediction"}
        </button>

        {error && (
          <div className="bg-[#FF4D4D]/10 border border-[#FF4D4D]/30 rounded-sm px-3 py-2 mb-4">
            <span className="text-[10px] text-[#FF4D4D]">{error}</span>
          </div>
        )}

        {result && (
          <div className="space-y-3">
            <div className="flex justify-between items-center border-b border-[#1E2A38] py-2">
              <span className="text-[10px] text-[#8B949E]">Signal</span>
              <span className={`text-xl font-black ${signalColor}`}>{result.signal}</span>
            </div>
            <div className="flex justify-between items-center border-b border-[#1E2A38] py-2">
              <span className="text-[10px] text-[#8B949E]">Prediction</span>
              <span className="text-[11px] font-bold font-mono text-[#E6EDF3]">{result.prediction.toFixed(6)}</span>
            </div>
            <div className="flex justify-between items-center border-b border-[#1E2A38] py-2">
              <span className="text-[10px] text-[#8B949E]">Model</span>
              <span className="text-[11px] font-mono text-[#E6EDF3]">{result.model_type ?? "N/A"}</span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-[10px] text-[#8B949E]">Timestamp</span>
              <span className="text-[10px] font-mono text-[#8B949E]">{result.timestamp}</span>
            </div>
          </div>
        )}
      </div>

      <div className="text-[9px] text-[#8B949E] text-center max-w-sm">
        Runs the trained model pipeline for {activeSymbol} and generates a LONG/SHORT/NEUTRAL signal.
        Requires a prior pipeline run to build the model.
      </div>
    </div>
  )
}
