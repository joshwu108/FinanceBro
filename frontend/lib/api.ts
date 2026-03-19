const API_BASE_URL="http://localhost:8000";

export interface PipelineConfig {
    symbols: string[]
    start_date: string;
    end_date: string;
    model_type: string;
    transaction_costs_bps: number;
    slippage_bps: number;
    max_position_size: number;
    benchmark: string;
}

export async function runPipeline(config: PipelineConfig) {
    const response = await fetch(`${API_BASE_URL}/run_pipeline`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
    });
    if (!response.ok) {
        throw new Error("Failed to run pipeline");
    }
    return response.json();
}