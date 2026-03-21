"""Pipeline Orchestrator — Chains all agents into the core quant pipeline.

Pipeline order (from Agents.MD):
    DataAgent → FeatureAgent → ModelAgent → WalkForwardAgent →
    BacktestAgent → OverfittingAgent → RiskAgent → PortfolioAgent →
    StatsAgent

Design decisions:
  - WalkForwardAgent runs first for OOS validation (trains models per fold)
  - A separate ModelAgent run on the temporal split produces the "final"
    model and OOS predictions for the full backtest
  - OverfittingAgent uses both the WalkForward fold results AND the
    final model's train/test gap
  - Multi-symbol: each symbol goes through Feature → Model → WF → Backtest
    independently; PortfolioAgent combines them at the end
  - Inference bundle saved per symbol for the /predict endpoint
  - All results logged to experiments/ as JSON
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.overfitting_agent import OverfittingAgent
from agents.portfolio_agent import PortfolioAgent
from agents.risk_agent import RiskAgent
from agents.stats_agent import StatsAgent
from agents.walkforward_agent import WalkForwardAgent
from api.serializers import (
    serialize_equity_curve,
    serialize_ohlcv,
    serialize_trade_log,
    serialize_weights,
)

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_ROOT / "models"
EXPERIMENTS_DIR = BACKEND_ROOT / "experiments"

# Default signal threshold: P(up) >= 0.5 → long, P(up) < 0.5 → short
_DEFAULT_SIGNAL_THRESHOLD = 0.5


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full quant research pipeline.

    Args:
        config: dict with keys:
            - symbols: list[str] — ticker symbols (required)
            - start_date: str — ISO date (optional)
            - end_date: str — ISO date (optional)
            - model_type: str — "logistic_regression" or "random_forest"
            - transaction_costs_bps: float (default 5.0)
            - slippage_bps: float (default 2.0)
            - max_position_size: float (default 0.1)
            - benchmark: str (default "SPY")
            - n_folds: int (default 5) — walk-forward folds
            - signal_threshold: float (default 0.5)

    Returns:
        dict with per-symbol results and aggregate portfolio metrics.
    """
    run_id = uuid.uuid4().hex[:12]
    symbols = config.get("symbols", [])
    if not symbols:
        raise ValueError("config must contain non-empty 'symbols' list")

    model_type = config.get("model_type", "logistic_regression")
    tx_cost_bps = config.get("transaction_costs_bps", 5.0)
    slippage_bps = config.get("slippage_bps", 2.0)
    max_position_size = config.get("max_position_size", 0.1)
    benchmark_symbol = config.get("benchmark", "SPY")
    n_folds = config.get("n_folds", 5)
    signal_threshold = config.get("signal_threshold", _DEFAULT_SIGNAL_THRESHOLD)

    backtest_config = {
        "transaction_cost_bps": tx_cost_bps,
        "slippage_bps": slippage_bps,
        "max_position_size": max_position_size,
    }

    model_config = {"model_type": model_type}

    logger.info(
        "Pipeline started: run_id=%s, symbols=%s, model=%s",
        run_id, symbols, model_type,
    )

    # ── Step 1: DataAgent — fetch and clean all symbols ──────────
    data_agent = DataAgent()
    data_inputs = {"symbols": symbols}
    if config.get("start_date"):
        data_inputs["start_date"] = config["start_date"]
    if config.get("end_date"):
        data_inputs["end_date"] = config["end_date"]

    data_outputs = data_agent.run(data_inputs)
    cleaned_data = data_outputs["cleaned_data"]
    data_quality = data_outputs["data_quality_report"]

    fetched_symbols = list(cleaned_data.keys())
    if not fetched_symbols:
        raise ValueError("DataAgent returned no usable symbols")

    logger.info(
        "DataAgent complete: %d/%d symbols fetched",
        len(fetched_symbols), len(symbols),
    )

    # ── Fetch benchmark data (for comparison) ────────────────────
    benchmark_data = _fetch_benchmark(benchmark_symbol, config, cleaned_data)

    # ── Steps 2–6: Per-symbol pipeline ───────────────────────────
    per_symbol_results: Dict[str, Dict[str, Any]] = {}
    agents_to_log: List[Any] = [data_agent]

    for symbol in fetched_symbols:
        logger.info("Processing symbol: %s", symbol)
        symbol_result = _run_symbol_pipeline(
            symbol=symbol,
            price_data=cleaned_data[symbol],
            model_config=model_config,
            backtest_config=backtest_config,
            n_folds=n_folds,
            signal_threshold=signal_threshold,
            benchmark_data=benchmark_data,
            agents_to_log=agents_to_log,
            include_ohlcv=True,
        )
        per_symbol_results[symbol] = symbol_result

    # ── Step 7: PortfolioAgent — multi-asset allocation ──────────
    portfolio_result = _run_portfolio(
        per_symbol_results=per_symbol_results,
        cleaned_data=cleaned_data,
        tx_cost_bps=tx_cost_bps,
        agents_to_log=agents_to_log,
    )

    # ── Step 8: StatsAgent — statistical significance ────────────
    stats_result = _run_stats(
        per_symbol_results=per_symbol_results,
        portfolio_result=portfolio_result,
        benchmark_data=benchmark_data,
        agents_to_log=agents_to_log,
    )

    # ── Log all agent metrics ────────────────────────────────────
    for agent in agents_to_log:
        try:
            agent.log_metrics()
        except Exception as exc:
            logger.warning("Failed to log metrics for %s: %s", type(agent).__name__, exc)

    # ── Build final results ──────────────────────────────────────
    results = _build_results(
        run_id=run_id,
        config=config,
        data_quality=data_quality,
        per_symbol_results=per_symbol_results,
        portfolio_result=portfolio_result,
        stats_result=stats_result,
    )

    # ── Save experiment log ──────────────────────────────────────
    _log_experiment(run_id, config, results)

    # ── Optional: persist to database ────────────────────────────
    _try_db_insert(results)

    logger.info("Pipeline complete: run_id=%s", run_id)

    return results


# ── Per-symbol pipeline ──────────────────────────────────────────


def _run_symbol_pipeline(
    symbol: str,
    price_data: pd.DataFrame,
    model_config: Dict[str, Any],
    backtest_config: Dict[str, Any],
    n_folds: int,
    signal_threshold: float,
    benchmark_data: Optional[pd.DataFrame],
    agents_to_log: List[Any],
    include_ohlcv: bool = False,
) -> Dict[str, Any]:
    """Run the full pipeline for a single symbol.

    Returns dict with feature_metadata, model, walkforward, backtest,
    overfitting, risk results, plus serialized equity_curve, trade_log,
    and optionally OHLCV data.
    """
    # ── Step 2: FeatureAgent ─────────────────────────────────────
    feature_agent = FeatureAgent()
    feature_outputs = feature_agent.run({"cleaned_data": price_data})
    feature_matrix = feature_outputs["feature_matrix"]
    target = feature_outputs["target"]
    feature_metadata = feature_outputs["feature_metadata"]
    agents_to_log.append(feature_agent)

    logger.info(
        "%s: FeatureAgent → %d features, %d rows",
        symbol, len(feature_metadata), len(feature_matrix),
    )

    # ── Step 3: ModelAgent — final model training ────────────────
    model_agent = ModelAgent()
    model_outputs = model_agent.run({
        "feature_matrix": feature_matrix,
        "target": target,
        "model_config": model_config,
    })
    agents_to_log.append(model_agent)

    # Save inference bundle for /predict endpoint
    _save_inference_bundle(model_agent, symbol)

    logger.info(
        "%s: ModelAgent → train_acc=%.4f, test_acc=%.4f",
        symbol,
        model_outputs["train_metrics"]["accuracy"],
        model_outputs["test_metrics"]["accuracy"],
    )

    # ── Step 4: WalkForwardAgent — OOS validation ────────────────
    wf_agent = WalkForwardAgent(config={
        "n_folds": n_folds,
        "model_config": model_config,
        "backtest_config": backtest_config,
        "signal_threshold": signal_threshold,
    })

    # Align price_data to feature_matrix index for WalkForward
    aligned_price = price_data.loc[feature_matrix.index]

    wf_outputs = wf_agent.run({
        "feature_matrix": feature_matrix,
        "target": target,
        "price_data": aligned_price,
    })
    agents_to_log.append(wf_agent)

    logger.info(
        "%s: WalkForwardAgent → %d folds, mean_sharpe=%.4f",
        symbol, wf_outputs["n_folds"], wf_outputs["aggregated_metrics"]["mean_sharpe"],
    )

    # ── Step 5: BacktestAgent — full backtest on OOS predictions ─
    predictions = model_outputs["predictions"]
    signals = _predictions_to_signals(predictions, signal_threshold)

    # Align price_data to the OOS prediction window
    oos_price = price_data.loc[signals.index]

    backtest_agent = BacktestAgent(config=backtest_config)
    backtest_outputs = backtest_agent.run({
        "price_data": oos_price,
        "predictions": signals,
        "benchmark_data": benchmark_data,
        "out_of_sample": True,
    })
    agents_to_log.append(backtest_agent)

    logger.info(
        "%s: BacktestAgent → sharpe=%.4f, max_dd=%.2f%%, trades=%d",
        symbol,
        backtest_outputs["performance_summary"]["sharpe"],
        backtest_outputs["performance_summary"]["max_drawdown"] * 100,
        backtest_outputs["performance_summary"]["total_trades"],
    )

    # ── Step 6a: OverfittingAgent ────────────────────────────────
    overfitting_agent = OverfittingAgent()
    overfitting_outputs = overfitting_agent.run({
        "train_metrics": model_outputs["train_metrics"],
        "test_metrics": model_outputs["test_metrics"],
        "fold_results": wf_outputs["fold_results"],
    })
    agents_to_log.append(overfitting_agent)

    logger.info(
        "%s: OverfittingAgent → score=%.4f, warnings=%d",
        symbol,
        overfitting_outputs["overfitting_score"],
        len(overfitting_outputs["warnings"]),
    )

    # ── Step 6b: RiskAgent ───────────────────────────────────────
    equity_curve = backtest_outputs["equity_curve"]
    strategy_returns = equity_curve.pct_change().dropna()

    risk_agent = RiskAgent()
    risk_outputs = risk_agent.run({
        "returns": strategy_returns,
        "price_data": oos_price.loc[strategy_returns.index],
        "signals": signals.loc[strategy_returns.index],
        "trade_log": backtest_outputs["trade_log"],
    })
    agents_to_log.append(risk_agent)

    logger.info(
        "%s: RiskAgent → VaR95=%.4f, max_exposure=%.4f",
        symbol,
        risk_outputs["risk_metrics"]["var_95"],
        risk_outputs["risk_metrics"]["max_position_exposure"],
    )

    result: Dict[str, Any] = {
        "feature_metadata": feature_metadata,
        "feature_count": len(feature_metadata),
        "model": {
            "model_type": model_outputs["model_type"],
            "train_metrics": _clean_dict(model_outputs["train_metrics"]),
            "test_metrics": _clean_dict(model_outputs["test_metrics"]),
            "train_test_gap": _clean_dict(model_outputs["train_test_gap"]),
            "feature_importances": _clean_dict(model_outputs["feature_importances"]),
            "split_info": _clean_dict(model_outputs["split_info"]),
        },
        "walk_forward": {
            "n_folds": wf_outputs["n_folds"],
            "aggregated_metrics": _clean_dict(wf_outputs["aggregated_metrics"]),
            "fold_results": _serialize_fold_results(wf_outputs["fold_results"]),
            "adjustments": _clean_dict(wf_outputs.get("adjustments", {})),
        },
        "backtest": {
            "performance_summary": _clean_dict(backtest_outputs["performance_summary"]),
            "trade_count": len(backtest_outputs["trade_log"]),
            "equity_start": float(equity_curve.iloc[0]),
            "equity_end": float(equity_curve.iloc[-1]),
            "equity_curve": serialize_equity_curve(equity_curve),
            "trade_log": serialize_trade_log(backtest_outputs["trade_log"]),
            "ohlcv": serialize_ohlcv(price_data) if include_ohlcv else [],
        },
        "overfitting": {
            "overfitting_score": float(overfitting_outputs["overfitting_score"]),
            "warnings": overfitting_outputs["warnings"],
            "failure_modes": overfitting_outputs["failure_modes"],
            "recommendations": overfitting_outputs["recommendations"],
            "diagnostics": _clean_dict(overfitting_outputs["diagnostics"]),
        },
        "risk": {
            "risk_metrics": _clean_dict(risk_outputs["risk_metrics"]),
            "mean_position_size": float(risk_outputs["position_sizes"].abs().mean()),
        },
        # Keep raw series for PortfolioAgent and StatsAgent
        "_equity_curve": equity_curve,
        "_strategy_returns": strategy_returns,
    }

    return result


# ── Portfolio construction ───────────────────────────────────────


def _run_portfolio(
    per_symbol_results: Dict[str, Dict[str, Any]],
    cleaned_data: Dict[str, pd.DataFrame],
    tx_cost_bps: float,
    agents_to_log: List[Any],
) -> Optional[Dict[str, Any]]:
    """Run PortfolioAgent if multiple symbols are available."""
    symbols = list(per_symbol_results.keys())

    if len(symbols) < 2:
        logger.info("Single symbol — skipping PortfolioAgent")
        return None

    # Build multi-asset returns DataFrame from cleaned price data
    # Use close-to-close returns on the common date range
    returns_dict = {}
    for sym in symbols:
        closes = cleaned_data[sym]["close"]
        returns_dict[sym] = closes.pct_change().dropna()

    returns_df = pd.DataFrame(returns_dict).dropna()

    if len(returns_df) < 30:
        logger.warning(
            "Only %d common return observations — insufficient for PortfolioAgent",
            len(returns_df),
        )
        return None

    portfolio_agent = PortfolioAgent(config={
        "transaction_cost_bps": tx_cost_bps,
    })
    portfolio_outputs = portfolio_agent.run({"returns": returns_df})
    agents_to_log.append(portfolio_agent)

    logger.info(
        "PortfolioAgent → Sharpe=%.4f, method=%s",
        portfolio_outputs["portfolio_metrics"]["sharpe_ratio"],
        "equal_weight",
    )

    return {
        "portfolio_metrics": _clean_dict(portfolio_outputs["portfolio_metrics"]),
        "n_assets": len(symbols),
        "equity_curve": serialize_equity_curve(portfolio_outputs["equity_curve"]),
        "weights": serialize_weights(portfolio_outputs["weights"]),
        # Keep raw series for StatsAgent
        "_portfolio_returns": portfolio_outputs["portfolio_returns"],
        "_equity_curve": portfolio_outputs["equity_curve"],
    }


# ── Statistical validation ───────────────────────────────────────


def _run_stats(
    per_symbol_results: Dict[str, Dict[str, Any]],
    portfolio_result: Optional[Dict[str, Any]],
    benchmark_data: Optional[pd.DataFrame],
    agents_to_log: List[Any],
) -> Dict[str, Any]:
    """Run StatsAgent on strategy returns."""
    # Use portfolio returns if available, else first symbol's returns
    if portfolio_result is not None:
        returns = portfolio_result["_portfolio_returns"]
    else:
        first_sym = next(iter(per_symbol_results))
        returns = per_symbol_results[first_sym]["_strategy_returns"]

    stats_inputs: Dict[str, Any] = {"returns": returns}

    # Benchmark comparison
    if benchmark_data is not None:
        bench_returns = benchmark_data["close"].pct_change().dropna()
        stats_inputs["benchmark_returns"] = bench_returns

    # Multiple testing correction if multiple symbols tested
    num_symbols = len(per_symbol_results)
    if num_symbols > 1:
        stats_inputs["num_tests"] = num_symbols

    stats_agent = StatsAgent()
    stats_outputs = stats_agent.run(stats_inputs)
    agents_to_log.append(stats_agent)

    logger.info(
        "StatsAgent → Sharpe=%.4f, p=%.4f, significant=%s",
        stats_outputs["metrics"]["sharpe"],
        stats_outputs["hypothesis_test"]["p_value"],
        stats_outputs["hypothesis_test"]["is_significant"],
    )

    return {
        "metrics": _clean_dict(stats_outputs["metrics"]),
        "bootstrap": _clean_dict(stats_outputs["bootstrap"]),
        "hypothesis_test": _clean_dict(stats_outputs["hypothesis_test"]),
        "multiple_testing": _clean_dict(stats_outputs["multiple_testing"]),
        "benchmark": _clean_dict(stats_outputs["benchmark"]) if stats_outputs.get("benchmark") else None,
    }


# ── Helpers ──────────────────────────────────────────────────────


def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    import numpy as np
    import math

    cleaned: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned[k] = _clean_dict(v)
        elif isinstance(v, (np.floating, np.complexfloating)):
            fval = float(v)
            cleaned[k] = None if (math.isnan(fval) or math.isinf(fval)) else round(fval, 6)
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        elif isinstance(v, np.bool_):
            cleaned[k] = bool(v)
        elif isinstance(v, float):
            cleaned[k] = None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
        elif isinstance(v, (pd.Timestamp, )):
            cleaned[k] = v.isoformat()
        else:
            cleaned[k] = v
    return cleaned


def _fetch_benchmark(
    benchmark_symbol: str,
    config: Dict[str, Any],
    cleaned_data: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Fetch benchmark data for comparison.

    If the benchmark symbol is already in cleaned_data, reuse it.
    Otherwise, fetch via DataAgent. Returns None on failure.
    """
    if benchmark_symbol in cleaned_data:
        return cleaned_data[benchmark_symbol]

    try:
        bench_agent = DataAgent()
        bench_inputs: Dict[str, Any] = {"symbols": [benchmark_symbol]}
        if config.get("start_date"):
            bench_inputs["start_date"] = config["start_date"]
        if config.get("end_date"):
            bench_inputs["end_date"] = config["end_date"]

        bench_outputs = bench_agent.run(bench_inputs)
        bench_data = bench_outputs["cleaned_data"]
        if benchmark_symbol in bench_data:
            return bench_data[benchmark_symbol]
    except Exception as exc:
        logger.warning("Failed to fetch benchmark %s: %s", benchmark_symbol, exc)

    return None


def _predictions_to_signals(
    predictions: pd.Series,
    threshold: float,
) -> pd.Series:
    """Convert probability predictions to trading signals {-1, 0, 1}.

    P(up) >= threshold → 1 (long)
    P(up) < (1 - threshold) → -1 (short)
    Otherwise → 0 (flat)
    """
    signals = pd.Series(0, index=predictions.index, dtype=int)
    signals[predictions >= threshold] = 1
    signals[predictions < (1 - threshold)] = -1
    return signals


def _save_inference_bundle(model_agent: ModelAgent, symbol: str) -> None:
    """Save model inference bundle for the /predict endpoint."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / f"{symbol.upper()}_inference.joblib"
    try:
        model_agent.save_inference_bundle(artifact_path)
        logger.info("Saved inference bundle: %s", artifact_path)
    except Exception as exc:
        logger.warning("Failed to save inference bundle for %s: %s", symbol, exc)

    # Clear cached bundle so next prediction uses the fresh model
    try:
        from api.predict_service import clear_inference_bundle_cache
        clear_inference_bundle_cache(symbol.upper())
    except ImportError:
        pass


def _serialize_fold_results(fold_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip non-serializable objects from fold results for JSON output."""
    serialized = []
    for fold in fold_results:
        model_metrics = fold.get("model_metrics", {})
        cleaned_model_metrics = {}
        for k, v in model_metrics.items():
            if isinstance(v, dict):
                cleaned_model_metrics[k] = _clean_dict(v)
            else:
                cleaned_model_metrics[k] = v
        backtest_metrics = fold.get("backtest_metrics", {})
        cleaned_backtest = _clean_dict(backtest_metrics) if isinstance(backtest_metrics, dict) else backtest_metrics
        serialized.append({
            "fold_index": fold["fold_index"],
            "split_info": _clean_dict(fold["split_info"]) if isinstance(fold.get("split_info"), dict) else fold.get("split_info"),
            "model_metrics": cleaned_model_metrics,
            "backtest_metrics": cleaned_backtest,
            "model_type": fold.get("model_type"),
        })
    return serialized


def _build_results(
    run_id: str,
    config: Dict[str, Any],
    data_quality: Dict[str, Any],
    per_symbol_results: Dict[str, Dict[str, Any]],
    portfolio_result: Optional[Dict[str, Any]],
    stats_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final results dict, stripping internal keys."""
    # Strip internal pandas objects from per-symbol results
    clean_per_symbol = {}
    for sym, result in per_symbol_results.items():
        clean_per_symbol[sym] = {
            k: v for k, v in result.items() if not k.startswith("_")
        }

    # Strip internal pandas objects from portfolio result
    clean_portfolio = None
    if portfolio_result is not None:
        clean_portfolio = {
            k: v for k, v in portfolio_result.items() if not k.startswith("_")
        }

    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": config.get("symbols", []),
            "model_type": config.get("model_type", "logistic_regression"),
            "transaction_costs_bps": config.get("transaction_costs_bps", 5.0),
            "slippage_bps": config.get("slippage_bps", 2.0),
            "max_position_size": config.get("max_position_size", 0.1),
            "benchmark": config.get("benchmark", "SPY"),
            "n_folds": config.get("n_folds", 5),
        },
        "data_quality": data_quality,
        "per_symbol": clean_per_symbol,
        "portfolio": clean_portfolio,
        "stats": stats_result,
    }


def _log_experiment(
    run_id: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
) -> None:
    """Save experiment results to experiments/ directory."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract primary symbol metrics for top-level summary
    per_symbol = results.get("per_symbol", {})
    first_sym = next(iter(per_symbol), None)
    primary = per_symbol.get(first_sym, {}) if first_sym else {}
    backtest = primary.get("backtest", {}).get("performance_summary", {})
    overfitting = primary.get("overfitting", {})
    stats = results.get("stats", {})

    experiment = {
        "experiment_id": f"pipeline_{run_id}",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "agent": "PipelineOrchestrator",
        "stage": "full_pipeline",
        "timestamp": results.get("timestamp"),
        "symbols": config.get("symbols", []),
        "model": config.get("model_type", "logistic_regression"),
        "parameters": {
            "transaction_costs_bps": config.get("transaction_costs_bps", 5.0),
            "slippage_bps": config.get("slippage_bps", 2.0),
            "max_position_size": config.get("max_position_size", 0.1),
            "benchmark": config.get("benchmark", "SPY"),
            "n_folds": config.get("n_folds", 5),
        },
        "out_of_sample": True,
        "walk_forward_folds": primary.get("walk_forward", {}).get("n_folds", 0),
        "metrics": {
            "sharpe": backtest.get("sharpe", 0.0),
            "sortino": backtest.get("sortino", 0.0),
            "max_drawdown": backtest.get("max_drawdown", 0.0),
            "calmar": backtest.get("calmar", 0.0),
            "win_rate": backtest.get("win_rate", 0.0),
            "turnover": backtest.get("turnover", 0.0),
            "total_return": backtest.get("total_return", 0.0),
        },
        "benchmark_comparison": backtest.get("benchmark_comparison"),
        "overfitting_score": overfitting.get("overfitting_score", 0.0),
        "statistical_significance": {
            "sharpe_p_value": stats.get("hypothesis_test", {}).get("p_value", 1.0),
            "is_significant": stats.get("hypothesis_test", {}).get("is_significant", False),
            "sharpe_ci_lower": stats.get("bootstrap", {}).get("sharpe_ci_lower"),
            "sharpe_ci_upper": stats.get("bootstrap", {}).get("sharpe_ci_upper"),
        },
        "notes": "",
    }

    filepath = EXPERIMENTS_DIR / f"pipeline_{run_id}.json"
    filepath.write_text(json.dumps(experiment, indent=2, default=str))
    logger.info("Experiment logged to %s", filepath)


def _try_db_insert(results: Dict[str, Any]) -> None:
    """Optionally persist results to Supabase. Fails silently."""
    try:
        from storage.db_client import DBClient
        db = DBClient()
        db.insert_experiment(results)
    except Exception as exc:
        logger.debug("DB insert skipped: %s", exc)
