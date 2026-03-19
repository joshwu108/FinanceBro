from pathlib import Path

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
#from agents.walkforward_agent import WalkForwardAgent
from agents.backtest_agent import BacktestAgent
#from agents.overfitting_agent import OverfittingAgent
#from agents.risk_agent import RiskAgent
#from agents.portfolio_agent import PortfolioAgent
from agents.stats_agent import StatsAgent

import json
import os
from datetime import datetime

from storage.db_client import DBClient

from api.predict_service import clear_inference_bundle_cache

db = DBClient()


def run_pipeline(config):
    """
    Execute the core quant research pipeline.

    Pipeline: Data → Features → Model → WalkForward → Backtest →
              Overfitting → Risk → Portfolio → Stats

    Args:
        config: dict with keys:
            - symbols: list of ticker symbols
            - start_date: str (YYYY-MM-DD)
            - end_date: str (YYYY-MM-DD)
            - model_type: str (logistic, random_forest, xgboost, lstm)
            - transaction_costs_bps: float (default 5.0)
            - slippage_bps: float (default 2.0)
            - max_position_size: float (default 0.1)
            - benchmark: str (default "SPY")
    """

    # Step 1: Data collection and cleaning
    data_agent = DataAgent()
    data = data_agent.run(config)
    data_agent.validate(config, data)

    # Step 2: Feature engineering
    first_ticker = config.get("symbols")[0]
    symbol_data = data["cleaned_data"][first_ticker]
    
    feature_agent = FeatureAgent()
    feature_outputs = feature_agent.run({"cleaned_data": symbol_data})
    features = feature_outputs["feature_matrix"]
    target = feature_outputs["target"]
    feature_agent.validate({"cleaned_data": symbol_data}, feature_outputs)

    # Step 3: Model training
    model_agent = ModelAgent(config)
    model_outputs = model_agent.run({
        "feature_matrix": features, 
        "target": target,
        "model_config": config
    })
    model_agent.validate({
        "feature_matrix": features, 
        "target": target,
        "model_config": config
    }, model_outputs)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"{first_ticker.upper()}_inference.joblib"
    model_agent.save_inference_bundle(artifact_path)
    clear_inference_bundle_cache(first_ticker.upper())

    # Step 4: Walk-forward validation (Implementation Pending)
    wf_results = {}

    # Step 5: Backtesting with transaction costs
    backtest_agent = BacktestAgent()
    backtest_outputs = backtest_agent.run({
        "price_data": symbol_data,
        "predictions": model_outputs["predicted_classes"],
        "config": config
    })
    backtest_agent.validate({
        "price_data": symbol_data,
        "predictions": model_outputs["predicted_classes"],
        "config": config
    }, backtest_outputs)

    # Step 6: Overfitting detection (Implementation Pending)
    overfit_report = {}

    # Step 7: Risk-adjusted position sizing (Implementation Pending)
    risk_adjusted = {}

    # Step 8: Portfolio construction (Implementation Pending)
    portfolio = {}

    # Step 9: Statistical significance testing
    stats_agent = StatsAgent()
    stats_outputs = stats_agent.run({
        "returns": backtest_outputs["equity_curve"].pct_change().dropna()
    })
    stats_agent.validate({"returns": backtest_outputs["equity_curve"].pct_change().dropna()}, stats_outputs)

    # Log all metrics
    for agent in [data_agent, feature_agent, model_agent,
                  backtest_agent, stats_agent]:
        agent.log_metrics()

    results = {
        "walk_forward": wf_results,
        "backtest": backtest_outputs,
        "overfitting": overfit_report,
        "risk": risk_adjusted,
        "portfolio": portfolio,
        "stats": stats_outputs,
    }

    # Save experiment log
    _log_experiment(config, results)

    return results


def _log_experiment(config, results):
    """Save experiment results to /experiments/ directory."""
    experiment = {
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.now().isoformat(),
        "symbols": config.get("symbols", []),
        "model": config.get("model_type", "unknown"),
        "features": config.get("feature_list", []),
        "parameters": {
            "transaction_costs_bps": config.get("transaction_costs_bps", 5.0),
            "slippage_bps": config.get("slippage_bps", 2.0),
            "max_position_size": config.get("max_position_size", 0.1),
            "benchmark": config.get("benchmark", "SPY"),
        },
        "out_of_sample": True,
        "walk_forward_folds": results.get("walk_forward", {}).get("n_folds", 0),
        "metrics": {
            "sharpe": results.get("backtest", {}).get("sharpe", 0.0),
            "sortino": results.get("backtest", {}).get("sortino", 0.0),
            "max_drawdown": results.get("backtest", {}).get("max_drawdown", 0.0),
            "calmar": results.get("backtest", {}).get("calmar", 0.0),
            "win_rate": results.get("backtest", {}).get("win_rate", 0.0),
            "turnover": results.get("backtest", {}).get("turnover", 0.0),
        },
        "benchmark_comparison": {
            "strategy_sharpe": results.get("backtest", {}).get("sharpe", 0.0),
            "benchmark_sharpe": results.get("backtest", {}).get("benchmark_sharpe", 0.0),
            "excess_return": results.get("backtest", {}).get("excess_return", 0.0),
        },
        "overfitting_score": results.get("overfitting", {}).get("score", 0.0),
        "statistical_significance": {
            "sharpe_p_value": results.get("stats", {}).get("p_value", 1.0),
            "is_significant": results.get("stats", {}).get("significance_flag", False),
        },
        "notes": "",
    }

    experiments_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    filepath = os.path.join(experiments_dir, f"{experiment['experiment_id']}.json")
    with open(filepath, "w") as f:
        json.dump(experiment, f, indent=2)

    try:
        db.insert_experiment(experiment)
    except Exception as e:
        print(f"Failed to save experiment to Supabase: {e}")
