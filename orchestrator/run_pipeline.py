from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.walkforward_agent import WalkForwardAgent
from agents.backtest_agent import BacktestAgent
from agents.overfitting_agent import OverfittingAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.stats_agent import StatsAgent

import json
import os
from datetime import datetime

from storage.db_client import DBClient

db = DBClient()

db.insert_experiment(results)


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

    # Step 2: Feature engineering (no look-ahead bias)
    feature_agent = FeatureAgent()
    features, target = feature_agent.run(data)
    feature_agent.validate(data, (features, target))

    # Step 3: Model training
    model_agent = ModelAgent(config)
    predictions = model_agent.run(features, target)
    model_agent.validate((features, target), predictions)

    # Step 4: Walk-forward validation
    wf_agent = WalkForwardAgent()
    wf_results = wf_agent.run(features, target, config)
    wf_agent.validate((features, target, config), wf_results)

    # Step 5: Backtesting with transaction costs
    backtest_agent = BacktestAgent()
    backtest_results = backtest_agent.run(data, predictions, config)
    backtest_agent.validate((data, predictions, config), backtest_results)

    # Step 6: Overfitting detection
    overfit_agent = OverfittingAgent()
    overfit_report = overfit_agent.run(wf_results, backtest_results)
    overfit_agent.validate((wf_results, backtest_results), overfit_report)

    # Step 7: Risk-adjusted position sizing
    risk_agent = RiskAgent()
    risk_adjusted = risk_agent.run(backtest_results, config)
    risk_agent.validate(backtest_results, risk_adjusted)

    # Step 8: Portfolio construction
    portfolio_agent = PortfolioAgent()
    portfolio = portfolio_agent.run(risk_adjusted, config)
    portfolio_agent.validate(risk_adjusted, portfolio)

    # Step 9: Statistical significance testing
    stats_agent = StatsAgent()
    stats = stats_agent.run(portfolio)
    stats_agent.validate(portfolio, stats)

    # Log all metrics
    for agent in [data_agent, feature_agent, model_agent, wf_agent,
                  backtest_agent, overfit_agent, risk_agent,
                  portfolio_agent, stats_agent]:
        agent.log_metrics()

    results = {
        "walk_forward": wf_results,
        "backtest": backtest_results,
        "overfitting": overfit_report,
        "risk": risk_adjusted,
        "portfolio": portfolio,
        "stats": stats,
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
