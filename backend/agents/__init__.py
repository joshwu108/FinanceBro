from agents.base_agent import BaseAgent
from agents.backtest_agent import BacktestAgent
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.overfitting_agent import OverfittingAgent
from agents.portfolio_agent import PortfolioAgent
from agents.risk_agent import RiskAgent
from agents.stats_agent import StatsAgent
from agents.walkforward_agent import WalkForwardAgent

# ExplainabilityAgent requires 'shap' which is an optional dependency.
# Import lazily to avoid breaking core pipeline when shap is not installed.
try:
    from agents.explainability_agent import ExplainabilityAgent
except ImportError:
    ExplainabilityAgent = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseAgent",
    "BacktestAgent",
    "DataAgent",
    "ExplainabilityAgent",
    "FeatureAgent",
    "ModelAgent",
    "OverfittingAgent",
    "PortfolioAgent",
    "RiskAgent",
    "StatsAgent",
    "WalkForwardAgent",
]
