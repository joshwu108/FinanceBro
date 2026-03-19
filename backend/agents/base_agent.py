"""BaseAgent — Abstract contract for all pipeline agents.
Every agent in the FinanceBro pipeline MUST subclass BaseAgent and
implement all abstract methods.  No agent may bypass this interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """Base contract that every pipeline agent must satisfy."""

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary task.

        Args:
            inputs: dict whose keys match ``input_schema``.

        Returns:
            dict whose keys match ``output_schema``.
        """

    @abstractmethod
    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate inputs/outputs for correctness and data integrity.
        Must raise on failure (AssertionError or ValueError).
        Returns True when all checks pass.
        """

    @abstractmethod
    def log_metrics(self) -> None:
        """Persist metrics from the most recent ``run()`` call."""

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Describe expected input keys and their types/formats."""

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Describe expected output keys and their types/formats."""
