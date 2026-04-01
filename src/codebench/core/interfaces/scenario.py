"""Scenario adapter interface — maps dataset instances to benchmark execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from codebench.core.models.common import (
    ExecutionResult,
    ProviderRequest,
    ProviderResponse,
    ScenarioType,
    ScoringResult,
)


class ScenarioAdapter(ABC):
    """Base interface for scenario adapters."""

    @property
    @abstractmethod
    def scenario_type(self) -> ScenarioType:
        """The scenario type this adapter handles."""

    @abstractmethod
    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        """Convert a dataset instance to a provider request."""

    @abstractmethod
    def extract_submission(self, response: ProviderResponse) -> str:
        """Extract the candidate submission from a provider response."""

    @abstractmethod
    def build_execution_payload(self, instance: dict[str, Any], submission: str) -> dict[str, Any]:
        """Build the sandbox execution payload (code, test harness, etc.)."""

    @abstractmethod
    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        """Score the result of a benchmark instance."""
