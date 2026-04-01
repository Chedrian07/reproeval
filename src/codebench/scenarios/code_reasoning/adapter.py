"""Scenario adapter for code reasoning benchmarks.

Handles CRUXEval and similar datasets where the task is to predict program
input or output without executing code.

TODO: implement fully once LM-judged scoring is wired up.
"""

from __future__ import annotations

from typing import Any

from codebench.core.interfaces.scenario import ScenarioAdapter
from codebench.core.models.common import (
    ExecutionResult,
    ProviderRequest,
    ProviderResponse,
    ScenarioType,
    ScoringResult,
)


class CodeReasoningAdapter(ScenarioAdapter):
    """Adapter for ``code_reasoning`` scenarios (stub)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CODE_REASONING

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        # TODO: build prompt from code snippet and prediction task
        raise NotImplementedError("code_reasoning prompt formatting not yet implemented")

    def extract_submission(self, response: ProviderResponse) -> str:
        # TODO: extract predicted value from response
        raise NotImplementedError("code_reasoning submission extraction not yet implemented")

    def build_execution_payload(self, instance: dict[str, Any], submission: str) -> dict[str, Any]:
        # TODO: code_reasoning may not need sandbox execution — return no-op
        raise NotImplementedError("code_reasoning execution payload not yet implemented")

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        # TODO: exact-match or fuzzy comparison against expected_output
        raise NotImplementedError("code_reasoning scoring not yet implemented")
