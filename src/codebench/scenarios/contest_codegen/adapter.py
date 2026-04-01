"""Scenario adapter for competitive-programming code generation benchmarks.

Handles LiveCodeBench and similar datasets where the task is to produce a
complete solution for a contest problem with stdin/stdout I/O.

TODO: implement fully once stdin/stdout sandbox harness is ready.
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


class ContestCodegenAdapter(ScenarioAdapter):
    """Adapter for ``contest_codegen`` scenarios (stub)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CONTEST_CODEGEN

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        # TODO: build prompt with problem statement, starter code, I/O examples
        raise NotImplementedError("contest_codegen prompt formatting not yet implemented")

    def extract_submission(self, response: ProviderResponse) -> str:
        # TODO: extract full program from response
        raise NotImplementedError("contest_codegen submission extraction not yet implemented")

    def build_execution_payload(self, instance: dict[str, Any], submission: str) -> dict[str, Any]:
        # TODO: combine program with stdin test cases
        raise NotImplementedError("contest_codegen execution payload not yet implemented")

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        # TODO: compare stdout against expected output
        raise NotImplementedError("contest_codegen scoring not yet implemented")
