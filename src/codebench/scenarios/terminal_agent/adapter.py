"""Scenario adapter for terminal-agent benchmarks.

Handles Terminal-Bench 2.0 and similar datasets where an agent must issue
shell commands inside a sandbox to accomplish a task.

TODO: implement fully once interactive sandbox sessions are supported.
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


class TerminalAgentAdapter(ScenarioAdapter):
    """Adapter for ``terminal_agent`` scenarios (stub)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.TERMINAL_AGENT

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        # TODO: build system/user prompt for agentic terminal interaction
        raise NotImplementedError("terminal_agent prompt formatting not yet implemented")

    def extract_submission(self, response: ProviderResponse) -> str:
        # TODO: extract shell command(s) from response
        raise NotImplementedError("terminal_agent submission extraction not yet implemented")

    def build_execution_payload(self, instance: dict[str, Any], submission: str) -> dict[str, Any]:
        # TODO: prepare interactive sandbox session
        raise NotImplementedError("terminal_agent execution payload not yet implemented")

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        # TODO: evaluate terminal state against success criteria
        raise NotImplementedError("terminal_agent scoring not yet implemented")
