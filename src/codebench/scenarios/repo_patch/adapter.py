"""Scenario adapter for repository-level patch generation benchmarks.

Handles SWE-bench Lite, SWE-bench Verified, and similar datasets where the
task is to produce a unified diff that resolves a GitHub issue.

TODO: implement fully once sandbox and git-apply infrastructure are ready.
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


class RepoPatchAdapter(ScenarioAdapter):
    """Adapter for ``repo_patch`` scenarios (stub)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.REPO_PATCH

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        # TODO: build prompt from problem_statement, repo context, etc.
        raise NotImplementedError("repo_patch prompt formatting not yet implemented")

    def extract_submission(self, response: ProviderResponse) -> str:
        # TODO: extract unified diff from response
        raise NotImplementedError("repo_patch submission extraction not yet implemented")

    def build_execution_payload(self, instance: dict[str, Any], submission: str) -> dict[str, Any]:
        # TODO: apply patch, run test suite inside sandbox
        raise NotImplementedError("repo_patch execution payload not yet implemented")

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        # TODO: check test-patch pass/fail
        raise NotImplementedError("repo_patch scoring not yet implemented")
