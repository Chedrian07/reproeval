"""Pass/fail scorer based on execution exit code."""

from __future__ import annotations

from typing import Any

from codebench.core.interfaces.scorer import Scorer
from codebench.core.models.common import ExecutionResult, ScoringResult


class PassFailScorer(Scorer):
    """Scores pass/fail based on execution exit code."""

    @property
    def name(self) -> str:
        return "pass_fail"

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        if execution_result is None:
            return ScoringResult(
                score=0.0,
                passed=False,
                details={"reason": "no_execution"},
            )

        if execution_result.timed_out:
            return ScoringResult(
                score=0.0,
                passed=False,
                details={"reason": "timeout"},
            )

        passed = execution_result.exit_code == 0
        return ScoringResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "exit_code": execution_result.exit_code,
                "stdout_snippet": execution_result.stdout[:500],
                "stderr_snippet": execution_result.stderr[:500],
            },
        )
