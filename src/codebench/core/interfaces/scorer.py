"""Scorer interface — judges and scores benchmark results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from codebench.core.models.common import ExecutionResult, ScoringResult


class Scorer(ABC):
    """Base interface for scoring/judging benchmark results."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique scorer identifier."""

    @abstractmethod
    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        """Score a single benchmark instance."""

    def aggregate(self, results: list[ScoringResult]) -> dict[str, Any]:
        """Aggregate multiple scoring results into summary statistics."""
        if not results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        passed = sum(1 for r in results if r.passed)
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "mean_score": sum(r.score for r in results) / len(results),
        }
