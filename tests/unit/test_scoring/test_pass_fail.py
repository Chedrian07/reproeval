"""Tests for pass/fail scorer."""

import pytest

from codebench.core.models.common import ExecutionResult, ScoringResult
from codebench.scoring.pass_fail import PassFailScorer


@pytest.mark.unit
class TestPassFailScorer:
    def setup_method(self) -> None:
        self.scorer = PassFailScorer()

    def test_name(self) -> None:
        assert self.scorer.name == "pass_fail"

    def test_pass(self) -> None:
        result = ExecutionResult(exit_code=0, stdout="OK", stderr="", duration_ms=100.0)
        scoring = self.scorer.score({}, "code", result)
        assert scoring.passed
        assert scoring.score == 1.0

    def test_fail(self) -> None:
        result = ExecutionResult(exit_code=1, stdout="", stderr="error", duration_ms=100.0)
        scoring = self.scorer.score({}, "code", result)
        assert not scoring.passed
        assert scoring.score == 0.0

    def test_timeout(self) -> None:
        result = ExecutionResult(
            exit_code=137, stdout="", stderr="", duration_ms=30000.0, timed_out=True
        )
        scoring = self.scorer.score({}, "code", result)
        assert not scoring.passed
        assert scoring.details["reason"] == "timeout"

    def test_no_execution(self) -> None:
        scoring = self.scorer.score({}, "code", None)
        assert not scoring.passed
        assert scoring.details["reason"] == "no_execution"

    def test_aggregate(self) -> None:
        results = [
            ScoringResult(score=1.0, passed=True),
            ScoringResult(score=0.0, passed=False),
            ScoringResult(score=1.0, passed=True),
        ]
        summary = self.scorer.aggregate(results)
        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert abs(summary["pass_rate"] - 2 / 3) < 0.001

    def test_aggregate_empty(self) -> None:
        summary = self.scorer.aggregate([])
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0
