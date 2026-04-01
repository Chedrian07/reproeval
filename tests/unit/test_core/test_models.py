"""Tests for core models."""

import pytest

from codebench.core.models.common import (
    ExecutionResult,
    InstanceResult,
    ProviderRequest,
    ProviderResponse,
    RunManifest,
    RunStatus,
    ScenarioType,
    ScoringResult,
    TokenUsage,
    TrackType,
)


@pytest.mark.unit
class TestTokenUsage:
    def test_total_auto_calculated(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_total_explicit(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200


@pytest.mark.unit
class TestProviderRequest:
    def test_defaults(self) -> None:
        req = ProviderRequest(prompt="hello")
        assert req.temperature == 0.0
        assert req.max_tokens == 40960
        assert req.seed is None
        assert req.stop_sequences == []
        assert req.metadata == {}

    def test_full(self) -> None:
        req = ProviderRequest(
            prompt="hello",
            system_prompt="you are helpful",
            temperature=0.5,
            max_tokens=1024,
            seed=42,
            stop_sequences=["END"],
        )
        assert req.system_prompt == "you are helpful"
        assert req.seed == 42


@pytest.mark.unit
class TestProviderResponse:
    def test_construction(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=20)
        resp = ProviderResponse(
            content="result",
            model="gpt-4",
            usage=usage,
            latency_ms=150.0,
        )
        assert resp.content == "result"
        assert resp.model == "gpt-4"
        assert resp.usage.total_tokens == 30
        assert resp.raw_response == {}


@pytest.mark.unit
class TestExecutionResult:
    def test_success(self) -> None:
        result = ExecutionResult(exit_code=0, stdout="ok", stderr="", duration_ms=100.0)
        assert result.exit_code == 0
        assert not result.timed_out

    def test_timeout(self) -> None:
        result = ExecutionResult(
            exit_code=1, stdout="", stderr="killed", duration_ms=30000.0, timed_out=True
        )
        assert result.timed_out


@pytest.mark.unit
class TestScoringResult:
    def test_passed(self) -> None:
        r = ScoringResult(score=1.0, passed=True)
        assert r.max_score == 1.0
        assert r.details == {}

    def test_failed_with_details(self) -> None:
        r = ScoringResult(
            score=0.0,
            passed=False,
            details={"reason": "wrong_answer"},
            judge_reasoning="Expected 42, got 41",
        )
        assert not r.passed
        assert r.judge_reasoning is not None


@pytest.mark.unit
class TestRunManifest:
    def test_defaults(self) -> None:
        m = RunManifest(
            dataset_name="test",
            dataset_version="1.0",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_type=TrackType.LM_ONLY,
            provider_name="openai",
            model_name="gpt-4",
            harness_version="0.1.0",
        )
        assert m.status == RunStatus.PENDING
        assert m.total_instances == 0
        assert len(m.run_id) == 12

    def test_serialization_roundtrip(self) -> None:
        m = RunManifest(
            dataset_name="test",
            dataset_version="1.0",
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_type=TrackType.LM_ONLY,
            provider_name="openai",
            model_name="gpt-4",
            harness_version="0.1.0",
            total_instances=10,
        )
        data = m.model_dump(mode="json")
        m2 = RunManifest.model_validate(data)
        assert m2.dataset_name == m.dataset_name
        assert m2.total_instances == 10


@pytest.mark.unit
class TestInstanceResult:
    def test_pending_by_default(self) -> None:
        r = InstanceResult(
            instance_id="abc",
            run_id="run1",
            dataset_instance_id="ds_1",
        )
        assert r.status == RunStatus.PENDING
        assert r.provider_response is None
        assert r.error is None
