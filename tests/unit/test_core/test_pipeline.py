"""Tests for the benchmark pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from codebench.artifacts.filesystem import FilesystemArtifactStore
from codebench.core.config.settings import (
    DatasetConfig,
    ProviderConfig,
    RunConfig,
)
from codebench.core.models.common import (
    ExecutionResult,
    ProviderRequest,
    ProviderResponse,
    RunStatus,
    ScenarioType,
    ScoringResult,
    TokenUsage,
    TrackType,
)
from codebench.core.runner.pipeline import BenchmarkPipeline


def _make_config() -> RunConfig:
    return RunConfig(
        provider=ProviderConfig(name="test", model="test-model"),
        dataset=DatasetConfig(name="test", version="1.0"),
        scenario_type=ScenarioType.FUNCTION_CODEGEN,
        track_type=TrackType.LM_ONLY,
        seed=42,
    )


def _make_provider_response() -> ProviderResponse:
    return ProviderResponse(
        content="def add(a, b):\n    return a + b",
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        latency_ms=100.0,
    )


def _make_execution_result(passed: bool = True) -> ExecutionResult:
    return ExecutionResult(
        exit_code=0 if passed else 1,
        stdout="OK" if passed else "FAIL",
        stderr="",
        duration_ms=50.0,
    )


@pytest.mark.unit
class TestBenchmarkPipeline:
    def _build_pipeline(
        self, tmp_path: Path, with_sandbox: bool = True
    ) -> tuple[BenchmarkPipeline, AsyncMock, MagicMock, AsyncMock | None]:
        config = _make_config()
        config.artifacts_dir = tmp_path

        provider = AsyncMock()
        provider.generate = AsyncMock(return_value=_make_provider_response())

        scenario = MagicMock()
        scenario.format_prompt.return_value = ProviderRequest(prompt="test prompt")
        scenario.extract_submission.return_value = "def add(a, b): return a + b"
        scenario.build_execution_payload.return_value = {
            "code": "def add(a, b): return a + b\nprint(add(2, 3))"
        }
        scenario.score.return_value = ScoringResult(score=1.0, passed=True)

        sandbox = None
        if with_sandbox:
            sandbox = AsyncMock()
            sandbox.execute = AsyncMock(return_value=_make_execution_result())

        artifact_store = FilesystemArtifactStore(tmp_path)

        pipeline = BenchmarkPipeline(
            config=config,
            provider=provider,
            scenario=scenario,
            sandbox=sandbox,
            artifact_store=artifact_store,
        )
        return pipeline, provider, scenario, sandbox

    @pytest.mark.asyncio
    async def test_run_single_instance(self, tmp_path: Path) -> None:
        pipeline, provider, scenario, sandbox = self._build_pipeline(tmp_path)
        instances = [{"id": "test_1", "prompt": "add function", "entry_point": "add"}]

        manifest = await pipeline.run(instances)

        assert manifest.status == RunStatus.COMPLETED
        assert manifest.total_instances == 1
        assert manifest.completed_instances == 1
        assert manifest.results_summary["passed"] == 1
        assert manifest.results_summary["pass_rate"] == 1.0
        provider.generate.assert_called_once()
        scenario.format_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multiple_instances(self, tmp_path: Path) -> None:
        pipeline, provider, scenario, sandbox = self._build_pipeline(tmp_path)
        instances = [{"id": f"test_{i}", "prompt": f"task {i}"} for i in range(3)]

        manifest = await pipeline.run(instances)

        assert manifest.completed_instances == 3
        assert manifest.results_summary["total"] == 3

    @pytest.mark.asyncio
    async def test_run_without_sandbox(self, tmp_path: Path) -> None:
        pipeline, provider, scenario, _ = self._build_pipeline(tmp_path, with_sandbox=False)
        instances = [{"id": "test_1"}]

        manifest = await pipeline.run(instances)
        assert manifest.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_provider_error_handled(self, tmp_path: Path) -> None:
        pipeline, provider, scenario, sandbox = self._build_pipeline(tmp_path)
        provider.generate = AsyncMock(side_effect=RuntimeError("API error"))
        instances = [{"id": "test_1"}]

        manifest = await pipeline.run(instances)

        assert manifest.status == RunStatus.COMPLETED
        assert manifest.results_summary["passed"] == 0

    @pytest.mark.asyncio
    async def test_artifacts_persisted(self, tmp_path: Path) -> None:
        pipeline, *_ = self._build_pipeline(tmp_path)
        instances = [{"id": "test_1"}]

        manifest = await pipeline.run(instances)

        store = FilesystemArtifactStore(tmp_path)
        loaded = store.load_manifest(manifest.run_id)
        assert loaded["dataset_name"] == "test"
        artifacts = store.list_artifacts(manifest.run_id)
        assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_manifest_has_seed(self, tmp_path: Path) -> None:
        pipeline, *_ = self._build_pipeline(tmp_path)
        manifest = pipeline.create_manifest([{"id": "1"}])
        assert manifest.seed == 42

    @pytest.mark.asyncio
    async def test_run_empty_instances(self, tmp_path: Path) -> None:
        pipeline, *_ = self._build_pipeline(tmp_path)
        manifest = await pipeline.run([])
        assert manifest.status == RunStatus.COMPLETED
        assert manifest.results_summary["total"] == 0
        assert manifest.results_summary["pass_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_instance_id_deterministic(self, tmp_path: Path) -> None:
        """Running the same instance with the same run_id produces identical instance_ids."""
        pipeline, *_ = self._build_pipeline(tmp_path)
        fixed_run_id = "fixed-run-id-for-test"
        instance = {"id": "test_deterministic", "prompt": "hello"}

        result_a = await pipeline.run_instance(instance, fixed_run_id)
        result_b = await pipeline.run_instance(instance, fixed_run_id)

        assert result_a.instance_id == result_b.instance_id
        assert len(result_a.instance_id) == 8
