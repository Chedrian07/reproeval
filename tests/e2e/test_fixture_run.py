"""End-to-end test with fixture data and mock provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from codebench.artifacts.filesystem import FilesystemArtifactStore
from codebench.core.config.settings import (
    DatasetConfig,
    ProviderConfig,
    RunConfig,
)
from codebench.core.models.common import (
    ProviderResponse,
    RunStatus,
    ScenarioType,
    TokenUsage,
    TrackType,
)
from codebench.core.runner.pipeline import BenchmarkPipeline

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture_instances() -> list[dict[str, str]]:
    path = FIXTURES_DIR / "humaneval_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.mark.e2e
class TestFixtureBackedRun:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_provider(self, tmp_path: Path) -> None:
        """Run the full pipeline with fixture data and a mock provider."""
        fixture_instances = _load_fixture_instances()

        config = RunConfig(
            provider=ProviderConfig(name="mock", model="mock-model"),
            dataset=DatasetConfig(name="humaneval_plus", version="1.0"),
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
            track_type=TrackType.LM_ONLY,
            artifacts_dir=tmp_path / "artifacts",
            seed=42,
        )

        mock_provider = AsyncMock()

        def generate_solution(request):  # type: ignore[no-untyped-def]
            for inst in fixture_instances:
                if inst["prompt"] in request.prompt:
                    return ProviderResponse(
                        content=inst["canonical_solution"],
                        model="mock-model",
                        usage=TokenUsage(input_tokens=100, output_tokens=50),
                        latency_ms=10.0,
                    )
            return ProviderResponse(
                content="pass",
                model="mock-model",
                usage=TokenUsage(input_tokens=100, output_tokens=10),
                latency_ms=10.0,
            )

        mock_provider.generate = AsyncMock(side_effect=generate_solution)

        from codebench.scenarios.function_codegen.adapter import FunctionCodegenAdapter

        scenario = FunctionCodegenAdapter()
        artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")

        pipeline = BenchmarkPipeline(
            config=config,
            provider=mock_provider,
            scenario=scenario,
            sandbox=None,
            artifact_store=artifact_store,
        )

        manifest = await pipeline.run(fixture_instances)

        assert manifest.status == RunStatus.COMPLETED
        assert manifest.total_instances == 2
        assert manifest.completed_instances == 2
        assert manifest.dataset_name == "humaneval_plus"
        assert manifest.seed == 42

        # Validate scoring correctness:
        # With sandbox=None, FunctionCodegenAdapter.score() should return passed=False
        assert manifest.results_summary["passed"] == 0
        assert manifest.results_summary["pass_rate"] == 0.0

        runs = artifact_store.list_runs()
        assert len(runs) == 1
        loaded_manifest = artifact_store.load_manifest(manifest.run_id)
        assert loaded_manifest["status"] == "completed"

        artifacts = artifact_store.list_artifacts(manifest.run_id)
        assert len(artifacts) >= 2
