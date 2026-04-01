"""Core benchmark runner pipeline."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from codebench.core.config.settings import RunConfig
from codebench.core.interfaces.artifact_store import ArtifactStore
from codebench.core.interfaces.provider import ProviderInterface
from codebench.core.interfaces.sandbox import SandboxRunner
from codebench.core.interfaces.scenario import ScenarioAdapter
from codebench.core.models.common import (
    InstanceResult,
    ProviderRequest,
    RunManifest,
    RunStatus,
)


class BenchmarkPipeline:
    """Orchestrates a complete benchmark run."""

    def __init__(
        self,
        config: RunConfig,
        provider: ProviderInterface,
        scenario: ScenarioAdapter,
        sandbox: SandboxRunner | None,
        artifact_store: ArtifactStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.scenario = scenario
        self.sandbox = sandbox
        self.artifact_store = artifact_store

    def create_manifest(self, instances: list[dict[str, Any]]) -> RunManifest:
        return RunManifest(
            dataset_name=self.config.dataset.name,
            dataset_version=self.config.dataset.version,
            scenario_type=self.config.scenario_type,
            track_type=self.config.track_type,
            provider_name=self.config.provider.name,
            model_name=self.config.provider.model,
            harness_version=self.config.harness_version,
            prompt_version=self.config.prompt_version,
            seed=self.config.seed,
            total_instances=len(instances),
            config=self.config.model_dump(mode="json"),
        )

    async def run_instance(self, instance: dict[str, Any], run_id: str) -> InstanceResult:
        instance_id = hashlib.sha256(f"{run_id}:{instance.get('id', '')}".encode()).hexdigest()[:8]
        result = InstanceResult(
            instance_id=instance_id,
            run_id=run_id,
            dataset_instance_id=str(instance.get("id", instance_id)),
        )

        try:
            # Step 1: Format prompt
            request: ProviderRequest = self.scenario.format_prompt(instance)
            if self.config.seed is not None:
                request.seed = self.config.seed

            # Step 2: Call provider
            result.status = RunStatus.RUNNING
            response = await self.provider.generate(request)
            result.provider_response = response

            # Short-circuit on provider error (API failure, network error, etc.)
            if response.metadata.get("error"):
                from codebench.core.models.common import ScoringResult

                result.scoring_result = ScoringResult(
                    score=0.0,
                    passed=False,
                    details={"reason": "provider_error", **response.metadata},
                )
                result.status = RunStatus.FAILED
                self._save_instance_artifacts(run_id, instance_id, instance, result)
                return result

            # Step 3: Extract submission
            submission = self.scenario.extract_submission(response)

            # Step 4: Execute in sandbox (if applicable)
            execution_result = None
            if self.sandbox is not None:
                payload = self.scenario.build_execution_payload(instance, submission)
                execution_result = await self.sandbox.execute(
                    code=payload.get("code", submission),
                    language=payload.get("language", "python"),
                    timeout_seconds=self.config.sandbox.timeout_seconds,
                    memory_limit_mb=self.config.sandbox.memory_limit_mb,
                    network_enabled=self.config.sandbox.network_enabled,
                    workdir_files=payload.get("files"),
                )
                result.execution_result = execution_result

            # Step 5: Score
            scoring = self.scenario.score(instance, submission, execution_result)
            result.scoring_result = scoring
            result.status = RunStatus.COMPLETED

        except Exception as exc:
            result.status = RunStatus.FAILED
            result.error = str(exc)

        # Step 6: Persist artifacts
        self._save_instance_artifacts(run_id, instance_id, instance, result)
        return result

    async def run(self, instances: list[dict[str, Any]]) -> RunManifest:
        manifest = self.create_manifest(instances)
        manifest.status = RunStatus.RUNNING
        self.artifact_store.save_manifest(manifest.run_id, manifest.model_dump(mode="json"))

        concurrency = self.config.concurrency
        results: list[InstanceResult] = []

        try:
            if concurrency <= 1:
                # Sequential execution
                for instance in instances:
                    instance_result = await self.run_instance(instance, manifest.run_id)
                    results.append(instance_result)
                    manifest.completed_instances += 1
            else:
                # Parallel execution with semaphore
                semaphore = asyncio.Semaphore(concurrency)

                async def _run_with_limit(inst: dict[str, Any]) -> InstanceResult:
                    async with semaphore:
                        return await self.run_instance(inst, manifest.run_id)

                tasks = [asyncio.create_task(_run_with_limit(inst)) for inst in instances]
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    manifest.completed_instances += 1

            manifest.status = RunStatus.COMPLETED
        except (KeyboardInterrupt, Exception):
            manifest.status = RunStatus.FAILED
            # Cancel remaining tasks on interrupt
            if concurrency > 1:
                for t in tasks:  # noqa: F821
                    t.cancel()
        finally:
            passed = sum(
                1 for r in results if r.scoring_result is not None and r.scoring_result.passed
            )
            total = len(results)
            manifest.results_summary = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total else 0.0,
            }
            self.artifact_store.save_manifest(manifest.run_id, manifest.model_dump(mode="json"))

        return manifest

    def _save_instance_artifacts(
        self,
        run_id: str,
        instance_id: str,
        instance: dict[str, Any],
        result: InstanceResult,
    ) -> None:
        import json

        self.artifact_store.save_artifact(
            run_id, instance_id, "instance_input.json", json.dumps(instance, default=str)
        )
        self.artifact_store.save_artifact(
            run_id, instance_id, "result.json", result.model_dump_json(indent=2)
        )
