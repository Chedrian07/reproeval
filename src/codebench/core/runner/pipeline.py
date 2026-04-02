"""Core benchmark runner pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

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
    ScoringResult,
)

_console = Console()


def _is_debug() -> bool:
    return os.environ.get("CODEBENCH_DEBUG", "").lower() in ("1", "on", "true", "yes")


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
        self._debug = _is_debug()

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
            # Step 1: Format prompt and apply config overrides
            request: ProviderRequest = self.scenario.format_prompt(instance)
            request.temperature = self.config.provider.temperature
            request.max_tokens = self.config.provider.max_tokens
            if self.config.seed is not None:
                request.seed = self.config.seed

            # Step 2: Call provider
            result.status = RunStatus.RUNNING
            response = await self.provider.generate(request)
            result.provider_response = response

            # Short-circuit on provider error
            if response.metadata.get("error"):
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

    # -- run orchestration ------------------------------------------------

    def _build_live_table(
        self,
        total: int,
        completed: int,
        passed: int,
        failed: int,
        latest: list[tuple[str, str, str]],
    ) -> Table:
        """Build the live-updating debug scoreboard."""
        rate = passed / completed if completed else 0.0
        table = Table(
            title=f"[bold]LIVE  {completed}/{total}  "
            f"PASS {passed}  FAIL {failed}  ({rate:.1%})[/bold]",
            show_header=True,
            expand=True,
        )
        table.add_column("#", style="dim", width=6, justify="right")
        table.add_column("Instance", style="cyan", ratio=3)
        table.add_column("Result", width=12, justify="center")
        table.add_column("Detail", style="dim", ratio=4)

        for idx, inst_id, verdict, detail in latest[-15:]:  # show last 15
            if verdict == "PASS":
                style = "[bold green]PASS[/bold green]"
            elif verdict == "FAIL":
                style = "[bold red]FAIL[/bold red]"
            else:
                style = f"[yellow]{verdict}[/yellow]"
            table.add_row(str(idx), inst_id, style, detail)

        return table

    def _verdict_detail(self, r: InstanceResult) -> tuple[str, str]:
        """Extract verdict and short detail from an instance result."""
        if r.scoring_result is None:
            return "ERROR", r.error or "unknown"
        if r.scoring_result.passed:
            parts: list[str] = []
            if r.provider_response:
                u = r.provider_response.usage
                parts.append(f"{r.provider_response.latency_ms:.0f}ms")
                parts.append(f"tokens={u.input_tokens}+{u.output_tokens}")
            if r.execution_result:
                parts.append(f"exec={r.execution_result.duration_ms:.0f}ms")
            return "PASS", "  ".join(parts) if parts else "ok"

        reason = r.scoring_result.details.get("reason", "")
        if reason == "provider_error":
            return "FAIL", "API error"
        if reason == "timeout":
            return "FAIL", "timeout"
        if reason == "no_execution":
            return "FAIL", "no sandbox"
        stderr = r.scoring_result.details.get("stderr_snippet", "")
        if stderr:
            lines = stderr.splitlines()
            # Priority 1: actual error type (NameError, SyntaxError, etc.)
            for line in reversed(lines):
                s = line.strip()
                is_err = ("Error" in s or "Exception" in s)
                if s and is_err and not s.startswith(("Traceback", "File")):
                    return "FAIL", s[:80]
            # Priority 2: last meaningful line
            for line in reversed(lines):
                s = line.strip()
                if s and len(s) > 3 and not s.startswith(("Traceback", "File", "^")):
                    return "FAIL", s[:80]
        return "FAIL", f"exit={r.scoring_result.details.get('exit_code', '?')}"

    async def _run_sequential_debug(
        self,
        instances: list[dict[str, Any]],
        manifest: RunManifest,
        results: list[InstanceResult],
    ) -> None:
        passed = 0
        failed = 0
        log: list[tuple[str, str, str, str]] = []

        with Live(
            _console.render_str("Starting..."), console=_console, refresh_per_second=4
        ) as live:
            for i, instance in enumerate(instances, 1):
                r = await self.run_instance(instance, manifest.run_id)
                results.append(r)
                manifest.completed_instances += 1

                verdict, detail = self._verdict_detail(r)
                if verdict == "PASS":
                    passed += 1
                else:
                    failed += 1
                log.append((str(i), r.dataset_instance_id, verdict, detail))

                live.update(
                    self._build_live_table(len(instances), i, passed, failed, log)
                )

    async def _run_parallel_debug(
        self,
        instances: list[dict[str, Any]],
        manifest: RunManifest,
        results: list[InstanceResult],
        concurrency: int,
    ) -> list:  # type: ignore[type-arg]
        semaphore = asyncio.Semaphore(concurrency)
        passed = 0
        failed = 0
        completed = 0
        log: list[tuple[str, str, str, str]] = []

        async def _run_with_limit(idx: int, inst: dict[str, Any]) -> InstanceResult:
            async with semaphore:
                return await self.run_instance(inst, manifest.run_id)

        tasks = [asyncio.create_task(_run_with_limit(i, inst)) for i, inst in enumerate(instances)]

        with Live(
            _console.render_str("Starting..."), console=_console, refresh_per_second=4
        ) as live:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                manifest.completed_instances += 1
                completed += 1

                verdict, detail = self._verdict_detail(r)
                if verdict == "PASS":
                    passed += 1
                else:
                    failed += 1
                log.append((str(completed), r.dataset_instance_id, verdict, detail))

                live.update(
                    self._build_live_table(len(instances), completed, passed, failed, log)
                )

        return tasks

    async def run(self, instances: list[dict[str, Any]]) -> RunManifest:
        manifest = self.create_manifest(instances)
        manifest.status = RunStatus.RUNNING
        self.artifact_store.save_manifest(manifest.run_id, manifest.model_dump(mode="json"))

        concurrency = self.config.concurrency
        results: list[InstanceResult] = []
        tasks: list[asyncio.Task[InstanceResult]] = []
        _interrupted = False

        try:
            if self._debug:
                # Debug mode: live scoreboard
                if concurrency <= 1:
                    await self._run_sequential_debug(instances, manifest, results)
                else:
                    tasks = await self._run_parallel_debug(
                        instances, manifest, results, concurrency
                    )
            else:
                # Normal mode: quiet
                if concurrency <= 1:
                    for instance in instances:
                        instance_result = await self.run_instance(instance, manifest.run_id)
                        results.append(instance_result)
                        manifest.completed_instances += 1
                else:
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
        except KeyboardInterrupt:
            manifest.status = RunStatus.FAILED
            for t in tasks:
                t.cancel()
            _interrupted = True
        except Exception:
            manifest.status = RunStatus.FAILED
            for t in tasks:
                t.cancel()
            _interrupted = False
        else:
            _interrupted = False
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
            if _interrupted:
                raise KeyboardInterrupt

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
