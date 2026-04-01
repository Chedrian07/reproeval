"""CLI command implementations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from codebench.artifacts.filesystem import FilesystemArtifactStore
from codebench.core.config.settings import (
    DatasetConfig,
    ProviderConfig,
    RunConfig,
    SandboxConfig,
)
from codebench.core.models.common import ScenarioType
from codebench.core.runner.pipeline import BenchmarkPipeline

console = Console()

# Maps dataset names to their scenario type and a bundled fixture path (if any).
_DATASET_SCENARIOS: dict[str, ScenarioType] = {
    "humaneval_plus": ScenarioType.FUNCTION_CODEGEN,
    "mbpp_plus": ScenarioType.FUNCTION_CODEGEN,
    "bigcodebench_hard": ScenarioType.FUNCTION_CODEGEN,
    "swe_bench_lite": ScenarioType.REPO_PATCH,
    "swe_bench_verified": ScenarioType.REPO_PATCH,
    "livecodebench_lite": ScenarioType.CONTEST_CODEGEN,
    "cruxeval": ScenarioType.CODE_REASONING,
}


# ---------------------------------------------------------------------------
# Config file based run
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> RunConfig:
    """Load a run config from YAML or JSON."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return RunConfig.model_validate(raw)


def _resolve_provider(config: RunConfig):  # type: ignore[no-untyped-def]
    """Instantiate the provider from config."""
    name = config.provider.name
    if name == "openai_chat_completions":
        from codebench.providers.openai_chat_completions.adapter import (
            OpenAIChatCompletionsProvider,
        )

        return OpenAIChatCompletionsProvider(config.provider)
    elif name == "openai_responses":
        from codebench.providers.openai_responses.adapter import (
            OpenAIResponsesProvider,
        )

        return OpenAIResponsesProvider(config.provider)
    else:
        raise ValueError(f"Unknown provider: {name}")


def _resolve_scenario(config: RunConfig):  # type: ignore[no-untyped-def]
    """Instantiate the scenario adapter from config."""
    scenario_map: dict[ScenarioType, tuple[str, str]] = {
        ScenarioType.FUNCTION_CODEGEN: (
            "codebench.scenarios.function_codegen.adapter",
            "FunctionCodegenAdapter",
        ),
        ScenarioType.REPO_PATCH: (
            "codebench.scenarios.repo_patch.adapter",
            "RepoPatchAdapter",
        ),
        ScenarioType.CONTEST_CODEGEN: (
            "codebench.scenarios.contest_codegen.adapter",
            "ContestCodegenAdapter",
        ),
        ScenarioType.CODE_REASONING: (
            "codebench.scenarios.code_reasoning.adapter",
            "CodeReasoningAdapter",
        ),
        ScenarioType.TERMINAL_AGENT: (
            "codebench.scenarios.terminal_agent.adapter",
            "TerminalAgentAdapter",
        ),
    }
    entry = scenario_map.get(config.scenario_type)
    if entry is None:
        raise ValueError(f"No adapter for scenario: {config.scenario_type}")

    import importlib

    module = importlib.import_module(entry[0])
    adapter_cls = getattr(module, entry[1])
    return adapter_cls()


def _resolve_sandbox(config: RunConfig):  # type: ignore[no-untyped-def]
    """Instantiate the sandbox from config."""
    if config.sandbox.backend == "docker":
        from codebench.sandbox.docker.runner import DockerSandboxRunner

        return DockerSandboxRunner(config.sandbox)
    elif config.sandbox.backend == "local":
        from codebench.sandbox.docker.local_runner import LocalProcessRunner

        return LocalProcessRunner(config.sandbox)
    return None


def _load_instances(config: RunConfig) -> list[dict]:  # type: ignore[type-arg]
    """Load dataset instances."""
    from codebench.datasets.registry.registry import DatasetRegistry

    registry = DatasetRegistry()
    return registry.load_instances(config.dataset)


async def execute_run(config_path: Path, dry_run: bool = False) -> None:
    """Execute a full benchmark run from a config file."""
    console.print(f"[bold]Loading config from {config_path}[/bold]")
    config = _load_config(config_path)

    if dry_run:
        config.dry_run = True
        console.print("[yellow]Dry run mode — will validate but not execute[/yellow]")

    console.print(f"  Provider: {config.provider.name} / {config.provider.model}")
    console.print(f"  Scenario: {config.scenario_type.value}")
    console.print(f"  Dataset:  {config.dataset.name} v{config.dataset.version}")

    provider = _resolve_provider(config)
    scenario = _resolve_scenario(config)
    sandbox = _resolve_sandbox(config)
    artifact_store = FilesystemArtifactStore(config.artifacts_dir)

    if dry_run:
        console.print("[green]Dry run validation passed.[/green]")
        return

    instances = _load_instances(config)
    if config.dataset.max_instances:
        instances = instances[: config.dataset.max_instances]

    console.print(f"  Instances: {len(instances)}")

    pipeline = BenchmarkPipeline(
        config=config,
        provider=provider,
        scenario=scenario,
        sandbox=sandbox,
        artifact_store=artifact_store,
    )

    try:
        manifest = await pipeline.run(instances)
        console.print(f"\n[bold green]Run complete: {manifest.run_id}[/bold green]")
        console.print(f"  Status: {manifest.status.value}")
        summary = manifest.results_summary
        console.print(
            f"  Results: {summary.get('passed', 0)}/{summary.get('total', 0)} passed "
            f"({summary.get('pass_rate', 0):.1%})"
        )
    finally:
        if hasattr(provider, "close"):
            await provider.close()
        if sandbox is not None and hasattr(sandbox, "cleanup"):
            await sandbox.cleanup()


# ---------------------------------------------------------------------------
# .env based run (no config file needed)
# ---------------------------------------------------------------------------


def _build_config_from_env(
    dataset_name: str,
    dataset_path: str | None = None,
    max_instances: int | None = None,
) -> RunConfig:
    """Build a RunConfig from environment variables (loaded from .env)."""
    provider_name = os.environ.get("CODEBENCH_PROVIDER", "openai_responses")
    model = os.environ.get("CODEBENCH_MODEL", "gpt-4o")
    api_key = os.environ.get("CODEBENCH_API_KEY", "")
    base_url = os.environ.get("CODEBENCH_BASE_URL", "")
    sandbox_backend = os.environ.get("CODEBENCH_SANDBOX", "local")
    env_max = os.environ.get("CODEBENCH_MAX_INSTANCES", "")
    reasoning_effort = os.environ.get("CODEBENCH_REASONING_EFFORT", "")
    concurrency_str = os.environ.get("CODEBENCH_CONCURRENCY", "1")

    if not api_key:
        raise ValueError("CODEBENCH_API_KEY not set. Create a .env file or export the variable.")

    # Resolve max_instances: CLI flag > env var > None
    resolved_max: int | None = max_instances
    if resolved_max is None and env_max:
        resolved_max = int(env_max)

    scenario_type = _DATASET_SCENARIOS.get(dataset_name)
    if scenario_type is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {', '.join(sorted(_DATASET_SCENARIOS))}"
        )

    # Write the API key to a temp env var so the provider can read it
    key_env_name = "CODEBENCH_API_KEY"
    os.environ[key_env_name] = api_key

    extra: dict[str, Any] = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    # Sampling parameters from env
    for env_key, param_key in [
        ("CODEBENCH_TOP_P", "top_p"),
        ("CODEBENCH_MIN_P", "min_p"),
        ("CODEBENCH_TOP_K", "top_k"),
        ("CODEBENCH_FREQUENCY_PENALTY", "frequency_penalty"),
        ("CODEBENCH_PRESENCE_PENALTY", "presence_penalty"),
    ]:
        val = os.environ.get(env_key, "")
        if val:
            extra[param_key] = float(val) if "." in val else int(val)

    temperature_str = os.environ.get("CODEBENCH_TEMPERATURE", "")
    temperature = float(temperature_str) if temperature_str else 0.0

    return RunConfig(
        provider=ProviderConfig(
            name=provider_name,
            model=model,
            api_key_env=key_env_name,
            base_url=base_url or None,
            temperature=temperature,
            extra=extra,
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            version="1.0",
            path=dataset_path,
            max_instances=resolved_max,
        ),
        scenario_type=scenario_type,
        sandbox=SandboxConfig(
            backend=sandbox_backend,
            image=os.environ.get("CODEBENCH_SANDBOX_IMAGE", "codebench-sandbox"),
        ),
        artifacts_dir=Path("artifacts"),
        concurrency=int(concurrency_str) if concurrency_str else 1,
    )


def _find_dataset_file(dataset_name: str) -> str | None:
    """Look for a local data file for a dataset. data/ takes priority."""
    candidates = [
        Path(f"data/{dataset_name}.jsonl"),
        Path(f"data/{dataset_name}.json"),
        Path(f"tests/fixtures/{dataset_name}_sample.json"),
        Path(f"tests/fixtures/{dataset_name}_sample.jsonl"),
        Path(f"tests/fixtures/{dataset_name}.json"),
        Path(f"tests/fixtures/{dataset_name}.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


async def _run_single_dataset(
    dataset_name: str,
    max_instances: int | None,
    dry_run: bool,
) -> dict | None:  # type: ignore[type-arg]
    """Run a single dataset benchmark. Returns results_summary or None on skip."""
    data_path = _find_dataset_file(dataset_name)
    if data_path is None:
        console.print(f"  [dim]{dataset_name}: no data file found — skipped[/dim]")
        return None

    try:
        config = _build_config_from_env(dataset_name, data_path, max_instances)
    except ValueError as e:
        console.print(f"  [red]{dataset_name}: {e}[/red]")
        return None

    if dry_run:
        console.print(f"  [yellow]{dataset_name}: dry run OK[/yellow]")
        return None

    provider = _resolve_provider(config)
    scenario = _resolve_scenario(config)
    sandbox = _resolve_sandbox(config)
    artifact_store = FilesystemArtifactStore(config.artifacts_dir)

    instances = _load_instances(config)
    if config.dataset.max_instances:
        instances = instances[: config.dataset.max_instances]

    if not instances:
        console.print(f"  [dim]{dataset_name}: 0 instances — skipped[/dim]")
        return None

    console.print(
        f"\n[bold]▶ {dataset_name}[/bold]  "
        f"({len(instances)} instances, {config.scenario_type.value})"
    )

    pipeline = BenchmarkPipeline(
        config=config,
        provider=provider,
        scenario=scenario,
        sandbox=sandbox,
        artifact_store=artifact_store,
    )

    try:
        manifest = await pipeline.run(instances)
        summary = manifest.results_summary
        passed = summary.get("passed", 0)
        total = summary.get("total", 0)
        rate = summary.get("pass_rate", 0)
        status_color = "green" if rate == 1.0 else ("yellow" if rate > 0 else "red")
        console.print(
            f"  [{status_color}]{passed}/{total} passed ({rate:.1%})[/{status_color}]"
            f"  run_id={manifest.run_id}"
        )
        return {"dataset": dataset_name, "run_id": manifest.run_id, **summary}
    finally:
        if hasattr(provider, "close"):
            await provider.close()
        if sandbox is not None and hasattr(sandbox, "cleanup"):
            await sandbox.cleanup()


async def execute_from_env(
    dataset_name: str = "all",
    max_instances: int | None = None,
    dry_run: bool = False,
) -> None:
    """Execute benchmarks using .env configuration."""
    provider_name = os.environ.get("CODEBENCH_PROVIDER", "openai_responses")
    model = os.environ.get("CODEBENCH_MODEL", "?")
    base_url = os.environ.get("CODEBENCH_BASE_URL", "")
    reasoning_effort = os.environ.get("CODEBENCH_REASONING_EFFORT", "")
    concurrency = os.environ.get("CODEBENCH_CONCURRENCY", "1")

    console.print("[bold]codebench — .env configuration[/bold]")
    console.print(f"  Provider: {provider_name} / {model}")
    if base_url:
        console.print(f"  Base URL: {base_url}")
    if reasoning_effort:
        console.print(f"  Reasoning: {reasoning_effort}")
    if concurrency and int(concurrency) > 1:
        console.print(f"  Concurrency: {concurrency}")

    if dataset_name == "all":
        datasets = list(_DATASET_SCENARIOS.keys())
        console.print(f"  Mode: benchmark all ({len(datasets)} datasets)")
    else:
        datasets = [dataset_name]

    if dry_run:
        console.print("[yellow]Dry run mode[/yellow]")

    results: list[dict] = []  # type: ignore[type-arg]
    for ds in datasets:
        result = await _run_single_dataset(ds, max_instances, dry_run)
        if result is not None:
            results.append(result)

    if not results:
        if dry_run:
            console.print("\n[green]Dry run complete.[/green]")
        else:
            console.print("\n[dim]No datasets with data files found.[/dim]")
        return

    # Summary table
    console.print("\n")
    table = Table(title="Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Total", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Run ID", style="dim")

    total_passed = 0
    total_count = 0
    for r in results:
        p, t = r.get("passed", 0), r.get("total", 0)
        rate = r.get("pass_rate", 0)
        total_passed += p
        total_count += t
        rate_str = f"{rate:.1%}"
        rate_style = "green" if rate == 1.0 else ("yellow" if rate > 0 else "red")
        table.add_row(
            r["dataset"],
            str(p),
            str(t),
            f"[{rate_style}]{rate_str}[/{rate_style}]",
            r.get("run_id", ""),
        )

    if len(results) > 1:
        overall_rate = total_passed / total_count if total_count else 0
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_passed}[/bold]",
            f"[bold]{total_count}[/bold]",
            f"[bold]{overall_rate:.1%}[/bold]",
            "",
        )

    console.print(table)
