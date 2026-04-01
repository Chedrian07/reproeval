"""CLI entrypoint for codebench."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codebench import __version__
from codebench.utils.dotenv import load_dotenv

app = typer.Typer(name="codebench", help="Benchmark framework for coding LLMs")
console = Console()


def _startup() -> None:
    """Load .env on every CLI invocation."""
    loaded = load_dotenv()
    if loaded:
        console.print(f"[dim].env loaded ({len(loaded)} vars)[/dim]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """codebench — Benchmark framework for coding LLMs."""
    _startup()
    if ctx.invoked_subcommand is None:
        console.print(f"codebench v{__version__} — use --help for commands")


@app.command()
def version() -> None:
    """Print version information."""
    console.print(f"codebench v{__version__}")


@app.command()
def run(
    config_path: Path = typer.Argument(None, help="Path to run config YAML/JSON (optional)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name to benchmark"),
    max_instances: int = typer.Option(None, "--max-instances", "-n", help="Max instances"),
) -> None:
    """Execute a benchmark run.

    Without a config file, reads settings from .env and benchmarks
    the specified dataset (or all available datasets with --dataset=all).
    """
    from codebench.cli.commands import execute_from_env, execute_run

    if config_path is not None:
        asyncio.run(execute_run(config_path, dry_run=dry_run))
    else:
        asyncio.run(
            execute_from_env(
                dataset_name=dataset or "all",
                max_instances=max_instances,
                dry_run=dry_run,
            )
        )


@app.command()
def list_datasets() -> None:
    """List available datasets in the registry."""
    from codebench.datasets.registry.registry import DatasetRegistry

    registry = DatasetRegistry()
    datasets = registry.list_datasets()
    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Scenario", style="yellow")
    table.add_column("Instances", style="magenta")

    for ds in datasets:
        table.add_row(ds.name, ds.version, ds.scenario_type, str(ds.instance_count))
    console.print(table)


@app.command()
def list_runs(
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Artifacts directory"),
) -> None:
    """List completed benchmark runs."""
    from codebench.artifacts.filesystem import FilesystemArtifactStore

    store = FilesystemArtifactStore(artifacts_dir)
    runs = store.list_runs()
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    for run_id in runs:
        manifest = store.load_manifest(run_id)
        status = manifest.get("status", "unknown")
        dataset = manifest.get("dataset_name", "?")
        total = manifest.get("total_instances", 0)
        summary = manifest.get("results_summary", {})
        pass_rate = summary.get("pass_rate", 0)
        console.print(
            f"  [cyan]{run_id}[/cyan]  {dataset}  "
            f"status={status}  instances={total}  "
            f"pass_rate={pass_rate:.1%}"
        )


@app.command()
def clean_runs(
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Artifacts directory"),
) -> None:
    """Remove incomplete (running/failed) runs from artifacts."""
    import shutil

    from codebench.artifacts.filesystem import FilesystemArtifactStore

    store = FilesystemArtifactStore(artifacts_dir)
    runs = store.list_runs()
    removed = 0
    for run_id in runs:
        manifest = store.load_manifest(run_id)
        status = manifest.get("status", "unknown")
        if status in ("running", "failed", "pending"):
            run_path = artifacts_dir / run_id
            shutil.rmtree(run_path, ignore_errors=True)
            console.print(f"  [red]removed[/red] {run_id} (status={status})")
            removed += 1
    if removed:
        console.print(f"\n{removed} run(s) removed.")
    else:
        console.print("[dim]No incomplete runs to clean.[/dim]")


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run ID to replay"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Artifacts directory"),
) -> None:
    """Replay and inspect a previous run."""
    from codebench.artifacts.filesystem import FilesystemArtifactStore

    store = FilesystemArtifactStore(artifacts_dir)
    manifest = store.load_manifest(run_id)
    console.print_json(json.dumps(manifest, indent=2, default=str))

    artifacts = store.list_artifacts(run_id)
    console.print(f"\n[bold]Artifacts ({len(artifacts)}):[/bold]")
    for a in artifacts:
        console.print(f"  {a}")


if __name__ == "__main__":
    app()
