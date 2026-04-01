"""Global configuration for codebench runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from codebench.core.models.common import ScenarioType, TrackType


class ProviderConfig(BaseModel):
    """Configuration for a specific provider."""

    name: str
    model: str
    api_key_env: str = ""
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 40960
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SandboxConfig(BaseModel):
    """Configuration for sandbox execution."""

    backend: str = "docker"
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    memory_limit_mb: int = Field(default=512, ge=64, le=16384)
    network_enabled: bool = False
    image: str = "python:3.12-slim"
    max_output_bytes: int = Field(default=1_000_000, ge=1024, le=100_000_000)


class DatasetConfig(BaseModel):
    """Configuration for a dataset source."""

    name: str
    version: str = "latest"
    path: str | None = None
    split: str = "test"
    max_instances: int | None = None


class RunConfig(BaseModel):
    """Top-level run configuration."""

    provider: ProviderConfig
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    dataset: DatasetConfig
    scenario_type: ScenarioType
    track_type: TrackType = TrackType.LM_ONLY
    artifacts_dir: Path = Path("artifacts")
    seed: int | None = None
    prompt_version: str | None = None
    harness_version: str = "0.1.0"
    concurrency: int = Field(default=1, ge=1, le=128)
    dry_run: bool = False
