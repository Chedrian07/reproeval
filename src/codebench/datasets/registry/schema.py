"""Dataset registry schema — Pydantic models for dataset manifests and entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from codebench.core.models.common import ScenarioType, TrackType


class DatasetManifest(BaseModel):
    """Declarative manifest describing a benchmark dataset.

    Each YAML file in ``datasets/manifests/`` is deserialised into one of these.
    The schema deliberately avoids dataset-specific fields — datasets that need
    extra metadata encode it in the ``metadata`` dict so the registry stays
    generic.
    """

    name: str
    version: str
    description: str
    scenario_type: ScenarioType
    track_types: list[TrackType]
    instance_count: int
    source_url: str | None = None
    license: str = ""
    language: str = "python"
    splits: list[str] = Field(default_factory=lambda: ["test"])
    format_type: str = "jsonl"
    required_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Canonical registry key: ``name@version``."""
        return f"{self.name}@{self.version}"


class DatasetEntry(BaseModel):
    """A manifest enriched with its resolved local path or loader reference.

    The registry stores these — callers can resolve a ``DatasetConfig`` to an
    entry and then use the attached ``local_path`` or ``loader_ref`` to actually
    read instances.
    """

    manifest: DatasetManifest
    local_path: Path | None = None
    loader_ref: str | None = None
