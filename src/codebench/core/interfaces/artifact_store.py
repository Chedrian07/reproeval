"""Artifact store interface — persists run artifacts for audit and replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactStore(ABC):
    """Base interface for artifact storage."""

    @abstractmethod
    def save_artifact(self, run_id: str, instance_id: str, name: str, data: bytes | str) -> str:
        """Save an artifact and return its storage path."""

    @abstractmethod
    def load_artifact(self, run_id: str, instance_id: str, name: str) -> bytes:
        """Load a previously saved artifact."""

    @abstractmethod
    def save_manifest(self, run_id: str, manifest: dict[str, Any]) -> str:
        """Save the run manifest."""

    @abstractmethod
    def load_manifest(self, run_id: str) -> dict[str, Any]:
        """Load a run manifest."""

    @abstractmethod
    def list_runs(self) -> list[str]:
        """List all run IDs."""

    @abstractmethod
    def list_artifacts(self, run_id: str, instance_id: str | None = None) -> list[str]:
        """List artifact names for a run (optionally filtered by instance)."""

    @abstractmethod
    def get_base_path(self) -> Path:
        """Return the base storage path."""
