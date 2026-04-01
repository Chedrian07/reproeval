"""Sandbox runner interface — executes code in an isolated environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from codebench.core.models.common import ExecutionResult


class SandboxRunner(ABC):
    """Base interface for sandbox execution backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique sandbox backend identifier."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_seconds: int = 30,
        memory_limit_mb: int = 512,
        network_enabled: bool = False,
        env: dict[str, str] | None = None,
        workdir_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run code in the sandbox and return structured results."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release sandbox resources."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Return True if the sandbox backend is available."""

    def get_default_policy(self) -> dict[str, Any]:
        """Return default execution policy."""
        return {
            "timeout_seconds": 30,
            "memory_limit_mb": 512,
            "network_enabled": False,
            "max_output_bytes": 1_000_000,
        }
