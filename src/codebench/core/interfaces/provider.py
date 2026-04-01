"""Provider interface — provider-neutral abstraction for LLM calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from codebench.core.models.common import ProviderRequest, ProviderResponse


class ProviderInterface(ABC):
    """Base interface for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier."""

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Send a prompt and return a provider-neutral response."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Return True if the provider is reachable."""

    def get_capabilities(self) -> dict[str, Any]:
        """Return provider capability flags (override in subclasses)."""
        return {"streaming": False, "tool_use": False, "vision": False}
