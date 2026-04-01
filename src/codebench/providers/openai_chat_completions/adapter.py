"""OpenAI Chat Completions provider adapter.

Maps the provider-neutral ProviderRequest/ProviderResponse to the
``POST /v1/chat/completions`` endpoint using ``httpx.AsyncClient``.
No OpenAI SDK dependency -- just raw HTTP.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from codebench.core.config.settings import ProviderConfig
from codebench.core.interfaces.provider import ProviderInterface
from codebench.core.models.common import ProviderRequest, ProviderResponse, TokenUsage


class OpenAIChatCompletionsProvider(ProviderInterface):
    """Adapter for the OpenAI Chat Completions API (``/v1/chat/completions``)."""

    _DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._api_key_env = config.api_key_env or "OPENAI_API_KEY"
        self._client: httpx.AsyncClient | None = None

    # -- ProviderInterface ----------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "openai_chat_completions"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Send *request* to the Chat Completions API and return a neutral response."""
        client = self._get_client()
        payload = self._build_payload(request)
        url = f"{self._base_url}/v1/chat/completions"

        t0 = time.monotonic()
        try:
            http_resp = await client.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=120.0,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

            if http_resp.status_code != 200:
                return self._error_response(
                    status_code=http_resp.status_code,
                    body=http_resp.text,
                    latency_ms=latency_ms,
                )

            data: dict[str, Any] = http_resp.json()
            return self._parse_response(data, latency_ms)

        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._error_response(
                status_code=0,
                body=str(exc),
                latency_ms=latency_ms,
            )

    async def check_health(self) -> bool:
        """Return *True* if the API responds to a lightweight models-list call."""
        client = self._get_client()
        try:
            resp = await client.get(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
                timeout=10.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        return {"streaming": False, "tool_use": True, "vision": False}

    # -- internal helpers -----------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self._api_key_env, "")
        if not key:
            raise ValueError(
                f"API key not found. Set the {self._api_key_env} environment variable."
            )
        return key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        return payload

    def _parse_response(
        self,
        data: dict[str, Any],
        latency_ms: float,
    ) -> ProviderResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens"),
        )

        return ProviderResponse(
            content=content,
            model=data.get("model", self._config.model),
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
            metadata={"finish_reason": choice.get("finish_reason")},
        )

    @staticmethod
    def _error_response(
        *,
        status_code: int,
        body: str,
        latency_ms: float,
    ) -> ProviderResponse:
        return ProviderResponse(
            content="",
            model="unknown",
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            latency_ms=latency_ms,
            raw_response={"error": body, "status_code": status_code},
            metadata={"error": True, "status_code": status_code},
        )

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
