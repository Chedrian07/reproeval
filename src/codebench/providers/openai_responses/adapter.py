"""OpenAI Responses API provider adapter.

Maps the provider-neutral ProviderRequest/ProviderResponse to the
``POST /v1/responses`` endpoint using ``httpx.AsyncClient``.
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


class OpenAIResponsesProvider(ProviderInterface):
    """Adapter for the OpenAI Responses API (``/v1/responses``)."""

    _DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._api_key_env = config.api_key_env or "OPENAI_API_KEY"
        self._client: httpx.AsyncClient | None = None

    # -- ProviderInterface ----------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return "openai_responses"

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Send *request* to the Responses API and return a neutral response."""
        client = self._get_client()
        payload = self._build_payload(request)
        url = f"{self._base_url}/v1/responses"

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
        return {"streaming": False, "tool_use": True, "vision": True}

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
        """Build the Responses API request body.

        The Responses API uses ``input`` (a string or list of input items)
        rather than the ``messages`` array.  When a *system_prompt* is
        provided it is passed via the ``instructions`` top-level field.
        """
        payload: dict[str, Any] = {
            "model": self._config.model,
            "input": request.prompt,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_prompt:
            payload["instructions"] = request.system_prompt
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        # Reasoning effort (low / medium / high)
        reasoning_effort = self._config.extra.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        return payload

    def _parse_response(
        self,
        data: dict[str, Any],
        latency_ms: float,
    ) -> ProviderResponse:
        """Extract content and usage from the Responses API JSON body.

        The Responses API returns ``output`` as a list of output items.
        Each item with ``type == "message"`` contains a ``content`` list;
        we concatenate all ``output_text`` entries from those content blocks.
        """
        content_parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        content_parts.append(block.get("text", ""))
        content = "".join(content_parts)

        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("total_tokens"),
        )

        return ProviderResponse(
            content=content,
            model=data.get("model", self._config.model),
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
            metadata={"response_id": data.get("id")},
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
