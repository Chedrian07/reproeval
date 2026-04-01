"""Unit tests for the OpenAI Chat Completions provider adapter.

Every test is fully offline -- ``httpx.AsyncClient`` is monkey-patched so no
real HTTP calls are made.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from codebench.core.config.settings import ProviderConfig
from codebench.core.models.common import ProviderRequest, ProviderResponse
from codebench.providers.openai_chat_completions.adapter import (
    OpenAIChatCompletionsProvider,
)

# Re-use shared fixtures
from tests.fixtures.provider_responses import (
    chat_completions_error_body,
    chat_completions_success_body,
    make_httpx_response,
    models_list_body,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ProviderConfig:
    defaults: dict[str, Any] = {
        "name": "openai_chat_completions",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {"prompt": "Say hello."}
    defaults.update(overrides)
    return ProviderRequest(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatCompletionsName:
    def test_name_is_correct(self) -> None:
        provider = OpenAIChatCompletionsProvider(_make_config())
        assert provider.name == "openai_chat_completions"


@pytest.mark.unit
class TestChatCompletionsCapabilities:
    def test_capabilities(self) -> None:
        provider = OpenAIChatCompletionsProvider(_make_config())
        caps = provider.get_capabilities()
        assert caps == {"streaming": False, "tool_use": True, "vision": False}


@pytest.mark.unit
class TestChatCompletionsRequestMapping:
    """Verify that ProviderRequest is correctly mapped to the Chat Completions payload."""

    async def test_basic_request_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        captured_kwargs: dict[str, Any] = {}

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured_kwargs.update(kwargs)
            captured_kwargs["url"] = url
            return make_httpx_response(
                status_code=200,
                json_body=chat_completions_success_body(),
            )

        client = provider._get_client()
        monkeypatch.setattr(client, "post", _mock_post)

        request = _make_request(prompt="Hello", temperature=0.5, max_tokens=100)
        await provider.generate(request)

        payload = captured_kwargs["json"]
        assert payload["model"] == "gpt-4o"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert "seed" not in payload
        assert "stop" not in payload
        assert captured_kwargs["url"] == "https://api.openai.com/v1/chat/completions"
        await provider.close()

    async def test_system_prompt_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        captured: dict[str, Any] = {}

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured.update(kwargs)
            return make_httpx_response(
                status_code=200,
                json_body=chat_completions_success_body(),
            )

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        request = _make_request(prompt="Hello", system_prompt="You are helpful.")
        await provider.generate(request)

        messages = captured["json"]["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        await provider.close()

    async def test_seed_and_stop_sequences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        captured: dict[str, Any] = {}

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured.update(kwargs)
            return make_httpx_response(
                status_code=200,
                json_body=chat_completions_success_body(),
            )

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        request = _make_request(seed=42, stop_sequences=["###", "END"])
        await provider.generate(request)

        payload = captured["json"]
        assert payload["seed"] == 42
        assert payload["stop"] == ["###", "END"]
        await provider.close()

    async def test_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(
            _make_config(base_url="https://custom.api.example.com/")
        )

        captured: dict[str, Any] = {}

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            return make_httpx_response(
                status_code=200,
                json_body=chat_completions_success_body(),
            )

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)
        await provider.generate(_make_request())

        assert captured["url"] == "https://custom.api.example.com/v1/chat/completions"
        await provider.close()


@pytest.mark.unit
class TestChatCompletionsResponseMapping:
    """Verify response parsing, usage capture, and latency measurement."""

    async def test_successful_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        body = chat_completions_success_body(
            content="Hi there!",
            model="gpt-4o-2024-05-13",
            prompt_tokens=10,
            completion_tokens=5,
        )

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=200, json_body=body)

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())

        assert isinstance(resp, ProviderResponse)
        assert resp.content == "Hi there!"
        assert resp.model == "gpt-4o-2024-05-13"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15
        assert resp.latency_ms > 0
        assert resp.raw_response == body
        assert resp.metadata["finish_reason"] == "stop"
        await provider.close()

    async def test_latency_is_positive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(
                status_code=200,
                json_body=chat_completions_success_body(),
            )

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())
        assert resp.latency_ms >= 0
        await provider.close()

    async def test_raw_response_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())
        body = chat_completions_success_body()

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=200, json_body=body)

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())
        assert resp.raw_response["id"] == "chatcmpl-abc123"
        assert "choices" in resp.raw_response
        await provider.close()


@pytest.mark.unit
class TestChatCompletionsErrorHandling:
    """Verify that errors produce structured responses instead of exceptions."""

    async def test_api_error_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        error_body = json.dumps(chat_completions_error_body(message="Rate limit exceeded"))

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=429, text=error_body)

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())

        assert resp.content == ""
        assert resp.model == "unknown"
        assert resp.metadata.get("error") is True
        assert resp.metadata.get("status_code") == 429
        assert resp.raw_response["status_code"] == 429
        assert resp.latency_ms >= 0
        await provider.close()

    async def test_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())

        assert resp.content == ""
        assert resp.metadata.get("error") is True
        assert resp.metadata.get("status_code") == 0
        assert "Connection refused" in resp.raw_response["error"]
        await provider.close()

    async def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIChatCompletionsProvider(_make_config(api_key_env="NONEXISTENT_KEY_VAR"))

        with pytest.raises(ValueError, match="API key not found"):
            await provider.generate(_make_request())
        await provider.close()

    async def test_server_error_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_post(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=500, text="Internal Server Error")

        monkeypatch.setattr(provider._get_client(), "post", _mock_post)

        resp = await provider.generate(_make_request())
        assert resp.metadata.get("error") is True
        assert resp.metadata.get("status_code") == 500
        await provider.close()


@pytest.mark.unit
class TestChatCompletionsHealthCheck:
    async def test_health_check_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_get(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=200, json_body=models_list_body())

        monkeypatch.setattr(provider._get_client(), "get", _mock_get)

        assert await provider.check_health() is True
        await provider.close()

    async def test_health_check_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_get(url: str, **kwargs: Any) -> httpx.Response:
            return make_httpx_response(status_code=401, text="Unauthorized")

        monkeypatch.setattr(provider._get_client(), "get", _mock_get)

        assert await provider.check_health() is False
        await provider.close()

    async def test_health_check_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        provider = OpenAIChatCompletionsProvider(_make_config())

        async def _mock_get(url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("unreachable")

        monkeypatch.setattr(provider._get_client(), "get", _mock_get)

        assert await provider.check_health() is False
        await provider.close()
