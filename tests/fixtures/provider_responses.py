"""Reusable mock HTTP responses for both OpenAI provider adapters.

Every helper returns a plain ``dict`` matching the JSON body the real API
would return, plus factory functions that produce ``httpx.Response`` objects
suitable for patching ``httpx.AsyncClient.post`` / ``.get``.
"""

from __future__ import annotations

from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Chat Completions fixtures
# ---------------------------------------------------------------------------


def chat_completions_success_body(
    *,
    content: str = "Hello, world!",
    model: str = "gpt-4o-2024-05-13",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Return a realistic ``/v1/chat/completions`` JSON body."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_completions_error_body(
    *,
    message: str = "Invalid API key",
    error_type: str = "invalid_request_error",
    code: str = "invalid_api_key",
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


# ---------------------------------------------------------------------------
# Responses API fixtures
# ---------------------------------------------------------------------------


def responses_success_body(
    *,
    content: str = "Hello, world!",
    model: str = "gpt-4o-2024-05-13",
    input_tokens: int = 12,
    output_tokens: int = 8,
    response_id: str = "resp_abc123",
) -> dict[str, Any]:
    """Return a realistic ``/v1/responses`` JSON body."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def responses_error_body(
    *,
    message: str = "Invalid API key",
    error_type: str = "invalid_request_error",
    code: str = "invalid_api_key",
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


# ---------------------------------------------------------------------------
# Models-list fixture (shared by both adapters for health-check)
# ---------------------------------------------------------------------------


def models_list_body() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        ],
    }


# ---------------------------------------------------------------------------
# httpx.Response factories
# ---------------------------------------------------------------------------


def make_httpx_response(
    *,
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    text: str = "",
) -> httpx.Response:
    """Build an ``httpx.Response`` suitable for mocking ``AsyncClient`` calls.

    Provide *either* ``json_body`` (for success responses) *or* ``text``
    (for error bodies).  ``httpx.Response`` only accepts one content source
    at a time.
    """
    request = httpx.Request("POST", "https://api.openai.com/mock")
    if json_body is not None:
        return httpx.Response(status_code=status_code, json=json_body, request=request)
    return httpx.Response(status_code=status_code, text=text, request=request)
