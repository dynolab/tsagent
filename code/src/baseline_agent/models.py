"""Model resolution: a model string plus optional endpoint overrides -> pydantic-ai Model.

Supported strings:

- ``"scripted"`` - the deterministic stand-in from ``scripted.py`` (no API needed)
- ``"test"`` - pydantic-ai's ``TestModel`` (calls every tool once, then answers)
- any pydantic-ai model string, e.g. ``"openai-chat:gpt-oss-20b"``, ``"ollama:qwen3:14b"``,
  ``"openrouter:openai/gpt-oss-20b"``, ``"anthropic:claude-sonnet-4-5"``; credentials come
  from the provider's usual environment variables (``OPENAI_API_KEY``, ...)

With ``base_url`` set, the model name is served through the OpenAI Chat Completions
protocol at that address - the common denominator of vLLM, Ollama, LM Studio, llama.cpp
and hosted gateways - which is how a ≤20B model on the project GPU or on a free endpoint
is wired in. ``anonymous=True`` strips the ``Authorization`` header the OpenAI SDK always
sends, for endpoints that serve unauthenticated traffic but reject a bogus bearer token
(OVHcloud AI Endpoints does). Retries and the timeout of the HTTP client are set here too,
because free tiers answer with 429 more often than not.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from pydantic_ai.models import Model, infer_model

from .scripted import MODEL_NAME as SCRIPTED
from .scripted import scripted_model

TEST = "test"


async def _strip_authorization(request: httpx.Request) -> None:
    request.headers.pop("Authorization", None)


def anonymous_http_client(timeout_s: float, *, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """An httpx client that drops the ``Authorization`` header from every request."""
    return httpx.AsyncClient(timeout=timeout_s, transport=transport, event_hooks={"request": [_strip_authorization]})


def build_model(
    name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    anonymous: bool = False,
    timeout_s: float = 120.0,
    max_retries: int = 5,
    scripted_options: dict[str, Any] | None = None,
) -> Model:
    if name == SCRIPTED:
        return scripted_model(**(scripted_options or {}))
    if name == TEST:
        from pydantic_ai.models.test import TestModel

        return TestModel()
    if base_url:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model_name = name.split(":", 1)[1] if ":" in name else name
        http_client = anonymous_http_client(timeout_s) if anonymous else httpx.AsyncClient(timeout=timeout_s)
        client = AsyncOpenAI(base_url=base_url, api_key=api_key or "unused", http_client=http_client, max_retries=max_retries, timeout=timeout_s)
        return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))
    return infer_model(name)


def model_slug(name: str) -> str:
    """A file-system-safe tag for a model string, e.g. ``openai-chat:gpt-oss-20b`` -> ``openai-chat_gpt-oss-20b``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"
