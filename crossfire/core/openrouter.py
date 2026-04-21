"""OpenRouter HTTP client with retry and cost extraction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from crossfire.core import logging as log
from crossfire.core.domain import CostEntry, Role

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def strip_model_prefix(model: str) -> str:
    """Strips the ``openrouter:`` vendor prefix from a model ID."""
    return model.removeprefix("openrouter:")


_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 2


class EmptyResponseError(Exception):
    """Raised when the LLM returns an empty or missing response."""


class InsufficientCreditsError(RuntimeError):
    """Raised when OpenRouter returns 402 Payment Required."""

    def __init__(self) -> None:
        super().__init__("OpenRouter ate all your credits. Feed it some more at https://openrouter.ai/credits")


async def call_openrouter(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    async with semaphore:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ianreppel/crossfire",
            },
            json={
                "model": strip_model_prefix(model),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result


def extract_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if choices:
        content: str = choices[0].get("message", {}).get("content", "")
        if content:
            return content
    raise EmptyResponseError("LLM returned empty content")


def extract_cost(data: dict[str, Any], model: str, role: Role, round_num: int) -> CostEntry:
    usage = data.get("usage", {})
    cost = usage["cost"] if "cost" in usage else usage.get("total_cost")
    return CostEntry(
        model=model,
        role=role,
        round=round_num,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        cost=float(cost) if cost is not None else None,
    )


async def call_with_retry(
    coroutine_factory: Callable[[], Awaitable[dict[str, Any]]],
    *,
    role: Role,
    model: str,
    round_num: int,
) -> dict[str, Any]:
    """Retries up to MAX_RETRIES on transient errors with exponential back-off."""
    last_exception: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await coroutine_factory()
        except httpx.HTTPStatusError as exception:
            if exception.response.status_code == 402:
                raise InsufficientCreditsError() from exception
            if exception.response.status_code not in _TRANSIENT_STATUS_CODES:
                raise
            last_exception = exception
            if attempt < MAX_RETRIES:
                delay = 2 ** (attempt + 1)
                log.log_retry(
                    round=round_num,
                    role=role,
                    model=model,
                    attempt=attempt + 1,
                    reason=f"HTTP {exception.response.status_code}",
                )
                await asyncio.sleep(delay)
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            json.JSONDecodeError,
            EmptyResponseError,
        ) as exception:
            last_exception = exception
            if attempt < MAX_RETRIES:
                delay = 2 ** (attempt + 1)
                log.log_retry(
                    round=round_num,
                    role=role,
                    model=model,
                    attempt=attempt + 1,
                    reason=str(type(exception).__name__),
                )
                await asyncio.sleep(delay)

    assert last_exception is not None
    if isinstance(last_exception, httpx.ConnectError):
        raise RuntimeError(
            f"Cannot reach OpenRouter after {MAX_RETRIES + 1} attempts. Check your internet connection."
        ) from last_exception
    if isinstance(last_exception, httpx.TimeoutException):
        raise RuntimeError(
            f"OpenRouter timed out after {MAX_RETRIES + 1} attempts. The model may be overloaded."
        ) from last_exception
    raise last_exception
