"""Multi-provider LLM adapters: one abstraction, three gateways, three wire protocols.

OpenCode Zen, OpenCode Go, and OpenRouter each speak a slightly different dialect of "please write my thing": OpenAI
chat completions, Anthropic messages, and OpenAI responses. Rather than hard-code a favourite, Crossfire dispatches
on the protocol resolved for each model, and every provider brings its own auth, base URL, and opinion about whether
it will deign to accept a ``temperature``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import httpx

from crossfire.core import logging as log
from crossfire.core.domain import ProviderConfiguration, Role

try:
    _VERSION: str = version("crossfire")
except PackageNotFoundError:  # pragma: no cover - source checkout without installed metadata
    _VERSION = "0.0.0"

USER_AGENT = f"crossfire/{_VERSION}"
ANTHROPIC_VERSION = "2023-06-01"

Protocol = Literal["chat", "messages", "responses"]

_ENDPOINTS: dict[str, str] = {
    "chat": "chat/completions",
    "messages": "messages",
    "responses": "responses",
}

# Models that use the Anthropic messages protocol on the OpenCode gateways.
_MESSAGES_PREFIXES: tuple[str, ...] = ("claude-", "qwen", "minimax-")
# Models that use the OpenAI responses protocol on the OpenCode gateways.
_RESPONSES_PREFIXES: tuple[str, ...] = ("gpt-6-", "gpt-5.6-", "grok-", "muse-")

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
# Two total attempts. Reasoning models on the OpenCode gateways can take minutes per call, so a larger retry
# budget turns one slow model into a many-minute stall.
MAX_RETRIES = 1

# Where to send money when the money runs out, and a line to soften the blow.
_CREDITS_LAST_WORDS: dict[str, tuple[str, str]] = {
    "opencode": (
        "https://opencode.ai/auth",
        "OpenCode Zen has gone through your balance like Homer through a box of doughnuts.",
    ),
    "opencode-go": (
        "https://opencode.ai/auth",
        "OpenCode Go has hit its monthly ceiling. Wait for the clock to reset, or enable 'Use balance'.",
    ),
    "openrouter": (
        "https://openrouter.ai/credits",
        "OpenRouter ate all your credits. Feed it some more.",
    ),
}


class EmptyResponseError(Exception):
    """Raised when the LLM returns an empty or missing response."""


class TruncatedResponseError(RuntimeError):
    """Raised when a response exhausted its output budget before emitting any content."""


class RefusalResponseError(RuntimeError):
    """Raised when the model declined to answer (content filter or refusal), which retrying will not fix."""


class AuthenticationError(RuntimeError):
    """Raised when the provider rejects the API key (HTTP 401).

    Fatal, so it aborts rather than dropping a model.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"{provider} rejected the API key. Check the key in your environment, and that it belongs to {provider}."
        )


class InsufficientCreditsError(RuntimeError):
    """Raised when the provider returns 402 Payment Required, with a nudge toward the till."""

    def __init__(self, provider: str) -> None:
        url, quip = _CREDITS_LAST_WORDS.get(
            provider,
            ("https://opencode.ai/auth", f"{provider} is out of credits."),
        )
        super().__init__(f"{quip} Top up at {url}")


class UnsupportedProtocolError(RuntimeError):
    """Raised when a model uses a wire protocol Crossfire does not implement."""


@dataclass(frozen=True)
class Usage:
    """Token and (when reported) cost information from a single call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float | None = None


def resolve_protocol(provider: ProviderConfiguration, wire_model_id: str) -> Protocol:
    """Returns the wire protocol for *wire_model_id* served by *provider*."""
    if provider.name == "openrouter":
        return "chat"
    lowered = wire_model_id.lower()
    if lowered.startswith(_MESSAGES_PREFIXES):
        return "messages"
    if lowered.startswith(_RESPONSES_PREFIXES):
        return "responses"
    if lowered.startswith("gemini-"):
        raise UnsupportedProtocolError(
            f"{wire_model_id} speaks Google, a dialect Crossfire hasn't learned yet. Stick to the other labs."
        )
    return "chat"


def _openrouter_routing(provider: ProviderConfiguration) -> dict[str, Any]:
    """Builds OpenRouter's per-request provider routing object from the gateway's privacy settings.

    ``data_collection: "deny"`` keeps a request off providers that store data or train on it, and ``zdr: true``
    keeps it on endpoints that retain nothing. Both are filters rather than promises: when no endpoint for the
    model qualifies, OpenRouter fails the request instead of routing it somewhere the filters exclude.
    """
    routing: dict[str, Any] = {}
    if provider.deny_data_collection:
        routing["data_collection"] = "deny"
    if provider.require_zdr:
        routing["zdr"] = True
    return routing


def build_request(
    *,
    provider: ProviderConfiguration,
    protocol: Protocol,
    wire_model_id: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float | None,
    session_id: str,
    reasoning_effort: str = "",
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Builds the URL, headers, and JSON payload for a single call."""
    headers: dict[str, str] = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if provider.name == "openrouter":
        # OpenRouter likes to know who is knocking. It also unlocks the odd free-tier nicety.
        headers["HTTP-Referer"] = "https://github.com/ianreppel/crossfire"
        headers["X-Title"] = "Crossfire"
    if provider.requires_session and session_id:
        headers["x-opencode-session"] = session_id

    payload: dict[str, Any] = {"model": wire_model_id}

    if provider.name == "openrouter":
        routing = _openrouter_routing(provider)
        if routing:
            payload["provider"] = routing

    if protocol == "chat":
        headers["Authorization"] = f"Bearer {api_key}"
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
    elif protocol == "messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = ANTHROPIC_VERSION
        payload["max_tokens"] = max_tokens
        payload["system"] = system_prompt
        payload["messages"] = [{"role": "user", "content": user_prompt}]
        # Anthropic's newer frontier models reject ``temperature`` outright, so it is never sent here.
        # Reasoning effort is applied on the responses protocol only: on the messages protocol there is no
        # effort control (only an explicit thinking-token budget), and enabling it would raise cost rather
        # than lower it.
    elif protocol == "responses":
        headers["Authorization"] = f"Bearer {api_key}"
        payload["instructions"] = system_prompt
        payload["input"] = user_prompt
        payload["max_output_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

    return f"{provider.base_url}/{_ENDPOINTS[protocol]}", headers, payload


def _is_truncated(protocol: Protocol, data: dict[str, Any]) -> bool:
    """Returns True when the model stopped because it hit the output-token ceiling."""
    if protocol == "chat":
        choices = data.get("choices") or []
        return bool(choices) and choices[0].get("finish_reason") == "length"
    if protocol == "messages":
        return data.get("stop_reason") == "max_tokens"
    return data.get("status") == "incomplete" or data.get("incomplete_details") is not None


def _is_refusal(protocol: Protocol, data: dict[str, Any]) -> bool:
    """Returns True when the model declined to answer, so retrying the identical call cannot help."""
    if protocol == "chat":
        choices = data.get("choices") or []
        return bool(choices) and choices[0].get("finish_reason") == "content_filter"
    if protocol == "messages":
        return data.get("stop_reason") == "refusal"
    details = data.get("incomplete_details") or {}
    return details.get("reason") == "content_filter"


def _empty_diagnostics(protocol: Protocol, data: dict[str, Any]) -> str:
    """Builds a short description of why a response carried no text, for logs and errors."""
    if protocol == "chat":
        choices = data.get("choices") or []
        reason = choices[0].get("finish_reason") if choices else None
        return f"empty content (finish_reason={reason})"
    if protocol == "messages":
        blocks = [block.get("type") for block in data.get("content", []) if isinstance(block, dict)]
        return f"empty content (stop_reason={data.get('stop_reason')}, blocks={blocks})"
    return f"empty content (status={data.get('status')}, incomplete={data.get('incomplete_details')})"


def extract_response_text(protocol: Protocol, data: dict[str, Any]) -> str:
    """Extracts the assistant text from a provider response.

    Raises :class:`TruncatedResponseError` when the model produced nothing because it ran out of output tokens
    (retrying the identical call would only waste money), and :class:`EmptyResponseError` for any other empty result.
    """
    text = ""
    if protocol == "chat":
        choices = data.get("choices") or []
        if choices:
            text = choices[0].get("message", {}).get("content") or ""
    elif protocol == "messages":
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif protocol == "responses":
        for item in data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                    text += block.get("text", "")

    if text.strip():
        return text
    if _is_refusal(protocol, data):
        raise RefusalResponseError(f"The model refused to answer ({_empty_diagnostics(protocol, data)}).")
    if _is_truncated(protocol, data):
        raise TruncatedResponseError(
            "The model spent its whole output budget thinking and had nothing left to say. "
            "Raise max_output_tokens or hand the job to someone less verbose."
        )
    raise EmptyResponseError(f"LLM returned empty content ({_empty_diagnostics(protocol, data)})")


def extract_usage(protocol: Protocol, data: dict[str, Any]) -> Usage:
    """Extracts token usage and any reported cost from a provider response."""
    usage = data.get("usage") or {}
    if protocol == "chat":
        details = usage.get("prompt_tokens_details") or {}
        reported = usage.get("cost", usage.get("total_cost"))
        return Usage(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
            cache_write_tokens=int(details.get("cache_write_tokens", 0) or 0),
            cost=float(reported) if reported is not None else None,
        )
    if protocol == "messages":
        return Usage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        )
    details = usage.get("input_tokens_details") or {}
    return Usage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
        cache_write_tokens=int(details.get("cache_write_tokens", 0) or 0),
    )


async def call_model(
    *,
    provider: ProviderConfiguration,
    protocol: Protocol,
    wire_model_id: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    session_id: str,
    reasoning_effort: str = "",
) -> dict[str, Any]:
    """Performs a single provider call and returns the decoded JSON response."""
    url, headers, payload = build_request(
        provider=provider,
        protocol=protocol,
        wire_model_id=wire_model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        session_id=session_id,
        reasoning_effort=reasoning_effort,
    )
    async with semaphore:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


async def call_with_retry(
    coroutine_factory: Callable[[], Awaitable[dict[str, Any]]],
    *,
    role: Role,
    model: str,
    round_num: int,
    provider_name: str,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Retries up to *max_retries* on transient errors with exponential back-off.

    :class:`TruncatedResponseError` and :class:`RefusalResponseError` are deliberately *not* retried: the same
    call would burn the same budget and fail the same way. Callers handle them through the replacement path.
    ``max_retries=0`` makes a call single-shot, which the optional enrichment step uses so a flaky model cannot
    stall the run.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await coroutine_factory()
        except httpx.HTTPStatusError as exception:
            if exception.response.status_code == 401:
                raise AuthenticationError(provider_name) from exception
            if exception.response.status_code == 402:
                raise InsufficientCreditsError(provider_name) from exception
            if exception.response.status_code not in _TRANSIENT_STATUS_CODES:
                raise
            last_exception = exception
            if attempt < max_retries:
                log.log_retry(
                    round=round_num,
                    role=role,
                    model=model,
                    attempt=attempt + 1,
                    reason=f"HTTP {exception.response.status_code}",
                )
                await asyncio.sleep(2 ** (attempt + 1))
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            json.JSONDecodeError,
            EmptyResponseError,
        ) as exception:
            last_exception = exception
            if attempt < max_retries:
                log.log_retry(
                    round=round_num,
                    role=role,
                    model=model,
                    attempt=attempt + 1,
                    reason=str(type(exception).__name__),
                )
                await asyncio.sleep(2 ** (attempt + 1))

    assert last_exception is not None
    attempts = max_retries + 1
    if isinstance(last_exception, httpx.ConnectError):
        raise RuntimeError(
            f"Cannot reach {provider_name} after {attempts} attempts. Check your internet connection."
        ) from last_exception
    if isinstance(last_exception, httpx.TimeoutException):
        raise RuntimeError(
            f"{provider_name} timed out after {attempts} attempts. The model may be overloaded."
        ) from last_exception
    raise last_exception
