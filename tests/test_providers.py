"""Tests for the provider abstraction and wire-protocol adapters."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import crossfire.core.providers as providers_module
from crossfire.core.domain import ProviderConfiguration, Role, strip_model_prefix
from crossfire.core.providers import (
    AuthenticationError,
    EmptyResponseError,
    InsufficientCreditsError,
    RefusalResponseError,
    TruncatedResponseError,
    UnsupportedProtocolError,
    build_request,
    call_model,
    call_with_retry,
    extract_response_text,
    extract_usage,
    resolve_protocol,
)

_OPENCODE = ProviderConfiguration(
    name="opencode",
    base_url="https://opencode.ai/zen/v1",
    api_key_env="OPENCODE_API_KEY",
    requires_session=True,
)
_OPENROUTER = ProviderConfiguration(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
)
_SYNTHETIC = ProviderConfiguration(
    name="synthetic",
    base_url="https://api.synthetic.new/openai/v1",
    api_key_env="SYNTHETIC_API_KEY",
)


class TestStripModelPrefix:
    def test_strips_known_provider_prefix(self):
        assert strip_model_prefix("opencode:glm-5.3") == "glm-5.3"
        assert strip_model_prefix("openrouter:anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"

    def test_leaves_neutral_slug_untouched(self):
        assert strip_model_prefix("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"

    def test_leaves_colon_in_wire_id_untouched(self):
        assert strip_model_prefix("anthropic/claude-sonnet-4:batch") == "anthropic/claude-sonnet-4:batch"


class TestResolveProtocol:
    @pytest.mark.parametrize(
        ("wire_model_id", "expected"),
        [
            ("glm-5.3", "chat"),
            ("kimi-k3", "chat"),
            ("deepseek-v4.1-flash", "chat"),
            ("claude-fable-5-1", "messages"),
            ("qwen3.8-flash", "messages"),
            ("minimax-m3", "messages"),
            ("gpt-6-astra", "responses"),
            ("gpt-6-luna", "responses"),
            ("gpt-5.6-luna", "responses"),
            ("grok-4.7", "responses"),
        ],
    )
    def test_opencode_protocols(self, wire_model_id: str, expected: str):
        assert resolve_protocol(_OPENCODE, wire_model_id) == expected

    def test_openrouter_is_always_chat(self):
        assert resolve_protocol(_OPENROUTER, "anthropic/claude-sonnet-4") == "chat"

    def test_synthetic_is_always_chat(self):
        assert resolve_protocol(_SYNTHETIC, "syn:large:text") == "chat"
        assert resolve_protocol(_SYNTHETIC, "hf:zai-org/GLM-5.3-Flash") == "chat"

    def test_gemini_raises(self):
        with pytest.raises(UnsupportedProtocolError):
            resolve_protocol(_OPENCODE, "gemini-3.8-flash")


class TestBuildRequest:
    def test_chat_request(self):
        url, headers, payload = build_request(
            provider=_OPENCODE,
            protocol="chat",
            wire_model_id="glm-5.3",
            system_prompt="sys",
            user_prompt="usr",
            api_key="k",
            max_tokens=64,
            temperature=0.7,
            session_id="session-1",
        )
        assert url == "https://opencode.ai/zen/v1/chat/completions"
        assert headers["Authorization"] == "Bearer k"
        assert headers["x-opencode-session"] == "session-1"
        assert payload["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 64

    def test_chat_omits_temperature_when_none(self):
        _, _, payload = build_request(
            provider=_OPENCODE,
            protocol="chat",
            wire_model_id="glm-5.3",
            system_prompt="sys",
            user_prompt="usr",
            api_key="k",
            max_tokens=64,
            temperature=None,
            session_id="session-1",
        )
        assert "temperature" not in payload

    def test_session_header_omitted_when_not_required(self):
        _, headers, _ = build_request(
            provider=_OPENROUTER,
            protocol="chat",
            wire_model_id="anthropic/claude-sonnet-4",
            system_prompt="sys",
            user_prompt="usr",
            api_key="k",
            max_tokens=64,
            temperature=0.2,
            session_id="session-1",
        )
        assert "x-opencode-session" not in headers

    def test_messages_request(self):
        url, headers, payload = build_request(
            provider=_OPENCODE,
            protocol="messages",
            wire_model_id="claude-fable-5-1",
            system_prompt="sys",
            user_prompt="usr",
            api_key="k",
            max_tokens=128,
            temperature=None,
            session_id="session-1",
            reasoning_effort="high",
        )
        assert url == "https://opencode.ai/zen/v1/messages"
        assert headers["x-api-key"] == "k"
        assert headers["anthropic-version"]
        assert payload["system"] == "sys"
        assert payload["messages"] == [{"role": "user", "content": "usr"}]
        assert "temperature" not in payload
        # Reasoning effort has no messages-protocol equivalent, so it is never sent.
        assert "thinking" not in payload

    def test_responses_request_with_reasoning(self):
        url, headers, payload = build_request(
            provider=_OPENCODE,
            protocol="responses",
            wire_model_id="gpt-6-luna",
            system_prompt="sys",
            user_prompt="usr",
            api_key="k",
            max_tokens=128,
            temperature=None,
            session_id="session-1",
            reasoning_effort="low",
        )
        assert url == "https://opencode.ai/zen/v1/responses"
        assert headers["Authorization"] == "Bearer k"
        assert payload["instructions"] == "sys"
        assert payload["input"] == "usr"
        assert payload["max_output_tokens"] == 128
        assert payload["reasoning"] == {"effort": "low"}


class TestExtractResponseText:
    def test_chat(self):
        data = {"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]}
        assert extract_response_text("chat", data) == "hello"

    def test_messages(self):
        data = {"content": [{"type": "text", "text": "hello"}]}
        assert extract_response_text("messages", data) == "hello"

    def test_responses(self):
        data = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]}
        assert extract_response_text("responses", data) == "hello"

    def test_empty_raises(self):
        with pytest.raises(EmptyResponseError):
            extract_response_text("chat", {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})

    def test_truncated_empty_raises_truncated(self):
        data = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        with pytest.raises(TruncatedResponseError):
            extract_response_text("chat", data)

    def test_messages_truncated(self):
        with pytest.raises(TruncatedResponseError):
            extract_response_text("messages", {"stop_reason": "max_tokens", "content": []})

    def test_messages_refusal_raises_refusal(self):
        with pytest.raises(RefusalResponseError):
            extract_response_text("messages", {"stop_reason": "refusal", "content": []})

    def test_chat_content_filter_raises_refusal(self):
        data = {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
        with pytest.raises(RefusalResponseError):
            extract_response_text("chat", data)

    def test_responses_truncated(self):
        data = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        with pytest.raises(TruncatedResponseError):
            extract_response_text("responses", data)

    def test_responses_content_filter_raises_refusal(self):
        data = {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}
        with pytest.raises(RefusalResponseError):
            extract_response_text("responses", data)

    def test_empty_diagnostics_included(self):
        with pytest.raises(EmptyResponseError, match="stop_reason"):
            extract_response_text("messages", {"stop_reason": "end_turn", "content": []})


class TestExtractUsage:
    def test_chat_with_cost_and_cache(self):
        data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost": 0.0005,
                "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 5},
            }
        }
        usage = extract_usage("chat", data)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 20
        assert usage.cache_read_tokens == 80
        assert usage.cache_write_tokens == 5
        assert usage.cost == pytest.approx(0.0005)

    def test_chat_without_cost(self):
        usage = extract_usage("chat", {"usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        assert usage.cost is None

    def test_messages(self):
        data = {
            "usage": {
                "input_tokens": 50,
                "output_tokens": 8,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 3,
            }
        }
        usage = extract_usage("messages", data)
        assert (usage.input_tokens, usage.output_tokens) == (50, 8)
        assert (usage.cache_read_tokens, usage.cache_write_tokens) == (40, 3)

    def test_responses(self):
        data = {
            "usage": {
                "input_tokens": 21,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1},
            }
        }
        usage = extract_usage("responses", data)
        assert (usage.input_tokens, usage.output_tokens) == (21, 5)
        assert (usage.cache_read_tokens, usage.cache_write_tokens) == (4, 1)


class TestCallModel:
    def test_posts_built_request(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = request.content
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        async def scenario() -> dict[str, object]:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                return await call_model(
                    provider=_OPENCODE,
                    protocol="chat",
                    wire_model_id="glm-5.3",
                    system_prompt="sys",
                    user_prompt="usr",
                    api_key="secret",
                    max_tokens=64,
                    temperature=0.5,
                    semaphore=asyncio.Semaphore(1),
                    client=client,
                    session_id="session-1",
                )
            finally:
                await client.aclose()

        result = asyncio.run(scenario())
        assert result == {"choices": [{"message": {"content": "ok"}}]}
        assert captured["url"] == "https://opencode.ai/zen/v1/chat/completions"
        assert captured["auth"] == "Bearer secret"


class TestCallWithRetry:
    def test_payment_required_maps_to_insufficient_credits(self):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(402, request=request)

        async def factory() -> dict[str, object]:
            raise httpx.HTTPStatusError("402 Payment Required", request=request, response=response)

        with pytest.raises(InsufficientCreditsError, match=r"doughnuts|credits"):
            asyncio.run(
                call_with_retry(
                    factory,
                    role=Role.GENERATOR,
                    model="glm-5.3",
                    round_num=1,
                    provider_name="opencode",
                )
            )

    def test_zero_retries_is_single_shot(self):
        calls = {"count": 0}

        async def factory() -> dict[str, object]:
            calls["count"] += 1
            raise httpx.TimeoutException("too slow")

        with pytest.raises(RuntimeError, match="timed out after 1 attempts"):
            asyncio.run(
                call_with_retry(
                    factory,
                    role=Role.ENRICHER,
                    model="gpt-6-luna",
                    round_num=0,
                    provider_name="opencode",
                    max_retries=0,
                )
            )
        assert calls["count"] == 1

    def test_refusal_is_not_retried(self):
        calls = {"count": 0}

        async def factory() -> dict[str, object]:
            calls["count"] += 1
            data: dict[str, object] = {"stop_reason": "refusal", "content": []}
            extract_response_text("messages", data)
            return data

        with pytest.raises(RefusalResponseError):
            asyncio.run(
                call_with_retry(
                    factory,
                    role=Role.SYNTHESIZER,
                    model="claude-fable-5-1",
                    round_num=1,
                    provider_name="opencode",
                )
            )
        assert calls["count"] == 1

    def test_unauthorized_maps_to_authentication_error(self):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(401, request=request)

        async def factory() -> dict[str, object]:
            raise httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

        with pytest.raises(AuthenticationError, match="opencode rejected the API key"):
            asyncio.run(
                call_with_retry(
                    factory,
                    role=Role.GENERATOR,
                    model="glm-5.3",
                    round_num=1,
                    provider_name="opencode",
                )
            )

    def test_transient_error_is_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch):
        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(providers_module.asyncio, "sleep", no_sleep)

        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(503, request=request)
        calls = {"count": 0}

        async def factory() -> dict[str, object]:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.HTTPStatusError("503 Service Unavailable", request=request, response=response)
            return {"ok": True}

        result = asyncio.run(
            call_with_retry(
                factory,
                role=Role.GENERATOR,
                model="glm-5.3",
                round_num=1,
                provider_name="opencode",
            )
        )
        assert result == {"ok": True}
        assert calls["count"] == 2
