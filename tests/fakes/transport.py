"""A fake provider transport for the offline suite.

Every test runs under ``env -i``: no network, no provider key, no live receipts. An autouse fixture replaces the async
HTTP client with an :class:`httpx.MockTransport` that replays canned completions per role, records every request, and
raises :class:`UnexpectedRequestError` for anything it was not told to expect. The role is read from the prompt itself,
so a run that reaches a model it should not have reached is loud rather than silently answered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx

# The fake key the fixture puts in the environment. It never leaves the process.
FAKE_API_KEY = "crossfire-offline-test-key"

_DEFAULT_TRANSLATOR_PREFIX = "[TRANSLATE "


class UnexpectedRequestError(AssertionError):
    """Raised when a request reaches the fake transport that the test did not plan for."""


@dataclass
class RecordedRequest:
    """One request the fake transport answered."""

    role: str
    model: str
    url: str
    user_prompt: str
    system_prompt: str


@dataclass
class FakeTransport:
    """Replays canned completions per role and refuses everything else."""

    requests: list[RecordedRequest] = field(default_factory=list)
    _queues: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _defaults: dict[str, str] = field(default_factory=dict)
    _allowed_roles: set[str] | None = None
    identity_translator: bool = True

    def queue(self, role: str, *texts: str) -> None:
        """Queues replies for *role*, consumed in order before any default reply."""
        self._queues[role].extend(texts)

    def default(self, *roles: str) -> None:
        """Opts *roles* into the canned default reply once their queue is empty."""
        for role in roles:
            self._defaults[role] = _DEFAULT_RESPONSES[role]

    def allow(self, *roles: str) -> None:
        """Restricts the fake to *roles*; a request for any other role raises."""
        self._allowed_roles = set(roles)

    def count(self, role: str) -> int:
        """Returns how many requests the fake answered for *role*."""
        return sum(1 for request in self.requests if request.role == role)

    def count_model(self, model: str) -> int:
        """Returns how many requests named *model*."""
        return sum(1 for request in self.requests if request.model == model)

    def texts(self, role: str) -> list[str]:
        """Returns the prompts the fake saw for *role*."""
        return [request.user_prompt for request in self.requests if request.role == role]

    def classify(self, user_prompt: str) -> str:
        """Maps a user prompt to the role that built it."""
        if _DEFAULT_TRANSLATOR_PREFIX in user_prompt:
            return "translator"
        if "[CROSS-BRANCH ADJUDICATION]" in user_prompt:
            return "adjudicator"
        if "[REVIEW for candidate" in user_prompt:
            return "synthesizer"
        if "[CANDIDATE (" in user_prompt:
            return "reviewer"
        if "[ORIGINAL INSTRUCTION]" in user_prompt:
            return "enricher"
        if "[INSTRUCTION]" in user_prompt:
            return "generator"
        raise UnexpectedRequestError(f"Unclassifiable request prompt: {user_prompt[:120]!r}")

    def reply_for(self, role: str, user_prompt: str) -> str:
        """Returns the next queued reply for *role*, else its default, else raises."""
        if self._allowed_roles is not None and role not in self._allowed_roles:
            raise UnexpectedRequestError(f"Unexpected {role} request: the test allowed {sorted(self._allowed_roles)}")
        queue = self._queues.get(role)
        if queue:
            return queue.pop(0)
        if role == "translator" and self.identity_translator:
            return _identity_translation(user_prompt)
        if role in self._defaults:
            return self._defaults[role]
        raise UnexpectedRequestError(
            f"Unexpected {role} request with no queued reply. Queue one, or set "
            "identity_translator=True on the transport when the test means to accept an identity translation."
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Answers one provider call in the protocol the URL asks for."""
        payload: dict[str, Any] = json.loads(request.content or b"{}")
        system_prompt, user_prompt = _split_prompts(payload)
        role = self.classify(user_prompt)
        model = str(payload.get("model", ""))
        self.requests.append(
            RecordedRequest(
                role=role,
                model=model,
                url=str(request.url),
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
        )
        text = self.reply_for(role, user_prompt)
        return httpx.Response(200, json=_wire_body(str(request.url), text), request=request)


def _identity_translation(user_prompt: str) -> str:
    """Returns the text after the ``[TRANSLATE ...]`` header unchanged.

    A no-op translation keeps the fidelity gate satisfied by construction; a test that cares about the translated words
    queues its own reply.
    """
    _, _, remainder = user_prompt.partition("\n")
    return remainder or user_prompt


def _split_prompts(payload: dict[str, Any]) -> tuple[str, str]:
    """Extracts ``(system_prompt, user_prompt)`` from a chat, messages or responses payload."""
    if "messages" in payload:
        system = ""
        user = ""
        for message in payload["messages"]:
            if message.get("role") == "system":
                system = str(message.get("content", ""))
            elif message.get("role") == "user":
                user = str(message.get("content", ""))
        if system or user:
            return system, user
    if "system" in payload or "input" in payload:
        return str(payload.get("system", "")), str(payload.get("input", ""))
    return "", ""


def _wire_body(url: str, text: str) -> dict[str, Any]:
    """Wraps *text* in the response shape for the endpoint in *url*."""
    usage = {"prompt_tokens": 100, "completion_tokens": 40, "input_tokens": 100, "output_tokens": 40}
    if url.endswith("/messages"):
        return {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": usage,
        }
    if url.endswith("/responses"):
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "status": "completed",
            "usage": usage,
        }
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": usage,
    }


_DEFAULT_RESPONSES: dict[str, str] = {
    "generator": "## Draft\n\nA drafted answer that cites [1] one source.\n",
    "reviewer": (
        "STRENGTHS: clear structure\n"
        "WEAKNESSES: one uncited claim\n"
        "SEVERITY: nitpick\n"
        'CROSSFIRE-VERDICT {"severity": "nitpick", "weaknesses": 1}'
    ),
    "synthesizer": (
        '{"crossfire_synthesis": {"attributions": [{"index": 0, "kept": ["draft"], "discarded": []}], '
        '"notes": "merged the draft"}}\n\n## Answer\n\nA synthesized answer that cites [1] one source.\n'
    ),
    "enricher": "## Enriched\n\nExpand the request into a structured brief.\n",
    "adjudicator": (
        '{"crossfire_adjudication": {"verdicts": []}}\nCROSSFIRE-ADJUDICATION {"claims": 0, "unavailable": 0}'
    ),
}


def install_fake_transport(monkeypatch: Any) -> FakeTransport:
    """Points every async HTTP client at a fresh fake transport and sets the fake API key."""
    transport = FakeTransport()
    real_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        # A test that brings its own transport keeps it; everything else is served by the fake.
        kwargs.setdefault("transport", httpx.MockTransport(transport.handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    for variable in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY", "SYNTHETIC_API_KEY"):
        monkeypatch.setenv(variable, FAKE_API_KEY)
    return transport
