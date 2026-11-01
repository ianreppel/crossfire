"""Live drift tripwires for the hand-maintained vendor tables.

``privacy.py`` and ``providers.py`` transcribe the OpenCode vendor pages by hand, and those pages rotate on a
schedule of days. These tests fetch both docs pages and both models endpoints and diff what they say against
the shipped tables, so a vendor policy change fails loudly instead of silently unprotecting runs. When the
pages cannot be reached (offline runs, CI sandboxes) every test here skips.
"""

from __future__ import annotations

import html
import re

import httpx
import pytest

from crossfire.core.domain import DEFAULT_PROVIDER_CONFIGURATIONS
from crossfire.core.privacy import _RETENTION_MODELS, _TRAINING_MODELS
from crossfire.core.providers import Protocol, resolve_protocol

_DOCS_URLS = {
    "zen": "https://opencode.ai/docs/zen/",
    "go": "https://opencode.ai/docs/go/",
}

_MODEL_ID_REGEX = re.compile(r"^[a-z0-9][a-z0-9.:-]*$")

_ENDPOINT_TO_PROTOCOL: list[tuple[str, Protocol]] = [
    ("openai/v1/chat/completions", "chat"),
    ("anthropic/v1/messages", "messages"),
    ("openai/v1/responses", "responses"),
]


def _strip_tags(raw: str) -> list[str]:
    """Reduces an HTML page to its non-empty text lines, one per rendered line."""
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", raw)
    text = re.sub(r"<[^>]+>", "\n", text)
    return [line.strip() for line in html.unescape(text).split("\n") if line.strip()]


def _normalise(name: str) -> str:
    """Maps a display name and its wire ID onto the same key ('MiniMax M3.1 Flash' == 'minimax-m3.1-flash')."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class _VendorPages:
    def __init__(self) -> None:
        self.docs: dict[str, list[str]] = {}
        self.endpoint_ids: dict[str, set[str]] = {}
        for name, url in _DOCS_URLS.items():
            self.docs[name] = _strip_tags(httpx.get(url, timeout=20, follow_redirects=True).text)
        for provider in DEFAULT_PROVIDER_CONFIGURATIONS:
            if provider.name not in ("opencode", "opencode-go"):
                continue
            payload = httpx.get(f"{provider.base_url}/models", timeout=20, follow_redirects=True).json()
            items = payload.get("data", payload) if isinstance(payload, dict) else payload
            self.endpoint_ids[provider.name] = {model["id"] for model in items}


@pytest.fixture(scope="module")
def vendor() -> _VendorPages:
    """Fetches both vendor docs pages and both models endpoints once, and skips offline runs."""
    try:
        return _VendorPages()
    except (httpx.HTTPError, OSError, ValueError) as exception:
        pytest.skip(f"vendor pages unreachable: {exception}")
        raise AssertionError("unreachable") from exception  # pragma: no cover


def _wire_id_for(display_name: str, candidates: set[str]) -> str | None:
    """Resolves a vendor display name to its wire ID among *candidates*, or None when it exists on neither."""
    matches = [model_id for model_id in candidates if _normalise(model_id) == _normalise(display_name)]
    return matches[0] if len(matches) == 1 else None


def _zen_training_ids(vendor: _VendorPages) -> set[str]:
    """Reads the Zen privacy section and returns the wire IDs it marks as using prompt data."""
    lines = vendor.docs["zen"]
    candidates = vendor.endpoint_ids["opencode"] | set(_TRAINING_MODELS["opencode"])
    start = next(index for index, line in enumerate(lines) if "All our models are hosted" in line)
    end = next((index for index, line in enumerate(lines[start:], start) if line.startswith("For teams")), len(lines))

    offenders: set[str] = set()
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            segment = "\n".join(buffer)
            if "used to improve the model" in segment or "improve NVIDIA products" in segment:
                offenders.add(current)

    for line in lines[start:end]:
        # Exception headers are display names followed by ':' at the start of a line, before the API notes.
        header = line.split(":", maxsplit=1)[0].replace("(NVIDIA free endpoints)", "").strip()
        matches = [model_id for model_id in candidates if _normalise(model_id) == _normalise(header)]
        if matches and "APIs" not in header:
            flush()
            current = matches[0] if len(matches) == 1 else None
            buffer = [line]
            continue
        if "APIs" in header:
            flush()
            current = None
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return offenders


def _go_policy_rows(vendor: _VendorPages) -> dict[str, tuple[str, str]]:
    """Reads Go's privacy table as display name -> (training, retention) rows."""
    lines = vendor.docs["go"]
    start = next(index for index, line in enumerate(lines) if line == "Data retention")
    rows: dict[str, tuple[str, str]] = {}
    index = start + 1
    while index + 2 < len(lines):
        name, training, retention = lines[index], lines[index + 1], lines[index + 2]
        if training not in ("Yes", "Not used"):
            break
        rows[name] = (training, retention)
        index += 3
    return rows


def _zen_endpoint_protocols(vendor: _VendorPages) -> dict[str, Protocol]:
    """Reads the Zen docs endpoint table as wire ID -> protocol."""
    lines = vendor.docs["zen"]
    protocols: dict[str, Protocol] = {}
    for index, line in enumerate(lines):
        if not line.startswith("https://opencode.ai/inference/"):
            continue
        wire_id = lines[index - 1] if index else ""
        if not _MODEL_ID_REGEX.match(wire_id):
            continue
        for suffix, protocol in _ENDPOINT_TO_PROTOCOL:
            if line.endswith(suffix):
                protocols[wire_id] = protocol
                break
    return protocols


def test_zen_training_table_matches_the_vendor_page(vendor: _VendorPages):
    vendors = _zen_training_ids(vendor)
    assert vendors == set(_TRAINING_MODELS["opencode"])


def test_go_training_and_retention_tables_match_the_vendor_page(vendor: _VendorPages):
    rows = _go_policy_rows(vendor)
    candidates = vendor.endpoint_ids["opencode-go"] | set(_TRAINING_MODELS["opencode-go"])
    classified = {
        name for name, (trains, retention) in rows.items() if trains == "Yes" or retention.strip("*").startswith("30")
    }
    training = {name for name in classified if rows[name][0] == "Yes"}
    retaining = classified - training
    unmapped = {name for name in classified if _wire_id_for(name, candidates) is None}
    assert unmapped == set(), f"vendor classifies models with no known wire ID: {unmapped}"
    assert {_wire_id_for(name, candidates) for name in training} == set(_TRAINING_MODELS["opencode-go"])
    assert {_wire_id_for(name, candidates) for name in retaining} == set(_RETENTION_MODELS["opencode-go"])


def test_no_dead_entries_in_the_policy_tables(vendor: _VendorPages):
    candidates_go = (
        vendor.endpoint_ids["opencode-go"]
        | set(_TRAINING_MODELS["opencode-go"])
        | set(_RETENTION_MODELS["opencode-go"])
    )
    live_and_documented = (
        set(_zen_training_ids(vendor))
        | vendor.endpoint_ids["opencode"]
        | vendor.endpoint_ids["opencode-go"]
        | {wire for name in _go_policy_rows(vendor) if (wire := _wire_id_for(name, candidates_go)) is not None}
    )
    stale = (
        set(_TRAINING_MODELS["opencode"]) | set(_TRAINING_MODELS["opencode-go"]) | set(_RETENTION_MODELS["opencode-go"])
    ) - live_and_documented
    assert stale == set(), f"policy tables name models on no live endpoint or vendor page: {stale}"


def test_protocol_table_matches_the_vendor_endpoints(vendor: _VendorPages):
    protocols = _zen_endpoint_protocols(vendor)
    opencode = next(provider for provider in DEFAULT_PROVIDER_CONFIGURATIONS if provider.name == "opencode")
    mismatches = {
        wire_id: (protocol, resolve_protocol(opencode, wire_id))
        for wire_id, protocol in protocols.items()
        if resolve_protocol(opencode, wire_id) != protocol
    }
    assert mismatches == {}, f"resolve_protocol disagrees with the vendor endpoint table: {mismatches}"
