"""Checks Crossfire's OpenCode policy tables and protocol mapping against vendor sources.

``privacy.py`` and ``providers.py`` transcribe OpenCode's vendor pages. Network tests compare Crossfire's complete list
of models whose input data OpenCode Zen may use for training or model improvement, OpenCode Go's training and retention
tables, and Zen's documented protocol mapping. An offline test checks that configured model IDs resolve to wire IDs for
every gateway.

Network tests skip as a group if any required documentation page or model-list endpoint for OpenCode Zen or OpenCode Go
is unreachable.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import httpx
import pytest

from crossfire.core.config import load_configuration
from crossfire.core.domain import DEFAULT_PROVIDER_CONFIGURATIONS, _model_targets_provider
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

_TOML_PATH = Path(__file__).resolve().parents[1] / "crossfire.toml"


def _bench_slugs() -> set[str]:
    """Collects every neutral slug the configured bench can run, across groups and mode overrides."""
    configuration = load_configuration(configuration_path=_TOML_PATH)
    slugs: set[str] = set()
    for group in (configuration.enricher, configuration.generators, configuration.reviewers, configuration.synthesizer):
        slugs.update(group.names)
    for overrides in configuration.mode_overrides.values():
        for mode_group in (overrides.generators, overrides.reviewers, overrides.synthesizer, overrides.enricher):
            if mode_group is not None:
                slugs.update(mode_group.names)
    return slugs


def _bench_wire_ids(provider_name: str) -> set[str]:
    """Resolves the bench slugs to *provider_name*'s wire IDs, keeping only models that target that gateway."""
    configuration = load_configuration(configuration_path=_TOML_PATH)
    provider = next(item for item in configuration.providers if item.name == provider_name)
    provider_names = {item.name for item in configuration.providers}
    return {
        provider.resolve_wire_model_id(slug)
        for slug in _bench_slugs()
        if _model_targets_provider(slug, provider_name, provider_names)
    }


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
            segment = "\n".join(buffer).lower()
            if any(
                phrase in segment
                for phrase in ("used to improve the model", "improve nvidia products", "for model training")
            ):
                offenders.add(current)

    for line in lines[start:end]:
        # The privacy page puts model exception names before a colon, followed by their API notes.
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


def test_bench_wire_ids_resolve_for_every_gateway():
    """Every bench slug resolves to a non-empty wire ID on every gateway, so a new model fails fast offline."""
    configuration = load_configuration(configuration_path=_TOML_PATH)
    provider_names = {item.name for item in configuration.providers}
    for provider in configuration.providers:
        for slug in _bench_slugs():
            if not _model_targets_provider(slug, provider.name, provider_names):
                continue
            assert provider.resolve_wire_model_id(slug), f"{provider.name} has no wire ID for bench slug {slug!r}"


def test_zen_training_parser_detects_explicit_training_language() -> None:
    vendor: _VendorPages = object.__new__(_VendorPages)
    vendor.docs = {
        "zen": [
            "All our models are hosted in the US and EU.",
            "Inkling 2 Small Free: During its free period, collected data may be used for model training.",
            "For teams",
        ]
    }
    vendor.endpoint_ids = {"opencode": {"inkling-2-small-free"}}
    assert _zen_training_ids(vendor) == {"inkling-2-small-free"}


def test_zen_training_matches_the_vendor_catalogue(vendor: _VendorPages) -> None:
    vendors = _zen_training_ids(vendor)
    shipped = set(_TRAINING_MODELS["opencode"])
    assert vendors == shipped
    bench = _bench_wire_ids("opencode")
    dead_bench_entries = (shipped & bench) - vendor.endpoint_ids["opencode"]
    assert dead_bench_entries == set(), f"bench training entries are off the live endpoint: {dead_bench_entries}"


def test_go_classification_matches_the_vendor_catalogue(vendor: _VendorPages) -> None:
    rows = _go_policy_rows(vendor)
    bench = _bench_wire_ids("opencode-go")
    candidates = (
        vendor.endpoint_ids["opencode-go"]
        | set(_TRAINING_MODELS["opencode-go"])
        | set(_RETENTION_MODELS["opencode-go"])
    )
    mapped = {name for name in rows if _wire_id_for(name, candidates) is not None}
    classified = {
        name
        for name, (trains, retention) in rows.items()
        if trains == "Yes" or retention.strip("* ").lower() != "0 days"
    }
    training = {_wire_id_for(name, candidates) for name in classified if rows[name][0] == "Yes"}
    retaining = {_wire_id_for(name, candidates) for name in classified if rows[name][0] != "Yes"}
    training.discard(None)
    retaining.discard(None)

    unmapped = {name for name in classified - mapped if _wire_id_for(name, candidates) is None}
    assert unmapped == set(), f"vendor classifies models with no known wire ID: {unmapped}"
    assert training == set(_TRAINING_MODELS["opencode-go"])
    assert retaining == set(_RETENTION_MODELS["opencode-go"])

    all_mapped = {_wire_id_for(name, candidates) for name in rows if _wire_id_for(name, candidates) is not None}
    silent_bench = bench - all_mapped
    assert silent_bench == set(), f"Go table is silent on bench wire IDs (classify or remove): {silent_bench}"


def test_protocols_match_the_vendor_catalogue(vendor: _VendorPages) -> None:
    protocols = _zen_endpoint_protocols(vendor)
    opencode = next(provider for provider in DEFAULT_PROVIDER_CONFIGURATIONS if provider.name == "opencode")
    mismatches = {
        wire_id: (protocol, resolve_protocol(opencode, wire_id))
        for wire_id, protocol in protocols.items()
        if resolve_protocol(opencode, wire_id) != protocol
    }
    assert mismatches == {}, f"resolve_protocol disagrees with the vendor endpoint catalogue: {mismatches}"
