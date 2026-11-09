"""Live drift tripwires for the bench's vendor tables.

``privacy.py`` and ``providers.py`` transcribe the OpenCode vendor pages by hand. The gating check here diffs
only the wire IDs the configured bench actually runs (resolved from ``crossfire.toml``), so a vendor move to a
model outside the bench stays quiet. A catalogue-wide diff still runs as a non-gating probe in
``scripts/drift_probe.py`` (cron, not CI).

When the vendor pages cannot be reached (offline runs, CI sandboxes) the live tests skip.
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


def test_bench_wire_ids_resolve_for_every_gateway():
    """Every bench slug resolves to a non-empty wire ID on every gateway, so a new model fails fast offline."""
    configuration = load_configuration(configuration_path=_TOML_PATH)
    provider_names = {item.name for item in configuration.providers}
    for provider in configuration.providers:
        for slug in _bench_slugs():
            if not _model_targets_provider(slug, provider.name, provider_names):
                continue
            assert provider.resolve_wire_model_id(slug), f"{provider.name} has no wire ID for bench slug {slug!r}"


def test_zen_bench_training_matches_the_vendor_page(vendor: _VendorPages):
    bench = _bench_wire_ids("opencode")
    vendors = _zen_training_ids(vendor)
    shipped = set(_TRAINING_MODELS["opencode"])
    assert vendors & bench == shipped & bench
    dead_bench_entries = (shipped & bench) - vendor.endpoint_ids["opencode"]
    assert dead_bench_entries == set(), (
        f"bench training entries name models off the live endpoint: {dead_bench_entries}"
    )


def test_go_bench_classification_matches_the_vendor_page(vendor: _VendorPages):
    rows = _go_policy_rows(vendor)
    bench = _bench_wire_ids("opencode-go")
    candidates = vendor.endpoint_ids["opencode-go"] | bench | set(_TRAINING_MODELS["opencode-go"])
    mapped = {name for name in rows if _wire_id_for(name, candidates) is not None}
    classified = {
        name for name, (trains, retention) in rows.items() if trains == "Yes" or retention.strip("*").startswith("30")
    }
    training = {_wire_id_for(name, candidates) for name in classified if rows[name][0] == "Yes"}
    retaining = {_wire_id_for(name, candidates) for name in classified if rows[name][0] != "Yes"}
    training.discard(None)
    retaining.discard(None)

    unmapped = {name for name in classified - mapped if _wire_id_for(name, candidates) is None}
    assert unmapped == set(), f"vendor classifies bench models with no known wire ID: {unmapped}"
    assert training & bench == set(_TRAINING_MODELS["opencode-go"]) & bench
    assert retaining & bench == set(_RETENTION_MODELS["opencode-go"]) & bench

    all_mapped = {_wire_id_for(name, candidates) for name in rows if _wire_id_for(name, candidates) is not None}
    silent_bench = bench - all_mapped
    assert silent_bench == set(), f"Go table is silent on bench wire IDs (classify or remove): {silent_bench}"


def test_bench_protocol_matches_the_vendor_endpoints(vendor: _VendorPages):
    protocols = _zen_endpoint_protocols(vendor)
    bench = _bench_wire_ids("opencode")
    opencode = next(provider for provider in DEFAULT_PROVIDER_CONFIGURATIONS if provider.name == "opencode")
    mismatches = {
        wire_id: (protocol, resolve_protocol(opencode, wire_id))
        for wire_id, protocol in protocols.items()
        if wire_id in bench and resolve_protocol(opencode, wire_id) != protocol
    }
    assert mismatches == {}, f"resolve_protocol disagrees with the vendor endpoint table for bench models: {mismatches}"
