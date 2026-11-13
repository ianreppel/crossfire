"""Data-policy tables for prompt use and retention.

A request can only ask a gateway for controls that it supports. OpenRouter accepts per-request data-collection and
zero-retention settings, so Crossfire sets them. The OpenCode gateways publish per-model policies but expose no
per-request switch, so Crossfire refuses models whose input data may be used for training or model improvement by
default. Synthetic states that it never trains on or stores API prompts, so it needs no table.

Sources:
- OpenCode Zen: https://opencode.ai/docs/zen/
- OpenCode Go: https://opencode.ai/docs/go/
- OpenRouter: https://openrouter.ai/docs/guides/routing/provider-selection
- Synthetic: https://synthetic.new/policies/privacy

Vendor catalogues change over time, and these tables are hand-copied snapshots.
``tests/test_drift.py`` compares the policy tables with live vendor pages when network access is available.
A mismatch fails the relevant test.
"""

from __future__ import annotations

from collections.abc import Iterable

# Wire model IDs whose input data the gateway may use for training or model improvement. Zen lists them as exceptions
# to its zero-retention pledge; Go's privacy table marks them "Yes" for model training.
_TRAINING_MODELS: dict[str, frozenset[str]] = {
    "opencode": frozenset(
        {
            "big-pickle",
            "flint-40b-free",
            "iceman-free",
            "inkling-2-small-free",
            "mimo-v2.6-flash-free",
            "nemotron-3-ultra-free",
            "nemotron-3.5-lightning-free",
            "nemotron-3.5-super-vl-free",
        }
    ),
    "opencode-go": frozenset(
        {
            "iceman-contributor-free",
            "muse-spark-1.2-contributor",
            "muse-spark-1.3-contributor",
        }
    ),
}

# Wire model IDs the gateway documents as retaining prompts, mapped to the retention it states. Go publishes the
# figure per model; every other Go model is zero-retention.
_RETENTION_MODELS: dict[str, dict[str, str]] = {
    "opencode-go": {
        "beam-free": "30 days",
        "claude-haiku-5-5": "30 days",
        "gpt-5.6-luna": "30 days",
        "gpt-6-luna": "30 days",
        "grok-4.7": "30 days",
    },
}

# Gateway-wide data-use and retention caveats that the vendor states without listing them in a per-model table.
_GATEWAY_DATA_POLICY_NOTES: dict[str, str] = {
    "opencode": (
        "OpenCode Zen forwards requests for OpenAI and Anthropic models to those providers, "
        "which retain those requests for 30 days. "
        "The Zen docs mention Mistral's retention policy but do not identify which models it applies to. "
        "Zen retains Beam Free prompts and other inputs for 30 days without using them for training. "
        "It retains Iceman Free data and uses it for model improvement, "
        "but the docs do not state the retention period. "
        "It retains Inkling 2 Small Free data and may use it for model training, but the docs do not state the "
        "retention period."
    ),
}


def training_models(provider_name: str) -> frozenset[str]:
    """Returns wire IDs for which the gateway may use input data for training or model improvement."""
    return _TRAINING_MODELS.get(provider_name, frozenset())


def training_violations(provider_name: str, wire_model_ids: Iterable[str]) -> list[str]:
    """Returns an error message for each given wire ID in the gateway's list of models whose input data may be used for
    training or model improvement.

    Gateways with no published per-model policy return nothing: OpenRouter filters the offending providers out of
    the request itself, and Synthetic states that it never trains on prompts.
    """
    offenders = sorted(set(wire_model_ids) & training_models(provider_name))
    return [
        f"{provider_name} may use input data sent to '{wire_model_id}' for training or model improvement. "
        f"Remove every model ID that resolves to '{wire_model_id}' from `[models.*]`, or set "
        f"`allow_training_models = true` under `[providers.{provider_name}]` after reviewing the gateway's terms."
        for wire_model_id in offenders
    ]


def retention_notices(provider_name: str, wire_model_ids: Iterable[str]) -> list[str]:
    """Returns a notice for each wire ID whose prompts the gateway documents as retaining."""
    retention = _RETENTION_MODELS.get(provider_name, {})
    return [
        f"{provider_name} retains prompts sent to '{wire_model_id}' for {retention[wire_model_id]}"
        for wire_model_id in sorted(set(wire_model_ids) & retention.keys())
    ]


def gateway_data_policy_note(provider_name: str) -> str:
    """Returns the gateway-wide data-use and retention note for *provider_name*, or an empty string when none exists."""
    return _GATEWAY_DATA_POLICY_NOTES.get(provider_name, "")
