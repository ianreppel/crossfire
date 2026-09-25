"""Data-policy tables: which models train on prompts, and which gateways retain them.

A request can only ask a gateway for what the gateway lets it ask for. OpenRouter takes both knobs
(``provider.data_collection`` and ``provider.zdr``) per request, so Crossfire sets them. The OpenCode gateways
publish a per-model policy and expose no per-request switch, so the only enforcement available there is refusing
to run the models that train. Synthetic states that it never trains on and never stores API prompts, so it needs
no table.

Sources: https://opencode.ai/docs/zen/ , https://opencode.ai/docs/go/ , https://openrouter.ai/docs/guides/routing/provider-selection
"""

from __future__ import annotations

from collections.abc import Iterable

# Wire model IDs the gateway documents as training on prompts. Zen lists these as exceptions to its
# zero-retention pledge; Go lists the Muse Spark Contributor rows as "Yes" for model training.
_TRAINING_MODELS: dict[str, frozenset[str]] = {
    "opencode": frozenset(
        {
            "big-pickle",
            "ling-3.0-flash-fin-free",
            "mimo-v2.5-free",
            "mimo-v2.6-flash-free",
            "muse-spark-1.3-contributor-free",
            "nemotron-3-5-lightning-free",
            "nemotron-3-ultra-free",
        }
    ),
    "opencode-go": frozenset(
        {
            "muse-spark-1.2-contributor",
            "muse-spark-1.3-contributor",
        }
    ),
}

# Wire model IDs the gateway documents as retaining prompts, mapped to the retention it states. Go publishes the
# figure per model; every other Go model is zero-retention.
_RETENTION_MODELS: dict[str, dict[str, str]] = {
    "opencode-go": {
        "gpt-5.6-luna": "30 days",
        "gpt-6-luna": "30 days",
        "grok-4.6": "30 days",
        "grok-4.7": "30 days",
    },
}

# Gateway-wide retention caveats that the vendor states without attributing them to individual models.
_GATEWAY_RETENTION_NOTES: dict[str, str] = {
    "opencode": "opencode forwards OpenAI and Anthropic API requests under 30-day retention, "
    "and documents no per-model attribution for it",
}


def training_models(provider_name: str) -> frozenset[str]:
    """Returns the wire model IDs *provider_name* documents as training on prompts."""
    return _TRAINING_MODELS.get(provider_name, frozenset())


def training_violations(provider_name: str, wire_model_ids: Iterable[str]) -> list[str]:
    """Returns one actionable error per wire model ID the gateway documents as training on prompts.

    Gateways with no published per-model policy return nothing: OpenRouter filters the offending providers out of
    the request itself, and Synthetic states that it never trains on prompts.
    """
    offenders = sorted(set(wire_model_ids) & training_models(provider_name))
    return [
        f"{provider_name} trains on prompts sent to '{wire_model_id}'. "
        f"Drop it from [models.*], or set allow_training_models = true under [providers.{provider_name}] "
        f"if you have read the gateway's terms and do not mind."
        for wire_model_id in offenders
    ]


def retention_notices(provider_name: str, wire_model_ids: Iterable[str]) -> list[str]:
    """Returns one notice per wire model ID the gateway documents as retaining prompts."""
    retention = _RETENTION_MODELS.get(provider_name, {})
    return [
        f"{provider_name} retains prompts sent to '{wire_model_id}' for {retention[wire_model_id]}"
        for wire_model_id in sorted(set(wire_model_ids) & retention.keys())
    ]


def gateway_retention_note(provider_name: str) -> str:
    """Returns the gateway-wide retention caveat for *provider_name*, or an empty string when it has none."""
    return _GATEWAY_RETENTION_NOTES.get(provider_name, "")
