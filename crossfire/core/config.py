"""Configuration loader from TOML file with CLI overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from crossfire.core.domain import (
    DEFAULT_PROVIDER_CONFIGURATIONS,
    CrossfireConfiguration,
    LimitsConfiguration,
    ModelGroup,
    ModelGroupOverrides,
    ProviderConfiguration,
    SearchConfiguration,
)

DEFAULT_CONFIGURATION_FILENAME = "crossfire.toml"


def _find_configuration_file(start: Path | None = None) -> Path | None:
    """Walks up from *start* (default: pwd) looking for crossfire.toml."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / DEFAULT_CONFIGURATION_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _parse_model_group(
    data: dict[str, Any],
    *,
    default_context_window: int,
    default_max_output_tokens: int,
) -> ModelGroup:
    names = data.get("names", [])
    context_window = int(data.get("context_window", default_context_window))

    raw_max_output_tokens = data.get("max_output_tokens")
    if raw_max_output_tokens is None:
        max_allowed = max(1, int(0.8 * context_window) - 1)
        max_output_tokens = min(default_max_output_tokens, max_allowed)
    else:
        max_output_tokens = int(raw_max_output_tokens)

    context_windows = data.get("context_windows", {})
    max_output_tokens_by_model = data.get("max_output_tokens_by_model", {})
    return ModelGroup(
        names=tuple(names),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        context_windows=tuple((str(model), int(value)) for model, value in context_windows.items()),
        max_output_tokens_by_model=tuple(
            (str(model), int(value)) for model, value in max_output_tokens_by_model.items()
        ),
    )


_ROLE_DEFAULTS: dict[str, tuple[int, int]] = {
    "enricher": (128000, 4096),
    "generators": (16000, 4096),
    "reviewers": (16000, 4096),
    "synthesizer": (200000, 32000),
}

# [limits.reasoning] uses the plural group names from [models.*]; roles are singular.
_ROLE_GROUP_TO_ROLE: dict[str, str] = {
    "generators": "generator",
    "reviewers": "reviewer",
    "synthesizer": "synthesizer",
    "enricher": "enricher",
}


def _parse_mode_overrides(modes_raw: dict[str, Any]) -> dict[str, ModelGroupOverrides]:
    result: dict[str, ModelGroupOverrides] = {}
    for mode_name, mode_data in modes_raw.items():
        if not isinstance(mode_data, dict):
            continue
        kwargs: dict[str, ModelGroup] = {}
        for role in ("generators", "reviewers", "synthesizer", "enricher"):
            if role in mode_data:
                context_window_default, max_output_tokens_default = _ROLE_DEFAULTS[role]
                kwargs[role] = _parse_model_group(
                    mode_data[role],
                    default_context_window=context_window_default,
                    default_max_output_tokens=max_output_tokens_default,
                )
        if kwargs:
            result[mode_name] = ModelGroupOverrides(**kwargs)
    return result


def _parse_providers(raw: dict[str, Any]) -> tuple[ProviderConfiguration, ...]:
    """Merges TOML provider overrides onto the built-in gateway definitions."""
    provider_overrides = raw.get("providers", {})
    builtin_providers: dict[str, ProviderConfiguration] = {
        provider.name: provider for provider in DEFAULT_PROVIDER_CONFIGURATIONS
    }
    providers: dict[str, ProviderConfiguration] = dict(builtin_providers)

    for provider_name, provider_data in provider_overrides.items():
        if not isinstance(provider_data, dict):
            continue
        base_provider = builtin_providers.get(provider_name)
        default_api_key_env = f"{provider_name.upper().replace('-', '_')}_API_KEY"
        model_ids = tuple(
            (str(neutral_slug), str(wire_model_id))
            for neutral_slug, wire_model_id in provider_data.get("model_ids", {}).items()
        )
        providers[provider_name] = ProviderConfiguration(
            name=provider_name,
            base_url=str(provider_data.get("base_url", base_provider.base_url if base_provider else "")),
            api_key_env=str(
                provider_data.get("api_key_env", base_provider.api_key_env if base_provider else default_api_key_env)
            ),
            requires_session=bool(
                provider_data.get("requires_session", base_provider.requires_session if base_provider else False)
            ),
            model_ids=model_ids,
            deny_data_collection=bool(
                provider_data.get("deny_data_collection", base_provider.deny_data_collection if base_provider else True)
            ),
            require_zdr=bool(provider_data.get("require_zdr", base_provider.require_zdr if base_provider else True)),
            allow_training_models=bool(
                provider_data.get(
                    "allow_training_models", base_provider.allow_training_models if base_provider else False
                )
            ),
        )

    # Backwards compatibility: an [openrouter] section with a custom api_key_env still wins.
    legacy_api_key_env = raw.get("openrouter", {}).get("api_key_env")
    if legacy_api_key_env and "openrouter" in providers:
        providers["openrouter"] = replace(providers["openrouter"], api_key_env=str(legacy_api_key_env))

    return tuple(providers.values())


def load_configuration(
    configuration_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> CrossfireConfiguration:
    """Loads configuration based on the following order of precedence: CLI > TOML > defaults."""
    raw: dict[str, Any] = {}

    path = configuration_path or _find_configuration_file()
    if path is not None and path.is_file():
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

    overrides = cli_overrides or {}

    providers = _parse_providers(raw)
    provider_name = str(
        overrides.get("provider") or raw.get("provider") or ("openrouter" if raw.get("openrouter") else "opencode")
    )
    active_provider = next((p for p in providers if p.name == provider_name), None)
    api_key_env = active_provider.api_key_env if active_provider else "OPENCODE_API_KEY"

    models_raw = raw.get("models", {})
    parsed_groups: dict[str, ModelGroup] = {}
    for role, (context_window_default, max_output_tokens_default) in _ROLE_DEFAULTS.items():
        parsed_groups[role] = _parse_model_group(
            models_raw.get(role, {}),
            default_context_window=context_window_default,
            default_max_output_tokens=max_output_tokens_default,
        )
    enricher = parsed_groups["enricher"]
    generators = parsed_groups["generators"]
    reviewers = parsed_groups["reviewers"]
    synthesizer = parsed_groups["synthesizer"]

    search_raw = raw.get("search", {})
    search = SearchConfiguration(
        enabled=search_raw.get("enabled", False),
        provider=search_raw.get("provider", "tavily"),
    )

    limits_raw = raw.get("limits", {})
    max_concurrent_requests = overrides.get(
        "max_concurrent_requests",
        limits_raw.get("max_concurrent_requests", 10),
    )
    http_timeout = overrides.get(
        "http_timeout",
        limits_raw.get("http_timeout", 120.0),
    )
    search_timeout = overrides.get(
        "search_timeout",
        limits_raw.get("search_timeout", 30.0),
    )
    reasoning_effort_by_role = tuple(
        (_ROLE_GROUP_TO_ROLE.get(str(group), str(group)), str(effort))
        for group, effort in limits_raw.get("reasoning", {}).items()
    )
    limits = LimitsConfiguration(
        max_concurrent_requests=int(max_concurrent_requests),
        temperature_generators=float(limits_raw.get("temperature_generators", 0.7)),
        temperature_reviewers=float(limits_raw.get("temperature_reviewers", 0.1)),
        temperature_synthesizer=float(limits_raw.get("temperature_synthesizer", 0.2)),
        temperature_enricher=float(limits_raw.get("temperature_enricher", 0.3)),
        http_timeout=float(http_timeout),
        search_timeout=float(search_timeout),
        reasoning_effort_by_role=reasoning_effort_by_role,
    )

    mode_overrides = _parse_mode_overrides(raw.get("modes", {}))

    return CrossfireConfiguration(
        api_key_env=api_key_env,
        provider=provider_name,
        providers=providers,
        enricher=enricher,
        generators=generators,
        reviewers=reviewers,
        synthesizer=synthesizer,
        search=search,
        limits=limits,
        mode_overrides=mode_overrides,
    )


def get_api_key(configuration: CrossfireConfiguration) -> str:
    """Resolves the API key for the active provider from the environment.

    Derived from the active provider rather than the stored ``api_key_env`` field, so the key can never drift
    from the selected gateway.
    """
    provider = configuration.active_provider()
    key = os.environ.get(provider.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"No API key. Set {provider.api_key_env} in your environment. "
            f"Crossfire can't talk to {provider.name} without it."
        )
    return key
