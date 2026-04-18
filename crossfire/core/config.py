"""Configuration loader from TOML file with CLI overrides."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from crossfire.core.domain import (
    CrossfireConfiguration,
    LimitsConfiguration,
    ModelGroup,
    ModelGroupOverrides,
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


def _parse_mode_overrides(modes_raw: dict[str, Any]) -> dict[str, ModelGroupOverrides]:
    result: dict[str, ModelGroupOverrides] = {}
    for mode_name, mode_data in modes_raw.items():
        if not isinstance(mode_data, dict):
            continue
        kwargs: dict[str, ModelGroup] = {}
        for role in ("generators", "reviewers", "synthesizer", "enricher"):
            if role in mode_data:
                cw_default, mot_default = _ROLE_DEFAULTS[role]
                kwargs[role] = _parse_model_group(
                    mode_data[role],
                    default_context_window=cw_default,
                    default_max_output_tokens=mot_default,
                )
        if kwargs:
            result[mode_name] = ModelGroupOverrides(**kwargs)
    return result


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

    openrouter = raw.get("openrouter", {})
    api_key_env = openrouter.get("api_key_env", "OPENROUTER_API_KEY")

    models_raw = raw.get("models", {})
    parsed_groups: dict[str, ModelGroup] = {}
    for role, (cw_default, mot_default) in _ROLE_DEFAULTS.items():
        parsed_groups[role] = _parse_model_group(
            models_raw.get(role, {}),
            default_context_window=cw_default,
            default_max_output_tokens=mot_default,
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
    max_concurrent = overrides.get(
        "max_concurrent_requests",
        limits_raw.get("max_concurrent_requests", 10),
    )
    temperature = overrides.get(
        "temperature_default",
        limits_raw.get("temperature_default", 0.2),
    )
    http_timeout = overrides.get(
        "http_timeout",
        limits_raw.get("http_timeout", 120.0),
    )
    search_timeout = overrides.get(
        "search_timeout",
        limits_raw.get("search_timeout", 30.0),
    )
    limits = LimitsConfiguration(
        max_concurrent_requests=int(max_concurrent),
        temperature_default=float(temperature),
        http_timeout=float(http_timeout),
        search_timeout=float(search_timeout),
    )

    mode_overrides = _parse_mode_overrides(raw.get("modes", {}))

    return CrossfireConfiguration(
        api_key_env=api_key_env,
        enricher=enricher,
        generators=generators,
        reviewers=reviewers,
        synthesizer=synthesizer,
        search=search,
        limits=limits,
        mode_overrides=mode_overrides,
    )


def get_api_key(configuration: CrossfireConfiguration) -> str:
    """Resolves the OpenRouter API key from the environment."""
    key = os.environ.get(configuration.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"No API key. Set {configuration.api_key_env} in your environment "
            "— Crossfire can't talk to OpenRouter without it."
        )
    return key
