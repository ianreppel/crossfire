"""Language resolution: which branches run, which language is the source, and which layer said so.

Language is configuration. A run may be told its languages by an explicit CLI flag, by a ``[language.modes.*]``
override, or by the global ``[language]`` section, and the resolution records which layer supplied each value so the
manifest can show it. No module, prompt or fixture hard-codes a language tag.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from crossfire.core.domain import LanguageConfiguration
from crossfire.core.presets import UnknownPresetError, expand_preset

# Field names of LanguageConfiguration, in declaration order.
_LANGUAGE_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(LanguageConfiguration))

Layer = dict[str, Any]


def _expanded(preset: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Expands *preset*, falling back to *fallback* when the preset is unknown.

    An unknown preset is a configuration error and language validation reports it before any model call, so
    resolution stays silent and total rather than raising halfway through a run.
    """
    try:
        return expand_preset(preset)
    except UnknownPresetError:
        return fallback


def resolve_language(
    cli: Layer | None = None,
    mode_override: Layer | None = None,
    global_: Layer | None = None,
) -> tuple[LanguageConfiguration, dict[str, str]]:
    """Resolves a language configuration from three layers, highest precedence first.

    Precedence is explicit CLI, then the mode override, then the global section, then the dataclass defaults.
    An explicit branch list beats a preset, because a list names exactly what an operator asked for. The second
    return value maps each field to the layer that supplied it, with ``"default"`` for fields nobody set.
    """
    defaults = LanguageConfiguration()
    layers: tuple[tuple[str, Layer | None], ...] = (
        ("cli", cli),
        ("mode", mode_override),
        ("global", global_),
    )

    merged: dict[str, Any] = {}
    supplied: dict[str, str] = {}
    # Lowest precedence first, so the highest layer overwrites the rest.
    for layer_name, layer in reversed(layers):
        for key, value in (layer or {}).items():
            merged[key] = value
            supplied[key] = layer_name

    preset_value = merged.get("preset")
    preset: str | None = str(preset_value) if preset_value is not None else None
    if "branches" in merged:
        branches = tuple(str(branch) for branch in merged["branches"])
        preset = None
        branch_layer = supplied["branches"]
    elif preset is not None:
        branches = _expanded(preset, defaults.branches)
        branch_layer = supplied["preset"]
    else:
        branches = defaults.branches
        branch_layer = "default"

    competence_branch = merged.get("competence_branch")
    render_language = merged.get("render_language")
    configuration = LanguageConfiguration(
        source=str(merged.get("source", defaults.source)),
        branches=branches,
        preset=preset,
        competence_branch=str(competence_branch) if competence_branch is not None else defaults.competence_branch,
        render_language=str(render_language) if render_language is not None else defaults.render_language,
        translate_instruction=bool(merged.get("translate_instruction", defaults.translate_instruction)),
        adjudicate=bool(merged.get("adjudicate", defaults.adjudicate)),
        translator_backend=str(merged.get("translator_backend", defaults.translator_backend)),
    )

    provenance: dict[str, str] = {field: supplied.get(field, "default") for field in _LANGUAGE_FIELDS}
    provenance["branches"] = branch_layer
    return configuration, provenance
