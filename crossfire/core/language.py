"""Language resolution: which branches run, which language is the source, and which layer said so.

Language is configuration. A run may be told its languages by an explicit CLI flag, by a ``[language.modes.*]``
override, or by the global ``[language]`` section, and the resolution records which layer supplied each value so the
manifest can show it. No module, prompt or fixture hard-codes a language tag.
"""

from __future__ import annotations

import re
from dataclasses import asdict, fields, replace
from typing import Any

from crossfire.core.detection import resolve_source_language
from crossfire.core.domain import CrossfireConfiguration, LanguageConfiguration, RunParameters
from crossfire.core.presets import LANGUAGE_PRESETS, expand_preset

# Translation backends this build can drive. The LLM backend is the only one: a named vendor backend was dropped by
# owner order because it costs money and the LLM path already covers translation.
TRANSLATOR_BACKENDS: tuple[str, ...] = ("llm",)

# A BCP-47 language tag: a two- or three-letter primary subtag, then optional alphanumeric subtags.
_LANGUAGE_TAG_REGEX: re.Pattern[str] = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$")

# Field names of LanguageConfiguration, in declaration order.
_LANGUAGE_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(LanguageConfiguration))

Layer = dict[str, Any]


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
    # Lowest precedence first, so the highest layer overwrites the rest. An explicit None means "not set here".
    for layer_name, layer in reversed(layers):
        for key, value in (layer or {}).items():
            if value is None:
                continue
            merged[key] = value
            supplied[key] = layer_name

    preset_value = merged.get("preset")
    preset: str | None = str(preset_value) if preset_value is not None else None
    branches = defaults.branches
    branch_layer = "default"
    # The branch decision belongs to one layer, not to the merged view: a preset expands at the layer that named it,
    # so a global branch list cannot silently defeat a mode-level preset.
    for layer_name, layer in layers:
        data = layer or {}
        if data.get("branches") and data.get("preset") is not None:
            raise ValueError(
                f"language.{layer_name} sets both branches and preset, which are mutually exclusive. Keep one."
            )
        if "branches" in data:
            branches = tuple(str(branch) for branch in data["branches"])
            preset = None
            branch_layer = layer_name
            break
        if data.get("preset") is not None:
            preset = str(data["preset"])
            branches = expand_preset(preset)
            branch_layer = layer_name
            break

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


def validate_language(configuration: LanguageConfiguration) -> list[str]:
    """Returns the language configuration errors that must stop a run before any model call.

    An empty or duplicated branch list is an operator error: a branch multiplies the whole loop, so the list is
    validated once, up front, rather than discovered mid-run.
    """
    errors: list[str] = []
    if configuration.preset is not None and configuration.preset not in LANGUAGE_PRESETS:
        errors.append(
            f"Unknown language preset '{configuration.preset}'. Available presets: {', '.join(LANGUAGE_PRESETS)}"
        )
    for field in ("source", "render_language"):
        tag = getattr(configuration, field)
        if tag is not None and not _LANGUAGE_TAG_REGEX.fullmatch(tag):
            errors.append(f"language.{field} has a malformed BCP-47 tag: {tag!r}")
    branches = configuration.branches
    if not branches:
        errors.append("language.branches must name at least one language")
    seen: set[str] = set()
    duplicates: list[str] = []
    for branch in branches:
        if branch in seen and branch not in duplicates:
            duplicates.append(branch)
        seen.add(branch)
    if duplicates:
        errors.append(f"language.branches has duplicate languages: {', '.join(duplicates)}")

    malformed: list[str] = [branch for branch in branches if not _LANGUAGE_TAG_REGEX.fullmatch(branch)]
    if malformed:
        errors.append(f"language.branches has malformed BCP-47 tags: {', '.join(malformed)}")

    # The source language is deliberately independent of the branch list: a Portuguese question may run one English
    # branch and render back to Portuguese. The competence branch is not, because it must name a branch that exists.
    if configuration.translator_backend not in TRANSLATOR_BACKENDS:
        errors.append(
            f"language.translator_backend '{configuration.translator_backend}' is not implemented. "
            f"Available backends: {', '.join(TRANSLATOR_BACKENDS)}"
        )

    competence_branch = configuration.competence_branch
    if competence_branch is not None and competence_branch not in branches:
        errors.append(
            f"language.competence_branch '{competence_branch}' is not one of language.branches "
            f"({', '.join(branches) or 'none'})"
        )
    return errors


def resolve_run_language(
    configuration: CrossfireConfiguration, parameters: RunParameters
) -> tuple[LanguageConfiguration, dict[str, str]]:
    """Resolves all language layers and detects the source before validation, pricing or execution."""
    global_layer = configuration.language_layer
    if not global_layer:
        global_layer = asdict(configuration.language)
        if configuration.language.preset is not None:
            global_layer.pop("branches")
    language, provenance = resolve_language(
        cli=parameters.language_cli,
        mode_override=configuration.language_modes.get(parameters.mode.value),
        global_=global_layer,
    )
    language = replace(language, source=resolve_source_language(language.source, parameters.task.instruction))
    return language, provenance


def branch_needs_translation(language: LanguageConfiguration, tag: str) -> bool:
    """Returns whether a branch needs its original instruction translated."""
    return language.translate_instruction and tag != language.source


def validate_run_language(configuration: CrossfireConfiguration, language: LanguageConfiguration) -> list[str]:
    """Validates the language surface and the model groups required by the resolved operations."""
    errors = validate_language(language)
    needs_translator = any(branch_needs_translation(language, tag) for tag in language.branches)
    if language.render_language:
        render_source = language.branches[0] if len(language.branches) == 1 else language.source
        needs_translator |= language.render_language != render_source or any(
            tag != language.render_language for tag in language.branches
        )
        if len(language.branches) > 1 and not language.adjudicate:
            errors.append("language.render_language requires adjudication for a multi-branch run")
    if needs_translator and not configuration.translator.names:
        errors.append("No translator models configured: set [models.translator] names")
    if len(language.branches) >= 2 and language.adjudicate and not configuration.adjudicator.names:
        errors.append("No adjudicator models configured: set [models.adjudicator] names, or set adjudicate = false")
    return errors
