"""Language presets: named branch lists held as data, never as code.

A preset expands only at the configuration layer that names it, and it is mutually exclusive with an explicit branch
list at that same layer. Every preset names languages, not models: a branch multiplies the whole loop, so the choice
is about evidence value (model coverage, archival base) rather than about competence.
"""

from __future__ import annotations

# English is the question and answer language. Chinese covers the origin of most open-weight models. German, French
# and Japanese carry large academic and archival bases. Spanish and Hindi are offered as candidates, not defaults.
LANGUAGE_PRESETS: dict[str, tuple[str, ...]] = {
    "multilingual": ("en", "zh", "de", "ja", "fr"),
    "multilingual-wide": ("en", "zh", "de", "ja", "fr", "es", "hi"),
}


class UnknownPresetError(ValueError):
    """Raised when a configuration names a language preset that does not exist."""


def expand_preset(name: str) -> tuple[str, ...]:
    """Returns the ordered branch list for preset *name*.

    Raises :class:`UnknownPresetError` for an unknown preset, so a typo fails before any model call.
    """
    try:
        return LANGUAGE_PRESETS[name]
    except KeyError as exception:
        known = ", ".join(sorted(LANGUAGE_PRESETS))
        raise UnknownPresetError(f"Unknown language preset '{name}'. Configured presets: {known}") from exception
