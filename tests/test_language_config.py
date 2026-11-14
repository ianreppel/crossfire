"""Tests for the ``[language]`` configuration surface: dataclass, presets, precedence and validation."""

from __future__ import annotations

from pathlib import Path

from crossfire.core.config import load_configuration
from crossfire.core.domain import LanguageConfiguration
from crossfire.core.language import resolve_language
from crossfire.core.presets import LANGUAGE_PRESETS


def write_configuration(tmp_path: Path, body: str) -> Path:
    """Writes *body* to a ``crossfire.toml`` under *tmp_path* and returns its path."""
    path = tmp_path / "crossfire.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLanguageConfiguration:
    def test_absent_section_means_one_english_branch(self, tmp_path: Path) -> None:
        path = write_configuration(tmp_path, '[models.generators]\nnames = ["gen-a"]\n')

        configuration = load_configuration(configuration_path=path)

        assert configuration.language == LanguageConfiguration()
        assert configuration.language.branches == ("en",)
        assert configuration.language.source == "en"
        assert configuration.language.preset is None

    def test_explicit_section_is_parsed(self, tmp_path: Path) -> None:
        path = write_configuration(
            tmp_path,
            '[language]\nsource = "pt"\nbranches = ["en", "de"]\nrender_language = "pt"\n'
            '[language.modes.research]\nbranches = ["en", "de", "ja"]\n',
        )

        configuration = load_configuration(configuration_path=path)

        assert configuration.language.source == "pt"
        assert configuration.language.branches == ("en", "de")
        assert configuration.language.render_language == "pt"
        assert configuration.language_modes["research"]["branches"] == ["en", "de", "ja"]

    def test_preset_multilingual_expands_to_five(self, tmp_path: Path) -> None:
        path = write_configuration(tmp_path, '[language]\npreset = "multilingual"\n')

        configuration = load_configuration(configuration_path=path)

        assert configuration.language.preset == "multilingual"
        assert configuration.language.branches == ("en", "zh", "de", "ja", "fr")
        assert LANGUAGE_PRESETS["multilingual-wide"] == ("en", "zh", "de", "ja", "fr", "es", "hi")

    def test_explicit_branches_beat_preset(self) -> None:
        resolved, provenance = resolve_language(
            cli={"branches": ["en", "ko"]},
            mode_override=None,
            global_={"preset": "multilingual"},
        )

        assert resolved.branches == ("en", "ko")
        assert resolved.preset is None
        assert provenance["branches"] == "cli"

    def test_mode_preset_beats_global_branches(self) -> None:
        resolved, provenance = resolve_language(
            cli=None,
            mode_override={"preset": "multilingual"},
            global_={"branches": ["en"], "source": "en"},
        )

        assert resolved.branches == ("en", "zh", "de", "ja", "fr")
        assert resolved.preset == "multilingual"
        assert provenance["branches"] == "mode"
        assert provenance["source"] == "global"
