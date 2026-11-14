"""Tests for the ``[language]`` configuration surface: dataclass, presets, precedence and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.config import load_configuration
from crossfire.core.domain import LanguageConfiguration
from crossfire.core.language import resolve_language, validate_language
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


class TestLanguageValidation:
    def test_duplicate_branch_is_a_config_error(self) -> None:
        errors = validate_language(LanguageConfiguration(branches=("en", "de", "en")))

        assert any("duplicate" in error.lower() for error in errors), errors
        assert validate_language(LanguageConfiguration(branches=("en", "de"))) == []

    def test_malformed_tag_is_a_config_error(self) -> None:
        errors = validate_language(LanguageConfiguration(branches=("en", "not a language")))

        assert any("bcp-47" in error.lower() or "malformed" in error.lower() for error in errors), errors
        assert validate_language(LanguageConfiguration(branches=("en", "zh-Hant", "pt-BR"))) == []

    def test_competence_branch_must_be_in_branches(self) -> None:
        errors = validate_language(LanguageConfiguration(branches=("en",), competence_branch="de"))

        assert any("competence_branch" in error for error in errors), errors
        assert validate_language(LanguageConfiguration(branches=("en", "de"), competence_branch="de")) == []
        assert validate_language(LanguageConfiguration(branches=("en",), source="pt")) == []

    def test_preset_and_branches_are_mutually_exclusive(self, tmp_path: Path) -> None:
        path = write_configuration(tmp_path, '[language]\npreset = "multilingual"\nbranches = ["en"]\n')

        with pytest.raises(ValueError, match="mutually exclusive"):
            load_configuration(configuration_path=path)

    def test_an_unimplemented_translator_backend_is_a_config_error(self) -> None:
        errors = validate_language(LanguageConfiguration(translator_backend="vendor-x"))

        assert any("translator_backend" in error for error in errors), errors
        assert validate_language(LanguageConfiguration(translator_backend="llm")) == []
