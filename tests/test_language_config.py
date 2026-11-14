"""Tests for the ``[language]`` configuration surface: dataclass, presets, precedence and validation."""

from __future__ import annotations

from pathlib import Path

from crossfire.core.config import load_configuration
from crossfire.core.domain import LanguageConfiguration


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
