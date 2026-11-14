"""Tests for the offline source-language detector: Unicode scripts, stopword scoring, and the margin rule."""

from __future__ import annotations

import pytest

from crossfire.core.detection import detect_language


class TestScriptRanges:
    def test_script_ranges_identify_ja_zh_ko(self) -> None:
        japanese = "1937年12月の南京では、多くの市民が殺害された。"
        chinese = "1937年12月，南京有许多市民被杀害。"  # noqa: RUF001 - the full-width comma is Chinese prose
        korean = "1937년 12월 난징에서 많은 민간인이 학살되었다."

        assert detect_language(japanese).language == "ja"
        assert detect_language(chinese).language == "zh"
        assert detect_language(korean).language == "ko"

    def test_empty_text_is_not_confident(self) -> None:
        result = detect_language("")

        assert result.language == ""
        assert result.confidence == pytest.approx(0.0)


class TestStopwordScoring:
    def test_stopwords_separate_pt_from_es_and_de(self) -> None:
        portuguese = "O governo disse que não vai comentar a decisão do tribunal sobre os desaparecidos."
        spanish = "El gobierno dijo que no va a comentar la decisión del tribunal sobre los desaparecidos."
        german = "Die Regierung sagte, dass sie die Entscheidung des Gerichts nicht kommentieren wird."

        assert detect_language(portuguese).language == "pt"
        assert detect_language(spanish).language == "es"
        assert detect_language(german).language == "de"
