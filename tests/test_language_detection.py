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
