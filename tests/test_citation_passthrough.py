"""Tests for citation and name pass-through, and for the translation fidelity gate."""

from __future__ import annotations

import pytest

from crossfire.core.translation import (
    FidelityError,
    assert_fidelity,
    extract_do_not_translate,
    fidelity_violations,
    restore_do_not_translate,
)

JAPANESE_REFERENCE = "[3] 洞 富雄, 「南京アッサム日記」, 文芸春秋, 1998."


class TestCitationPassthrough:
    def test_japanese_title_survives_byte_identical(self) -> None:
        source = f"The death toll is disputed in the sources.\n{JAPANESE_REFERENCE}\n"

        masked, spans = extract_do_not_translate(source)

        assert "南京アッサム日記" not in masked, "the reference line must not reach the translator"
        assert len(spans) == 1
        translated = f"Die Opferzahl ist in den Quellen umstritten.\n{spans[0].placeholder}\n"
        restored = restore_do_not_translate(translated, spans)

        assert JAPANESE_REFERENCE in restored
        assert fidelity_violations(source, translated, spans) == []
        assert_fidelity(source, translated, spans)

    def test_translation_that_drops_a_span_fails_the_gate(self) -> None:
        source = f"The death toll is disputed in the sources.\n{JAPANESE_REFERENCE}"
        _, spans = extract_do_not_translate(source)

        violations = fidelity_violations(source, "Die Opferzahl ist umstritten.", spans)

        assert violations, "a translation that drops the reference must fail the gate"
        with pytest.raises(FidelityError):
            assert_fidelity(source, "Die Opferzahl ist umstritten.", spans)
