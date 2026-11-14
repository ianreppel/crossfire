"""A numeral must survive as digits, not as one exact spelling: a German thousands separator is fidelity."""

from __future__ import annotations

import pytest

from crossfire.core.translation import ProtectedSpan, assert_fidelity, fidelity_violations

SOURCE = "Explain the Nanjing death toll of 300,000 people in 1937 with sources."
SPANS: tuple[ProtectedSpan, ...] = ()


def test_a_thousands_separator_in_the_target_style_is_accepted() -> None:
    """German writes 300.000 for 300,000; the digits are unchanged, so the translation is faithful."""
    translated = "Erklären Sie die Opferzahl von 300.000 Menschen in Nanjing 1937 anhand von Quellen."
    assert fidelity_violations(SOURCE, translated, SPANS) == []


def test_a_space_separated_number_is_accepted() -> None:
    """French and Russian group thousands with a space; the digit run is the same."""
    translated = "Expliquez le bilan de 300 000 morts de Nanjing en 1937 avec des sources."
    assert fidelity_violations(SOURCE, translated, SPANS) == []


def test_a_changed_number_is_still_rejected() -> None:
    """A translator that rounds 300,000 to 300,00 or reports a different toll must fail the check."""
    translated = "Erklären Sie die Opferzahl von 30.000 Menschen in Nanjing 1937 anhand von Quellen."
    violations = fidelity_violations(SOURCE, translated, SPANS)
    assert violations, "a numeral with fewer digits must be a violation"
    assert "300,000" in violations[0]


def test_a_dropped_number_is_still_rejected() -> None:
    """Omitting the year entirely is a fidelity violation, whatever the separator style."""
    translated = "Erklären Sie die Opferzahl von 300.000 Menschen in Nanjing anhand von Quellen."
    with pytest.raises(Exception) as raised:
        assert_fidelity(SOURCE, translated, SPANS)
    assert "1937" in str(raised.value)


def test_a_protected_span_must_survive_once() -> None:
    """A span the translator paraphrased instead of copying is a violation, as before."""
    span = ProtectedSpan(placeholder="\u27e6DT0\u27e7", text="https://example.org/2024/report")
    translated = "Erklären Sie die Opferzahl von 300.000 Menschen in Nanjing 1937 anhand des Berichts."
    violations = fidelity_violations(SOURCE + " See https://example.org/2024/report.", translated, (span,))
    assert any("DT0" in violation for violation in violations)
