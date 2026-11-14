"""The numeral gate must reject a moved, merged, dropped or restated number, and accept a locale separator."""

from __future__ import annotations

import pytest

from crossfire.core.translation import FidelityError, ProtectedSpan, assert_fidelity, fidelity_violations

SPANS: tuple[ProtectedSpan, ...] = ()


def violations(source: str, translated: str) -> list[str]:
    """Returns the fidelity violations for one pair of strings with no protected spans."""
    return fidelity_violations(source, translated, SPANS)


def test_merged_numbers_are_rejected() -> None:
    """Two source numerals that arrive as one number are not the same two numbers."""
    merged = violations("The report lists 12 and 34 cases.", "Der Bericht nennt 1234 Fälle.")
    assert merged, "12 plus 34 must not pass as 1234"


def test_a_reworded_sentence_with_the_same_values_is_accepted() -> None:
    """German may name the toll before the year; the values are unchanged, so the branch must survive."""
    source = "In 1937 the toll was 300,000."
    assert violations(source, "Die Opferzahl betrug 300.000 im Jahr 1937.") == []


def test_duplicate_occurrences_must_both_survive() -> None:
    """A numeral repeated in the source must appear as often in the translation."""
    source = "The 1937 report and the 1937 review agree."
    translated = "Der Bericht von 1937 und die Überprüfung stimmen überein."
    assert violations(source, translated), "the second 1937 is missing"


def test_a_thousand_reported_as_ten_thousand_is_rejected() -> None:
    """A different value in the same place is the defect this gate exists for."""
    assert violations("The toll was 300,000.", "Die Opferzahl lag bei 30.000.")


def test_equal_values_in_different_separator_styles_are_accepted() -> None:
    """German, French and plain spellings of one number are the same number."""
    source = "The toll was 300,000 in 1937."
    for translated in (
        "Die Opferzahl betrug 300.000 im Jahr 1937.",
        "Le bilan était de 300 000 en 1937.",
        "The toll was 300000 in 1937.",
    ):
        assert violations(source, translated) == [], translated


def test_a_fraction_is_not_read_as_a_whole_number() -> None:
    """1.5 and 15 are different values and must not compare equal."""
    assert violations("The rate was 1.5 percent.", "Die Rate lag bei 15 Prozent."), "15 must not pass for 1.5"


def test_a_protected_span_must_survive_once() -> None:
    """A span the translator paraphrased instead of copying is a violation."""
    span = ProtectedSpan(placeholder="\u27e6DT0\u27e7", text="https://example.org/2024/report")
    source = "Explain the 300,000 toll. See https://example.org/2024/report."
    translated = "Erklären Sie die Zahl von 300.000 anhand des Berichts."
    assert any("DT0" in violation for violation in fidelity_violations(source, translated, (span,)))


def test_a_clean_translation_passes_the_assertion() -> None:
    """The accepting path raises nothing."""
    assert_fidelity("The toll was 300,000 in 1937.", "Die Opferzahl betrug 300.000 im Jahr 1937.", SPANS)


def test_the_assertion_raises_the_named_error() -> None:
    """The strict entry point raises a FidelityError, not a bare exception."""
    with pytest.raises(FidelityError):
        assert_fidelity("The toll was 300,000.", "Die Opferzahl lag bei 30.000.", SPANS)
