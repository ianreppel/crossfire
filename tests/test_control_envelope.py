"""Tests for the machine-readable control envelope that ends every reviewer and adjudicator reply."""

from __future__ import annotations

import pytest

from crossfire.core.domain import Review
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.prompts import EnvelopeError, parse_control_envelope, parse_review_verdict


def review(text: str) -> Review:
    """Wraps *text* in a single review record."""
    return Review(text=text, model="rev-a", round=1, candidate_index=0)


class TestControlEnvelope:
    def test_missing_envelope_never_stops_early(self) -> None:
        prose_only = review(
            "The claim about the death toll is contradicted by the sources.\n"
            "WEAKNESSES: contradicted claim\n"
            "SEVERITY: none"
        )
        clean = review('Solid work, nothing material.\nCROSSFIRE-VERDICT {"severity": "none", "weaknesses": 0}')

        assert Orchestrator._should_stop_early([prose_only]) is False
        assert Orchestrator._should_stop_early([clean]) is True
        assert parse_control_envelope(clean.text).severity == "none"
        assert parse_control_envelope(clean.text).weaknesses == 0

    def test_translated_trailer_without_envelope_is_invalid(self) -> None:
        german = review(
            "Der Bericht enthält sachliche Fehler bei der Opferzahl.\n"
            "STRENGTHS: klare Struktur\n"
            "WEAKNESSES: falsche Opferzahl\n"
            "SEVERITY: keine"
        )
        # The legacy parser keeps the translated word, which is why the loop reads the envelope instead.
        assert parse_review_verdict(german.text).severity == "keine"
        with pytest.raises(EnvelopeError):
            parse_control_envelope(german.text)
        assert Orchestrator._should_stop_early([german]) is False

        buried = review('CROSSFIRE-VERDICT {"severity": "none", "weaknesses": 0}\nHowever: the toll is contradicted.')
        with pytest.raises(EnvelopeError):
            parse_control_envelope(buried.text)
        assert Orchestrator._should_stop_early([buried]) is False
