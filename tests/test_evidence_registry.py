"""Tests for the evidence registry: verbatim spans, hashes, and verdict validation."""

from __future__ import annotations

from crossfire.core.domain import CrossBranchVerdict, VerdictState
from crossfire.core.evidence import EvidenceRegistry


class TestEvidenceRegistry:
    def test_verdict_with_unknown_evidence_id_is_rejected(self) -> None:
        registry = EvidenceRegistry()
        span = registry.add("The IMTFE finding puts the toll at 200,000 or more.", origin="source")
        branch_text = "The toll was 200,000 or more [1]."
        registry.add("[1] IMTFE judgement, 1948.", origin="branch_citation", branch="en")

        assert registry.resolve(span.id) == span
        assert registry.resolve(span.id).sha256 == span.sha256

        verdict = CrossBranchVerdict(
            claim="the death toll",
            state=VerdictState.AGREES,
            by_branch={"en": branch_text, "de": "Die Opferzahl lag bei 200,000 oder mehr."},
            evidence_ids=[span.id, "ev-source-000000000000"],
        )

        violations = registry.validate_verdict(verdict, branch_texts={"en": branch_text})

        assert violations, "an evidence id that does not resolve must invalidate the verdict"
        assert "ev-source-000000000000" in violations[0]

        known_only = CrossBranchVerdict(
            claim="the death toll",
            state=VerdictState.AGREES,
            by_branch={"en": branch_text},
            evidence_ids=[span.id],
        )
        assert registry.validate_verdict(known_only, branch_texts={"en": branch_text}) == []
