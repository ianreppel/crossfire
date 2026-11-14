"""The evidence registry: immutable spans, verbatim, each with the hash that resolves it.

A translation is never evidence. Every span the adjudicator may cite is stored here once, in the script and language it
appeared in, with a sha256. A verdict that names an id this registry cannot resolve, or a span that no longer appears in
the branch synthesis it was taken from, is invalid: the comparison says so rather than promoting a rendering to a fact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from crossfire.core.domain import CrossBranchVerdict, EvidenceSpan, VerdictState
from crossfire.core.hashing import sha256_text

# Where a span came from: the instruction, the user context, or a citation a branch produced.
EVIDENCE_ORIGINS: tuple[str, ...] = ("source", "context", "branch_citation")


class EvidenceRegistry:
    """Content-addressed store of verbatim spans, keyed by a deterministic id."""

    def __init__(self) -> None:
        self._spans: dict[str, EvidenceSpan] = {}

    def add(self, text: str, *, origin: str, branch: str = "") -> EvidenceSpan:
        """Stores *text* verbatim and returns its span.

        The id is content-addressed, so the same span added twice resolves to one entry, while the same words cited by
        two branches stay two spans, because each names its own branch.
        """
        if origin not in EVIDENCE_ORIGINS:
            raise ValueError(f"Unknown evidence origin '{origin}'. Known origins: {', '.join(EVIDENCE_ORIGINS)}")
        if not text.strip():
            raise ValueError("An evidence span cannot be empty.")
        digest = sha256_text(text)
        suffix = f"-{branch}" if branch else ""
        span_id = f"ev-{origin.replace('_', '-')}{suffix}-{digest[:12]}"
        span = self._spans.get(span_id)
        if span is None:
            span = EvidenceSpan(id=span_id, text=text, sha256=digest, origin=origin, branch=branch)
            self._spans[span_id] = span
        return span

    def resolve(self, span_id: str) -> EvidenceSpan | None:
        """Returns the stored span for *span_id*, or ``None`` when it does not resolve."""
        return self._spans.get(span_id)

    def spans(self) -> tuple[EvidenceSpan, ...]:
        """Returns every stored span, in insertion order."""
        return tuple(self._spans.values())

    def validate_verdict(self, verdict: CrossBranchVerdict, branch_texts: Mapping[str, str]) -> list[str]:
        """Returns every way *verdict* fails to rest on stored evidence.

        Each cited id must resolve, and a span taken from a branch must still appear verbatim in that branch's text.
        Anything else is a verdict built on a rendering, and it is not accepted.
        """
        violations: list[str] = []
        for span_id in verdict.evidence_ids:
            span = self._spans.get(span_id)
            if span is None:
                violations.append(f"evidence id {span_id} does not resolve to a stored span")
                continue
            branch_text = branch_texts.get(span.branch, "")
            if span.branch and branch_text and span.text not in branch_text:
                violations.append(
                    f"evidence span {span_id} does not appear verbatim in the {span.branch} synthesis"
                )
        return violations


def apply_evidence_rules(
    verdict: CrossBranchVerdict,
    registry: EvidenceRegistry,
    branch_texts: Mapping[str, str],
) -> tuple[CrossBranchVerdict, list[str]]:
    """Returns *verdict* after the evidence rules, with the violations that were found.

    A verdict built on an id the registry cannot resolve becomes ``no_evidence`` and its violations are returned. An
    ``agrees`` with no evidence at all is ``no_evidence`` too: agreement without a stored span is a coincidence of two
    renderings, not a finding. Disagreement and absence need no evidence to be published, because those states say the
    branches do not line up and that is worth printing as it stands.
    """
    violations = registry.validate_verdict(verdict, branch_texts)
    if violations:
        return replace(verdict, state=VerdictState.NO_EVIDENCE), violations
    if verdict.state is VerdictState.AGREES and not verdict.evidence_ids:
        return replace(verdict, state=VerdictState.NO_EVIDENCE), []
    return verdict, []
