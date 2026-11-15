"""Cross-branch adjudication: claim extraction, packet sizing and verdict normalisation.

The adjudicator compares branch syntheses claim by claim against the evidence registry. Everything here is
deterministic: which lines are claims, which claims are the same claim, how they are packed into calls, and what a
verdict is allowed to mean. The model only judges, and its judgement has to name evidence to count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from crossfire.core.domain import CrossBranchVerdict, EvidenceSpan
from crossfire.core.prompts import build_cross_branch_adjudicator_prompt
from crossfire.core.tokens import compute_token_budget, estimate_tokens

_IDENTIFIER_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+"),
    re.compile(r"\b10\.\d{4,9}/[^\s\"<>,;]+"),
    re.compile(r"\bISBN(?:-1[03])?:?\s*[\dXx-]+"),
)

# Native-script sentence boundaries do not require whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。\uff01\uff1f])")


@dataclass(frozen=True)
class Claim:
    """One claim, with each branch's verbatim passage for it (empty when the branch omits it)."""

    key: str
    text: str
    by_branch: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdjudicationResult:
    """Carries published verdicts and any packets whose comparison did not complete."""

    verdicts: tuple[CrossBranchVerdict, ...]
    published: str
    incomplete_packets: tuple[int, ...] = ()

    @property
    def warning(self) -> str:
        """Returns the publication warning for missing packets."""
        if not self.incomplete_packets:
            return ""
        named = ", ".join(str(index) for index in self.incomplete_packets)
        return f"Incomplete packets: {named}.\nThe run is adjudication_incomplete."


def _is_claim(line: str) -> bool:
    """Returns True when *line* is a claim rather than a heading, a rule or a table separator."""
    if not line or line.startswith(("#", "```")):
        return False
    return any(character.isalnum() for character in line)


def _claim_key(line: str) -> str:
    """Returns the identity of *line*: an identifier when it carries one, else the verbatim string."""
    for pattern in _IDENTIFIER_REGEXES:
        match = pattern.search(line)
        if match:
            return f"id:{match.group(0)}"
    return f"text:{line}"


def extract_claims(syntheses: Mapping[str, str]) -> list[Claim]:
    """Extracts passages and matches them by identifier or verbatim text."""
    order: list[str] = []
    texts: dict[str, str] = {}
    passages: dict[str, dict[str, str]] = {}

    for branch, synthesis in syntheses.items():
        for line in synthesis.split("\n"):
            stripped = line.strip()
            if not _is_claim(stripped):
                continue
            segments = [stripped] if stripped.startswith("|") else _SENTENCE_BOUNDARY.split(stripped)
            for segment in segments:
                passage = segment.strip()
                if not _is_claim(passage):
                    continue
                key = _claim_key(passage)
                if key not in passages:
                    order.append(key)
                    texts[key] = passage
                    passages[key] = {}
                passages[key].setdefault(branch, passage)

    return [Claim(key=key, text=texts[key], by_branch=passages[key]) for key in order]


class PacketSizeError(ValueError):
    """Raised when an indivisible claim and its prompt cannot fit the adjudicator's window."""


def partition_claims(
    claims: Sequence[Claim],
    *,
    context_window: int,
    max_output_tokens: int,
    syntheses: Mapping[str, str] | None = None,
    evidence: Sequence[EvidenceSpan] = (),
) -> list[list[Claim]]:
    """Packs claims by measuring both prompts and the output reservation, rejecting oversized passages."""
    if syntheses is None:
        syntheses = {branch: "" for claim in claims for branch in claim.by_branch}
    budget = compute_token_budget(context_window)
    packets: list[list[Claim]] = []
    current: list[Claim] = []

    def packet_tokens(packet: Sequence[Claim]) -> int:
        """Measures both prompts and the reserved completion."""
        system, user = build_cross_branch_adjudicator_prompt(packet, syntheses, evidence)
        return estimate_tokens(system) + estimate_tokens(user) + max_output_tokens

    for claim in claims:
        candidate = [*current, claim]
        if packet_tokens(candidate) > budget:
            if current:
                packets.append(current)
                current = []
            required = packet_tokens([claim])
            if required > budget:
                raise PacketSizeError(
                    f"Claim {claim.key[:80]!r} cannot fit the adjudication window: "
                    f"{required} tokens including output reservation, budget {budget}."
                )
        current.append(claim)

    if current:
        packets.append(current)
    return packets


def render_cross_branch(
    verdicts: Sequence[CrossBranchVerdict],
    *,
    syntheses: Mapping[str, str],
    adjudicator: str,
    incomplete: Sequence[int] = (),
) -> str:
    """Renders the adjudication as markdown: each claim, its state, and every branch's verbatim passage.

    Disagreement is published rather than collapsed, and a packet that did not complete is named, because a comparison
    that did not happen must not read as a clean one.
    """
    lines: list[str] = ["# Cross-branch adjudication", ""]
    lines.append(f"Branches: {', '.join(syntheses)}.")
    lines.append(f"Adjudicator: {adjudicator}.")
    lines.append(f"Claims compared: {len(verdicts)}.")
    if incomplete:
        named = ", ".join(str(index) for index in incomplete)
        lines.append(f"Incomplete packets: {named}.")
        lines.append("The run is adjudication_incomplete.")
    for verdict in verdicts:
        lines.extend(["", f"## {verdict.claim}", "", f"State: {verdict.state.value}."])
        for branch, passage in verdict.by_branch.items():
            lines.append(f"- {branch}: {passage or '(omitted)'}")
        if verdict.evidence_ids:
            lines.append(f"Evidence: {', '.join(verdict.evidence_ids)}.")
    return "\n".join(lines).rstrip() + "\n"
