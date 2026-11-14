"""Cross-branch adjudication: claim extraction, packet sizing and verdict normalisation.

The adjudicator compares branch syntheses claim by claim against the evidence registry. Everything here is
deterministic: which lines are claims, which claims are the same claim, how they are packed into calls, and what a
verdict is allowed to mean. The model only judges, and its judgement has to name evidence to count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from crossfire.core.domain import CrossBranchVerdict
from crossfire.core.tokens import compute_token_budget, estimate_tokens

_IDENTIFIER_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+"),
    re.compile(r"\b10\.\d{4,9}/[^\s\"<>,;]+"),
    re.compile(r"\bISBN(?:-1[03])?:?\s*[\dXx-]+"),
)

# A claim needs enough words to be a statement rather than a label or a fragment.
_MINIMUM_CLAIM_WORDS = 4


@dataclass(frozen=True)
class Claim:
    """One claim, with each branch's verbatim passage for it (empty when the branch omits it)."""

    key: str
    text: str
    by_branch: dict[str, str] = field(default_factory=dict)


def _is_claim(line: str) -> bool:
    """Returns True when *line* is a claim rather than a heading, a rule or a table separator."""
    if not line or line.startswith(("#", "|", "```", "---", "***")):
        return False
    return len(line.split()) >= _MINIMUM_CLAIM_WORDS


def _claim_key(line: str) -> str:
    """Returns the identity of *line*: an identifier when it carries one, else the verbatim string."""
    for pattern in _IDENTIFIER_REGEXES:
        match = pattern.search(line)
        if match:
            return f"id:{match.group(0)}"
    return f"text:{line}"


def extract_claims(syntheses: Mapping[str, str]) -> list[Claim]:
    """Extracts claims from each branch synthesis and matches them across branches.

    A line is a claim when it carries at least a few words. Two branches' lines are one claim when they carry the same
    identifier (a DOI, an ISBN or a URL) or when they are verbatim identical: reconciliation never runs through a
    translated title, because two renderings of one work are not the same string.
    """
    order: list[str] = []
    texts: dict[str, str] = {}
    passages: dict[str, dict[str, str]] = {}

    for branch, synthesis in syntheses.items():
        for line in synthesis.split("\n"):
            stripped = line.strip()
            if not _is_claim(stripped):
                continue
            key = _claim_key(stripped)
            if key not in passages:
                order.append(key)
                texts[key] = stripped
                passages[key] = {}
            passages[key].setdefault(branch, stripped)

    return [Claim(key=key, text=texts[key], by_branch=passages[key]) for key in order]


def partition_claims(
    claims: Sequence[Claim],
    *,
    context_window: int,
    max_output_tokens: int,
    overhead_tokens: int = 0,
) -> list[list[Claim]]:
    """Packs *claims* into packets whose estimated size fits the adjudicator's context window.

    The split is deterministic: claims keep their order and a packet is closed only when the next claim would not fit.
    A single claim larger than the whole budget still gets a packet of its own, because dropping it would silently
    shrink the comparison.
    """
    budget = compute_token_budget(context_window) - max_output_tokens
    packets: list[list[Claim]] = []
    current: list[Claim] = []
    used = overhead_tokens

    for claim in claims:
        cost = estimate_tokens(claim.text)
        if current and used + cost > budget:
            packets.append(current)
            current = []
            used = overhead_tokens
        current.append(claim)
        used += cost

    if current or not packets:
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
