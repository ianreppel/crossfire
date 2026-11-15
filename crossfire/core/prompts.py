"""Prompt construction for all modes and roles."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crossfire.core.domain import (
    Candidate,
    CandidateDecision,
    CrossBranchVerdict,
    Mode,
    Review,
    VerdictState,
)

MODE_RULES: dict[Mode, str] = {
    Mode.RESEARCH: (
        "RULES:\n"
        "- All factual claims MUST have inline [N] citation markers\n"
        "- End with a numbered References section in this exact format: "
        "[N] Authors, Title, Journal/Venue, Year. "
        "For websites/reports: Author/Organization, Title, URL, Year. "
        "NEVER fabricate DOIs. Omit DOIs entirely unless you are certain they are correct\n"
        "- Every [N] marker in the body MUST correspond to the correct [N] entry in the References section. "
        "After drafting, re-read each citation in context and verify the referenced paper actually supports "
        "the specific claim made. If unsure, flag with a bracketed note\n"
        "- Prioritize cutting-edge research (last 3-5 years) alongside foundational work\n"
        "- Flag recent results that have not yet been independently reproduced\n"
        "- Note any cited work that has been retracted, corrected, or substantially challenged\n"
        "- Quantify claims where possible (thresholds, error rates, resource counts), not just directional statements\n"
        "- Structured output with sections and bullet points\n"
        "- No unsupported claims in final output\n"
        "- Steelman counterarguments before dismissing them\n"
        "- Synthesize across sources, don't just summarize each"
    ),
    Mode.CODE: (
        "RULES:\n"
        "- Production-quality code with associated tests\n"
        "- Separate code and tests with clear section headings\n"
        "- Validate inputs, avoid injection risks, no hardcoded secrets\n"
        "- Explicit error handling, resource cleanup, edge cases\n"
        "- Tests must cover normal flow, error paths, and edge cases"
    ),
    Mode.EDIT: (
        "RULES:\n"
        "- Maximize clarity and concision while preserving meaning\n"
        "- Improve structure, remove redundancy, sharpen language"
    ),
    Mode.CHECK: (
        "RULES:\n"
        "- Identify every core claim and its reasoning chain\n"
        "- Classify each claim as true, false, or uncertain\n"
        "- Flag logical fallacies, unstated assumptions, evidence gaps"
    ),
    Mode.WRITE: (
        "RULES:\n"
        "- Originality and strong voice above all\n"
        "- Commit to a distinctive tone and structure\n"
        "- Vary section lengths; never mirror or balance sections symmetrically\n"
        "- Select and discard, never average"
    ),
}

_ENRICHER_SYSTEM = (
    "Expand the user instruction into a detailed, structured brief "
    "that a team of AI generators can follow.\n\n"
    "- Preserve the original intent exactly\n"
    "- Add explicit constraints: output format, sections, length guidance\n"
    "- Incorporate the mode-specific rules\n"
    "- Clarify ambiguities with reasonable defaults\n"
    "- Output ONLY the enriched instruction"
)

_GENERATOR_SYSTEM: dict[Mode, str] = {
    Mode.RESEARCH: (
        "You are a meticulous researcher who builds arguments from evidence. "
        "Anchor your analysis in foundational work but prioritize the latest published results. "
        "When citing recent findings, note whether they have been independently replicated."
    ),
    Mode.CODE: "You are a careful engineer who thinks several steps ahead.",
    Mode.EDIT: "You are a sharp editor with zero tolerance for waffle.",
    Mode.CHECK: "You are a methodical analyst who questions everything.",
    Mode.WRITE: "You are a lyrical writer with a distinctive voice.",
}

_REVIEWER_STRUCTURED_SUFFIX = (
    "\n\nEnd with a structured summary on separate lines:\n"
    "STRENGTHS: <comma-separated list>\n"
    "WEAKNESSES: <comma-separated list>\n"
    "SEVERITY: material (factual errors, logic flaws, broken code) | nitpick (style only) | none\n"
    'CROSSFIRE-VERDICT {"severity": "material|nitpick|none", "weaknesses": <integer>}\n'
    "The last line is that JSON line exactly: ASCII keys, ASCII enum values, and a whole number of weaknesses. "
    "The prose above it may be in any language; the envelope is never translated."
)

_REVIEWER_SYSTEM: dict[Mode, str] = {
    Mode.RESEARCH: (
        "You are a sceptical peer reviewer. Verify every claim with rigour:\n\n"
        "**CITATIONS:** uncited claims, citations that don't support the claim, "
        "citation markers that don't match the corresponding reference entry "
        "(e.g. [8] in the body describes result X but reference [8] is a different paper), "
        "single-source over-reliance, outdated or non-authoritative sources\n\n"
        "**RECENCY:** flag claims supported only by pre-2020 work when newer results exist; "
        "flag recent results that lack independent replication; "
        "note any retracted, corrected, or substantially challenged sources\n\n"
        "**QUANTIFICATION:** reject vague hedges ('generally lower', 'roughly comparable') "
        "when specific numbers, ranges, or order-of-magnitude estimates are available\n\n"
        "**REASONING:** cherry-picked evidence, false equivalences, "
        "correlation-as-causation, unstated assumptions, logical jumps\n\n"
        "**COMPLETENESS:** omitted counterarguments (steelman the strongest objections), "
        "unacknowledged limitations, overreaching scope, missing context\n\n"
        "**CLARITY:** claims buried in prose, ambiguous language, "
        "repetitive sections, missing cross-source synthesis\n\n"
        "Cite the specific claim. Classify as supported / uncertain / unsupported." + _REVIEWER_STRUCTURED_SUFFIX
    ),
    Mode.CODE: (
        "You are a paranoid code reviewer. Rubber-duck debug each section:\n\n"
        "**LOGIC:** off-by-one, null derefs, wrong conditionals, "
        "unhandled edge cases, infinite loops\n\n"
        "**TYPES:** mismatches, unsafe casts, missing hints, "
        "implicit conversions\n\n"
        "**SECURITY:** SQL/XSS/command injection, path traversal, "
        "hardcoded secrets, unsafe eval/deserialization\n\n"
        "**ERRORS:** unhandled exceptions, resource leaks, "
        "race conditions, silent failures\n\n"
        "**TESTS:** missing edge-case/error-path tests, "
        "assertions that don't verify behaviour, missing integration tests\n\n"
        "**STRUCTURE:** single-responsibility violations, tight coupling, "
        "magic values, O(n²) where O(n) suffices\n\n"
        "Cite exact line/section. Provide a specific fix. "
        "Focus on bugs, security, and maintenance risks." + _REVIEWER_STRUCTURED_SUFFIX
    ),
    Mode.EDIT: (
        "You are a ruthless editor. Eliminate every unnecessary word:\n\n"
        "**PRECISION:** vague qualifiers (somewhat, rather, quite), "
        "hedging (it seems that, one might argue), weak verbs (is/was + adj), "
        "unclear antecedents, imprecise nouns (thing, aspect, factor)\n\n"
        "**CONCISENESS:** redundancy (completely eliminate, future plans), "
        "wordy constructions (in order to → to), unnecessary intensifiers, "
        "bloated transitions, filler (basically, literally, actually)\n\n"
        "**JARGON & LLM WAFFLE:** corporate speak (leverage, utilize, facilitate), "
        "LLM hedging (it's worth noting, importantly), buzzword clusters, "
        "throat-clearing, academic bloat\n\n"
        "**STRUCTURE:** buried main points, missing topic sentences, "
        "repetitive patterns, fake transitions, logic gaps in prose\n\n"
        "For each problem: exact text + tighter replacement. Cut mercilessly." + _REVIEWER_STRUCTURED_SUFFIX
    ),
    Mode.CHECK: (
        "You are a hostile fact-checker. Every claim is wrong until proven right:\n\n"
        "**FACTS:** contradicted claims, implausible statistics, "
        "wrong dates/names/attributions, technical misrepresentations, outdated claims\n\n"
        "**LOGIC:** non sequiturs, circular reasoning, false dichotomies, "
        "straw men, appeals to authority\n\n"
        "**ASSUMPTIONS:** unstated premises, undefined terms, "
        "counterfactuals that break the argument, selection bias\n\n"
        "**REASONING CHAIN:** challengeable steps, justification gaps, "
        "unsupported conclusions, unconsidered alternatives\n\n"
        "Classify each claim as true / false / uncertain. Explain why." + _REVIEWER_STRUCTURED_SUFFIX
    ),
    Mode.WRITE: (
        "You are a jaded literary critic. Be sceptical; make the work convince you:\n\n"
        "**STORY:** plot holes, forced motivations, pacing problems, "
        "artificial stakes, world-building contradictions\n\n"
        "**CRAFT:** genuine vs performed voice, style consistency, "
        "clichés-as-insights, purple prose, wooden dialogue, show vs tell\n\n"
        "**EMOTION:** genuine resonance vs manipulation, earned vs forced sentiment, "
        "complete vs abandoned arcs, universal themes in specific details\n\n"
        "**STRUCTURE:** confusing transitions, scenes that don't advance anything, "
        "symmetric section lengths/patterns (LLM tell), "
        "repetitive patterns, earned vs deus-ex-machina endings\n\n"
        "**ORIGINALITY:** derivative plots, rehashed concepts, "
        "predictable turns, depth of insight\n\n"
        "If previously marked insufficient: what would elevate it from competent to compelling?\n\n"
        "Be ruthlessly honest. Acknowledge what works." + _REVIEWER_STRUCTURED_SUFFIX
    ),
}

_REVIEW_TRIAGE_INSTRUCTION = (
    "\n\nUse the REVIEW TRIAGE to identify which specific elements to keep "
    "or discard from each candidate. You may keep parts of an otherwise "
    "weak candidate."
)

_BANNED_PHRASES = (
    "delve into, leverage/navigate/anchor/tapestry/unpack (metaphorical), "
    "utilize (say 'use'), cutting-edge, groundbreaking, paradigm shift, "
    "comprehensive overview, holistic, synergy/synergistic, empower, "
    "deep dive, think outside the box, move the needle, take offline, "
    "best-in-class, learnings (say 'lessons learnt'), alignment (say 'agreement'), "
    "it is worth noting, it's important to note, it's crucial to, "
    "it goes without saying, needless to say, "
    "as previously mentioned, in the context of, in today's world, "
    "at the end of the day, when it comes to, in conclusion, "
    "verbs abused as nouns (the ask, the solve, the build)"
)

_TONE_BASE = (
    "Vary sentence length naturally. No \"it's not X, it's Y\" patterns. "
    "Em dashes only for genuine parentheticals. "
    f"Banned: {_BANNED_PHRASES}. "
    "Cut filler. Output must be tighter than combined inputs."
)

_TONE_INSTRUCTION = f"\n\nTone: expert human, not an LLM. Precise vocabulary. {_TONE_BASE}"

_TONE_INSTRUCTION_WRITER = (
    f"\n\nTone: skilled human author with a distinctive voice. "
    f"Lyrical where warranted, never overembellished. Rich vocabulary, no purple prose. {_TONE_BASE}"
)

_SYNTHESIZER_SYSTEM: dict[Mode, str] = {
    Mode.RESEARCH: (
        "Merge strongest candidate elements. Resolve contradictions. "
        "Remove unsupported claims. Verify every [N] marker matches the correct reference entry. "
        "Steelman objections before resolving them. "
        "Produce a coherent, well-cited summary.\n\n"
        "DISCARD weak or unsupported content. Never hedge or include blindly."
        + _REVIEW_TRIAGE_INSTRUCTION
        + _TONE_INSTRUCTION
    ),
    Mode.CODE: (
        "Fix and refine code using reviewer feedback. "
        "Produce consistent, testable, production-ready output.\n\n"
        "DISCARD inferior implementations. Never merge mechanically." + _REVIEW_TRIAGE_INSTRUCTION + _TONE_INSTRUCTION
    ),
    Mode.EDIT: (
        "Apply reviewer suggestions. Produce the clearest, most concise version.\n\n"
        "DISCARD weaker phrasings. Resolve conflicting suggestions." + _REVIEW_TRIAGE_INSTRUCTION + _TONE_INSTRUCTION
    ),
    Mode.CHECK: (
        "Produce a structured validation report: main claims, validity assessment for each, evidence gaps.\n\n"
        "Make definitive assessments. Never hedge." + _REVIEW_TRIAGE_INSTRUCTION + _TONE_INSTRUCTION
    ),
    Mode.WRITE: (
        "Select best elements across drafts. Discard weak material. "
        "Produce a single coherent piece with a unified voice.\n\n"
        "DISCARD weaker drafts or sections." + _REVIEW_TRIAGE_INSTRUCTION + _TONE_INSTRUCTION_WRITER
    ),
}


_SYNTHESIS_FORMAT = (
    "\n\n---\nBefore your final output, emit a JSON block on a single line:\n"
    '{"crossfire_synthesis": {"attributions": ['
    '{"index": 0, "kept": ["element a", "element b"], '
    '"discarded": ["element c"]}, ...], '
    '"notes": "<brief rationale>"}}\n'
    "Then produce your synthesized output below."
)


_SEARCH_SECTION_HEADER = "\n\n[SEARCH RESULTS]\n"


def _format_search_block(search_results: str) -> str:
    """Wrap non-empty *search_results* in a ``[SEARCH RESULTS]`` section."""
    if not search_results:
        return ""
    return _SEARCH_SECTION_HEADER + search_results


_SEARCH_INSTRUCTION = (
    "\n\nIf you need to search the web for information, emit a JSON object "
    "on the LAST non-empty line of your response:\n"
    '{"crossfire_search": {"query": "your search query"}}\n'
    "You may request at most one search."
)


_MARKDOWN_DECORATION_REGEX = re.compile(r"^#{1,6}\s+|[*_]{1,3}")

_STRENGTHS_REGEX = re.compile(r"^STRENGTHS:\s*(.+)", re.IGNORECASE)
_WEAKNESSES_REGEX = re.compile(r"^WEAKNESSES:\s*(.+)", re.IGNORECASE)
_SEVERITY_REGEX = re.compile(r"^SEVERITY:\s*(\w+)", re.IGNORECASE)

# Split on commas that are NOT inside parentheses or brackets.
_ITEM_SPLIT_REGEX = re.compile(r",\s*(?![^()]*\))(?![^\[\]]*\])")


@dataclass
class ReviewVerdict:
    """Parsed structured output from a reviewer."""

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    severity: str = ""


def parse_review_verdict(text: str) -> ReviewVerdict:
    """Extracts STRENGTHS, WEAKNESSES, and SEVERITY from review text into a verdict.

    ``severity`` is one of ``"material"``, ``"nitpick"``, ``"none"``, or ``""`` (empty)
    when the reviewer did not include a SEVERITY line.
    """
    verdict = ReviewVerdict()
    for line in text.split("\n"):
        line = _MARKDOWN_DECORATION_REGEX.sub("", line).strip()
        match = _STRENGTHS_REGEX.match(line)
        if match:
            verdict.strengths.extend(item.strip() for item in _ITEM_SPLIT_REGEX.split(match.group(1)) if item.strip())
            continue
        match = _WEAKNESSES_REGEX.match(line)
        if match:
            verdict.weaknesses.extend(item.strip() for item in _ITEM_SPLIT_REGEX.split(match.group(1)) if item.strip())
            continue
        match = _SEVERITY_REGEX.match(line)
        if match:
            verdict.severity = match.group(1).lower()
    return verdict


_CONTROL_ENVELOPE_REGEX = re.compile(r"^CROSSFIRE-(VERDICT|ADJUDICATION)\s+(\{.*\})\s*$")
_VERDICT_SEVERITIES: frozenset[str] = frozenset({"material", "nitpick", "none"})


class EnvelopeError(ValueError):
    """Raised when a machine-readable control envelope is missing or malformed."""


@dataclass(frozen=True)
class ControlEnvelope:
    """Parsed control envelope: a fixed ASCII line that ends a reviewer or adjudicator reply.

    The prose may be in any language, so the loop never reads a verdict out of it. Only this line decides whether a
    round may stop early, and an absent or invalid line is never treated as approval.
    """

    kind: str
    severity: str = ""
    weaknesses: int = 0
    claims: int = 0
    unavailable: int = 0


def _parse_envelope_payload(kind: str, payload: str) -> ControlEnvelope:
    """Validates one envelope payload against its closed schema."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exception:
        raise EnvelopeError(f"Malformed CROSSFIRE-{kind} envelope: the payload is not JSON.") from exception
    if not isinstance(data, dict):
        raise EnvelopeError(f"Malformed CROSSFIRE-{kind} envelope: the payload is not a JSON object.")

    if kind == "VERDICT":
        severity = data.get("severity")
        weaknesses = data.get("weaknesses")
        if severity not in _VERDICT_SEVERITIES:
            raise EnvelopeError(
                f"Malformed CROSSFIRE-VERDICT envelope: severity must be one of "
                f"{', '.join(sorted(_VERDICT_SEVERITIES))}, got {severity!r}."
            )
        if not isinstance(weaknesses, int) or isinstance(weaknesses, bool) or weaknesses < 0:
            raise EnvelopeError("Malformed CROSSFIRE-VERDICT envelope: weaknesses must be a non-negative integer.")
        return ControlEnvelope(kind="verdict", severity=str(severity), weaknesses=weaknesses)

    claims = data.get("claims")
    unavailable = data.get("unavailable")
    for name, value in (("claims", claims), ("unavailable", unavailable)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EnvelopeError(f"Malformed CROSSFIRE-ADJUDICATION envelope: {name} must be a non-negative integer.")
    assert isinstance(claims, int) and isinstance(unavailable, int)
    return ControlEnvelope(kind="adjudication", claims=claims, unavailable=unavailable)


def parse_control_envelope(text: str) -> ControlEnvelope:
    """Parses the last control envelope in *text*.

    The envelope must be the last non-empty line. An envelope followed by more prose is a quotation, not a verdict, and
    the model may still deny it in the text after it. Raises :class:`EnvelopeError` when the trailer is absent or
    malformed; the loop treats both as ``envelope_invalid``: never clean, and early stop is forbidden for that round.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        raise EnvelopeError("No CROSSFIRE-VERDICT or CROSSFIRE-ADJUDICATION envelope found.")
    match = _CONTROL_ENVELOPE_REGEX.match(lines[-1])
    if match is None:
        raise EnvelopeError("No CROSSFIRE-VERDICT or CROSSFIRE-ADJUDICATION envelope on the final line.")
    return _parse_envelope_payload(match.group(1), match.group(2))


def _build_review_triage(
    candidates: list[Candidate],
    reviews: list[Review],
) -> str:
    """Builds a per-candidate review triage from reviewer verdicts."""
    keep: dict[int, list[str]] = defaultdict(list)
    discard: dict[int, list[str]] = defaultdict(list)

    for review in reviews:
        verdict = parse_review_verdict(review.text)
        for strength in verdict.strengths:
            if strength not in keep[review.candidate_index]:
                keep[review.candidate_index].append(strength)
        for weakness in verdict.weaknesses:
            if weakness not in discard[review.candidate_index]:
                discard[review.candidate_index].append(weakness)

    if not keep and not discard:
        return ""

    lines = ["[REVIEW TRIAGE]"]
    for candidate in candidates:
        kept = keep.get(candidate.index, [])
        discarded = discard.get(candidate.index, [])
        if not kept and not discarded:
            continue
        lines.append(f"Candidate {candidate.index}:")
        if kept:
            lines.append(f"  KEEP: {', '.join(kept)}")
        if discarded:
            lines.append(f"  DISCARD: {', '.join(discarded)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_enrichment_prompt(
    *,
    mode: Mode,
    instruction: str,
    context: str = "",
) -> tuple[str, str]:
    rules_text = MODE_RULES[mode]
    parts: list[str] = [
        f"[ORIGINAL INSTRUCTION]\n{instruction}",
        f"\n[MODE RULES]\n{rules_text}",
    ]
    if context:
        parts.append(f"\n[CONTEXT]\n{context}")
    parts.append(
        "\n\nRewrite the instruction above into a detailed, structured brief "
        "that incorporates the mode rules. Output only the enriched instruction."
    )
    return _ENRICHER_SYSTEM, "".join(parts)


def build_generator_prompt(
    *,
    mode: Mode,
    instruction: str,
    context: str = "",
    rules: str | None = None,
    previous_synthesis: str = "",
    round_num: int = 1,
    search_results: str = "",
    search_enabled: bool = False,
    language_tag: str = "",
) -> tuple[str, str]:
    rules_text = rules or MODE_RULES[mode]
    system = _GENERATOR_SYSTEM[mode]

    parts: list[str] = [
        f"[INSTRUCTION]\n{instruction}",
        f"\n[RULES]\n{rules_text}",
    ]

    if context:
        parts.append(f"\n[CONTEXT]\n{context}")

    if round_num > 1 and previous_synthesis:
        parts.append(f"\n[PREVIOUS SYNTHESIS (round {round_num - 1})]\n{previous_synthesis}")

    parts.append(_format_search_block(search_results))

    if search_enabled:
        parts.append(_SEARCH_INSTRUCTION)

    if language_tag:
        parts.append(language_tag)

    user = "".join(parts)
    return system, user


def build_reviewer_prompt(
    *,
    mode: Mode,
    instruction: str,
    candidate: Candidate,
    rules: str | None = None,
    search_results: str = "",
    search_enabled: bool = False,
    language_tag: str = "",
) -> tuple[str, str]:
    rules_text = rules or MODE_RULES[mode]
    system = _REVIEWER_SYSTEM[mode]

    parts: list[str] = [
        f"[INSTRUCTION]\n{instruction}",
        f"\n[RULES]\n{rules_text}",
        f"\n[CANDIDATE ({candidate.label})]\n{candidate.text}",
    ]

    parts.append(_format_search_block(search_results))

    if search_enabled:
        parts.append(_SEARCH_INSTRUCTION)

    user = "".join(parts)
    return system, user


def build_synthesizer_prompt(
    *,
    mode: Mode,
    instruction: str,
    candidates: list[Candidate],
    reviews: list[Review],
    rules: str | None = None,
    language_tag: str = "",
) -> tuple[str, str]:
    rules_text = rules or MODE_RULES[mode]
    # The attribution block is essential. Fable refuses it (a safety-classifier false positive on
    # instructions to keep/discard/attribute candidate content, reproduced live across many phrasings),
    # so Fable reviews instead. Opus 5.5 accepts the block, so Claude can still synthesise.
    # See the synthesizer pool in crossfire.toml.
    system = _SYNTHESIZER_SYSTEM[mode] + _SYNTHESIS_FORMAT

    parts: list[str] = [
        f"[INSTRUCTION]\n{instruction}",
        f"\n[RULES]\n{rules_text}",
    ]

    for candidate in candidates:
        parts.append(f"\n[CANDIDATE {candidate.index} ({candidate.label})]\n{candidate.text}")

    brief = _build_review_triage(candidates, reviews)
    if brief:
        parts.append(f"\n{brief}")

    for review in reviews:
        parts.append(f"\n[REVIEW for candidate {review.candidate_index} by {review.model}]\n{review.text}")

    if language_tag:
        parts.append(language_tag)

    user = "".join(parts)
    return system, user


_ADJUDICATOR_SYSTEM = (
    "You compare several answers to the same question, each written in its own language, claim by claim.\n\n"
    "- Judge each claim only on the passages the branches actually wrote; never translate a passage into another "
    "language and treat the translation as a source\n"
    "- Cite evidence ids from the registry, and nothing else\n"
    "- A claim one branch omits is absent_in_branch, not a disagreement\n"
    "- A branch you cannot read is branch_unreadable: say so rather than guessing\n"
    "- Agreement with no evidence id is no_evidence\n"
    "- Publish disagreement; never collapse it into a single answer"
)

_ADJUDICATION_KEY = "crossfire_adjudication"
_ADJUDICATION_STATES: tuple[str, ...] = (
    "agrees",
    "disagrees",
    "absent_in_branch",
    "branch_unreadable",
    "no_evidence",
)


def build_cross_branch_adjudicator_prompt(
    claims: Sequence[Any],
    syntheses: Mapping[str, str],
    evidence: Sequence[Any],
) -> tuple[str, str]:
    """Builds a packet prompt with only its verbatim passages and their cited evidence."""
    passages: dict[str, list[str]] = {branch: [] for branch in syntheses}
    for claim in claims:
        for branch, passage in claim.by_branch.items():
            if branch in passages and passage and passage not in passages[branch]:
                passages[branch].append(passage)
    packet_texts = {branch: "\n".join(texts) for branch, texts in passages.items()}
    markers = {branch: set(re.findall(r"\[\d+\]", text)) for branch, text in packet_texts.items()}
    relevant_evidence: list[Any] = []
    for span in evidence:
        if span.branch:
            text = packet_texts.get(span.branch, "")
            reference = re.match(r"^\[\d+\]", span.text)
            if span.text in text or (reference and reference.group(0) in markers.get(span.branch, set())):
                relevant_evidence.append(span)
        elif any(span.text in text for text in packet_texts.values()):
            relevant_evidence.append(span)

    parts: list[str] = ["[CROSS-BRANCH ADJUDICATION]"]
    for branch, text in packet_texts.items():
        parts.append(f"\n[BRANCH {branch}]\n{text or '(omitted)'}")
    parts.append("\n[EVIDENCE]")
    for span in relevant_evidence:
        parts.append(f"\n{span.id} ({span.origin}): {span.text}")
    parts.append("\n[CLAIMS]")
    for claim in claims:
        parts.append(f"\n{claim.key}: {claim.text}")
        for branch, passage in claim.by_branch.items():
            parts.append(f"\n  {branch}: {passage or '(omitted)'}")
    parts.append(
        "\n\nReply with one JSON object on a single line, then the envelope:\n"
        '{"crossfire_adjudication": {"verdicts": [{"claim": "<claim key>", "state": "'
        + "|".join(_ADJUDICATION_STATES)
        + '", "by_branch": {"<language>": "<verbatim passage or empty>"}, "evidence_ids": ["<id>", ...]}]}}\n'
        'CROSSFIRE-ADJUDICATION {"claims": <integer>, "unavailable": <integer>}\n'
        "The envelope is the last line, with ASCII keys and whole numbers."
    )
    return _ADJUDICATOR_SYSTEM, "".join(parts)


def _json_object_with_key(text: str, key: str) -> dict[str, Any] | None:
    """Returns the first JSON object carrying *key* at the top level, on its own line or inside a fence."""
    for line in text.split("\n"):
        candidate = line.strip()
        if key not in candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get(key), dict):
            payload: dict[str, Any] = parsed[key]
            return payload
    return None


def parse_cross_branch_verdicts(text: str) -> list[CrossBranchVerdict]:
    """Parses the verdicts out of an adjudicator reply.

    Raises :class:`EnvelopeError` when the reply carries no adjudication object, when a state is outside the closed enum
    or when a verdict names no claim. A caller that cannot parse a reply records ``adjudication_incomplete`` rather than
    publishing a comparison it did not get.
    """
    payload = _json_object_with_key(text, _ADJUDICATION_KEY)
    if payload is None:
        raise EnvelopeError("No crossfire_adjudication object found in the adjudicator reply.")

    verdicts: list[CrossBranchVerdict] = []
    entries = payload.get("verdicts")
    if not isinstance(entries, list):
        raise EnvelopeError("The crossfire_adjudication object carries no verdicts list.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise EnvelopeError("A verdict is not a JSON object.")
        claim = str(entry.get("claim", ""))
        if "\n" in claim or "\r" in claim:
            raise EnvelopeError("A verdict claim must occupy a single line.")
        claim = claim.strip()
        if not claim:
            raise EnvelopeError("A verdict must name its claim.")
        state_text = str(entry.get("state", ""))
        if state_text not in _ADJUDICATION_STATES:
            raise EnvelopeError(
                f"Unknown verdict state {state_text!r}. Known states: {', '.join(_ADJUDICATION_STATES)}"
            )
        raw_by_branch = entry.get("by_branch") or {}
        raw_ids = entry.get("evidence_ids") or []
        if not isinstance(raw_by_branch, dict) or not isinstance(raw_ids, list):
            raise EnvelopeError("A verdict's by_branch must be an object and its evidence_ids a list.")
        verdicts.append(
            CrossBranchVerdict(
                claim=claim,
                state=VerdictState(state_text),
                by_branch={str(branch): str(passage) for branch, passage in raw_by_branch.items()},
                evidence_ids=[str(span_id) for span_id in raw_ids],
            )
        )
    return verdicts


def build_language_tag(language: str) -> str:
    """Returns the response-language instruction for a branch answering in *language*.

    Evidence is quoted in the language it appears in. A quotation is never translated, because a rendering is not the
    passage a reviewer or an adjudicator judged.
    """
    return (
        f"\n\nRespond in {language}. Quote evidence verbatim in the language it appears; never translate a quotation."
    )


_TRANSLATOR_SYSTEM = (
    "You are a translator. Translate the text you are given into the target language.\n\n"
    "- Preserve the structure, headings and line breaks exactly\n"
    "- Reproduce every number, date, citation, identifier, URL and quoted title byte for byte\n"
    "- Do not translate names of people, journals, publishers or works\n"
    "- Add no commentary, no notes and no preamble\n"
    "- Output the translation alone"
)


def build_translation_prompt(direction: str, path: str) -> tuple[str, str]:
    """Builds the translation prompt for one instruction.

    *direction* is ``"<source>-><target>"`` and *path* is the text to translate. Citations, numbers and identifiers
    travel through unchanged, and the fidelity gate checks that they did.
    """
    user = f"[TRANSLATE {direction}]\n{path}"
    return _TRANSLATOR_SYSTEM, user


_FENCED_BLOCK_REGEX = re.compile(r"```([a-zA-Z0-9_-]*)[ \t]*\n(.*?)```[ \t]*\n?", re.DOTALL)


def _parse_decision_json(text: str) -> dict[str, Any] | None:
    """Parses the decision object from JSON text with a top-level ``crossfire_synthesis`` key.

    Returns ``None`` if the JSON is invalid or lacks an object under that key.
    """
    stripped = text.strip()
    if "crossfire_synthesis" not in stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("crossfire_synthesis"), dict):
        return None
    decision: dict[str, Any] = parsed["crossfire_synthesis"]
    return decision


def _parse_decision_block(body: str) -> dict[str, Any] | None:
    """Parses a decision object from a complete JSON body or a standalone JSON object on its own line.

    Returns ``None`` if no object exists under the top-level ``crossfire_synthesis`` key.
    """
    decision = _parse_decision_json(body)
    if decision is not None:
        return decision
    for line in body.split("\n"):
        decision = _parse_decision_json(line)
        if decision is not None:
            return decision
    return None


def _first_synthesis_decision(text: str) -> dict[str, Any] | None:
    """Returns the first decision from a standalone JSON object on its own line or from a fenced object.

    Standalone objects must be outside code fences. Fenced objects must be unlabelled or marked ``json``.
    Returns ``None`` if no decision is found.
    """
    cursor = 0
    for match in _FENCED_BLOCK_REGEX.finditer(text):
        for line in text[cursor : match.start()].split("\n"):
            decision = _parse_decision_json(line)
            if decision is not None:
                return decision
        language, body = match.group(1), match.group(2)
        if language.lower() in ("", "json"):
            decision = _parse_decision_block(body)
            if decision is not None:
                return decision
        cursor = match.end()
    for line in text[cursor:].split("\n"):
        decision = _parse_decision_json(line)
        if decision is not None:
            return decision
    return None


def parse_synthesis_decision(text: str) -> tuple[list[CandidateDecision], str]:
    """Extracts candidate decisions and notes from synthesizer output.

    The parser reads standalone JSON objects, each on its own line, outside code fences.
    It also reads JSON objects inside unlabelled fences or fences marked ``json``.
    Each object must have the top-level ``crossfire_synthesis`` key.
    Returns ``([], "")`` when it finds no valid decision object.
    """
    decision_data = _first_synthesis_decision(text)
    if decision_data is None:
        return [], ""

    notes = decision_data.get("notes", "")

    decisions: list[CandidateDecision] = []
    for entry in decision_data.get("attributions", []):
        decisions.append(
            CandidateDecision(
                index=entry.get("index", 0),
                kept=entry.get("kept", []),
                discarded=entry.get("discarded", []),
            )
        )
    return decisions, notes


def _strip_bare_decision_lines(text: str) -> str:
    """Removes standalone JSON objects, each on its own line, with the top-level key outside code fences."""
    lines = [line for line in text.split("\n") if _parse_decision_json(line) is None]
    return "\n".join(lines)


def strip_synthesis_decision(text: str) -> str:
    """Removes standalone JSON objects, each on its own line, and fenced blocks with a top-level ``crossfire_synthesis``
    key.

    Standalone objects are removed only outside code fences. Fenced blocks are removed only when unlabelled or marked
    ``json`` and they contain a JSON object with the key. Other fences and text outside removed blocks remain unchanged.
    """

    parts: list[str] = []
    cursor = 0
    for match in _FENCED_BLOCK_REGEX.finditer(text):
        parts.append(_strip_bare_decision_lines(text[cursor : match.start()]))
        language, body = match.group(1), match.group(2)
        if language.lower() not in ("", "json") or _parse_decision_block(body) is None:
            parts.append(match.group(0))
        cursor = match.end()
    parts.append(_strip_bare_decision_lines(text[cursor:]))
    return "".join(parts).strip()
