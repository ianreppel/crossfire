"""Prompt construction for all modes and roles."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from crossfire.core.domain import Candidate, CandidateDecision, Mode, Review

MODE_RULES: dict[Mode, str] = {
    Mode.RESEARCH: (
        "RULES:\n"
        "- All factual claims MUST have inline [N] citation markers\n"
        "- End with a numbered References section in this exact format: "
        "[N] Authors, Title, Journal/Venue, Year. "
        "For websites/reports: Author/Organization, Title, URL, Year. "
        "NEVER fabricate DOIs. Omit DOIs entirely unless you are certain they are correct\n"
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
    "SEVERITY: material (factual errors, logic flaws, broken code) | nitpick (style only) | none"
)

_REVIEWER_SYSTEM: dict[Mode, str] = {
    Mode.RESEARCH: (
        "You are a skeptical peer reviewer. Verify every claim with rigour:\n\n"
        "**CITATIONS:** uncited claims, citations that don't support the claim, "
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
        "You are a jaded literary critic. Be skeptical; make the work convince you:\n\n"
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
        "Remove unsupported claims. Steelman objections before resolving them. "
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
        line = line.strip()
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
) -> tuple[str, str]:
    rules_text = rules or MODE_RULES[mode]
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

    user = "".join(parts)
    return system, user


def parse_synthesis_decision(text: str) -> tuple[list[CandidateDecision], str]:
    """Extracts the structured attribution decision from synthesizer output.

    Returns ``(decisions, notes)`` where *decisions* is a list of
    :class:`CandidateDecision` and *notes* is the rationale string.
    Returns ``([], "")`` when no valid JSON block is found.
    """
    for line in text.split("\n"):
        line = line.strip()
        if "crossfire_synthesis" not in line:
            continue
        try:
            parsed = json.loads(line)
            dec = parsed.get("crossfire_synthesis", {})
        except (json.JSONDecodeError, TypeError):
            continue

        notes = dec.get("notes", "")

        decisions = []
        for entry in dec.get("attributions", []):
            decisions.append(
                CandidateDecision(
                    index=entry.get("index", 0),
                    kept=entry.get("kept", []),
                    discarded=entry.get("discarded", []),
                )
            )
        return decisions, notes

    return [], ""


def strip_synthesis_decision(text: str) -> str:
    """Removes the ``crossfire_synthesis`` JSON line from synthesizer output."""
    lines: list[str] = text.split("\n")
    cleaned: list[str] = [line for line in lines if "crossfire_synthesis" not in line]
    return "\n".join(cleaned).strip()
