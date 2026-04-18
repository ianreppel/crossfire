"""Deterministic simulation for dry runs: fake LLM responses and fake search results"""

from __future__ import annotations

import hashlib

from crossfire.core.domain import Phase, Role


def simulate_response(
    *,
    instruction: str,
    mode: str,
    phase: Phase,
    role: Role,
    model: str,
    round_num: int,
    candidate_index: int | None = None,
) -> str:
    """Generates a fake LLM response deterministically.

    All inputs are hashed with SHA-256 to produce a hex digest that serves as a seedable source of pseudo-random values.
    Slices of the digest are used as unique-looking content chunks, so that different models, rounds, and candidates
    yield structurally varied but fully reproducible output.
    """
    extra = f"-c{candidate_index}" if candidate_index is not None else ""
    hash_input = f"{instruction}{mode}{phase}{role}{model}{round_num}{extra}"
    digest = hashlib.sha256(hash_input.encode()).hexdigest()

    third_hex_digit = int(digest[2], 16)
    emit_search = third_hex_digit > 10  # deterministic ~31% chance of a search request

    if role == "enricher":
        sections = [
            f"## Enriched Instruction (model={model})",
            f"Hash: {digest[:16]}",
            "",
            instruction,
            "",
            f"### Additional constraints ({digest[16:20]})",
            "- Output must be structured with clear section headers.",
            "- Include concrete examples where applicable.",
            "- Target length: 1500-3000 words.",
            f"- Ensure all claims are verifiable. Digest: {digest[20:28]}.",
        ]
        return "\n".join(sections)

    if role == "generator":
        sections = [
            f"## Generated Output (model={model}, round={round_num})",
            f"Hash: {digest[:16]}",
            "",
            f"### Section A ({digest[16:20]})",
            f"Content block alpha for instruction digest {digest[:8]}.",
            f"Claim: Statement-{digest[20:24]} is factual. [citation-{digest[24:28]}]",
            "",
            f"### Section B ({digest[28:32]})",
            f"Content block beta with analysis {digest[32:40]}.",
            "",
            "```python",
            f"def generated_{digest[40:44]}():",
            f'    return "{digest[44:52]}"',
            "```",
        ]
        if emit_search:
            sections.append(f'{{"crossfire_search": {{"query": "evidence for {digest[48:56]}"}}}}')
    elif role == "reviewer":
        simulated_score = (int(digest[0], 16) % 5) + 5  # deterministic 5-9
        claim_assessment = "supported" if int(digest[1], 16) > 7 else "uncertain"
        sections = [
            f"## Review (model={model}, round={round_num}, candidate={candidate_index})",
            f"Hash: {digest[:16]}",
            "",
            f"**Overall Score:** {simulated_score}/10",
            f"**Claim Assessment:** {claim_assessment}",
            "",
            "### Strengths",
            f"- Good structure ({digest[8:12]})",
            f"- Clear citations ({digest[12:16]})",
            "",
            "### Issues",
            f"- Missing evidence for claim {digest[16:20]}",
            f"- Redundancy in section {digest[20:24]}",
            "",
            f"STRENGTHS: good structure ({digest[8:12]}), clear citations ({digest[12:16]})",
            f"WEAKNESSES: missing evidence for claim {digest[16:20]}, redundancy in section {digest[20:24]}",
        ]
        if emit_search:
            sections.append(f'{{"crossfire_search": {{"query": "verify claim {digest[24:32]}"}}}}')
    else:  # synthesizer
        per_cand = [
            f'{{"index": 0, "kept": ["merged content ({digest[8:12]})"], "discarded": []}}',
        ]
        if int(digest[0], 16) > 7:
            per_cand.append(f'{{"index": 1, "kept": [], "discarded": ["weak structure ({digest[12:16]})"]}}')
        per_cand_json = ", ".join(per_cand)
        sections = [
            f'{{"crossfire_synthesis": {{"attributions": [{per_cand_json}], '
            f'"notes": "Kept strongest elements ({digest[:8]})"}}}}',
            "",
            f"## Synthesized Output (model={model}, round={round_num})",
            f"Hash: {digest[:16]}",
            "",
            f"### Merged Content ({digest[16:24]})",
            f"Refined content combining best elements. Digest: {digest[24:32]}.",
            f"Verified claim: Statement-{digest[32:36]} [citation-{digest[36:40]}]",
            "",
            "```python",
            f"def synthesized_{digest[40:44]}():",
            f'    return "{digest[44:52]}"',
            "```",
        ]
    return "\n".join(sections)


def simulate_search(
    *,
    instruction: str,
    mode: str,
    role: Role,
    model: str,
    round_num: int,
    query: str,
) -> str:
    """Produces (fake) search results deterministically."""
    hash_input = f"{instruction}{mode}{role}{model}{round_num}search{query}"
    digest = hashlib.sha256(hash_input.encode()).hexdigest()

    first_hex_digit = int(digest[0], 16)
    result_count = (first_hex_digit % 3) + 2  # yields 2-4 results
    lines: list[str] = []
    for result_index in range(result_count):
        slug = digest[result_index * 8 : (result_index + 1) * 8]  # 8-char slices as unique identifiers
        lines.append(
            f"- Result {result_index + 1}: [{query[:40]}] "
            f"(https://example.com/{slug}) "
            f"Simulated finding #{slug[:4]} relevant to the query."
        )
    return "\n".join(lines)
