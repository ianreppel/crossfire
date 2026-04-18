"""Deterministic compression for prompts when the prompt exceeds the models' token budget.

Priority order for compression:
1. Candidates
2. Reviews
3. Context

Note that task instructions are *never* compressed!

Key trade-offs:
- Extractive compression is deterministic and fast, but it can produce choppy output when many lines are dropped
- Generative compression is stochastic, slower, and it may produce smoother output, but it introduces model bias
- The two-pass approach drops entire (superfluous) sections first, and falls back to line-level
  filtering only when needed with a heuristic
- The heuristic is relatively simple: code > citations > headers > bullet points > long lines >
  short lines > blank lines; it always preserves code blocks, citations, and headers
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crossfire.core import logging as log
from crossfire.core.domain import Phase, Role
from crossfire.core.tokens import compute_token_budget, count_tokens, estimate_tokens

_CITATION_REGEX = re.compile(r"\[.+?\]")
_HEADER_REGEX = re.compile(r"^#{1,6}\s", re.MULTILINE)

# Scoring scale (0-100): lines scoring at or above the threshold are NEVER dropped.
# 100 = code fence markers, 95 = code content / blank lines inside code, 90 = citations,
# 80 = headers, 1-3 = prose (bullets and long lines score slightly higher).
_PROTECTED_SCORE_THRESHOLD = 80.0


@dataclass
class CompressedText:
    """Result of a compression pass, with both the original and compressed text included."""

    original: str
    compressed: str
    tokens_before: int
    tokens_after: int


def _is_code_block_line(line: str) -> bool:
    return line.strip().startswith("```")


def _is_citation_line(line: str) -> bool:
    return bool(_CITATION_REGEX.search(line))


def _is_header(line: str) -> bool:
    return bool(_HEADER_REGEX.match(line))


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Splits text into (headers, body_lines) sections."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_body: list[str] = []

    for line in lines:
        if _is_header(line):
            if current_header or current_body:
                sections.append((current_header, current_body))
            current_header = line
            current_body = []
        else:
            current_body.append(line)

    if current_header or current_body:
        sections.append((current_header, current_body))

    return sections


def _score_line(line: str, *, in_code_block: bool = False) -> float:
    """Scores a line by importance: higher means more important to keep."""
    score = 1.0
    stripped = line.strip()
    if not stripped:
        return 0.1 if not in_code_block else 95.0
    if _is_code_block_line(line):
        return 100.0
    if in_code_block:
        return 95.0
    if _is_citation_line(line):
        return 90.0
    if _is_header(line):
        return _PROTECTED_SCORE_THRESHOLD
    if stripped.startswith(("-", "*", "•")):
        score += 2.0
    if len(stripped) > 80:
        score += 1.0
    return score


def _score_lines_with_code_tracking(lines: list[str]) -> list[tuple[int, float, str]]:
    """Scores all lines while tracking code-block boundaries to protect code content."""
    in_code = False
    scored: list[tuple[int, float, str]] = []
    for index, line in enumerate(lines):
        if _is_code_block_line(line):
            scored.append((index, 100.0, line))
            in_code = not in_code
        else:
            scored.append((index, _score_line(line, in_code_block=in_code), line))
    return scored


def _trim_sections(text: str, target_tokens: int) -> str:
    """Drops lowest-priority sections first to preserve structure (pass 1)."""
    sections = _split_sections(text)
    if not sections:
        return text

    kept: list[tuple[str, list[str]]] = list(sections)

    while len(kept) > 1:
        assembled = _assemble(kept)
        if estimate_tokens(assembled) <= target_tokens:
            return assembled
        _last_header, last_body = kept[-1]
        has_protected = any(_is_citation_line(line) or _is_code_block_line(line) for line in last_body)
        if not has_protected:
            kept.pop()
        else:
            break

    assembled = _assemble(kept)
    if estimate_tokens(assembled) <= target_tokens:
        return assembled

    return _trim_section_bodies(kept, target_tokens)


def _estimate_line_tokens(line: str) -> int:
    """Returns the token count for a single line (1 for blank lines)."""
    if not line.strip():
        return 1
    return count_tokens(line)


def _trim_section_bodies(sections: list[tuple[str, list[str]]], target_tokens: int) -> str:
    """Drops low-scoring lines within section bodies when section-level trimming is insufficient."""
    assembled = _assemble(sections)
    current_tokens = estimate_tokens(assembled)

    for section_index in range(len(sections) - 1, -1, -1):
        header, body = sections[section_index]
        original_blanks = {index for index, line in enumerate(body) if line == ""}
        scored = _score_lines_with_code_tracking(body)
        scored.sort(key=lambda x: x[1])
        for drop_index, score, line in scored:
            if score >= _PROTECTED_SCORE_THRESHOLD:
                break
            saved = _estimate_line_tokens(line)
            body[drop_index] = ""
            current_tokens -= saved
            if current_tokens <= target_tokens:
                cleaned = [line for index, line in enumerate(body) if line or index in original_blanks]
                sections[section_index] = (header, cleaned)
                return _assemble(sections)

    return _assemble(sections)


def _assemble(sections: list[tuple[str, list[str]]]) -> str:
    """Re-joins header + body sections."""
    parts: list[str] = []
    for header, body in sections:
        if header:
            parts.append(header)
        parts.extend(body)
    return "\n".join(parts)


def _compress_by_rank(text: str, target_tokens: int) -> str:
    """Filters lines aggressively by heuristic score (pass 2)."""
    lines = text.split("\n")
    scored = _score_lines_with_code_tracking(lines)
    scored.sort(key=lambda x: x[1])

    current_tokens = estimate_tokens(text)
    removed: set[int] = set()
    for index, score, line in scored:
        if score >= _PROTECTED_SCORE_THRESHOLD:
            break
        removed.add(index)
        current_tokens -= _estimate_line_tokens(line)
        if current_tokens <= target_tokens:
            break

    result = "\n".join(line for index, line in enumerate(lines) if index not in removed)
    return result


def compress(text: str, target_tokens: int, max_passes: int = 2) -> CompressedText:
    """Applies up to *max_passes* of compression."""
    tokens_before = estimate_tokens(text)
    if tokens_before <= target_tokens:
        return CompressedText(
            original=text,
            compressed=text,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
        )

    result = _trim_sections(text, target_tokens) if max_passes >= 1 else text

    if max_passes >= 2 and estimate_tokens(result) > target_tokens:
        result = _compress_by_rank(result, target_tokens)

    return CompressedText(
        original=text,
        compressed=result,
        tokens_before=tokens_before,
        tokens_after=estimate_tokens(result),
    )


# -- prompt fitting ---

_MIN_COMPRESSION_TARGET = 100


def _find_span(container: str, text: str, search_from: int = 0) -> tuple[int, int]:
    """Finds *text* in *container* starting at *search_from*, returning (start, end) or (-1, -1)."""
    position = container.find(text, search_from)
    if position < 0:
        return (-1, -1)
    return (position, position + len(text))


def _try_compress_part(
    prompt: str,
    text: str,
    reason: str,
    available_tokens: int,
    max_passes: int,
    *,
    phase: Phase,
    role: Role,
    model: str,
    round_num: int,
) -> tuple[str, str]:
    """Attempts to compress one part within *prompt* to fit the token budget.

    Returns (updated_prompt, updated_text).  If the part cannot be found or compression wouldn't help, both are returned
    unchanged.
    """
    start, end = _find_span(prompt, text)
    if start < 0:
        return prompt, text

    overshoot = estimate_tokens(prompt) - available_tokens
    if overshoot <= 0:
        return prompt, text

    text_tokens = estimate_tokens(text)
    target = max(text_tokens - overshoot, _MIN_COMPRESSION_TARGET)
    if target >= text_tokens:
        return prompt, text

    compressed = compress(text, target, max_passes=max_passes)
    if compressed.tokens_after >= compressed.tokens_before:
        return prompt, text

    log.log_compression_applied(
        phase=phase,
        role=role,
        model=model,
        round=round_num,
        tokens_before=compressed.tokens_before,
        tokens_after=compressed.tokens_after,
        reason=reason,
    )
    updated_prompt = prompt[:start] + compressed.compressed + prompt[end:]
    return updated_prompt, compressed.compressed


def compress_prompt_components(
    *,
    system_prompt: str,
    user_prompt: str,
    context_window: int,
    max_output_tokens: int,
    phase: Phase,
    role: Role,
    model: str,
    round_num: int,
    compressible_parts: list[tuple[str, str]],
) -> tuple[str, bool]:
    """Fits *user_prompt* into the token budget by compressing parts in priority order.

    Each entry in *compressible_parts* is a ``(text, reason)`` pair.
    Two passes are made: gentle (section-level) then aggressive (line-level).
    """
    budget = compute_token_budget(context_window)
    available = budget - max_output_tokens - estimate_tokens(system_prompt)

    if estimate_tokens(user_prompt) <= available:
        return user_prompt, True

    result = user_prompt
    current_texts = [text for text, _ in compressible_parts]

    for pass_num in range(1, 3):
        for index, (_, reason) in enumerate(compressible_parts):
            if not current_texts[index]:
                continue
            result, current_texts[index] = _try_compress_part(
                result,
                current_texts[index],
                reason,
                available,
                max_passes=pass_num,
                phase=phase,
                role=role,
                model=model,
                round_num=round_num,
            )
            if estimate_tokens(result) <= available:
                return result, True

    return result, estimate_tokens(result) <= available
