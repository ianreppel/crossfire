"""Do-not-translate extraction and the translation fidelity gate.

Translation covers question and answer prose only. Reference blocks, author names, titles, journal and publisher names,
DOIs, ISBNs and URLs pass through byte-identical, and the gate rejects any translation that moves a numeral, a date, a
name or an identifier. A translation never stands in for evidence, so this gate is what keeps a rendering from
quietly replacing a citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A token a translator has no reason to translate; the renderer swaps the original text back in.
PLACEHOLDER_TEMPLATE = "\u27e6DT{n}\u27e7"

# A reference entry: a bracketed number followed by at least a name and a title.
_REFERENCE_LINE_REGEX = re.compile(r"^\s*\[\d+\]\s+\S.*\S")
_URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
_DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[^\s\"<>,;]+")
_ISBN_REGEX = re.compile(r"\b(?:ISBN(?:-1[03])?:?\s*)?(?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]\b")
_NUMERAL_REGEX = re.compile(r"\d+(?:[.,:/]\d+)*")


@dataclass(frozen=True)
class ProtectedSpan:
    """One span that travels untranslated, with the placeholder that carries it through a translator."""

    placeholder: str
    text: str


class FidelityError(ValueError):
    """Raised when a translation moved a numeral, a date, a name or an identifier."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def _placeholder(index: int) -> str:
    """Returns the placeholder token for the *index*-th protected span."""
    return PLACEHOLDER_TEMPLATE.format(n=index)


def _protect_identifiers(line: str, spans: list[ProtectedSpan]) -> str:
    """Replaces every URL, DOI and ISBN in *line* with a placeholder, appending each span to *spans*."""
    for pattern in (_URL_REGEX, _DOI_REGEX, _ISBN_REGEX):
        pieces: list[str] = []
        cursor = 0
        for match in pattern.finditer(line):
            if match.start() < cursor:
                continue
            span = ProtectedSpan(placeholder=_placeholder(len(spans)), text=match.group(0))
            spans.append(span)
            pieces.append(line[cursor : match.start()])
            pieces.append(span.placeholder)
            cursor = match.end()
        pieces.append(line[cursor:])
        line = "".join(pieces)
    return line


def extract_do_not_translate(text: str) -> tuple[str, tuple[ProtectedSpan, ...]]:
    """Returns *text* with its protected spans replaced by placeholders, plus the spans themselves.

    A whole reference line is protected: author, title, journal and year travel together, because a translator that
    renders a title has already broken the link between the citation and the work it names.
    """
    spans: list[ProtectedSpan] = []
    masked_lines: list[str] = []
    for line in text.split("\n"):
        if _REFERENCE_LINE_REGEX.match(line):
            span = ProtectedSpan(placeholder=_placeholder(len(spans)), text=line)
            spans.append(span)
            masked_lines.append(span.placeholder)
            continue
        masked_lines.append(_protect_identifiers(line, spans))
    return "\n".join(masked_lines), tuple(spans)


def restore_do_not_translate(text: str, spans: tuple[ProtectedSpan, ...]) -> str:
    """Returns *text* with every present placeholder replaced by its original span, byte for byte.

    A placeholder the translator dropped stays dropped: the gate reports it rather than inventing the text back.
    """
    for span in spans:
        text = text.replace(span.placeholder, span.text)
    return text


def fidelity_violations(source: str, translated: str, spans: tuple[ProtectedSpan, ...]) -> list[str]:
    """Returns every way *translated* fails to carry *source* faithfully.

    Each protected span must survive exactly once, and every numeral in the source must reappear byte-identical. The
    check runs on the translated text as the model returned it, so a translator that renders a title instead of copying
    the placeholder is caught.
    """
    violations: list[str] = []
    for span in spans:
        occurrences = translated.count(span.placeholder)
        if occurrences != 1:
            violations.append(
                f"the protected span {span.placeholder} appears {occurrences} times, expected exactly once: "
                f"{span.text[:80]!r}"
            )
    stripped = restore_do_not_translate(translated, spans)
    for numeral in dict.fromkeys(_NUMERAL_REGEX.findall(source)):
        if numeral not in stripped:
            violations.append(f"the numeral {numeral!r} is not byte-identical in the translation")
    return violations


def assert_fidelity(source: str, translated: str, spans: tuple[ProtectedSpan, ...]) -> None:
    """Raises :class:`FidelityError` when *translated* breaks fidelity to *source*."""
    violations = fidelity_violations(source, translated, spans)
    if violations:
        raise FidelityError(violations)
