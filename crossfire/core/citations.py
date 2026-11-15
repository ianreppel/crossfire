"""Shared citation boundaries and identifier extraction for translation and evidence."""

from __future__ import annotations

import re

_CITATION_LINE_REGEX = re.compile(r"^\s*\[\d+\]\s+\S.*\S")
_IDENTIFIER_REGEX = re.compile(
    r"https?://\S+|www\.\S+|\b10\.\d{4,9}/[^\s\"<>,;]+|\bISBN(?:-1[03])?:?\s*[\dXx-]+"
    r"|\b(?:97[89][- ]?(?:\d[- ]?){9}\d|\d[- ]?(?:\d[- ]?){8}[\dXx])\b"
)


def is_citation_line(text: str) -> bool:
    """Returns whether a line is a numbered reference entry."""
    return _CITATION_LINE_REGEX.match(text) is not None


def identifiers(text: str) -> tuple[str, ...]:
    """Extracts URL, DOI and ISBN identifiers in reading order."""
    return tuple(match.group(0) for match in _IDENTIFIER_REGEX.finditer(text))
