"""Deterministic hashing for attested artefacts.

A hash is how the journal proves an artefact is the one it recorded, and how ``--resume`` decides which units are
already done. One function, one encoding, so two writers can never disagree about the bytes.
"""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Returns the hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
