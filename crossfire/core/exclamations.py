"""The Simpsons exclamations prepended to exception messages for a touch of levity."""

from __future__ import annotations

import hashlib

_EXCLAMATIONS: tuple[str, ...] = (
    "D'oh!",
    "Craptastic!",
    "Well, that both sucks and blows!",
    "I'm in danger!",
    "Ay, caramba!",
    "Don't have a cow, man!",
    "Worst. Error. Ever.",
    "Inflammable means flammable?!",
    "Why you little...",
    "Stupid Flanders!",
    "Whatever, I'll be at Moe's.",
    "Eat my shorts!",
    "Mmm... errors.",
    "I didn't do it!",
    "It tastes like burning!",
    "Save me, Jeebus!",
    "Eeep!",
    "Ah, geez!",
    "Ha!",
    "Haw haw!",
    "Clean up in all the aisles!",
    "Ah, nuts!",
)


def exclaim(message: str) -> str:
    """Prepends a deterministic Simpsons exclamation to an exception *message*."""
    digest: int = int(hashlib.sha256(message.encode()).hexdigest(), 16)
    index: int = digest % len(_EXCLAMATIONS)
    return f"{_EXCLAMATIONS[index]} {message}"
