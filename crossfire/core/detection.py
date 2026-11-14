"""Offline source-language detection: Unicode scripts first, then stopword scoring.

``source = "auto"`` must never call a model, so detection runs on character ranges and word lists only. Confidence is
the margin between the top two candidate scores: a thin margin means the text does not commit to one language, and the
caller must ask for an explicit ``--source-language`` rather than guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A script must cover at least this share of the alphabetic characters to name the language.
_SCRIPT_MINIMUM_SHARE = 0.3

# Below this margin between the top two candidates the detector refuses to commit.
_MARGIN_FLOOR = 0.15

# (language, inclusive code-point ranges). Kana and Hangul are checked before CJK ideographs, because Japanese and
# Korean text carries ideographs too, while Chinese text carries no kana or Hangul.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("ja", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("ru", ((0x0400, 0x04FF),)),
    ("ar", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("el", ((0x0370, 0x03FF),)),
    ("he", ((0x0590, 0x05FF),)),
    ("hi", ((0x0900, 0x097F),)),
)

_WORD_REGEX = re.compile(r"[^\W\d_]+", re.UNICODE)

def _words(text: str) -> frozenset[str]:
    """Splits a whitespace-separated word list into a set, keeping accented forms intact."""
    return frozenset(text.split())


# Function words that separate the Latin-script languages this bench cares about. Accents are kept, because they are
# part of the evidence: Portuguese "não" and Spanish "no" are different words.
_STOPWORDS: dict[str, frozenset[str]] = {
    "de": _words(
        "der die das den dem des und oder aber ist sind war nicht kein keine ein eine einer eines zu mit von "
        "für auf aus bei nach über unter sich auch es sie er wir ich dass wenn wird werden wurde hat haben"
    ),
    "fr": _words(
        "le la les un une des du et ou mais est sont ne pas que qui quoi dont où à au aux en dans pour par sur "
        "avec sans ce cette ces il elle ils elles nous vous je se son sa ses leur leurs plus"
    ),
    "es": _words(
        "el la los las un una unos unas y o pero es son no ni de del en para por con sin que quien como se su sus "
        "más este esta estos estas al lo"
    ),
    "pt": _words(
        "o a os as um uma uns umas e ou mas é são não nem de do da dos das em para por com sem que quem como se seu "
        "sua seus suas mais este esta estes estas ao à"
    ),
    "it": _words(
        "il lo la i gli le un una e o ma è sono non né di del della dei delle in per con senza che chi come si suo "
        "sua suoi sue più questo questa al alla"
    ),
    "nl": _words(
        "de het een en of maar is zijn niet geen van in op voor met zonder dat die dit deze zich ook als er aan bij "
        "uit over naar wordt worden was waren hij zij wij ik"
    ),
}

# A stopword share below this floor means the text carries no usable signal in any scored language.
_STOPWORD_MINIMUM_SCORE = 0.05


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of offline detection: the best language, the margin, and the ranked candidates."""

    language: str = ""
    confidence: float = 0.0
    candidates: tuple[tuple[str, float], ...] = field(default_factory=tuple)


def _script_hits(text: str) -> dict[str, int]:
    """Counts the characters of *text* that fall inside each script range, kana and Hangul first."""
    hits: dict[str, int] = {}
    for character in text:
        code_point = ord(character)
        for language, ranges in _SCRIPT_RANGES:
            if any(low <= code_point <= high for low, high in ranges):
                hits[language] = hits.get(language, 0) + 1
                break
    return hits


def _ranked(scores: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Returns candidate scores, highest first, then by language tag for a stable order."""
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def _margin(ranked: tuple[tuple[str, float], ...]) -> float:
    """Returns the difference between the top two scores, or the top score when there is only one."""
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return ranked[0][1]
    return ranked[0][1] - ranked[1][1]


def _stopword_scores(text: str) -> dict[str, float]:
    """Returns each Latin-script language's stopword share of the words in *text*."""
    words = [word.lower() for word in _WORD_REGEX.findall(text)]
    if not words:
        return {}
    total = float(len(words))
    return {
        language: sum(1 for word in words if word in stopwords) / total
        for language, stopwords in _STOPWORDS.items()
    }


def _score_stopwords(text: str) -> DetectionResult:
    """Ranks the Latin-script languages by stopword share and applies the minimum-score and margin rules."""
    scores = _stopword_scores(text)
    ranked = _ranked(scores)
    if not ranked or ranked[0][1] < _STOPWORD_MINIMUM_SCORE:
        return DetectionResult(candidates=ranked)
    confidence = _margin(ranked)
    if confidence < _MARGIN_FLOOR:
        return DetectionResult(confidence=confidence, candidates=ranked)
    return DetectionResult(language=ranked[0][0], confidence=confidence, candidates=ranked)


def detect_language(text: str) -> DetectionResult:
    """Detects the language of *text* offline.

    Script coverage decides first: kana means Japanese, Hangul means Korean, and ideographs without either mean
    Chinese. Text in one of the other non-Latin scripts resolves to that script's language. Latin text falls through to
    stopword scoring. The detector returns an empty language when the top two
    candidates are too close to separate, so the caller asks for an explicit source language instead of guessing.
    """
    alphabetic = [character for character in text if character.isalpha()]
    if not alphabetic:
        return _score_stopwords(text)

    hits = _script_hits(text)
    # Japanese and Korean carry ideographs too, and Chinese carries neither kana nor Hangul, so the presence of kana
    # or Hangul rules Chinese out rather than splitting the score with it.
    if hits.get("ja") or hits.get("ko"):
        hits.pop("zh", None)
    if hits:
        total = float(len(alphabetic))
        scores = {language: count / total for language, count in hits.items()}
        ranked = _ranked(scores)
        top_language, top_score = ranked[0]
        confidence = _margin(ranked)
        if top_score < _SCRIPT_MINIMUM_SHARE or confidence < _MARGIN_FLOOR:
            return DetectionResult(language="", confidence=confidence, candidates=ranked)
        return DetectionResult(language=top_language, confidence=confidence, candidates=ranked)

    return _score_stopwords(text)
