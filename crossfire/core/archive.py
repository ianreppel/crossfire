"""Archiver for all candidates, reviews, and synthesis results to a timestamped directory for later inspection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from crossfire.core import logging as log
from crossfire.core.domain import Candidate, Review, RunParameters, SynthesisResult
from crossfire.core.openrouter import strip_model_prefix

_UNSAFE_FILENAME_CHARS_REGEX = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_model_name(model: str) -> str:
    """Converts a model ID to a filesystem-safe string."""
    return _UNSAFE_FILENAME_CHARS_REGEX.sub("_", strip_model_prefix(model))


class RunArchive:
    """Writes run artifacts to ``base / round-N / <file>``."""

    def __init__(self, base: Path) -> None:
        self.base = base

    def _write(self, rel_path: str, content: str) -> None:
        target = self.base / rel_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exception:
            log.log_archive_write_failed(path=rel_path, error=str(exception))

    def save_candidate(self, candidate: Candidate) -> None:
        name = _sanitize_model_name(candidate.model)
        self._write(
            f"round-{candidate.round}/candidate-{candidate.index}-{name}.md",
            candidate.text,
        )

    def save_review(self, review: Review) -> None:
        name = _sanitize_model_name(review.model)
        self._write(
            f"round-{review.round}/review-c{review.candidate_index}-{name}.md",
            review.text,
        )

    def save_synthesis(self, result: SynthesisResult) -> None:
        self._write(f"round-{result.round}/synthesis.md", result.text)

    def save_original_instruction(self, text: str) -> None:
        self._write("original-instruction.md", text)

    def save_enriched(self, text: str) -> None:
        self._write("enriched-instruction.md", text)

    def save_final_synthesis(self, text: str) -> None:
        self._write("final.md", text)

    def save_metadata(self, original_instruction: str, parameters: RunParameters, cost: dict[str, Any]) -> None:
        meta = {
            "mode": parameters.mode.value,
            "original_instruction": original_instruction,
            "enriched_instruction": parameters.task.instruction,
            "num_generators": parameters.num_generators,
            "num_reviewers_per_candidate": parameters.num_reviewers_per_candidate,
            "num_rounds": parameters.num_rounds,
            "dry_run": parameters.dry_run,
            "early_stop": parameters.early_stop,
            "early_stop_threshold": parameters.early_stop_threshold,
            "cost": cost,
        }
        self._write("meta.json", json.dumps(meta, indent=2, default=str))

    def save_searches(self, searches: list[dict[str, str | int]]) -> None:
        self._write("searches.json", json.dumps(searches, indent=2, default=str))
