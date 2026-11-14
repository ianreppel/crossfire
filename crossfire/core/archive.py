"""Archiver for all candidates, reviews, and synthesis results to a timestamped directory for later inspection.

Every artefact is written to ``path.tmp`` and then moved into place, so a reader never sees half a file, and its sha256
is appended to an fsync'd journal. A write that fails raises: an artefact the run depends on is never logged away.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from crossfire.core import logging as log
from crossfire.core.domain import Candidate, Review, RunParameters, SynthesisResult, strip_provider_pin
from crossfire.core.hashing import sha256_text

_UNSAFE_FILENAME_CHARS_REGEX = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_model_name(model: str) -> str:
    """Converts a model ID to a filesystem-safe string."""
    return _UNSAFE_FILENAME_CHARS_REGEX.sub("_", strip_provider_pin(model))


class ArchiveWriteError(RuntimeError):
    """Raised when an artefact or a journal line cannot be written."""


class RunJournal:
    """Append-only, fsync'd record of what a run did, one JSON object per line.

    The journal is how a killed run is resumed: each artefact line carries the hash of what was written, so ``--resume``
    can tell a completed unit from a half-written one without trusting any file's modification time.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Appends one event and returns the record that was written."""
        record: dict[str, Any] = {"event": event, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exception:
            raise ArchiveWriteError(f"Could not append to {self.path}: {exception}") from exception
        return record

    def replay(self) -> list[dict[str, Any]]:
        """Returns every recorded event, in the order it was appended."""
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


class RunArchive:
    """Writes run artifacts to ``base / round-N / <file>``, with an append-only journal at ``base``."""

    def __init__(self, base: Path, journal: RunJournal | None = None, prefix: str = "") -> None:
        self.base = base
        self.journal = journal or RunJournal(base / "journal.jsonl")
        self.prefix = prefix

    def child(self, *parts: str) -> RunArchive:
        """Returns an archive in a subdirectory that appends to this archive's journal.

        One run, one journal: a branch writes into its own directory but records into the same append-only file, so
        ``--resume`` reads a single record of everything that happened.
        """
        segments = [segment for segment in (self.prefix, *parts) if segment]
        return RunArchive(self.base.joinpath(*parts), journal=self.journal, prefix="/".join(segments))

    def record_path(self, rel_path: str) -> str:
        """Returns *rel_path* relative to the run root, which is what the journal records."""
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path

    def _write(self, rel_path: str, content: str) -> str:
        """Writes *content* atomically, journals its hash, and returns the hash.

        A failed write raises :class:`ArchiveWriteError` after the failure has been logged, so a caller can never
        mistake a missing artefact for a saved one.
        """
        target = self.base / rel_path
        temporary = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, target)
        except OSError as exception:
            log.log_archive_write_failed(path=rel_path, error=str(exception))
            temporary.unlink(missing_ok=True)
            raise ArchiveWriteError(f"Could not write {rel_path}: {exception}") from exception

        digest = sha256_text(content)
        self.journal.append(
            "artefact",
            path=self.record_path(rel_path),
            sha256=digest,
            bytes=len(content.encode("utf-8")),
        )
        return digest

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

    def save_branch_output(self, text: str) -> None:
        self._write("output.md", text)

    def save_branch_instruction(self, language: str, text: str) -> str:
        return self._write(f"branches/{language}/instruction.md", text)

    def save_enriched(self, text: str) -> None:
        self._write("enriched-instruction.md", text)

    def save_cross_branch(self, text: str) -> None:
        self._write("cross-branch.md", text)

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
