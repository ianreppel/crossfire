"""Archiver for all candidates, reviews, and synthesis results to a timestamped directory for later inspection.

Every artefact is written to ``path.tmp`` and then moved into place, so a reader never sees half a file, and its sha256
is appended to an fsync'd journal. A write that fails raises: an artefact the run depends on is never logged away.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from crossfire.core import logging as log
from crossfire.core.domain import (
    Candidate,
    EvidenceSpan,
    Review,
    RunParameters,
    SynthesisResult,
    strip_provider_pin,
)
from crossfire.core.hashing import sha256_text

_UNSAFE_FILENAME_CHARS_REGEX = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_model_name(model: str) -> str:
    """Converts a model ID to a filesystem-safe string."""
    return _UNSAFE_FILENAME_CHARS_REGEX.sub("_", strip_provider_pin(model))


class ArchiveWriteError(RuntimeError):
    """Raised when an artefact or a journal line cannot be written."""


class ArchiveOwnerError(RuntimeError):
    """Raised when another live process already owns the run directory."""


@dataclass(frozen=True)
class RunOwner:
    """The process that owns a run directory, so two writers never interleave one run's journal."""

    pid: int
    started_at: str


def _process_is_alive(pid: int) -> bool:
    """Returns True when a process with *pid* exists and this user may signal it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

    def _parse_records(self, contents: bytes) -> list[dict[str, Any]]:
        """Parses complete records and identifies the line of any corruption."""
        records: list[dict[str, Any]] = []
        for number, line in enumerate(contents.split(b"\n"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("a journal record must be an object")
            except (ValueError, UnicodeDecodeError) as exception:
                raise ValueError(f"Corrupt journal {self.path} at line {number}: {exception}") from exception
            records.append(record)
        return records

    def recover_tail(self) -> None:
        """Recovers an unterminated final record while the caller holds the owner lock."""
        if not self.path.is_file():
            return
        contents = self.path.read_bytes()
        boundary = contents.rfind(b"\n") + 1
        self._parse_records(contents[:boundary])
        tail = contents[boundary:]
        if not tail:
            return
        try:
            self._parse_records(tail)
        except ValueError:
            with self.path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            # A complete JSON object may have been written before the terminating newline was lost.
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def replay(self) -> list[dict[str, Any]]:
        """Returns every recorded event, rejecting corruption with its line number."""
        if not self.path.is_file():
            return []
        return self._parse_records(self.path.read_bytes())


class RunArchive:
    """Writes run artifacts to ``base / round-N / <file>``, with an append-only journal at ``base``."""

    def __init__(self, base: Path, journal: RunJournal | None = None, prefix: str = "") -> None:
        self.base = base
        self.journal = journal or RunJournal(base / "journal.jsonl")
        self.prefix = prefix
        self._owner: RunOwner | None = None

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

    @contextmanager
    def _locked_owner_file(self) -> Iterator[TextIO]:
        """Serializes ownership changes on the current pid file, retrying if a release removed it."""
        owner_path = self.base / "owner.json"
        self.base.mkdir(parents=True, exist_ok=True)
        while True:
            with open(owner_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    current = owner_path.stat()
                except FileNotFoundError:
                    continue
                opened = os.fstat(handle.fileno())
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    continue
                handle.seek(0)
                yield handle
                return

    def claim_owner(self, *, adopt_stale: bool = False) -> RunOwner:
        """Claims the run directory for this process.

        A second writer is refused while the recorded owner is alive, because two processes appending one journal would
        interleave.
        Resume adopts only a stale pid, and ownership changes are serialized by a file lock.
        """
        owner = RunOwner(pid=os.getpid(), started_at=datetime.now(UTC).isoformat(timespec="seconds"))
        with self._locked_owner_file() as handle:
            content = handle.read()
            if content:
                recorded = json.loads(content)
                recorded_pid = int(recorded.get("pid", 0) or 0)
                if (
                    self._owner is not None
                    and self._owner.pid == owner.pid
                    and recorded
                    == {
                        "pid": self._owner.pid,
                        "started_at": self._owner.started_at,
                    }
                ):
                    self.journal.recover_tail()
                    return self._owner
                alive = _process_is_alive(recorded_pid)
                if alive:
                    raise ArchiveOwnerError(
                        f"Run directory {self.base} is owned by process {recorded_pid}, which is still alive. "
                        "Wait for it to exit, or pick another --run-dir."
                    )
                if not adopt_stale:
                    raise ArchiveOwnerError(f"Run directory {self.base} has a stale owner. Use --resume to adopt it.")
                self.journal.recover_tail()
                self.journal.append(
                    "owner_adopted",
                    previous_pid=recorded_pid,
                    pid=owner.pid,
                    previous_alive=alive,
                )
            self.journal.recover_tail()
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": owner.pid, "started_at": owner.started_at}))
            handle.flush()
            os.fsync(handle.fileno())
            self._owner = owner
        return owner

    def release_owner(self) -> None:
        """Releases the run directory, so a later run may claim it without ``--resume``."""
        if self._owner is None or self._owner.pid != os.getpid():
            return
        with self._locked_owner_file() as handle:
            content = handle.read()
            if content and json.loads(content) == {"pid": self._owner.pid, "started_at": self._owner.started_at}:
                (self.base / "owner.json").unlink(missing_ok=True)
        self._owner = None

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

    def save_render(self, language: str, text: str) -> None:
        self._write(f"render/{language}.md", text)

    def save_evidence(self, spans: Sequence[EvidenceSpan]) -> None:
        lines = [
            json.dumps(
                {
                    "id": span.id,
                    "sha256": span.sha256,
                    "origin": span.origin,
                    "branch": span.branch,
                    "text": span.text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for span in spans
        ]
        self._write("evidence.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    def save_cross_branch(self, text: str) -> None:
        self._write("cross-branch.md", text)

    def save_adjudication_packet(self, index: int, text: str, request_sha256: str) -> None:
        """Saves a covered packet and journals the request and reply hashes for resume."""
        path = f"adjudication/packet-{index}.txt"
        digest = self._write(path, text)
        self.journal.append(
            "adjudication_packet_completed",
            packet=index,
            path=self.record_path(path),
            sha256=digest,
            request_sha256=request_sha256,
        )

    def save_final_synthesis(self, text: str) -> None:
        self._write("final.md", text)

    def save_metadata(
        self,
        original_instruction: str,
        parameters: RunParameters,
        cost: dict[str, Any],
        language: dict[str, Any] | None = None,
        status: str = "COMPLETE",
        adjudication: dict[str, Any] | None = None,
    ) -> None:
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
            "status": status,
        }
        if language is not None:
            meta["language"] = language
        if adjudication is not None:
            meta["adjudication"] = adjudication
        self._write("meta.json", json.dumps(meta, indent=2, default=str))

    def save_searches(self, searches: list[dict[str, str | int]]) -> None:
        self._write("searches.json", json.dumps(searches, indent=2, default=str))
