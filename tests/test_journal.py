"""Tests for the archive journal: atomic writes, hashes, and a write failure that is never swallowed."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.archive import ArchiveWriteError, RunArchive
from crossfire.core.hashing import sha256_text


class TestJournal:
    def test_failed_archive_write_raises(self, tmp_path: Path) -> None:
        archive = RunArchive(tmp_path / "run")
        archive.save_final_synthesis("the first synthesis")

        journal = archive.journal.replay()
        assert journal[-1]["event"] == "artefact"
        assert journal[-1]["path"] == "final.md"
        assert journal[-1]["sha256"] == sha256_text("the first synthesis")

        target = tmp_path / "run" / "final.md"
        target.unlink()
        target.mkdir()

        with pytest.raises(ArchiveWriteError):
            archive.save_final_synthesis("a synthesis that cannot be written")

        assert not (tmp_path / "run" / "final.md.tmp").exists(), "a failed write leaves no temporary file behind"
        assert [record["path"] for record in archive.journal.replay() if record["event"] == "artefact"] == ["final.md"]
