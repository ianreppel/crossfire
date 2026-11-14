"""Tests for the archive journal: atomic writes, hashes, and a write failure that is never swallowed."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.archive import ArchiveWriteError, RunArchive
from crossfire.core.hashing import sha256_text
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for


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


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_skips_hash_matched_rounds(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        first = orchestrator_for(instruction, tmp_path, num_rounds=2, branches=("en", "de"))
        await first.run()
        settled = (
            fake_transport.count("generator"),
            fake_transport.count("translator"),
            fake_transport.count("synthesizer"),
        )

        resumed = orchestrator_for(instruction, tmp_path, num_rounds=3, branches=("en", "de"), resume=True)
        await resumed.run()

        generators = fake_transport.count("generator") - settled[0]
        translators = fake_transport.count("translator") - settled[1]
        syntheses = fake_transport.count("synthesizer") - settled[2]
        assert generators == 2, "only round 3 runs, once per branch"
        assert syntheses == 2
        assert translators == 0, "the attested translation is reused, not paid for twice"
        final = (tmp_path / "final.md").read_text(encoding="utf-8")
        assert final.strip()

    @pytest.mark.asyncio
    async def test_resume_reruns_a_round_whose_artefact_was_torn(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        """A kill between two units leaves a torn artefact; the hash rule redoes exactly that round."""
        instruction = "Explain the Nanjing death toll with sources."
        first = orchestrator_for(instruction, tmp_path, num_rounds=3, branches=("en", "de"))
        await first.run()
        settled = fake_transport.count("generator")

        torn = tmp_path / "branches" / "en" / "round-2" / "synthesis.md"
        torn.write_text("half a synthesis", encoding="utf-8")

        resumed = orchestrator_for(instruction, tmp_path, num_rounds=3, branches=("en", "de"), resume=True)
        await resumed.run()

        assert fake_transport.count("generator") - settled == 1, "only the torn round is re-run, for one branch"
        assert torn.read_text(encoding="utf-8") != "half a synthesis"
