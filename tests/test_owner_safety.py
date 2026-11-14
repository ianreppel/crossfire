"""Tests for exclusive run ownership and stale pid adoption."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from crossfire.core import archive as archive_module
from crossfire.core.archive import ArchiveOwnerError, RunArchive
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for


@pytest.fixture(autouse=True)
def fixed_owner_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps ownership tests independent of the clock."""
    monkeypatch.setattr(archive_module, "datetime", Mock(now=Mock(return_value=datetime(2026, 11, 14, tzinfo=UTC))))
    monkeypatch.setattr(archive_module.os, "getpid", lambda: 12345)


@pytest.mark.parametrize("resume", [False, True])
async def test_live_owner_is_refused_without_removing_its_lock(
    resume: bool, fake_transport: FakeTransport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_path = tmp_path / "owner.json"
    owner_text = '{"pid": 54321, "started_at": "2026-11-13T00:00:00+00:00"}'
    owner_path.write_text(owner_text, encoding="utf-8")
    monkeypatch.setattr(archive_module, "_process_is_alive", lambda pid: True)
    fake_transport.default("generator", "reviewer", "synthesizer")
    orchestrator = orchestrator_for("Explain the findings.", tmp_path, branches=("en",), resume=resume)

    with pytest.raises(ArchiveOwnerError, match=r"54321.*still alive"):
        await orchestrator.run()

    assert owner_path.read_text(encoding="utf-8") == owner_text
    assert not (tmp_path / "journal.jsonl").exists()
    assert fake_transport.requests == []


def test_resume_adopts_only_a_stale_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owner_path = tmp_path / "owner.json"
    owner_path.write_text('{"pid": 54321, "started_at": "old"}', encoding="utf-8")
    monkeypatch.setattr(archive_module, "_process_is_alive", lambda pid: False)
    archive = RunArchive(tmp_path)

    with pytest.raises(ArchiveOwnerError, match="stale owner"):
        archive.claim_owner()
    owner = archive.claim_owner(adopt_stale=True)

    assert owner.pid == 12345
    assert json.loads(owner_path.read_text(encoding="utf-8"))["pid"] == 12345
    assert archive.journal.replay()[-1] == {
        "event": "owner_adopted",
        "previous_pid": 54321,
        "pid": 12345,
        "previous_alive": False,
    }
    archive.release_owner()
    assert not owner_path.exists()


def test_another_archive_cannot_claim_the_same_live_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_module, "_process_is_alive", lambda pid: True)
    archive = RunArchive(tmp_path)
    owner = archive.claim_owner()
    assert archive.claim_owner() == owner

    with pytest.raises(ArchiveOwnerError):
        RunArchive(tmp_path).claim_owner(adopt_stale=True)

    archive.release_owner()


async def test_resume_checks_owner_before_replaying_live_journal(
    fake_transport: FakeTransport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "owner.json").write_text('{"pid": 54321, "started_at": "old"}', encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    journal.write_text('{"event":', encoding="utf-8")
    monkeypatch.setattr(archive_module, "_process_is_alive", lambda pid: True)

    with pytest.raises(ArchiveOwnerError):
        await orchestrator_for("Explain the findings.", tmp_path, branches=("en",), resume=True).run()

    assert journal.read_text(encoding="utf-8") == '{"event":'
    assert fake_transport.requests == []


def test_inherited_archive_cannot_reclaim_or_release_parent_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive_module, "_process_is_alive", lambda pid: True)
    archive = RunArchive(tmp_path)
    owner = archive.claim_owner()
    owner_path = tmp_path / "owner.json"
    original = owner_path.read_text(encoding="utf-8")
    monkeypatch.setattr(archive_module.os, "getpid", lambda: 22222)

    with pytest.raises(ArchiveOwnerError):
        archive.claim_owner(adopt_stale=True)
    archive.release_owner()

    assert owner_path.read_text(encoding="utf-8") == original
    monkeypatch.setattr(archive_module.os, "getpid", lambda: owner.pid)
    archive.release_owner()
    assert not owner_path.exists()
