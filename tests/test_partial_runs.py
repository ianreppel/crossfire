"""Tests for preserving usable output after consecutive branch failures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from crossfire.core.archive import RunArchive
from crossfire.core.orchestrator import RunFailedError
from tests.fakes.transport import FakeTransport
from tests.helpers import LogCapture, orchestrator_for

DRAFT = "A draft with a supported finding."
REFUSAL = "I cannot fulfill this request."
ANSWER = "The supported finding remains usable."


@pytest.mark.parametrize(
    ("branches", "replies", "expected_calls"),
    [
        (("en",), [DRAFT, REFUSAL, REFUSAL], 3),
        (("en", "de"), [DRAFT, REFUSAL, REFUSAL] * 2, 6),
        (("en", "de"), [DRAFT] * 4 + [REFUSAL] * 2, 6),
        (("en", "de"), [REFUSAL] * 2 + [DRAFT] * 4, 6),
    ],
    ids=["single", "both_partial", "later_failed", "earlier_failed"],
)
async def test_consecutive_failures_preserve_output_as_partial(
    branches: tuple[str, ...],
    replies: list[str],
    expected_calls: int,
    fake_transport: FakeTransport,
    tmp_path: Path,
    clean_logger: logging.Logger,
) -> None:
    fake_transport.default("reviewer")
    fake_transport.queue(
        "adjudicator",
        json.dumps({"crossfire_adjudication": {"verdicts": [{"claim": f"text:{ANSWER}", "state": "no_evidence"}]}})
        + '\nCROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 1}',
    )
    fake_transport.queue("generator", *replies)
    fake_transport.queue("synthesizer", *([ANSWER] * 4))
    capture = LogCapture()
    clean_logger.addHandler(capture)

    result = await orchestrator_for("Explain the finding.", tmp_path, branches=branches, num_rounds=4).run()

    assert result == ANSWER
    assert (tmp_path / "final.md").read_text(encoding="utf-8") == ANSWER
    assert fake_transport.count("generator") == expected_calls
    metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "PARTIAL"
    for records in (capture.records, RunArchive(tmp_path).journal.replay()):
        shape = [record for record in records if record["event"] == "run_shape"][-1]
        assert shape["shape"] == "partial"
        assert shape["status"] == "PARTIAL"


@pytest.mark.parametrize("branches", [("en",), ("en", "de")])
async def test_consecutive_failures_without_output_still_raise(
    branches: tuple[str, ...], fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.queue("generator", *([REFUSAL] * (2 * len(branches))))
    with pytest.raises(RunFailedError):
        await orchestrator_for("Explain the finding.", tmp_path, branches=branches, num_rounds=4).run()

    assert not (tmp_path / "final.md").exists()
    assert not (tmp_path / "meta.json").exists()
    assert fake_transport.count("synthesizer") == 0


async def test_resumed_output_survives_consecutive_failures(fake_transport: FakeTransport, tmp_path: Path) -> None:
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("synthesizer", ANSWER)
    await orchestrator_for("Explain the finding.", tmp_path, branches=("en",)).run()
    fake_transport.queue("generator", REFUSAL, REFUSAL)

    result = await orchestrator_for("Explain the finding.", tmp_path, branches=("en",), num_rounds=4, resume=True).run()

    assert result == ANSWER
    assert fake_transport.count("synthesizer") == 1
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["status"] == "PARTIAL"


async def test_complete_single_branch_is_not_partial(fake_transport: FakeTransport, tmp_path: Path) -> None:
    fake_transport.default("generator", "reviewer", "synthesizer")
    await orchestrator_for("Explain the finding.", tmp_path, branches=("en",)).run()

    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"
    shape = [record for record in RunArchive(tmp_path).journal.replay() if record["event"] == "run_shape"][-1]
    assert shape["shape"] == "single-branch"
    assert shape["status"] == "COMPLETE"
