"""Tests for refusal to publish verdicts that cite invalid evidence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from crossfire.core.adjudication import extract_claims
from crossfire.core.archive import RunArchive
from crossfire.core.orchestrator import RunFailedError
from tests.fakes.transport import FakeTransport
from tests.helpers import LogCapture, orchestrator_for


@pytest.mark.parametrize("state", ["agrees", "disagrees", "no_evidence"])
async def test_invalid_evidence_refuses_publication_and_logs_id(
    state: str, fake_transport: FakeTransport, tmp_path: Path, clean_logger: logging.Logger
) -> None:
    answer = "The source records a supported finding."
    claim = extract_claims({"en": answer})[0]
    invalid_id = "ev-source-000000000000"
    response = {
        "crossfire_adjudication": {
            "verdicts": [
                {
                    "claim": claim.key,
                    "state": state,
                    "by_branch": {"en": answer, "de": answer},
                    "evidence_ids": [invalid_id],
                }
            ]
        }
    }
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("synthesizer", answer, answer)
    fake_transport.queue(
        "adjudicator", json.dumps(response) + '\nCROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 0}'
    )
    capture = LogCapture()
    clean_logger.addHandler(capture)
    orchestrator = orchestrator_for("Explain the finding.", tmp_path, render_language="en")

    with pytest.raises(RunFailedError, match=invalid_id):
        await orchestrator.run()

    assert not (tmp_path / "final.md").exists()
    assert not (tmp_path / "cross-branch.md").exists()
    assert not (tmp_path / "render" / "en.md").exists()
    for records in (capture.records, RunArchive(tmp_path).journal.replay()):
        invalid = [record for record in records if record["event"] == "evidence_invalid"]
        assert len(invalid) == 1
        assert invalid_id in invalid[0]["violations"][0]
