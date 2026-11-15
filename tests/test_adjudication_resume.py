"""Tests for packet coverage and hash-matched adjudication resume."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from crossfire.core.adjudication import extract_claims
from crossfire.core.archive import RunArchive
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.prompts import build_cross_branch_adjudicator_prompt
from crossfire.core.tokens import compute_token_budget, estimate_tokens
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for

ANSWER = (
    "The first source supports this claim.\n"
    "The second source supports another claim.\n"
    "The third finding needs evidence."
)
CLAIMS = extract_claims({"en": ANSWER, "de": ANSWER})


def packet_reply(keys: list[str], state: str = "disagrees") -> str:
    """Builds a reply in the existing adjudicator wire format."""
    payload = {"crossfire_adjudication": {"verdicts": [{"claim": key, "state": state} for key in keys]}}
    return json.dumps(payload) + f'\nCROSSFIRE-ADJUDICATION {{"claims": {len(keys)}, "unavailable": 0}}'


def packet_orchestrator(tmp_path: Path, *, resume: bool = False, model: str = "adj-a") -> Orchestrator:
    """Builds a run whose adjudication window gives each claim its own packet."""
    orchestrator = orchestrator_for("Explain the findings.", tmp_path, resume=resume)
    group = replace(orchestrator.configuration.adjudicator, names=(model,), context_window=1000, max_output_tokens=350)
    orchestrator.configuration = replace(orchestrator.configuration, adjudicator=group)
    sizes: list[int] = []
    for count in (1, 2):
        system, user = build_cross_branch_adjudicator_prompt(CLAIMS[:count], {"en": ANSWER, "de": ANSWER}, ())
        sizes.append(estimate_tokens(system) + estimate_tokens(user) + group.max_output_tokens)
    assert sizes[0] <= compute_token_budget(group.context_window) < sizes[1]
    return orchestrator


@pytest.mark.parametrize("reply_kind", ["omitted", "malformed", "explicit_unavailable"])
async def test_adjudication_accounts_for_every_claim(
    reply_kind: str, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("synthesizer", ANSWER, ANSWER)
    if reply_kind == "malformed":
        fake_transport.queue("adjudicator", "unparseable", "unparseable")
    elif reply_kind == "omitted":
        fake_transport.queue("adjudicator", packet_reply([CLAIMS[0].key]))
    else:
        fake_transport.queue("adjudicator", packet_reply([claim.key for claim in CLAIMS], "no_evidence"))

    await orchestrator_for("Explain the findings.", tmp_path).run()

    published = (tmp_path / "cross-branch.md").read_text(encoding="utf-8")
    assert all(claim.key in published for claim in CLAIMS)
    records = RunArchive(tmp_path).journal.replay()
    adjudicated = [record for record in records if record["event"] == "adjudicated"][-1]
    unavailable = [record for record in records if record["event"] == "adjudication_packet_unavailable"]
    if reply_kind == "explicit_unavailable":
        assert adjudicated["incomplete"] == []
        assert not unavailable
    else:
        assert adjudicated["incomplete"] == [1]
        expected_missing = CLAIMS if reply_kind == "malformed" else CLAIMS[1:]
        assert unavailable[0]["claims"] == [claim.key for claim in expected_missing]
        assert "adjudication_incomplete" in published


@pytest.mark.parametrize(("change", "expected_calls"), [("intact", 0), ("torn", 1), ("request", 0)])
async def test_adjudication_resume_checks_request_and_reply_hashes(
    change: str, expected_calls: int, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("synthesizer", ANSWER, ANSWER)
    fake_transport.queue("adjudicator", *(packet_reply([claim.key]) for claim in CLAIMS))
    await packet_orchestrator(tmp_path).run()
    assert fake_transport.count("adjudicator") == 3
    original = (tmp_path / "cross-branch.md").read_text(encoding="utf-8")
    if change == "torn":
        (tmp_path / "adjudication" / "packet-2.txt").write_text("torn reply", encoding="utf-8")
        fake_transport.queue("adjudicator", packet_reply([CLAIMS[1].key]))
    elif change == "request":
        fake_transport.queue("adjudicator", *(packet_reply([claim.key]) for claim in CLAIMS))

    if change == "request":
        with pytest.raises(ValueError, match="Resume refused"):
            await packet_orchestrator(tmp_path, resume=True, model="adj-b").run()
    else:
        await packet_orchestrator(tmp_path, resume=True).run()

    assert fake_transport.count("adjudicator") - 3 == expected_calls
    assert fake_transport.count("synthesizer") == 2
    assert (tmp_path / "cross-branch.md").read_text(encoding="utf-8") == original


async def test_adjudication_resume_retries_only_unavailable_packet(
    fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("synthesizer", ANSWER, ANSWER)
    fake_transport.queue(
        "adjudicator", packet_reply([CLAIMS[0].key]), "invalid reply", "invalid reply", packet_reply([CLAIMS[2].key])
    )
    await packet_orchestrator(tmp_path).run()
    assert fake_transport.count("adjudicator") == 4
    fake_transport.queue("adjudicator", packet_reply([CLAIMS[1].key]))

    await packet_orchestrator(tmp_path, resume=True).run()

    assert fake_transport.count("adjudicator") == 5
    published = (tmp_path / "cross-branch.md").read_text(encoding="utf-8")
    assert "adjudication_incomplete" not in published
    assert all(claim.key in published for claim in CLAIMS)
