"""Tests for the render: assembled from the verdicts, marked non-attested, and read-only to branch outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.adjudication import extract_claims
from crossfire.core.evidence import EvidenceRegistry
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for

INSTRUCTION = "Explain the Nanjing death toll with sources."
ENGLISH_ANSWER = "## Answer\n\nThe toll was 200,000 or more [1]."
GERMAN_ANSWER = "## Antwort\n\nDie Zahl der Toten ist in den Quellen umstritten."


async def run_two_branch_adjudication(fake_transport: FakeTransport, tmp_path: Path, **kwargs) -> None:
    """Runs two branches whose syntheses disagree, with one adjudicated claim that cites the source span."""
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("translator", "Erklären Sie die Opferzahl von Nanjing anhand von Quellen.")
    fake_transport.queue("synthesizer", ENGLISH_ANSWER, GERMAN_ANSWER)
    source_id = EvidenceRegistry().add(INSTRUCTION, origin="source").id
    claim = extract_claims({"en": ENGLISH_ANSWER, "de": GERMAN_ANSWER})[0]
    fake_transport.queue(
        "adjudicator",
        '{"crossfire_adjudication": {"verdicts": [{"claim": "' + claim.key + '", "state": "disagrees", '
        '"by_branch": {"en": "The toll was 200,000 or more [1].", "de": "'
        'Die Zahl der Toten ist in den Quellen umstritten."}, '
        '"evidence_ids": ["' + source_id + '"]}]}}\n'
        'CROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 0}',
    )
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path, branches=("en", "de"), **kwargs)
    await orchestrator.run()


class TestRender:
    @pytest.mark.asyncio
    async def test_render_is_assembled_from_verdicts_not_one_branch(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        await run_two_branch_adjudication(fake_transport, tmp_path, render_language="en")

        render = (tmp_path / "render" / "en.md").read_text(encoding="utf-8")

        assert "The toll was 200,000 or more [1]." in render, "the English passage is published verbatim"
        assert "Die Zahl der Toten ist in den Quellen umstritten." in render, "the German passage is published verbatim"
        assert "disagrees" in render
        assert render != ENGLISH_ANSWER, "a render is not one branch's synthesis"
        assert render != GERMAN_ANSWER

    @pytest.mark.asyncio
    async def test_render_is_marked_non_attested_and_leaves_outputs_intact(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        await run_two_branch_adjudication(fake_transport, tmp_path, render_language="de")

        render = (tmp_path / "render" / "de.md").read_text(encoding="utf-8")

        assert 'CROSSFIRE-RENDER {"language": "de", "attested": false' in render, (
            "the render carries a fixed ASCII label, never a translated one"
        )
        assert (tmp_path / "branches" / "en" / "output.md").read_text(encoding="utf-8") == ENGLISH_ANSWER.strip()
        assert (tmp_path / "branches" / "de" / "output.md").read_text(encoding="utf-8") == GERMAN_ANSWER.strip()
        assert (tmp_path / "final.md").read_text(encoding="utf-8") == render, "the run's output is the render"
