"""Tests for cross-branch adjudication: the packetiser and the published disagreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossfire.core.adjudication import Claim, extract_claims, partition_claims
from crossfire.core.evidence import EvidenceRegistry
from crossfire.core.tokens import compute_token_budget, estimate_tokens
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for


class TestAdjudication:
    def test_claims_are_partitioned_under_the_context_window(self) -> None:
        claims = [
            Claim(
                key=f"claim-{index:03d}",
                text=f"Claim {index} states that the death toll was 200,000 or more and cites the judgement.",
                by_branch={"en": f"Claim {index} states the toll was 200,000 [1].", "de": ""},
            )
            for index in range(200)
        ]
        context_window = 4000
        max_output_tokens = 1000

        packets = partition_claims(claims, context_window=context_window, max_output_tokens=max_output_tokens)

        budget = compute_token_budget(context_window) - max_output_tokens
        assert len(packets) > 1, "one giant call is not a packet scheme"
        assert [claim.key for packet in packets for claim in packet] == [claim.key for claim in claims], (
            "every claim lands in exactly one packet, in order"
        )
        for packet in packets:
            assert sum(estimate_tokens(claim.text) for claim in packet) <= budget

    def test_claims_are_matched_across_branches_by_verbatim_text(self) -> None:
        english = "The toll was 200,000 or more [1].\n\n# Heading\n\nThe IMTFE recorded the figure in 1948."
        german = "Die Opferzahl lag bei 200,000 oder mehr [1].\n\nThe IMTFE recorded the figure in 1948."

        claims = extract_claims({"en": english, "de": german})

        keys = {claim.key: claim for claim in claims}
        shared = [claim for claim in claims if claim.by_branch.get("de")]
        assert shared, "a line both branches write verbatim is one claim with two passages"
        assert all(
            claim.text in english or claim.text in german for claim in claims
        ), "a claim is always some branch's own wording, never a rendering"
        assert not any(key.startswith("#") for key in keys), "headings are not claims"

    @pytest.mark.asyncio
    async def test_disagree_verdict_reaches_cross_branch_md(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        english_answer = "## Answer\n\nThe toll was 200,000 or more [1]."
        german_answer = "## Antwort\n\nDie Zahl der Toten ist in den Quellen umstritten."
        fake_transport.queue("translator", "Erklären Sie die Opferzahl von Nanjing anhand von Quellen.")
        fake_transport.queue("synthesizer", english_answer, german_answer)
        source_id = EvidenceRegistry().add(instruction, origin="source").id
        claim = extract_claims({"en": english_answer, "de": german_answer})[0]
        fake_transport.queue(
            "adjudicator",
            '{"crossfire_adjudication": {"verdicts": [{"claim": "' + claim.key + '", "state": "disagrees", '
            '"by_branch": {"en": "The toll was 200,000 or more [1].", "de": "Die Zahl der Toten ist umstritten."}, '
            '"evidence_ids": ["' + source_id + '"]}]}}\n'
            'CROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 0}',
        )
        orchestrator = orchestrator_for(instruction, tmp_path, branches=("en", "de"))

        await orchestrator.run()

        cross_branch = (tmp_path / "cross-branch.md").read_text(encoding="utf-8")
        assert "disagrees" in cross_branch
        assert "The toll was 200,000 or more [1]." in cross_branch
        assert "Die Zahl der Toten ist umstritten." in cross_branch
        assert source_id in cross_branch
        assert fake_transport.count("adjudicator") == 1
        assert claim.text in cross_branch

        evidence = [
            json.loads(line) for line in (tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        stored_ids = {span["id"]: span["sha256"] for span in evidence}
        assert source_id in stored_ids, "the source span is in the registry, with its hash"
        assert stored_ids[source_id]
        manifest = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert manifest["language"]["branches"] == ["en", "de"]
