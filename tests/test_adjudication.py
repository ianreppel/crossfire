"""Tests for cross-branch adjudication: the packetiser and the published disagreement."""

from __future__ import annotations

from crossfire.core.adjudication import Claim, extract_claims, partition_claims
from crossfire.core.tokens import compute_token_budget, estimate_tokens


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
