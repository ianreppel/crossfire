"""Tests for reviewer and generator independence."""

from __future__ import annotations

from crossfire.core.domain import Candidate, Mode, Review
from crossfire.core.prompts import (
    build_generator_prompt,
    build_reviewer_prompt,
    build_synthesizer_prompt,
)


class TestReviewerIsolation:
    """Reviewer prompts must contain only their assigned candidate."""

    def test_reviewer_sees_only_assigned_candidate(self):
        candidate_zero = Candidate(text="Candidate ZERO content", model="g0", round=1, index=0)

        _, prompt = build_reviewer_prompt(
            mode=Mode.RESEARCH,
            instruction="Test instruction",
            candidate=candidate_zero,
        )

        assert "Candidate ZERO content" in prompt
        assert "Candidate ONE content" not in prompt

    def test_reviewer_prompt_has_no_review_section(self):
        candidate = Candidate(text="Candidate text", model="g0", round=1, index=0)

        _, prompt = build_reviewer_prompt(
            mode=Mode.RESEARCH,
            instruction="Test instruction",
            candidate=candidate,
        )

        assert "REVIEW" not in prompt.upper().replace("[CANDIDATE", "").replace("candidate", "")

    def test_reviewer_prompt_contains_rules_and_instruction(self):
        candidate = Candidate(text="Candidate text", model="g0", round=1, index=0)

        _, prompt = build_reviewer_prompt(
            mode=Mode.RESEARCH,
            instruction="My specific instruction",
            candidate=candidate,
        )

        assert "My specific instruction" in prompt
        assert "RULES" in prompt


class TestGeneratorIsolation:
    """Generators must not see each other's outputs within a round."""

    def test_round1_generator_has_no_previous_candidate(self):
        _, prompt = build_generator_prompt(
            mode=Mode.CODE,
            instruction="Write code",
            context="Some context",
            round_num=1,
        )

        assert "PREVIOUS SYNTHESIS" not in prompt

    def test_round2_generator_sees_only_previous_synthesis(self):
        _, prompt = build_generator_prompt(
            mode=Mode.CODE,
            instruction="Write code",
            context="Some context",
            previous_synthesis="Previous round output",
            round_num=2,
        )

        assert "Previous round output" in prompt
        assert "PREVIOUS SYNTHESIS" in prompt

    def test_generator_prompt_does_not_contain_other_generators(self):
        _, prompt_a = build_generator_prompt(
            mode=Mode.CODE,
            instruction="Write code",
            round_num=1,
        )
        _, prompt_b = build_generator_prompt(
            mode=Mode.CODE,
            instruction="Write code",
            round_num=1,
        )

        assert prompt_a == prompt_b, "Same shite in, same shite out"

    def test_generator_includes_instruction_context_rules(self):
        _, prompt = build_generator_prompt(
            mode=Mode.CODE,
            instruction="My instruction",
            context="My context",
            round_num=1,
        )

        assert "My instruction" in prompt
        assert "My context" in prompt
        assert "RULES" in prompt


class TestSynthesizerSeesAll:
    """Synthesizer must see all candidates and reviews."""

    def test_synthesizer_sees_all_candidates_and_reviews(self):
        candidates = [
            Candidate(text="Cand 0", model="g0", round=1, index=0),
            Candidate(text="Cand 1", model="g1", round=1, index=1),
        ]
        reviews = [
            Review(text="Review for 0", model="r0", round=1, candidate_index=0),
            Review(text="Review for 1", model="r1", round=1, candidate_index=1),
        ]

        _, prompt = build_synthesizer_prompt(
            mode=Mode.RESEARCH,
            instruction="Instruction",
            candidates=candidates,
            reviews=reviews,
        )

        assert "Cand 0" in prompt
        assert "Cand 1" in prompt
        assert "Review for 0" in prompt
        assert "Review for 1" in prompt

    def test_synthesizer_includes_review_triage(self):
        candidates = [
            Candidate(text="Cand 0", model="g0", round=1, index=0),
            Candidate(text="Cand 1", model="g1", round=1, index=1),
        ]
        reviews = [
            Review(
                text="Decent.\nSTRENGTHS: clear structure, solid refs\nWEAKNESSES: weak ending",
                model="r0",
                round=1,
                candidate_index=0,
            ),
            Review(
                text="Pffft!\nSTRENGTHS: novel framing\nWEAKNESSES: no citations",
                model="r1",
                round=1,
                candidate_index=1,
            ),
        ]

        _, prompt = build_synthesizer_prompt(
            mode=Mode.RESEARCH,
            instruction="Instruction",
            candidates=candidates,
            reviews=reviews,
        )

        assert "REVIEW TRIAGE" in prompt
        assert "KEEP: clear structure, solid refs" in prompt
        assert "DISCARD: weak ending" in prompt
        assert "KEEP: novel framing" in prompt
        assert "DISCARD: no citations" in prompt

    def test_synthesizer_no_triage_without_structured_reviews(self):
        candidates = [
            Candidate(text="Cand 0", model="g0", round=1, index=0),
        ]
        reviews = [
            Review(
                text="This candidate is a bit bla and could be better.",
                model="r0",
                round=1,
                candidate_index=0,
            ),
        ]

        _, prompt = build_synthesizer_prompt(
            mode=Mode.RESEARCH,
            instruction="Instruction",
            candidates=candidates,
            reviews=reviews,
        )

        assert "REVIEW TRIAGE" not in prompt
