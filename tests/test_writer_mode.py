"""Tests for writer mode."""

from __future__ import annotations

import pytest

from crossfire.core.domain import (
    Candidate,
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    RunParameters,
    Task,
)
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.prompts import build_reviewer_prompt, parse_review_verdict


class TestWriterRubberDuckReviewing:
    """Tests that catch creative writing issues."""

    def test_nobody_expects_literary_inquisition(self):
        candidate = Candidate(
            text="Once upon a time, there was a writer who struggled with creativity.",
            model="test-gen",
            round=1,
            index=0,
        )
        system, _user = build_reviewer_prompt(
            mode=Mode.WRITE,
            instruction="Write a compelling short story",
            candidate=candidate,
        )

        assert "STORY:" in system
        assert "CRAFT:" in system
        assert "EMOTION:" in system
        assert "STRUCTURE:" in system
        assert "ORIGINALITY:" in system

        assert "plot holes" in system.lower()
        assert "forced motivations" in system.lower()
        assert "clichés" in system.lower()
        assert "dialogue" in system.lower()
        assert "show vs tell" in system.lower()
        assert "abandoned arcs" in system.lower()

        assert "sceptical" in system.lower()
        assert "convince" in system.lower()

    def test_deus_ex_llm_detector(self):
        problematic_writing = """The protagonist was very sad. She felt terrible.
"I can't believe this is happening," she said sadly.

Suddenly, everything changed. The magical solution appeared out of nowhere
and solved all her problems instantly. She lived happily ever after.

The end.
"""

        candidate = Candidate(
            text=problematic_writing,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.WRITE,
            instruction="Write a compelling character arc",
            candidate=candidate,
        )

        assert "show vs tell" in system.lower()
        assert "deus ex machina" in system.lower() or "earned" in system.lower()
        assert "dialogue" in system.lower()
        assert "sentiment" in system.lower()
        assert "purple prose" in system.lower()

    def test_roast_crap_writing(self):
        derivative_writing = """
        It was a dark and stormy night. The handsome stranger entered the tavern
        with mysterious eyes that held ancient secrets. The beautiful maiden
        gasped, for she had never seen such a man. Their eyes met across the
        crowded room, and destiny called to them both.
        """

        candidate = Candidate(
            text=derivative_writing,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.WRITE,
            instruction="Write original fantasy romance",
            candidate=candidate,
        )

        assert "insufficient" in system.lower()
        assert "previously" in system.lower()
        assert "elevate" in system.lower()
        assert "compelling" in system.lower()

        assert "derivative" in system.lower()
        assert "original" in system.lower()
        assert "clichés" in system.lower()

    @pytest.mark.asyncio
    async def test_whole_shebang_produces_something(self):
        challenging_instruction = (
            "Write a short story about loss that avoids sentimentality "
            "while still being emotionally powerful. Make it original and compelling."
        )

        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-creative",), context_window=32000),
            reviewers=ModelGroup(names=("rev-critic", "rev-editor"), context_window=32000),
            synthesizer=ModelGroup(names=("synth-writer",), context_window=32000),
            limits=LimitsConfiguration(),
        )

        parameters = RunParameters(
            mode=Mode.WRITE,
            task=Task(instruction=challenging_instruction),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
        )

        orchestrator = Orchestrator(configuration, parameters)
        result = await orchestrator.run()

        assert len(result.strip()) > 100
        assert "Synthesized" in result or "content" in result.lower()

    def test_smack_flat_dialogue_even_flatter(self):
        harsh_but_fair_review = """
        This piece struggles with fundamental craft issues that undermine its impact:

        1. Plot convenience - conflict resolves through coincidence not character action
        2. Flat dialogue - characters speak in exposition rather than authentic voice
        3. Emotional manipulation - trying to force tears through cheap tragedy
        4. Clichéd metaphors - comparing love to weather patterns and seasons
        5. Inconsistent voice - switching between lyrical and conversational mid-paragraph

        However, there are glimpses of genuine insight in the middle section.

        STRENGTHS: authentic details in workplace scenes, strong sensory descriptions
        WEAKNESSES: plot convenience, forced emotion, clichéd metaphors, inconsistent voice
        SEVERITY: material
        """

        verdict = parse_review_verdict(harsh_but_fair_review)

        assert verdict.severity == "material"
        assert len(verdict.weaknesses) >= 3  # Should identify multiple creative issues
        assert any("plot" in w.lower() or "convenience" in w.lower() for w in verdict.weaknesses)
        assert any("dialogue" in w.lower() or "emotion" in w.lower() for w in verdict.weaknesses)

    def test_ackshually_nitpicks(self):
        nitpick_review = """
        This is solid creative work with strong fundamentals. Minor suggestions:

        1. Could vary sentence length more in the opening paragraph
        2. The metaphor in paragraph 3 could be more specific
        3. One adverb that could be cut for stronger verb choice
        4. Consider a more active voice in the final sentence

        STRENGTHS: compelling character voice, authentic emotional core, satisfying structure
        WEAKNESSES: minor sentence rhythm variations, one weak metaphor, occasional passive voice
        SEVERITY: nitpick
        """

        verdict = parse_review_verdict(nitpick_review)

        assert verdict.severity == "nitpick"
        assert len(verdict.strengths) >= 3  # Should acknowledge strong fundamentals
        assert len(verdict.weaknesses) <= 5  # Should be limited minor issues
