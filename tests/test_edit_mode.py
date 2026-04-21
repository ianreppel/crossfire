"""Tests for edit mode."""

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
from crossfire.core.prompts import _BANNED_PHRASES, build_reviewer_prompt, parse_review_verdict


class TestEditPrecisionReviewing:
    """Tests that eliminate waffle and imprecision.

    If you happen to be an executive, you might want to stop reading right now. Otherwise, grab your comfort blanket
    stuffed with dollars now.
    """

    def test_bullshit_radar_is_on(self):
        candidate = Candidate(
            text="It is quite clear that we should optimize our approach going forward.",
            model="test-gen",
            round=1,
            index=0,
        )
        system, _user = build_reviewer_prompt(
            mode=Mode.EDIT,
            instruction="Edit this text for clarity and conciseness",
            candidate=candidate,
        )

        assert "PRECISION:" in system
        assert "CONCISENESS:" in system
        assert "JARGON & LLM WAFFLE:" in system
        assert "STRUCTURE:" in system

        assert "vague qualifiers" in system.lower()
        assert "hedging" in system.lower()
        assert "weak verbs" in system.lower()
        assert "redundancy" in system.lower()
        assert "corporate speak" in system.lower()
        assert "llm" in system.lower()

    def test_edit_reviewer_blows_up_on_business_bullshit(self):
        jargon_heavy_text = """
        It's worth noting that we should leverage our core competencies to optimize
        our robust, comprehensive approach. Furthermore, it's important to streamline
        our holistic framework going forward. Notably, we can utilize best practices
        to facilitate synergistic solutions that deliver value-added outcomes.
        """

        candidate = Candidate(
            text=jargon_heavy_text,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.EDIT,
            instruction="Edit this business communication",
            candidate=candidate,
        )

        assert "leverage" in system.lower()
        assert "utilize" in system.lower()
        assert "facilitate" in system.lower()
        assert "corporate speak" in system.lower()
        assert "it's worth noting" in system.lower()
        assert "importantly" in system.lower()
        assert "buzzword" in system.lower()
        assert "throat-clearing" in system.lower()

    def test_edit_reviewer_gobbles_up_waffle(self):
        waffling_text = """
        On the whole, it seems that one might argue that this approach could
        potentially be somewhat effective, relatively speaking. It appears that
        the results are fairly positive, though one could say they are rather
        limited in scope. Basically, it's essentially a quite good solution,
        all things considered.
        """

        candidate = Candidate(
            text=waffling_text,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.EDIT,
            instruction="Edit for directness and confidence",
            candidate=candidate,
        )

        assert "hedging" in system.lower()
        assert "vague qualifiers" in system.lower()
        assert "it seems that" in system.lower()
        assert "somewhat" in system.lower()
        assert "rather" in system.lower()
        assert "quite" in system.lower()
        assert "basically" in system.lower()
        assert "literally" in system.lower()

    def test_wordy_mcwordface(self):
        wordy_text = """
        In order to completely eliminate the past history of previous problems,
        we must first begin to start the process of making improvements to enhance
        our current existing systems. Due to the fact that our future plans require
        advance planning ahead of time, we should continue to keep moving forward.
        """

        candidate = Candidate(
            text=wordy_text,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.EDIT,
            instruction="Edit for conciseness",
            candidate=candidate,
        )

        assert "redundancy" in system.lower()
        assert "wordy constructions" in system.lower()
        assert "in order to" in system.lower()
        assert "unnecessary intensifiers" in system.lower()
        assert "completely eliminate" in system.lower()
        assert "future plans" in system.lower()
        assert "bloated transitions" in system.lower()

    @pytest.mark.asyncio
    async def test_whole_shebang_for_gobbledygook(self):
        verbose_instruction = (
            "Edit this text to make it as clear and concise as possible "
            "while preserving all meaning and removing any unnecessary words or jargon."
        )

        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-editor",), context_window=32000),
            reviewers=ModelGroup(names=("rev-precision", "rev-concision"), context_window=32000),
            synthesizer=ModelGroup(names=("synth-editor",), context_window=32000),
            limits=LimitsConfiguration(),
        )

        parameters = RunParameters(
            mode=Mode.EDIT,
            task=Task(instruction=verbose_instruction),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
        )

        orchestrator = Orchestrator(configuration, parameters)
        result = await orchestrator.run()

        assert len(result.strip()) > 50
        assert "Synthesized" in result or "content" in result.lower()

    def test_hairs_in_jargon_soup(self):
        precision_review = """
        This text suffers from multiple clarity and concision problems:

        1. Business jargon throughout - "leverage", "optimize", "streamline" appear 8 times
        2. Hedging language weakens every claim - "seems", "might", "could potentially"
        3. Redundant phrases waste words - "completely eliminate", "future plans"
        4. Vague referents make sentences unclear - "this", "it" without clear antecedents
        5. Wordy constructions triple sentence length unnecessarily

        STRENGTHS: logical structure, factual accuracy
        WEAKNESSES: business jargon, hedging language, redundancy, vague referents, wordiness
        SEVERITY: material
        """

        verdict = parse_review_verdict(precision_review)

        assert verdict.severity == "material"
        assert len(verdict.weaknesses) >= 4  # Should identify multiple precision issues
        assert any("jargon" in w.lower() or "hedging" in w.lower() for w in verdict.weaknesses)
        assert any("redundancy" in w.lower() or "wordiness" in w.lower() for w in verdict.weaknesses)

    def test_nits_are_not_worth_picking(self):
        minor_style_review = """
        This is clear, concise writing with good flow. Minor polish opportunities:

        1. One sentence could vary its structure for better rhythm
        2. A single transition could be smoother
        3. One word choice could be more precise

        STRENGTHS: clear meaning, good pace, direct language, no jargon
        WEAKNESSES: minor rhythm variation, one transition, word choice precision
        SEVERITY: nitpick
        """

        verdict = parse_review_verdict(minor_style_review)

        assert verdict.severity == "nitpick"
        assert len(verdict.strengths) >= 3  # Should acknowledge good fundamentals
        assert len(verdict.weaknesses) <= 4  # Should be limited minor issues


class TestBannedPhrases:
    def test_the_cursed_sentence(self):
        cursed = (
            "Let's unpack that deep dive into the holistic synergy of our "
            "paradigm shift. It is worth noting that, in today's world, "
            "when it comes to cutting-edge solutions, it's crucial to "
            "leverage our comprehensive overview to move the needle and "
            "empower groundbreaking transformation. Needless to say, "
            "at the end of the day, the ask is to think outside the box."
        )
        hits = [
            phrase
            for phrase in [
                "unpack",
                "deep dive",
                "holistic",
                "synergy",
                "paradigm shift",
                "it is worth noting",
                "in today's world",
                "when it comes to",
                "cutting-edge",
                "it's crucial to",
                "leverage",
                "comprehensive overview",
                "move the needle",
                "empower",
                "groundbreaking",
                "needless to say",
                "at the end of the day",
                "the ask",
                "think outside the box",
            ]
            if phrase in cursed.lower()
        ]

        assert (
            len(hits) >= 15
        ), f"You sound like a CEO! The cursed sentence should trigger most bans, got {len(hits)} hits"
        for phrase in hits:
            assert phrase in _BANNED_PHRASES.lower(), f"'{phrase}' missing from banned list"
