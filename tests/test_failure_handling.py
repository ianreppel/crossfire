"""Tests for failure handling."""

from __future__ import annotations

import pytest

from crossfire.core.domain import (
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    Review,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.reviewers import assign_reviewers
from tests.helpers import LogCapture


class TestReviewerAssignment:
    def test_valid_assignment(self):
        result = assign_reviewers(
            reviewers=["r1", "r2", "r3", "r4"],
            num_candidates=2,
            num_reviewers_per_candidate=2,
            round_num=1,
            models_used_this_round=set(),
        )
        assert result is not None
        assert len(result) == 2
        assigned_models = [name for group in result.values() for name in group]
        assert len(assigned_models) == len(set(assigned_models)), "Thou shalt not reuse reviewers!"

    def test_insufficient_reviewers_returns_none(self):
        result = assign_reviewers(
            reviewers=["r1", "r2"],
            num_candidates=2,
            num_reviewers_per_candidate=2,
            round_num=1,
            models_used_this_round=set(),
        )
        assert result is None

    def test_filters_explicitly_excluded_models(self):
        result = assign_reviewers(
            reviewers=["r1", "r2", "r3", "r4"],
            num_candidates=1,
            num_reviewers_per_candidate=2,
            round_num=1,
            models_used_this_round={"r1", "r3"},
        )
        assert result is not None
        assigned_models = [name for group in result.values() for name in group]
        assert "r1" not in assigned_models
        assert "r3" not in assigned_models


class TestRoundFailure:
    @pytest.mark.asyncio
    async def test_consecutive_round_failures_abort_run(self, clean_logger):
        """Uses 3 reviewers for a 2*2=4 requirement to force runtime failures (bypasses validation)."""
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a", "gen-b"), context_window=16000),
            reviewers=ModelGroup(
                names=("rev-a", "rev-b", "rev-c"),
                context_window=16000,
            ),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test", context=""),
            num_generators=2,
            num_reviewers_per_candidate=2,
            num_rounds=3,
            dry_run=True,
        )

        capture = LogCapture()
        clean_logger.addHandler(capture)

        # Bypass upfront validation (which catches the pool-size mismatches)
        # to exercise the runtime consecutive-failure abort path.
        orchestrator = Orchestrator(configuration, parameters)
        previous: str = ""
        for round_num in range(1, parameters.num_rounds + 1):
            orchestrator._progress.on_round_start(round_num, parameters.num_rounds)
            result = await orchestrator._run_round(round_num, previous)
            if result is not None:
                orchestrator._consecutive_round_failures = 0
                previous = result.synthesis_text
            else:
                orchestrator._consecutive_round_failures += 1
                if orchestrator._consecutive_round_failures >= 2:
                    break

        assert orchestrator._consecutive_round_failures >= 2

        round_failures = [e for e in capture.records if e.get("event") == "round_failed"]
        assert len(round_failures) >= 2
        assert all(rf["reason"] == "insufficient_reviewers" for rf in round_failures)

    async def test_cross_group_overlap_succeeds(self, clean_logger):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a", "gen-b"), context_window=16000),
            reviewers=ModelGroup(names=("gen-a", "rev-b", "rev-c", "rev-d"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test", context=""),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=3,
            dry_run=True,
        )

        orchestrator = Orchestrator(configuration, parameters)
        result = await orchestrator.run()

        assert result.strip(), "If it's empty, something went horribly wrong!"


class TestConfigurationValidation:
    def test_insufficient_reviewer_pool_fails_validation(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a",), context_window=16000),
            reviewers=ModelGroup(names=("rev-a",), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        )
        errors = configuration.validate(num_generators=2, num_reviewers_per_candidate=2)
        assert any("reviewer" in e.lower() for e in errors)

    def test_valid_configuration_passes(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a",), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        )
        errors = configuration.validate(num_generators=1, num_reviewers_per_candidate=2)
        assert errors == []

    def test_no_generators_fails(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=(), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        )
        errors = configuration.validate(num_generators=1, num_reviewers_per_candidate=2)
        assert any("generator" in e.lower() for e in errors)

    def test_no_synthesizer_fails(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a",), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b"), context_window=16000),
            synthesizer=ModelGroup(names=(), context_window=32000),
        )
        errors = configuration.validate(num_generators=1, num_reviewers_per_candidate=2)
        assert any("synthesizer" in e.lower() for e in errors)


class TestShouldStopEarly:
    def test_empty_reviews_never_stop(self):
        assert Orchestrator._should_stop_early([]) is False

    def test_material_severity_keeps_going(self):
        review = Review(text="SEVERITY: material\nWEAKNESSES: -", model="rev-a", round=1, candidate_index=0)
        assert Orchestrator._should_stop_early([review]) is False

    def test_weaknesses_below_threshold_stops(self):
        review = Review(text="STRENGTHS: solid", model="rev-a", round=1, candidate_index=0)
        assert Orchestrator._should_stop_early([review], threshold=1) is True
