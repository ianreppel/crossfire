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
    SynthesisResult,
    Task,
)
from crossfire.core.orchestrator import _REFUSAL_REGEX, Orchestrator, RefusalError
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


class TestSynthesisRegression:
    def test_refusal_phrase_is_regression(self):
        previous = "A detailed analysis with citations [1][2][3]." * 20
        current = "I must be transparent about a limitation."
        assert Orchestrator._is_synthesis_regression(previous, current) is True

    def test_length_drop_is_regression(self):
        previous = "Substantive content. " * 200
        current = "Brief meta-commentary."
        assert Orchestrator._is_synthesis_regression(previous, current) is True

    def test_similar_length_not_regression(self):
        previous = "Substantive content. " * 50
        current = "Different but equally substantive. " * 50
        assert Orchestrator._is_synthesis_regression(previous, current) is False

    def test_empty_previous_never_regression(self):
        current = "Short output."
        assert Orchestrator._is_synthesis_regression("", current) is False

    def test_modest_trim_not_regression(self):
        previous = "Content here. " * 100
        current = "Content here. " * 60
        assert Orchestrator._is_synthesis_regression(previous, current) is False


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "I must be transparent about a significant limitation: the search results are insufficient.",
            "I cannot fulfill the detailed quantitative comparative analysis without better sources.",
            "I'm unable to fulfill this request given the available information.",
            "The data is insufficient for the analysis to support this kind of comparison.",
            "Would you like me to search for more specialized sources?",
            "I appreciate the detailed instructions, but the sources lack depth.",
        ],
    )
    def test_refusal_phrases_detected(self, text: str):
        assert _REFUSAL_REGEX.search(text[:500]) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Surface codes dominate superconducting QEC implementations due to compatibility.",
            "The error threshold for surface codes is approximately 1%.",
            "Transparent reporting of results is essential in quantum computing research.",
        ],
    )
    def test_legitimate_content_not_flagged(self, text: str):
        assert _REFUSAL_REGEX.search(text[:500]) is None

    @pytest.mark.asyncio
    async def test_refusal_triggers_replacement_generator(self, clean_logger):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a", "gen-b"), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b", "rev-c"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test"),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=1,
            dry_run=True,
        )

        capture = LogCapture()
        clean_logger.addHandler(capture)

        orchestrator = Orchestrator(configuration, parameters)
        original_generate = orchestrator._generate_candidate
        call_count: int = 0

        async def _mock_generate(round_num, index, model, previous_synthesis):
            nonlocal call_count
            call_count += 1
            if model == "gen-a":
                raise RefusalError("gen-a refused")
            return await original_generate(round_num, index, model, previous_synthesis)

        orchestrator._generate_candidate = _mock_generate  # type: ignore[assignment]
        result = await orchestrator.run()

        assert result.strip(), "Replacement generator should have produced output"
        assert call_count == 2
        dropped = [e for e in capture.records if e.get("event") == "model_dropped" and e.get("reason") == "refusal"]
        assert len(dropped) == 1

    @pytest.mark.asyncio
    async def test_synthesis_regression_carries_forward_previous(self, clean_logger):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a",), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b", "rev-c"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test"),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
            early_stop=False,
        )

        capture = LogCapture()
        clean_logger.addHandler(capture)

        orchestrator = Orchestrator(configuration, parameters)
        original_synthesis = orchestrator._run_synthesis

        async def _mock_synthesis(round_num, candidates, reviews):
            if round_num == 2:
                return SynthesisResult(
                    text="I must be transparent about a limitation: the sources are insufficient.",
                    model="synth-a",
                    round=round_num,
                )
            return await original_synthesis(round_num, candidates, reviews)

        orchestrator._run_synthesis = _mock_synthesis  # type: ignore[assignment]
        result = await orchestrator.run()

        assert "I must be transparent" not in result
        assert result.strip(), "Should have carried forward round 1 synthesis"
        regressions = [e for e in capture.records if e.get("event") == "synthesis_regression"]
        assert len(regressions) == 1
        assert regressions[0]["round"] == 2

    @pytest.mark.asyncio
    async def test_length_regression_carries_forward_previous(self, clean_logger):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a",), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b", "rev-c"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test"),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
            early_stop=False,
        )

        capture = LogCapture()
        clean_logger.addHandler(capture)

        orchestrator = Orchestrator(configuration, parameters)
        original_synthesis = orchestrator._run_synthesis

        async def _mock_synthesis(round_num, candidates, reviews):
            if round_num == 2:
                return SynthesisResult(
                    text="Critical analysis: sources are inadequate for comparison.",
                    model="synth-a",
                    round=round_num,
                )
            return await original_synthesis(round_num, candidates, reviews)

        orchestrator._run_synthesis = _mock_synthesis  # type: ignore[assignment]
        result = await orchestrator.run()

        assert "sources are inadequate" not in result
        assert result.strip(), "Should have carried forward round 1 synthesis"
        regressions = [e for e in capture.records if e.get("event") == "synthesis_regression"]
        assert len(regressions) == 1
