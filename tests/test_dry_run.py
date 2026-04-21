"""Tests for dry runs."""

from __future__ import annotations

import pytest

from crossfire.core.domain import (
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    Phase,
    Role,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.simulation import simulate_response, simulate_search


class TestDryRunDeterminism:
    def test_same_inputs_same_output(self) -> None:
        output_a: str = simulate_response(
            instruction="Test",
            mode="research",
            phase=Phase.GENERATION,
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
        )
        output_b: str = simulate_response(
            instruction="Test",
            mode="research",
            phase=Phase.GENERATION,
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
        )
        assert output_a == output_b

    @pytest.mark.parametrize(
        ("overrides_a", "overrides_b", "description"),
        [
            ({"model": "gen-a"}, {"model": "gen-b"}, "different models"),
            ({"candidate_index": 0}, {"candidate_index": 1}, "different candidate slots"),
            ({"round_num": 1}, {"round_num": 2}, "different rounds"),
            (
                {"phase": Phase.GENERATION, "role": Role.GENERATOR},
                {"phase": Phase.REVIEW, "role": Role.REVIEWER},
                "different roles",
            ),
        ],
    )
    def test_varying_one_parameter_changes_output(
        self,
        overrides_a: dict,
        overrides_b: dict,
        description: str,
    ) -> None:
        base: dict[str, str | Phase | Role | int] = dict(
            instruction="Test", mode="research", phase=Phase.GENERATION, role=Role.GENERATOR, model="gen-a", round_num=1
        )
        output_a: str = simulate_response(**{**base, **overrides_a})
        output_b: str = simulate_response(**{**base, **overrides_b})
        assert output_a != output_b, f"Sorry, but that's not what we expected for {description}"


class TestDryRunSearch:
    def test_deterministic_search(self) -> None:
        result_a: str = simulate_search(
            instruction="Test",
            mode="research",
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
            query="quantum computing",
        )
        result_b: str = simulate_search(
            instruction="Test",
            mode="research",
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
            query="quantum computing",
        )
        assert result_a == result_b

    def test_different_queries_different_results(self) -> None:
        result_a: str = simulate_search(
            instruction="Test",
            mode="research",
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
            query="quantum computing",
        )
        result_b: str = simulate_search(
            instruction="Test",
            mode="research",
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
            query="classical computing",
        )
        assert result_a != result_b

    def test_results_contain_structure(self) -> None:
        result: str = simulate_search(
            instruction="Test",
            mode="research",
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
            query="test query",
        )
        assert "Result" in result
        assert "https://example.com" in result


class TestDryRunStructure:
    def test_generator_output_has_sections(self):
        output = simulate_response(
            instruction="Test",
            mode="code",
            phase=Phase.GENERATION,
            role=Role.GENERATOR,
            model="gen-a",
            round_num=1,
        )
        assert "## Generated Output" in output
        assert "### Section" in output

    def test_reviewer_output_has_score(self):
        output = simulate_response(
            instruction="Test",
            mode="code",
            phase=Phase.REVIEW,
            role=Role.REVIEWER,
            model="rev-a",
            round_num=1,
        )
        assert "## Review" in output
        assert "Score" in output

    def test_synthesizer_output_has_decision(self):
        output = simulate_response(
            instruction="Test",
            mode="code",
            phase=Phase.SYNTHESIS,
            role=Role.SYNTHESIZER,
            model="synth-a",
            round_num=1,
        )
        assert "crossfire_synthesis" in output
        assert "## Synthesized Output" in output


@pytest.mark.asyncio
async def test_full_dry_run_end_to_end(clean_logger):
    """Full pipeline in dry-run returns non-empty, deterministic output."""
    configuration = CrossfireConfiguration(
        generators=ModelGroup(names=("gen-a",), context_window=16000),
        reviewers=ModelGroup(names=("rev-a", "rev-b"), context_window=16000),
        synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        search=SearchConfiguration(enabled=False),
        limits=LimitsConfiguration(),
    )
    parameters = RunParameters(
        mode=Mode.RESEARCH,
        task=Task(instruction="Test instruction"),
        num_generators=1,
        num_reviewers_per_candidate=2,
        num_rounds=2,
        dry_run=True,
    )

    orchestrator = Orchestrator(configuration, parameters)
    result = await orchestrator.run()
    assert result
    assert "Synthesized Output" in result
