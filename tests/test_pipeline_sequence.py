"""Tests for sequencing and determinism of generation -> review -> synthesis."""

from __future__ import annotations

from dataclasses import replace

import pytest

from crossfire.core.domain import RunParameters
from crossfire.core.orchestrator import Orchestrator
from tests.helpers import LogCapture


@pytest.fixture()
def three_round_parameters(basic_parameters: RunParameters) -> RunParameters:
    """Extends basic_parameters with 3 rounds for sequencing tests."""
    return replace(basic_parameters, num_rounds=3)


@pytest.mark.asyncio
async def test_full_round_sequence(basic_configuration, three_round_parameters, clean_logger):
    """Phases execute in order: generation → review → synthesis for each round."""
    capture = LogCapture()
    clean_logger.addHandler(capture)

    orchestrator = Orchestrator(basic_configuration, three_round_parameters)
    result = await orchestrator.run()

    assert result, "Erm... you might need more than empty stuff here"

    phase_events = [
        (e["event"], e.get("round"), e.get("phase"))
        for e in capture.records
        if e.get("event") in ("phase_start", "phase_end")
    ]

    for round_num in range(1, three_round_parameters.num_rounds + 1):
        round_phases = [(ev, ph) for ev, r, ph in phase_events if r == round_num]
        starts = [ph for ev, ph in round_phases if ev == "phase_start"]
        assert starts == ["generation", "review", "synthesis"], f"Round {round_num} phases out of order: {starts}"


@pytest.mark.asyncio
async def test_rounds_are_sequential(basic_configuration, three_round_parameters, clean_logger):
    capture = LogCapture()
    clean_logger.addHandler(capture)

    orchestrator = Orchestrator(basic_configuration, three_round_parameters)
    await orchestrator.run()

    phase_events = [
        (e["event"], e.get("round"), e.get("phase"))
        for e in capture.records
        if e.get("event") in ("phase_start", "phase_end")
    ]

    last_end_round: int = 0
    for event, round_num, phase in phase_events:
        assert isinstance(round_num, int)
        if event == "phase_start" and phase == "generation":
            assert (
                round_num >= last_end_round
            ), f"Learn to count! Round {round_num} started before round {last_end_round} ended."
        if event == "phase_end" and phase == "synthesis":
            last_end_round = round_num


@pytest.mark.asyncio
async def test_dry_run_determinism(basic_configuration, three_round_parameters, clean_logger):
    orchestrator1 = Orchestrator(basic_configuration, three_round_parameters)
    result1 = await orchestrator1.run()

    clean_logger.handlers.clear()

    orchestrator2 = Orchestrator(basic_configuration, three_round_parameters)
    result2 = await orchestrator2.run()

    assert result1 == result2, "Deterministic, dude! Dry runs spit out predictable output, or at least they ought to."


@pytest.mark.asyncio
async def test_synthesis_decision_logged(basic_configuration, three_round_parameters, clean_logger):
    capture = LogCapture()
    clean_logger.addHandler(capture)

    orchestrator = Orchestrator(basic_configuration, three_round_parameters)
    await orchestrator.run()

    synth_decisions = [e for e in capture.records if e.get("event") == "synthesis_decision"]
    assert len(synth_decisions) == three_round_parameters.num_rounds
    for decision in synth_decisions:
        assert "attributions" in decision
        assert isinstance(decision["attributions"], list)
        for entry in decision["attributions"]:
            assert "index" in entry
            assert "kept" in entry
            assert "discarded" in entry
        assert "selected_candidates" in decision
        assert "discarded_candidates" in decision
        assert decision["phase"] == "synthesis"


@pytest.mark.asyncio
async def test_cost_summary_logged(basic_configuration, three_round_parameters, clean_logger):
    """A cost_summary event is emitted at the end of the run."""
    capture = LogCapture()
    clean_logger.addHandler(capture)

    orchestrator = Orchestrator(basic_configuration, three_round_parameters)
    await orchestrator.run()

    cost_events = [e for e in capture.records if e.get("event") == "cost_summary"]
    assert len(cost_events) == 1
