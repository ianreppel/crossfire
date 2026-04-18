"""Shared test fixtures."""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

from crossfire.core.domain import (
    Candidate,
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    Review,
    RunParameters,
    SearchConfiguration,
    Task,
)


@pytest.fixture()
def clean_logger() -> Generator[logging.Logger, None, None]:
    """Clears before (setup/clear) and after (teardown)."""
    logger: logging.Logger = logging.getLogger("crossfire")
    logger.handlers.clear()
    yield logger
    logger.handlers.clear()


@pytest.fixture()
def basic_task() -> Task:
    return Task(instruction="Write a summary of AI in quantum computing.", context="Background info.")


@pytest.fixture()
def basic_configuration() -> CrossfireConfiguration:
    return CrossfireConfiguration(
        generators=ModelGroup(names=("gen-a", "gen-b"), context_window=16000),
        reviewers=ModelGroup(
            names=("rev-a", "rev-b", "rev-c", "rev-d"),
            context_window=16000,
        ),
        synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        search=SearchConfiguration(enabled=False),
        limits=LimitsConfiguration(max_concurrent_requests=10, temperature_default=0.2),
    )


@pytest.fixture()
def basic_parameters(basic_task: Task) -> RunParameters:
    return RunParameters(
        mode=Mode.RESEARCH,
        task=basic_task,
        num_generators=2,
        num_reviewers_per_candidate=2,
        num_rounds=2,
        dry_run=True,
    )


@pytest.fixture()
def sample_candidate() -> Candidate:
    return Candidate(
        text="This is candidate text with [citation-1].",
        model="gen-a",
        round=1,
        index=0,
    )


@pytest.fixture()
def sample_review() -> Review:
    return Review(
        text="The candidate is mostly supported.",
        model="rev-a",
        round=1,
        candidate_index=0,
    )
