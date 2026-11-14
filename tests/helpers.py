"""Shared test utilities."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from crossfire.core.archive import RunArchive
from crossfire.core.domain import (
    CrossfireConfiguration,
    LanguageConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.orchestrator import Orchestrator


class LogCapture(logging.Handler):
    """Captures structured log events for test assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {}
        if hasattr(record, "event"):
            entry["event"] = record.event  # type: ignore[attr-defined]
        if hasattr(record, "data"):
            entry.update(record.data)  # type: ignore[attr-defined]
        if entry:
            self.records.append(entry)


def two_branch_configuration(
    *,
    source: str = "en",
    branches: tuple[str, ...] = ("en", "de"),
) -> CrossfireConfiguration:
    """A configuration whose source is *source* and whose branches are *branches*."""
    return CrossfireConfiguration(
        generators=ModelGroup(names=("gen-a",), context_window=16000),
        reviewers=ModelGroup(names=("rev-a",), context_window=16000),
        synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        translator=ModelGroup(names=("trans-a",), context_window=16000),
        adjudicator=ModelGroup(names=("adj-a",), context_window=200000),
        search=SearchConfiguration(enabled=False),
        limits=LimitsConfiguration(max_concurrent_requests=2),
        language=LanguageConfiguration(source=source, branches=branches),
    )


def orchestrator_for(
    instruction: str,
    archive_base: Path,
    *,
    num_rounds: int = 1,
    source: str = "en",
    branches: tuple[str, ...] = ("en", "de"),
    resume: bool = False,
    render_language: str | None = None,
) -> Orchestrator:
    """Builds an orchestrator over *num_rounds* rounds, ready to call models through the fake transport."""
    parameters = RunParameters(
        mode=Mode.RESEARCH,
        task=Task(instruction=instruction),
        num_generators=1,
        num_reviewers_per_candidate=1,
        num_rounds=num_rounds,
        resume=resume,
        early_stop=False,
    )
    configuration = two_branch_configuration(source=source, branches=branches)
    if render_language is not None:
        language = replace(configuration.language, render_language=render_language)
        configuration = replace(configuration, language=language)
    orchestrator = Orchestrator(configuration, parameters, archive=RunArchive(archive_base))
    orchestrator._api_key = "offline-test-key"
    orchestrator._http_client = httpx.AsyncClient()
    return orchestrator
