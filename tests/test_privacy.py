"""Tests for the gateway data-policy guard."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from crossfire.core.config import load_configuration
from crossfire.core.domain import Candidate, CrossfireConfiguration, Mode, ModelGroup, Review, RunParameters, Task
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.privacy import (
    gateway_retention_note,
    retention_notices,
    training_models,
    training_violations,
)
from tests.helpers import LogCapture

_ZEN_RETENTION_NOTE = (
    "opencode forwards OpenAI and Anthropic API requests under 30-day retention, "
    "and documents no per-model attribution for it"
)


def _configuration(provider: str, **groups: ModelGroup) -> CrossfireConfiguration:
    base = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
    defaults = ModelGroup(names=("z-ai/glm-5.3",), context_window=200000)
    return replace(
        base,
        provider=provider,
        generators=groups.get("generators", defaults),
        reviewers=groups.get("reviewers", defaults),
        synthesizer=groups.get("synthesizer", defaults),
    )


def _with_provider_replacements(
    configuration: CrossfireConfiguration, **replacements: object
) -> CrossfireConfiguration:
    return replace(
        configuration,
        providers=tuple(
            replace(provider, **replacements[provider.name])  # type: ignore[arg-type]
            if provider.name in replacements
            else provider
            for provider in configuration.providers
        ),
    )


class TestTrainingModels:
    def test_lists_the_documented_offenders_per_gateway(self):
        assert "big-pickle" in training_models("opencode")
        assert "muse-spark-1.3-contributor" in training_models("opencode-go")
        assert training_models("synthetic") == frozenset()

    def test_reports_only_the_offending_models(self):
        violations = training_violations("opencode-go", ["muse-spark-1.3-contributor", "kimi-k3"])
        assert len(violations) == 1
        assert "muse-spark-1.3-contributor" in violations[0]
        assert "allow_training_models = true" in violations[0]

    def test_zero_retention_models_are_not_offenders(self):
        assert training_violations("opencode", ["space-bunny-free", "claude-opus-5-5"]) == []

    def test_openrouter_relies_on_request_routing_not_model_names(self):
        assert training_violations("openrouter", ["big-pickle"]) == []


class TestRetentionNotices:
    def test_reports_only_the_retaining_models(self):
        notices = retention_notices("opencode-go", ["grok-4.7", "kimi-k3", "gpt-6-luna"])
        assert len(notices) == 2
        assert any("grok-4.7" in notice and "30 days" in notice for notice in notices)

    def test_silent_for_zero_retention_gateways(self):
        assert retention_notices("synthetic", ["syn:large:text"]) == []

    def test_gateway_note_covers_zen_api_traffic(self):
        assert gateway_retention_note("opencode") == _ZEN_RETENTION_NOTE
        assert gateway_retention_note("synthetic") == ""


class TestValidation:
    def test_training_model_in_the_bench_fails_the_run(self):
        configuration = _configuration(
            "opencode-go",
            generators=ModelGroup(names=("muse-spark-1.3-contributor",), context_window=200000),
        )
        errors = configuration.validate(1, 0)
        assert any("muse-spark-1.3-contributor" in error for error in errors)

    def test_allow_training_models_opts_back_in(self):
        configuration = _configuration(
            "opencode-go",
            generators=ModelGroup(names=("muse-spark-1.3-contributor",), context_window=200000),
        )
        permitted = _with_provider_replacements(configuration, **{"opencode-go": {"allow_training_models": True}})
        assert not any("trains on prompts" in error for error in permitted.validate(1, 0))

    def test_clean_bench_raises_no_privacy_errors(self):
        assert not any("trains on prompts" in error for error in _configuration("opencode-go").validate(1, 0))


class TestDataPolicyNotices:
    def test_lists_retaining_models_resolved_through_the_alias_table(self):
        configuration = _configuration("opencode-go")
        mapped = _with_provider_replacements(
            configuration,
            **{"opencode-go": {"model_ids": (("z-ai/glm-5.3", "grok-4.7"),)}},
        )
        assert mapped.data_policy_notices() == ["opencode-go retains prompts sent to 'grok-4.7' for 30 days"]

    def test_zen_runs_report_the_api_retention_caveat(self):
        assert _configuration("opencode").data_policy_notices() == [_ZEN_RETENTION_NOTE]

    def test_synthetic_runs_report_nothing(self):
        assert _configuration("synthetic").data_policy_notices() == []


@pytest.mark.asyncio
async def test_run_logs_the_retention_notice(clean_logger: logging.Logger):
    capture = LogCapture()
    clean_logger.addHandler(capture)
    configuration = _configuration("opencode-go")
    configuration = _with_provider_replacements(
        configuration, **{"opencode-go": {"model_ids": (("z-ai/glm-5.3", "grok-4.7"),)}}
    )
    parameters = RunParameters(
        mode=Mode.CHECK,
        task=Task(instruction="Test"),
        num_generators=1,
        num_reviewers_per_candidate=1,
        num_rounds=1,
        dry_run=True,
    )
    orchestrator = Orchestrator(configuration, parameters)

    async def fake_generation(round_num: int, previous: str) -> list[Candidate]:
        return [Candidate(text="candidate", model="z-ai/glm-5.3", round=round_num, index=0)]

    async def fake_review(round_num: int, candidates: list[Candidate]) -> list[Review] | None:
        return [
            Review(text="review", model="z-ai/glm-5.3", round=round_num, candidate_index=candidate.index)
            for candidate in candidates
        ]

    orchestrator._run_generation = fake_generation  # type: ignore[assignment]
    orchestrator._run_review = fake_review  # type: ignore[assignment]

    await orchestrator.run()

    notices = [event for event in capture.records if event.get("event") == "data_policy_notice"]
    assert [event["notice"] for event in notices] == ["opencode-go retains prompts sent to 'grok-4.7' for 30 days"]
