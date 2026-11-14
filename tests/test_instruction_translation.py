"""Tests for the per-branch instruction translation step: the source short-circuit, attestation and failure state."""

from __future__ import annotations

import pytest

from crossfire.core.domain import (
    BranchState,
    CrossfireConfiguration,
    LanguageConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.hashing import sha256_text
from crossfire.core.orchestrator import Orchestrator
from tests.fakes.transport import FakeTransport


def two_branch_configuration() -> CrossfireConfiguration:
    """A configuration whose source is English and whose branches are English and German."""
    return CrossfireConfiguration(
        generators=ModelGroup(names=("gen-a",), context_window=16000),
        reviewers=ModelGroup(names=("rev-a",), context_window=16000),
        synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
        search=SearchConfiguration(enabled=False),
        limits=LimitsConfiguration(max_concurrent_requests=2),
        language=LanguageConfiguration(source="en", branches=("en", "de")),
    )


def orchestrator_for(instruction: str) -> Orchestrator:
    """Builds an orchestrator over one round, ready to translate without a dry run."""
    parameters = RunParameters(mode=Mode.RESEARCH, task=Task(instruction=instruction), num_rounds=1)
    orchestrator = Orchestrator(two_branch_configuration(), parameters)
    orchestrator._api_key = "offline-test-key"
    return orchestrator


class TestInstructionTranslation:
    @pytest.mark.asyncio
    async def test_source_branch_spends_no_translation(self, fake_transport: FakeTransport) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        orchestrator = orchestrator_for(instruction)

        branches = orchestrator._build_branches(orchestrator.configuration)
        source = await orchestrator._translate_instruction(branches[0])

        assert [branch.language for branch in branches] == ["en", "de"]
        assert source.instruction_text == instruction
        assert source.instruction_hash == sha256_text(instruction)
        assert source.instruction_model == ""
        assert source.state is BranchState.PENDING
        assert branches[1].instruction_text == "", "the German branch is not translated yet"
        assert fake_transport.count("translator") == 0
