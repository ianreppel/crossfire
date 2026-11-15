"""Tests for the per-branch instruction translation step: the source short-circuit, attestation and failure state."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.domain import BranchState
from crossfire.core.hashing import sha256_text
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for


class TestInstructionTranslation:
    @pytest.mark.asyncio
    async def test_source_branch_spends_no_translation(self, fake_transport: FakeTransport, tmp_path: Path) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        orchestrator = orchestrator_for(instruction, tmp_path)

        branches = orchestrator._build_branches(orchestrator.configuration)
        source = await orchestrator._translate_instruction(branches[0])

        assert [branch.language for branch in branches] == ["en", "de"]
        assert source.instruction_text == instruction
        assert source.instruction_hash == sha256_text(instruction)
        assert source.instruction_model == ""
        assert source.state is BranchState.PENDING
        assert branches[1].instruction_text == "", "the German branch is not translated yet"
        assert fake_transport.count("translator") == 0

    @pytest.mark.asyncio
    async def test_translation_is_attested_with_model_and_hash(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        german = "Erklären Sie die Opferzahl von Nanjing anhand von Quellen."
        fake_transport.queue("translator", german)
        orchestrator = orchestrator_for(instruction, tmp_path)

        branch = await orchestrator._translate_instruction(orchestrator._build_branches(orchestrator.configuration)[1])

        assert branch.language == "de"
        assert branch.instruction_text == german
        assert branch.instruction_model == "trans-a"
        assert branch.instruction_hash == sha256_text(german)
        assert branch.state is BranchState.TRANSLATED
        assert fake_transport.count("translator") == 1
        assert (tmp_path / "branches" / "de" / "instruction.md").read_text(encoding="utf-8") == german
        translate_prompt = fake_transport.texts("translator")[0]
        assert translate_prompt.startswith("[TRANSLATE en->de]")
        assert instruction in translate_prompt

    @pytest.mark.parametrize("reply", ["", "I cannot fulfill this request."])
    @pytest.mark.asyncio
    async def test_translation_refusal_marks_branch_translation_failed(
        self, reply: str, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        fake_transport.queue("translator", reply)
        orchestrator = orchestrator_for("Explain the Nanjing death toll with sources.", tmp_path)

        branch = await orchestrator._translate_instruction(orchestrator._build_branches(orchestrator.configuration)[1])

        assert branch.state is BranchState.TRANSLATION_FAILED
        assert branch.instruction_text == ""
        assert branch.instruction_model == ""
        assert fake_transport.count("translator") == 1
        assert not (tmp_path / "branches" / "de" / "instruction.md").exists()
