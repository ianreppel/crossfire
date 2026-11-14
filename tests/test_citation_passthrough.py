"""Tests for citation and name pass-through, and for the translation fidelity gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossfire.core.domain import BranchState
from crossfire.core.translation import (
    assert_fidelity,
    extract_do_not_translate,
    fidelity_violations,
    restore_do_not_translate,
)
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for

JAPANESE_REFERENCE = "[3] 洞 富雄, 「南京アッサム日記」, 文藝春秋, 1998."


class TestCitationPassthrough:
    def test_japanese_title_survives_byte_identical(self) -> None:
        source = f"The death toll is disputed in the sources.\n{JAPANESE_REFERENCE}\n"

        masked, spans = extract_do_not_translate(source)

        assert "南京アッサム日記" not in masked, "the reference line must not reach the translator"
        assert len(spans) == 1
        translated = f"Die Opferzahl ist in den Quellen umstritten.\n{spans[0].placeholder}\n"
        restored = restore_do_not_translate(translated, spans)

        assert JAPANESE_REFERENCE in restored
        assert fidelity_violations(source, translated, spans) == []
        assert_fidelity(source, translated, spans)

    @pytest.mark.asyncio
    async def test_render_that_translates_a_title_fails_the_gate(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        instruction = f"Summarise the disputed toll.\n{JAPANESE_REFERENCE}"
        rendered = "[3] Nobuo Doi, Nanjing Assam Diary, Bungeishunju, 1998."
        fake_transport.queue("translator", f"Fassen Sie die umstrittene Opferzahl zusammen.\n{rendered}")
        orchestrator = orchestrator_for(instruction, tmp_path)

        branch = await orchestrator._translate_instruction(orchestrator._build_branches(orchestrator.configuration)[1])

        assert branch.state is BranchState.TRANSLATION_FAILED
        assert branch.instruction_text == ""
        translate_prompt = fake_transport.texts("translator")[0]
        assert JAPANESE_REFERENCE not in translate_prompt, "the reference line must never reach the translator"
