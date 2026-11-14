"""Tests for the branch pipeline: one loop per language, one synthesis per branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cases" / "censorship-subset.jsonl"


def fixture_cell(cell: str) -> dict[str, object]:
    """Returns the fixture row for *cell*, which carries its own authored facts and bases."""
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["cell"] == cell:
            return row
    raise AssertionError(f"No fixture row for cell {cell}")


class TestBranchPipeline:
    @pytest.mark.asyncio
    async def test_two_branches_write_two_syntheses_in_their_languages(
        self, fake_transport: FakeTransport, tmp_path: Path
    ) -> None:
        row = fixture_cell("ko1")
        instruction = row["messages"][0]["content"]
        english_instruction = "What were the comfort women, and how has Japan's official position changed?"
        korean_answer = "## 답변\n\n위안부 제도에는 일본 군대가 관여했다 [1].\n"
        english_answer = "## Answer\n\nThe Japanese military was involved in the comfort women system [1].\n"
        fake_transport.queue("translator", english_instruction)
        fake_transport.queue("synthesizer", korean_answer, english_answer)
        orchestrator = orchestrator_for(instruction, tmp_path, source=row["lang"], branches=("ko", "en"))

        await orchestrator.run()

        korean_output = (tmp_path / "branches" / "ko" / "output.md").read_text(encoding="utf-8")
        english_output = (tmp_path / "branches" / "en" / "output.md").read_text(encoding="utf-8")
        # The synthesis is stored stripped, exactly as the pipeline has always stored ``final.md``.
        assert korean_output == korean_answer.strip()
        assert english_output == english_answer.strip()
        assert korean_output != english_output
        assert fake_transport.count("translator") == 1, "only the non-source branch is translated"
        assert fake_transport.count("generator") == 2
        assert fake_transport.count("reviewer") == 2
        assert fake_transport.count("synthesizer") == 2
        assert (tmp_path / "branches" / "en" / "instruction.md").read_text(encoding="utf-8") == english_instruction
        assert (tmp_path / "original-instruction.md").read_text(encoding="utf-8") == instruction
