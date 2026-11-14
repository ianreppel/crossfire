"""Tests for the branch pipeline: one loop per language, one synthesis per branch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossfire.core.archive import RunArchive
from crossfire.core.domain import (
    CrossfireConfiguration,
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

# The output and layout this run produced before the dual-language work started, captured from a894f86. The journal is
# additive: the plan's rollback section allows new archive files, so the guard is the synthesis bytes plus the layout.
GOLDEN_SHA256 = "77d5b8f3c360206843269cc4d015bf1f34d4bbbbd7d70397306b2c072fd31b59"
GOLDEN_FILES = {
    "final.md",
    "journal.jsonl",
    "meta.json",
    "original-instruction.md",
    "round-1/candidate-0-gen-a.md",
    "round-1/candidate-1-gen-b.md",
    "round-1/review-c0-rev-a.md",
    "round-1/review-c0-rev-b.md",
    "round-1/review-c1-rev-c.md",
    "round-1/review-c1-rev-d.md",
    "round-1/synthesis.md",
    "round-2/candidate-0-gen-a.md",
    "round-2/candidate-1-gen-b.md",
    "round-2/review-c0-rev-b.md",
    "round-2/review-c0-rev-c.md",
    "round-2/review-c1-rev-a.md",
    "round-2/review-c1-rev-d.md",
    "round-2/synthesis.md",
}


class TestSingleBranchCompatibility:
    @pytest.mark.asyncio
    async def test_single_branch_output_matches_todays_bytes(self, tmp_path: Path) -> None:
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-a", "gen-b"), context_window=16000),
            reviewers=ModelGroup(names=("rev-a", "rev-b", "rev-c", "rev-d"), context_window=16000),
            synthesizer=ModelGroup(names=("synth-a",), context_window=32000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(max_concurrent_requests=10, temperature_generators=0.2),
        )
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Write a summary of AI in quantum computing.", context="Background info."),
            num_generators=2,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
        )

        result = await Orchestrator(configuration, parameters, archive=RunArchive(tmp_path)).run()

        assert sha256_text(result) == GOLDEN_SHA256
        files = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()}
        assert files == GOLDEN_FILES, "a one-branch run keeps today's archive layout"
