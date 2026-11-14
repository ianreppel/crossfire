"""Tests for branch isolation: each branch owns its task, search cache and failure counters."""

from __future__ import annotations

from pathlib import Path

from tests.helpers import orchestrator_for


class TestBranchIsolation:
    def test_second_branch_starts_with_empty_search_cache(self, tmp_path: Path) -> None:
        instruction = "Explain the Nanjing death toll with sources."
        orchestrator = orchestrator_for(instruction, tmp_path)
        branches = orchestrator._build_branches(orchestrator.configuration)

        english = orchestrator._prepare_branch_context(branches[0])
        english.search_cache[0] = "cached search results"
        english.synthesis_history.append("an earlier synthesis")
        english.consecutive_round_failures = 2

        german = orchestrator._prepare_branch_context(branches[1])

        assert german is not english
        assert german.search_cache == {}, "a branch must not inherit another branch's searches"
        assert german.synthesis_history == []
        assert german.consecutive_round_failures == 0
        assert german.branch.language == "de"
        assert german.task.language == "de"
        # The German branch is not translated yet, so its context falls back to the original rather than to nothing.
        assert branches[1].instruction_text == ""
        assert german.task.instruction == instruction
        assert orchestrator._active_context.search_cache == {}
        assert orchestrator.parameters.task.instruction == instruction
