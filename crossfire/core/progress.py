"""Progress reporting and no-op implementation for the orchestration pipeline."""

from __future__ import annotations

from typing import Protocol

from crossfire.core.domain import Phase


class ProgressCallback(Protocol):
    """Optional progress reporting hook consumed by the TUI."""

    def on_phase_start(
        self,
        round_num: int,
        phase: Phase,
        total_tasks: int,
        models: list[str] | None = None,
        candidate_indices: list[int | None] | None = None,
    ) -> None: ...
    def on_task_done(
        self,
        round_num: int,
        phase: Phase,
        model: str = "",
        candidate_index: int | None = None,
    ) -> None: ...
    def on_phase_end(self, round_num: int, phase: Phase) -> None: ...
    def on_round_start(self, round_num: int, total_rounds: int) -> None: ...
    def on_run_end(self) -> None: ...


class NoOpProgress:
    """No-op fallback when no TUI is attached."""

    def on_phase_start(
        self,
        round_num: int,
        phase: Phase,
        total_tasks: int,
        models: list[str] | None = None,
        candidate_indices: list[int | None] | None = None,
    ) -> None:
        pass

    def on_task_done(
        self,
        round_num: int,
        phase: Phase,
        model: str = "",
        candidate_index: int | None = None,
    ) -> None:
        pass

    def on_phase_end(self, round_num: int, phase: Phase) -> None:
        pass

    def on_round_start(self, round_num: int, total_rounds: int) -> None:
        pass

    def on_run_end(self) -> None:
        pass
