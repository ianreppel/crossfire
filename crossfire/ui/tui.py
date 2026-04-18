"""Live progress display for the orchestration pipeline.

The :class:`TUI` class implements the ``ProgressCallback`` expected by
:class:`~crossfire.core.orchestrator.Orchestrator` and renders
a spinner with progress bar for each phase via Rich's ``Live`` display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from crossfire.core.domain import Phase, RunParameters

_PHASE_LABELS: dict[Phase, tuple[str, str]] = {
    Phase.ENRICHMENT: ("Enriching", "blue"),
    Phase.GENERATION: ("Generating", "cyan"),
    Phase.REVIEW: ("Reviewing", "magenta"),
    Phase.SYNTHESIS: ("Synthesizing", "yellow"),
}

_MAX_PHASE_LABEL_LENGTH: int = max(len(label) for label, _ in _PHASE_LABELS.values()) + 1

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _shorten_model(model: str) -> str:
    """Strips the ``openrouter:vendor/`` prefix, keeping only the model slug."""
    return model.split("/")[-1]


def _format_elapsed(seconds: float) -> str:
    """Formats *seconds* as ``H:MM:SS``."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@dataclass
class _TaskRow:
    """Tracks one in-flight or completed LLM call for rendering."""

    model: str
    phase: Phase
    candidate_index: int | None = None
    start_time: float = field(default_factory=time.monotonic)
    done: bool = False
    elapsed: float = 0.0


class TUILogHandler(logging.Handler):
    """Captures structured log events for the summary table."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        event: dict[str, Any] = {}
        # Custom attributes set by logging._emit via makeRecord (not on LogRecord's type stub)
        if hasattr(record, "event"):
            event["event"] = record.event  # type: ignore[attr-defined]
        if hasattr(record, "data"):
            event.update(record.data)  # type: ignore[attr-defined]
        if event:
            self.events.append(event)


class TUI:
    """Implements the ``ProgressCallback`` protocol expected by
    :class:`crossfire.core.orchestrator.Orchestrator`.
    """

    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self._handler = TUILogHandler()

        self._total_rounds: int = 0
        self._rounds_completed: int = 0
        self._run_start: float = 0.0
        self._current_round: int = 0

        self._rounds: dict[int, list[_TaskRow]] = {}
        self._active_phase: Phase | None = None
        self._active_phase_total: int = 0
        self._active_phase_done: int = 0

        self._spinner_index: int = 0
        self._live: Live | None = None

    def start(self, parameters: RunParameters) -> None:
        """Prints the header panel and begins the live display."""
        cf_logger = logging.getLogger("crossfire")
        cf_logger.addHandler(self._handler)

        self.console.print(
            Panel(
                f"[bold]Crossfire[/bold] — {parameters.mode.value} mode\n"
                f"Rounds: {parameters.num_rounds} | "
                f"Generators: {parameters.num_generators} | "
                f"Reviewers/candidate: {parameters.num_reviewers_per_candidate}"
                + (" | [yellow]DRY RUN[/yellow]" if parameters.dry_run else ""),
                title="Crossfire",
                border_style="blue",
            )
        )
        self._run_start = time.monotonic()
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
            get_renderable=self._build_display,
        )
        self._live.start()

    # -- ProgressCallback protocol --

    def on_round_start(self, round_num: int, total_rounds: int) -> None:
        self._total_rounds = total_rounds
        self._current_round = round_num
        self._rounds.setdefault(round_num, [])
        self._refresh()

    def on_phase_start(
        self,
        round_num: int,
        phase: Phase,
        total_tasks: int,
        models: list[str] | None = None,
        candidate_indices: list[int | None] | None = None,
    ) -> None:
        self._active_phase = phase
        self._active_phase_total = total_tasks
        self._active_phase_done = 0

        if models:
            indices = candidate_indices or [None] * len(models)
            rows = self._rounds.setdefault(round_num, [])
            for model, candidate_index in zip(models, indices, strict=True):
                rows.append(
                    _TaskRow(
                        model=model,
                        phase=phase,
                        candidate_index=candidate_index,
                    )
                )
        self._refresh()

    def on_task_done(
        self,
        round_num: int,
        phase: Phase,
        model: str = "",
        candidate_index: int | None = None,
    ) -> None:
        self._active_phase_done += 1
        now = time.monotonic()
        for row in reversed(self._rounds.get(round_num, [])):
            if row.done:
                continue
            if row.phase != phase:
                continue
            if model and row.model != model:
                continue
            if candidate_index is not None and row.candidate_index != candidate_index:
                continue
            row.done = True
            row.elapsed = now - row.start_time
            break
        self._refresh()

    def on_phase_end(self, round_num: int, phase: Phase) -> None:
        now = time.monotonic()
        for row in self._rounds.get(round_num, []):
            if row.phase == phase and not row.done:
                row.done = True
                row.elapsed = now - row.start_time
        self._refresh()

    def on_run_end(self) -> None:
        self._rounds_completed = self._current_round
        self._refresh()
        if self._live:
            self._live.stop()
            self._live = None

    def _remove_handler(self) -> None:
        logging.getLogger("crossfire").removeHandler(self._handler)

    def finish(self, cost_summary: dict[str, Any]) -> None:
        """Stops the live display and prints the summary table."""
        self._remove_handler()
        events = self._handler.events

        table = Table(title="Run Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        rounds_completed = max(
            (event.get("round", 0) for event in events if event.get("event") == "phase_end"),
            default=0,
        )
        compressions = sum(1 for event in events if event.get("event") == "compression_applied")
        drops = sum(1 for event in events if event.get("event") == "model_dropped")
        failures = sum(1 for event in events if event.get("event") == "round_failed")

        table.add_row("Rounds completed", str(rounds_completed))
        table.add_row("Compressions applied", str(compressions))
        table.add_row("Models dropped", str(drops))
        table.add_row("Round failures", str(failures))
        table.add_row("Total input tokens", str(cost_summary.get("total_input_tokens", 0)))
        table.add_row("Total output tokens", str(cost_summary.get("total_output_tokens", 0)))
        table.add_row("Total cost", f"${cost_summary.get('total_cost', 0):.4f}")

        self.console.print(table)

    def report_error(self) -> None:
        """Stops the live display and prints a failure banner."""
        self._remove_handler()
        if self._live:
            self._live.stop()
            self._live = None
        self.console.print("[bold red]Run failed. Better luck next round.[/bold red]")

    # -- internal rendering --

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build_display())

    def _build_display(self) -> Group:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        parts: list[Any] = []
        now = time.monotonic()

        max_model_length: int = 0
        for rows in self._rounds.values():
            for row in rows:
                name = _shorten_model(row.model)
                suffix = f" → candidate {row.candidate_index}" if row.candidate_index is not None else ""
                max_model_length = max(max_model_length, len(name) + len(suffix))
        max_model_length = max(max_model_length, 10)  # floor to avoid cramped columns before any tasks arrive

        for round_num in sorted(self._rounds):
            rows = self._rounds[round_num]
            if round_num == 0:
                parts.append(Text("  Enrichment", style="bold blue"))
            else:
                parts.append(Text(f"  Round {round_num}/{self._total_rounds}", style="bold"))

            for row in rows:
                label_text, label_style = _PHASE_LABELS.get(row.phase, (row.phase, "white"))
                model_name = _shorten_model(row.model)
                suffix = f" → candidate {row.candidate_index}" if row.candidate_index is not None else ""
                model_display = f"{model_name}{suffix}"

                if row.done:
                    icon = "[green]✓[/green]"
                    elapsed = _format_elapsed(row.elapsed)
                else:
                    icon = f"[{label_style}]{_SPINNER_FRAMES[self._spinner_index]}[/{label_style}]"
                    elapsed = _format_elapsed(now - row.start_time)

                line = Text.from_markup(
                    f"    {icon} [{label_style}]{label_text:<{_MAX_PHASE_LABEL_LENGTH}}[/{label_style}]"
                    f" {model_display:<{max_model_length}}  [dim]{elapsed}[/dim]"
                )
                parts.append(line)

            if (
                rows
                and self._active_phase
                and round_num == self._current_round
                and self._active_phase_done < self._active_phase_total
            ):
                phase_bar = Progress(
                    TextColumn("    {task.description}"),
                    BarColumn(bar_width=24),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=self.console,
                    transient=True,
                )
                fallback = (self._active_phase, "white")
                phase_label = _PHASE_LABELS.get(self._active_phase, fallback)[0]
                tid = phase_bar.add_task(
                    phase_label,
                    total=self._active_phase_total,
                    completed=self._active_phase_done,
                )

                first_active = next((row for row in rows if row.phase == self._active_phase), None)
                if first_active:
                    phase_bar.tasks[tid].start_time = first_active.start_time
                parts.append(phase_bar)

        overall_bar = Progress(
            TextColumn("  [bold green]Rounds"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        rounds_done = max(self._current_round - 1, 0)
        if self._rounds_completed == self._total_rounds and self._total_rounds > 0:
            rounds_done = self._total_rounds
        tid = overall_bar.add_task(
            "Rounds",
            total=max(self._total_rounds, 1),
            completed=rounds_done,
        )
        overall_bar.tasks[tid].start_time = self._run_start
        parts.append(Text(""))
        parts.append(overall_bar)

        return Group(*parts)
