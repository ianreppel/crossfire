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
        now: float = time.monotonic()

        max_model_display_width: int = self._compute_max_model_display_width()

        history_lines: list[Any] = []
        current_round_lines: list[Any] = []

        for round_number in sorted(self._rounds):
            rows: list[_TaskRow] = self._rounds[round_number]
            is_current: bool = round_number == self._current_round
            round_finished: bool = bool(rows) and all(row.done for row in rows)

            if not is_current and round_finished:
                history_lines.append(self._build_collapsed_round(round_number, rows))
                continue

            if round_number == 0:
                current_round_lines.append(Text("  Enrichment", style="bold blue"))
            else:
                current_round_lines.append(Text(f"  Round {round_number}/{self._total_rounds}", style="bold"))

            current_round_lines.extend(
                self._build_round_rows(round_number, rows, max_model_display_width, now)
            )

            if self._should_show_phase_bar(round_number, rows):
                current_round_lines.append(self._build_phase_bar(rows))

        overall_bar: Progress = self._build_overall_bar()
        fixed_line_count: int = len(current_round_lines) + 2  # spacer + overall bar
        terminal_height: int = self.console.size.height
        available_for_history: int = max(terminal_height - fixed_line_count, 0)

        trimmed_history: list[Any] = self._trim_history(history_lines, available_for_history)

        parts: list[Any] = trimmed_history + current_round_lines
        parts.append(Text(""))
        parts.append(overall_bar)

        return Group(*parts)

    def _compute_max_model_display_width(self) -> int:
        widest: int = 10
        for rows in self._rounds.values():
            for row in rows:
                name: str = _shorten_model(row.model)
                suffix: str = f" → candidate {row.candidate_index}" if row.candidate_index is not None else ""
                widest = max(widest, len(name) + len(suffix))
        return widest

    def _build_collapsed_round(self, round_number: int, rows: list[_TaskRow]) -> Text:
        """Renders a completed round as a single summary line."""
        task_count: int = len(rows)
        max_elapsed: float = max(row.elapsed for row in rows) if rows else 0.0
        elapsed: str = _format_elapsed(max_elapsed)

        if round_number == 0:
            model_name: str = _shorten_model(rows[0].model) if rows else "enricher"
            return Text.from_markup(f"  Enrichment  [green]✓[/green]  {model_name}  [dim]{elapsed}[/dim]")

        return Text.from_markup(
            f"  Round {round_number}/{self._total_rounds}  [green]✓[/green]"
            f"  {task_count} tasks  [dim]{elapsed}[/dim]"
        )

    def _build_collapsed_phase(self, phase: Phase, phase_rows: list[_TaskRow]) -> Text:
        """Renders a completed phase within the current round as a single summary line."""
        label_text: str = _PHASE_LABELS.get(phase, (phase, "white"))[0]
        task_count: int = len(phase_rows)
        max_elapsed: float = max(row.elapsed for row in phase_rows) if phase_rows else 0.0
        elapsed: str = _format_elapsed(max_elapsed)
        return Text.from_markup(
            f"    [green]✓[/green] {label_text}  {task_count} tasks  [dim]{elapsed}[/dim]"
        )

    def _build_round_rows(
        self,
        round_number: int,
        rows: list[_TaskRow],
        max_model_display_width: int,
        now: float,
    ) -> list[Any]:
        """Builds display lines for the current round, collapsing completed phases."""
        lines: list[Any] = []
        is_current: bool = round_number == self._current_round

        phases_in_order: list[Phase] = []
        rows_by_phase: dict[Phase, list[_TaskRow]] = {}
        for row in rows:
            if row.phase not in rows_by_phase:
                phases_in_order.append(row.phase)
                rows_by_phase[row.phase] = []
            rows_by_phase[row.phase].append(row)

        for phase in phases_in_order:
            phase_rows: list[_TaskRow] = rows_by_phase[phase]
            phase_finished: bool = all(row.done for row in phase_rows)
            has_later_phase: bool = phase != phases_in_order[-1]

            if is_current and phase_finished and has_later_phase:
                lines.append(self._build_collapsed_phase(phase, phase_rows))
                continue

            for row in phase_rows:
                lines.append(self._build_task_line(row, max_model_display_width, now))

        return lines

    def _build_task_line(self, row: _TaskRow, max_model_display_width: int, now: float) -> Text:
        """Renders a single task row with spinner or checkmark."""
        label_text, label_style = _PHASE_LABELS.get(row.phase, (row.phase, "white"))
        model_name: str = _shorten_model(row.model)
        suffix: str = f" → candidate {row.candidate_index}" if row.candidate_index is not None else ""
        model_display: str = f"{model_name}{suffix}"

        if row.done:
            icon: str = "[green]✓[/green]"
            elapsed: str = _format_elapsed(row.elapsed)
        else:
            icon = f"[{label_style}]{_SPINNER_FRAMES[self._spinner_index]}[/{label_style}]"
            elapsed = _format_elapsed(now - row.start_time)

        return Text.from_markup(
            f"    {icon} [{label_style}]{label_text:<{_MAX_PHASE_LABEL_LENGTH}}[/{label_style}]"
            f" {model_display:<{max_model_display_width}}  [dim]{elapsed}[/dim]"
        )

    def _should_show_phase_bar(self, round_number: int, rows: list[_TaskRow]) -> bool:
        return bool(
            rows
            and self._active_phase
            and round_number == self._current_round
            and self._active_phase_done < self._active_phase_total
        )

    def _build_phase_bar(self, rows: list[_TaskRow]) -> Progress:
        phase_bar: Progress = Progress(
            TextColumn("    {task.description}"),
            BarColumn(bar_width=24),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        fallback: tuple[Phase | str, str] = (self._active_phase, "white")  # type: ignore[assignment]
        phase_label: str = _PHASE_LABELS.get(self._active_phase, fallback)[0]  # type: ignore[arg-type]
        task_id = phase_bar.add_task(
            phase_label,
            total=self._active_phase_total,
            completed=self._active_phase_done,
        )

        first_active: _TaskRow | None = next(
            (row for row in rows if row.phase == self._active_phase), None
        )
        if first_active:
            phase_bar.tasks[task_id].start_time = first_active.start_time
        return phase_bar

    def _build_overall_bar(self) -> Progress:
        overall_bar: Progress = Progress(
            TextColumn("  [bold green]Rounds"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        rounds_done: int = max(self._current_round - 1, 0)
        if self._rounds_completed == self._total_rounds and self._total_rounds > 0:
            rounds_done = self._total_rounds
        task_id = overall_bar.add_task(
            "Rounds",
            total=max(self._total_rounds, 1),
            completed=rounds_done,
        )
        overall_bar.tasks[task_id].start_time = self._run_start
        return overall_bar

    @staticmethod
    def _trim_history(history_lines: list[Any], available_lines: int) -> list[Any]:
        """Trims collapsed round history from the top to fit the available lines."""
        if len(history_lines) <= available_lines:
            return history_lines

        if available_lines <= 1:
            hidden_count: int = len(history_lines)
            return [Text.from_markup(f"  [dim]… {hidden_count} earlier rounds …[/dim]")]

        visible_count: int = available_lines - 1
        hidden_count = len(history_lines) - visible_count
        return [
            Text.from_markup(f"  [dim]… {hidden_count} earlier rounds …[/dim]"),
            *history_lines[-visible_count:],
        ]
