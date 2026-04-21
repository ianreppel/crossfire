"""CLI for ``crossfire run``, ``crossfire clean``, and ``crossfire prices``."""

from __future__ import annotations

import asyncio
import logging as _logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

from crossfire.core.archive import RunArchive
from crossfire.core.config import load_configuration
from crossfire.core.domain import CostEstimate, CrossfireConfiguration, Mode, RunParameters, Task
from crossfire.core.logging import set_stderr_level
from crossfire.core.orchestrator import Orchestrator, RunFailedError
from crossfire.core.pricing import (
    PRICING_FILENAME,
    estimate_cost,
    fetch_pricing,
    load_pricing,
    parse_api_response,
    save_pricing,
)
from crossfire.core.search import get_search_api_key
from crossfire.ui.tui import TUI


@click.group()
def cli() -> None:
    """Crossfire — adversarial LLM refinement."""


@cli.command()
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in Mode]),
    required=True,
    help="Operating mode.",
)
@click.option(
    "--instruction",
    default=None,
    help="Task instruction (mutually exclusive with --instruction-file).",
)
@click.option(
    "--instruction-file",
    type=click.Path(exists=True),
    default=None,
    help="Read instruction from a Markdown file (mutually exclusive with --instruction).",
)
@click.option(
    "--context-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to context file.",
)
@click.option("--num-generators", type=int, default=1, show_default=True)
@click.option("--num-reviewers-per-candidate", type=int, default=3, show_default=True)
@click.option("--num-rounds", type=int, default=3, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--enrich/--no-enrich", default=True, help="Enrich instruction before generation.")
@click.option(
    "--early-stop/--no-early-stop",
    default=True,
    help="Stop early when reviews find no weaknesses.",
)
@click.option(
    "--early-stop-threshold",
    type=int,
    default=1,
    help=(
        "Weakness threshold for early stopping (reviews with more weaknesses"
        " trigger another round, provided num-rounds allows it)."
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show JSON log events on stderr.",
)
@click.option("--output", type=click.Path(), default=None)
@click.option("--run-dir", type=click.Path(), default=None, help="Archive directory.")
@click.option(
    "--config",
    "configuration_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to crossfire.toml.",
)
def run(
    mode: str,
    instruction: str | None,
    instruction_file: str | None,
    context_file: str | None,
    num_generators: int,
    num_reviewers_per_candidate: int,
    num_rounds: int,
    dry_run: bool,
    enrich: bool,
    early_stop: bool,
    early_stop_threshold: int,
    verbose: bool,
    output: str | None,
    run_dir: str | None,
    configuration_path: str | None,
) -> None:
    """Runs the Crossfire pipeline."""
    if not verbose:
        set_stderr_level(_logging.WARNING)

    if instruction and instruction_file:
        click.echo("Pick one: --instruction or --instruction-file, not both.", err=True)
        sys.exit(1)
    if not instruction and not instruction_file:
        click.echo("Nothing to do: provide --instruction or --instruction-file.", err=True)
        sys.exit(1)

    if instruction_file:
        effective_instruction: str = Path(instruction_file).read_text(encoding="utf-8")
    else:
        assert instruction is not None
        effective_instruction = instruction

    base_configuration: CrossfireConfiguration = load_configuration(
        configuration_path=Path(configuration_path) if configuration_path else None,
    )
    configuration: CrossfireConfiguration = base_configuration.resolve_for_mode(mode)

    context: str = ""
    if context_file:
        context = Path(context_file).read_text(encoding="utf-8")

    task = Task(instruction=effective_instruction, context=context)

    parameters = RunParameters(
        mode=Mode(mode),
        task=task,
        num_generators=num_generators,
        num_reviewers_per_candidate=num_reviewers_per_candidate,
        num_rounds=num_rounds,
        dry_run=dry_run,
        enrich=enrich,
        early_stop=early_stop,
        early_stop_threshold=early_stop_threshold,
    )

    errors = configuration.validate(parameters.num_generators, parameters.num_reviewers_per_candidate)
    if errors:
        for error in errors:
            click.echo(f"Configuration error: {error}", err=True)
        sys.exit(1)

    # Fail fast on a missing search key rather than burning tokens through generation
    # and review only to discover the misconfiguration at the first search request.
    if configuration.search.enabled and not parameters.dry_run:
        try:
            get_search_api_key()
        except RuntimeError as exception:
            click.echo(f"Configuration error: {exception}", err=True)
            sys.exit(1)

    cost_estimate: CostEstimate | None = None
    if parameters.dry_run:
        cost_estimate = _try_estimate_cost(configuration, parameters)

    archive_path: Path = Path(run_dir or f"runs/{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}")

    try:
        result = asyncio.run(
            _execute(configuration, parameters, archive_path, verbose=verbose, cost_estimate=cost_estimate)
        )
    except RunFailedError as exception:
        click.echo(f"Run failed: {exception}", err=True)
        sys.exit(1)
    except ValueError as exception:
        click.echo(f"Configuration error: {exception}", err=True)
        sys.exit(1)
    except RuntimeError as exception:
        click.echo(f"Error: {exception}", err=True)
        sys.exit(1)

    # The archive is the single source of truth: it always writes final.md. We only
    # write an additional copy when the user explicitly asks with --output.
    if output:
        output_path: Path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        click.echo(f"Output written to {output_path}", err=True)
    click.echo(f"Artifacts saved to {archive_path}", err=True)


_CLEAN_DIRS = ["runs", ".venv", ".ruff_cache", ".pytest_cache", ".mypy_cache", "dist", "build"]
_CLEAN_GLOBS = ["**/__pycache__", "**/*.pyc", "**/*.egg-info"]


@cli.command()
@click.confirmation_option(prompt="Remove all generated/cached files (including .venv)?")
def clean() -> None:
    """Removes all generated and cached files."""
    if not Path("crossfire.toml").is_file():
        click.echo(
            "No can do! There is no crossfire.toml in the current directory. "
            "Run this from the project root to avoid nuking an unrelated .venv or caches.",
            err=True,
        )
        sys.exit(1)

    removed: list[str] = []
    for name in _CLEAN_DIRS:
        target = Path(name)
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    for pattern in _CLEAN_GLOBS:
        for match in sorted(Path(".").glob(pattern), reverse=True):
            if match.is_dir():
                shutil.rmtree(match)
            else:
                match.unlink()
            removed.append(str(match))
    if removed:
        click.echo(f"Removed {len(removed)} item(s): {', '.join(removed)}")
    else:
        click.echo("Nothing to clean.")


@cli.command()
def prices() -> None:
    """Fetches current model pricing from OpenRouter and writes pricing.json."""
    if not Path("crossfire.toml").is_file():
        click.echo(
            "No can do! There is no crossfire.toml in the current directory. " "Run this from the project root.",
            err=True,
        )
        sys.exit(1)

    click.echo("Fetching pricing from OpenRouter...", err=True)
    try:
        raw = fetch_pricing()
    except Exception as exception:
        click.echo(f"Failed to fetch pricing: {exception}", err=True)
        sys.exit(1)

    pricing = parse_api_response(raw)
    fetched_at: str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path = Path(PRICING_FILENAME)
    save_pricing(pricing, fetched_at, output_path)
    click.echo(f"Saved pricing for {len(pricing)} models to {output_path}", err=True)


def _try_estimate_cost(
    configuration: CrossfireConfiguration,
    parameters: RunParameters,
) -> CostEstimate | None:
    """Loads pricing.json and returns a cost estimate, or None if the file is missing."""
    pricing_path = Path(PRICING_FILENAME)
    if not pricing_path.is_file():
        click.echo(
            "No pricing data found. Run 'crossfire prices' to enable cost estimates.",
            err=True,
        )
        return None
    try:
        pricing, fetched_at = load_pricing(pricing_path)
    except (ValueError, OSError) as exception:
        click.echo(f"Could not load {PRICING_FILENAME}: {exception}", err=True)
        return None
    return estimate_cost(configuration, parameters, pricing, fetched_at)


async def _execute(
    configuration: CrossfireConfiguration,
    parameters: RunParameters,
    archive_path: Path,
    *,
    verbose: bool = False,
    cost_estimate: CostEstimate | None = None,
) -> str:
    """Bootstraps the TUI (unless verbose), runs the orchestrator, and returns the final output."""
    archive: RunArchive = RunArchive(archive_path)

    # In verbose mode, JSON logs go to stderr -- skip the TUI to avoid collision.
    tui: TUI | None = None
    if not verbose:
        tui = TUI()
        tui.start(parameters)

    try:
        orchestrator = Orchestrator(configuration, parameters, progress=tui, archive=archive)
        result: str = await orchestrator.run()
        if tui:
            tui.finish(orchestrator.cost_tracker.summarize(), cost_estimate=cost_estimate)
        elif cost_estimate is not None:
            _print_cost_estimate(cost_estimate)
        return result
    except Exception:
        if tui:
            tui.report_error()
        raise


def _print_cost_estimate(cost_estimate: CostEstimate) -> None:
    """Prints the cost estimate to stderr (verbose mode fallback when TUI is not active)."""
    click.echo(f"Estimated cost: ${cost_estimate.total_usd:.2f}", err=True)
    if cost_estimate.fetched_at:
        click.echo(f"Prices from: {cost_estimate.fetched_at[:10]}", err=True)


def main() -> None:
    """Entrypoint for the ``crossfire`` console script."""
    cli()


if __name__ == "__main__":
    main()
