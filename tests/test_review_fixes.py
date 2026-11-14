"""Two findings pinned as tests: the cost snapshot covers every phase, and a configured automatic source is detected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from crossfire.cli import cli
from tests.fakes.transport import FakeTransport
from tests.test_render import run_two_branch_adjudication


@pytest.mark.asyncio
async def test_meta_cost_includes_adjudication_and_render_charges(
    fake_transport: FakeTransport, tmp_path: Path
) -> None:
    """The saved cost snapshot must charge every phase, including the ones after synthesis."""
    await run_two_branch_adjudication(fake_transport, tmp_path, render_language="en")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert fake_transport.count("adjudicator") >= 1, "the adjudicator must be billed in this run"
    assert "adj-a" in meta["cost"]["per_model"], (
        "meta.json must charge adjudication, not only the phases before it; "
        f"charged models were {sorted(meta['cost']['per_model'])}"
    )


def test_configured_automatic_source_is_detected(fake_transport: FakeTransport, tmp_path: Path) -> None:
    """`source = "auto"` set in configuration, with no CLI flag, must still resolve before the run."""
    fake_transport.default("generator", "reviewer", "synthesizer", "adjudicator")
    fake_transport.queue("translator", "Erklären Sie die Opferzahl von Nanjing anhand von Quellen.")
    spanish = "Explique la cifra de muertos de Nanjing con fuentes."
    configuration = tmp_path / "crossfire.toml"
    source = (Path(__file__).parent / "fixtures" / "crossfire-dual-language.toml").read_text(encoding="utf-8")
    configuration.write_text(source.replace('source = "en"', 'source = "auto"'), encoding="utf-8")
    run_dir = tmp_path / "run"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--mode",
            "research",
            "--num-reviewers-per-candidate",
            "1",
            "--no-enrich",
            "--instruction",
            spanish,
            "--config",
            str(configuration),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    resolved = meta["language"]
    assert resolved["source"] == "es", (
        "a configured automatic source must resolve to the detected language, not stay as the literal 'auto'; "
        f"resolved source was {resolved['source']!r}"
    )
