"""The end-to-end CLI run: one instruction, two language branches, an adjudication and a render."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from crossfire.cli import cli
from tests.fakes.transport import FakeTransport

CONFIGURATION_PATH = Path(__file__).parent / "fixtures" / "crossfire-dual-language.toml"
INSTRUCTION = "Explain the Nanjing death toll with sources."


def test_cli_run_writes_two_branches_an_adjudication_and_a_render(
    fake_transport: FakeTransport,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("translator", "Erklären Sie die Opferzahl von Nanjing anhand von Quellen.")
    fake_transport.queue(
        "synthesizer",
        "## Answer\n\nThe toll was 200,000 or more [1].",
        "## Antwort\n\nDie Zahl der Toten ist in den Quellen umstritten.",
    )
    fake_transport.queue(
        "adjudicator",
        '{"crossfire_adjudication": {"verdicts": [{"claim": "the death toll", "state": "absent_in_branch", '
        '"by_branch": {"en": "The toll was 200,000 or more [1].", "de": ""}}]}}\n'
        'CROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 1}',
    )

    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--mode",
            "research",
            "--instruction",
            INSTRUCTION,
            "--config",
            str(CONFIGURATION_PATH),
            "--run-dir",
            str(run_dir),
            "--num-rounds",
            "1",
            "--num-reviewers-per-candidate",
            "1",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    english = (run_dir / "branches" / "en" / "output.md").read_text(encoding="utf-8")
    german = (run_dir / "branches" / "de" / "output.md").read_text(encoding="utf-8")
    assert english == "## Answer\n\nThe toll was 200,000 or more [1]."
    assert german == "## Antwort\n\nDie Zahl der Toten ist in den Quellen umstritten."
    assert (run_dir / "branches" / "de" / "instruction.md").read_text(encoding="utf-8") == (
        "Erklären Sie die Opferzahl von Nanjing anhand von Quellen."
    )
    assert "absent_in_branch" in (run_dir / "cross-branch.md").read_text(encoding="utf-8")
    render = (run_dir / "render" / "en.md").read_text(encoding="utf-8")
    assert 'CROSSFIRE-RENDER {"language": "en", "attested": false' in render
    assert "The toll was 200,000 or more [1]." in render
    assert (run_dir / "final.md").read_text(encoding="utf-8") == render
    assert fake_transport.count("translator") >= 1
    assert fake_transport.count("adjudicator") == 1
