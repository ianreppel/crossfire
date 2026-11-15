"""Regression tests for the P1 language, evidence and reporting defects."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from rich.console import Console

from crossfire.cli import cli
from crossfire.core import archive as archive_module
from crossfire.core.adjudication import AdjudicationResult, extract_claims
from crossfire.core.archive import ArchiveWriteError, RunArchive
from crossfire.core.config import load_configuration
from crossfire.core.domain import BranchState, CrossBranchVerdict, LanguageConfiguration, ModelGroup, Role, VerdictState
from crossfire.core.evidence import EvidenceRegistry, apply_evidence_rules
from crossfire.core.language import validate_language
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.pricing import estimate_cost
from crossfire.core.prompts import build_language_tag
from crossfire.core.translation import extract_do_not_translate, fidelity_violations
from crossfire.ui.tui import TUI
from tests.fakes.transport import FakeTransport
from tests.helpers import orchestrator_for

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Explain the historical event with sources."
GERMAN = "Erklären Sie die Ereignisse mit den Quellen und der Geschichte."
ANSWER = "The supported claim has a source."


@pytest.fixture(autouse=True)
def fixed_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps regression tests independent of the wall clock."""
    fixed_datetime = Mock(now=Mock(return_value=datetime(2026, 11, 15, tzinfo=UTC)))
    monkeypatch.setattr(archive_module, "datetime", fixed_datetime)
    monkeypatch.setattr("crossfire.ui.tui.time.monotonic", lambda: 100.0)


def seed_run(fake_transport: FakeTransport) -> None:
    """Queues two complete branches and one covered comparison."""
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("translator", GERMAN)
    fake_transport.queue("synthesizer", ANSWER, ANSWER)
    fake_transport.queue(
        "adjudicator",
        json.dumps({"crossfire_adjudication": {"verdicts": [{"claim": f"text:{ANSWER}", "state": "disagrees"}]}})
        + '\nCROSSFIRE-ADJUDICATION {"claims": 1, "unavailable": 0}',
    )


async def test_unknown_preset_fails_before_calls(fake_transport: FakeTransport, tmp_path: Path) -> None:
    language = LanguageConfiguration(preset="multilingaul")
    assert any("preset" in error for error in validate_language(language))
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path)
    configuration = replace(orchestrator.configuration, language=language)
    with pytest.raises(ValueError, match="preset"):
        await Orchestrator(configuration, orchestrator.parameters).run()
    assert fake_transport.requests == []


@pytest.mark.parametrize("through_cli", [False, True])
async def test_mode_auto_source_is_detected(through_cli: bool, fake_transport: FakeTransport, tmp_path: Path) -> None:
    instruction = "Die Geschichte ist in den Quellen und in der Forschung umstritten."
    fake_transport.default("generator", "reviewer", "synthesizer")
    if through_cli:
        configuration_path = tmp_path / "config.toml"
        configuration_path.write_text(
            '[models.generators]\nnames=["gen-a"]\n[models.reviewers]\nnames=["rev-a"]\n'
            '[models.synthesizer]\nnames=["synth-a"]\n[search]\nenabled=false\n'
            '[language]\nbranches=["de"]\n[language.modes.research]\nsource="auto"\n'
        )
        # Click owns its event loop, so this path is exercised in a worker thread.
        result = await asyncio.to_thread(
            CliRunner().invoke,
            cli,
            [
                "run",
                "--mode",
                "research",
                "--instruction",
                instruction,
                "--config",
                str(configuration_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--num-generators",
                "1",
                "--num-reviewers-per-candidate",
                "1",
                "--num-rounds",
                "1",
                "--verbose",
            ],
        )
        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / "run" / "meta.json").read_text())["language"]
    else:
        orchestrator = orchestrator_for(instruction, tmp_path, branches=("de",))
        configuration = replace(orchestrator.configuration, language_modes={"research": {"source": "auto"}})
        await Orchestrator(configuration, orchestrator.parameters, archive=RunArchive(tmp_path)).run()
        manifest = json.loads((tmp_path / "meta.json").read_text())["language"]
    assert manifest["source"] == "de"
    assert manifest["provenance"]["source"] == "mode"
    assert not any("auto->" in request.user_prompt for request in fake_transport.requests)


def test_surviving_auto_is_invalid() -> None:
    assert any("source" in error for error in validate_language(LanguageConfiguration(source="auto")))


def test_cli_configuration_conflict_has_one_line_error(tmp_path: Path, fake_transport: FakeTransport) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[language]\nbranches=["en"]\npreset="multilingual"\n')
    result = CliRunner().invoke(cli, ["run", "--mode", "research", "--instruction", INSTRUCTION, "--config", str(path)])
    assert result.exit_code == 1
    assert result.output.startswith("Configuration error: ")
    assert len(result.output.splitlines()) == 1
    assert "Traceback" not in result.output
    assert fake_transport.requests == []


def test_isbn_claim_harvests_citable_span() -> None:
    text = "The edition carries ISBN 978-4-12-345678-9 and supports this claim."
    claim = extract_claims({"en": text})[0]
    registry = EvidenceRegistry()
    spans = registry.harvest("en", text)
    assert claim.key == "id:ISBN 978-4-12-345678-9"
    assert len(spans) == 1
    assert spans[0].text == text
    masked, protected = extract_do_not_translate(text)
    assert any(span.text == claim.key.removeprefix("id:") for span in protected)
    assert "ISBN" not in masked


@pytest.mark.parametrize("translate_instruction", [True, False])
async def test_estimate_and_run_resolve_same_branches(
    translate_instruction: bool, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    seed_run(fake_transport)
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path, branches=("en",))
    configuration = replace(
        orchestrator.configuration,
        language_modes={"research": {"branches": ["en", "de"], "translate_instruction": translate_instruction}},
    )
    prices = {"trans-a": (0.01, 0.01)}
    estimate = estimate_cost(configuration, orchestrator.parameters, prices, "offline")
    runner = Orchestrator(configuration, orchestrator.parameters, archive=RunArchive(tmp_path))
    await runner.run()
    assert [tag for tag, _cost in estimate.per_branch] == list(runner.language.branches)
    assert (dict(estimate.per_branch)["de"] > dict(estimate.per_branch)["en"]) is translate_instruction
    assert [entry.branch for entry in runner.cost_tracker.entries if entry.role is Role.TRANSLATOR] == (
        ["de"] if translate_instruction else []
    )


class RecordingTUI(TUI):
    """Records the terminal callback without starting a live display."""

    def __init__(self) -> None:
        super().__init__()
        self.ends = 0

    def on_run_end(self) -> None:
        self.ends += 1
        super().on_run_end()


async def test_progress_ends_once_and_labels_both_branches(fake_transport: FakeTransport, tmp_path: Path) -> None:
    seed_run(fake_transport)
    progress = RecordingTUI()
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path)
    orchestrator._progress = progress
    await orchestrator.run()
    assert progress.ends == 1
    console = Console(record=True, width=120, height=100)
    with console.capture() as captured:
        console.print(progress._build_display())
    display = captured.get()
    assert "en" in display and "de" in display
    assert ("en", 1) in progress._rounds and ("de", 1) in progress._rounds


@pytest.mark.parametrize("field", ["source", "render_language"])
@pytest.mark.parametrize("value", ["../../escaped", "/tmp/escaped", "en\n", "auto"])
def test_source_and_render_tags_are_validated(field: str, value: str) -> None:
    language = (
        replace(LanguageConfiguration(), source=value)
        if field == "source"
        else replace(LanguageConfiguration(), render_language=value)
    )
    assert any(field in error for error in validate_language(language))


@pytest.mark.parametrize("through_symlink", [False, True])
def test_archive_refuses_escape(tmp_path: Path, through_symlink: bool) -> None:
    archive = RunArchive(tmp_path / "run")
    if through_symlink:
        archive.base.mkdir()
        (archive.base / "render").symlink_to(tmp_path, target_is_directory=True)
        language = "escaped"
    else:
        language = "../../escaped"
    with pytest.raises(ArchiveWriteError, match=r"outside|escape"):
        archive.save_render(language, "must stay within the archive")
    assert not (tmp_path / "escaped.md").exists()


@pytest.mark.parametrize(
    "reply", [INSTRUCTION, "La recherche est dans les sources et les documents.", "I cannot fulfill this request."]
)
async def test_invalid_translation_is_not_attested(reply: str, fake_transport: FakeTransport, tmp_path: Path) -> None:
    fake_transport.queue("translator", reply)
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path)
    branch = await orchestrator._translate_instruction(orchestrator._build_branches(orchestrator.configuration)[1])
    assert branch.state is BranchState.TRANSLATION_FAILED
    assert not (tmp_path / "branches/de/instruction.md").exists()


async def test_reviewer_request_carries_language_tag(fake_transport: FakeTransport, tmp_path: Path) -> None:
    seed_run(fake_transport)
    await orchestrator_for(INSTRUCTION, tmp_path).run()
    assert build_language_tag("de") in fake_transport.texts("reviewer")[1]


@pytest.mark.parametrize("field", ["Author", "Title", "Journal", "Publisher"])
def test_explicit_metadata_is_protected(field: str) -> None:
    source = f"{field}: Original Work\nSummarize the argument."
    masked, spans = extract_do_not_translate(source)
    invented = f"{field}: Invented Work\nFassen Sie das Argument zusammen."
    assert fidelity_violations(masked, invented, spans)
    assert spans[0].text == f"{field}: Original Work"


@pytest.mark.parametrize("operation", ["instruction", "render", "adjudication"])
async def test_missing_model_group_fails_before_calls(
    operation: str, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    seed_run(fake_transport)
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path)
    language = replace(
        orchestrator.language,
        render_language="fr" if operation == "render" else None,
        translate_instruction=operation == "instruction",
    )
    group = "adjudicator" if operation == "adjudication" else "translator"
    configuration = replace(orchestrator.configuration, language=language)
    missing_group = ModelGroup(names=(), context_window=16000)
    configuration = (
        replace(configuration, adjudicator=missing_group)
        if group == "adjudicator"
        else replace(configuration, translator=missing_group)
    )
    with pytest.raises(ValueError, match=group):
        await Orchestrator(configuration, orchestrator.parameters).run()
    assert fake_transport.requests == []


async def test_cost_identity_and_persisted_subtotals(fake_transport: FakeTransport, tmp_path: Path) -> None:
    seed_run(fake_transport)
    fake_transport.queue("translator", "Die Zusammenfassung ist in der Sprache der Leser.")
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path, render_language="de")
    orchestrator._pricing = {model: (0.001, 0.002) for model in ("gen-a", "rev-a", "synth-a", "trans-a", "adj-a")}
    await orchestrator.run()
    entries = orchestrator.cost_tracker.entries
    assert [entry.branch for entry in entries if entry.role is Role.TRANSLATOR] == ["de", "render:de"]
    assert [entry.branch for entry in entries if entry.role is Role.ADJUDICATOR] == ["adjudication"]
    summary = json.loads((tmp_path / "meta.json").read_text())["cost"]
    assert sum(value["cost"] for value in summary["per_branch"].values()) == pytest.approx(summary["total_cost"])
    assert sum(value["input_tokens"] for value in summary["per_branch"].values()) == summary["total_input_tokens"]
    assert set(summary["per_branch"]) == {"en", "de", "adjudication", "render:de"}


@pytest.mark.parametrize("fallback", [False, True])
async def test_adjudicator_wire_identity_is_independent(
    fallback: bool, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path)
    provider = orchestrator.configuration.active_provider()
    provider = replace(provider, model_ids=(("adj-a", "shared"), ("synth-a", "shared"), ("other", "different")))
    configuration = replace(
        orchestrator.configuration,
        providers=(provider,),
        synthesizer=replace(
            orchestrator.configuration.synthesizer, names=("other", "synth-a") if fallback else ("synth-a",)
        ),
    )
    assert any("adjudicator" in error and "synthesizer" in error for error in configuration.validate(1, 1))
    with pytest.raises(ValueError, match="adjudicator"):
        await Orchestrator(configuration, orchestrator.parameters).run()
    assert fake_transport.requests == []


def test_shipped_adjudicator_example_is_independent() -> None:
    text = (ROOT / "crossfire.toml").read_text()
    example = text.split("#   [models.adjudicator]", 1)[1].split("#\n", 1)[0]
    adjudicator = example.split('names = ["', 1)[1].split('"', 1)[0]
    configuration = load_configuration(ROOT / "crossfire.toml")
    for mode in ("research", "write", "code", "edit", "check"):
        resolved = configuration.resolve_for_mode(mode)
        resolved = replace(resolved, adjudicator=replace(resolved.adjudicator, names=(adjudicator,)))
        assert not any("adjudicator" in error and "synthesizer" in error for error in resolved.validate(1, 1))
        provider = resolved.active_provider()
        assert provider.resolve_wire_model_id(adjudicator) not in {
            provider.resolve_wire_model_id(model) for model in resolved.synthesizer.names
        }


async def test_single_foreign_branch_renders_back_to_source(fake_transport: FakeTransport, tmp_path: Path) -> None:
    fake_transport.default("generator", "reviewer")
    fake_transport.queue("translator", GERMAN, "The supported claim has a source.")
    fake_transport.queue("synthesizer", "Die Aussage ist in den Quellen belegt.")
    result = await orchestrator_for(INSTRUCTION, tmp_path, branches=("de",), render_language="en").run()
    assert (tmp_path / "render/en.md").read_text() == result
    assert (tmp_path / "final.md").read_text() == result
    assert '"attested": false' in result
    assert "The supported claim has a source." in result
    assert (tmp_path / "output.md").read_text() == "Die Aussage ist in den Quellen belegt."
    assert fake_transport.count("adjudicator") == 0
    assert fake_transport.texts("translator")[1].startswith("[TRANSLATE de->en]")


async def test_translation_exclusion_has_single_branch_shape(fake_transport: FakeTransport, tmp_path: Path) -> None:
    fake_transport.default("generator", "reviewer", "synthesizer")
    fake_transport.queue("translator", "")
    await orchestrator_for(INSTRUCTION, tmp_path).run()
    record = [record for record in RunArchive(tmp_path).journal.replay() if record["event"] == "run_shape"][-1]
    assert record["shape"] == "single-branch"
    assert record["surviving_branches"] == 1
    assert record["status"] == "PARTIAL"


async def test_render_uses_published_claim_layout(fake_transport: FakeTransport, tmp_path: Path) -> None:
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path, render_language="en")
    verdict = CrossBranchVerdict(claim="text:Claim", state=VerdictState.DISAGREES)
    body = "## text:Claim\n\nState: disagrees.\nCanonical claim layout from the adjudication renderer.\n"
    adjudication = AdjudicationResult(verdicts=(verdict,), published="# Cross-branch adjudication\n\n" + body)
    context = orchestrator._prepare_branch_context(orchestrator._build_branches(orchestrator.configuration)[0])
    rendered = await orchestrator._render_output(adjudication, [context], status="COMPLETE")
    assert body in rendered


@pytest.mark.parametrize(
    "text", ["> The archival record supports this shared quotation.", "The archive records 42 witnesses."]
)
def test_shared_passage_can_support_agreement(text: str) -> None:
    registry = EvidenceRegistry()
    spans = registry.harvest("en", text) + registry.harvest("de", text)
    assert len(spans) == 2
    verdict = CrossBranchVerdict(
        claim=extract_claims({"en": text, "de": text})[0].key,
        state=VerdictState.AGREES,
        by_branch={"en": text, "de": text},
        evidence_ids=[span.id for span in spans],
    )
    checked, violations = apply_evidence_rules(verdict, registry, {"en": text, "de": text})
    assert checked.state is VerdictState.AGREES
    assert violations == []


def test_readme_documents_language_flags() -> None:
    readme = (ROOT / "README.md").read_text()
    for flag in (
        "--language-branch",
        "--source-language",
        "--competence-branch",
        "--render-language",
        "--no-translate-instruction",
        "--no-adjudicate",
        "--resume",
    ):
        assert flag in readme


def test_numeral_evidence_is_a_verbatim_sentence() -> None:
    registry = EvidenceRegistry()
    text = "An introductory sentence. The archive records 42 witnesses. A closing sentence."
    spans = registry.harvest("en", text)
    assert [span.text for span in spans] == ["The archive records 42 witnesses."]


async def test_translation_exclusion_recomputes_multi_branch_shape(
    fake_transport: FakeTransport, tmp_path: Path
) -> None:
    seed_run(fake_transport)
    fake_transport.queue("translator", "")
    await orchestrator_for(INSTRUCTION, tmp_path, branches=("en", "de", "fr")).run()
    record = [record for record in RunArchive(tmp_path).journal.replay() if record["event"] == "run_shape"][-1]
    assert record["shape"] == "full"
    assert record["surviving_branches"] == 2
    assert record["status"] == "PARTIAL"


async def test_render_fallback_requires_translator_before_calls(fake_transport: FakeTransport, tmp_path: Path) -> None:
    seed_run(fake_transport)
    orchestrator = orchestrator_for(INSTRUCTION, tmp_path, render_language="en")
    configuration = replace(
        orchestrator.configuration,
        language=replace(orchestrator.language, translate_instruction=False),
        translator=replace(orchestrator.configuration.translator, names=()),
    )
    with pytest.raises(ValueError, match="translator"):
        await Orchestrator(configuration, orchestrator.parameters).run()
    assert fake_transport.requests == []


async def test_render_matches_published_claims(fake_transport: FakeTransport, tmp_path: Path) -> None:
    seed_run(fake_transport)
    result = await orchestrator_for(INSTRUCTION, tmp_path, render_language="en").run()
    published = (tmp_path / "cross-branch.md").read_text()
    assert result[result.index("## text:") :] == published[published.index("## text:") :]


@pytest.mark.parametrize("field", ["Author", "Title", "Journal", "Publisher"])
def test_preserved_metadata_passes_fidelity(field: str) -> None:
    source = f"{field}: Original Work\nSummarize the argument."
    masked, spans = extract_do_not_translate(source)
    translated = f"{spans[0].placeholder}\nFassen Sie das Argument zusammen."
    assert fidelity_violations(masked, translated, spans) == []


@pytest.mark.parametrize(
    "identifier",
    [
        "ISBN 978-4-12-345678-9",
        "978-4-12-345678-9",
        "0-306-40615-2",
        "10.1234/example",
        "https://example.org/source",
        "www.example.org/source",
    ],
)
def test_identifier_rules_are_shared(identifier: str) -> None:
    text = f"The source is {identifier} and supports the finding"
    claim = extract_claims({"en": text})[0]
    assert claim.key == f"id:{identifier}"
    spans = EvidenceRegistry().harvest("en", text)
    assert [span.text for span in spans] == [text]
    masked, protected = extract_do_not_translate(text)
    assert [span.text for span in protected] == [identifier]
    assert fidelity_violations(masked, masked, protected) == []
