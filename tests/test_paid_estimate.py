"""Tests for cost reporting before a paid run starts."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from crossfire import cli as cli_module
from crossfire.core.domain import CostEstimate, CrossfireConfiguration, RunParameters
from crossfire.core.pricing import save_pricing
from tests.fakes.transport import FakeTransport

CONFIGURATION = Path(__file__).parent / "fixtures" / "crossfire-dual-language.toml"


@pytest.mark.parametrize("verbose", [False, True])
def test_paid_run_prints_estimate_before_launch(
    verbose: bool, fake_transport: FakeTransport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pricing = {f"opencode::{model}": (0.00001, 0.00002) for model in ("gen-a", "rev-a", "synth-a")}
    save_pricing(pricing, "2026-11-14T00:00:00Z", tmp_path / "pricing.json")
    fake_transport.default("generator", "reviewer", "synthesizer")
    printed: list[CostEstimate] = []
    original_print = cli_module._print_cost_estimate
    original_execute = cli_module._execute

    def capture_estimate(estimate: CostEstimate) -> None:
        printed.append(estimate)
        original_print(estimate)

    async def check_launch(
        configuration: CrossfireConfiguration,
        parameters: RunParameters,
        archive_path: Path,
        *,
        verbose: bool = False,
        cost_estimate: CostEstimate | None = None,
    ) -> str:
        assert printed, "the estimate must be printed before launching the paid run"
        assert cost_estimate == printed[0]
        assert cost_estimate is not None and cost_estimate.total_usd > 0
        assert cost_estimate.missing_models == ()
        assert fake_transport.requests == []
        return await original_execute(
            configuration, parameters, archive_path, verbose=verbose, cost_estimate=cost_estimate
        )

    monkeypatch.setattr(cli_module, "_print_cost_estimate", capture_estimate)
    monkeypatch.setattr(cli_module, "_execute", check_launch)
    arguments = [
        "run",
        "--mode",
        "research",
        "--instruction",
        "Explain the finding.",
        "--config",
        str(CONFIGURATION),
        "--run-dir",
        str(tmp_path / "run"),
        "--language-branch",
        "en",
        "--num-rounds",
        "1",
        "--num-reviewers-per-candidate",
        "1",
        "--no-enrich",
    ]
    if verbose:
        arguments.append("--verbose")

    result = CliRunner().invoke(cli_module.cli, arguments)

    assert result.exit_code == 0, result.output
    assert "Estimated cost (upper bound for configured rounds): $" in result.output
    assert len(printed) == 1
    assert fake_transport.count("generator") == 1
