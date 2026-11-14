"""Observe one fail-closed research pass: what it sent, and what it produced.

The compatibility gate compares HEAD against a stored observation of a pinned baseline. The same module runs in
either tree, because the configuration dataclasses differ across the dual-language work and a fixed constructor call
would not survive both.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

INSTRUCTION = "Explain the Nanjing death toll with sources."
GENERATOR = "## Draft\n\nA drafted answer that cites [1] one source.\n"
REVIEWER = (
    "STRENGTHS: clear structure\n"
    "WEAKNESSES: one uncited claim\n"
    "SEVERITY: nitpick\n"
    'CROSSFIRE-VERDICT {"severity": "nitpick", "weaknesses": 1}'
)
SYNTHESIZER = (
    '{"crossfire_synthesis": {"attributions": [{"index": 0, "kept": ["draft"], "discarded": []}], '
    '"notes": "merged the draft"}}\n\n## Answer\n\nA synthesized answer that cites [1] one source.\n'
)


def digest(text: str) -> str:
    """Returns a short, stable digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _configuration(domain: Any) -> Any:
    """Builds a one-generation configuration across both dataclass shapes."""
    names = {field.name for field in fields(domain.CrossfireConfiguration)}
    model_group = domain.ModelGroup
    values: dict[str, Any] = {
        "generators": model_group(names=("gen-a",), context_window=16000),
        "reviewers": model_group(names=("rev-a",), context_window=16000),
        "synthesizer": model_group(names=("synth-a",), context_window=32000),
        "limits": domain.LimitsConfiguration(max_concurrent_requests=1),
    }
    if hasattr(domain, "SearchConfiguration"):
        values["search"] = domain.SearchConfiguration(enabled=False)
    if "translator" in names:
        values["translator"] = model_group(names=("trans-a",), context_window=16000)
    if "adjudicator" in names:
        values["adjudicator"] = model_group(names=("adj-a",), context_window=200000)
    if "language" in names:
        values["language"] = domain.LanguageConfiguration(source="en", branches=("en",))
    return domain.CrossfireConfiguration(**values)


async def _run(tree: Path, out_dir: Path) -> dict[str, Any]:
    """Runs one pass in *tree* and returns the observation."""
    import httpx

    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "tests"))
    from fakes.transport import FakeTransport

    import crossfire.core.domain as domain
    from crossfire.core.archive import RunArchive
    from crossfire.core.orchestrator import Orchestrator

    transport = FakeTransport()
    transport.default("generator", "reviewer", "synthesizer", "adjudicator", "enricher")
    for role, text in (("generator", GENERATOR), ("reviewer", REVIEWER), ("synthesizer", SYNTHESIZER)):
        transport.queue(role, *([text] * 24))

    real_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        """Serves every client from the fake transport."""
        kwargs.setdefault("transport", httpx.MockTransport(transport.handler))
        return real_client(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[assignment]
    for variable in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY", "SYNTHETIC_API_KEY"):
        os.environ[variable] = "sk-differential-fixture"

    configuration = _configuration(domain)
    parameters = domain.RunParameters(
        mode=domain.Mode.RESEARCH,
        task=domain.Task(instruction=INSTRUCTION),
        num_generators=1,
        num_reviewers_per_candidate=1,
        num_rounds=1,
        dry_run=False,
        enrich=False,
        early_stop=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = Orchestrator(configuration, parameters, archive=RunArchive(out_dir))
    answer = await orchestrator.run()
    httpx.AsyncClient = real_client  # type: ignore[assignment]

    files = {
        str(path.relative_to(out_dir)): digest(path.read_text(encoding="utf-8", errors="replace"))
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
    }
    return {
        "answer": answer,
        "answer_digest": digest(answer),
        "requests": [
            {
                "role": request.role,
                "model": request.model,
                "user": digest(request.user_prompt),
                "system": digest(request.system_prompt),
            }
            for request in transport.requests
        ],
        "files": files,
    }


def observe(tree: Path, out_dir: Path) -> dict[str, Any]:
    """Runs one pass in *tree* and returns the observation, writing its archive under *out_dir*."""
    return asyncio.run(_run(Path(tree).resolve(), Path(out_dir).resolve()))
