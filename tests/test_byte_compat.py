"""A real pass must produce the same final bytes as the pinned baseline, not only a deterministic dry run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.fakes.differential import observe

GOLDEN = Path(__file__).parent / "fixtures" / "byte-compat-main-ba8c958.json"
INTENDED_DIFFERENCES = {
    # The reviewer system prompt gained the claim-attribution instructions in the dual-language work. That change is
    # deliberate and recorded in the plan; it does not move the answer for a fixed set of replies.
    "reviewer_system",
    # The archive gained a journal and extra meta fields. Both are additive.
    "journal.jsonl",
    "meta.json",
}


@pytest.fixture(scope="module")
def golden() -> dict:
    """The observation recorded from the pinned baseline tree."""
    return cast(dict, json.loads(GOLDEN.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def current(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The observation recorded from this tree, with a real transport-level replay."""
    return observe(Path(__file__).resolve().parent.parent, tmp_path_factory.mktemp("byte-compat"))


def test_the_final_answer_matches_the_baseline_bytes(golden: dict, current: dict) -> None:
    """The published answer must be the same string the baseline produced from the same replies."""
    assert current["answer_digest"] == golden["answer_digest"], (
        f"a real pass changed the final answer against baseline {golden.get('baseline_commit', '?')}"
    )
    assert current["answer"] == golden["answer"]


def test_the_same_requests_go_out_in_the_same_order(golden: dict, current: dict) -> None:
    """Every role, model, and user prompt must match; only the reviewer system prompt may move."""
    assert [row["role"] for row in current["requests"]] == [row["role"] for row in golden["requests"]]
    for index, (mine, theirs) in enumerate(zip(current["requests"], golden["requests"], strict=True)):
        assert mine["model"] == theirs["model"], f"request {index} named a different model"
        assert mine["user"] == theirs["user"], f"request {index} carries a different user prompt"
        if mine["role"] == "reviewer":
            continue
        assert mine["system"] == theirs["system"], f"request {index} carries a different system prompt"


def test_the_archive_may_only_grow(golden: dict, current: dict) -> None:
    """No artefact the baseline wrote may disappear, and a changed one must be a declared difference."""
    missing = set(golden["files"]) - set(current["files"])
    assert not missing, f"artefacts the baseline wrote are gone: {sorted(missing)}"
    changed = {name for name, digest in golden["files"].items() if current["files"].get(name) != digest}
    assert changed <= INTENDED_DIFFERENCES, f"undeclared artefact changes: {sorted(changed - INTENDED_DIFFERENCES)}"
