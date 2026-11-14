#!/usr/bin/env python3
"""Regenerate the byte-compatibility golden from a pinned baseline tree.

    python3 tools/differential-golden.py <baseline-tree> tests/fixtures/byte-compat.json

The baseline for the current golden is main at ba8c958. Record the commit in the JSON so a later reader knows which
tree produced it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fakes.differential import observe  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as scratch:
        payload = observe(tree, Path(scratch) / "run")
    payload["baseline_commit"] = commit
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"golden written from {tree} at {commit[:7]}: answer {payload['answer_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
