"""Reports differences between Crossfire's OpenCode tables and vendor sources.

The probe compares Crossfire's complete list of models whose input data OpenCode Zen may use for training or model
improvement, OpenCode Go's training and retention tables, and Zen's documented protocol mapping with current vendor docs
and model lists. Scheduled runs report mismatches between the vendor sources and Crossfire's tables. Suggested cron
line:

0 9 * * * cd /path/to/crossfire && uv run python scripts/drift_probe.py

The probe always exits with status 0. Pytest fails on a mismatch when all required sources are reachable.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

# Reuse the parsing helpers from the gating tests without importing pytest fixtures.
import test_drift


def main() -> int:
    try:
        vendor = test_drift._VendorPages()
    except Exception as error:
        print(f"drift-probe: vendor pages unreachable: {error}")
        return 0

    warnings: list[str] = []

    zen_vendors = test_drift._zen_training_ids(vendor)
    zen_shipped = set(test_drift._TRAINING_MODELS["opencode"])
    if zen_vendors != zen_shipped:
        warnings.append(f"zen training: vendor={sorted(zen_vendors)} shipped={sorted(zen_shipped)}")

    rows = test_drift._go_policy_rows(vendor)
    candidates = vendor.endpoint_ids["opencode-go"] | set(test_drift._TRAINING_MODELS["opencode-go"])
    mapped = {name for name in rows if test_drift._wire_id_for(name, candidates) is not None}
    classified = {
        name
        for name, (trains, retention) in rows.items()
        if trains == "Yes" or retention.strip("* ").lower() != "0 days"
    }
    training = {test_drift._wire_id_for(name, candidates) for name in classified if rows[name][0] == "Yes"}
    retaining = {test_drift._wire_id_for(name, candidates) for name in classified if rows[name][0] != "Yes"}
    training.discard(None)
    retaining.discard(None)
    if training != set(test_drift._TRAINING_MODELS["opencode-go"]):
        warnings.append(
            f"go training: vendor={sorted(training)} shipped={sorted(test_drift._TRAINING_MODELS['opencode-go'])}"
        )
    if retaining != set(test_drift._RETENTION_MODELS["opencode-go"]):
        warnings.append(
            f"go retention: vendor={sorted(retaining)} shipped={sorted(test_drift._RETENTION_MODELS['opencode-go'])}"
        )
    unmapped = {name for name in classified - mapped if test_drift._wire_id_for(name, candidates) is None}
    if unmapped:
        warnings.append(f"go unmapped vendor rows: {sorted(unmapped)}")

    protocols = test_drift._zen_endpoint_protocols(vendor)
    opencode = next(item for item in test_drift.DEFAULT_PROVIDER_CONFIGURATIONS if item.name == "opencode")
    mismatches = {
        wire_id: (protocol, test_drift.resolve_protocol(opencode, wire_id))
        for wire_id, protocol in protocols.items()
        if test_drift.resolve_protocol(opencode, wire_id) != protocol
    }
    if mismatches:
        warnings.append(f"protocols: {mismatches}")

    if warnings:
        print("drift-probe: catalogue drift detected")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("drift-probe: catalogue matches shipped tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
