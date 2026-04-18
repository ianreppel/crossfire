"""Shared test utilities."""

from __future__ import annotations

import logging
from typing import Any


class LogCapture(logging.Handler):
    """Captures structured log events for test assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {}
        if hasattr(record, "event"):
            entry["event"] = record.event  # type: ignore[attr-defined]
        if hasattr(record, "data"):
            entry.update(record.data)  # type: ignore[attr-defined]
        if entry:
            self.records.append(entry)
