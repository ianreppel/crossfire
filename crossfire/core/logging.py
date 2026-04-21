"""Structured logging with local timestamps, as the output is meant for the user."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from crossfire.core.domain import Phase, Role

_LOGGER_NAME = "crossfire"
_logger: logging.Logger | None = None


class _JsonFormatter(logging.Formatter):
    """Renders log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "data"):
            payload.update(record.data)
        if not hasattr(record, "event"):
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


_JSON_HANDLER_ATTR = "_cf_json_handler"


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(_LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    if not getattr(_logger, _JSON_HANDLER_ATTR, False):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        _logger.addHandler(handler)
        setattr(_logger, _JSON_HANDLER_ATTR, True)

    return _logger


def set_stderr_level(level: int) -> None:
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            handler.setLevel(level)


def _emit(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    logger = get_logger()
    record = logger.makeRecord(
        name=_LOGGER_NAME,
        level=level,
        fn="",
        lno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record.event = event
    record.data = fields
    logger.handle(record)


def log_compression_applied(
    *,
    phase: Phase,
    role: Role,
    model: str,
    round: int,
    tokens_before: int,
    tokens_after: int,
    reason: str,
) -> None:
    _emit(
        "compression_applied",
        phase=phase,
        role=role,
        model=model,
        round=round,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        reason=reason,
    )


def log_model_dropped(
    *,
    phase: Phase,
    role: Role,
    model: str,
    round: int,
    tokens_before: int | None = None,
    tokens_after: int | None = None,
    reason: str,
) -> None:
    fields: dict[str, Any] = dict(phase=phase, role=role, model=model, round=round, reason=reason)
    if tokens_before is not None:
        fields["tokens_before"] = tokens_before
    if tokens_after is not None:
        fields["tokens_after"] = tokens_after
    _emit("model_dropped", **fields)


def log_synthesis_decision(
    *,
    round: int,
    model: str,
    attributions: list[dict[str, Any]],
    selected_candidates: list[int],
    discarded_candidates: list[int],
    notes: str,
) -> None:
    _emit(
        "synthesis_decision",
        round=round,
        model=model,
        phase=Phase.SYNTHESIS,
        attributions=attributions,
        selected_candidates=selected_candidates,
        discarded_candidates=discarded_candidates,
        notes=notes,
    )


def log_round_failed(*, round: int, reason: str, details: str = "") -> None:
    _emit("round_failed", level=logging.WARNING, round=round, reason=reason, details=details)


def log_run_failed(*, round: int, reason: str, details: str = "") -> None:
    _emit("run_failed", level=logging.ERROR, round=round, reason=reason, details=details)


def log_search_failure(*, round: int, role: Role, model: str, query: str, error: str) -> None:
    _emit(
        "search_failure",
        level=logging.WARNING,
        round=round,
        role=role,
        model=model,
        query=query,
        error=error,
    )


def log_retry(*, round: int, role: Role, model: str, attempt: int, reason: str) -> None:
    _emit("retry", round=round, role=role, model=model, attempt=attempt, reason=reason)


def log_cost_summary(summary: dict[str, Any]) -> None:
    _emit("cost_summary", **summary)


def log_phase_start(*, round: int, phase: Phase) -> None:
    _emit("phase_start", round=round, phase=phase)


def log_phase_end(*, round: int, phase: Phase) -> None:
    _emit("phase_end", round=round, phase=phase)


def log_prompt_enriched(*, model: str, original_tokens: int, enriched_tokens: int) -> None:
    _emit(
        "prompt_enriched",
        model=model,
        original_tokens=original_tokens,
        enriched_tokens=enriched_tokens,
    )


def log_early_stop(*, round: int, remaining_rounds: int, reason: str) -> None:
    _emit("early_stop", round=round, remaining_rounds=remaining_rounds, reason=reason)


def log_synthesis_regression(*, round: int, model: str, reason: str) -> None:
    _emit("synthesis_regression", level=logging.WARNING, round=round, model=model, reason=reason)


def log_archive_write_failed(*, path: str, error: str) -> None:
    _emit("archive_write_failed", level=logging.WARNING, path=path, error=error)


def log_enrichment_failed(*, model: str, error: str) -> None:
    _emit("enrichment_failed", level=logging.WARNING, model=model, error=error)
