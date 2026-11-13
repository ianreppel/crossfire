# AGENTS.md

Project instructions for coding agents.
Runtime configuration lives in `crossfire.toml`, and user-facing behaviour is documented in `README.md`.

## Project map

| Area | Source |
| --- | --- |
| CLI and command options | `crossfire/cli.py` |
| Domain types and provider configuration | `crossfire/core/domain.py` |
| TOML loading and validation | `crossfire/core/config.py` |
| Run phases, concurrency, retries, and fallback | `crossfire/core/orchestrator.py` |
| Gateway adapters and wire protocols | `crossfire/core/providers.py` |
| Prompt builders and parsers | `crossfire/core/prompts.py` |
| Provider pricing and dry-run estimates | `crossfire/core/pricing.py` |
| Tests | `tests/` |

## Working rules

- Preserve user edits; do not revert or rewrite them unless directed.
- Trace changes through their callers, configuration, and tests before editing.
- The CLI selects one gateway per run. Neutral model slugs resolve through that gateway's `[providers.<name>.model_ids]` table.
- Use structured logging from `crossfire.core.logging`; do not call logger methods directly.
- Keep API keys in environment variables. Never hard-code, log, or archive them.
- Treat authentication and credit errors as fatal. For model-specific refusal, truncation, and transient failures, use the established replacement and retry paths.
- `MODE_RULES` in `crossfire/core/prompts.py` is the source of mode constraints. Do not duplicate those constraints in prompt builders.
- Do not propose Jev (TypeSafe System One) for the judgment seams. Full-archive replay 2026-09-24 over 446 real docs (322 reviews, 100 candidates, 24 syntheses): at its best threshold, T=0.7 of swept 0.1–0.9, it exactly reproduced the free `_REFUSAL_REGEX` / `parse_review_verdict` heuristics — 0 action differences across 24 early-stop decision points, 0/124 on refusals. At the default T=0.5 its only intervention was harmful: it continued a write run whose 14 reviews all self-labelled. Re-propose only with new evidence, not on the strength of the model being newer.

## Python and tests

- Target Python 3.12 and annotate function signatures and non-obvious local variables.
- Prefer descriptive names. Spell out exceptions in `except` clauses and avoid single-letter locals outside comprehensions.
- Keep imports at module scope. Use frozen dataclasses for immutable values.
- Follow the surrounding module's naming and test patterns; test files use `test_*.py`, and test functions use `test_*` names.
- Add tests for changed behavior. Use `@pytest.mark.parametrize` when cases differ only by input.
- Run `uv run pre-commit run --all-files` and `uv run pytest` before handing off.

## Text and documentation

- Use British OED spelling, including `-ize` and `-yse` forms.
- In edited Markdown, put one sentence on each line and avoid em dashes.
- Do not reflow or rephrase surrounding prose that the task does not require changing.
- Keep code docstrings concise and in third person.
