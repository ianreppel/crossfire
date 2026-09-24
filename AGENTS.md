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
