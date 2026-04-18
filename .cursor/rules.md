# Crossfire — Cursor Rules

## Naming

- No abbreviations or acronyms in names: `configuration` not `config`, `parameters` not `params`, `index` not `idx`, `length` not `len` (except when using said built-in function)
- No single-letter variables: `match` not `m`, `line` not `ln`, `candidate` not `c` (short names are acceptable inside list/dict/set comprehensions)
- Spell out `exception` in except clauses (not `exc`)
- Regex constants use `_REGEX` suffix (not `_RE`)
- Class names spelled out: `Configuration` not `Config`, `Parameters` not `Params`

## Docstrings

- Function/method docstrings use third person ("Rewrites the instruction" not "Rewrite the instruction")
- One-liner and module docstrings: end with a full stop
- British OED spelling throughout (-ize/-yse)
- Human-readable and concise, not technically precise but unintelligible

## Architecture

- `domain.py` holds all data structures (not `models.py`, which is ambiguous in an LLM project)
- `openrouter.py` is the HTTP client layer (separate from orchestration)
- `simulation.py` holds all dry-run fakes (LLM responses and search results)
- `prompts.py` contains prompt builders (pure functions) and output parsers (review verdict, synthesis decision)
- `compression.py` handles both low-level text compression and prompt fitting within token budgets
- `progress.py` defines the `ProgressCallback` protocol and `NoOpProgress` fallback
- `reviewers.py` holds the reviewer-to-candidate assignment algorithm
- `exclamations.py` holds the Simpsons-quote prefix helper used to dress up error messages
- Cross-group model overlap is allowed: a model may generate and review in the same round
- Compression priority: candidates first, then reviews, then context; the task instruction is never compressed

### Dependency direction

- `cli.py` depends on `core/` and `ui/`; nothing in `core/` or `ui/` imports from `cli.py`
- `ui/tui.py` depends on `core/progress.py` (protocol) and `core/domain.py` (types); it never imports from `core/orchestrator.py`
- Within `core/`, `orchestrator.py` is the only module that imports from `openrouter.py`, `compression.py`, and `reviewers.py`; `search.py` is imported by `orchestrator.py` only but may itself import `simulation.py` for its dry-run path
- `prompts.py` depends only on `domain.py`; it has no runtime dependencies on other core modules
- `exclamations.py` is a leaf module (stdlib-only); it may be imported from anywhere in `core/`

## Prompts

- `MODE_RULES` is the single source of constraints (never duplicate rules in system prompts)
- Generator system prompts: identity only, e.g. "You are a careful engineer who thinks several steps ahead"
- Reviewer system prompts: punchy adversarial identity and checklist (no overlap with `MODE_RULES`)
- `_BANNED_PHRASES`: context-free filler only; words with legitimate uses are excluded or marked `(metaphorical)`
- Write mode: enforce asymmetry (vary section lengths, never mirror structure)
- Code mode: language-agnostic (the user specifies their stack)

## Style

- Readability is paramount: when in doubt, optimize for the reader, not the writer
- Always include type annotations: function signatures, return types, and local variables
- Python 3.12+
- Ruff line length: 120
- `frozen=True` on immutable dataclasses (including `CrossfireConfiguration`)
- Top-level imports only (no lazy imports inside method bodies)
- `...` for Protocol method stubs, `pass` for concrete no-op implementations
- All files must end with a trailing newline (for git)
- In Markdown files, each sentence goes on its own line (no mid-sentence line breaks); let word wrapping handle display

## Error handling

- Always use structured `log.log_*` functions; never use `log.get_logger().warning(...)` or similar direct calls
- Group related catch-all exceptions into named module-level tuples (e.g. `_RETRIABLE_ERRORS`) rather than inline multi-type except clauses
- Methods should return new state, not silently mutate instance attributes (especially on frozen dataclasses)
- `RunFailedError` for fatal orchestration failures; `RuntimeError` for recoverable per-model failures and configuration errors surfaced before or during a run (e.g. missing API keys)
- Exception messages inside `core/` are wrapped in `exclaim(...)` from `exclamations.py` for a consistent tone; CLI-layer `click.echo(..., err=True)` messages stay plain

## Testing

- Module docstrings follow the pattern `"""Tests for <topic>."""`
- Skip test-function docstrings when the test name already explains the intent
- Prefer `@pytest.mark.parametrize` over copy-pasted test functions with different inputs
- Use fixtures from `conftest.py` rather than hardcoding shared setup inline
- New shared test objects belong in `conftest.py` as fixtures, not as module-level constants in test files
- One assertion concept per test (multiple `assert` lines are fine if they test one logical thing)
- Dry-run determinism via SHA-256 hashing of all call parameters
- Test files may import private functions when needed (e.g. `_build_review_triage`)
- All tests + linting + typing must pass before merging
