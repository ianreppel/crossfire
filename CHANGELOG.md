# Changelog

## [0.3.0] - 2026-09-24

Crossfire now runs on **OpenCode Zen and OpenCode Go**, not only OpenRouter, and `opencode` is the default gateway.
One neutral model bench follows you across gateways, each mapping the shared slugs to its own wire IDs and prices.

### Added
- OpenCode Zen and Go as native gateways (via `--provider` or the `provider` key), sharing `OPENCODE_API_KEY`.
- One adapter for three wire protocols (chat, messages, responses), resolved per gateway and model family.
- Per-gateway pricing from models.dev, per-role temperature and reasoning effort, cache-token accounting.
- Provider-level refusal and truncation handling: a declining or budget-exhausted model is replaced, not retried.

### Changed
- **Breaking:** a new `provider` key and `[providers.*]` sections; `temperature_default` becomes per-role keys.
- **Breaking:** the default gateway is `opencode`, so set `provider = "openrouter"` to keep the old behaviour.
- Fable reviews, because it refuses the synthesis attribution block; Opus 5.5 synthesises in its place.

### Migration
Set `OPENCODE_API_KEY` (or `provider = "openrouter"`), drop `temperature_default`, and run `crossfire prices`.
