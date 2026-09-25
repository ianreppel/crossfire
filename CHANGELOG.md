# Changelog

## [Unreleased]

### Added
- OpenRouter requests carry `provider.data_collection = "deny"` and `provider.zdr = true`, so the gateway keeps them off providers that train on prompts and off endpoints that retain them.
- A data-policy guard refuses a run whose bench names a model the gateway documents as training on prompts, and logs at run start when the chosen models retain prompts.
- `deny_data_collection`, `require_zdr`, and `allow_training_models` per provider in `crossfire.toml`, all defaulting to the safe setting.

## [0.3.0] - 2026-09-24

Crossfire supports OpenCode Zen, OpenCode Go, OpenRouter, and Synthetic. Zen is the default gateway.
Model slugs stay neutral; each gateway maps them to its own wire IDs and prices.

### Added
- OpenCode Zen, OpenCode Go, and Synthetic gateways, selected with `--provider` or `provider` in TOML.
- An adapter for OpenAI chat, Anthropic messages, and OpenAI responses, selected by gateway and model family.
- Per-gateway pricing, per-role temperature and reasoning effort, and cache-token accounting.
- Refusals and output truncation trigger model replacement.

### Changed
- **Breaking:** `crossfire.toml` now uses `provider` and `[providers.*]`; `temperature_default` is replaced by per-role settings.
- **Breaking:** `opencode` is the default and requires `OPENCODE_API_KEY`; set `provider = "openrouter"` to retain OpenRouter.
- Fable reviews because it refuses the synthesis attribution block. Opus 5.5 synthesises instead.

### Migration
Set `OPENCODE_API_KEY` (or `provider = "openrouter"`), remove `temperature_default`, and run `crossfire prices`.
