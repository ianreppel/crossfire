# Changelog

## [Unreleased]

### Added
- OpenRouter requests carry `provider.data_collection = "deny"` and `provider.zdr = true`, so the gateway keeps them off providers that train on prompts and off endpoints that retain them.
- A data-policy guard refuses a run whose bench names a model the gateway documents as training on prompts, and logs at run start when the chosen models retain prompts.
- `deny_data_collection`, `require_zdr`, and `allow_training_models` per provider in `crossfire.toml`, all defaulting to the safe setting.

### Fixed
- Refreshed the policy tables from the vendor pages: Zen's training list matches its documented exceptions again (fixing a Nemotron wire ID that could never match), and Go's lists gain Iceman Contributor Free, Claude Haiku 5.5, and Beam Free while dropping the departed Grok 4.6.
- When OpenCode Zen is active, Crossfire refuses runs that use Iceman Free because Zen uses Iceman Free input data for model improvement, as stated in the [OpenCode Zen privacy docs](https://opencode.ai/docs/zen/#privacy).
- When OpenCode Zen is active, Crossfire refuses runs that use Inkling 2 Small Free because Zen may use Inkling 2 Small Free input data for model training, as stated in the [OpenCode Zen privacy docs](https://opencode.ai/docs/zen/#privacy).
- OpenCode Zen policy drift tests compare Crossfire's complete list of models whose input data OpenCode Zen may use for training or model improvement with the [OpenCode Zen privacy docs](https://opencode.ai/docs/zen/#privacy).
- OpenCode Go policy drift tests compare Crossfire's complete training and retention tables with the [OpenCode Go privacy docs](https://opencode.ai/docs/go/#privacy).
- OpenCode Zen protocol drift tests compare Crossfire's protocol mapping with the protocol documented for each endpoint in the [OpenCode Zen docs](https://opencode.ai/docs/zen/).
- A module-scoped fixture skips all network drift tests if any required documentation page or model-list endpoint for OpenCode Zen or OpenCode Go is unreachable.
- Validation requires each provider's base URL to start with `http://` or `https://`.
- Crossfire restricts a model with a configured provider pin to runs that use the named gateway.
- Prefixes that do not match a configured provider name remain part of the model ID.
- OpenCode model families follow the vendor's documented wire protocols: minimax and qwen3.8-max route to chat, and the whole gpt-5.x family to responses.
- `model_dropped` events and the progress display name the billed wire model ID when the gateway substitutes the neutral slug.
- Crossfire checks refusal and truncation statuses before returning response text.
- Crossfire removes the provider pin for the selected gateway before alias lookup, so protocol selection and cost estimates use the resolved wire ID.
- Synthesis cleanup removes standalone JSON objects, each on its own line, with a top-level `crossfire_synthesis` key outside code fences.
  It also removes entire unlabelled or `json`-labelled fenced blocks that contain an object with that top-level key.
- Synthesis cleanup preserves other code fences and text outside removed blocks, including prose that mentions `crossfire_synthesis`.
- If the usage report includes cache reads or writes and the provider does not report a cost, Crossfire lists the model in `unpriced_models` and omits the call's cost estimate.
- Removed the dead `fatal_on_overflow` parameter.
- The non-gating drift probe reports mismatches but exits with status 0.
- Removed the dead `models_used_this_round` parameter.
  Round rotation already varies the model set each round.

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
