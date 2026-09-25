"""Pricing cache and cost estimation for dry runs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from crossfire.core.domain import CostEstimate, CrossfireConfiguration, ModelGroup, RunParameters, strip_model_prefix
from crossfire.core.reviewers import assign_reviewers
from crossfire.core.tokens import estimate_tokens

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
PRICING_FILENAME = "pricing.json"

# models.dev reports prices per million tokens; pricing.json stores per-token values.
_PER_MILLION = 1_000_000
_MODELS_DEV_PROVIDER_IDS = ("opencode", "opencode-go", "synthetic")

# Default output tokens per call when the instruction has no explicit length signal.
# ~3,500 words: a substantial article, neither a tweet nor a novel.
_DEFAULT_GENERATOR_OUTPUT = 5000
_DEFAULT_REVIEWER_OUTPUT = 2000
_DEFAULT_SYNTHESIZER_OUTPUT = 5000
_DEFAULT_ENRICHER_OUTPUT = 2000

_TOKENS_PER_WORD = 1.4
_WORDS_PER_PAGE = 500

_WORD_COUNT_REGEX = re.compile(r"(\d[\d,]*)\s*[-\u2013]?\s*words?", re.IGNORECASE)
_PAGE_COUNT_REGEX = re.compile(r"(\d[\d,]*)\s*[-\u2013]?\s*pages?", re.IGNORECASE)


def _parse_pricing_entry(raw_pricing: Any) -> tuple[float, float]:
    """Extracts ``(prompt_price, completion_price)`` per token from an OpenRouter pricing object.

    Handles both flat objects and tiered arrays (uses the first tier).
    Returns ``(0.0, 0.0)`` when the pricing data is missing or unparseable.
    """
    entry: Any = raw_pricing
    if isinstance(entry, list):
        entry = entry[0] if entry else {}
    if not isinstance(entry, dict):
        return 0.0, 0.0
    try:
        prompt_price: float = float(entry.get("prompt", "0") or "0")
        completion_price: float = float(entry.get("completion", "0") or "0")
    except (ValueError, TypeError):
        return 0.0, 0.0
    return prompt_price, completion_price


def parse_api_response(data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Parses the OpenRouter ``/api/v1/models`` response into provider-qualified price entries."""
    models: dict[str, tuple[float, float]] = {}
    for entry in data.get("data", []):
        model_id: str = entry.get("id", "")
        if not model_id:
            continue
        models[f"openrouter::{model_id}"] = _parse_pricing_entry(entry.get("pricing"))
    return models


def parse_models_dev_response(data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Parses the models.dev catalog into provider-qualified price entries for Zen, Go and Synthetic."""
    models: dict[str, tuple[float, float]] = {}
    for provider_id in _MODELS_DEV_PROVIDER_IDS:
        provider_data = data.get(provider_id, {})
        for model_id, entry in provider_data.get("models", {}).items():
            cost = entry.get("cost") or {}
            models[f"{provider_id}::{model_id}"] = (
                float(cost.get("input", 0) or 0) / _PER_MILLION,
                float(cost.get("output", 0) or 0) / _PER_MILLION,
            )
    return models


def fetch_pricing() -> dict[str, Any]:
    """Fetches all model pricing from OpenRouter (synchronous)."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def fetch_models_dev_pricing() -> dict[str, Any]:
    """Fetches the models.dev catalog (synchronous), the source of OpenCode Zen/Go pricing."""
    with httpx.Client(timeout=60.0, headers={"User-Agent": "crossfire"}) as client:
        response = client.get(MODELS_DEV_URL)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def save_pricing(pricing: dict[str, tuple[float, float]], fetched_at: str, path: Path) -> None:
    """Writes the pricing cache to *path* as JSON."""
    payload: dict[str, Any] = {
        "fetched_at": fetched_at,
        "models": {
            model_id: {"prompt": prompt, "completion": completion}
            for model_id, (prompt, completion) in sorted(pricing.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_pricing(path: Path) -> tuple[dict[str, tuple[float, float]], str]:
    """Loads the pricing from *path*.

    Returns ``(models, fetched_at)``.
    Raises :class:`FileNotFoundError` if the file does not exist,
    :class:`ValueError` if the JSON is malformed.
    """
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    fetched_at: str = raw.get("fetched_at", "")
    models: dict[str, tuple[float, float]] = {}
    for model_id, prices in raw.get("models", {}).items():
        if isinstance(prices, dict):
            try:
                models[model_id] = (
                    float(prices.get("prompt", 0)),
                    float(prices.get("completion", 0)),
                )
            except (ValueError, TypeError):
                continue
    return models, fetched_at


def _pricing_keys_for(
    configuration: CrossfireConfiguration,
) -> Callable[[str], tuple[str, ...]]:
    """Builds a resolver returning the candidate pricing keys for a configured model name.

    Handles explicit ``provider:model`` overrides, neutral slugs, and a bare-slug fallback so older
    pricing caches without provider qualification still resolve.
    """
    provider_names = {provider.name for provider in configuration.providers}

    def keys_for(name: str) -> tuple[str, ...]:
        prefix, separator, rest = name.partition(":")
        if separator and prefix in provider_names:
            provider_name, neutral = prefix, rest
        else:
            provider_name, neutral = configuration.provider, name
        provider = next((p for p in configuration.providers if p.name == provider_name), None)
        wire_model_id = provider.resolve_wire_model_id(neutral) if provider else strip_model_prefix(name)
        return (f"{provider_name}::{wire_model_id}", wire_model_id, strip_model_prefix(name))

    return keys_for


def _price_for(
    name: str,
    pricing: dict[str, tuple[float, float]],
    keys_for: Callable[[str], tuple[str, ...]],
) -> tuple[float, float] | None:
    """Resolves the per-token price for *name*, trying each provider-qualified key in turn."""
    return next((pricing[key] for key in keys_for(name) if key in pricing), None)


def _selected_generator_models(group: ModelGroup, num_generators: int) -> list[str]:
    """The generator models a run actually calls: the first *num_generators* of the cheapest-first list."""
    if not group.names or num_generators <= 0:
        return []
    return [group.names[index % len(group.names)] for index in range(num_generators)]


def _selected_reviewer_models(
    group: ModelGroup,
    num_candidates: int,
    reviewers_per_candidate: int,
    round_num: int,
) -> list[str]:
    """The reviewer models a round actually calls, mirroring the runtime window selection."""
    if not group.names or num_candidates <= 0 or reviewers_per_candidate <= 0:
        return []
    assignments = assign_reviewers(
        reviewers=group.names,
        num_candidates=num_candidates,
        num_reviewers_per_candidate=reviewers_per_candidate,
        round_num=round_num,
        models_used_this_round=set(),
    )
    if assignments is None:
        return []
    return [model for models in assignments.values() for model in models]


def parse_length_hint(instruction: str) -> int | None:
    """Extracts an output token estimate from explicit word or page counts in the *instruction*.

    Looks for patterns like "1,200 words" or "10 pages".  This is a heuristic:
    the number may refer to the input rather than the desired output (e.g.
    "analyse this 200 page document").  When in doubt the estimate errs on the
    high side, which is acceptable for an upper-bound cost estimate.

    Returns the estimated token count, or ``None`` when no length signal is found.
    """
    match = _WORD_COUNT_REGEX.search(instruction)
    if match:
        words: int = int(match.group(1).replace(",", ""))
        return int(words * _TOKENS_PER_WORD)
    match = _PAGE_COUNT_REGEX.search(instruction)
    if match:
        pages: int = int(match.group(1).replace(",", ""))
        return int(pages * _WORDS_PER_PAGE * _TOKENS_PER_WORD)
    return None


def estimate_cost(
    configuration: CrossfireConfiguration,
    parameters: RunParameters,
    pricing: dict[str, tuple[float, float]],
    fetched_at: str,
) -> CostEstimate:
    """Estimates the cost of a run by following the same model selection the runtime uses.

    Prices the models each round actually picks (the first N generators, the reviewer window, the rotating synthesizer)
    rather than averaging each group, so the estimate tracks the run instead of an abstract pool.
    """
    missing: list[str] = []
    keys_for = _pricing_keys_for(configuration)

    def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
        entry = _price_for(model, pricing, keys_for)
        if entry is None:
            missing.append(model)
            return 0.0
        input_price, output_price = entry
        return input_tokens * input_price + output_tokens * output_price

    instruction_tokens: int = estimate_tokens(parameters.task.instruction)
    context_tokens: int = estimate_tokens(parameters.task.context) if parameters.task.context else 0

    hint: int | None = parse_length_hint(parameters.task.instruction)
    generator_output: int = min(hint or _DEFAULT_GENERATOR_OUTPUT, configuration.generators.max_output_tokens)
    reviewer_output: int = min(_DEFAULT_REVIEWER_OUTPUT, configuration.reviewers.max_output_tokens)
    synthesizer_output: int = min(hint or _DEFAULT_SYNTHESIZER_OUTPUT, configuration.synthesizer.max_output_tokens)
    enricher_output: int = min(_DEFAULT_ENRICHER_OUTPUT, configuration.enricher.max_output_tokens)

    num_generators: int = parameters.num_generators
    num_reviewers: int = parameters.num_reviewers_per_candidate
    num_rounds: int = parameters.num_rounds

    enrichment_active: bool = parameters.enrich and bool(configuration.enricher.names)

    # After enrichment the enriched instruction replaces the original in all
    # downstream phases (generation, review, synthesis).
    effective_instruction_tokens: int = enricher_output if enrichment_active else instruction_tokens

    total: float = 0.0

    # -- enrichment: real input tokens --
    if enrichment_active:
        total += cost_of(configuration.enricher.names[0], instruction_tokens + context_tokens, enricher_output)

    generator_models: list[str] = _selected_generator_models(configuration.generators, num_generators)
    generation_input_round_1: int = effective_instruction_tokens + context_tokens
    generation_input_round_n: int = generation_input_round_1 + synthesizer_output
    reviewer_input: int = effective_instruction_tokens + generator_output
    synthesis_input: int = (
        effective_instruction_tokens
        + num_generators * generator_output
        + num_generators * num_reviewers * reviewer_output
    )

    for round_num in range(1, num_rounds + 1):
        generation_input: int = generation_input_round_1 if round_num == 1 else generation_input_round_n
        for model in generator_models:
            total += cost_of(model, generation_input, generator_output)

        for model in _selected_reviewer_models(configuration.reviewers, num_generators, num_reviewers, round_num):
            total += cost_of(model, reviewer_input, reviewer_output)

        if configuration.synthesizer.names:
            synthesizer_model: str = configuration.synthesizer.names[
                (round_num - 1) % len(configuration.synthesizer.names)
            ]
            total += cost_of(synthesizer_model, synthesis_input, synthesizer_output)

    unique_missing: tuple[str, ...] = tuple(dict.fromkeys(missing))
    return CostEstimate(
        total_usd=total,
        missing_models=unique_missing,
        fetched_at=fetched_at,
    )
