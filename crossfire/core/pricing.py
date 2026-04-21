"""Pricing cache and cost estimation for dry runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from crossfire.core.domain import CostEstimate, CrossfireConfiguration, ModelGroup, RunParameters
from crossfire.core.openrouter import strip_model_prefix
from crossfire.core.tokens import estimate_tokens

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICING_FILENAME = "pricing.json"

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
    """Parses the OpenRouter ``/api/v1/models`` response into a ``{model_id: (prompt, completion)}`` map."""
    models: dict[str, tuple[float, float]] = {}
    for entry in data.get("data", []):
        model_id: str = entry.get("id", "")
        if not model_id:
            continue
        models[model_id] = _parse_pricing_entry(entry.get("pricing"))
    return models


def fetch_pricing() -> dict[str, Any]:
    """Fetches all model pricing from OpenRouter (synchronous)."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(OPENROUTER_MODELS_URL)
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


def _average_group_price(
    group: ModelGroup,
    pricing: dict[str, tuple[float, float]],
    missing: list[str],
) -> tuple[float, float]:
    """Computes the average per-token price across a model group.

    Returns ``(average_price_in, average_price_out)``.
    Models missing from *pricing* are appended to *missing*.
    """
    total_price_in: float = 0.0
    total_price_out: float = 0.0
    found: int = 0

    for name in group.names:
        api_id: str = strip_model_prefix(name)
        if api_id not in pricing:
            missing.append(name)
            continue
        prompt_price, completion_price = pricing[api_id]
        total_price_in += prompt_price
        total_price_out += completion_price
        found += 1

    if found == 0:
        return 0.0, 0.0
    return total_price_in / found, total_price_out / found


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
    """Estimates the cost of the run described by *configuration* and *parameters*."""
    missing: list[str] = []
    instruction_tokens: int = estimate_tokens(parameters.task.instruction)
    context_tokens: int = estimate_tokens(parameters.task.context) if parameters.task.context else 0

    enricher_price_in, enricher_price_out = _average_group_price(configuration.enricher, pricing, missing)
    generator_price_in, generator_price_out = _average_group_price(configuration.generators, pricing, missing)
    reviewer_price_in, reviewer_price_out = _average_group_price(configuration.reviewers, pricing, missing)
    synthesizer_price_in, synthesizer_price_out = _average_group_price(configuration.synthesizer, pricing, missing)

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
        enrichment_input: int = instruction_tokens + context_tokens
        total += enrichment_input * enricher_price_in + enricher_output * enricher_price_out

    # -- generation: real input for round 1, estimated for rounds 2+ --
    generation_input_round_1: int = effective_instruction_tokens + context_tokens
    generation_input_round_n: int = generation_input_round_1 + synthesizer_output

    total += num_generators * (generation_input_round_1 * generator_price_in + generator_output * generator_price_out)
    if num_rounds > 1:
        total += (
            num_generators
            * (num_rounds - 1)
            * (generation_input_round_n * generator_price_in + generator_output * generator_price_out)
        )

    # -- review: instruction (enriched if applicable) + estimated candidate output --
    reviewer_input: int = effective_instruction_tokens + generator_output
    total += (
        num_generators
        * num_reviewers
        * num_rounds
        * (reviewer_input * reviewer_price_in + reviewer_output * reviewer_price_out)
    )

    # -- synthesis: instruction (enriched if applicable) + estimated candidates and reviews --
    synthesis_input: int = (
        effective_instruction_tokens
        + num_generators * generator_output
        + num_generators * num_reviewers * reviewer_output
    )
    total += num_rounds * (synthesis_input * synthesizer_price_in + synthesizer_output * synthesizer_price_out)

    unique_missing: tuple[str, ...] = tuple(dict.fromkeys(missing))
    return CostEstimate(
        total_usd=total,
        missing_models=unique_missing,
        fetched_at=fetched_at,
    )
