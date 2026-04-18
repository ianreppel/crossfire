"""Token estimation using tiktoken's cl100k_base, which is GPT-4's default tokenizer and a good approximation for most
LLMs' native tokenizers."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

SAFETY_MARGIN = 50
TOKEN_BUDGET_RATIO = 0.8


@lru_cache(maxsize=1)
def _load_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Counts the number of tokens, excluding any safety margin."""
    if not text.strip():
        return 1
    return len(_load_encoding().encode(text))


def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens, including the safety margin."""
    if not text:
        return 0
    return len(_load_encoding().encode(text)) + SAFETY_MARGIN


def compute_token_budget(context_window: int) -> int:
    """Calculates the usable token budget, i.e. 80 % of *context_window*"""
    return int(TOKEN_BUDGET_RATIO * context_window)


def fits_token_budget(
    input_tokens: int,
    max_output_tokens: int,
    context_window: int,
) -> bool:
    """Checks whether *input_tokens* + *max_output_tokens* fits within the token budget."""
    return input_tokens + max_output_tokens <= compute_token_budget(context_window)
