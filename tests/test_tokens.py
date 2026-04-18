"""Tests for token estimation utilities."""

from __future__ import annotations

from crossfire.core.tokens import SAFETY_MARGIN, compute_token_budget, count_tokens, estimate_tokens, fits_token_budget


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_nonempty_string_positive(self):
        assert estimate_tokens("hello world") > 0

    def test_longer_text_more_tokens(self):
        short_estimate = estimate_tokens("hello")
        long_estimate = estimate_tokens("hello " * 100)
        assert long_estimate > short_estimate

    def test_includes_safety_margin(self):
        tokens = estimate_tokens("test")
        assert tokens >= SAFETY_MARGIN

    def test_count_raw_tokens_no_margin(self):
        assert count_tokens("hello") < estimate_tokens("hello")


class TestBudget:
    def test_compute_token_budget_is_80_percent(self):
        assert compute_token_budget(10000) == 8000

    def test_fits_budget_true(self):
        assert fits_token_budget(1000, 2000, 10000) is True

    def test_fits_budget_false(self):
        assert fits_token_budget(5000, 4000, 10000) is False

    def test_fits_budget_exact(self):
        assert fits_token_budget(4000, 4000, 10000) is True
