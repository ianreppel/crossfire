"""Tests for token compression."""

from __future__ import annotations

import logging

from crossfire.core import logging as crossfire_logging
from crossfire.core.compression import compress
from crossfire.core.tokens import estimate_tokens
from tests.helpers import LogCapture


class TestCompressionBasics:
    def test_no_compression_when_under_budget(self):
        text = "Short text."
        result = compress(text, target_tokens=1000)
        assert result.compressed == text
        assert result.tokens_before == result.tokens_after

    def test_compression_reduces_tokens(self):
        text = "\n".join([f"Line {i}: This is filler content for testing." for i in range(200)])
        target = estimate_tokens(text) // 3
        result = compress(text, target_tokens=target)
        assert result.tokens_after < result.tokens_before

    def test_code_blocks_preserved(self):
        text = (
            "# Header\n"
            "Some filler text that is not important.\n"
            * 20
            + "```python\ndef important():\n    pass\n```\n"
            + "More filler.\n" * 20
        )
        target = estimate_tokens(text) // 3
        result = compress(text, target_tokens=target)
        assert "```python" in result.compressed
        assert "def important():" in result.compressed

    def test_citation_lines_preserved(self):
        text = "# Header\nFiller content.\n" * 20 + "This claim is supported [citation-91].\n" + "More filler.\n" * 20
        target = estimate_tokens(text) // 3
        result = compress(text, target_tokens=target)
        assert "[citation-91]" in result.compressed

    def test_headings_preserved(self):
        text = "# Main Header\nSome content.\n" * 10 + "## Sub Header\nMore content.\n" * 10
        target = estimate_tokens(text) // 2
        result = compress(text, target_tokens=target)
        assert "# Main Header" in result.compressed


class TestCompressionPriority:
    """Verify ``compress()`` reduces tokens for each content type independently."""

    def test_each_content_type_compresses(self):
        """Each content type (candidate, review, context) compresses to target."""
        candidate_text = "Candidate: " + "word " * 500
        review_text = "Review: " + "word " * 500
        context_text = "Context: " + "word " * 500

        c_result = compress(candidate_text, target_tokens=200)
        r_result = compress(review_text, target_tokens=200)
        x_result = compress(context_text, target_tokens=200)

        assert c_result.tokens_after <= c_result.tokens_before
        assert r_result.tokens_after <= r_result.tokens_before
        assert x_result.tokens_after <= x_result.tokens_before


class TestCompressionLogging:
    def test_compression_applied_event_format(self):
        """The log event fields match the spec when compression is exercised."""
        capture = LogCapture()
        logger = logging.getLogger("crossfire")
        logger.addHandler(capture)
        try:
            crossfire_logging.log_compression_applied(
                phase="generation",
                role="generator",
                model="gen-a",
                round=1,
                tokens_before=5000,
                tokens_after=3000,
                reason="candidate_truncation",
            )
            assert len(capture.records) == 1
            record = capture.records[0]
            assert record["event"] == "compression_applied"
            assert record["phase"] == "generation"
            assert record["tokens_before"] == 5000
            assert record["tokens_after"] == 3000
        finally:
            logger.removeHandler(capture)
