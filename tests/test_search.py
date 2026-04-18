"""Tests for search request extraction, stripping (safe for work!), and API key resolution."""

from __future__ import annotations

import pytest

from crossfire.core.search import (
    extract_search_request,
    get_search_api_key,
    strip_search_request,
)


class TestExtractSearchRequest:
    def test_valid_json_last_line(self):
        output = 'Some content\n{"crossfire_search": {"query": "quantum computing"}}'
        assert extract_search_request(output) == "quantum computing"

    def test_no_search_request(self):
        output = "Just regular content\nNo search here."
        assert extract_search_request(output) is None

    def test_search_not_on_last_line_ignored(self):
        output = '{"crossfire_search": {"query": "test"}}\nMore content after.'
        assert extract_search_request(output) is None

    def test_empty_query_returns_none(self):
        output = '{"crossfire_search": {"query": ""}}'
        assert extract_search_request(output) is None

    def test_trailing_whitespace_handled(self):
        output = 'Content\n{"crossfire_search": {"query": "test"}}  \n  \n'
        assert extract_search_request(output) == "test"


class TestStripSearchRequest:
    def test_strips_search_line(self):
        output = 'Line 1\nLine 2\n{"crossfire_search": {"query": "test"}}'
        result = strip_search_request(output)
        assert "crossfire_search" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_no_strip_when_no_search(self):
        output = "Line 1\nLine 2"
        assert strip_search_request(output) == output


class TestGetSearchApiKey:
    def test_returns_key_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        assert get_search_api_key() == "tvly-xxx"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            get_search_api_key()
