"""Web search (Tavily) with a deterministic dry-run simulation, because LLMs may request a search."""

from __future__ import annotations

import json
import os
import re

import httpx

from crossfire.core import logging as log
from crossfire.core.domain import Role
from crossfire.core.exclamations import exclaim
from crossfire.core.simulation import simulate_search

# Backup in case the LLM emits malformed JSON and json.loads() cannot handle it
SEARCH_REQUEST_REGEX = re.compile(r'\{"crossfire_search"\s*:\s*\{.*?"query"\s*:\s*"(.+?)"\s*\}\s*\}')


def get_search_api_key() -> str:
    """Resolves the Tavily search API key from the environment.

    Raises :class:`RuntimeError` if the key is missing, so callers can fail fast rather
    than burning tokens in earlier phases only to discover the problem at search time.
    """
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError(
            exclaim(
                "Search is enabled but you did not set TAVILY_API_KEY. "
                "Export it or disable search in crossfire.toml."
            )
        )
    return key


def extract_search_request(llm_output: str) -> str | None:
    """Parse a search request from the last non-empty line of the LLM's output."""
    lines = [line.strip() for line in llm_output.strip().split("\n") if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    try:
        parsed = json.loads(last_line)
        if isinstance(parsed, dict) and "crossfire_search" in parsed:
            query = parsed["crossfire_search"].get("query")
            return query if isinstance(query, str) and query else None
    except (json.JSONDecodeError, TypeError):
        pass

    match = SEARCH_REQUEST_REGEX.search(last_line)
    return match.group(1) if match else None


def strip_search_request(llm_output: str) -> str:
    lines = llm_output.rstrip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return llm_output

    last = lines[-1].strip()
    try:
        parsed = json.loads(last)
        if isinstance(parsed, dict) and "crossfire_search" in parsed:
            lines.pop()
    except (json.JSONDecodeError, TypeError):
        if SEARCH_REQUEST_REGEX.search(last):
            lines.pop()

    return "\n".join(lines)


async def query_tavily(
    query: str,
    *,
    max_results: int = 5,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> str:
    api_key = get_search_api_key()

    owned_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owned_client:
            await client.aclose()

    results = data.get("results", [])
    lines: list[str] = []
    for result in results[:max_results]:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        lines.append(f"- {title} ({url}): {content}")

    return "\n".join(lines) if lines else "(no results)"


async def perform_search(
    query: str,
    *,
    dry_run: bool,
    instruction: str,
    mode: str,
    role: Role,
    model: str,
    round_num: int,
    client: httpx.AsyncClient | None = None,
    search_timeout: float = 30.0,
) -> str:
    """Unified entry point for a real Tavily API call or a simulated dry run."""
    if dry_run:
        return simulate_search(
            instruction=instruction,
            mode=mode,
            role=role,
            model=model,
            round_num=round_num,
            query=query,
        )

    try:
        return await query_tavily(query, client=client, timeout=search_timeout)
    except (
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.ConnectError,
        RuntimeError,
        ValueError,
    ) as exception:
        log.log_search_failure(
            round=round_num,
            role=role,
            model=model,
            query=query,
            error=str(exception),
        )
        return ""
