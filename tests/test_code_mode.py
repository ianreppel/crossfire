"""Tests for code mode."""

from __future__ import annotations

import pytest

from crossfire.core.domain import (
    Candidate,
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    RunParameters,
    Task,
)
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.prompts import build_reviewer_prompt, parse_review_verdict


class TestCodeRubberDuckReviewing:
    def test_rubber_duck_is_loaded_for_bear(self):
        candidate = Candidate(
            text="def authenticate(user): return True",
            model="test-gen",
            round=1,
            index=0,
        )
        system, _user = build_reviewer_prompt(
            mode=Mode.CODE,
            instruction="Implement a user authentication system",
            candidate=candidate,
        )

        assert "LOGIC:" in system
        assert "TYPES:" in system
        assert "SECURITY:" in system
        assert "ERRORS:" in system
        assert "TESTS:" in system
        assert "STRUCTURE:" in system

        assert "SQL/XSS/command injection" in system
        assert "command injection" in system
        assert "hardcoded secrets" in system

        assert "off-by-one" in system.lower()
        assert "null" in system.lower() or "none" in system.lower()
        assert "race conditions" in system.lower()
        assert "resource leaks" in system.lower()

    def test_bobby_tables_would_not_survive(self):
        vulnerable_code = """def authenticate_user(username, password):
    import sqlite3
    conn = sqlite3.connect('users.db')
    # VULNERABILITY: SQL injection
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = conn.execute(query).fetchone()
    return result is not None

def process_file(filename):
    # VULNERABILITY: Path traversal
    with open(f"/uploads/{filename}", 'r') as f:
        return f.read()

# VULNERABILITY: Hardcoded secret
API_KEY = "sk-1234567890abcdef"
"""

        candidate = Candidate(
            text=vulnerable_code,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.CODE,
            instruction="Review this authentication code",
            candidate=candidate,
        )

        assert "SQL/XSS/command injection" in system
        assert "path traversal" in system.lower() or "../../" in system
        assert "hardcoded" in system.lower()

    def test_off_by_one_dumbshittery(self):
        buggy_code = """def binary_search(arr, target):
    left, right = 0, len(arr)  # BUG: Should be len(arr) - 1

    while left <= right:
        mid = (left + right) / 2  # BUG: Should use // for integer division

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def process_users(users):
    # BUG: No null/empty check
    for user in users:
        print(user.name)  # Could throw AttributeError

def calculate_average(numbers):
    return sum(numbers) / len(numbers)  # BUG: Division by zero if empty
"""

        candidate = Candidate(
            text=buggy_code,
            model="test-gen",
            round=1,
            index=0,
        )

        system, _user = build_reviewer_prompt(
            mode=Mode.CODE,
            instruction="Review this utility code",
            candidate=candidate,
        )

        assert "off-by-one" in system.lower()
        assert "edge cases" in system.lower()
        assert "unhandled edge cases" in system.lower()
        assert "implicit conversions" in system.lower()
        assert "null" in system.lower() or "none" in system.lower()

    @pytest.mark.asyncio
    async def test_whole_shebang_for_code_smells(self):
        problematic_instruction = (
            "Create a user registration system that stores passwords and handles file uploads. "
            "Include a search function that finds users by username."
        )

        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("gen-code",), context_window=32000),
            reviewers=ModelGroup(names=("rev-security", "rev-logic"), context_window=32000),
            synthesizer=ModelGroup(names=("synth-code",), context_window=32000),
            limits=LimitsConfiguration(),
        )

        parameters = RunParameters(
            mode=Mode.CODE,
            task=Task(instruction=problematic_instruction),
            num_generators=1,
            num_reviewers_per_candidate=2,
            num_rounds=2,
            dry_run=True,
        )

        orchestrator = Orchestrator(configuration, parameters)
        result = await orchestrator.run()

        assert len(result.strip()) > 100
        assert "Synthesized Output" in result or "python" in result.lower()

    def test_string_interpolation_of_sql_parameters_is_naughty(self):
        security_review = """This code has several critical security vulnerabilities:

1. SQL injection in the login function - user input is directly interpolated
2. Path traversal vulnerability allows reading arbitrary files
3. Hardcoded API key exposed in source code
4. No input validation on file uploads

STRENGTHS: code structure is clear, follows naming conventions
WEAKNESSES: SQL injection, path traversal, hardcoded secrets, no input validation
SEVERITY: material
"""

        verdict = parse_review_verdict(security_review)

        assert verdict.severity == "material"
        assert len(verdict.weaknesses) >= 3  # Should identify multiple security issues
        assert any("sql injection" in w.lower() for w in verdict.weaknesses)
        assert any("path traversal" in w.lower() or "hardcoded" in w.lower() for w in verdict.weaknesses)
