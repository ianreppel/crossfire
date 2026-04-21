"""Tests for review triage construction and parsing of synthesis decision."""

from __future__ import annotations

from crossfire.core.domain import Candidate, CandidateDecision, Review
from crossfire.core.prompts import _build_review_triage, parse_review_verdict, parse_synthesis_decision


class TestParseReviewVerdict:
    def test_extracts_strengths_and_weaknesses(self):
        text = (
            "Some review prose.\nSTRENGTHS: clear structure, good citations\nWEAKNESSES: weak conclusion, missing data"
        )
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["clear structure", "good citations"]
        assert verdict.weaknesses == ["weak conclusion", "missing data"]

    def test_case_insensitive(self):
        text = "strengths: item a\nweaknesses: item b"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["item a"]
        assert verdict.weaknesses == ["item b"]

    def test_empty_when_no_structured_lines(self):
        text = "Just a plain review without any structure."
        verdict = parse_review_verdict(text)
        assert verdict.strengths == []
        assert verdict.weaknesses == []

    def test_strengths_only(self):
        text = "Review.\nSTRENGTHS: solid analysis, novel approach"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["solid analysis", "novel approach"]
        assert verdict.weaknesses == []

    def test_weaknesses_only(self):
        text = "Review.\nWEAKNESSES: poor formatting"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == []
        assert verdict.weaknesses == ["poor formatting"]

    def test_strips_whitespace(self):
        text = "STRENGTHS:  spaced out ,  extra spaces  "
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["spaced out", "extra spaces"]

    def test_commas_inside_parentheses_preserved(self):
        text = "STRENGTHS: good references (Smith, 2024), clear structure"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["good references (Smith, 2024)", "clear structure"]

    def test_commas_inside_brackets_preserved(self):
        text = "WEAKNESSES: missing citations [1, 2, 3], poor formatting"
        verdict = parse_review_verdict(text)
        assert verdict.weaknesses == ["missing citations [1, 2, 3]", "poor formatting"]

    def test_extracts_severity(self):
        text = "STRENGTHS: good\nWEAKNESSES: minor formatting\nSEVERITY: nitpick"
        verdict = parse_review_verdict(text)
        assert verdict.severity == "nitpick"

    def test_severity_material(self):
        text = "WEAKNESSES: broken logic\nSEVERITY: material"
        verdict = parse_review_verdict(text)
        assert verdict.severity == "material"
        assert verdict.weaknesses == ["broken logic"]

    def test_severity_none(self):
        text = "STRENGTHS: perfect\nWEAKNESSES: none\nSEVERITY: none"
        verdict = parse_review_verdict(text)
        assert verdict.severity == "none"

    def test_severity_missing(self):
        text = "STRENGTHS: good\nWEAKNESSES: bad"
        verdict = parse_review_verdict(text)
        assert verdict.severity == ""

    def test_bold_markdown_stripped(self):
        text = "**STRENGTHS:** clear structure\n**WEAKNESSES:** weak ending\n**SEVERITY:** material"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["clear structure"]
        assert verdict.weaknesses == ["weak ending"]
        assert verdict.severity == "material"

    def test_italic_markdown_stripped(self):
        text = "*STRENGTHS:* item a\n*WEAKNESSES:* item b\n*SEVERITY:* nitpick"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["item a"]
        assert verdict.weaknesses == ["item b"]
        assert verdict.severity == "nitpick"

    def test_mixed_markdown_stripped(self):
        text = "___STRENGTHS:___ good\n**WEAKNESSES:** bad\n***SEVERITY:*** material"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["good"]
        assert verdict.weaknesses == ["bad"]
        assert verdict.severity == "material"

    def test_heading_prefix_stripped(self):
        text = "## STRENGTHS: clear logic\n## WEAKNESSES: vague intro\n## SEVERITY: material"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["clear logic"]
        assert verdict.weaknesses == ["vague intro"]
        assert verdict.severity == "material"

    def test_heading_with_bold_stripped(self):
        text = "### **Strengths:** solid\n### **Weaknesses:** shaky\n### **Severity:** nitpick"
        verdict = parse_review_verdict(text)
        assert verdict.strengths == ["solid"]
        assert verdict.weaknesses == ["shaky"]
        assert verdict.severity == "nitpick"

    def test_deep_heading_stripped(self):
        text = "###### Severity: none"
        verdict = parse_review_verdict(text)
        assert verdict.severity == "none"


class TestBuildReviewTriage:
    def test_builds_per_candidate_brief(self):
        candidates = [
            Candidate(text="C0", model="g0", round=1, index=0),
            Candidate(text="C1", model="g1", round=1, index=1),
        ]
        reviews = [
            Review(
                text="STRENGTHS: good intro\nWEAKNESSES: bad ending",
                model="r0",
                round=1,
                candidate_index=0,
            ),
            Review(
                text="STRENGTHS: novel idea\nWEAKNESSES: no citations",
                model="r1",
                round=1,
                candidate_index=1,
            ),
        ]
        brief = _build_review_triage(candidates, reviews)
        assert "REVIEW TRIAGE" in brief
        assert "Candidate 0:" in brief
        assert "KEEP: good intro" in brief
        assert "DISCARD: bad ending" in brief
        assert "Candidate 1:" in brief
        assert "KEEP: novel idea" in brief
        assert "DISCARD: no citations" in brief

    def test_deduplicates_across_reviewers(self):
        candidates = [
            Candidate(text="C0", model="g0", round=1, index=0),
        ]
        reviews = [
            Review(
                text="STRENGTHS: clear structure\nWEAKNESSES: weak ending",
                model="r0",
                round=1,
                candidate_index=0,
            ),
            Review(
                text="STRENGTHS: clear structure, good data\nWEAKNESSES: weak ending",
                model="r1",
                round=1,
                candidate_index=0,
            ),
        ]
        brief = _build_review_triage(candidates, reviews)
        assert brief.count("clear structure") == 1
        assert brief.count("weak ending") == 1
        assert "good data" in brief

    def test_returns_empty_without_signals(self):
        candidates = [
            Candidate(text="C0", model="g0", round=1, index=0),
        ]
        reviews = [
            Review(
                text="Unstructured review text.",
                model="r0",
                round=1,
                candidate_index=0,
            ),
        ]
        brief = _build_review_triage(candidates, reviews)
        assert brief == ""

    def test_partial_signals_still_produce_brief(self):
        candidates = [
            Candidate(text="C0", model="g0", round=1, index=0),
            Candidate(text="C1", model="g1", round=1, index=1),
        ]
        reviews = [
            Review(
                text="STRENGTHS: decent intro",
                model="r0",
                round=1,
                candidate_index=0,
            ),
            Review(
                text="No structured output here.",
                model="r1",
                round=1,
                candidate_index=1,
            ),
        ]
        brief = _build_review_triage(candidates, reviews)
        assert "REVIEW TRIAGE" in brief
        assert "Candidate 0:" in brief
        assert "Candidate 1:" not in brief


class TestParseSynthesisDecision:
    def test_parses_attributions_format(self):
        text = (
            '{"crossfire_synthesis": {"attributions": ['
            '{"index": 0, "kept": ["intro", "refs"], "discarded": ["conclusion"]}, '
            '{"index": 1, "kept": [], "discarded": ["everything"]}], '
            '"notes": "Mixed and matched"}}\n'
            "## Output here"
        )
        decisions, notes = parse_synthesis_decision(text)
        assert len(decisions) == 2
        assert decisions[0] == CandidateDecision(index=0, kept=["intro", "refs"], discarded=["conclusion"])
        assert decisions[1] == CandidateDecision(index=1, kept=[], discarded=["everything"])
        assert notes == "Mixed and matched"

    def test_ignores_legacy_format(self):
        """Old selected/discarded format is no longer supported."""
        text = (
            '{"crossfire_synthesis": {"selected": [0, 2], "discarded": [1], "notes": "Legacy output"}}\n## Output here'
        )
        decisions, notes = parse_synthesis_decision(text)
        assert decisions == []
        assert notes == "Legacy output"

    def test_returns_empty_on_no_json(self):
        text = "Just plain synthesis output with no JSON."
        decisions, notes = parse_synthesis_decision(text)
        assert decisions == []
        assert notes == ""

    def test_handles_malformed_json(self):
        text = '{"crossfire_synthesis": broken}\n## Output'
        decisions, notes = parse_synthesis_decision(text)
        assert decisions == []
        assert notes == ""
