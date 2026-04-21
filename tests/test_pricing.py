"""Tests for pricing cache and cost estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossfire.core.domain import (
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.pricing import (
    _average_group_price,
    _parse_pricing_entry,
    estimate_cost,
    load_pricing,
    parse_api_response,
    parse_length_hint,
    save_pricing,
)


class TestParsePricingEntry:
    def test_flat_pricing(self):
        prompt, completion = _parse_pricing_entry({"prompt": "0.000003", "completion": "0.000015"})
        assert prompt == pytest.approx(0.000003)
        assert completion == pytest.approx(0.000015)

    def test_tiered_pricing_uses_first_tier(self):
        tiered = [
            {"prompt": "0.000001", "completion": "0.000005"},
            {"prompt": "0.000002", "completion": "0.000010"},
        ]
        prompt, completion = _parse_pricing_entry(tiered)
        assert prompt == pytest.approx(0.000001)
        assert completion == pytest.approx(0.000005)

    def test_free_model(self):
        prompt, completion = _parse_pricing_entry({"prompt": "0", "completion": "0"})
        assert prompt == 0.0
        assert completion == 0.0

    def test_missing_fields(self):
        prompt, completion = _parse_pricing_entry({})
        assert prompt == 0.0
        assert completion == 0.0

    def test_none_pricing(self):
        prompt, completion = _parse_pricing_entry(None)
        assert prompt == 0.0
        assert completion == 0.0

    def test_empty_tiered_list(self):
        prompt, completion = _parse_pricing_entry([])
        assert prompt == 0.0
        assert completion == 0.0

    def test_null_string_values(self):
        prompt, completion = _parse_pricing_entry({"prompt": None, "completion": None})
        assert prompt == 0.0
        assert completion == 0.0


class TestParseApiResponse:
    def test_parses_data_array(self):
        raw = {
            "data": [
                {"id": "vendor/model-a", "pricing": {"prompt": "0.000002", "completion": "0.000008"}},
                {"id": "vendor/model-b", "pricing": {"prompt": "0", "completion": "0"}},
            ]
        }
        result = parse_api_response(raw)
        assert len(result) == 2
        assert result["vendor/model-a"] == pytest.approx((0.000002, 0.000008))
        assert result["vendor/model-b"] == (0.0, 0.0)

    def test_empty_data(self):
        assert parse_api_response({"data": []}) == {}

    def test_missing_data_key(self):
        assert parse_api_response({}) == {}

    def test_skips_entries_without_id(self):
        raw = {"data": [{"pricing": {"prompt": "0.001", "completion": "0.002"}}]}
        assert parse_api_response(raw) == {}


class TestPricingRoundTrip:
    def test_save_and_load(self, tmp_path: Path):
        pricing = {
            "vendor/model-a": (0.000002, 0.000008),
            "vendor/model-b": (0.0, 0.0),
        }
        fetched_at = "2026-04-20T14:30:00Z"
        path = tmp_path / "pricing.json"

        save_pricing(pricing, fetched_at, path)

        loaded, loaded_at = load_pricing(path)
        assert loaded_at == fetched_at
        assert loaded["vendor/model-a"] == pytest.approx((0.000002, 0.000008))
        assert loaded["vendor/model-b"] == (0.0, 0.0)

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_pricing(tmp_path / "nonexistent.json")

    def test_load_malformed_json(self, tmp_path: Path):
        path = tmp_path / "pricing.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_pricing(path)

    def test_saved_file_is_valid_json(self, tmp_path: Path):
        pricing = {"vendor/model-x": (0.001, 0.002)}
        path = tmp_path / "pricing.json"
        save_pricing(pricing, "2026-01-01T00:00:00Z", path)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert "fetched_at" in parsed
        assert "models" in parsed
        assert parsed["models"]["vendor/model-x"]["prompt"] == 0.001


class TestAverageGroupPrice:
    def test_averages_prices_across_group(self):
        group = ModelGroup(
            names=("openrouter:vendor/cheap", "openrouter:vendor/expensive"),
            context_window=16000,
            max_output_tokens=4096,
        )
        pricing = {
            "vendor/cheap": (0.000001, 0.000010),
            "vendor/expensive": (0.000005, 0.000002),
        }
        missing: list[str] = []
        price_in, price_out = _average_group_price(group, pricing, missing)
        assert price_in == pytest.approx(0.000003)
        assert price_out == pytest.approx(0.000006)
        assert missing == []

    def test_missing_model_tracked(self):
        group = ModelGroup(
            names=("openrouter:vendor/known", "openrouter:vendor/unknown"),
            context_window=16000,
            max_output_tokens=4096,
        )
        pricing = {"vendor/known": (0.001, 0.002)}
        missing: list[str] = []
        _average_group_price(group, pricing, missing)
        assert missing == ["openrouter:vendor/unknown"]

    def test_all_missing_returns_zeros(self):
        group = ModelGroup(
            names=("openrouter:vendor/unknown",),
            context_window=16000,
            max_output_tokens=4096,
        )
        missing: list[str] = []
        price_in, price_out = _average_group_price(group, {}, missing)
        assert price_in == 0.0
        assert price_out == 0.0


class TestParseLengthHint:
    def test_word_count(self):
        assert parse_length_hint("Write a 1,200 words essay on picking your nose in public") == int(1200 * 1.4)

    def test_word_count_hyphenated(self):
        assert parse_length_hint("Write a 15,000-word paper on farting the national anthem") == int(15000 * 1.4)

    def test_word_range_uses_upper_bound(self):
        result = parse_length_hint("Roughly 900\u20131,200 words")
        assert result == int(1200 * 1.4)

    def test_page_count(self):
        assert parse_length_hint("Write a 10 page report on the history of popping balloons") == int(10 * 500 * 1.4)

    def test_input_description_triggers_false_positive(self):
        result = parse_length_hint("Analyse this 200 pages document")
        assert result is not None

    def test_incidental_word_does_not_match(self):
        assert parse_length_hint("Summarize the key words in this text") is None

    def test_no_hint(self):
        assert parse_length_hint("Compare designs for underpants in space") is None

    def test_no_hint_on_empty(self):
        assert parse_length_hint("") is None


@pytest.fixture()
def _estimation_configuration() -> CrossfireConfiguration:
    return CrossfireConfiguration(
        enricher=ModelGroup(
            names=("openrouter:vendor/enricher",),
            context_window=128000,
            max_output_tokens=4096,
        ),
        generators=ModelGroup(
            names=("openrouter:vendor/gen-a", "openrouter:vendor/gen-b"),
            context_window=16000,
            max_output_tokens=12000,
        ),
        reviewers=ModelGroup(
            names=("openrouter:vendor/rev-a", "openrouter:vendor/rev-b", "openrouter:vendor/rev-c"),
            context_window=16000,
            max_output_tokens=8000,
        ),
        synthesizer=ModelGroup(
            names=("openrouter:vendor/synth-a",),
            context_window=200000,
            max_output_tokens=32000,
        ),
        search=SearchConfiguration(enabled=False),
        limits=LimitsConfiguration(),
    )


@pytest.fixture()
def _estimation_pricing() -> dict[str, tuple[float, float]]:
    return {
        "vendor/enricher": (0.0000004, 0.0000016),
        "vendor/gen-a": (0.000003, 0.000015),
        "vendor/gen-b": (0.0000003, 0.0000004),
        "vendor/rev-a": (0.000001, 0.000005),
        "vendor/rev-b": (0.0000004, 0.000002),
        "vendor/rev-c": (0.0000003, 0.000001),
        "vendor/synth-a": (0.000015, 0.000075),
    }


class TestEstimateCost:
    def test_positive_total(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test instruction"),
            num_generators=2,
            num_reviewers_per_candidate=1,
            num_rounds=3,
            dry_run=True,
            enrich=True,
        )
        estimate = estimate_cost(_estimation_configuration, parameters, _estimation_pricing, "2026-04-20T00:00:00Z")
        assert estimate.total_usd > 0
        assert estimate.missing_models == ()
        assert estimate.fetched_at == "2026-04-20T00:00:00Z"

    def test_no_enrichment_uses_real_input(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters_with = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test instruction"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=True,
        )
        parameters_without = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test instruction"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        estimate_with = estimate_cost(_estimation_configuration, parameters_with, _estimation_pricing, "")
        estimate_without = estimate_cost(_estimation_configuration, parameters_without, _estimation_pricing, "")
        assert estimate_with.total_usd > estimate_without.total_usd

    def test_more_rounds_costs_more(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters_1 = RunParameters(
            mode=Mode.CODE,
            task=Task(instruction="Build something"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        parameters_5 = RunParameters(
            mode=Mode.CODE,
            task=Task(instruction="Build something"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=5,
            dry_run=True,
            enrich=False,
        )
        estimate_1 = estimate_cost(_estimation_configuration, parameters_1, _estimation_pricing, "")
        estimate_5 = estimate_cost(_estimation_configuration, parameters_5, _estimation_pricing, "")
        assert estimate_5.total_usd > estimate_1.total_usd

    def test_missing_models_flagged(
        self,
        _estimation_configuration: CrossfireConfiguration,
    ):
        partial_pricing = {"vendor/gen-a": (0.001, 0.002)}
        parameters = RunParameters(
            mode=Mode.EDIT,
            task=Task(instruction="Edit something"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        estimate = estimate_cost(_estimation_configuration, parameters, partial_pricing, "")
        assert len(estimate.missing_models) > 0

    def test_zero_reviewers(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test instruction"),
            num_generators=1,
            num_reviewers_per_candidate=0,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        estimate = estimate_cost(_estimation_configuration, parameters, _estimation_pricing, "")
        assert estimate.total_usd > 0

    def test_large_context_increases_estimate(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters_no_context = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Summarize"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        parameters_with_context = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Summarize", context="x " * 20000),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        estimate_small = estimate_cost(_estimation_configuration, parameters_no_context, _estimation_pricing, "")
        estimate_large = estimate_cost(_estimation_configuration, parameters_with_context, _estimation_pricing, "")
        assert estimate_large.total_usd > estimate_small.total_usd

    def test_word_count_hint_increases_estimate(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters_default = RunParameters(
            mode=Mode.WRITE,
            task=Task(instruction="Write an essay"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        parameters_long = RunParameters(
            mode=Mode.WRITE,
            task=Task(instruction="Write a 10,000-word essay"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        estimate_default = estimate_cost(_estimation_configuration, parameters_default, _estimation_pricing, "")
        estimate_long = estimate_cost(_estimation_configuration, parameters_long, _estimation_pricing, "")
        assert estimate_long.total_usd > estimate_default.total_usd

    def test_estimate_is_frozen(
        self,
        _estimation_configuration: CrossfireConfiguration,
        _estimation_pricing: dict[str, tuple[float, float]],
    ):
        parameters = RunParameters(
            mode=Mode.RESEARCH,
            task=Task(instruction="Test"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
        )
        estimate = estimate_cost(_estimation_configuration, parameters, _estimation_pricing, "")
        with pytest.raises(AttributeError):
            estimate.total_usd = 0.0  # type: ignore[misc]
