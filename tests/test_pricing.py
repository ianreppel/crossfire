"""Tests for pricing cache and cost estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crossfire.core.domain import (
    CrossfireConfiguration,
    LimitsConfiguration,
    Mode,
    ModelGroup,
    ProviderConfiguration,
    RunParameters,
    SearchConfiguration,
    Task,
)
from crossfire.core.orchestrator import Orchestrator
from crossfire.core.pricing import (
    _parse_pricing_entry,
    _pricing_keys_for,
    estimate_cost,
    load_pricing,
    parse_api_response,
    parse_length_hint,
    parse_models_dev_response,
    save_pricing,
)
from crossfire.core.providers import Usage


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
        assert result["openrouter::vendor/model-a"] == pytest.approx((0.000002, 0.000008))
        assert result["openrouter::vendor/model-b"] == (0.0, 0.0)

    def test_empty_data(self):
        assert parse_api_response({"data": []}) == {}

    def test_missing_data_key(self):
        assert parse_api_response({}) == {}

    def test_skips_entries_without_id(self):
        raw = {"data": [{"pricing": {"prompt": "0.001", "completion": "0.002"}}]}
        assert parse_api_response(raw) == {}


class TestParseModelsDevResponse:
    def test_converts_per_million_to_per_token_and_qualifies_provider(self):
        raw = {
            "opencode": {"models": {"glm-5.3": {"cost": {"input": 1.4, "output": 4.4}}}},
            "opencode-go": {"models": {"glm-5.3": {"cost": {"input": 1.4, "output": 4.4}}}},
        }
        result = parse_models_dev_response(raw)
        assert result["opencode::glm-5.3"] == pytest.approx((1.4e-6, 4.4e-6))
        assert result["opencode-go::glm-5.3"] == pytest.approx((1.4e-6, 4.4e-6))

    def test_ignores_non_opencode_providers(self):
        raw = {"anthropic": {"models": {"x": {"cost": {"input": 1, "output": 2}}}}}
        assert parse_models_dev_response(raw) == {}

    def test_missing_cost_defaults_to_zero(self):
        raw: dict[str, Any] = {"opencode": {"models": {"free": {}}}}
        assert parse_models_dev_response(raw)["opencode::free"] == (0.0, 0.0)


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


class TestSelectionAwareEstimator:
    """The estimator follows the runtime's model selection rather than averaging each group."""

    @staticmethod
    def _configuration(generator_names: tuple[str, ...], reviewer_names: tuple[str, ...]) -> CrossfireConfiguration:
        return CrossfireConfiguration(
            generators=ModelGroup(names=generator_names, context_window=16000, max_output_tokens=1000),
            reviewers=ModelGroup(names=reviewer_names, context_window=16000, max_output_tokens=1000),
            synthesizer=ModelGroup(names=("v/synth",), context_window=16000, max_output_tokens=1000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )

    @staticmethod
    def _parameters() -> RunParameters:
        return RunParameters(
            mode=Mode.CODE,
            task=Task(instruction="Build something"),
            num_generators=1,
            num_reviewers_per_candidate=1,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )

    def test_only_the_first_generator_is_priced(self):
        pricing = {"v/expensive": (0.01, 0.05), "v/cheap": (0.0001, 0.0005), "v/synth": (0.001, 0.002)}
        both = self._configuration(("v/expensive", "v/cheap"), ("v/rev",))
        only_first = self._configuration(("v/expensive",), ("v/rev",))
        pricing["v/rev"] = (0.001, 0.002)

        estimate_both = estimate_cost(both, self._parameters(), pricing, "")
        estimate_first = estimate_cost(only_first, self._parameters(), pricing, "")
        assert estimate_both.total_usd == pytest.approx(estimate_first.total_usd)

    def test_unused_models_are_not_flagged_missing(self):
        pricing = {"v/gen": (0.001, 0.002), "v/rev": (0.001, 0.002), "v/synth": (0.001, 0.002)}
        configuration = self._configuration(("v/gen",), ("v/rev", "v/rev-unused"))
        estimate = estimate_cost(configuration, self._parameters(), pricing, "")
        assert estimate.missing_models == ()


class TestProviderQualifiedPricing:
    @staticmethod
    def _configuration(provider: str) -> CrossfireConfiguration:
        active = ProviderConfiguration(
            name=provider,
            base_url="https://example.test/v1",
            api_key_env="KEY",
            model_ids=(("vendor/model", "wire-model"),),
        )
        openrouter = ProviderConfiguration(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
        providers = (active,) if provider == "openrouter" else (active, openrouter)
        return CrossfireConfiguration(
            provider=provider,
            providers=providers,
            generators=ModelGroup(names=("vendor/model",), context_window=16000, max_output_tokens=1000),
            reviewers=ModelGroup(names=(), context_window=16000, max_output_tokens=1000),
            synthesizer=ModelGroup(names=("vendor/model",), context_window=16000, max_output_tokens=1000),
            enricher=ModelGroup(names=(), context_window=16000, max_output_tokens=1000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )

    def test_keys_are_provider_qualified_and_keep_fallbacks(self):
        keys = _pricing_keys_for(self._configuration("opencode"))("vendor/model")
        assert keys == ("opencode::wire-model", "wire-model", "vendor/model")

    def test_explicit_provider_prefix_overrides_active_provider(self):
        keys = _pricing_keys_for(self._configuration("opencode"))("openrouter:vendor/model")
        assert keys[0] == "openrouter::vendor/model"

    def test_estimate_picks_matching_provider_price(self):
        pricing = {
            "opencode::wire-model": (0.00001, 0.00005),
            "opencode-go::wire-model": (0.000001, 0.000005),
        }
        parameters = RunParameters(
            mode=Mode.CODE,
            task=Task(instruction="Build something"),
            num_generators=1,
            num_reviewers_per_candidate=0,
            num_rounds=1,
            dry_run=True,
            enrich=False,
        )
        zen = estimate_cost(self._configuration("opencode"), parameters, pricing, "")
        go = estimate_cost(self._configuration("opencode-go"), parameters, pricing, "")
        assert zen.missing_models == ()
        assert go.missing_models == ()
        assert zen.total_usd > go.total_usd


class TestComputeCostFromPricing:
    def test_computes_cost_from_provider_qualified_price(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("glm-5.3",), context_window=16000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(mode=Mode.CODE, task=Task(instruction="x"), dry_run=True)
        orchestrator = Orchestrator(configuration, parameters)
        orchestrator._pricing = {"opencode::glm-5.3": (1e-6, 2e-6)}
        cost = orchestrator._compute_cost("opencode", "glm-5.3", Usage(input_tokens=100, output_tokens=50))
        assert cost == pytest.approx(100 * 1e-6 + 50 * 2e-6)

    def test_unknown_model_returns_none(self):
        configuration = CrossfireConfiguration(
            generators=ModelGroup(names=("glm-5.3",), context_window=16000),
            search=SearchConfiguration(enabled=False),
            limits=LimitsConfiguration(),
        )
        parameters = RunParameters(mode=Mode.CODE, task=Task(instruction="x"), dry_run=True)
        orchestrator = Orchestrator(configuration, parameters)
        orchestrator._pricing = {}
        assert orchestrator._compute_cost("opencode", "missing", Usage(input_tokens=1, output_tokens=1)) is None


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
