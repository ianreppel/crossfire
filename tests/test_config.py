"""Tests for configuration loading and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from crossfire.core.config import get_api_key, load_configuration
from crossfire.core.domain import ModelGroup, Role


class TestConfigurationLoading:
    def test_load_from_toml(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[openrouter]
api_key_env = "MY_KEY"

[models.generators]
names = ["gen-x"]
context_window = 8000

[models.reviewers]
names = ["rev-x", "rev-y"]
context_window = 8000

[models.synthesizer]
names = ["synth-x"]
context_window = 16000

[search]
enabled = true
provider = "tavily"

[limits]
max_concurrent_requests = 5
temperature_generators = 0.3
"""
        )

        configuration = load_configuration(configuration_path=configuration_file)

        assert configuration.api_key_env == "MY_KEY"
        assert configuration.generators.names == ("gen-x",)
        assert configuration.generators.context_window == 8000
        assert configuration.generators.max_output_tokens == 4096
        assert configuration.reviewers.names == ("rev-x", "rev-y")
        assert configuration.synthesizer.names == ("synth-x",)
        assert configuration.synthesizer.max_output_tokens == int(0.80 * 16000) - 1
        assert configuration.search.enabled is True
        assert configuration.limits.max_concurrent_requests == 5
        assert configuration.limits.temperature_generators == 0.3

    def test_defaults_when_no_file(self):
        configuration = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
        assert configuration.provider == "opencode"
        assert configuration.api_key_env == "OPENCODE_API_KEY"
        assert configuration.generators.names == ()
        assert configuration.limits.max_concurrent_requests == 10

    def test_legacy_openrouter_section_selects_openrouter(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text('[openrouter]\napi_key_env = "MY_KEY"\n')
        configuration = load_configuration(configuration_path=configuration_file)
        assert configuration.provider == "openrouter"
        assert configuration.api_key_env == "MY_KEY"

    def test_provider_aliases_and_cli_override(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
provider = "opencode"

[providers.opencode.model_ids]
"vendor/model" = "vendor-wire"

[providers.custom]
base_url = "https://example.test/v1"
api_key_env = "CUSTOM_KEY"
"""
        )
        configuration = load_configuration(configuration_path=configuration_file)
        opencode = next(provider for provider in configuration.providers if provider.name == "opencode")
        assert opencode.resolve_wire_model_id("vendor/model") == "vendor-wire"
        assert opencode.resolve_wire_model_id("other/model") == "other/model"

        overridden = load_configuration(
            configuration_path=configuration_file,
            cli_overrides={"provider": "custom"},
        )
        assert overridden.provider == "custom"
        assert overridden.api_key_env == "CUSTOM_KEY"

    def test_provider_pinned_model_is_scoped_to_that_gateway(self):
        configuration = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
        mixed = replace(
            configuration,
            generators=ModelGroup(
                names=("openrouter:perplexity/sonar-pro", "z-ai/glm-5.3"),
                context_window=16000,
            ),
        )
        on_opencode = replace(mixed, provider="opencode").resolve_for_mode("code")
        assert on_opencode.generators.names == ("z-ai/glm-5.3",)

        on_openrouter = replace(mixed, provider="openrouter").resolve_for_mode("code")
        assert on_openrouter.generators.names == ("openrouter:perplexity/sonar-pro", "z-ai/glm-5.3")

    def test_api_key_follows_the_selected_provider(self, monkeypatch):
        configuration = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
        monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        assert get_api_key(replace(configuration, provider="opencode")) == "oc-key"
        assert get_api_key(replace(configuration, provider="openrouter")) == "or-key"

        monkeypatch.delenv("OPENROUTER_API_KEY")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            get_api_key(replace(configuration, provider="openrouter"))

    def test_opencode_and_go_share_one_key(self):
        configuration = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
        by_name = {provider.name: provider for provider in configuration.providers}
        assert by_name["opencode"].api_key_env == "OPENCODE_API_KEY"
        assert by_name["opencode-go"].api_key_env == "OPENCODE_API_KEY"

    def test_unknown_provider_is_flagged(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
provider = "typo"

[models.generators]
names = ["gen-x"]

[models.synthesizer]
names = ["synth-x"]
"""
        )
        configuration = load_configuration(configuration_path=configuration_file)
        errors = configuration.validate(num_generators=1, num_reviewers_per_candidate=0)
        assert any("typo" in error for error in errors)

    def test_reasoning_effort_parsed(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text('[limits.reasoning]\nreviewers = "low"\nsynthesizer = "high"\n')
        configuration = load_configuration(configuration_path=configuration_file)
        assert configuration.limits.reasoning_effort_for(Role.REVIEWER) == "low"
        assert configuration.limits.reasoning_effort_for(Role.SYNTHESIZER) == "high"
        assert configuration.limits.reasoning_effort_for(Role.GENERATOR) == ""

    def test_cli_overrides(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[limits]
max_concurrent_requests = 5
temperature_generators = 0.3
"""
        )

        configuration = load_configuration(
            configuration_path=configuration_file,
            cli_overrides={"max_concurrent_requests": 20},
        )
        assert configuration.limits.max_concurrent_requests == 20
        assert configuration.limits.temperature_generators == 0.3

    def test_max_output_tokens_from_toml(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[models.synthesizer]
names = ["synth-x"]
context_window = 200000
max_output_tokens = 32000
"""
        )

        configuration = load_configuration(configuration_path=configuration_file)
        assert configuration.synthesizer.max_output_tokens == 32000
        assert configuration.generators.max_output_tokens == 4096

    def test_per_model_overrides(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[models.reviewers]
names = ["rev-a", "rev-b"]
context_window = 16000
max_output_tokens = 4096

[models.reviewers.context_windows]
"rev-a" = 32000

[models.reviewers.max_output_tokens_by_model]
"rev-a" = 6000
"""
        )

        configuration = load_configuration(configuration_path=configuration_file)
        assert configuration.reviewers.resolve_context_window("rev-a") == 32000
        assert configuration.reviewers.resolve_context_window("rev-b") == 16000
        assert configuration.reviewers.resolve_max_output_tokens("rev-a") == 6000
        assert configuration.reviewers.resolve_max_output_tokens("rev-b") == 4096

    def test_default_max_output_must_fit_each_models_context_window(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[models.reviewers]
names = ["rev-small", "rev-b"]
context_window = 128000
max_output_tokens = 8000

[models.reviewers.context_windows]
"rev-small" = 10000
"""
        )

        configuration = load_configuration(configuration_path=configuration_file)
        errors = configuration.validate(num_generators=1, num_reviewers_per_candidate=2)
        assert any("rev-small" in e and "max_output_tokens" in e for e in errors)


class TestSearchConfiguration:
    def test_search_disabled_by_default(self):
        configuration = load_configuration(configuration_path=Path("/nonexistent"))
        assert configuration.search.enabled is False
