"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from crossfire.core.config import load_configuration


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
temperature_default = 0.3
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
        assert configuration.limits.temperature_default == 0.3

    def test_defaults_when_no_file(self):
        configuration = load_configuration(configuration_path=Path("/nonexistent/crossfire.toml"))
        assert configuration.api_key_env == "OPENROUTER_API_KEY"
        assert configuration.generators.names == ()
        assert configuration.limits.max_concurrent_requests == 10

    def test_cli_overrides(self, tmp_path: Path):
        configuration_file = tmp_path / "crossfire.toml"
        configuration_file.write_text(
            """
[limits]
max_concurrent_requests = 5
temperature_default = 0.3
"""
        )

        configuration = load_configuration(
            configuration_path=configuration_file,
            cli_overrides={"max_concurrent_requests": 20},
        )
        assert configuration.limits.max_concurrent_requests == 20
        assert configuration.limits.temperature_default == 0.3

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
