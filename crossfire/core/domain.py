"""Primitives for roles, phases, modes, and data structures that flow through the generate → review → synthesize
loop."""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from crossfire.core import privacy
from crossfire.core.tokens import compute_token_budget


class Role(enum.StrEnum):
    """LLM role within a round: enricher, generator, reviewer, synthesizer"""

    ENRICHER = "enricher"
    GENERATOR = "generator"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"


class Phase(enum.StrEnum):
    """Sequential phase within a round (plus an (optional) preparatory enrichment step)"""

    ENRICHMENT = "enrichment"
    GENERATION = "generation"
    REVIEW = "review"
    SYNTHESIS = "synthesis"


class Mode(enum.StrEnum):
    """Operating mode: determines prompt templates and review protocols."""

    RESEARCH = "research"
    CODE = "code"
    EDIT = "edit"
    CHECK = "check"
    WRITE = "write"


_KNOWN_PROVIDER_PREFIXES: tuple[str, ...] = ("opencode-go", "opencode", "openrouter", "synthetic")


def strip_model_prefix(model: str) -> str:
    """Strips a leading ``<provider>:`` prefix (e.g. ``opencode:glm-5.3``) when one is present."""
    prefix, separator, rest = model.partition(":")
    if separator and prefix in _KNOWN_PROVIDER_PREFIXES:
        return rest
    return model


def _model_targets_provider(model: str, provider_name: str, provider_names: set[str]) -> bool:
    """A model pinned with a ``provider:`` prefix runs only on that gateway; an unprefixed model runs anywhere."""
    prefix, separator, _ = model.partition(":")
    if not separator or prefix not in provider_names:
        return True
    return prefix == provider_name


@dataclass(frozen=True)
class ProviderConfiguration:
    """Connection details, model alias table, and data policy for a single inference gateway.

    ``model_ids`` maps a neutral model slug (one that works on every gateway) to the provider's wire model ID.
    Slugs absent from the table are passed through unchanged, which is what makes OpenRouter lists work as-is.

    The three privacy fields default to the safe setting. ``deny_data_collection`` and ``require_zdr`` become
    OpenRouter's per-request routing filters, which are the only such knobs any gateway exposes; on the OpenCode
    gateways they record intent and change nothing on the wire, where :mod:`crossfire.core.privacy` enforces the
    policy instead by refusing models that train. ``allow_training_models`` is the opt-out from that refusal.
    """

    name: str
    base_url: str
    api_key_env: str
    requires_session: bool = False
    model_ids: tuple[tuple[str, str], ...] = ()
    deny_data_collection: bool = True
    require_zdr: bool = True
    allow_training_models: bool = False

    def resolve_wire_model_id(self, model: str) -> str:
        """Returns the provider's wire model ID for a neutral *model* slug.

        Handles the alias table, this provider's own ``<name>:`` prefix, and the built-in prefixes, so a model
        pinned to this gateway resolves to the right wire ID.
        """
        for neutral, wire in self.model_ids:
            if neutral == model:
                return wire
        own_prefix = f"{self.name}:"
        if model.startswith(own_prefix):
            return model[len(own_prefix) :]
        return strip_model_prefix(model)


DEFAULT_PROVIDER_CONFIGURATIONS: tuple[ProviderConfiguration, ...] = (
    ProviderConfiguration(
        name="opencode",
        base_url="https://opencode.ai/zen/v1",
        api_key_env="OPENCODE_API_KEY",
    ),
    ProviderConfiguration(
        name="opencode-go",
        base_url="https://opencode.ai/zen/go/v1",
        api_key_env="OPENCODE_API_KEY",
        requires_session=True,
    ),
    ProviderConfiguration(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    ProviderConfiguration(
        name="synthetic",
        base_url="https://api.synthetic.new/openai/v1",
        api_key_env="SYNTHETIC_API_KEY",
    ),
)


@dataclass(frozen=True)
class Task:
    """User-provided instructions and an (optional) context that is fed to every generator."""

    instruction: str
    context: str = ""


@dataclass
class Candidate:
    """A single generator's output for a given round."""

    text: str
    model: str
    round: int
    index: int
    search_results: str = ""

    @property
    def label(self) -> str:
        return f"candidate-r{self.round}-i{self.index}"


@dataclass
class Review:
    """A single reviewer's critique of a single candidate for a given round."""

    text: str
    model: str
    round: int
    candidate_index: int
    search_results: str = ""

    @property
    def label(self) -> str:
        return f"review-r{self.round}-c{self.candidate_index}-{self.model}"


@dataclass
class CandidateDecision:
    """What the synthesizer kept and discarded from a single candidate in a given round."""

    index: int
    kept: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Output of the synthesis in a given round: the merged text plus per-candidate attribution"""

    text: str
    model: str
    round: int
    attributions: list[CandidateDecision] = field(default_factory=list)
    notes: str = ""

    @property
    def selected_candidates(self) -> list[int]:
        """Indices of candidates with at least one element kept."""
        return [decision.index for decision in self.attributions if decision.kept]

    @property
    def discarded_candidates(self) -> list[int]:
        """Indices of candidates with only discarded elements."""
        return [decision.index for decision in self.attributions if decision.discarded and not decision.kept]


@dataclass(frozen=True)
class ModelGroup:
    """Pool of LLM model IDs that serve a single pipeline function (e.g. enrichment, generation, review, synthesis)

    Each pool has token limits shared by all its models, with optional per-model overrides for
    context window and max output tokens.
    Overrides are stored as tuples of ``(model_id, value)`` so the dataclass stays frozen and hashable.
    """

    names: tuple[str, ...]
    context_window: int
    max_output_tokens: int = 4096
    context_windows: tuple[tuple[str, int], ...] = ()
    max_output_tokens_by_model: tuple[tuple[str, int], ...] = ()

    def resolve_context_window(self, model: str) -> int:
        """Returns the effective context window for *model*, with a fallback to the group default."""
        for model_id, window in self.context_windows:
            if model_id == model:
                return window
        return self.context_window

    def resolve_max_output_tokens(self, model: str) -> int:
        """Returns the effective max output tokens for *model*, with a fallback to the group default."""
        for model_id, tokens in self.max_output_tokens_by_model:
            if model_id == model:
                return tokens
        return self.max_output_tokens


@dataclass
class RoundResult:
    """Outcome of a single round, including both synthesis text and reviews."""

    synthesis_text: str
    reviews: list[Review]


@dataclass(frozen=True)
class SearchConfiguration:
    """Web search settings (for Tavily): disabled by default."""

    enabled: bool = False
    provider: str = "tavily"


@dataclass(frozen=True)
class LimitsConfiguration:
    """Concurrency, temperature, reasoning effort, and timeout defaults."""

    max_concurrent_requests: int = 10
    temperature_generators: float = 0.7
    temperature_reviewers: float = 0.1
    temperature_synthesizer: float = 0.2
    temperature_enricher: float = 0.3
    http_timeout: float = 120.0
    search_timeout: float = 30.0
    reasoning_effort_by_role: tuple[tuple[str, str], ...] = ()

    def temperature_for(self, role: Role) -> float:
        """Returns the configured sampling temperature for *role* (frontier models ignore it entirely)."""
        return {
            Role.ENRICHER: self.temperature_enricher,
            Role.GENERATOR: self.temperature_generators,
            Role.REVIEWER: self.temperature_reviewers,
            Role.SYNTHESIZER: self.temperature_synthesizer,
        }[role]

    def reasoning_effort_for(self, role: Role) -> str:
        """Returns the configured reasoning effort for *role*, or ``""`` when unset."""
        for role_name, effort in self.reasoning_effort_by_role:
            if role_name == role.value:
                return effort
        return ""


@dataclass
class ModelGroupOverrides:
    """Overrides for model groups (with ``None`` as the global default)"""

    generators: ModelGroup | None = None
    reviewers: ModelGroup | None = None
    synthesizer: ModelGroup | None = None
    enricher: ModelGroup | None = None


@dataclass(frozen=True)
class CrossfireConfiguration:
    """Top-level configuration: model groups, search, limits, and per-mode overrides"""

    api_key_env: str = "OPENROUTER_API_KEY"
    provider: str = "opencode"
    providers: tuple[ProviderConfiguration, ...] = DEFAULT_PROVIDER_CONFIGURATIONS
    enricher: ModelGroup = field(
        default_factory=lambda: ModelGroup(
            names=(),
            context_window=128000,
            max_output_tokens=4096,
        )
    )
    generators: ModelGroup = field(default_factory=lambda: ModelGroup(names=(), context_window=16000))
    reviewers: ModelGroup = field(default_factory=lambda: ModelGroup(names=(), context_window=16000))
    synthesizer: ModelGroup = field(
        default_factory=lambda: ModelGroup(
            names=(),
            context_window=200000,
            max_output_tokens=32000,
        )
    )
    search: SearchConfiguration = field(default_factory=SearchConfiguration)
    limits: LimitsConfiguration = field(default_factory=LimitsConfiguration)
    mode_overrides: dict[str, ModelGroupOverrides] = field(default_factory=dict)

    def active_provider(self) -> ProviderConfiguration:
        """Returns the provider selected by :attr:`provider`, falling back to the first configured one."""
        for provider in self.providers:
            if provider.name == self.provider:
                return provider
        if self.providers:
            return self.providers[0]
        return ProviderConfiguration(name=self.provider, base_url="", api_key_env=self.api_key_env)

    def for_active_provider(self) -> CrossfireConfiguration:
        """Restricts every group to the models the active provider can serve.

        A model may be pinned to one gateway with a ``provider:`` prefix (e.g. ``openrouter:perplexity/sonar-pro``),
        in which case it drops out of every other gateway's groups. Unprefixed models serve on all gateways. This
        is what keeps OpenRouter-only models (Perplexity) out of OpenCode runs, and vice versa.
        """
        provider_names = {provider.name for provider in self.providers}

        def keep(group: ModelGroup) -> ModelGroup:
            names = tuple(name for name in group.names if _model_targets_provider(name, self.provider, provider_names))
            if names == group.names:
                return group
            context_windows = tuple(entry for entry in group.context_windows if entry[0] in names)
            max_output_tokens_by_model = tuple(entry for entry in group.max_output_tokens_by_model if entry[0] in names)
            return replace(
                group,
                names=names,
                context_windows=context_windows,
                max_output_tokens_by_model=max_output_tokens_by_model,
            )

        return replace(
            self,
            enricher=keep(self.enricher),
            generators=keep(self.generators),
            reviewers=keep(self.reviewers),
            synthesizer=keep(self.synthesizer),
        )

    def resolve_for_mode(self, mode: str) -> CrossfireConfiguration:
        """Resolves a configuration with overrides for a certain *mode*, scoped to the active provider."""
        overrides = self.mode_overrides.get(mode)
        if not overrides:
            return self.for_active_provider()
        resolved = CrossfireConfiguration(
            api_key_env=self.api_key_env,
            provider=self.provider,
            providers=self.providers,
            enricher=overrides.enricher or self.enricher,
            generators=overrides.generators or self.generators,
            reviewers=overrides.reviewers or self.reviewers,
            synthesizer=overrides.synthesizer or self.synthesizer,
            search=self.search,
            limits=self.limits,
            mode_overrides=self.mode_overrides,
        )
        return resolved.for_active_provider()

    def data_policy_notices(self) -> list[str]:
        """Returns what the active gateway does with prompts on the models this run would use.

        These are notices rather than errors: no gateway lets a request waive its own retention, so a run against
        a retaining model is still a valid run. Training is a different matter, and :meth:`validate` refuses it.
        """
        provider = self.active_provider()
        notices: list[str] = []
        for group in (self.enricher, self.generators, self.reviewers, self.synthesizer):
            wire_model_ids = [provider.resolve_wire_model_id(model) for model in group.names]
            notices.extend(privacy.retention_notices(provider.name, wire_model_ids))
        gateway_note = privacy.gateway_retention_note(provider.name)
        if gateway_note:
            notices.append(gateway_note)
        return sorted(set(notices))

    def validate(
        self,
        num_generators: int,
        num_reviewers_per_candidate: int,
    ) -> list[str]:
        """Returns a list of validation errors (empty = valid)."""
        errors: list[str] = []
        active_provider = self.active_provider()

        provider_names = sorted(provider.name for provider in self.providers)
        if self.provider not in provider_names:
            errors.append(
                f"Unknown provider '{self.provider}'. Configured providers: {', '.join(provider_names) or 'none'}"
            )

        if not self.generators.names:
            errors.append("No generator models configured")
        if not self.synthesizer.names:
            errors.append("No synthesizer models configured")

        required_reviewers = num_generators * num_reviewers_per_candidate
        if num_reviewers_per_candidate > 0 and len(self.reviewers.names) < required_reviewers:
            errors.append(
                f"Need at least {required_reviewers} reviewer models "
                f"(num_generators={num_generators} * num_reviewers_per_candidate="
                f"{num_reviewers_per_candidate}), got {len(self.reviewers.names)}. "
                "Add more models to [models.reviewers] in crossfire.toml"
            )

        for label, group in [
            ("enricher", self.enricher),
            ("generators", self.generators),
            ("reviewers", self.reviewers),
            ("synthesizer", self.synthesizer),
        ]:
            if not active_provider.allow_training_models:
                wire_model_ids = [active_provider.resolve_wire_model_id(model) for model in group.names]
                errors.extend(
                    f"{label}: {violation}"
                    for violation in privacy.training_violations(active_provider.name, wire_model_ids)
                )

            if group.context_window <= 0:
                errors.append(f"{label}.context_window must be positive")

            if group.max_output_tokens <= 0:
                errors.append(f"{label}.max_output_tokens must be positive")

            budget = compute_token_budget(group.context_window)
            if group.max_output_tokens >= budget:
                errors.append(
                    f"{label}.max_output_tokens ({group.max_output_tokens}) "
                    f"must be less than 80% of context_window ({budget})"
                )

            for model, cw in group.context_windows:
                if model not in group.names:
                    errors.append(f"{label}.context_windows references unknown model: {model}")
                if cw <= 0:
                    errors.append(f"{label}.context_windows[{model}] must be positive")

            for model, max_output in group.max_output_tokens_by_model:
                if model not in group.names:
                    errors.append(f"{label}.max_output_tokens_by_model references unknown model: {model}")
                if max_output <= 0:
                    errors.append(f"{label}.max_output_tokens_by_model[{model}] must be positive")

            for model in group.names:
                model_cw = group.resolve_context_window(model)
                model_budget = compute_token_budget(model_cw)
                max_out = group.resolve_max_output_tokens(model)
                if max_out >= model_budget:
                    errors.append(
                        f"{label}: max_output_tokens for {model} ({max_out}) must be less than "
                        f"80% of that model's effective context_window ({model_budget})"
                    )

        return errors


@dataclass
class RunParameters:
    """Parameters for a single orchestration run, resolved from CLI and configuration."""

    mode: Mode
    task: Task
    num_generators: int = 1
    num_reviewers_per_candidate: int = 3
    num_rounds: int = 3
    dry_run: bool = False
    enrich: bool = True
    early_stop: bool = True
    early_stop_threshold: int = 1


@dataclass(frozen=True)
class CostEstimate:
    """Upper-bound cost estimate for a dry run, based on cached provider pricing."""

    total_usd: float
    missing_models: tuple[str, ...] = ()
    fetched_at: str = ""


@dataclass
class CostEntry:
    """Token and cost information from a single LLM call."""

    model: str
    role: Role
    round: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float | None = None
    provider: str = ""
    wire_model_id: str = ""


@dataclass
class CostTracker:
    """Accumulates :class:`CostEntry` records and produces an aggregate summary."""

    entries: list[CostEntry] = field(default_factory=list)

    def record(self, entry: CostEntry) -> None:
        self.entries.append(entry)

    def summarize(self) -> dict[str, Any]:
        per_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0.0,
            }
        )
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0
        unpriced_models: set[str] = set()

        for entry in self.entries:
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            total_cache_read += entry.cache_read_tokens
            total_cache_write += entry.cache_write_tokens
            if entry.cost is not None:
                total_cost += entry.cost
            else:
                unpriced_models.add(f"{entry.provider}::{entry.wire_model_id or entry.model}")

            per_model[entry.model]["input_tokens"] += entry.input_tokens
            per_model[entry.model]["output_tokens"] += entry.output_tokens
            per_model[entry.model]["cache_read_tokens"] += entry.cache_read_tokens
            per_model[entry.model]["cache_write_tokens"] += entry.cache_write_tokens
            if entry.cost is not None:
                per_model[entry.model]["cost"] += entry.cost

        return {
            "per_model": dict(per_model),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_cost": total_cost,
            "unpriced_models": sorted(unpriced_models),
        }
