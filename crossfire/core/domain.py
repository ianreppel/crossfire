"""Primitives for roles, phases, modes, and data structures that flow through the generate → review → synthesize
loop."""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

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
    """Operating mode — determines prompt templates and review protocols."""

    RESEARCH = "research"
    CODE = "code"
    EDIT = "edit"
    CHECK = "check"
    WRITE = "write"


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
    """Concurrency, temperature, and timeout defaults."""

    max_concurrent_requests: int = 10
    temperature_default: float = 0.2
    http_timeout: float = 120.0
    search_timeout: float = 30.0


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

    def resolve_for_mode(self, mode: str) -> CrossfireConfiguration:
        """Resolves a configuration with overrides for a certain *mode*."""
        overrides = self.mode_overrides.get(mode)
        if not overrides:
            return self
        return CrossfireConfiguration(
            api_key_env=self.api_key_env,
            enricher=overrides.enricher or self.enricher,
            generators=overrides.generators or self.generators,
            reviewers=overrides.reviewers or self.reviewers,
            synthesizer=overrides.synthesizer or self.synthesizer,
            search=self.search,
            limits=self.limits,
            mode_overrides=self.mode_overrides,
        )

    def validate(
        self,
        num_generators: int,
        num_reviewers_per_candidate: int,
    ) -> list[str]:
        """Returns a list of validation errors (empty = valid)."""
        errors: list[str] = []

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
    """Upper-bound cost estimate for a dry run, based on cached OpenRouter pricing."""

    total_usd: float
    missing_models: tuple[str, ...] = ()
    fetched_at: str = ""


@dataclass
class CostEntry:
    """Token and cost information from a single LLM call in OpenRouter."""

    model: str
    role: Role
    round: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


@dataclass
class CostTracker:
    """Accumulates :class:`CostEntry` records and produces an aggregate summary."""

    entries: list[CostEntry] = field(default_factory=list)

    def record(self, entry: CostEntry) -> None:
        self.entries.append(entry)

    def summarize(self) -> dict[str, Any]:
        per_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for entry in self.entries:
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            if entry.cost is not None:
                total_cost += entry.cost

            per_model[entry.model]["input_tokens"] += entry.input_tokens
            per_model[entry.model]["output_tokens"] += entry.output_tokens
            if entry.cost is not None:
                per_model[entry.model]["cost"] += entry.cost

        return {
            "per_model": dict(per_model),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": total_cost,
        }
