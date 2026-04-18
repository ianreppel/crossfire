"""Orchestration loop: generation -> review -> synthesis."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, cast

import httpx

from crossfire.core import logging as log
from crossfire.core.archive import RunArchive
from crossfire.core.compression import compress_prompt_components
from crossfire.core.config import get_api_key
from crossfire.core.domain import (
    Candidate,
    CostTracker,
    CrossfireConfiguration,
    ModelGroup,
    Phase,
    Review,
    Role,
    RoundResult,
    RunParameters,
    SynthesisResult,
    Task,
)
from crossfire.core.exclamations import exclaim
from crossfire.core.openrouter import (
    MAX_RETRIES,
    EmptyResponseError,
    call_openrouter,
    call_with_retry,
    extract_cost,
    extract_response_text,
)
from crossfire.core.progress import NoOpProgress, ProgressCallback
from crossfire.core.prompts import (
    build_enrichment_prompt,
    build_generator_prompt,
    build_reviewer_prompt,
    build_synthesizer_prompt,
    parse_review_verdict,
    parse_synthesis_decision,
    strip_synthesis_decision,
)
from crossfire.core.reviewers import assign_reviewers
from crossfire.core.search import (
    extract_search_request,
    perform_search,
    strip_search_request,
)
from crossfire.core.simulation import simulate_response
from crossfire.core.tokens import estimate_tokens

MAX_CONSECUTIVE_ROUND_FAILURES = 2

_RETRIABLE_ERRORS: tuple[type[Exception], ...] = (
    httpx.HTTPStatusError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    json.JSONDecodeError,
    EmptyResponseError,
)

ROLE_TO_PHASE: dict[Role, Phase] = {
    Role.ENRICHER: Phase.ENRICHMENT,
    Role.GENERATOR: Phase.GENERATION,
    Role.REVIEWER: Phase.REVIEW,
    Role.SYNTHESIZER: Phase.SYNTHESIS,
}


class RunFailedError(Exception):
    """Raised when the entire run must abort (e.g. synthesis failure)."""


class Orchestrator:
    """Orchestrates the generate -> review -> synthesize loop."""

    def __init__(
        self,
        configuration: CrossfireConfiguration,
        parameters: RunParameters,
        progress: ProgressCallback | None = None,
        archive: RunArchive | None = None,
    ) -> None:
        self.configuration = configuration
        self.parameters = parameters
        self.cost_tracker = CostTracker()
        # Shared across all phases to cap concurrent OpenRouter HTTP requests,
        # preventing rate-limit (429) errors and connection exhaustion.
        self._semaphore = asyncio.Semaphore(configuration.limits.max_concurrent_requests)
        self._api_key: str = ""
        self._last_generator_search: dict[int, str] = {}
        self._progress: ProgressCallback = progress or NoOpProgress()
        self._archive = archive
        self._http_client: httpx.AsyncClient | None = None
        self._consecutive_round_failures = 0
        self._searches_performed: list[dict[str, str | int]] = []

    # -- high-level entry point ---

    async def run(self) -> str:
        """Runs all rounds and returns the final synthesis."""
        errors = self.configuration.validate(
            self.parameters.num_generators,
            self.parameters.num_reviewers_per_candidate,
        )
        if errors:
            raise ValueError("; ".join(errors))

        if not self.parameters.dry_run:
            self._api_key = get_api_key(self.configuration)
            self._http_client = httpx.AsyncClient(timeout=self.configuration.limits.http_timeout)

        original_instruction: str = self.parameters.task.instruction
        previous_synthesis: str = ""
        try:
            if self._archive:
                self._archive.save_original_instruction(original_instruction)

            if self.parameters.enrich and self.configuration.enricher.names:
                try:
                    self.parameters = await self._enrich_instruction()
                except Exception as exception:
                    log.log_enrichment_failed(
                        model=self.configuration.enricher.names[0],
                        error=str(exception),
                    )

            for round_num in range(1, self.parameters.num_rounds + 1):
                self._progress.on_round_start(round_num, self.parameters.num_rounds)
                result = await self._run_round(round_num, previous_synthesis)

                if result is not None:
                    # Round succeeded: reset failure counter and update synthesis
                    self._consecutive_round_failures = 0
                    previous_synthesis = result.synthesis_text

                    if (
                        self.parameters.early_stop
                        and round_num < self.parameters.num_rounds
                        and result.reviews is not None
                        and self._should_stop_early(result.reviews, self.parameters.early_stop_threshold)
                    ):
                        remaining = self.parameters.num_rounds - round_num
                        log.log_early_stop(
                            round=round_num,
                            remaining_rounds=remaining,
                            reason="no_weaknesses",
                        )
                        self._progress.on_run_end()
                        break
                else:
                    # Round failed: increment failure counter and check if we should abort
                    self._consecutive_round_failures += 1
                    if self._consecutive_round_failures >= MAX_CONSECUTIVE_ROUND_FAILURES:
                        start_round = round_num - self._consecutive_round_failures + 1
                        raise RunFailedError(exclaim(
                            f"Giving up: {self._consecutive_round_failures} consecutive "
                            f"round failures ({start_round}-{round_num}). "
                            "The models are having a bad day."
                        ))
            else:
                # When the loop completes without an early stop
                self._progress.on_run_end()

            cost = self.cost_tracker.summarize()
            log.log_cost_summary(cost)

            # Abort if no content was generated across all rounds
            if not previous_synthesis.strip():
                raise RunFailedError(exclaim(
                    f"{self.parameters.num_rounds} rounds and nothing to show for it. "
                    "Every generator failed or was dropped. Check the logs."
                ))

            if self._archive:
                self._archive.save_final_synthesis(previous_synthesis)
                self._archive.save_metadata(original_instruction, self.parameters, cost)
                if self._searches_performed:
                    self._archive.save_searches(self._searches_performed)
            return previous_synthesis
        finally:
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None

    # -- enrichment pre-step ---

    async def _enrich_instruction(self) -> RunParameters:
        """Rewrites the task instruction through a lightweight enrichment model, returning updated parameters."""
        model = self.configuration.enricher.names[0]
        system, user = build_enrichment_prompt(
            mode=self.parameters.mode,
            instruction=self.parameters.task.instruction,
            context=self.parameters.task.context,
        )

        self._progress.on_phase_start(
            0,
            Phase.ENRICHMENT,
            1,
            models=[model],
            candidate_indices=[None],
        )
        original_tokens = estimate_tokens(self.parameters.task.instruction)
        enriched = await self._call_llm(
            model=model,
            system_prompt=system,
            user_prompt=user,
            role=Role.ENRICHER,
            round_num=0,
        )
        self._progress.on_task_done(0, Phase.ENRICHMENT, model=model)
        self._progress.on_phase_end(0, Phase.ENRICHMENT)

        enriched_tokens = estimate_tokens(enriched)
        log.log_prompt_enriched(
            model=model,
            original_tokens=original_tokens,
            enriched_tokens=enriched_tokens,
        )

        if self._archive:
            self._archive.save_enriched(enriched)

        return replace(
            self.parameters,
            task=Task(instruction=enriched, context=self.parameters.task.context),
        )

    # -- early stopping ---

    @staticmethod
    def _should_stop_early(reviews: list[Review], threshold: int = 1) -> bool:
        """Returns True if reviews indicate no material weaknesses remain.

        When SEVERITY is missing, falls back to counting weakness items against a *threshold*.
        An empty review list never triggers early stop: if nobody looked, we can't claim approval.
        """
        if not reviews:
            return False
        for review in reviews:
            verdict = parse_review_verdict(review.text)
            if verdict.severity == "material":
                return False
            if not verdict.severity and len(verdict.weaknesses) > threshold:
                return False
        return True

    # -- single round ---

    async def _run_round(self, round_num: int, previous_synthesis: str) -> RoundResult | None:
        """Executes one generate -> review -> synthesize cycle, returning None on failure."""

        # --- generation phase ---
        log.log_phase_start(round=round_num, phase=Phase.GENERATION)
        candidates = await self._run_generation(round_num, previous_synthesis)
        log.log_phase_end(round=round_num, phase=Phase.GENERATION)

        if not candidates:
            log.log_round_failed(
                round=round_num,
                reason="no_candidates",
                details="All generators failed or were dropped",
            )
            return None

        if self._archive:
            for candidate in candidates:
                self._archive.save_candidate(candidate)

        # --- review phase ---
        log.log_phase_start(round=round_num, phase=Phase.REVIEW)
        reviews = await self._run_review(round_num, candidates)
        log.log_phase_end(round=round_num, phase=Phase.REVIEW)

        if reviews is None:
            log.log_round_failed(
                round=round_num,
                reason="insufficient_reviewers",
                details="Could not meet minimum reviewer requirements",
            )
            return None

        if self._archive:
            for review in reviews:
                self._archive.save_review(review)

        # --- synthesis phase ---
        log.log_phase_start(round=round_num, phase=Phase.SYNTHESIS)
        synthesis = await self._run_synthesis(round_num, candidates, reviews)
        log.log_phase_end(round=round_num, phase=Phase.SYNTHESIS)

        if self._archive:
            self._archive.save_synthesis(synthesis)

        return RoundResult(synthesis_text=synthesis.text, reviews=reviews)

    # -- generation ---

    async def _run_generation(self, round_num: int, previous_synthesis: str) -> list[Candidate]:
        """Runs all generators in parallel and returns the surviving candidates."""
        generator_names = self.configuration.generators.names
        generator_count = self.parameters.num_generators
        generator_models = [generator_names[index % len(generator_names)] for index in range(generator_count)]
        self._progress.on_phase_start(
            round_num,
            Phase.GENERATION,
            generator_count,
            models=generator_models,
            candidate_indices=[None] * generator_count,
        )

        async def _generate(index: int, model: str) -> Candidate:
            try:
                return await self._generate_candidate(round_num, index, model, previous_synthesis)
            finally:
                self._progress.on_task_done(round_num, Phase.GENERATION, model=model)

        tasks = []
        for index in range(generator_count):
            tasks.append(_generate(index, generator_models[index]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[Candidate] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                model = generator_names[index % len(generator_names)]
                log.log_model_dropped(
                    phase=Phase.GENERATION,
                    role=Role.GENERATOR,
                    model=model,
                    round=round_num,
                    reason=f"generation_error:{type(result).__name__}",
                )
            else:
                candidates.append(cast(Candidate, result))

        self._progress.on_phase_end(round_num, Phase.GENERATION)
        return candidates

    async def _generate_candidate(
        self,
        round_num: int,
        index: int,
        model: str,
        previous_synthesis: str,
    ) -> Candidate:
        """Builds a prompt for one generator, calls the LLM, and handles any search request."""
        prior_search = self._last_generator_search.get(index, "")
        system, user = build_generator_prompt(
            mode=self.parameters.mode,
            instruction=self.parameters.task.instruction,
            context=self.parameters.task.context,
            previous_synthesis=previous_synthesis,
            round_num=round_num,
            search_results=prior_search,
            search_enabled=self.configuration.search.enabled,
        )

        # Priority order for compression when the prompt exceeds the token budget:
        # 1) synthesis, 2) search, and 3) context.
        # Note: task instructions are never compressed.
        compressible_parts: list[tuple[str, str]] = []
        if previous_synthesis:
            compressible_parts.append((previous_synthesis, "candidate_truncation"))
        if prior_search:
            compressible_parts.append((prior_search, "search_truncation"))
        if self.parameters.task.context:
            compressible_parts.append((self.parameters.task.context, "context_truncation"))

        user = self._prepare_prompt(
            system_prompt=system,
            user_prompt=user,
            compressible_parts=compressible_parts,
            context_window=self.configuration.generators.resolve_context_window(model),
            max_output_tokens=self.configuration.generators.resolve_max_output_tokens(model),
            phase=Phase.GENERATION,
            role=Role.GENERATOR,
            model=model,
            round_num=round_num,
        )

        text = await self._call_llm(
            model=model,
            system_prompt=system,
            user_prompt=user,
            role=Role.GENERATOR,
            round_num=round_num,
            dry_run_candidate_index=index,
        )

        text, search_results = await self._process_search_request(
            text,
            role=Role.GENERATOR,
            model=model,
            round_num=round_num,
        )
        if search_results:
            self._last_generator_search[index] = search_results

        return Candidate(
            text=text,
            model=model,
            round=round_num,
            index=index,
            search_results=search_results,
        )

    # -- review ---

    async def _run_review(self, round_num: int, candidates: list[Candidate]) -> list[Review] | None:
        """Assigns reviewers and runs all reviews in parallel."""
        if self.parameters.num_reviewers_per_candidate == 0:
            return []

        # Cross-group overlap is allowed: a model that generated a candidate
        # may also review (with a clean context).
        assignments = assign_reviewers(
            reviewers=self.configuration.reviewers.names,
            num_candidates=len(candidates),
            num_reviewers_per_candidate=self.parameters.num_reviewers_per_candidate,
            round_num=round_num,
            models_used_this_round=set(),
        )

        if assignments is None:
            return None

        total_reviews = sum(len(group) for group in assignments.values())
        review_models: list[str] = []
        review_cand_indices: list[int | None] = []
        for candidate_index, reviewer_models in assignments.items():
            for reviewer_model_name in reviewer_models:
                review_models.append(reviewer_model_name)
                review_cand_indices.append(candidate_index)
        self._progress.on_phase_start(
            round_num,
            Phase.REVIEW,
            total_reviews,
            models=review_models,
            candidate_indices=review_cand_indices,
        )

        async def _review_task(
            candidate_index: int,
            reviewer_model: str,
            assigned_reviewers: set[str],
            replacement_claim_lock: asyncio.Lock,
        ) -> Review:
            try:
                review: Review = await self._review_candidate(
                    round_num=round_num,
                    candidate=candidates[candidate_index],
                    reviewer_model=reviewer_model,
                    all_reviewer_names=self.configuration.reviewers.names,
                    assigned_reviewers=assigned_reviewers,
                    replacement_claim_lock=replacement_claim_lock,
                )
            except BaseException:
                self._progress.on_task_done(
                    round_num,
                    Phase.REVIEW,
                    model=reviewer_model,
                    candidate_index=candidate_index,
                )
                raise
            self._progress.on_task_done(
                round_num,
                Phase.REVIEW,
                model=review.model,
                candidate_index=candidate_index,
            )
            return review

        assigned_reviewers: set[str] = {name for group in assignments.values() for name in group}
        # Guards concurrent access to assigned_reviewers when failed reviews try to claim a
        # replacement model from the remaining pool.
        replacement_claim_lock: asyncio.Lock = asyncio.Lock()
        tasks = []
        task_metadata: list[tuple[int, str]] = []
        for candidate_index, reviewer_models in assignments.items():
            for reviewer_model in reviewer_models:
                tasks.append(_review_task(candidate_index, reviewer_model, assigned_reviewers, replacement_claim_lock))
                task_metadata.append((candidate_index, reviewer_model))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews: list[Review] = []
        for (_candidate_index, reviewer_model), result in zip(task_metadata, results, strict=True):
            if isinstance(result, Exception):
                log.log_model_dropped(
                    phase=Phase.REVIEW,
                    role=Role.REVIEWER,
                    model=reviewer_model,
                    round=round_num,
                    reason=f"review_error:{type(result).__name__}",
                )
                continue
            reviews.append(cast(Review, result))

        self._progress.on_phase_end(round_num, Phase.REVIEW)

        reviewed_candidates = {review.candidate_index for review in reviews}
        if len(reviewed_candidates) < len(candidates):
            return None

        return reviews

    async def _review_candidate(
        self,
        *,
        round_num: int,
        candidate: Candidate,
        reviewer_model: str,
        all_reviewer_names: list[str] | tuple[str, ...],
        assigned_reviewers: set[str],
        replacement_claim_lock: asyncio.Lock,
    ) -> Review:
        """Runs one review, and upon failure, attempts a replacement reviewer."""
        system, user_raw = build_reviewer_prompt(
            mode=self.parameters.mode,
            instruction=self.parameters.task.instruction,
            candidate=candidate,
            search_results=candidate.search_results,
            search_enabled=self.configuration.search.enabled,
        )

        try:
            text = await self._review(
                model=reviewer_model,
                system_prompt=system,
                user_prompt_raw=user_raw,
                candidate=candidate,
                round_num=round_num,
            )
        except _RETRIABLE_ERRORS:
            replacement = await self._find_replacement_reviewer(
                all_reviewer_names,
                assigned_reviewers,
                reviewer_model,
                replacement_claim_lock,
            )
            if replacement:
                log.log_retry(
                    round=round_num,
                    role=Role.REVIEWER,
                    model=reviewer_model,
                    attempt=MAX_RETRIES + 1,
                    reason=f"replacing with {replacement}",
                )
                text = await self._review(
                    model=replacement,
                    system_prompt=system,
                    user_prompt_raw=user_raw,
                    candidate=candidate,
                    round_num=round_num,
                )
                reviewer_model = replacement
            else:
                raise

        text, search_results = await self._process_search_request(
            text,
            role=Role.REVIEWER,
            model=reviewer_model,
            round_num=round_num,
        )

        return Review(
            text=text,
            model=reviewer_model,
            round=round_num,
            candidate_index=candidate.index,
            search_results=search_results,
        )

    async def _review(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt_raw: str,
        candidate: Candidate,
        round_num: int,
    ) -> str:
        """Compresses the review prompt for the *model* and calls the LLM."""
        user = self._prepare_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt_raw,
            compressible_parts=[(candidate.text, "candidate_truncation")],
            context_window=self.configuration.reviewers.resolve_context_window(model),
            max_output_tokens=self.configuration.reviewers.resolve_max_output_tokens(model),
            phase=Phase.REVIEW,
            role=Role.REVIEWER,
            model=model,
            round_num=round_num,
        )
        return await self._call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user,
            role=Role.REVIEWER,
            round_num=round_num,
            dry_run_candidate_index=candidate.index,
        )

    @staticmethod
    async def _find_replacement_reviewer(
        all_names: list[str] | tuple[str, ...],
        assigned_reviewers: set[str],
        failed: str,
        lock: asyncio.Lock,
    ) -> str | None:
        """Atomically claims an unused reviewer model as a substitute for *failed*."""
        async with lock:
            for name in all_names:
                if name not in assigned_reviewers and name != failed:
                    assigned_reviewers.add(name)
                    return name
        return None

    # -- synthesis ---

    def _pick_synthesizer_model(self, round_num: int) -> str:
        """Picks the synthesizer model by rotating through the pool."""
        names = self.configuration.synthesizer.names
        return names[(round_num - 1) % len(names)]

    async def _run_synthesis(
        self,
        round_num: int,
        candidates: list[Candidate],
        reviews: list[Review],
    ) -> SynthesisResult:
        """Manages synthesis progress and error handling, delegating the actual work to ``_synthesize``."""
        model = self._pick_synthesizer_model(round_num)
        self._progress.on_phase_start(
            round_num,
            Phase.SYNTHESIS,
            1,
            models=[model],
            candidate_indices=[None],
        )

        try:
            return await self._synthesize(round_num, model, candidates, reviews)
        except RunFailedError:
            raise
        except Exception as exception:
            log.log_run_failed(
                round=round_num,
                reason="synth_failure",
                details=str(exception),
            )
            raise RunFailedError(exclaim(
                f"Synthesis blew up in round {round_num}: {exception}"
            )) from exception
        finally:
            self._progress.on_phase_end(round_num, Phase.SYNTHESIS)

    async def _synthesize(
        self,
        round_num: int,
        model: str,
        candidates: list[Candidate],
        reviews: list[Review],
    ) -> SynthesisResult:
        """Builds the synthesis prompt, calls the LLM, and parses the decision."""
        system, user = build_synthesizer_prompt(
            mode=self.parameters.mode,
            instruction=self.parameters.task.instruction,
            candidates=candidates,
            reviews=reviews,
        )

        compressible_parts: list[tuple[str, str]] = [
            (candidate.text, "candidate_truncation") for candidate in candidates
        ]
        compressible_parts += [(review.text, "review_truncation") for review in reviews]

        user = self._prepare_prompt(
            system_prompt=system,
            user_prompt=user,
            compressible_parts=compressible_parts,
            context_window=self.configuration.synthesizer.resolve_context_window(model),
            max_output_tokens=self.configuration.synthesizer.resolve_max_output_tokens(model),
            phase=Phase.SYNTHESIS,
            role=Role.SYNTHESIZER,
            model=model,
            round_num=round_num,
            fatal_on_overflow=True,
        )

        text = await self._call_llm(
            model=model,
            system_prompt=system,
            user_prompt=user,
            role=Role.SYNTHESIZER,
            round_num=round_num,
        )
        self._progress.on_task_done(round_num, Phase.SYNTHESIS, model=model)

        attributions, notes = parse_synthesis_decision(text)
        cleaned_text: str = strip_synthesis_decision(text)
        result = SynthesisResult(
            text=cleaned_text,
            model=model,
            round=round_num,
            attributions=attributions,
            notes=notes,
        )
        log.log_synthesis_decision(
            round=round_num,
            model=model,
            attributions=[
                {"index": decision.index, "kept": decision.kept, "discarded": decision.discarded}
                for decision in attributions
            ],
            selected_candidates=result.selected_candidates,
            discarded_candidates=result.discarded_candidates,
            notes=notes,
        )

        return result

    # -- shared helpers ---

    def _prepare_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        compressible_parts: list[tuple[str, str]],
        context_window: int,
        max_output_tokens: int,
        phase: Phase,
        role: Role,
        model: str,
        round_num: int,
        fatal_on_overflow: bool = False,
    ) -> str:
        """Compresses *user_prompt* to fit the token budget.

        When *fatal_on_overflow* is True, raises RunFailedError (for synthesizer); otherwise raises
        RuntimeError (for generator/reviewer).
        """
        user_prompt, fits = compress_prompt_components(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            phase=phase,
            role=role,
            model=model,
            round_num=round_num,
            compressible_parts=compressible_parts,
        )
        if not fits:
            tokens: int = estimate_tokens(user_prompt)
            log.log_model_dropped(
                phase=phase,
                role=role,
                model=model,
                round=round_num,
                tokens_before=tokens,
                tokens_after=tokens,
                reason="token_overflow",
            )
            msg: str = f"Token overflow for {role} {model}"
            raise RunFailedError(exclaim(msg)) if fatal_on_overflow else RuntimeError(exclaim(msg))
        return user_prompt

    async def _process_search_request(
        self,
        text: str,
        *,
        role: Role,
        model: str,
        round_num: int,
    ) -> tuple[str, str]:
        """Extracts a search request from LLM output, performs it if enabled.

        Returns (cleaned_text, search_results).
        """
        if not self.configuration.search.enabled:
            return text, ""
        query = extract_search_request(text)
        if not query:
            return text, ""
        text = strip_search_request(text)
        results: str = await perform_search(
            query,
            dry_run=self.parameters.dry_run,
            instruction=self.parameters.task.instruction,
            mode=self.parameters.mode.value,
            role=role,
            model=model,
            round_num=round_num,
            client=self._http_client,
            search_timeout=self.configuration.limits.search_timeout,
        )
        if results:
            self._searches_performed.append({
                "round": round_num,
                "role": role,
                "model": model,
                "query": query,
            })
        return text, results

    # -- LLM call (real or dry-run) ---

    def _role_group(self, role: Role) -> ModelGroup:
        """Returns the model group configuration for the *role*."""
        if role is Role.GENERATOR:
            return self.configuration.generators
        if role is Role.REVIEWER:
            return self.configuration.reviewers
        if role is Role.SYNTHESIZER:
            return self.configuration.synthesizer
        return self.configuration.enricher

    async def _call_llm(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        role: Role,
        round_num: int,
        dry_run_candidate_index: int | None = None,
    ) -> str:
        """Dispatches to a simulated response (dry run) or the real OpenRouter API."""
        if self.parameters.dry_run:
            return simulate_response(
                instruction=self.parameters.task.instruction,
                mode=self.parameters.mode.value,
                phase=ROLE_TO_PHASE[role],
                role=role,
                model=model,
                round_num=round_num,
                candidate_index=dry_run_candidate_index,
            )

        if self._http_client is None:
            raise RuntimeError(exclaim("HTTP client not initialized. Are you perhaps in dry-run mode?"))
        http_client: httpx.AsyncClient = self._http_client

        # Both the HTTP call and response extraction are inside the retry loop
        # so that empty-but-200 responses (EmptyResponseError) are retried.
        async def _call_and_extract() -> dict[str, Any]:
            data: dict[str, Any] = await call_openrouter(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=self._api_key,
                temperature=self.configuration.limits.temperature_default,
                max_tokens=self._role_group(role).resolve_max_output_tokens(model),
                semaphore=self._semaphore,
                client=http_client,
            )
            extract_response_text(data)  # raises EmptyResponseError if empty
            return data

        data: dict[str, Any] = await call_with_retry(
            _call_and_extract,
            role=role,
            model=model,
            round_num=round_num,
        )

        self.cost_tracker.record(extract_cost(data, model, role, round_num))

        return extract_response_text(data)
