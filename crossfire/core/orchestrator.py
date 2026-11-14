"""Orchestration loop: generation -> review -> synthesis."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx

from crossfire.core import logging as log
from crossfire.core.adjudication import Claim, extract_claims, partition_claims, render_cross_branch
from crossfire.core.archive import RunArchive
from crossfire.core.compression import compress_prompt_components
from crossfire.core.config import get_api_key
from crossfire.core.domain import (
    Branch,
    BranchContext,
    BranchState,
    Candidate,
    CostEntry,
    CostTracker,
    CrossBranchVerdict,
    CrossfireConfiguration,
    LanguageConfiguration,
    ModelGroup,
    Phase,
    Review,
    Role,
    RoundResult,
    RunParameters,
    SynthesisResult,
    Task,
    VerdictState,
)
from crossfire.core.evidence import EvidenceRegistry, apply_evidence_rules
from crossfire.core.exclamations import exclaim
from crossfire.core.hashing import sha256_text
from crossfire.core.language import resolve_language, validate_language
from crossfire.core.pricing import PRICING_FILENAME, load_pricing
from crossfire.core.progress import NoOpProgress, ProgressCallback
from crossfire.core.prompts import (
    EnvelopeError,
    build_cross_branch_adjudicator_prompt,
    build_enrichment_prompt,
    build_generator_prompt,
    build_language_tag,
    build_reviewer_prompt,
    build_synthesizer_prompt,
    build_translation_prompt,
    parse_control_envelope,
    parse_cross_branch_verdicts,
    parse_synthesis_decision,
    strip_synthesis_decision,
)
from crossfire.core.providers import (
    MAX_RETRIES,
    AuthenticationError,
    EmptyResponseError,
    InsufficientCreditsError,
    RefusalResponseError,
    TruncatedResponseError,
    Usage,
    call_model,
    call_with_retry,
    extract_response_text,
    extract_usage,
    resolve_protocol,
)
from crossfire.core.reviewers import assign_reviewers
from crossfire.core.search import (
    extract_search_request,
    perform_search,
    strip_search_request,
)
from crossfire.core.simulation import simulate_response
from crossfire.core.tokens import estimate_tokens
from crossfire.core.translation import (
    extract_do_not_translate,
    fidelity_violations,
    restore_do_not_translate,
)

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

# Errors that warrant swapping in a different model rather than retrying the same call.
_REPLACEABLE_ERRORS: tuple[type[Exception], ...] = (
    *_RETRIABLE_ERRORS,
    TruncatedResponseError,
    RefusalResponseError,
)

# A rejected key or an empty wallet aborts the run: retrying the call or substituting a model changes nothing.
_FATAL_ERRORS: tuple[type[Exception], ...] = (AuthenticationError, InsufficientCreditsError)

ROLE_TO_PHASE: dict[Role, Phase] = {
    Role.ENRICHER: Phase.ENRICHMENT,
    Role.TRANSLATOR: Phase.TRANSLATE,
    Role.ADJUDICATOR: Phase.ADJUDICATE,
    Role.GENERATOR: Phase.GENERATION,
    Role.REVIEWER: Phase.REVIEW,
    Role.SYNTHESIZER: Phase.SYNTHESIS,
}


class RunFailedError(Exception):
    """Raised when the entire run must abort (e.g. synthesis failure)."""


class RefusalError(RuntimeError):
    """Raised when a model refuses to produce output (e.g. "insufficient data")."""


_REFUSAL_PHRASES: tuple[str, ...] = (
    "i must be transparent about",
    "i cannot fulfill",
    "i'm unable to fulfill",
    r"insufficient .* to support",
    "would you like me to search",
    "i appreciate the detailed instructions, but",
)

_REFUSAL_REGEX: re.Pattern[str] = re.compile("|".join(_REFUSAL_PHRASES), re.IGNORECASE)


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
        # Shared across all phases to cap concurrent provider HTTP requests,
        # preventing rate-limit (429) errors and connection exhaustion.
        self._semaphore = asyncio.Semaphore(configuration.limits.max_concurrent_requests)
        self._api_key: str = ""
        self._progress: ProgressCallback = progress or NoOpProgress()
        self._archive = archive
        self._http_client: httpx.AsyncClient | None = None
        self._searches_performed: list[dict[str, str | int]] = []
        self._session_id = f"crossfire-{uuid4()}"
        self._pricing: dict[str, tuple[float, float]] | None = None
        self.language, self.language_provenance = self._resolve_language(configuration)
        # The branch currently running: its own task, search cache and failure counter, never another branch's.
        self._active_context: BranchContext = BranchContext(
            branch=Branch(language=self.language.source),
            task=parameters.task,
        )
        self._active_task: Task = parameters.task
        self._active_archive: RunArchive | None = archive
        # On --resume the journal decides which units are already done: a unit counts only while its artefact hash
        # still matches the file on disk.
        self._journal_records: list[dict[str, Any]] = archive.journal.replay() if archive and parameters.resume else []

    # -- language surface ---

    @staticmethod
    def _global_language_layer(configuration: CrossfireConfiguration) -> dict[str, Any]:
        """Returns the explicit global language layer, falling back to the resolved global view.

        A configuration built in code rather than parsed from TOML carries its global values on ``language`` alone, and
        that view is exactly what the global layer means for an empty ``[language]`` section.
        """
        if configuration.language_layer:
            return configuration.language_layer
        resolved = configuration.language
        layer: dict[str, Any] = {
            "source": resolved.source,
            "branches": list(resolved.branches),
            "translate_instruction": resolved.translate_instruction,
            "adjudicate": resolved.adjudicate,
            "translator_backend": resolved.translator_backend,
        }
        for key in ("preset", "competence_branch", "render_language"):
            value = getattr(resolved, key)
            if value is not None:
                layer[key] = value
        return layer

    def _resolve_language(self, configuration: CrossfireConfiguration) -> tuple[LanguageConfiguration, dict[str, str]]:
        """Resolves this run's language configuration from the CLI, mode and global layers."""
        return resolve_language(
            cli=self.parameters.language_cli,
            mode_override=configuration.language_modes.get(self.parameters.mode.value),
            global_=self._global_language_layer(configuration),
        )

    def _build_branches(self, configuration: CrossfireConfiguration) -> list[Branch]:
        """Builds one pending branch per configured language.

        The branch whose language is the source carries the original instruction, attested with its own hash and an
        empty model id. Every other branch starts empty and is translated by :meth:`_translate_instruction`.
        """
        original = self._active_task.instruction
        branches: list[Branch] = []
        for tag in self.language.branches:
            branch = Branch(language=tag, is_competence_branch=tag == self.language.competence_branch)
            if tag == self.language.source:
                branch = replace(branch, instruction_text=original, instruction_hash=sha256_text(original))
            branches.append(branch)
        return branches

    def _prepare_branch_context(self, branch: Branch) -> BranchContext:
        """Returns a fresh context for *branch*, built from the immutable original task.

        The instruction is the branch's own (translated or original) and the context is the user's, unchanged. The
        search cache and the failure counter start empty: a branch never inherits another branch's state.
        """
        task = Task(
            instruction=branch.instruction_text or self._active_task.instruction,
            context=self._active_task.context,
            language=branch.language,
        )
        return BranchContext(branch=branch, task=task)

    async def _translate_instruction(self, branch: Branch) -> Branch:
        """Returns *branch* with its instruction ready for enrichment.

        Translation is skipped only when the branch language equals the source language, never on a guess. The source
        branch therefore spends no translation call and is attested as its own original. A translated branch is attested
        with the translator's model id, the hash of what came back, and its own artefact under ``branches/<lang>``.
        """
        original = self._active_task.instruction
        if branch.instruction_text:
            return branch
        needs_translation = self.language.translate_instruction and branch.language != self.language.source
        if not needs_translation:
            return replace(branch, instruction_text=original, instruction_hash=sha256_text(original))

        model = self.configuration.translator.names[0]
        masked, spans = extract_do_not_translate(original)
        system, user = build_translation_prompt(f"{self.language.source}->{branch.language}", masked)
        try:
            text = await self._call_llm(
                model=model,
                system_prompt=system,
                user_prompt=user,
                role=Role.TRANSLATOR,
                round_num=0,
                max_retries=0,
            )
        except _FATAL_ERRORS:
            raise
        except Exception as exception:
            # A refusal, a truncation or an empty reply excludes this branch and lets the run continue.
            log.log_translation_failed(branch=branch.language, model=model, error=str(exception))
            return replace(branch, state=BranchState.TRANSLATION_FAILED)

        translated = text.strip()
        if not translated:
            log.log_translation_failed(
                branch=branch.language,
                model=model,
                error="the translator returned an empty instruction",
            )
            return replace(branch, state=BranchState.TRANSLATION_FAILED)

        # Citations, numbers and identifiers must come back byte-identical, or the branch is not usable.
        violations = fidelity_violations(masked, translated, spans)
        if violations:
            log.log_translation_failed(
                branch=branch.language,
                model=model,
                error="fidelity: " + "; ".join(violations),
            )
            return replace(branch, state=BranchState.TRANSLATION_FAILED)
        translated = restore_do_not_translate(translated, spans)
        if self._archive:
            digest = self._archive.save_branch_instruction(branch.language, translated)
            self._archive.journal.append(
                "instruction_attested",
                branch=branch.language,
                model=model,
                path=self._archive.record_path(f"branches/{branch.language}/instruction.md"),
                sha256=digest,
            )
        return replace(
            branch,
            instruction_text=translated,
            instruction_model=model,
            instruction_hash=sha256_text(translated),
            state=BranchState.TRANSLATED,
        )

    # -- high-level entry point ---

    async def run(self) -> str:
        """Runs every branch in turn and returns the reader's output.

        A single branch behaves exactly as the pipeline always has: one enrichment, one round loop, one final synthesis.
        More branches run sequentially, each in its own context with its own translated instruction, and their syntheses
        are reconciled after the loops rather than inside them.
        """
        errors = self.configuration.validate(
            self.parameters.num_generators,
            self.parameters.num_reviewers_per_candidate,
        )
        errors.extend(validate_language(self.language))
        if errors:
            raise ValueError("; ".join(errors))

        # No gateway lets a request waive its own retention, so the best Crossfire can do on the OpenCode
        # gateways is say plainly what the chosen models do with the prompts.
        for notice in self.configuration.data_policy_notices():
            log.log_data_policy_notice(notice=notice)
        log.log_language_resolved(language=self._language_manifest())

        if not self.parameters.dry_run:
            self._api_key = get_api_key(self.configuration)
            self._http_client = httpx.AsyncClient(timeout=self.configuration.limits.http_timeout)

        original_instruction: str = self._active_task.instruction
        try:
            if self._archive:
                self._archive.claim_owner(adopt_stale=self.parameters.resume)
                self._archive.save_original_instruction(original_instruction)
                self._archive.journal.append(
                    "run_started",
                    mode=self.parameters.mode.value,
                    branches=list(self.language.branches),
                    source=self.language.source,
                    resume=self.parameters.resume,
                )

            branches = self._build_branches(self.configuration)
            if len(branches) >= 2 and self.language.adjudicate and not self.configuration.adjudicator.names:
                raise ValueError(
                    "No adjudicator models configured: set [models.adjudicator] names, or set adjudicate = false"
                )
            contexts: list[BranchContext] = []
            for branch in branches:
                translated = self._resumed_instruction(branch) or await self._translate_instruction(branch)
                if translated.state is BranchState.TRANSLATION_FAILED:
                    log.log_branch_excluded(branch=translated.language, reason=BranchState.TRANSLATION_FAILED)
                    continue
                contexts.append(await self._run_branch(self._prepare_branch_context(translated)))

            cost = self.cost_tracker.summarize()
            log.log_cost_summary(cost)

            synthesised = [context for context in contexts if context.branch.state is BranchState.SYNTHESIZED]
            if not synthesised:
                raise RunFailedError(
                    exclaim(
                        f"{self.parameters.num_rounds} rounds and nothing to show for it. "
                        "Every generator failed or was dropped. Check the logs."
                    )
                )

            shape = self._run_shape(synthesised, branches)
            log.log_run_shape(shape=shape, branches=len(branches), synthesised=len(synthesised))
            if self._archive:
                self._archive.journal.append(
                    "run_shape",
                    shape=shape,
                    synthesised=[context.branch.language for context in synthesised],
                )

            primary = self._primary_context(synthesised)
            result: str = primary.synthesis_history[-1]

            if len(synthesised) >= 2 and self.language.adjudicate:
                verdicts, published = await self._adjudicate_branches(synthesised)
                if self._archive and verdicts:
                    self._archive.save_cross_branch(published)
                if verdicts and self.language.render_language:
                    result = await self._render_output(verdicts, synthesised)
                    if self._archive:
                        self._archive.save_render(self.language.render_language, result)

            if self._archive:
                if len(branches) > 1:
                    self._archive.save_final_synthesis(result)
                self._archive.save_metadata(
                    original_instruction,
                    replace(self.parameters, task=primary.task),
                    cost,
                    language=self._language_manifest(),
                )
                if self._searches_performed:
                    self._archive.save_searches(self._searches_performed)
            return result
        finally:
            if self._archive is not None:
                self._archive.release_owner()
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None

    async def _adjudicate_branches(
        self,
        contexts: list[BranchContext],
    ) -> tuple[list[CrossBranchVerdict], str]:
        """Compares the branch syntheses claim by claim and returns the verdicts plus the published markdown.

        Claims are packed into packets under the adjudicator's context window and one call is made per packet. A packet
        that fails twice leaves the run ``adjudication_incomplete``: the branch outputs still publish, and the missing
        packet is named, because a comparison that did not happen is not a clean comparison.
        """
        syntheses = {context.branch.language: context.synthesis_history[-1] for context in contexts}
        claims: list[Claim] = extract_claims(syntheses)
        registry = self._build_evidence_registry(syntheses)
        if self._archive:
            self._archive.save_evidence(registry.spans())
        model = self.configuration.adjudicator.names[0]
        # The syntheses and the evidence listing travel with every packet, so they count against the window too.
        overhead = estimate_tokens("".join(syntheses.values()) + "".join(span.text for span in registry.spans()))
        packets = partition_claims(
            claims,
            context_window=self.configuration.adjudicator.resolve_context_window(model),
            max_output_tokens=self.configuration.adjudicator.resolve_max_output_tokens(model),
            overhead_tokens=overhead,
        )

        verdicts: list[CrossBranchVerdict] = []
        incomplete: list[int] = []
        for index, packet in enumerate(packets, start=1):
            system, user = build_cross_branch_adjudicator_prompt(packet, syntheses, registry.spans())
            parsed: list[CrossBranchVerdict] | None = None
            error = ""
            for _attempt in range(2):
                try:
                    text = await self._call_llm(
                        model=model,
                        system_prompt=system,
                        user_prompt=user,
                        role=Role.ADJUDICATOR,
                        round_num=0,
                        max_retries=0,
                    )
                    parse_control_envelope(text)
                    parsed = parse_cross_branch_verdicts(text)
                except _FATAL_ERRORS:
                    raise
                except Exception as exception:
                    error = str(exception)
                    continue
                break
            if parsed is None:
                incomplete.append(index)
                log.log_adjudication_incomplete(packet=index, error=error)
                continue
            for verdict in parsed:
                normalised, _violations = apply_evidence_rules(verdict, registry, syntheses)
                normalised.packet = index
                normalised.adjudicator = model
                verdicts.append(normalised)

        unavailable = sum(1 for verdict in verdicts if verdict.state is not VerdictState.AGREES)
        log.log_adjudication(packets=len(packets), verdicts=len(verdicts), unavailable=unavailable)
        if self._archive:
            self._archive.journal.append(
                "adjudicated",
                packets=len(packets),
                incomplete=incomplete,
                verdicts=len(verdicts),
                path="cross-branch.md",
            )
        return verdicts, render_cross_branch(verdicts, syntheses=syntheses, adjudicator=model, incomplete=incomplete)

    async def _render_output(
        self,
        verdicts: list[CrossBranchVerdict],
        contexts: list[BranchContext],
    ) -> str:
        """Assembles the reader's language from the verdicts, never from a single branch's synthesis.

        Per claim the render carries the state, every branch's verbatim passage in its own script, and the evidence ids
        that were checked. The prose around the passages is translated into the render language, and a translation that
        fails leaves the source prose in place rather than inventing any.
        """
        render_language = self.language.render_language or self.language.source
        branches = [context.branch.language for context in contexts]
        # A fixed ASCII label, never translated: it is the machine-readable record that this file is a rendering.
        marker = (
            f'CROSSFIRE-RENDER {{"language": "{render_language}", "attested": false, '
            f'"branches": {json.dumps(branches)}, "claims": {len(verdicts)}}}'
        )
        header = (
            f"# Crossfire render ({render_language})\n\n"
            "This file is a non-attested translation assembled from the cross-branch comparison.\n"
            "It is not evidence and it is not any one branch's synthesis.\n\n"
            f"Branches: {', '.join(branches)}. Claims compared: {len(verdicts)}."
        )
        lines: list[str] = [marker, "", await self._translate_render_prose(header, render_language)]
        for verdict in verdicts:
            prose = f"## {verdict.claim}\n\nState: {verdict.state.value}."
            lines.append("")
            lines.append(await self._translate_render_prose(prose, render_language))
            for branch, passage in verdict.by_branch.items():
                lines.append(f"- {branch}: {passage or '(omitted)'}")
            if verdict.evidence_ids:
                lines.append(f"Evidence: {', '.join(verdict.evidence_ids)}.")
        rendered = "\n".join(lines).rstrip() + "\n"
        if self._archive:
            self._archive.journal.append("rendered", language=render_language, claims=len(verdicts))
        return rendered

    async def _translate_render_prose(self, text: str, target_language: str) -> str:
        """Translates one render section into *target_language*, keeping its protected spans byte-identical.

        A failed or unfaithful translation returns the source prose unchanged: a render is not attested, so a missing
        rendering is reported rather than shouted, but nothing invented is ever substituted for it.
        """
        if target_language == self.language.source:
            return text
        masked, spans = extract_do_not_translate(text)
        system, user = build_translation_prompt(f"{self.language.source}->{target_language}", masked)
        try:
            reply = await self._call_llm(
                model=self.configuration.translator.names[0],
                system_prompt=system,
                user_prompt=user,
                role=Role.TRANSLATOR,
                round_num=0,
                max_retries=0,
            )
        except _FATAL_ERRORS:
            raise
        except Exception as exception:
            log.log_translation_failed(
                branch=target_language,
                model=self.configuration.translator.names[0],
                error=f"render prose: {exception}",
            )
            return text
        translated = reply.strip()
        if not translated or fidelity_violations(masked, translated, spans):
            log.log_translation_failed(
                branch=target_language,
                model=self.configuration.translator.names[0],
                error="render prose failed the fidelity gate",
            )
            return text
        return restore_do_not_translate(translated, spans)

    def _build_evidence_registry(self, syntheses: dict[str, str]) -> EvidenceRegistry:
        """Stores the source, the context and every citation the branches produced."""
        registry = EvidenceRegistry()
        registry.add(self.parameters.task.instruction, origin="source")
        if self.parameters.task.context:
            registry.add(self.parameters.task.context, origin="context")
        for branch, synthesis in syntheses.items():
            registry.harvest(branch, synthesis)
        return registry

    def _journal_round(self, branch: str, round_num: int, synthesis_text: str) -> None:
        """Records a completed round with the hash of its synthesis artefact."""
        if self._active_archive is None:
            return
        self._active_archive.journal.append(
            "round_completed",
            branch=branch,
            round=round_num,
            path=self._active_archive.record_path(f"round-{round_num}/synthesis.md"),
            sha256=sha256_text(synthesis_text),
        )

    def _journal_artefact_matches(self, record: dict[str, Any]) -> str | None:
        """Returns the artefact's text when it still hashes to the journalled value, else ``None``."""
        if self._archive is None:
            return None
        path = str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        target = self._archive.base / path
        if not path or not expected or not target.is_file():
            return None
        text = target.read_text(encoding="utf-8")
        return text if sha256_text(text) == expected else None

    def _completed_round(self, branch: str, round_num: int) -> str | None:
        """Returns the synthesis of a journalled round whose artefact still matches, else ``None``."""
        for record in reversed(self._journal_records):
            if (
                record.get("event") == "round_completed"
                and record.get("branch") == branch
                and record.get("round") == round_num
            ):
                return self._journal_artefact_matches(record)
        return None

    def _resumed_instruction(self, branch: Branch) -> Branch | None:
        """Returns *branch* with its attested translation restored from the journal, or ``None``."""
        if branch.instruction_text or not self._journal_records:
            return None
        for record in reversed(self._journal_records):
            if record.get("event") == "instruction_attested" and record.get("branch") == branch.language:
                text = self._journal_artefact_matches(record)
                if text is None:
                    return None
                return replace(
                    branch,
                    instruction_text=text,
                    instruction_model=str(record.get("model", "")),
                    instruction_hash=sha256_text(text),
                    state=BranchState.TRANSLATED,
                )
        return None

    @staticmethod
    def _run_shape(synthesised: list[BranchContext], branches: list[Branch]) -> str:
        """Returns the run's shape: ``full``, ``partial`` or ``single-branch``.

        Fewer than two syntheses means there is nothing to compare, so the run says so instead of pretending to
        adjudicate. A branch that failed after a usable synthesis leaves the run partial and is named.
        """
        if len(synthesised) < 2:
            return "single-branch"
        if len(synthesised) < len(branches):
            return "partial"
        return "full"

    def _primary_context(self, contexts: list[BranchContext]) -> BranchContext:
        """Returns the branch whose synthesis is the run's own output: the source branch when it survived."""
        for context in contexts:
            if context.branch.language == self.language.source:
                return context
        return contexts[0]

    def _language_manifest(self) -> dict[str, Any]:
        """Returns the resolved language configuration and the layer that supplied each value."""
        language = self.language
        return {
            "source": language.source,
            "branches": list(language.branches),
            "preset": language.preset,
            "competence_branch": language.competence_branch,
            "render_language": language.render_language,
            "translate_instruction": language.translate_instruction,
            "adjudicate": language.adjudicate,
            "translator_backend": language.translator_backend,
            "provenance": dict(self.language_provenance),
        }

    def _language_tag(self) -> str:
        """Returns the response-language instruction for the running branch, or an empty string.

        A single branch in the source language already speaks the instruction's language, so it gets no tag and its
        prompts stay byte-identical to the single-language pipeline. Every other branch is told which language to answer
        in, and told to quote evidence in the language it appears in.
        """
        language = self._active_task.language
        if len(self.language.branches) == 1 and language == self.language.source:
            return ""
        return build_language_tag(language)

    def _archive_for(self, branch: Branch) -> RunArchive | None:
        """Returns the archive a branch writes to.

        One branch keeps today's layout, with every artefact at the root of the run directory. More than one branch
        writes into ``branches/<language>``, so two branches can never overwrite each other.
        """
        if self._archive is None:
            return None
        if len(self.language.branches) == 1:
            return self._archive
        return self._archive.child("branches", branch.language)

    async def _run_branch(self, context: BranchContext) -> BranchContext:
        """Runs today's round loop for one branch and returns its context.

        Enrichment runs per branch, on the branch's own instruction. A branch that produces no synthesis is marked
        ``branch_failed`` and left out of the reconciliation rather than aborting the other branches.
        """
        self._active_context = context
        self._active_task = context.task
        self._active_archive = self._archive_for(context.branch)
        if self._active_archive:
            self._active_archive.journal.append(
                "branch_started",
                branch=context.branch.language,
                instruction_hash=context.branch.instruction_hash,
            )

        if self.parameters.enrich and self.configuration.enricher.names:
            try:
                context = await self._enrich_instruction(context)
            except _FATAL_ERRORS as exception:
                raise RunFailedError(exclaim(str(exception))) from exception
            except Exception as exception:
                log.log_enrichment_failed(
                    model=self.configuration.enricher.names[0],
                    error=str(exception),
                )
            self._active_context = context
            self._active_task = context.task

        previous_synthesis: str = ""
        for round_num in range(1, self.parameters.num_rounds + 1):
            self._progress.on_round_start(round_num, self.parameters.num_rounds)
            resumed = self._completed_round(context.branch.language, round_num)
            if resumed is not None:
                context.consecutive_round_failures = 0
                previous_synthesis = resumed
                context.synthesis_history.append(previous_synthesis)
                log.log_round_resumed(branch=context.branch.language, round=round_num)
                continue
            result = await self._run_round(round_num, previous_synthesis)

            if result is not None:
                # Round succeeded: reset failure counter and update synthesis
                context.consecutive_round_failures = 0
                previous_synthesis = result.synthesis_text
                context.synthesis_history.append(previous_synthesis)
                self._journal_round(context.branch.language, round_num, previous_synthesis)

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
                context.consecutive_round_failures += 1
                if context.consecutive_round_failures >= MAX_CONSECUTIVE_ROUND_FAILURES:
                    start_round = round_num - context.consecutive_round_failures + 1
                    raise RunFailedError(
                        exclaim(
                            f"Giving up: {context.consecutive_round_failures} consecutive "
                            f"round failures ({start_round}-{round_num}). "
                            "The models are having a bad day."
                        )
                    )
        else:
            # When the loop completes without an early stop
            self._progress.on_run_end()

        if not previous_synthesis.strip():
            context.branch = replace(context.branch, state=BranchState.BRANCH_FAILED)
            if self._active_archive:
                self._active_archive.journal.append(
                    "branch_failed",
                    branch=context.branch.language,
                    reason=BranchState.BRANCH_FAILED.value,
                )
            return context

        context.branch = replace(context.branch, state=BranchState.SYNTHESIZED)
        if self._active_archive:
            self._active_archive.journal.append(
                "synthesis",
                branch=context.branch.language,
                rounds=len(context.synthesis_history),
                sha256=sha256_text(previous_synthesis),
            )
            if len(self.language.branches) > 1:
                self._active_archive.save_branch_output(previous_synthesis)
            else:
                self._active_archive.save_final_synthesis(previous_synthesis)
        return context

    # -- enrichment pre-step ---

    async def _enrich_instruction(self, context: BranchContext) -> BranchContext:
        """Rewrites this branch's instruction through the enrichment model and returns the updated context."""
        model = self.configuration.enricher.names[0]
        system, user = build_enrichment_prompt(
            mode=self.parameters.mode,
            instruction=context.task.instruction,
            context=context.task.context,
        )

        self._progress.on_phase_start(
            0,
            Phase.ENRICHMENT,
            1,
            models=[model],
            candidate_indices=[None],
            wire_model_ids=[self._wire_model_id(model)],
        )
        original_tokens = estimate_tokens(context.task.instruction)
        enriched = await self._call_llm(
            model=model,
            system_prompt=system,
            user_prompt=user,
            role=Role.ENRICHER,
            round_num=0,
            max_retries=0,
        )
        self._progress.on_task_done(0, Phase.ENRICHMENT, model=model)
        self._progress.on_phase_end(0, Phase.ENRICHMENT)

        enriched_tokens = estimate_tokens(enriched)
        log.log_prompt_enriched(
            model=model,
            original_tokens=original_tokens,
            enriched_tokens=enriched_tokens,
        )

        if self._active_archive:
            self._active_archive.save_enriched(enriched)

        context.branch = replace(
            context.branch,
            enriched_text=enriched,
            enriched_hash=sha256_text(enriched),
            state=BranchState.ENRICHED,
        )
        context.task = Task(
            instruction=enriched,
            context=context.task.context,
            language=context.task.language,
        )
        return context

    # -- early stopping ---

    @staticmethod
    def _should_stop_early(reviews: list[Review], threshold: int = 1) -> bool:
        """Returns True only when every review carries a valid control envelope reporting no material defect.

        A review's prose may be written in any language, so the loop never reads a verdict out of it. A missing or
        malformed envelope is ``envelope_invalid``: never clean, and early stop is forbidden for that round. An empty
        review list never triggers early stop either: if nobody looked, we cannot claim approval.
        """
        if not reviews:
            return False
        for review in reviews:
            try:
                envelope = parse_control_envelope(review.text)
            except EnvelopeError:
                return False
            if envelope.kind != "verdict":
                return False
            if envelope.severity == "material":
                return False
            if envelope.weaknesses > threshold:
                return False
        return True

    # -- single round ---

    def _wire_model_id(self, model: str) -> str:
        """Returns the wire model ID the active gateway actually bills for *model*."""
        return self.configuration.active_provider().resolve_wire_model_id(model)

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

        if self._active_archive:
            for candidate in candidates:
                self._active_archive.save_candidate(candidate)

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

        if self._active_archive:
            for review in reviews:
                self._active_archive.save_review(review)

        # A review without a valid envelope cannot be read as approval, so the round records it and carries on.
        for review in reviews:
            try:
                parse_control_envelope(review.text)
            except EnvelopeError as exception:
                log.log_envelope_invalid(
                    round=round_num,
                    role=Role.REVIEWER,
                    model=review.model,
                    reason=str(exception),
                )

        reviewed_candidates = {review.candidate_index for review in reviews}
        if self.parameters.num_reviewers_per_candidate > 0 and len(reviewed_candidates) < len(candidates):
            missing = sorted({candidate.index for candidate in candidates} - reviewed_candidates)
            log.log_round_failed(
                round=round_num,
                reason="incomplete_reviews",
                details=f"No reviews for candidates {missing}",
            )
            return None

        # --- synthesis phase ---
        log.log_phase_start(round=round_num, phase=Phase.SYNTHESIS)
        synthesis = await self._run_synthesis(round_num, candidates, reviews)
        log.log_phase_end(round=round_num, phase=Phase.SYNTHESIS)

        if synthesis is None:
            log.log_round_failed(
                round=round_num,
                reason="synth_failure",
                details="All synthesizer models failed",
            )
            return None

        if self._active_archive:
            self._active_archive.save_synthesis(synthesis)

        if previous_synthesis and self._is_synthesis_regression(previous_synthesis, synthesis.text):
            log.log_synthesis_regression(
                round=round_num,
                model=synthesis.model,
                reason="synthesis_regression",
            )
            return RoundResult(synthesis_text=previous_synthesis, reviews=reviews)

        return RoundResult(synthesis_text=synthesis.text, reviews=reviews)

    # Refusals and regressions come in many forms (meta-critiques, source complaints,
    # "I can't do this" responses). Regex catches obvious phrases but sophisticated
    # refusals slip through. A length check catches more: legitimate refinement may
    # trim content, but a 50%+ drop in tokens signals that the synthesizer replaced
    # substance with commentary.
    _SYNTHESIS_REGRESSION_RATIO = 0.50

    @classmethod
    def _is_synthesis_regression(cls, previous: str, current: str) -> bool:
        """Detects whether the current synthesis is a regression from the previous one."""
        if _REFUSAL_REGEX.search(current[:500]):
            return True
        previous_tokens: int = estimate_tokens(previous)
        if previous_tokens == 0:
            return False
        current_tokens: int = estimate_tokens(current)
        return current_tokens / previous_tokens < cls._SYNTHESIS_REGRESSION_RATIO

    # -- generation ---

    async def _run_generation(self, round_num: int, previous_synthesis: str) -> list[Candidate]:
        """Runs all generators in parallel and returns the surviving candidates."""
        generator_names = self.configuration.generators.names
        generator_count = self.parameters.num_generators
        generator_models = [generator_names[index % len(generator_names)] for index in range(generator_count)]
        assigned_generators: set[str] = set(generator_models)
        replacement_claim_lock: asyncio.Lock = asyncio.Lock()
        self._progress.on_phase_start(
            round_num,
            Phase.GENERATION,
            generator_count,
            models=generator_models,
            candidate_indices=[None] * generator_count,
            wire_model_ids=[self._wire_model_id(model) for model in generator_models],
        )

        async def _generate(index: int, model: str) -> Candidate:
            try:
                return await self._generate_with_fallback(
                    round_num,
                    index,
                    model,
                    previous_synthesis,
                    assigned_generators,
                    replacement_claim_lock,
                )
            finally:
                self._progress.on_task_done(round_num, Phase.GENERATION, model=model)

        tasks = []
        for index in range(generator_count):
            tasks.append(_generate(index, generator_models[index]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[Candidate] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                if isinstance(result, _FATAL_ERRORS):
                    raise RunFailedError(exclaim(str(result))) from result
                model = generator_names[index % len(generator_names)]
                log.log_model_dropped(
                    phase=Phase.GENERATION,
                    role=Role.GENERATOR,
                    model=model,
                    round=round_num,
                    wire_model_id=self._wire_model_id(model),
                    reason=f"generation_error:{type(result).__name__}",
                )
            else:
                candidates.append(cast(Candidate, result))

        self._progress.on_phase_end(round_num, Phase.GENERATION)
        return candidates

    async def _generate_with_fallback(
        self,
        round_num: int,
        index: int,
        model: str,
        previous_synthesis: str,
        assigned_generators: set[str],
        lock: asyncio.Lock,
    ) -> Candidate:
        """Attempts generation, falling back to a different model on refusal or truncation."""
        try:
            return await self._generate_candidate(round_num, index, model, previous_synthesis)
        except (RefusalError, TruncatedResponseError, RefusalResponseError) as exception:
            if isinstance(exception, TruncatedResponseError):
                reason = "truncated"
            elif isinstance(exception, RefusalResponseError):
                reason = "model_refusal"
            else:
                reason = "refusal"
            log.log_model_dropped(
                phase=Phase.GENERATION,
                role=Role.GENERATOR,
                model=model,
                round=round_num,
                wire_model_id=self._wire_model_id(model),
                reason=reason,
            )
            replacement = await self._find_replacement_generator(
                self.configuration.generators.names,
                assigned_generators,
                model,
                lock,
            )
            if replacement:
                log.log_retry(
                    round=round_num,
                    role=Role.GENERATOR,
                    model=model,
                    attempt=MAX_RETRIES + 1,
                    reason=f"{reason}, replacing with {replacement}",
                )
                return await self._generate_candidate(round_num, index, replacement, previous_synthesis)
            raise

    @staticmethod
    async def _find_replacement_generator(
        all_names: list[str] | tuple[str, ...],
        assigned: set[str],
        failed: str,
        lock: asyncio.Lock,
    ) -> str | None:
        """Atomically claims an unused generator model as a substitute for *failed*."""
        async with lock:
            for name in all_names:
                if name not in assigned and name != failed:
                    assigned.add(name)
                    return name
        return None

    async def _generate_candidate(
        self,
        round_num: int,
        index: int,
        model: str,
        previous_synthesis: str,
    ) -> Candidate:
        """Builds a prompt for one generator, calls the LLM, and handles any search request."""
        prior_search = self._active_context.search_cache.get(index, "")
        system, user = build_generator_prompt(
            mode=self.parameters.mode,
            instruction=self._active_task.instruction,
            context=self._active_task.context,
            previous_synthesis=previous_synthesis,
            round_num=round_num,
            search_results=prior_search,
            search_enabled=self.configuration.search.enabled,
            language_tag=self._language_tag(),
        )

        # Priority order for compression when the prompt exceeds the token budget:
        # 1) synthesis, 2) search, and 3) context.
        # Note: task instructions are never compressed.
        compressible_parts: list[tuple[str, str]] = []
        if previous_synthesis:
            compressible_parts.append((previous_synthesis, "candidate_truncation"))
        if prior_search:
            compressible_parts.append((prior_search, "search_truncation"))
        if self._active_task.context:
            compressible_parts.append((self._active_task.context, "context_truncation"))

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

        if _REFUSAL_REGEX.search(text[:500]):
            raise RefusalError(exclaim(f"{model} doesn't wanna: {text[:120]}"))

        text, search_results = await self._process_search_request(
            text,
            role=Role.GENERATOR,
            model=model,
            round_num=round_num,
        )
        if search_results:
            self._active_context.search_cache[index] = search_results

        return Candidate(
            text=text,
            model=model,
            round=round_num,
            index=index,
            search_results=search_results,
            language=self._active_task.language,
        )

    # -- review ---

    async def _run_review(self, round_num: int, candidates: list[Candidate]) -> list[Review] | None:
        """Assigns reviewers and runs all reviews in parallel.

        Returns ``None`` only when the reviewer pool is too small to cover every candidate. Otherwise it
        returns whatever reviews succeeded, which may be partial: coverage is judged by the caller so that
        successful reviews survive a failed round rather than being discarded with it.
        """
        if self.parameters.num_reviewers_per_candidate == 0:
            return []

        # Cross-group overlap is allowed: a model that generated a candidate
        # may also review (with a clean context).
        assignments = assign_reviewers(
            reviewers=self.configuration.reviewers.names,
            num_candidates=len(candidates),
            num_reviewers_per_candidate=self.parameters.num_reviewers_per_candidate,
            round_num=round_num,
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
            wire_model_ids=[self._wire_model_id(model) for model in review_models],
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
                if isinstance(result, _FATAL_ERRORS):
                    raise RunFailedError(exclaim(str(result))) from result
                log.log_model_dropped(
                    phase=Phase.REVIEW,
                    role=Role.REVIEWER,
                    model=reviewer_model,
                    round=round_num,
                    wire_model_id=self._wire_model_id(reviewer_model),
                    reason=f"review_error:{type(result).__name__}",
                )
                continue
            reviews.append(cast(Review, result))

        self._progress.on_phase_end(round_num, Phase.REVIEW)
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
            instruction=self._active_task.instruction,
            candidate=candidate,
            search_results=candidate.search_results,
            search_enabled=self.configuration.search.enabled,
            language_tag=self._language_tag(),
        )

        try:
            text = await self._review(
                model=reviewer_model,
                system_prompt=system,
                user_prompt_raw=user_raw,
                candidate=candidate,
                round_num=round_num,
            )
        except _REPLACEABLE_ERRORS:
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
            language=self._active_task.language,
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

    async def _run_synthesis(
        self,
        round_num: int,
        candidates: list[Candidate],
        reviews: list[Review],
    ) -> SynthesisResult | None:
        """Runs synthesis, rotating through the pool and falling back on failure.

        Returns None when every synthesizer model fails, which the round loop treats as a failed round. A refusal or
        truncation is model-specific, so the next synthesizer may still succeed; only a rejected key or an empty wallet
        aborts the run outright.
        """
        names = self.configuration.synthesizer.names
        start = (round_num - 1) % len(names)
        ordered = [names[(start + offset) % len(names)] for offset in range(len(names))]
        self._progress.on_phase_start(
            round_num,
            Phase.SYNTHESIS,
            1,
            models=[ordered[0]],
            candidate_indices=[None],
            wire_model_ids=[self._wire_model_id(ordered[0])],
        )
        try:
            for model in ordered:
                try:
                    return await self._synthesize(round_num, model, candidates, reviews)
                except _FATAL_ERRORS as exception:
                    raise RunFailedError(exclaim(str(exception))) from exception
                except Exception as exception:
                    log.log_model_dropped(
                        phase=Phase.SYNTHESIS,
                        role=Role.SYNTHESIZER,
                        model=model,
                        round=round_num,
                        wire_model_id=self._wire_model_id(model),
                        reason=f"synth_error:{type(exception).__name__}",
                    )
            return None
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
            instruction=self._active_task.instruction,
            candidates=candidates,
            reviews=reviews,
            language_tag=self._language_tag(),
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
            language=self._active_task.language,
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
    ) -> str:
        """Compresses *user_prompt* to fit the token budget.

        Raises RuntimeError on overflow, which drops the model and degrades the run instead of aborting it.
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
                wire_model_id=self._wire_model_id(model),
                tokens_before=tokens,
                tokens_after=tokens,
                reason="token_overflow",
            )
            msg: str = f"Token overflow for {role} {model}"
            raise RuntimeError(exclaim(msg))
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
            instruction=self._active_task.instruction,
            mode=self.parameters.mode.value,
            role=role,
            model=model,
            round_num=round_num,
            client=self._http_client,
            search_timeout=self.configuration.limits.search_timeout,
        )
        if results:
            self._searches_performed.append(
                {
                    "round": round_num,
                    "role": role,
                    "model": model,
                    "query": query,
                }
            )
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
        if role is Role.TRANSLATOR:
            return self.configuration.translator
        if role is Role.ADJUDICATOR:
            return self.configuration.adjudicator
        return self.configuration.enricher

    def _load_pricing(self) -> dict[str, tuple[float, float]]:
        """Lazily loads the pricing cache; returns an empty map when unavailable."""
        if self._pricing is not None:
            return self._pricing
        self._pricing = {}
        path = Path(PRICING_FILENAME)
        if path.is_file():
            try:
                self._pricing, _ = load_pricing(path)
            except (ValueError, OSError):
                self._pricing = {}
        return self._pricing

    def _compute_cost(self, provider_name: str, wire_model_id: str, usage: Usage) -> float | None:
        """Derives a cost from the pricing table when the provider does not report one.

        The pricing table has no cache-specific rates. If the usage report includes cache reads or writes and the
        provider reports no cost figure, this function returns ``None``. The caller lists the model in
        ``unpriced_models``.
        """
        if usage.cache_read_tokens or usage.cache_write_tokens:
            return None
        pricing = self._load_pricing()
        entry = pricing.get(f"{provider_name}::{wire_model_id}") or pricing.get(wire_model_id)
        if entry is None:
            return None
        input_price, output_price = entry
        return usage.input_tokens * input_price + usage.output_tokens * output_price

    async def _call_llm(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        role: Role,
        round_num: int,
        dry_run_candidate_index: int | None = None,
        max_retries: int | None = None,
    ) -> str:
        """Dispatches to a simulated response (dry run) or the active provider's API.

        *max_retries* defaults to ``MAX_RETRIES``; the optional enrichment step passes 0 so a flaky model cannot
        stall the run with repeated timeouts.
        """
        if self.parameters.dry_run:
            return simulate_response(
                instruction=self._active_task.instruction,
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

        provider = self.configuration.active_provider()
        wire_model_id = provider.resolve_wire_model_id(model)
        protocol = resolve_protocol(provider, wire_model_id)
        # Frontier models on the messages/responses protocols reject sampling controls, so temperature is only
        # sent on the OpenAI-compatible chat protocol.
        temperature = self.configuration.limits.temperature_for(role) if protocol == "chat" else None
        reasoning_effort = self.configuration.limits.reasoning_effort_for(role)

        # Both the HTTP call and response extraction are inside the retry loop
        # so that empty-but-200 responses (EmptyResponseError) are retried.
        async def _call_and_extract() -> dict[str, Any]:
            data: dict[str, Any] = await call_model(
                provider=provider,
                protocol=protocol,
                wire_model_id=wire_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=self._api_key,
                max_tokens=self._role_group(role).resolve_max_output_tokens(model),
                temperature=temperature,
                semaphore=self._semaphore,
                client=http_client,
                session_id=self._session_id,
                reasoning_effort=reasoning_effort,
            )
            extract_response_text(protocol, data)  # raises Empty/TruncatedResponseError
            return data

        data: dict[str, Any] = await call_with_retry(
            _call_and_extract,
            role=role,
            model=model,
            round_num=round_num,
            provider_name=provider.name,
            max_retries=MAX_RETRIES if max_retries is None else max_retries,
        )

        usage = extract_usage(protocol, data)
        cost = usage.cost if usage.cost is not None else self._compute_cost(provider.name, wire_model_id, usage)
        self.cost_tracker.record(
            CostEntry(
                model=model,
                role=role,
                round=round_num,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                cost=cost,
                provider=provider.name,
                wire_model_id=wire_model_id,
                branch=self._active_task.language,
            )
        )

        return extract_response_text(protocol, data)
