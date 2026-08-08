"""
The weakness-mining stage: completed runs in, evidence bundle out.

The miner does not execute the harness. It takes runs that already happened,
which keeps the experiment driver's concerns (splits, repetitions, budgets) out
of the mining code and keeps mining testable without a live model.

Verdicts may be precomputed. By default the miner recomputes each verdict from
the completion's response, but a caller that already holds one -- the driver
persists verdicts at run time, including RESOURCE_TERMINATED verdicts the
Verifier protocol can never produce because it is handed only a response
string, never an exception -- passes it through instead and the verifier is not
consulted for that run.
"""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.attribution import (
    ATTRIBUTOR_SYSTEM_PROMPT,
    VALIDATOR_VERSION,
    AttributionRejection,
    AttributionTransportError,
    LLMAttributor,
)
from shrlm.optimization.bundle import build_evidence_bundle
from shrlm.optimization.clustering import (
    ACTIONABILITY_WEIGHTS,
    ClusteringConfig,
    cluster_failures,
    compute_marginals,
)
from shrlm.optimization.digest import DIGEST_VERSION, DigestConfig, build_digest
from shrlm.optimization.grounding import apply_sub_verifier
from shrlm.optimization.taxonomy import TAXONOMY_VERSION
from shrlm.optimization.types import (
    AttributionErrorKind,
    EvidenceBundle,
    FailureRecord,
    MiningConfig,
    MiningTotals,
    RunTraceLink,
    SubVerifier,
    Verdict,
    Verifier,
)
from shrlm.optimization.walker import walk


@dataclass
class MiningResult:
    """The bundle, plus the per-record material behind it.

    ``digest_texts`` maps each failure record's ``digest_sha256`` to the exact
    digest text the attributor saw, and ``attributor_prompts`` maps prompt
    sha256 to each rendered system prompt variant used during the round --
    together they make every attribution replayable from disk, not just
    hash-checkable. ``errors`` names the runs whose attribution failed at the
    transport level (LM unreachable after retries): those records are still
    present, marked unattributed, but the round completed instead of raising.
    """

    bundle: EvidenceBundle
    records: list[FailureRecord] = field(default_factory=list)
    raw_attributions: list[dict[str, Any]] = field(default_factory=list)
    digest_texts: dict[str, str] = field(default_factory=dict)
    attributor_prompts: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FailureOutcome:
    """What one run contributed to the round.

    ``record`` is None for a passing run. ``transport_error`` is set when the
    attributor's LM stayed unreachable after bounded retries -- the record is
    then marked unattributed and ``mine`` checkpoints instead of raising.
    """

    record: FailureRecord | None
    raw: dict[str, Any] | None
    coverage: float
    digest_text: str | None = None
    prompt_text: str | None = None
    prompt_sha256: str | None = None
    transport_error: str | None = None


class WeaknessMiner:
    """
    Converts verifier-grounded traces into recurring, actionable failure patterns.

    ``sub_verifier`` is the single ablation switch. Supplying one makes the
    failing level a checkable fact derived from per-sub-call outcomes;
    withholding it hands that judgment to the attributor and marks every record
    ungrounded. Nothing besides the checkability of the failing level and the
    sub-verdict evidence the digest deliberately surfaces (the per-call verdict
    column and verdict-aware focus-excerpt selection) varies with the choice.
    """

    def __init__(
        self,
        verifier: Verifier,
        attributor: LLMAttributor,
        sub_verifier: SubVerifier | None = None,
        digest_config: DigestConfig | None = None,
        clustering_config: ClusteringConfig | None = None,
    ):
        self.verifier = verifier
        self.attributor = attributor
        self.sub_verifier = sub_verifier
        self.digest_config = digest_config or DigestConfig()
        self.clustering_config = clustering_config or ClusteringConfig()

    def record_failure(
        self,
        instance: dict[str, Any],
        completion: RLMChatCompletion,
        verdict: Verdict | None = None,
    ) -> FailureOutcome:
        """
        Verify one run and, if it failed, attribute it.

        Returns a ``FailureOutcome``; its record is None for a passing run.
        When ``verdict`` is supplied it is used as-is and the verifier is not
        called -- the path a driver takes when replaying persisted verdicts,
        and the only way a RESOURCE_TERMINATED verdict can enter mining.

        The raw attribution payload is produced for every failure record,
        including rejected and transport-failed ones, and carries the full
        per-attempt audit trail: attempt number, raw response, and the named
        violation each rejected response was re-asked over.
        """
        instance_id = str(instance["id"])
        if verdict is None:
            verdict = self.verifier(instance, completion.response)
        if verdict.passed:
            return FailureOutcome(record=None, raw=None, coverage=1.0)

        root, stats = walk(completion)
        grounding = apply_sub_verifier(instance, root, self.sub_verifier)
        digest = build_digest(
            instance_id=instance_id,
            question=str(instance.get("question", "")),
            root=root,
            stats=stats,
            verdict=verdict,
            cfg=self.digest_config,
        )

        record = FailureRecord(
            instance_id=instance_id,
            verdict=verdict,
            stats=stats,
            signature=None,
            detail=None,
            level_grounded=grounding.grounded,
            sub_verdicts=grounding.verdicts,
            digest_sha256=digest.sha256,
        )

        no_subcalls = digest.no_subcalls
        prompt_text = self.attributor.system_prompt(
            grounding.grounded, digest.aggregated, no_subcalls
        )
        prompt_sha = self.attributor.prompt_sha256(
            grounding.grounded, digest.aggregated, no_subcalls
        )
        raw: dict[str, Any] = {
            "instance_id": instance_id,
            "digest_sha256": digest.sha256,
            "prompt_sha256": prompt_sha,
            "level_grounded": grounding.grounded,
        }
        outcome = FailureOutcome(
            record=record,
            raw=raw,
            coverage=digest.coverage,
            digest_text=digest.text,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha,
        )

        # The only try/except over attribution in this package. One unusable
        # attribution must not abort a mining round, but it must remain visible
        # in the totals rather than being coerced into a label the model did
        # not produce; a transport failure must checkpoint, not raise away the
        # round's completed records.
        try:
            result = self.attributor.attribute(digest, root, verdict, grounding)
        except AttributionRejection as exc:
            record.attribution_failed = True
            record.attribution_error = str(exc)
            record.attribution_error_kind = AttributionErrorKind.REJECTION
            raw.update(
                signature=None,
                detail=None,
                attributed=False,
                error=str(exc),
                attribution_error_kind=record.attribution_error_kind.value,
                attempts=[attempt.to_dict() for attempt in exc.attempts],
            )
            return outcome
        except AttributionTransportError as exc:
            record.attribution_failed = True
            record.attribution_error = f"transport failure: {exc}"
            record.attribution_error_kind = AttributionErrorKind.TRANSPORT
            raw.update(
                signature=None,
                detail=None,
                attributed=False,
                error=record.attribution_error,
                attribution_error_kind=record.attribution_error_kind.value,
                attempts=[attempt.to_dict() for attempt in exc.attempts],
            )
            outcome.transport_error = str(exc)
            return outcome

        record.signature = result.signature
        record.detail = result.detail
        raw.update(
            signature=result.signature.to_dict(),
            detail=result.detail.to_dict(),
            attributed=True,
            attempts=[attempt.to_dict() for attempt in result.attempts],
        )
        return outcome

    def mine(
        self,
        runs: list[tuple[dict[str, Any], RLMChatCompletion]],
        round_index: int,
        harness_version: str,
        split_id: str,
        created_at: str | None = None,
        verdicts: Sequence[Verdict | None] | None = None,
        trace_links: Sequence[RunTraceLink | None] | None = None,
        harness_hash: str = "",
        sampling_seed: int | None = None,
        attribution_cache_path: str | None = None,
    ) -> MiningResult:
        """Run the full stage over a set of completed runs.

        ``verdicts``, when given, is aligned index-for-index with ``runs``: a
        Verdict replaces the verifier's judgment for that run, and None falls
        back to recomputing it. Omitting the argument keeps the original
        behavior for every existing caller.

        ``trace_links`` is likewise aligned index-for-index with ``runs`` and
        stamps each failure record with the run_id / trace_path / trace_sha256
        of its persisted trace; ``mine_round`` supplies it from ``runs.jsonl``,
        and legacy in-memory callers leave records unlinked.

        Provenance: ``harness_hash`` is the round's ``harness.json`` content
        hash (``harness_version`` stays the caller's label, which
        ``mine_round`` defaults to that same hash); ``sampling_seed`` is the
        dataset sampling seed the instances carry; ``attribution_cache_path``
        overrides the attributor cache's own path (``mine_round`` passes it
        round-dir-relative), falling back to the cache's configured path --
        but either value is stamped into the config only when the cache file
        actually exists once mining finishes, since an all-pass round never
        writes one and a stamped path to a nonexistent file would break the
        audit's ``attribution_cache`` link on a perfectly clean round. The
        verifier's ``config()`` payload, when it defines one, is recorded as
        ``verifier_config``. All of these serialize into the MiningConfig and
        therefore into the bundle id.

        Never raises for an unattributable run: rejected attributions become
        unattributed records, and transport failures (LM unreachable after
        bounded retries) additionally land in ``MiningResult.errors`` so the
        round checkpoints with every completed record intact.
        """
        if verdicts is not None and len(verdicts) != len(runs):
            raise ValueError(
                f"verdicts must align with runs one-to-one: got {len(verdicts)} verdicts "
                f"for {len(runs)} runs"
            )
        if trace_links is not None and len(trace_links) != len(runs):
            raise ValueError(
                f"trace_links must align with runs one-to-one: got {len(trace_links)} links "
                f"for {len(runs)} runs"
            )

        records: list[FailureRecord] = []
        raw_attributions: list[dict[str, Any]] = []
        coverages: list[float] = []
        digest_texts: dict[str, str] = {}
        attributor_prompts: dict[str, str] = {}
        errors: list[dict[str, Any]] = []

        for index, (instance, completion) in enumerate(runs):
            verdict = verdicts[index] if verdicts is not None else None
            outcome = self.record_failure(instance, completion, verdict=verdict)
            if outcome.record is None:
                continue
            link = trace_links[index] if trace_links is not None else None
            if link is not None:
                outcome.record.run_id = link.run_id
                outcome.record.trace_path = link.trace_path
                outcome.record.trace_sha256 = link.trace_sha256
            records.append(outcome.record)
            coverages.append(outcome.coverage)
            if outcome.raw is not None:
                raw_attributions.append(outcome.raw)
            if outcome.digest_text is not None:
                digest_texts[outcome.record.digest_sha256] = outcome.digest_text
            if outcome.prompt_text is not None and outcome.prompt_sha256 is not None:
                attributor_prompts[outcome.prompt_sha256] = outcome.prompt_text
            if outcome.transport_error is not None:
                errors.append(
                    {
                        "instance_id": outcome.record.instance_id,
                        "run_index": index,
                        "error": outcome.transport_error,
                    }
                )

        patterns = cluster_failures(records, self.clustering_config)
        marginals = compute_marginals(records)

        totals = MiningTotals(
            n_runs=len(runs),
            n_failures=len(records),
            n_attributed=sum(1 for record in records if not record.attribution_failed),
            n_unattributed=sum(1 for record in records if record.attribution_failed),
            n_grounded=sum(1 for record in records if record.level_grounded),
            n_degraded_trees=sum(
                1 for record in records if record.stats.suspected_lost_subcalls > 0
            ),
        )

        # The verifier's own configuration is provenance when it exposes one
        # (duck-typed ``config()``, as GraphWalksVerifier does): two rounds
        # judged under different verifier settings must not share a bundle id.
        config_method = getattr(self.verifier, "config", None)
        verifier_config: dict[str, Any] = dict(config_method()) if callable(config_method) else {}

        # Stamp the cache path only when the cache file exists now that every
        # attribution has run: an all-pass round never calls put(), so the
        # file is never created and a stamped path would be a broken link the
        # audit rightly rejects.
        cache_path = self.attributor.cache.path
        stamped_cache_path = (
            attribution_cache_path if attribution_cache_path is not None else cache_path
        )
        if cache_path is not None and not os.path.exists(cache_path):
            stamped_cache_path = None

        config = MiningConfig(
            round_index=round_index,
            harness_version=harness_version,
            split_id=split_id,
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version=self.attributor.config.prompt_version,
            digest_version=DIGEST_VERSION,
            # The TEMPLATE pin: the sha256 of the raw system-prompt template,
            # before any per-variant rendering. Records render per-variant
            # prompts, so no single rendered variant can speak for the round;
            # the rendered shas live on each attributions.jsonl entry, which
            # the audit resolves against the persisted prompt files.
            prompt_sha256=hashlib.sha256(ATTRIBUTOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            attributor_model=self.attributor.lm.model_name,
            attributor_sampling_args=dict(self.attributor.lm.sampling_args),
            digest_char_budget=self.digest_config.char_budget,
            digest_focus_k=self.digest_config.focus_k,
            max_attempts=self.attributor.config.max_attempts,
            sub_verifier_enabled=self.sub_verifier is not None,
            min_support=self.clustering_config.min_support,
            actionability_weights=dict(ACTIONABILITY_WEIGHTS),
            verifier_config=verifier_config,
            sampling_seed=sampling_seed,
            validator_version=VALIDATOR_VERSION,
            attribution_cache_path=stamped_cache_path,
            harness_hash=harness_hash,
        )

        bundle = build_evidence_bundle(
            config=config,
            records=records,
            patterns=patterns,
            marginals=marginals,
            totals=totals,
            digest_coverages=coverages,
            created_at=created_at,
        )

        return MiningResult(
            bundle=bundle,
            records=records,
            raw_attributions=raw_attributions,
            digest_texts=digest_texts,
            attributor_prompts=attributor_prompts,
            errors=errors,
        )
