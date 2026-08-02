"""
The weakness-mining stage: completed runs in, evidence bundle out.

The miner does not execute the harness. It takes runs that already happened,
which keeps the experiment driver's concerns (splits, repetitions, budgets) out
of the mining code and keeps mining testable without a live model.
"""

from dataclasses import dataclass, field
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.attribution import AttributionRejection, LLMAttributor
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
    EvidenceBundle,
    FailureRecord,
    MiningConfig,
    MiningTotals,
    SubVerifier,
    Verifier,
)
from shrlm.optimization.walker import walk


@dataclass
class MiningResult:
    """The bundle, plus the per-record material behind it."""

    bundle: EvidenceBundle
    records: list[FailureRecord] = field(default_factory=list)
    raw_attributions: list[dict[str, Any]] = field(default_factory=list)


class WeaknessMiner:
    """
    Converts verifier-grounded traces into recurring, actionable failure patterns.

    ``sub_verifier`` is the single ablation switch. Supplying one makes the
    failing level a checkable fact derived from per-sub-call outcomes;
    withholding it hands that judgment to the attributor and marks every record
    ungrounded. Nothing else in the pipeline varies with the choice.
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
        self, instance: dict[str, Any], completion: RLMChatCompletion
    ) -> tuple[FailureRecord | None, dict[str, Any] | None, float]:
        """
        Verify one run and, if it failed, attribute it.

        Returns (record, raw attribution, digest coverage). The record is None
        for a passing run.
        """
        instance_id = str(instance["id"])
        verdict = self.verifier(instance, completion.response)
        if verdict.passed:
            return None, None, 1.0

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

        # The single try in this package. One unusable attribution must not
        # abort a mining round, but it must remain visible in the totals rather
        # than being coerced into a label the model did not produce.
        try:
            signature, detail = self.attributor.attribute(digest, root, verdict, grounding)
        except AttributionRejection as exc:
            record.attribution_failed = True
            record.attribution_error = str(exc)
            return record, None, digest.coverage

        record.signature = signature
        record.detail = detail
        raw = {
            "instance_id": instance_id,
            "digest_sha256": digest.sha256,
            "signature": signature.to_dict(),
            "detail": detail.to_dict(),
            "level_grounded": grounding.grounded,
        }
        return record, raw, digest.coverage

    def mine(
        self,
        runs: list[tuple[dict[str, Any], RLMChatCompletion]],
        round_index: int,
        harness_version: str,
        split_id: str,
        created_at: str | None = None,
    ) -> MiningResult:
        """Run the full stage over a set of completed runs."""
        records: list[FailureRecord] = []
        raw_attributions: list[dict[str, Any]] = []
        coverages: list[float] = []

        for instance, completion in runs:
            record, raw, coverage = self.record_failure(instance, completion)
            if record is None:
                continue
            records.append(record)
            coverages.append(coverage)
            if raw is not None:
                raw_attributions.append(raw)

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

        config = MiningConfig(
            round_index=round_index,
            harness_version=harness_version,
            split_id=split_id,
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version=self.attributor.config.prompt_version,
            digest_version=DIGEST_VERSION,
            prompt_sha256=self.attributor.prompt_sha256(self.sub_verifier is not None),
            attributor_model=self.attributor.lm.model_name,
            attributor_sampling_args=dict(self.attributor.lm.sampling_args),
            digest_char_budget=self.digest_config.char_budget,
            digest_focus_k=self.digest_config.focus_k,
            max_attempts=self.attributor.config.max_attempts,
            sub_verifier_enabled=self.sub_verifier is not None,
            min_support=self.clustering_config.min_support,
            actionability_weights=dict(ACTIONABILITY_WEIGHTS),
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

        return MiningResult(bundle=bundle, records=records, raw_attributions=raw_attributions)
