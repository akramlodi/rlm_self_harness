"""Stage-3 proposal validation, promotion half: the pure offline decision (R6, R7).

Everything here is arithmetic over U3's persisted ``summary.json`` payloads and
composition over live harnesses -- no model calls, no filesystem (KTD5). Three
layers, in the order U6's ``validate_round`` will compose them:

1. **The acceptance rule plus the band** (``score_candidate``). The paper's
   rule on pass counts aggregated across repeats, with configurable noise
   margins: delta_in >= -tau_reg and delta_ho >= -tau_reg and
   max(delta_in, delta_ho) > tau_imp. Both thresholds default to 0, which
   reproduces the paper's exact rule (delta >= 0 on both splits, max > 0);
   the pilot's preregistered margins drop into ``PromotionConfig`` and both
   values are recorded on every decision record. On top of the rule sits
   proposal.tex 3.3.3's preregistered band: candidate mean cost and mean
   sub-calls must fall within multiplier bounds relative to the baseline's
   means (inclusive at both ends; the defaults are unconstrained, again
   reproducing the paper). Both the rule and the band must pass.
2. **Round assessment** (``assess_round`` / ``decide_subject``). Candidates
   already rejected upstream -- at a loader gate or by the spend breaker --
   enter as their structured rejections and are never re-scored; each becomes
   a ``rejected`` or ``over_budget`` decision record with its reason.
3. **Selection and merge construction** (``plan_promotion``,
   ``merge_harnesses``). Among accepted candidates, edits are compatible when
   their edited surfaces are pairwise disjoint. Same-surface accepted edits
   cannot merge: the one with the higher held-out delta (tiebreak: held-in
   delta, then lexicographically smaller candidate id) survives, the rest are
   recorded as excluded. One survivor promotes alone; several survivors are
   composed onto the incumbent -- each edit's surface value replacing the
   incumbent's -- into the merged harness, which is the round's promotion
   artifact. U6 re-evaluates that artifact through U3 and this same rule
   before promotion is final; ``apply_merge_verdict`` turns the re-evaluation
   into the round's outcome. When the merge fails, the round promotes
   *nothing*: falling back to individually accepted candidates after seeing
   merge results would be post-hoc selection the preregistered rule never
   validated, so the constituents are ledgered as ``merged_failed`` instead.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import CandidateRejection, LoadedCandidate
from shrlm.optimization.costs import OUTCOME_COMPLETED
from shrlm.optimization.validation import (
    SPLIT_HELDIN,
    SPLIT_HELDOUT,
    RoundEvaluation,
    SubjectEvaluation,
)
from shrlm.rlm_harness import Harness

# The decision vocabulary the promotion ledger (U5) records. ``accepted`` and
# ``rejected`` come from the rule+band; ``over_budget`` from the breaker;
# ``merged_failed`` marks accepted constituents of a merge that failed its own
# re-evaluation; ``promoted`` is final and assigned only after any merge leg.
DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISION_OVER_BUDGET = "over_budget"
DECISION_MERGED_FAILED = "merged_failed"
DECISION_PROMOTED = "promoted"

# What one round plans to promote: nothing, one candidate, or a merged harness.
PLAN_NONE = "none"
PLAN_SINGLE = "single"
PLAN_MERGE = "merge"

# The subject id a merged harness evaluates under (filesystem-safe, and -- like
# ``BASELINE_ID`` -- never a legal candidate id, so directories cannot collide).
MERGED_SUBJECT_ID = "merged"

# The two band-checked metrics: the recorded name (the overall per-run mean
# across both splits), the summary's per-split total it is computed from, and
# the ``PromotionConfig`` field naming its band.
_BAND_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mean_cost", "total_cost", "cost_band"),
    ("mean_sub_calls", "total_sub_calls", "sub_call_band"),
)

# The ``Harness`` fields each surface id owns (builder convention: build_<x>
# fills <x>). The merge builder composes edits through this map; a test asserts
# it covers ``SURFACE_SERIALIZATION_KEYS`` exactly.
SURFACE_HARNESS_FIELDS: dict[str, tuple[str, ...]] = {
    "S1": ("repl_contract",),
    "S2": ("decomposition_instruction",),
    "S3": ("execution_instruction",),
    "S4": ("verification_instruction",),
    "S5": ("recovery_instruction",),
    "S6": ("runtime_policy",),
    "S7": ("metadata",),
    "S8": ("repl_helpers", "sub_repl_helpers"),
    "S9": ("answer_middleware",),
}


@dataclass(frozen=True)
class Band:
    """Inclusive multiplier bounds relative to the baseline's mean.

    A candidate mean is within the band when
    ``lower * baseline <= candidate <= upper * baseline``, both ends
    inclusive. The default ``[0, inf)`` constrains nothing -- the paper's rule
    with no band -- so the preregistered multipliers are opt-in config, never
    invented here. A zero baseline mean with a finite upper bound demands a
    zero candidate mean (the bound is computed by multiplication, never by a
    ratio, so a zero baseline is ordinary arithmetic, not a special case).
    """

    lower: float = 0.0
    upper: float = math.inf

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, bool)
            or not isinstance(self.lower, int | float)
            or self.lower < 0
        ):
            raise ValueError(f"band lower multiplier must be a number >= 0, got {self.lower!r}")
        if isinstance(self.upper, bool) or not isinstance(self.upper, int | float):
            raise ValueError(f"band upper multiplier must be a number, got {self.upper!r}")
        if self.upper < self.lower:
            raise ValueError(
                f"band upper multiplier {self.upper} is below the lower multiplier {self.lower}"
            )

    def contains(self, baseline_mean: float, candidate_mean: float) -> bool:
        """Whether ``candidate_mean`` sits within the band around ``baseline_mean``."""
        if candidate_mean < self.lower * baseline_mean:
            return False
        # inf * 0.0 is nan, so an infinite upper bound short-circuits instead.
        return math.isinf(self.upper) or candidate_mean <= self.upper * baseline_mean


@dataclass(frozen=True)
class PromotionConfig:
    """The preregistered promotion parameters (R6).

    ``tau_regression`` (noise tolerance on either split's pass-count delta)
    and ``tau_improvement`` (the margin at least one split must clear) default
    to 0, reproducing the paper's exact rule; the bands default to
    unconstrained. The pilot's preregistered values drop in here, and every
    decision record carries the thresholds it was decided under.
    """

    tau_regression: float = 0.0
    tau_improvement: float = 0.0
    cost_band: Band = field(default_factory=Band)
    sub_call_band: Band = field(default_factory=Band)

    def __post_init__(self) -> None:
        for name in ("tau_regression", "tau_improvement"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class CandidateDecision:
    """One subject's promotion decision, in the shape the ledger serializes.

    ``rule`` carries per-split pass counts and deltas, ``band`` the per-metric
    baseline/candidate means and bounds; both are None exactly when the
    subject was never scored (rejected upstream or over budget, in which case
    ``upstream`` or ``reasons`` say why). The thresholds are recorded on every
    record, scored or not -- they are round-level preregistration, and a
    ledger row must be interpretable without the config that produced it.
    """

    subject_id: str
    decision: str
    reasons: tuple[str, ...]
    tau_regression: float
    tau_improvement: float
    harness_hash: str | None = None
    rule: dict[str, Any] | None = None
    band: dict[str, Any] | None = None
    upstream: dict[str, str] | None = None

    @property
    def accepted(self) -> bool:
        return self.decision == DECISION_ACCEPTED

    def delta(self, split_id: str) -> int:
        """The pass-count delta on one split; only scored decisions have one."""
        if self.rule is None:
            raise ValueError(f"decision for {self.subject_id!r} was never scored; it has no deltas")
        return int(self.rule[split_id]["delta"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "tau_regression": self.tau_regression,
            "tau_improvement": self.tau_improvement,
            "harness_hash": self.harness_hash,
            "rule": self.rule,
            "band": self.band,
            "upstream": self.upstream,
        }


@dataclass(frozen=True)
class PromotionPlan:
    """What one round's accepted candidates resolve to (R7's pure half).

    ``harness`` is the promotion artifact: the single winner's own harness, or
    the merged composition -- which U6 must re-evaluate through U3 and
    ``score_candidate`` before promotion is final. ``excluded`` records
    accepted candidates the selection dropped (same-surface losers) with the
    reason, so the ledger never loses them.
    """

    kind: str
    constituent_ids: tuple[str, ...]
    harness: Harness | None
    harness_hash: str | None
    excluded: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The rule and the band: scoring one summary against the baseline's
# ---------------------------------------------------------------------------


def _scoring_violation(baseline: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """Why these two summaries cannot be compared, or None when they can."""
    if baseline["outcome"] != OUTCOME_COMPLETED:
        return (
            f"the baseline summary is {baseline['outcome']!r}, not completed; a partial "
            "baseline cannot anchor any delta"
        )
    if candidate["outcome"] != OUTCOME_COMPLETED:
        return (
            f"candidate {candidate['subject_id']!r} is {candidate['outcome']!r}; over_budget "
            "candidates are ledgered as such, never scored"
        )
    for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
        base_runs = baseline["splits"][split_id]["n_runs"]
        cand_runs = candidate["splits"][split_id]["n_runs"]
        if base_runs != cand_runs or base_runs == 0:
            return (
                f"{split_id} n_runs differ (baseline {base_runs}, candidate {cand_runs}) or are "
                "zero; pass-count deltas need one shared, non-empty denominator"
            )
    return None


def _overall_mean(summary: dict[str, Any], total_key: str) -> float:
    """One metric's per-run mean across both splits, from the persisted totals.

    The band compares whole evaluation footprints, so the figure aggregates
    across splits (weighting each split by its run count) instead of banding
    each split separately.
    """
    total = 0.0
    runs = 0
    for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
        split = summary["splits"][split_id]
        total += float(split[total_key])
        runs += int(split["n_runs"])
    return total / runs


def score_candidate(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    config: PromotionConfig,
) -> CandidateDecision:
    """Apply the acceptance rule and the band to one completed evaluation (R6).

    Pure arithmetic over two persisted ``summary.json`` payloads. Every
    failing check contributes a reason -- the ledger reports all of them, not
    the first.

    Args:
        baseline_summary: The incumbent's summary payload.
        candidate_summary: The candidate's (or merged harness's) summary.
        config: The preregistered thresholds and bands.

    Returns:
        An ``accepted`` or ``rejected`` decision with the rule and band
        figures recorded.

    Raises:
        ValueError: If either summary is not a completed evaluation or the
            run counts differ per split -- misuses, not candidate rejections.
    """
    violation = _scoring_violation(baseline_summary, candidate_summary)
    if violation:
        raise ValueError(violation)

    reasons: list[str] = []
    rule: dict[str, Any] = {}
    deltas: dict[str, int] = {}
    for split_id in (SPLIT_HELDIN, SPLIT_HELDOUT):
        base_split = baseline_summary["splits"][split_id]
        cand_split = candidate_summary["splits"][split_id]
        delta = int(cand_split["pass_count"]) - int(base_split["pass_count"])
        deltas[split_id] = delta
        rule[split_id] = {
            "n_runs": base_split["n_runs"],
            "baseline_pass_count": base_split["pass_count"],
            "candidate_pass_count": cand_split["pass_count"],
            "delta": delta,
        }
        if delta < -config.tau_regression:
            reasons.append(
                f"{split_id} pass-count delta {delta} regresses beyond "
                f"tau_regression={config.tau_regression}"
            )
    if max(deltas.values()) <= config.tau_improvement:
        reasons.append(
            f"no split improves beyond tau_improvement={config.tau_improvement} "
            f"(heldin {deltas[SPLIT_HELDIN]:+d}, heldout {deltas[SPLIT_HELDOUT]:+d})"
        )

    band: dict[str, Any] = {}
    for metric, total_key, band_field in _BAND_METRICS:
        metric_band: Band = getattr(config, band_field)
        baseline_mean = _overall_mean(baseline_summary, total_key)
        candidate_mean = _overall_mean(candidate_summary, total_key)
        within = metric_band.contains(baseline_mean, candidate_mean)
        band[metric] = {
            "baseline": baseline_mean,
            "candidate": candidate_mean,
            "lower": metric_band.lower,
            "upper": metric_band.upper,
            "within": within,
        }
        if not within:
            reasons.append(
                f"{metric} {candidate_mean} is outside the band "
                f"[{metric_band.lower}, {metric_band.upper}] x baseline {baseline_mean}"
            )

    return CandidateDecision(
        subject_id=str(candidate_summary["subject_id"]),
        decision=DECISION_REJECTED if reasons else DECISION_ACCEPTED,
        reasons=tuple(reasons),
        tau_regression=config.tau_regression,
        tau_improvement=config.tau_improvement,
        harness_hash=str(candidate_summary["harness_hash"]),
        rule=rule,
        band=band,
    )


# ---------------------------------------------------------------------------
# Assessing subjects and rounds: upstream rejections pass through
# ---------------------------------------------------------------------------


def decide_subject(
    baseline_summary: dict[str, Any],
    subject: SubjectEvaluation | CandidateRejection,
    config: PromotionConfig,
) -> CandidateDecision:
    """One subject's decision record, whatever happened to it upstream.

    A loader/caps ``CandidateRejection`` becomes a ``rejected`` record
    carrying the gate verdict verbatim; an over-budget evaluation becomes
    ``over_budget``; only a completed evaluation is scored. This is also the
    entry point U6's merge leg uses on the merged harness's evaluation.
    """
    if isinstance(subject, CandidateRejection):
        return CandidateDecision(
            subject_id=subject.candidate_id,
            decision=DECISION_REJECTED,
            reasons=(f"rejected upstream at gate {subject.gate}: {subject.reason}",),
            tau_regression=config.tau_regression,
            tau_improvement=config.tau_improvement,
            upstream=subject.to_dict(),
        )
    if subject.over_budget:
        return CandidateDecision(
            subject_id=subject.subject_id,
            decision=DECISION_OVER_BUDGET,
            reasons=(
                f"evaluation stopped over budget after spending {subject.summary['spent']}; "
                "a partial evaluation is never scored",
            ),
            tau_regression=config.tau_regression,
            tau_improvement=config.tau_improvement,
            harness_hash=subject.harness_hash,
        )
    return score_candidate(baseline_summary, subject.summary, config)


def assess_round(evaluation: RoundEvaluation, config: PromotionConfig) -> list[CandidateDecision]:
    """One decision record per candidate, in the round's order.

    Raises:
        ValueError: If the baseline itself ran over budget -- no candidate
            delta is measurable against a partial baseline, so the round is a
            misconfigured experiment, not a promotes-nothing outcome.
    """
    if evaluation.baseline.over_budget:
        raise ValueError(
            "the baseline evaluation ran over budget; no delta is measurable against a "
            "partial baseline, so the round must be re-run with a workable budget"
        )
    return [
        decide_subject(evaluation.baseline.summary, candidate, config)
        for candidate in evaluation.candidates
    ]


# ---------------------------------------------------------------------------
# Selection and merge construction (R7, the pure half)
# ---------------------------------------------------------------------------


def merge_harnesses(incumbent: Harness, edits: Iterable[LoadedCandidate]) -> Harness:
    """Compose disjoint-surface edits onto the incumbent (R7's merge builder).

    Each edit contributes its edited surface's live field values from its own
    gated harness; every other surface stays the incumbent's object -- the
    same reuse stance as the loader's host materialization (KTD2). The name
    concatenates incumbent and constituent ids (sorted, so composition order
    cannot matter), and names are excluded from the harness hash anyway.

    Raises:
        ValueError: If two edits claim the same surface; same-surface edits
            are incompatible by definition and must go through selection.
    """
    ordered = sorted(edits, key=lambda edit: edit.candidate_id)
    claimed: dict[str, str] = {}
    fields: dict[str, Any] = {}
    for edit in ordered:
        if edit.surface in claimed:
            raise ValueError(
                f"candidates {claimed[edit.surface]!r} and {edit.candidate_id!r} both edit "
                f"surface {edit.surface}; same-surface edits cannot merge"
            )
        claimed[edit.surface] = edit.candidate_id
        for field_name in SURFACE_HARNESS_FIELDS[edit.surface]:
            fields[field_name] = getattr(edit.harness, field_name)
    name = "+".join([incumbent.name, *(edit.candidate_id for edit in ordered)])
    return replace(incumbent, name=name, **fields)


def plan_promotion(
    incumbent: Harness,
    decisions: Iterable[CandidateDecision],
    candidates: Iterable[LoadedCandidate],
) -> PromotionPlan:
    """Resolve a round's accepted candidates into its promotion artifact (R7).

    Accepted candidates are grouped by edited surface; within each group only
    the best survives -- higher held-out delta, tiebreak: held-in delta, then
    the lexicographically smaller candidate id -- because same-surface edits
    cannot compose. Zero survivors plan nothing; one plans a single promotion
    of that candidate's own (already evaluated) harness; several plan the
    merged harness, which U6 must re-evaluate before promotion is final.

    Args:
        incumbent: The harness the round evaluated against.
        decisions: The round's decision records (non-accepted ones are
            ignored here; the ledger keeps them).
        candidates: The loaded candidates, keyed by id; every accepted
            decision must have one (its surface and live harness live there).

    Returns:
        The plan, with same-surface losers recorded in ``excluded``.

    Raises:
        ValueError: If an accepted decision has no loaded candidate to
            promote.
    """
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    accepted = [decision for decision in decisions if decision.accepted]
    missing = [decision.subject_id for decision in accepted if decision.subject_id not in by_id]
    if missing:
        raise ValueError(
            f"accepted decision(s) {', '.join(sorted(missing))} have no loaded candidate; "
            "a decision cannot promote a harness it does not hold"
        )

    by_surface: dict[str, list[CandidateDecision]] = {}
    for decision in accepted:
        by_surface.setdefault(by_id[decision.subject_id].surface, []).append(decision)

    winners: list[CandidateDecision] = []
    excluded: dict[str, str] = {}
    for surface in sorted(by_surface):
        ranked = sorted(
            by_surface[surface],
            key=lambda decision: (
                -decision.delta(SPLIT_HELDOUT),
                -decision.delta(SPLIT_HELDIN),
                decision.subject_id,
            ),
        )
        winner = ranked[0]
        winners.append(winner)
        for loser in ranked[1:]:
            excluded[loser.subject_id] = (
                f"same-surface {surface}: {winner.subject_id} wins on held-out delta "
                "(tiebreak: held-in delta, then candidate id); same-surface edits cannot merge"
            )

    if not winners:
        return PromotionPlan(
            kind=PLAN_NONE, constituent_ids=(), harness=None, harness_hash=None, excluded=excluded
        )
    if len(winners) == 1:
        winner = by_id[winners[0].subject_id]
        return PromotionPlan(
            kind=PLAN_SINGLE,
            constituent_ids=(winner.candidate_id,),
            harness=winner.harness,
            harness_hash=winner.harness_hash,
            excluded=excluded,
        )
    merged = merge_harnesses(incumbent, [by_id[winner.subject_id] for winner in winners])
    return PromotionPlan(
        kind=PLAN_MERGE,
        constituent_ids=tuple(sorted(winner.subject_id for winner in winners)),
        harness=merged,
        harness_hash=harness_hash(merged),
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# The merge verdict and final promotion
# ---------------------------------------------------------------------------


def promote_decision(decision: CandidateDecision) -> CandidateDecision:
    """Mark one accepted decision as the round's promotion.

    Raises:
        ValueError: If the decision is not ``accepted`` -- only a candidate
            (or merged harness) that passed the rule and the band may promote.
    """
    if not decision.accepted:
        raise ValueError(
            f"only accepted decisions may be promoted; {decision.subject_id!r} is "
            f"{decision.decision!r}"
        )
    return replace(decision, decision=DECISION_PROMOTED)


def apply_merge_verdict(
    plan: PromotionPlan,
    merge_decision: CandidateDecision,
    decisions: Iterable[CandidateDecision],
) -> tuple[CandidateDecision, list[CandidateDecision]]:
    """Turn the merged harness's re-evaluation into the round's outcome (R7).

    When the merge passed the rule and the band on its own, it is promoted and
    the constituents keep their ``accepted`` records. When it did not --
    rejected by the rule/band, or over budget -- the round promotes nothing:
    every constituent is re-ledgered as ``merged_failed``, because falling
    back to an individually accepted candidate after seeing merge results is
    post-hoc selection the preregistered rule never validated.

    Args:
        plan: The ``PLAN_MERGE`` plan whose harness was re-evaluated.
        merge_decision: ``decide_subject``'s record for that re-evaluation.
        decisions: The round's candidate decision records.

    Returns:
        ``(final_merge_decision, final_decisions)``: the merge's record
        (``promoted`` on success, unchanged otherwise) and the candidate
        records with constituents re-marked on failure. Inputs are never
        mutated.

    Raises:
        ValueError: If the plan is not a merge, or the re-evaluated harness
            hash is not the planned one -- the verdict must be about the
            artifact this plan built.
    """
    if plan.kind != PLAN_MERGE:
        raise ValueError(f"apply_merge_verdict needs a {PLAN_MERGE!r} plan, got {plan.kind!r}")
    if merge_decision.harness_hash is not None and merge_decision.harness_hash != plan.harness_hash:
        raise ValueError(
            f"merge decision is about harness hash {merge_decision.harness_hash}, but the plan "
            f"built {plan.harness_hash}; the verdict must re-evaluate the planned merge"
        )

    if merge_decision.accepted:
        return promote_decision(merge_decision), list(decisions)

    failure = (
        f"constituent of merged harness {merge_decision.subject_id!r}, which failed its own "
        "re-evaluation; the round promotes nothing (no post-hoc fallback to individual "
        "candidates)"
    )
    constituents = set(plan.constituent_ids)
    final_decisions = [
        replace(
            decision,
            decision=DECISION_MERGED_FAILED,
            reasons=(*decision.reasons, failure),
        )
        if decision.subject_id in constituents
        else decision
        for decision in decisions
    ]
    return merge_decision, final_decisions


__all__ = [
    "DECISION_ACCEPTED",
    "DECISION_MERGED_FAILED",
    "DECISION_OVER_BUDGET",
    "DECISION_PROMOTED",
    "DECISION_REJECTED",
    "MERGED_SUBJECT_ID",
    "PLAN_MERGE",
    "PLAN_NONE",
    "PLAN_SINGLE",
    "SURFACE_HARNESS_FIELDS",
    "Band",
    "CandidateDecision",
    "PromotionConfig",
    "PromotionPlan",
    "apply_merge_verdict",
    "assess_round",
    "decide_subject",
    "merge_harnesses",
    "plan_promotion",
    "promote_decision",
    "score_candidate",
]
