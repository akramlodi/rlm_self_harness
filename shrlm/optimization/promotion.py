"""Pure composition and promotion decisions for held-out batch validation.

``plan_batch`` composes all admitted edits before validation and rejects any
surface collision. ``score_candidate`` compares held-out pass counts and
held-out cost/sub-call means against the incumbent. The batch must clear the
configured improvement margin and resource bands. No constituent is scored
individually, and a failed batch has no individual fallback."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import CandidateRejection, LoadedCandidate
from shrlm.optimization.costs import OUTCOME_COMPLETED
from shrlm.optimization.driver import ACCOUNTING_VERSION_KEY, LEGACY_ACCOUNTING_VERSION
from shrlm.optimization.validation import (
    SPLIT_HELDOUT,
    VALIDATION_PROTOCOL,
    RoundEvaluation,
    SubjectEvaluation,
)
from shrlm.rlm_harness import Harness

# The decision vocabulary the promotion ledger (U5) records. ``accepted`` and
# ``rejected`` come from the rule+band; ``over_budget`` from the breaker;
# ``bundled`` constituents have no individual measurements. ``merged_failed``
# is retained only as vocabulary for readers of legacy ledgers.
DECISION_BUNDLED = "bundled"
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
# on held-out instances), the summary's per-split total it is computed from, and
# the ``PromotionConfig`` field naming its band.
_BAND_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mean_cost", "total_cost", "cost_band"),
    ("mean_sub_calls", "total_sub_calls", "sub_call_band"),
)

# The ``Harness`` fields each surface id owns (builder convention: build_<x>
# fills <x>). The merge builder composes edits through this map; a test asserts
# it covers the declared ``SURFACES`` (S1-S10) exactly.
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
    "S10": ("skills",),
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
    """Promotion thresholds and resource multiplier bands.

    The held-out delta must exceed ``tau_improvement`` and must not fall below
    ``-tau_regression``. Bands use held-out means only. Every decision records
    these round-level parameters, including unscored constituent records."""

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
    baseline/candidate means and bounds (a null bound means unconstrained:
    ``math.inf`` has no JSON rendering, so it never enters this shape); both
    are None exactly when the subject was never scored (rejected upstream or
    over budget, in which case ``upstream`` or ``reasons`` say why). The thresholds are recorded on every
    record, scored or not -- they are round-level preregistration, and a
    ledger row must be interpretable without the config that produced it.
    ``surface`` is the one S1-S10 surface the candidate's proposal edited
    (from its ``LoadedCandidate``); it is None for a loader rejection whose
    surface was never resolved, and for the merged harness's evaluation
    record, which spans more than one surface by construction.
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
    surface: str | None = None

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
            "surface": self.surface,
        }


@dataclass(frozen=True)
class PromotionPlan:
    """The fixed candidate composition, constructed before any validation result.

    ``harness`` is the sole candidate to evaluate. ``excluded`` remains an empty
    legacy ledger field: no proposal is selected or excluded using measurements."""

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
    if any(
        summary.get("validation_protocol") != VALIDATION_PROTOCOL
        for summary in (baseline, candidate)
    ):
        return (
            "promotion requires the current validation protocol; legacy evidence cannot be scored"
        )
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
    for split_id in (SPLIT_HELDOUT,):
        base_runs = baseline["splits"][split_id]["n_runs"]
        cand_runs = candidate["splits"][split_id]["n_runs"]
        if base_runs != cand_runs or base_runs == 0:
            return (
                f"{split_id} n_runs differ (baseline {base_runs}, candidate {cand_runs}) or are "
                "zero; pass-count deltas need one shared, non-empty denominator"
            )
    # The cost band is a ratio between the two arms, so it is only meaningful
    # when both were priced the same way. A terminated run's cost changed from
    # absent to real between accounting versions, which moves the ratio by more
    # than the band's own width on a realistic termination count -- comparing
    # across versions would decide promotions on the accounting rather than on
    # the candidate. A summary written before the version was recorded reads as
    # legacy, which is what it is.
    base_accounting = baseline.get(ACCOUNTING_VERSION_KEY, LEGACY_ACCOUNTING_VERSION)
    cand_accounting = candidate.get(ACCOUNTING_VERSION_KEY, LEGACY_ACCOUNTING_VERSION)
    if base_accounting != cand_accounting:
        return (
            f"baseline is priced under cost-accounting {base_accounting!r} but candidate "
            f"{candidate['subject_id']!r} under {cand_accounting!r}; a cost band compared "
            "across accounting versions measures the accounting, not the candidate"
        )
    return None


def _overall_mean(summary: dict[str, Any], total_key: str) -> float:
    """One metric's per-run held-out mean from persisted totals."""
    split = summary["splits"][SPLIT_HELDOUT]
    return float(split[total_key]) / int(split["n_runs"])


def _recorded_bound(bound: float) -> float | None:
    """A band bound as the decision record carries it: None when non-finite.

    ``Band`` keeps ``math.inf`` for its in-memory arithmetic, but the record
    is what the ledger serializes, and bare ``Infinity`` is not RFC-8259
    JSON -- so an unconstrained bound is recorded as null.
    """
    return bound if math.isfinite(bound) else None


def score_candidate(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    config: PromotionConfig,
    surface: str | None = None,
) -> CandidateDecision:
    """Apply the acceptance rule and the band to one completed evaluation (R6).

    Pure arithmetic over two persisted ``summary.json`` payloads. Every
    failing check contributes a reason -- the ledger reports all of them, not
    the first.

    Args:
        baseline_summary: The incumbent's summary payload.
        candidate_summary: The candidate's (or merged harness's) summary.
        config: The preregistered thresholds and bands.
        surface: The candidate's edited surface (S1-S10), recorded on the
            decision verbatim; None for the merged harness's evaluation.

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
    for split_id in (SPLIT_HELDOUT,):
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
            f"(heldout {deltas[SPLIT_HELDOUT]:+d})"
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
            "lower": _recorded_bound(metric_band.lower),
            "upper": _recorded_bound(metric_band.upper),
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
        surface=surface,
    )


# ---------------------------------------------------------------------------
# Assessing subjects and rounds: upstream rejections pass through
# ---------------------------------------------------------------------------


def decide_subject(
    baseline_summary: dict[str, Any],
    subject: SubjectEvaluation | CandidateRejection,
    config: PromotionConfig,
    surface: str | None = None,
) -> CandidateDecision:
    """One subject's decision record, whatever happened to it upstream.

    A loader/caps ``CandidateRejection`` becomes a ``rejected`` record
    carrying the gate verdict verbatim; an over-budget evaluation becomes
    ``over_budget``; only a completed evaluation is scored. This also handles the combined batch subject.

    Args:
        surface: The candidate's edited surface (S1-S10), when known -- a
            loader rejection whose surface never resolved, or the merged
            harness's evaluation, passes None.
    """
    if isinstance(subject, CandidateRejection):
        return CandidateDecision(
            subject_id=subject.candidate_id,
            decision=DECISION_REJECTED,
            reasons=(f"rejected upstream at gate {subject.gate}: {subject.reason}",),
            tau_regression=config.tau_regression,
            tau_improvement=config.tau_improvement,
            upstream=subject.to_dict(),
            surface=surface,
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
            surface=surface,
        )
    return score_candidate(baseline_summary, subject.summary, config, surface=surface)


def assess_round(
    evaluation: RoundEvaluation,
    config: PromotionConfig,
    surfaces: Mapping[str, str] | None = None,
) -> list[CandidateDecision]:
    """One decision record per candidate, in the round's order.

    Args:
        surfaces: ``candidate_id -> surface``, from the round's loaded
            candidates; a candidate absent from this mapping records
            ``surface=None``.

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
    surfaces = surfaces or {}
    return [
        decide_subject(
            evaluation.baseline.summary,
            candidate,
            config,
            surface=surfaces.get(
                candidate.candidate_id
                if isinstance(candidate, CandidateRejection)
                else candidate.subject_id
            ),
        )
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
            are incompatible and must be fixed before evaluation.
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


def plan_batch(incumbent: Harness, candidates: Iterable[LoadedCandidate]) -> PromotionPlan:
    """Freeze all admitted, disjoint edits before observing validation outcomes."""
    candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    if not candidates:
        return PromotionPlan(PLAN_NONE, (), None, None)
    merged = merge_harnesses(incumbent, candidates)  # also rejects duplicate surfaces
    if len(candidates) == 1:
        candidate = candidates[0]
        return PromotionPlan(
            PLAN_SINGLE, (candidate.candidate_id,), candidate.harness, candidate.harness_hash
        )
    return PromotionPlan(
        PLAN_MERGE, tuple(c.candidate_id for c in candidates), merged, harness_hash(merged)
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


__all__ = [
    "DECISION_ACCEPTED",
    "DECISION_BUNDLED",
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
    "assess_round",
    "decide_subject",
    "merge_harnesses",
    "plan_batch",
    "promote_decision",
    "score_candidate",
]
