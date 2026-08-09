"""Stage-3 proposal validation, evaluation half: persisted, resumable, comparable rounds.

Baseline and every candidate are evaluated on the held-in and the disjoint
held-out split with a configurable repetition count (R5), by reusing the
persist-first round infrastructure (KTD4): each (subject x split) evaluation is
one ``run_round`` with its own directory and ``harness.json`` identity,
executed through the cost governor's ``run_governed_round`` so the
experiment-owned caps bind at the RLM constructor and one cumulative circuit
breaker spans all of a subject's splits and repetitions (R3, R4).
Resumability, sha-linked traces, and recomputable pass rates come for free; a
crashed evaluation resumes by re-invoking with the same configuration, and
only the missing runs are paid for again.

Directory layout under one validation round, ``<out_dir>/round_NN``::

    round_NN/
        baseline/                     # the incumbent, evaluated once per round
            heldin/                   # the out_dir handed to run_round, so its
                round_00/             #   contents land under a nested round_00
                    harness.json      # the subject's identity (KTD4)
                    instances.jsonl
                    runs.jsonl        # verdict, trace sha, cost per run
                    runs/<run_id>.json
            heldout/
                round_00/...
            summary.json              # the validation-owned aggregate
        <candidate_id>/
            heldin/round_00/...
            heldout/round_00/...
            summary.json

Repetition uses the existing ``(instance, attempt)`` run ids: ``repetitions``
becomes ``RoundConfig.attempts``, so rep ``v`` of instance ``i`` is run
``<i>__aNN`` inside the split's single nested round. Anything the ledger cites
links through this nesting: a split's round directory is
``<subject_id>/<split_id>/round_00`` relative to ``round_NN`` (each summary
records the ``<split_id>/round_00`` tail relative to its own subject
directory).

Aggregation is disk-only (KTD5's consume-not-execute stance): pass counts and
costs come from ``runs.jsonl``, sub-call counts from rehydrating the persisted
traces through ``load_round`` + ``run_metrics`` (sha-verified). Each subject's
aggregate is then written exactly once as ``summary.json`` -- non-clobbering
in the bundle style, byte-identical rewrites allowed, divergence refused -- so
the band check (U4) and the ledger (U5) never re-read traces, and the shared
mining manifest format is never extended.

The promotion half (U4) lives in the sibling ``promotion`` module; the ledger
writer (U5) lives here: ``write_promotion_ledger`` persists one
``promotions.jsonl`` record per candidate (loader-rejected and over-budget ones
included, with their structured reasons) plus one for a re-evaluated merged
harness, and a ``decision.json`` naming the promoted harness hash or "no
promotion" (R8). Every record links -- by paths relative to the round
directory, in ``audit.py``'s walkable style -- to the subject's summary, split
round directories, and ``harness.json`` identities, so an audit can walk ledger
-> round dirs -> sha-verified traces. Both files are non-clobbering in the
bundle style: byte-identical rewrites are no-ops, divergence is refused. The
evaluation surface here plus the promotion module are what U6's
``validate_round`` composes.
"""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN, round_dir
from shrlm.optimization.candidates import CandidateRejection, LoadedCandidate
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import HARNESS_FILE, RoundConfig, load_round
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verifier
from shrlm.rlm_harness import Harness
from shrlm.runner import run_metrics

if TYPE_CHECKING:  # promotion imports this module, so its types are annotations only
    from shrlm.optimization.promotion import CandidateDecision, PromotionPlan

# The incumbent's directory name under a validation round; reserved, so no
# candidate may claim it.
BASELINE_ID = "baseline"

SPLIT_HELDIN = "heldin"
SPLIT_HELDOUT = "heldout"

# Each split directory is the out_dir of exactly one run_round, so the nested
# round index is always 0 (see the layout in the module docstring).
EVAL_ROUND_INDEX = 0

SUMMARY_FILENAME = "summary.json"
SUMMARY_FORMAT = "shrlm-validation-summary/v1"

# The promotion ledger (U5): one JSONL record per candidate (and per merged
# harness) under the round directory, plus the round's decision summary.
PROMOTIONS_FILENAME = "promotions.jsonl"
DECISION_FILENAME = "decision.json"
LEDGER_RECORD_FORMAT = "shrlm-promotion-record/v1"
DECISION_FORMAT = "shrlm-promotion-decision/v1"

# Merge participation roles a ledger record can carry: a constituent of the
# round's merge plan, or the merged harness's own re-evaluation record.
ROLE_CONSTITUENT = "constituent"
ROLE_MERGED = "merged"


def _canonical(instance: dict[str, Any]) -> str:
    return json.dumps(instance, sort_keys=True)


@dataclass(frozen=True)
class ValidationSplits:
    """The held-in and held-out instance lists one validation round evaluates.

    Both splits must be non-empty and disjoint (R5): an instance appearing
    verbatim in both would leak the held-out measurement. Instance *ids* may
    repeat across splits -- each split runs in its own round directory, and the
    breaker namespaces charges by that directory -- but identical instances
    may not.
    """

    heldin: list[dict[str, Any]]
    heldout: list[dict[str, Any]]

    def __post_init__(self) -> None:
        for name, instances in self.items():
            if not instances:
                raise ValueError(f"the {name} split must hold at least one instance")
        heldin_canonical = {_canonical(instance) for instance in self.heldin}
        shared = [instance for instance in self.heldout if _canonical(instance) in heldin_canonical]
        if shared:
            ids = ", ".join(repr(instance.get("id")) for instance in shared)
            raise ValueError(
                f"held-in and held-out splits share instance(s) {ids}; R5 demands disjoint "
                "splits, so a shared instance is a held-out leak, not a coincidence."
            )

    def items(self) -> tuple[tuple[str, list[dict[str, Any]]], ...]:
        """The (split_id, instances) pairs, in evaluation order."""
        return ((SPLIT_HELDIN, self.heldin), (SPLIT_HELDOUT, self.heldout))


@dataclass(frozen=True)
class EvaluationConfig:
    """Everything one validation round's evaluations share.

    ``repetitions`` becomes each round's ``attempts``; ``caps`` are the
    experiment-owned limits every subject runs under (merged tighten-only
    against its S6 policy by ``governed_limits``).
    """

    splits: ValidationSplits
    verifier: Verifier
    caps: ValidationCaps
    out_dir: Path | str
    round_index: int
    repetitions: int = 1
    backend: str = "openrouter"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")


def subject_dir(out_dir: Path | str, round_index: int, subject_id: str) -> Path:
    """One subject's directory: ``<out_dir>/round_NN/<subject_id>``."""
    if not FILESYSTEM_SAFE_ID_PATTERN.fullmatch(subject_id):
        raise ValueError(
            f"subject id {subject_id!r} is not filesystem-safe; ids become directory "
            f"names and must match {FILESYSTEM_SAFE_ID_PATTERN.pattern}"
        )
    return round_dir(out_dir, round_index) / subject_id


def split_dir(out_dir: Path | str, round_index: int, subject_id: str, split_id: str) -> Path:
    """One subject's split directory -- the ``out_dir`` its ``run_round`` persists into."""
    return subject_dir(out_dir, round_index, subject_id) / split_id


@dataclass(frozen=True)
class SubjectEvaluation:
    """One subject's completed (or budget-stopped) evaluation over both splits.

    ``summary`` is exactly the persisted ``summary.json`` payload -- the
    aggregate shape the promotion module and the ledger consume.
    """

    subject_id: str
    harness_hash: str
    path: Path
    summary_path: Path
    summary: dict[str, Any]

    @property
    def outcome(self) -> str:
        return str(self.summary["outcome"])

    @property
    def over_budget(self) -> bool:
        return self.outcome == OUTCOME_OVER_BUDGET


@dataclass(frozen=True)
class RoundEvaluation:
    """One validation round's evaluations: the baseline plus every candidate.

    ``candidates`` preserves the caller's order; a candidate whose enabled S6
    policy exceeds the caps appears as its ``CandidateRejection``, never
    silently dropped.
    """

    round_path: Path
    baseline: SubjectEvaluation
    candidates: list[SubjectEvaluation | CandidateRejection]


# ---------------------------------------------------------------------------
# Disk-only aggregation
# ---------------------------------------------------------------------------


def split_aggregate(split_path: Path | str) -> dict[str, Any]:
    """Recompute one split's aggregate from its persisted round alone.

    Pass counts and costs come straight from the manifest lines; sub-call
    counts come from rehydrating every trace through ``load_round`` (which
    sha-verifies each file first) and ``run_metrics``. ``total_cost`` sums the
    costs that were actually persisted: a resource-terminated run on a
    cost-less backend records no cost and contributes nothing here (the spend
    *breaker* prices such runs at the per-run ceiling, but that is worst-case
    accounting, not a measurement). A terminated run persisted before any
    trajectory existed counts zero sub-calls; a non-terminated run without a
    trajectory is an error, surfaced by ``run_metrics``.

    Ratio fields (``pass_rate``, ``mean_cost``, ``mean_sub_calls``) are None
    when the split holds no runs at all -- the shape a fully budget-skipped
    split persists.
    """
    runs, _verdicts, envelope, entries = load_round(split_path, EVAL_ROUND_INDEX)
    total_sub_calls = 0
    for entry, (_instance, completion) in zip(entries, runs, strict=True):
        terminated = entry.get("cause") == VerifierCause.RESOURCE_TERMINATED.value
        if completion.metadata is None and terminated:
            continue  # terminated before any trajectory existed: no sub-call evidence
        total_sub_calls += int(run_metrics(completion)["sub_call_count"])

    n_runs = len(entries)
    pass_count = sum(1 for entry in entries if entry["passed"])
    total_cost = float(sum(entry["cost"] for entry in entries if entry.get("cost") is not None))
    return {
        "harness_hash": str(envelope["hash"]),
        "n_runs": n_runs,
        "pass_count": pass_count,
        "pass_rate": pass_count / n_runs if n_runs else None,
        "n_resource_terminated": sum(
            1 for entry in entries if entry.get("cause") == VerifierCause.RESOURCE_TERMINATED.value
        ),
        "total_cost": total_cost,
        "mean_cost": total_cost / n_runs if n_runs else None,
        "total_sub_calls": total_sub_calls,
        "mean_sub_calls": total_sub_calls / n_runs if n_runs else None,
    }


def load_summary(subject_path: Path | str) -> dict[str, Any]:
    """Read one subject's persisted ``summary.json`` back, checking its format."""
    path = Path(subject_path) / SUMMARY_FILENAME
    payload = json.loads(path.read_text())
    if payload.get("format") != SUMMARY_FORMAT:
        raise ValueError(f"{path} is not a {SUMMARY_FORMAT} summary")
    return payload


def _persist_once(path: Path, text: str, diverging: str) -> None:
    """Write ``text`` exactly once, in the bundle's non-clobbering style.

    A byte-identical rewrite is an idempotent no-op; different bytes raise
    ``diverging`` instead of silently replacing what may already have been
    audited. The write goes through a same-directory tmp file and
    ``os.replace``, so a crash mid-write never leaves a truncated file.
    """
    if path.exists():
        if path.read_text() == text:
            return
        raise ValueError(diverging)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Persist one subject's summary, refusing to clobber a diverging one.

    The payload is a pure function of the persisted round state and the
    evaluation config (no timestamps), so a resume that completes the same
    evaluation rewrites identical bytes -- allowed as a no-op. Different bytes
    mean the configuration changed under an already-summarized subject (e.g. a
    raised budget re-running a previously over-budget candidate); that summary
    may already have fed a promotion decision, so the rewrite is refused and
    the operator must delete the stale file deliberately.
    """
    _persist_once(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        f"{path} already holds a diverging summary; the evaluation configuration "
        "changed under an already-summarized subject. Refusing to overwrite an "
        "aggregate the promotion decision may already cite -- delete the stale "
        "summary deliberately to re-aggregate.",
    )


# ---------------------------------------------------------------------------
# Evaluation: one subject, then the whole round
# ---------------------------------------------------------------------------


def evaluate_subject(
    subject_id: str,
    harness: Harness,
    config: EvaluationConfig,
    *,
    breaker: CandidateSpendBreaker | None = None,
) -> SubjectEvaluation | CandidateRejection:
    """Evaluate one harness over both splits, persist-first and cost-governed.

    Each split is one ``run_governed_round`` into
    ``<out_dir>/round_NN/<subject_id>/<split_id>/`` (nesting ``round_00``; see
    the module docstring), under ONE spend breaker shared across the splits so
    the candidate budget is truly cumulative (R4). Limits are merged
    tighten-only by ``governed_limits``; an enabled S6 policy above the caps
    comes back as a ``CandidateRejection`` before anything touches disk.

    Idempotent over the round directory: persisted runs are skipped (after
    trace re-verification), so a crashed evaluation resumes by re-invoking
    with the same configuration and only the missing runs execute.

    Args:
        subject_id: ``BASELINE_ID`` or a candidate id; becomes the directory
            name.
        harness: The live harness to run.
        config: The round's shared evaluation config.
        breaker: Optional pre-charged breaker (tests, merged re-evaluations);
            by default a fresh one per subject.

    Returns:
        The ``SubjectEvaluation`` with its persisted summary, or the
        ``CandidateRejection`` from the caps gate.
    """
    limits = governed_limits(subject_id, harness.runtime_policy, config.caps)
    if isinstance(limits, CandidateRejection):
        return limits
    if breaker is None:
        breaker = CandidateSpendBreaker(config.caps)

    subject_path = subject_dir(config.out_dir, config.round_index, subject_id)
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_id, instances in config.splits.items():
        split_path = split_dir(config.out_dir, config.round_index, subject_id, split_id)
        round_config = RoundConfig(
            round_index=EVAL_ROUND_INDEX,
            harness=harness,
            instances=instances,
            verifier=config.verifier,
            out_dir=split_path,
            backend=config.backend,
            backend_kwargs=dict(config.backend_kwargs),
            attempts=config.repetitions,
            **limits,
        )
        result = run_governed_round(round_config, breaker)
        split_summaries[split_id] = {
            "round_path": f"{split_id}/round_{EVAL_ROUND_INDEX:02d}",
            "n_instances": len(instances),
            "outcome": result.outcome,
            "skipped_run_ids": list(result.skipped_run_ids),
            **split_aggregate(split_path),
        }

    summary = {
        "format": SUMMARY_FORMAT,
        "subject_id": subject_id,
        "harness_hash": harness_hash(harness),
        "repetitions": config.repetitions,
        "outcome": OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
        "spent": breaker.spent,
        "splits": split_summaries,
    }
    summary_path = subject_path / SUMMARY_FILENAME
    _write_summary(summary_path, summary)
    return SubjectEvaluation(
        subject_id=subject_id,
        harness_hash=str(summary["harness_hash"]),
        path=subject_path,
        summary_path=summary_path,
        summary=summary,
    )


def evaluate_validation_round(
    incumbent: Harness,
    candidates: Iterable[LoadedCandidate],
    config: EvaluationConfig,
) -> RoundEvaluation:
    """Evaluate the baseline once, then every candidate against it (R5).

    The baseline is the incumbent under ``BASELINE_ID``; each candidate runs
    under its own id with its own fresh spend breaker. A candidate rejected at
    the caps gate stays in the result as its ``CandidateRejection`` -- the
    ledger records every candidate, including the ones that never ran.

    Raises:
        ValueError: If a candidate claims the reserved ``baseline`` id, two
            candidates share an id (their directories would collide), or the
            incumbent itself violates the experiment-owned caps (an experiment
            misconfiguration, not expected-invalid stage-2 output).
    """
    candidates = list(candidates)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id == BASELINE_ID:
            raise ValueError(
                f"candidate id {BASELINE_ID!r} is reserved for the incumbent's directory"
            )
        if candidate.candidate_id in seen:
            raise ValueError(
                f"duplicate candidate id {candidate.candidate_id!r}: evaluation directories "
                "are keyed by id, so repeats would mix two candidates' evidence"
            )
        seen.add(candidate.candidate_id)

    baseline = evaluate_subject(BASELINE_ID, incumbent, config)
    if isinstance(baseline, CandidateRejection):
        raise ValueError(
            f"the incumbent violates the experiment-owned caps ({baseline.reason}); a "
            "baseline that cannot run under the validation limits is a misconfigured "
            "experiment, not a rejectable candidate"
        )
    results: list[SubjectEvaluation | CandidateRejection] = [
        evaluate_subject(candidate.candidate_id, candidate.harness, config)
        for candidate in candidates
    ]
    return RoundEvaluation(
        round_path=round_dir(config.out_dir, config.round_index),
        baseline=baseline,
        candidates=results,
    )


# ---------------------------------------------------------------------------
# The promotion ledger (U5): every candidate's outcome, auditable and linkable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionLedger:
    """One round's persisted ledger: where it lives and what it holds.

    ``records`` are the ``promotions.jsonl`` lines in file order; ``decision``
    is the ``decision.json`` payload.
    """

    round_path: Path
    ledger_path: Path
    decision_path: Path
    records: list[dict[str, Any]]
    decision: dict[str, Any]


def _subject_links(evaluation: SubjectEvaluation) -> dict[str, Any]:
    """One evaluated subject's audit links, relative to the ledger's directory.

    Built from the persisted summary alone: each split's ``round_path`` tail is
    recorded relative to the subject directory, so prefixing the subject id
    yields the round-relative path the audit walk resolves. Links exist even
    for an over-budget subject -- its split rounds were prepared (harness
    identity and all) before the breaker tripped.
    """
    splits: dict[str, Any] = {}
    for split_id, split_summary in evaluation.summary["splits"].items():
        nested = f"{evaluation.subject_id}/{split_summary['round_path']}"
        splits[split_id] = {"round_dir": nested, "harness": f"{nested}/{HARNESS_FILE}"}
    return {"summary": f"{evaluation.subject_id}/{SUMMARY_FILENAME}", "splits": splits}


def _ledger_record(
    decision: "CandidateDecision",
    subject: SubjectEvaluation | CandidateRejection,
    role: str | None,
    constituent_ids: tuple[str, ...],
    excluded_reason: str | None,
) -> dict[str, Any]:
    """One ``promotions.jsonl`` line: the decision, merge participation, links.

    A ``CandidateRejection`` subject never touched disk, so its ``links`` are
    None -- what exists (the gate, the violation, the proposal path) is already
    in the decision's ``upstream``. An evaluated subject links to everything
    the audit walk needs.
    """
    return {
        "format": LEDGER_RECORD_FORMAT,
        **decision.to_dict(),
        "merge": {
            "role": role,
            "constituent_ids": list(constituent_ids) if role is not None else None,
        },
        "selection_excluded": excluded_reason,
        "links": _subject_links(subject) if isinstance(subject, SubjectEvaluation) else None,
    }


def write_promotion_ledger(
    evaluation: RoundEvaluation,
    decisions: Iterable["CandidateDecision"],
    plan: "PromotionPlan",
    *,
    loader_rejections: Iterable[CandidateRejection] = (),
    merge_evaluation: "SubjectEvaluation | CandidateRejection | None" = None,
    merge_decision: "CandidateDecision | None" = None,
) -> PromotionLedger:
    """Persist one round's promotion ledger and decision summary (R8).

    Writes ``promotions.jsonl`` (one record per candidate: loader rejections
    first, then the round's candidates in evaluation order, then the merged
    harness's record when the plan merged) and ``decision.json`` (the promoted
    harness hash, or "no promotion") into ``evaluation.round_path``. Both
    payloads are pure functions of the inputs -- no timestamps -- so re-running
    the same round rewrites identical bytes (a no-op), while a divergent
    rewrite is refused in the bundle's non-clobbering style.

    Args:
        evaluation: The round's evaluations (U3's ``RoundEvaluation``).
        decisions: One ``CandidateDecision`` per candidate -- covering every
            loader rejection and every evaluated candidate exactly, with any
            promotion / merge-failure re-marking already applied.
        plan: The round's ``PromotionPlan``.
        loader_rejections: Candidates the U1 loader refused; they never reached
            evaluation, and are ledgered with their structured reasons.
        merge_evaluation: The merged harness's evaluation (or its caps
            rejection); required with ``merge_decision`` exactly when the plan
            is a merge.
        merge_decision: The merged harness's decision record, after
            ``apply_merge_verdict``.

    Returns:
        The ``PromotionLedger`` with both persisted payloads.

    Raises:
        ValueError: If decisions do not cover the candidates exactly, the
            merge leg is missing or spurious for the plan kind, more than one
            record is promoted, the promoted hash contradicts the plan, or an
            existing ledger diverges from this one.
    """
    # promotion imports this module, so its vocabulary is imported at call time.
    from shrlm.optimization.promotion import DECISION_PROMOTED, PLAN_MERGE

    subjects: list[SubjectEvaluation | CandidateRejection] = [
        *loader_rejections,
        *evaluation.candidates,
    ]
    subject_ids = [
        subject.subject_id if isinstance(subject, SubjectEvaluation) else subject.candidate_id
        for subject in subjects
    ]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError(f"duplicate ledger subject ids in {subject_ids}")
    by_id: dict[str, CandidateDecision] = {}
    for decision in decisions:
        if decision.subject_id in by_id:
            raise ValueError(f"duplicate decision for subject {decision.subject_id!r}")
        by_id[decision.subject_id] = decision
    if set(by_id) != set(subject_ids):
        missing = sorted(set(subject_ids) - set(by_id))
        extra = sorted(set(by_id) - set(subject_ids))
        raise ValueError(
            f"decisions must cover the round's candidates exactly: missing {missing}, "
            f"extra {extra}; the ledger records every candidate, never a subset"
        )

    if (merge_decision is None) != (merge_evaluation is None):
        raise ValueError("merge_decision and merge_evaluation must be given together")
    if (plan.kind == PLAN_MERGE) != (merge_decision is not None):
        raise ValueError(
            f"a {PLAN_MERGE!r} plan requires the merge re-evaluation leg (and only a merge "
            f"plan may carry one); plan kind is {plan.kind!r}"
        )

    constituents = set(plan.constituent_ids) if plan.kind == PLAN_MERGE else set()
    records = [
        _ledger_record(
            by_id[subject_id],
            subject,
            ROLE_CONSTITUENT if subject_id in constituents else None,
            plan.constituent_ids,
            plan.excluded.get(subject_id),
        )
        for subject_id, subject in zip(subject_ids, subjects, strict=True)
    ]
    if merge_decision is not None and merge_evaluation is not None:
        records.append(
            _ledger_record(
                merge_decision, merge_evaluation, ROLE_MERGED, plan.constituent_ids, None
            )
        )

    promoted = [record for record in records if record["decision"] == DECISION_PROMOTED]
    if len(promoted) > 1:
        ids = ", ".join(record["subject_id"] for record in promoted)
        raise ValueError(f"a round promotes at most one harness; got promoted records for {ids}")
    promoted_record = promoted[0] if promoted else None
    if promoted_record is not None and promoted_record["harness_hash"] != plan.harness_hash:
        raise ValueError(
            f"promoted record {promoted_record['subject_id']!r} carries harness hash "
            f"{promoted_record['harness_hash']}, but the plan built {plan.harness_hash}; "
            "the ledger must name the artifact the plan promoted"
        )

    decision_payload = {
        "format": DECISION_FORMAT,
        "plan": plan.kind,
        "constituent_ids": list(plan.constituent_ids),
        "excluded": dict(plan.excluded),
        "promoted": promoted_record is not None,
        "promoted_subject_id": promoted_record["subject_id"] if promoted_record else None,
        "promoted_harness_hash": promoted_record["harness_hash"] if promoted_record else None,
        "baseline": {
            "subject_id": evaluation.baseline.subject_id,
            "harness_hash": evaluation.baseline.harness_hash,
            "links": _subject_links(evaluation.baseline),
        },
        "n_candidates": len(subjects),
        "ledger": PROMOTIONS_FILENAME,
    }

    ledger_path = evaluation.round_path / PROMOTIONS_FILENAME
    _persist_once(
        ledger_path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        f"{ledger_path} already holds a diverging promotion ledger; the round's decisions "
        "changed under an already-ledgered round. Refusing to rewrite audit history -- "
        "delete the stale ledger deliberately to re-ledger.",
    )
    decision_path = evaluation.round_path / DECISION_FILENAME
    _persist_once(
        decision_path,
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
        f"{decision_path} already holds a diverging promotion decision; the round's outcome "
        "changed under an already-ledgered round. Refusing to rewrite audit history -- "
        "delete the stale decision deliberately to re-ledger.",
    )
    return PromotionLedger(
        round_path=evaluation.round_path,
        ledger_path=ledger_path,
        decision_path=decision_path,
        records=records,
        decision=decision_payload,
    )


def load_promotion_ledger(round_path: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one round's persisted ledger back, checking both formats.

    Returns ``(records, decision)``: the ``promotions.jsonl`` lines in file
    order and the ``decision.json`` payload -- the shape stage 2 consumes as
    prior-edit history.
    """
    path = Path(round_path)
    ledger_path = path / PROMOTIONS_FILENAME
    records = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    for record in records:
        if record.get("format") != LEDGER_RECORD_FORMAT:
            raise ValueError(
                f"{ledger_path} holds a record for {record.get('subject_id')!r} that is not "
                f"a {LEDGER_RECORD_FORMAT} record"
            )
    decision_path = path / DECISION_FILENAME
    decision = json.loads(decision_path.read_text())
    if decision.get("format") != DECISION_FORMAT:
        raise ValueError(f"{decision_path} is not a {DECISION_FORMAT} decision summary")
    return records, decision


__all__ = [
    "BASELINE_ID",
    "DECISION_FILENAME",
    "DECISION_FORMAT",
    "EVAL_ROUND_INDEX",
    "LEDGER_RECORD_FORMAT",
    "PROMOTIONS_FILENAME",
    "ROLE_CONSTITUENT",
    "ROLE_MERGED",
    "SPLIT_HELDIN",
    "SPLIT_HELDOUT",
    "SUMMARY_FILENAME",
    "SUMMARY_FORMAT",
    "EvaluationConfig",
    "PromotionLedger",
    "RoundEvaluation",
    "SubjectEvaluation",
    "ValidationSplits",
    "evaluate_subject",
    "evaluate_validation_round",
    "load_promotion_ledger",
    "load_summary",
    "split_aggregate",
    "split_dir",
    "subject_dir",
    "write_promotion_ledger",
]
