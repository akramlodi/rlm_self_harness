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
bundle style: byte-identical rewrites are no-ops, divergence is refused.

Subject parallelism: ``EvaluationConfig.workers`` above ``1`` evaluates the
baseline and the candidates concurrently, each subject in its own child
process (``shrlm.optimization.subject_worker``, which also owns the parent-side
dispatcher), at most ``workers`` alive at once. Subjects share nothing -- directory, breaker, hard deadline, and
manifests are all per subject -- so the persisted artifacts are byte-identical
to the sequential path's; only the wall clock changes. The parent gates the
caps before spawning (a rejection never gets a child), rebuilds each result
from the child's persisted ``summary.json``, keeps results in loader order,
and raises ``SubjectWorkerError`` only after every child has exited, so a
failed subject never discards its siblings' persisted runs. The merged
re-evaluation stays on the sequential in-process path.

``validate_round`` (U6) is the stage as one call: loader -> evaluation ->
promotion -> merged re-evaluation -> ledger. It gates a proposals directory
with the U1 loader under the round's caps, evaluates the baseline and every
loaded candidate, applies the U4 rule and band, re-evaluates a merged plan's
harness through the same evaluation path and rule before promotion is final
(R7), and persists the U5 ledger. A round with zero loadable candidates
short-circuits before the baseline runs -- no model calls are ever made for a
round with nothing to compare -- but R8 still demands every candidate's
outcome on disk: when the loader rejected at least one candidate, the round
directory gets a rejection-only ledger (upstream reasons, null ``links``, a
``decision.json`` with plan "none" and a null ``baseline``). Only a proposals
directory with zero candidates altogether stays a clean no-op that creates no
round directory at all.
"""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN, round_dir
from shrlm.optimization.candidates import (
    DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
    CandidateRejection,
    LoadedCandidate,
    load_candidates,
)
from shrlm.optimization.costs import (
    OUTCOME_COMPLETED,
    OUTCOME_OVER_BUDGET,
    CandidateSpendBreaker,
    ValidationCaps,
    governed_limits,
    run_governed_round,
)
from shrlm.optimization.driver import (
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_KEY,
    HARNESS_FILE,
    RoundConfig,
    load_round,
    manifest_accounting_version,
    read_run_workers,
    reject_sensitive_backend_kwargs,
)
from shrlm.optimization.subject_worker import (
    BASELINE_REJECTED_MESSAGE,
    evaluate_subjects_in_processes,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verifier
from shrlm.rlm_harness import Harness
from shrlm.runner import run_metrics

if TYPE_CHECKING:  # promotion imports this module, so its types are annotations only
    from shrlm.optimization.promotion import CandidateDecision, PromotionConfig, PromotionPlan

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

    ``workers`` caps how many subjects ``evaluate_validation_round`` evaluates
    concurrently. ``1`` is the sequential in-process path. Above ``1`` every
    subject runs in its own child process (``shrlm.optimization.subject_worker``),
    which rebuilds the verifier from ``verifier_factory`` -- a dotted path
    (``pkg.mod:attr`` or ``pkg.mod.attr``) to a zero-argument callable
    returning a ``Verifier`` -- so the factory is mandatory once ``workers > 1``.
    ``client_factory`` is the test-only seam for those children: a dotted path
    to a callable taking one JSON-safe ``dict`` and returning a ``get_client``
    replacement, plus per-subject-id args; a child installs it on
    ``rlm.core.rlm.get_client`` before evaluating (KTD9).
    """

    splits: ValidationSplits
    verifier: Verifier
    caps: ValidationCaps
    out_dir: Path | str
    round_index: int
    repetitions: int = 1
    backend: str = "openrouter"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    workers: int = 1
    verifier_factory: str | None = None
    client_factory: tuple[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise ValueError(f"workers must be an integer, got {self.workers!r}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.workers > 1 and not self.verifier_factory:
            raise ValueError(
                f"workers={self.workers} evaluates subjects in child processes, which rebuild "
                "the verifier from verifier_factory; pass the dotted path of a zero-argument "
                "verifier factory (or keep workers=1 for the in-process path)"
            )
        # The parallel path writes these kwargs into every worker_request.json
        # before any round runs, so the driver's credential scan must fire here.
        reject_sensitive_backend_kwargs(self.backend_kwargs)


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
    path: Path
    summary_path: Path
    summary: dict[str, Any]

    @property
    def harness_hash(self) -> str:
        return str(self.summary["harness_hash"])

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

    Skill-load counts (R16) come from the same rehydrated traces and sit beside
    the sub-call counts: a candidate whose runs never invoked the skill loader
    reports ``total_skill_loads`` of zero here, visible before it is scored.

    Ratio fields (``pass_rate``, ``mean_cost``, ``mean_sub_calls``,
    ``mean_skill_loads``) are None when the split holds no runs at all -- the
    shape a fully budget-skipped split persists.
    """
    runs, _verdicts, envelope, entries = load_round(split_path, EVAL_ROUND_INDEX)
    total_sub_calls = 0
    total_skill_loads = 0
    for entry, (_instance, completion) in zip(entries, runs, strict=True):
        terminated = entry.get("cause") == VerifierCause.RESOURCE_TERMINATED.value
        if completion.metadata is None and terminated:
            continue  # terminated before any trajectory existed: no sub-call evidence
        metrics = run_metrics(completion)
        total_sub_calls += int(metrics["sub_call_count"])
        total_skill_loads += int(metrics["skill_load_count"])

    n_runs = len(entries)
    pass_count = sum(1 for entry in entries if entry["passed"])
    total_cost = float(sum(entry["cost"] for entry in entries if entry.get("cost") is not None))
    return {
        "harness_hash": str(envelope["hash"]),
        # The accounting rules these figures were produced under, read from the
        # lines themselves rather than assumed to be this build's. A split
        # aggregated after the correction but whose runs predate it must say so,
        # or promotion would compare it against a differently-priced arm.
        ACCOUNTING_VERSION_KEY: manifest_accounting_version(entries),
        # What conditions produced these numbers (R5). Concurrency is a
        # confound on measured cost, so it is recorded beside the measurement
        # rather than left for the reader to reconstruct.
        "run_workers": read_run_workers(round_dir(split_path, EVAL_ROUND_INDEX)),
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
        "total_skill_loads": total_skill_loads,
        "mean_skill_loads": total_skill_loads / n_runs if n_runs else None,
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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

    # Both splits' aggregates carry the hash their persisted harness.json
    # recorded for this same harness, so reuse it rather than paying for a
    # second full serialization here.
    split_versions = {
        str(split[ACCOUNTING_VERSION_KEY])
        for split in split_summaries.values()
        if ACCOUNTING_VERSION_KEY in split
    }
    if len(split_versions) > 1:
        raise ValueError(
            f"subject {subject_id!r} mixes cost-accounting versions "
            f"{sorted(split_versions)} across its splits; its held-in and held-out "
            "figures are not comparable with each other, let alone with another "
            "subject's."
        )
    summary = {
        "format": SUMMARY_FORMAT,
        # Which cost-accounting rules produced every figure below -- taken from
        # the runs, never assumed to be this build's. Stamping the current
        # version unconditionally would let a legacy round re-aggregated after
        # the correction claim an accounting it was not priced under.
        ACCOUNTING_VERSION_KEY: next(iter(split_versions), ACCOUNTING_VERSION),
        "subject_id": subject_id,
        "harness_hash": next(iter(split_summaries.values()))["harness_hash"],
        "repetitions": config.repetitions,
        "outcome": OUTCOME_OVER_BUDGET if breaker.tripped else OUTCOME_COMPLETED,
        "spent": breaker.spent,
        "splits": split_summaries,
    }
    summary_path = subject_path / SUMMARY_FILENAME
    _write_summary(summary_path, summary)
    return SubjectEvaluation(
        subject_id=subject_id,
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

    if config.workers > 1:
        subjects = [(BASELINE_ID, incumbent, harness_hash(incumbent))] + [
            (candidate.candidate_id, candidate.harness, candidate.harness_hash)
            for candidate in candidates
        ]
        outcomes = evaluate_subjects_in_processes(subjects, config)
        baseline = outcomes[0]
        results = outcomes[1:]
    else:
        baseline = evaluate_subject(BASELINE_ID, incumbent, config)
        if isinstance(baseline, CandidateRejection):
            raise ValueError(BASELINE_REJECTED_MESSAGE.format(reason=baseline.reason))
        results = [
            evaluate_subject(candidate.candidate_id, candidate.harness, config)
            for candidate in candidates
        ]
    assert isinstance(baseline, SubjectEvaluation), "the baseline is caps-gated before any spawn"
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
    evaluation: "RoundEvaluation | None",
    decisions: Iterable["CandidateDecision"],
    plan: "PromotionPlan",
    *,
    round_path: Path | None = None,
    loader_rejections: Iterable[CandidateRejection] = (),
    merge_evaluation: "SubjectEvaluation | CandidateRejection | None" = None,
    merge_decision: "CandidateDecision | None" = None,
) -> PromotionLedger:
    """Persist one round's promotion ledger and decision summary (R8).

    Writes ``promotions.jsonl`` (one record per candidate: loader rejections
    first, then the round's candidates in evaluation order, then the merged
    harness's record when the plan merged) and ``decision.json`` (the promoted
    harness hash, or "no promotion") into the round directory. Both payloads
    are pure functions of the inputs -- no timestamps -- so re-running the
    same round rewrites identical bytes (a no-op), while a divergent rewrite
    is refused in the bundle's non-clobbering style.

    A round whose every candidate was rejected by the loader never evaluates
    anything, but R8 still demands its outcomes on disk: pass
    ``evaluation=None`` with an explicit ``round_path`` and the rejections,
    and the ledger holds their records (upstream reasons, null ``links``)
    plus a ``decision.json`` whose ``baseline`` is null -- there was no
    evaluated incumbent to name or link.

    Args:
        evaluation: The round's evaluations (U3's ``RoundEvaluation``), or
            None for a loader-rejection-only round.
        decisions: One ``CandidateDecision`` per candidate -- covering every
            loader rejection and every evaluated candidate exactly, with any
            promotion / merge-failure re-marking already applied.
        plan: The round's ``PromotionPlan``.
        round_path: Where the ledger lands when ``evaluation`` is None; the
            two are mutually exclusive (an evaluated round anchors its own
            directory).
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
        ValueError: If neither or both of ``evaluation`` and ``round_path``
            are given, a rejection-only ledger holds no rejections or a plan
            other than "none", decisions do not cover the candidates exactly,
            the merge leg is missing or spurious for the plan kind, more than
            one record is promoted, the promoted hash contradicts the plan, or
            an existing ledger diverges from this one.
    """
    # promotion imports this module, so its vocabulary is imported at call time.
    from shrlm.optimization.promotion import DECISION_PROMOTED, PLAN_MERGE, PLAN_NONE

    if (evaluation is None) == (round_path is None):
        raise ValueError(
            "pass exactly one round anchor: an evaluated round's RoundEvaluation, or an "
            "explicit round_path for a loader-rejection-only round"
        )
    rejections = list(loader_rejections)
    if evaluation is None:
        if not rejections:
            raise ValueError(
                "a rejection-only ledger needs at least one loader rejection; a round "
                "with zero candidates has nothing to record and writes no ledger"
            )
        if plan.kind != PLAN_NONE:
            raise ValueError(
                f"a round with no evaluations cannot plan {plan.kind!r}; only a "
                f"{PLAN_NONE!r} plan may be ledgered without an evaluated round"
            )
    ledger_root = evaluation.round_path if evaluation is not None else round_path
    assert ledger_root is not None  # exactly one anchor, checked above

    subjects: list[SubjectEvaluation | CandidateRejection] = [
        *rejections,
        *(evaluation.candidates if evaluation is not None else ()),
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
        # Null exactly when the round evaluated nothing (loader-rejection-only):
        # there is no incumbent evaluation to name or link.
        "baseline": {
            "subject_id": evaluation.baseline.subject_id,
            "harness_hash": evaluation.baseline.harness_hash,
            "links": _subject_links(evaluation.baseline),
        }
        if evaluation is not None
        else None,
        "n_candidates": len(subjects),
        "ledger": PROMOTIONS_FILENAME,
    }

    ledger_path = ledger_root / PROMOTIONS_FILENAME
    _persist_once(
        ledger_path,
        # allow_nan=False: bare Infinity/NaN tokens are not RFC-8259 JSON, so a
        # non-finite value anywhere in a record fails loudly instead of
        # persisting an unparseable ledger.
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
        f"{ledger_path} already holds a diverging promotion ledger; the round's decisions "
        "changed under an already-ledgered round. Refusing to rewrite audit history -- "
        "delete the stale ledger deliberately to re-ledger.",
    )
    decision_path = ledger_root / DECISION_FILENAME
    _persist_once(
        decision_path,
        json.dumps(decision_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        f"{decision_path} already holds a diverging promotion decision; the round's outcome "
        "changed under an already-ledgered round. Refusing to rewrite audit history -- "
        "delete the stale decision deliberately to re-ledger.",
    )
    return PromotionLedger(
        round_path=ledger_root,
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


# ---------------------------------------------------------------------------
# The whole stage as one call (U6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationRound:
    """One completed ``validate_round``: every stage output, ready for audit.

    ``evaluation`` is None exactly when the loader admitted no candidate: the
    round short-circuited before the baseline ran (zero model calls) and no
    evaluation directories exist. Such a round still persists a rejection-only
    ledger when the loader rejected at least one candidate (R8); ``ledger`` is
    None only when the proposals directory held no candidates at all, in which
    case nothing exists on disk and ``round_path`` is where the round *would*
    have lived. ``merge_evaluation``/``merge_decision`` are populated exactly
    when the plan was a merge -- the R7 re-evaluation leg.
    """

    round_path: Path
    loader_rejections: list[CandidateRejection]
    evaluation: RoundEvaluation | None
    decisions: list["CandidateDecision"]
    plan: "PromotionPlan"
    merge_evaluation: SubjectEvaluation | CandidateRejection | None
    merge_decision: "CandidateDecision | None"
    ledger: PromotionLedger | None

    @property
    def promoted(self) -> bool:
        """Whether this round promoted a harness (final, after any merge leg)."""
        return self.ledger is not None and bool(self.ledger.decision["promoted"])

    @property
    def promoted_harness(self) -> Harness | None:
        """The promoted live harness -- the plan's artifact -- or None."""
        return self.plan.harness if self.promoted else None

    @property
    def promoted_harness_hash(self) -> str | None:
        return str(self.ledger.decision["promoted_harness_hash"]) if self.promoted else None


def validate_round(
    incumbent: Harness,
    proposals_dir: Path | str,
    config: EvaluationConfig,
    promotion: "PromotionConfig | None" = None,
    *,
    loader_timeout_seconds: float = DEFAULT_MATERIALIZATION_TIMEOUT_SECONDS,
) -> ValidationRound:
    """Run one whole validation round: load, evaluate, decide, ledger (R9).

    The composition, in order:

    1. **Loader (U1).** Every candidate directory under ``proposals_dir`` is
       gated against the incumbent (with the caps' S6 comparison); rejections
       are carried through to the ledger, never dropped. When *nothing* loads,
       the round short-circuits: the baseline is never evaluated (no model
       calls) and no evaluation directories are created, but any loader
       rejections still persist as a rejection-only ledger in the round
       directory (R8) -- ``evaluation`` is None and ``ledger`` carries their
       records with a null baseline. Only an empty proposals directory (zero
       candidates, zero rejections) leaves nothing on disk, with ``ledger``
       also None.
    2. **Evaluation (U3).** The baseline and every loaded candidate run over
       both splits under the caps and per-candidate breakers.
    3. **Promotion (U4).** Loader rejections become decision records, the
       evaluated candidates are scored by the rule and band, and the accepted
       set resolves to a plan.
    4. **The merge leg (R7).** A ``merge`` plan's harness is re-evaluated
       through the same U3 path (under ``MERGED_SUBJECT_ID``) and the same
       rule; ``apply_merge_verdict`` turns that into the round's outcome -- a
       failed merge promotes nothing and re-marks its constituents
       ``merged_failed``. A ``single`` plan promotes its already-evaluated
       winner directly, never re-running it.
    5. **Ledger (U5).** ``write_promotion_ledger`` persists every candidate's
       record and the round decision, non-clobbering.

    Idempotent end to end: re-invoking with the same inputs replays persisted
    runs (zero new model calls), recomputes identical summaries and decisions,
    and rewrites the ledger byte-identically (a no-op).

    Args:
        incumbent: The harness the round defends.
        proposals_dir: Stage 2's candidate directories (``shrlm-proposal/v1``).
        config: The round's evaluation config (splits, verifier, caps, out
            dir, round index, repetitions, backend).
        promotion: The preregistered thresholds and bands; defaults to the
            paper's exact rule with unconstrained bands.
        loader_timeout_seconds: Wall-clock bound on each candidate's
            materialization/check subprocess.

    Returns:
        The ``ValidationRound`` with every intermediate the audit needs.

    Raises:
        ValueError: If a loaded candidate claims a reserved subject id
            (``baseline``, ``merged``) -- a directory collision, not an
            expected-invalid proposal.
    """
    from shrlm.optimization.promotion import (
        MERGED_SUBJECT_ID,
        PLAN_MERGE,
        PLAN_NONE,
        PLAN_SINGLE,
        PromotionConfig,
        PromotionPlan,
        apply_merge_verdict,
        assess_round,
        decide_subject,
        plan_promotion,
        promote_decision,
    )

    pconfig = promotion if promotion is not None else PromotionConfig()
    round_path = round_dir(config.out_dir, config.round_index)
    loaded, rejections = load_candidates(
        proposals_dir, incumbent, caps=config.caps.s6_caps(), timeout_seconds=loader_timeout_seconds
    )
    for candidate in loaded:
        if candidate.candidate_id == MERGED_SUBJECT_ID:
            raise ValueError(
                f"candidate id {MERGED_SUBJECT_ID!r} is reserved for the merged harness's "
                "re-evaluation directory"
            )

    if not loaded:
        # Nothing to compare: never run the baseline, make zero model calls.
        # R8 still demands every candidate's outcome in a persisted ledger, so
        # a round with at least one loader rejection writes a rejection-only
        # ledger (upstream reasons, null links, a null baseline) into the
        # round directory. Only a truly empty proposals directory -- zero
        # candidates, zero rejections -- stays a clean no-op that creates no
        # round directory at all.
        decisions = [decide_subject({}, rejection, pconfig) for rejection in rejections]
        plan = PromotionPlan(kind=PLAN_NONE, constituent_ids=(), harness=None, harness_hash=None)
        ledger = (
            write_promotion_ledger(
                None, decisions, plan, round_path=round_path, loader_rejections=rejections
            )
            if rejections
            else None
        )
        return ValidationRound(
            round_path=round_path,
            loader_rejections=rejections,
            evaluation=None,
            decisions=decisions,
            plan=plan,
            merge_evaluation=None,
            merge_decision=None,
            ledger=ledger,
        )

    evaluation = evaluate_validation_round(incumbent, loaded, config)
    surface_by_id = {candidate.candidate_id: candidate.surface for candidate in loaded}
    decisions = [
        decide_subject(evaluation.baseline.summary, rejection, pconfig) for rejection in rejections
    ]
    decisions += assess_round(evaluation, pconfig, surfaces=surface_by_id)
    plan = plan_promotion(incumbent, decisions, loaded)

    merge_evaluation: SubjectEvaluation | CandidateRejection | None = None
    merge_decision: CandidateDecision | None = None
    if plan.kind == PLAN_SINGLE:
        # The winner's own evaluation already is the evidence; single winners
        # promote directly, never re-run (U4's contract).
        winner_id = plan.constituent_ids[0]
        decisions = [
            promote_decision(decision) if decision.subject_id == winner_id else decision
            for decision in decisions
        ]
    elif plan.kind == PLAN_MERGE:
        assert plan.harness is not None  # a merge plan always carries its artifact
        merge_evaluation = evaluate_subject(MERGED_SUBJECT_ID, plan.harness, config)
        merge_decision = decide_subject(evaluation.baseline.summary, merge_evaluation, pconfig)
        merge_decision, decisions = apply_merge_verdict(plan, merge_decision, decisions)

    ledger = write_promotion_ledger(
        evaluation,
        decisions,
        plan,
        loader_rejections=rejections,
        merge_evaluation=merge_evaluation,
        merge_decision=merge_decision,
    )
    return ValidationRound(
        round_path=evaluation.round_path,
        loader_rejections=rejections,
        evaluation=evaluation,
        decisions=decisions,
        plan=plan,
        merge_evaluation=merge_evaluation,
        merge_decision=merge_decision,
        ledger=ledger,
    )


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
    "ValidationRound",
    "ValidationSplits",
    "evaluate_subject",
    "evaluate_validation_round",
    "load_promotion_ledger",
    "load_summary",
    "split_aggregate",
    "split_dir",
    "subject_dir",
    "validate_round",
    "write_promotion_ledger",
]
