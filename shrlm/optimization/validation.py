"""Held-out validation of one fixed batch, with resumable evidence.

``validate_round`` loads proposals, composes all admitted disjoint edits onto
an incumbent, then evaluates baseline and candidate once each on held-out
instances (with the configured repetitions). Held-in instances remain mining
inputs only. The whole candidate is promoted or rejected as one unit.

Artifacts under ``<out_dir>/round_NN``:
    validation.json                       # protocol, inputs, batch, promotion parameters
    baseline/evaluation.json              # subject contract
    baseline/heldout/round_00/             # harness, instances, runs, sha-linked traces
    baseline/summary.json
    <candidate_id or merged>/...           # the same subject layout
    promotions.jsonl                      # unscored constituents + measured candidate
    decision.json

Contracts are checked before model calls. Identical inputs replay saved runs;
legacy or changed inputs require a fresh directory. v1 evidence remains readable
for analysis. Empty rounds make no calls; loader-rejection-only rounds persist
a ledger with no baseline. Subject workers evaluate baseline and batch in
independent processes under the existing caps, breakers, and recovery rules."""

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shrlm.harness_identity import harness_hash
from shrlm.optimization.behavior import BEHAVIOR_SCHEMA, summarize_behavior
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
from shrlm.optimization.types import QualityDefinition, Verifier
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
VALIDATION_PROTOCOL = "heldout-batch/v1"
SUMMARY_FORMAT = "shrlm-validation-summary/v3"

# The promotion ledger (U5): one JSONL record per candidate (and per merged
# harness) under the round directory, plus the round's decision summary.
PROMOTIONS_FILENAME = "promotions.jsonl"
DECISION_FILENAME = "decision.json"
LEDGER_RECORD_FORMAT = "shrlm-promotion-record/v2"
DECISION_FORMAT = "shrlm-promotion-decision/v2"

# Merge participation roles a ledger record can carry: a constituent of the
# round's merge plan, or the merged harness's evaluation record.
ROLE_CONSTITUENT = "constituent"
ROLE_MERGED = "merged"


def _canonical(instance: dict[str, Any]) -> str:
    return json.dumps(instance, sort_keys=True)


@dataclass(frozen=True)
class ValidationSplits:
    """Disjoint mining and held-out validation instance lists.

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

    def evaluation_items(self) -> tuple[tuple[str, list[dict[str, Any]]], ...]:
        """Only held-out instances contribute promotion evidence."""
        return ((SPLIT_HELDOUT, self.heldout),)

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
    # How many of one subject's runs execute concurrently. Multiplies with
    # ``workers``: the two knobs are independent process tiers, so total
    # in-flight runs is ``workers x run_workers`` (KTD15).
    run_workers: int = 1
    verifier_factory: str | None = None
    client_factory: tuple[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise ValueError(f"workers must be an integer, got {self.workers!r}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if isinstance(self.run_workers, bool) or not isinstance(self.run_workers, int):
            raise ValueError(f"run_workers must be an integer, got {self.run_workers!r}")
        if self.run_workers < 1:
            raise ValueError(f"run_workers must be >= 1, got {self.run_workers}")
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
    """One subject's completed (or budget-stopped) held-out evaluation.

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
    observed_runs = []
    for entry, (_instance, completion) in zip(entries, runs, strict=True):
        terminated = entry.get("cause") in (
            VerifierCause.RESOURCE_TERMINATED.value,
            VerifierCause.RUNTIME_ERROR.value,
        )
        observed_runs.append((completion, terminated))
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
        "behavior": summarize_behavior(observed_runs),
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
        "n_runtime_errors": sum(
            1 for entry in entries if entry.get("cause") == VerifierCause.RUNTIME_ERROR.value
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
    if payload.get("format") not in (
        SUMMARY_FORMAT,
        "shrlm-validation-summary/v2",
        "shrlm-validation-summary/v1",
    ):
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


def _any_split_skipped(split_summaries: dict[str, dict[str, Any]]) -> bool:
    """Whether any split left a run unexecuted.

    A skipped run means the evaluated sample is smaller than the configured
    one, whatever the reason -- and that is a property of the subject, not of
    one split, because the promotion rule scores the subject.
    """
    return any(split.get("skipped_run_ids") for split in split_summaries.values())


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Persist one subject's summary, refusing to clobber a diverging one.

    The payload is a pure function of the persisted round state and the
    evaluation config (no timestamps), so a resume that completes the same
    evaluation rewrites identical bytes -- allowed as a no-op. Different bytes
    mean the configuration changed under an already-summarized subject (e.g. a
    raised budget re-running a previously over-budget candidate); that summary
    may already have fed a promotion decision, so the rewrite is refused and
    the operator must delete the stale file deliberately. A legacy missing
    runtime-error count is equivalent to zero and leaves the saved file intact.
    """
    if path.exists():
        previous = json.loads(path.read_text())
        # Summaries written before runtime-error containment lack this count.
        # Accept only an equivalent zero count, preserving the saved bytes.
        for split in previous.get("splits", {}).values():
            split.setdefault("n_runtime_errors", 0)
        if previous == payload:
            return
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
    """Evaluate one harness on held-out instances under the configured caps.

    A cumulative spend breaker covers every repetition. The subject contract is
    checked before preparing runs; identical saved runs are verified and reused.
    Caps violations return a structured rejection before any subject files exist.
    An optional pre-charged breaker supports resuming governed execution."""
    limits = governed_limits(subject_id, harness.runtime_policy, config.caps)
    if isinstance(limits, CandidateRejection):
        return limits
    if breaker is None:
        breaker = CandidateSpendBreaker(config.caps)

    subject_path = subject_dir(config.out_dir, config.round_index, subject_id)
    check_subject_contract(subject_id, harness, config)
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_id, instances in config.splits.evaluation_items():
        split_path = split_dir(config.out_dir, config.round_index, subject_id, split_id)
        round_config = RoundConfig(
            round_index=EVAL_ROUND_INDEX,
            run_workers=config.run_workers,
            client_factory=config.client_factory,
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

    # The split aggregate carries the hash its persisted harness.json
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
            f"{sorted(split_versions)}; its figures cannot anchor promotion"
        )
    summary = {
        "format": SUMMARY_FORMAT,
        "validation_protocol": VALIDATION_PROTOCOL,
        "behavior_contract": BEHAVIOR_SCHEMA,
        # Which cost-accounting rules produced every figure below -- taken from
        # the runs, never assumed to be this build's. Stamping the current
        # version unconditionally would let a legacy round re-aggregated after
        # the correction claim an accounting it was not priced under.
        ACCOUNTING_VERSION_KEY: next(iter(split_versions), ACCOUNTING_VERSION),
        "subject_id": subject_id,
        "harness_hash": next(iter(split_summaries.values()))["harness_hash"],
        "repetitions": config.repetitions,
        # Derived from the splits' skipped sets as well as the breaker. The
        # promotion rule reads THIS outcome, and dispatch can stop while spend
        # is still inside the budget -- the reservation gate holds back a
        # slot's worth of headroom per in-flight run, so a subject can end with
        # runs never executed and the breaker never tripped. Deriving from the
        # breaker alone would mark such a subject `completed` and let the
        # preregistered rule score a sample that never ran.
        "outcome": (
            OUTCOME_OVER_BUDGET
            if (breaker.tripped or _any_split_skipped(split_summaries))
            else OUTCOME_COMPLETED
        ),
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


@dataclass(frozen=True)
class ValidationSubject:
    """The harness to measure, independent of one-surface proposal metadata."""

    candidate_id: str
    harness: Harness
    harness_hash: str


def evaluation_contract(config: EvaluationConfig) -> dict[str, Any]:
    """Behavior-changing evaluation inputs, excluding operational concurrency."""
    config_method = getattr(config.verifier, "config", None)
    verifier_config = dict(config_method()) if callable(config_method) else {}
    if "primary_quality" in verifier_config:
        QualityDefinition.from_dict(verifier_config["primary_quality"])
    return {
        "validation_protocol": VALIDATION_PROTOCOL,
        "verifier_config": verifier_config,
        "heldout": config.splits.heldout,
        "repetitions": config.repetitions,
        "caps": asdict(config.caps),
        "backend": config.backend,
        "backend_kwargs": config.backend_kwargs,
        "verifier_type": f"{type(config.verifier).__module__}.{type(config.verifier).__qualname__}",
    }


def check_contract(path: Path, payload: dict[str, Any]) -> None:
    """Refuse legacy or changed evidence before any paid evaluation work."""
    if not path.exists() and path.parent.exists():
        if any(path.parent.rglob("harness.json")) or any(path.parent.rglob(SUMMARY_FILENAME)):
            raise ValueError(f"{path.parent} has legacy validation evidence; use a fresh directory")
        if (path.parent / DECISION_FILENAME).exists():
            raise ValueError(
                f"{path.parent} has a legacy validation decision; use a fresh directory"
            )
    _persist_once(
        path,
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        f"{path} has a different validation contract; use a fresh directory",
    )


def check_subject_contract(subject_id: str, harness: Harness, config: EvaluationConfig) -> None:
    check_contract(
        subject_dir(config.out_dir, config.round_index, subject_id) / "evaluation.json",
        {**evaluation_contract(config), "harness_hash": harness_hash(harness)},
    )


def evaluate_validation_round(
    incumbent: Harness,
    candidates: Iterable[LoadedCandidate | ValidationSubject],
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
        for subject_id, harness in [(BASELINE_ID, incumbent)] + [
            (candidate.candidate_id, candidate.harness) for candidate in candidates
        ]:
            limits = governed_limits(subject_id, harness.runtime_policy, config.caps)
            if isinstance(limits, CandidateRejection):
                if subject_id == BASELINE_ID:
                    raise ValueError(BASELINE_REJECTED_MESSAGE.format(reason=limits.reason))
                continue
            check_subject_contract(subject_id, harness, config)
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
    subject: SubjectEvaluation | CandidateRejection | LoadedCandidate,
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
        "validation_protocol": VALIDATION_PROTOCOL,
        **decision.to_dict(),
        "merge": {
            "role": role,
            "constituent_ids": list(constituent_ids) if role is not None else None,
        },
        "selection_excluded": excluded_reason,
        "batch_subject_id": "merged" if role == ROLE_CONSTITUENT else None,
        "links": _subject_links(subject) if isinstance(subject, SubjectEvaluation) else None,
    }


def write_promotion_ledger(
    evaluation: "RoundEvaluation | None",
    decisions: Iterable["CandidateDecision"],
    plan: "PromotionPlan",
    *,
    round_path: Path | None = None,
    loader_rejections: Iterable[CandidateRejection] = (),
    constituents: Iterable[LoadedCandidate] = (),
) -> PromotionLedger:
    """Persist one record per rejected proposal, unscored constituent, and measured subject.

    ``decisions`` must cover those subjects exactly. A multi-edit batch supplies
    its admitted proposals as ``constituents``; their records have no measurements
    or evaluation links. The combined subject alone may be promoted. Rejection-only
    rounds supply ``round_path`` instead of an evaluation. Identical rewrites are
    no-ops; divergent audit history is refused."""
    # promotion imports this module, so its vocabulary is imported at call time.
    from shrlm.optimization.promotion import (
        DECISION_BUNDLED,
        DECISION_PROMOTED,
        MERGED_SUBJECT_ID,
        PLAN_MERGE,
        PLAN_NONE,
        PLAN_SINGLE,
    )

    if (evaluation is None) == (round_path is None):
        raise ValueError(
            "pass exactly one round anchor: an evaluated round's RoundEvaluation, or an "
            "explicit round_path for a loader-rejection-only round"
        )
    rejections = list(loader_rejections)
    admitted = list(constituents)
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

    subjects: list[SubjectEvaluation | CandidateRejection | LoadedCandidate] = [
        *rejections,
        *admitted,
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

    if evaluation is not None:
        if len(evaluation.candidates) != 1 or plan.kind not in (PLAN_SINGLE, PLAN_MERGE):
            raise ValueError("a batch ledger requires exactly one evaluated candidate")
        measured = evaluation.candidates[0]
        expected_id = MERGED_SUBJECT_ID if plan.kind == PLAN_MERGE else plan.constituent_ids[0]
        if not isinstance(measured, SubjectEvaluation) or (
            measured.subject_id != expected_id or measured.harness_hash != plan.harness_hash
        ):
            raise ValueError("the evaluated subject must match the planned batch id and hash")
        for subject in (evaluation.baseline, measured):
            if subject.summary.get("validation_protocol") != VALIDATION_PROTOCOL:
                raise ValueError("a batch ledger cannot use legacy validation evidence")
        expected_constituents = set(plan.constituent_ids) if plan.kind == PLAN_MERGE else set()
        if {candidate.candidate_id for candidate in admitted} != expected_constituents:
            raise ValueError("a batch ledger requires every unscored constituent exactly once")
        for candidate in admitted:
            decision = by_id[candidate.candidate_id]
            if (
                decision.decision != DECISION_BUNDLED
                or decision.rule is not None
                or decision.band is not None
                or decision.harness_hash != candidate.harness_hash
            ):
                raise ValueError("batch constituents must be bundled without individual scores")

    constituent_ids = set(plan.constituent_ids) if plan.kind == PLAN_MERGE else set()
    records = [
        _ledger_record(
            by_id[subject_id],
            subject,
            ROLE_CONSTITUENT
            if subject_id in constituent_ids
            else (ROLE_MERGED if subject_id == MERGED_SUBJECT_ID else None),
            plan.constituent_ids,
            plan.excluded.get(subject_id),
        )
        for subject_id, subject in zip(subject_ids, subjects, strict=True)
    ]

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
        "validation_protocol": VALIDATION_PROTOCOL,
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
        "n_candidates": len(subjects) - (1 if plan.kind == PLAN_MERGE else 0),
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
        if record.get("format") not in (LEDGER_RECORD_FORMAT, "shrlm-promotion-record/v1"):
            raise ValueError(
                f"{ledger_path} holds a record for {record.get('subject_id')!r} that is not "
                f"a {LEDGER_RECORD_FORMAT} record"
            )
    decision_path = path / DECISION_FILENAME
    decision = json.loads(decision_path.read_text())
    if decision.get("format") not in (DECISION_FORMAT, "shrlm-promotion-decision/v1"):
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
    have lived. ``evaluation.candidates`` contains only the measured batch subject.
    """

    round_path: Path
    loader_rejections: list[CandidateRejection]
    evaluation: RoundEvaluation | None
    decisions: list["CandidateDecision"]
    plan: "PromotionPlan"
    ledger: PromotionLedger | None

    @property
    def promoted(self) -> bool:
        """Whether this round promoted the evaluated batch harness."""
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
    preflight_profile: str | None = None,
    prior_evaluations: list[dict[str, Any]] | None = None,
) -> ValidationRound:
    """Load and freeze a disjoint proposal batch, evaluate it, then persist one verdict.

    Only baseline and the combined candidate run, on held-out instances. There
    is no individual scoring, post-validation selection, or second merge leg.
    Single edits keep their candidate id; multi-edit batches evaluate as ``merged``.
    Duplicate surfaces or reserved ids fail before model calls. Loader rejections
    remain visible in the ledger. Empty proposal directories create no artifacts."""
    from shrlm.optimization.candidates import select_preflight_profile
    from shrlm.optimization.promotion import (
        DECISION_BUNDLED,
        MERGED_SUBJECT_ID,
        PLAN_MERGE,
        CandidateDecision,
        PromotionConfig,
        decide_subject,
        plan_batch,
        promote_decision,
    )

    round_path = round_dir(config.out_dir, config.round_index)
    contract_path = round_path / "validation.json"
    persisted_contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if preflight_profile is None:
        preflight_profile = (
            persisted_contract.get("preflight_profile", "generic/v1")
            if persisted_contract is not None
            else select_preflight_profile(evaluation_contract(config)["verifier_config"])
        )
    if persisted_contract is not None and preflight_profile != persisted_contract.get(
        "preflight_profile", "generic/v1"
    ):
        raise ValueError("different validation contract: preflight profile changed")
    profile_fields = (
        {"preflight_profile": preflight_profile}
        if persisted_contract is None or "preflight_profile" in persisted_contract
        else {}
    )
    pconfig = promotion if promotion is not None else PromotionConfig()
    loaded, rejections = load_candidates(
        proposals_dir,
        incumbent,
        caps=config.caps.s6_caps(),
        timeout_seconds=loader_timeout_seconds,
        preflight_profile=preflight_profile,
    )
    for candidate in loaded:
        if candidate.candidate_id in (BASELINE_ID, MERGED_SUBJECT_ID):
            raise ValueError(f"candidate id {candidate.candidate_id!r} is reserved")
    plan = plan_batch(incumbent, loaded)
    admitted_inputs = [
        {"id": c.candidate_id, "surface": c.surface, "hash": c.harness_hash} for c in loaded
    ]
    batch_hash = plan.harness_hash
    duplicate = next(
        (
            record
            for record in (prior_evaluations or [])
            if batch_hash is not None
            and record["incumbent_hash"] == harness_hash(incumbent)
            and record["harness_hash"] == batch_hash
        ),
        None,
    )
    if duplicate:
        reason = f"not evaluated: identical rejected evaluated harness under this incumbent; prior round {duplicate['round']} subject {duplicate['subject_id']}; shared batch outcome, no individual score"
        rejections.extend(
            CandidateRejection(c.candidate_id, "duplicate_evaluation", reason, str(c.path))
            for c in loaded
        )
        loaded = []
        plan = plan_batch(incumbent, [])
    decisions = [decide_subject({}, rejection, pconfig) for rejection in rejections]
    evaluation = None
    constituents = loaded if plan.kind == PLAN_MERGE else []

    if loaded or rejections or (round_path / "validation.json").exists():
        # Freeze the admitted batch and rejected inputs before the baseline runs.
        check_contract(
            round_path / "validation.json",
            {
                **evaluation_contract(config),
                **profile_fields,
                "incumbent_hash": harness_hash(incumbent),
                "batch_hash": batch_hash,
                "constituents": admitted_inputs,
                "prior_evaluations": prior_evaluations or [],
                "duplicate_evaluation": duplicate,
                "rejections": [r.to_dict() for r in rejections],
                "promotion": {
                    "tau_regression": pconfig.tau_regression,
                    "tau_improvement": pconfig.tau_improvement,
                    "cost_band": [pconfig.cost_band.lower, str(pconfig.cost_band.upper)],
                    "sub_call_band": [
                        pconfig.sub_call_band.lower,
                        str(pconfig.sub_call_band.upper),
                    ],
                },
            },
        )
    if loaded:
        assert plan.harness is not None and plan.harness_hash is not None
        subject_id = MERGED_SUBJECT_ID if constituents else loaded[0].candidate_id
        subject = ValidationSubject(subject_id, plan.harness, plan.harness_hash)
        limits = governed_limits(subject_id, plan.harness.runtime_policy, config.caps)
        if isinstance(limits, CandidateRejection):
            raise ValueError(f"combined candidate violates caps: {limits.reason}")
        evaluation = evaluate_validation_round(incumbent, [subject], config)
        if evaluation.baseline.over_budget:
            raise ValueError(
                "the baseline evaluation ran over budget; a partial baseline cannot anchor promotion"
            )
        verdict = decide_subject(
            evaluation.baseline.summary,
            evaluation.candidates[0],
            pconfig,
            surface=None if constituents else loaded[0].surface,
        )
        if verdict.accepted:
            verdict = promote_decision(verdict)
        decisions.extend(
            CandidateDecision(
                subject_id=c.candidate_id,
                decision=DECISION_BUNDLED,
                reasons=(
                    f"evaluated only as part of {subject_id}; shared decision: {verdict.decision}",
                ),
                tau_regression=pconfig.tau_regression,
                tau_improvement=pconfig.tau_improvement,
                harness_hash=c.harness_hash,
                surface=c.surface,
            )
            for c in constituents
        )
        decisions.append(verdict)

    ledger = (
        write_promotion_ledger(
            evaluation,
            decisions,
            plan,
            round_path=round_path if evaluation is None else None,
            loader_rejections=rejections,
            constituents=constituents,
        )
        if (loaded or rejections)
        else None
    )
    return ValidationRound(
        round_path=round_path,
        loader_rejections=rejections,
        evaluation=evaluation,
        decisions=decisions,
        plan=plan,
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
    "VALIDATION_PROTOCOL",
    "EvaluationConfig",
    "PromotionLedger",
    "RoundEvaluation",
    "SubjectEvaluation",
    "ValidationRound",
    "ValidationSplits",
    "ValidationSubject",
    "check_contract",
    "check_subject_contract",
    "evaluate_subject",
    "evaluate_validation_round",
    "evaluation_contract",
    "load_promotion_ledger",
    "load_summary",
    "split_aggregate",
    "split_dir",
    "subject_dir",
    "validate_round",
    "write_promotion_ledger",
]
