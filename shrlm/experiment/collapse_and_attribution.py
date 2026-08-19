"""Whole-input collapse rate and root/child failure-attribution share (proposal 3.5, 1.2).

Two investigation findings shaped this script's design (see the reference doc
for the full writeup):

1. ``records.jsonl`` only ever carries ``collapse_ratio`` for FAILING runs --
   the gate is a pure mining-stage choice (``mining.py``'s ``record_failure``
   returns before walking a passing run's trace), not a structural one:
   ``walker.walk()`` needs only a completion's trajectory metadata, which is
   persisted identically for every run, pass or fail, at ``runs/<run_id>.json``.
2. ``eval_summary.json`` has zero per-run attribution -- ``evaluation.py``
   never calls ``walker``/``grounding``/``attribution`` anywhere. But eval
   traces are persisted through the exact same ``driver.run_round`` path as
   mining traces, and ``failing_level`` is fully deterministic whenever a
   ``SubVerifier`` exists for the environment (``grounding.py``'s
   ``apply_sub_verifier`` needs no LLM call at all -- only ``agent_mechanism``
   / ``causal_status``, which this script does not compute, need one).

Given both, this script re-walks every run's persisted trace directly (via
``shrlm.optimization.driver.load_round``, the same rehydration
``split_aggregate`` already uses for sub-call counts) rather than reading
``records.jsonl``'s failures-only figures. That makes ``collapse_rate`` a true
outcome-independent rate (matching the proposal's own framing: "how often the
harness routes... instead of decomposing," not "...among failures") and makes
the optimization and evaluation phases directly comparable, using the
identical mechanism -- there is no ``records.jsonl`` equivalent on the
evaluation side, so re-walking is the only way to get evaluation-phase numbers
at all.

Denominators, deliberately different per metric
    ``collapse_rate`` is over every WALKABLE run (pass or fail) -- it is a
    harness-behavior metric, not a failure-cause metric. ``root_failure_share``
    / ``child_failure_share`` / ``ungrounded_share`` are over FAILURES only
    (the proposal names them "failure attribution share"); a passing run has
    no failure to locate, so it never enters these three denominators. A run
    whose trace cannot be walked at all (no trajectory metadata -- e.g. a
    RESOURCE_TERMINATED run that never produced one) is excluded from every
    numerator and denominator and counted in ``n_unwalkable`` instead, so it
    is never silently folded into a rate as if grounded=False or collapsed.

``no_recursion_failure_share`` -- an addition beyond the four named columns
    ``FailingLevel`` has four members: ROOT, CHILD, NO_RECURSION (grounded,
    but a failure with no sub-calls to attribute to either level), and
    UNDETERMINED (folded into "ungrounded" here, alongside runs with no
    registered SubVerifier at all). Reporting only root/child/ungrounded would
    silently drop NO_RECURSION failures from view -- a real category, not
    noise -- so this script adds one more share to keep the four columns
    (root + child + no_recursion + ungrounded) exhaustive over ``n_failures``.

``sub_verifier_available`` -- environment coverage, not a per-run fact
    Only GraphWalks has a registered ``SubVerifier``; OOLONG-Pairs deliberately
    has none ("child correctness here is a semantic-labeling judgment no
    deterministic check can recompute" -- its own module docstring). A group
    with ``sub_verifier_available=False`` shows ``ungrounded_share=1.0``
    structurally, not because grounding failed at any individual run -- this
    column exists so a reader can tell the two apart at a glance.

Completeness, carried beside every rate (R2, KTD5, KTD9)
    A rate computed over a truncated round is not wrong, but read as if it
    were complete it is misleading -- and before this the two were
    indistinguishable once they reached a CSV. So every row carries the
    completeness of the group it summarizes, taken from the shared inventory
    rather than re-derived here: mining rounds report their expected and
    missing run counts plus ``runs_complete`` / ``evidence_complete`` /
    ``round_complete``; test sets report the summary's ``outcome``, their
    skipped-run count, and ``complete``.

    Nothing is dropped for being incomplete (KTD5). Every discovered round is
    emitted, including a completed no-op round whose mining stage persisted no
    manifest at all -- that round is ``round_complete`` true with zero runs and
    null rates, which is a real outcome rather than missing data, and skipping
    it would make the round vanish from the series entirely. The same holds on
    the evaluation side: test sets come from the inventory rather than from
    ``eval_summary.json``'s own grid, so a set with run directories but no
    summary entry is emitted with ``complete`` unknown instead of being
    invisible to this table.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shrlm.environments.graphwalks import GraphWalksSubVerifier
from shrlm.experiment.analysis_io import (
    PROVENANCE_FILENAME,
    Snapshot,
    allocate_snapshot,
    tristate,
    write_csv,
)
from shrlm.experiment.evaluation import EVAL_DIR, EVAL_SUMMARY_FILENAME
from shrlm.experiment.rounds import EvalSetRecord, RoundRecord, discover_rounds
from shrlm.optimization.bundle import BUNDLE_FILENAME
from shrlm.optimization.driver import INSTANCES_FILE, MANIFEST_FILE, TRACES_DIR, load_round
from shrlm.optimization.grounding import apply_sub_verifier
from shrlm.optimization.taxonomy import FailingLevel
from shrlm.optimization.types import SubVerifier
from shrlm.optimization.validation import EVAL_ROUND_INDEX
from shrlm.optimization.walker import walk

# clustering.py's own convention for "collapsed" (clustering.py:109,
# the >0.8 threshold behind its shared_symptoms prose) -- reused here rather
# than inventing a second threshold for the same signal.
COLLAPSE_THRESHOLD = 0.8

# oolong_pairs deliberately has no SubVerifier; see this module's docstring.
SUB_VERIFIERS: dict[str, SubVerifier] = {"graphwalks": GraphWalksSubVerifier()}

PHASE_OPTIMIZATION = "optimization"
PHASE_EVALUATION = "evaluation"

OPTIMIZATION_FILENAME = "collapse_and_attribution_optimization.csv"
EVALUATION_FILENAME = "collapse_and_attribution_evaluation.csv"

# The names this aggregation's two phases are recorded under in a snapshot's
# provenance -- separate, because a batch may run one and not the other.
TOOL_NAME_OPTIMIZATION = "collapse_and_attribution_optimization"
TOOL_NAME_EVALUATION = "collapse_and_attribution_evaluation"

# Declared field order per phase (KTD8): a phase that finds no group still has
# to write a header-only CSV, which it could not do from an absent first row.
GROUP_METRICS_FIELDNAMES = (
    "n_runs",
    "n_walked",
    "n_unwalkable",
    "n_failures",
    "collapse_rate",
    "root_failure_share",
    "child_failure_share",
    "no_recursion_failure_share",
    "ungrounded_share",
    "sub_verifier_available",
)

# Completeness travels with the numbers it qualifies (R2), appended after them
# so the metric columns keep the positions a pre-change reader knows.
ROUND_COMPLETENESS_FIELDNAMES = (
    "n_expected_runs",
    "n_missing_runs",
    "runs_complete",
    "evidence_complete",
    "round_complete",
)
EVAL_COMPLETENESS_FIELDNAMES = ("outcome", "n_skipped", "complete")

OPTIMIZATION_FIELDNAMES = (
    "round_index",
    "environment",
    *GROUP_METRICS_FIELDNAMES,
    *ROUND_COMPLETENESS_FIELDNAMES,
)
EVALUATION_FIELDNAMES = (
    "condition_id",
    "test_set_id",
    "environment",
    "length",
    *GROUP_METRICS_FIELDNAMES,
    *EVAL_COMPLETENESS_FIELDNAMES,
)


@dataclass(frozen=True)
class GroupMetrics:
    """One group's (a round, or a condition x test set) collapse/attribution reading."""

    n_runs: int
    n_walked: int
    n_unwalkable: int
    n_failures: int
    collapse_rate: float | None
    root_failure_share: float | None
    child_failure_share: float | None
    no_recursion_failure_share: float | None
    ungrounded_share: float | None
    sub_verifier_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_walked": self.n_walked,
            "n_unwalkable": self.n_unwalkable,
            "n_failures": self.n_failures,
            "collapse_rate": self.collapse_rate,
            "root_failure_share": self.root_failure_share,
            "child_failure_share": self.child_failure_share,
            "no_recursion_failure_share": self.no_recursion_failure_share,
            "ungrounded_share": self.ungrounded_share,
            "sub_verifier_available": self.sub_verifier_available,
        }


@dataclass(frozen=True)
class RoundCompleteness:
    """One mining round's KTD9 completeness, as a row carries it.

    Read straight off the shared inventory's ``RoundRecord`` -- discovery owns
    the truth table, and a second opinion computed here is exactly the drift
    this plan removed. ``n_expected_runs`` and ``n_missing_runs`` are ``None``
    when the experiment's configuration cannot be resolved (KTD6); the
    ``runs_complete`` flag beside them then reads ``unknown``, which is what
    keeps an unmeasurable round distinguishable from a truncated one in the
    written CSV rather than both showing an empty cell.
    """

    n_expected_runs: int | None
    n_missing_runs: int | None
    runs_complete: bool | None
    evidence_complete: bool
    round_complete: bool

    @classmethod
    def from_record(cls, record: RoundRecord) -> "RoundCompleteness":
        return cls(
            n_expected_runs=record.mining_runs.n_expected,
            n_missing_runs=record.mining_runs.n_missing,
            runs_complete=record.mining_runs.runs_complete,
            evidence_complete=record.evidence_complete,
            round_complete=record.round_complete,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_expected_runs": self.n_expected_runs,
            "n_missing_runs": self.n_missing_runs,
            "runs_complete": tristate(self.runs_complete),
            "evidence_complete": tristate(self.evidence_complete),
            "round_complete": tristate(self.round_complete),
        }


@dataclass(frozen=True)
class EvalCompleteness:
    """One test set's KTD9 completeness, as the evaluation summary recorded it.

    ``complete`` is ``None`` -- ``unknown`` once written -- for a set that has
    run directories but no ``eval_summary.json`` entry: an evaluation that
    crashed before it aggregated is unmeasured, not failed.
    """

    outcome: str | None
    n_skipped: int
    complete: bool | None

    @classmethod
    def from_record(cls, record: EvalSetRecord) -> "EvalCompleteness":
        return cls(outcome=record.outcome, n_skipped=record.n_skipped, complete=record.complete)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "n_skipped": self.n_skipped,
            "complete": tristate(self.complete),
        }


def empty_group_metrics(*, sub_verifier_available: bool) -> GroupMetrics:
    """A group with zero runs on disk (e.g. entirely spend-skipped) -- explicit nulls, not zeros."""
    return GroupMetrics(
        n_runs=0,
        n_walked=0,
        n_unwalkable=0,
        n_failures=0,
        collapse_rate=None,
        root_failure_share=None,
        child_failure_share=None,
        no_recursion_failure_share=None,
        ungrounded_share=None,
        sub_verifier_available=sub_verifier_available,
    )


def compute_group_metrics(
    runs: list[tuple[dict[str, Any], Any]],
    verdicts: list[Any],
    sub_verifier: SubVerifier | None,
) -> GroupMetrics:
    """Walk every run's trace once; collapse over all walked runs, attribution over failures only."""
    n_unwalkable = 0
    n_walked = 0
    collapsed = 0
    failure_levels: list[FailingLevel | None] = []  # None means ungrounded

    for (instance, completion), verdict in zip(runs, verdicts, strict=True):
        try:
            root, stats = walk(completion)
        except ValueError:
            n_unwalkable += 1
            continue
        n_walked += 1
        if stats.collapse_ratio > COLLAPSE_THRESHOLD:
            collapsed += 1
        if not verdict.passed:
            grounding = apply_sub_verifier(instance, root, sub_verifier)
            failure_levels.append(grounding.failing_level if grounding.grounded else None)

    n_failures = len(failure_levels)
    root_count = sum(1 for level in failure_levels if level is FailingLevel.ROOT)
    child_count = sum(1 for level in failure_levels if level is FailingLevel.CHILD)
    no_recursion_count = sum(1 for level in failure_levels if level is FailingLevel.NO_RECURSION)
    ungrounded_count = sum(1 for level in failure_levels if level is None)

    return GroupMetrics(
        n_runs=len(runs),
        n_walked=n_walked,
        n_unwalkable=n_unwalkable,
        n_failures=n_failures,
        collapse_rate=(collapsed / n_walked) if n_walked else None,
        root_failure_share=(root_count / n_failures) if n_failures else None,
        child_failure_share=(child_count / n_failures) if n_failures else None,
        no_recursion_failure_share=(no_recursion_count / n_failures) if n_failures else None,
        ungrounded_share=(ungrounded_count / n_failures) if n_failures else None,
        sub_verifier_available=sub_verifier is not None,
    )


# ---------------------------------------------------------------------------
# Optimization phase: grouped by round
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationRow:
    round_index: int
    environment: str | None
    metrics: GroupMetrics
    completeness: RoundCompleteness

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "environment": self.environment,
            **self.metrics.to_dict(),
            **self.completeness.to_dict(),
        }


def _environment_for_round(mining_round_path: Path) -> str | None:
    bundle_path = mining_round_path / BUNDLE_FILENAME
    if not bundle_path.exists():
        return None
    bundle = json.loads(bundle_path.read_text())
    environment = bundle.get("config", {}).get("verifier_config", {}).get("environment")
    return str(environment) if environment else None


def collapse_and_attribution_optimization(out_dir: Path | str) -> list[OptimizationRow]:
    """One row per discovered round: collapse rate + failure-attribution shares, re-walked from traces.

    Every round discovery finds gets a row, flagged with its completeness
    (KTD5). A round whose mining stage persisted no manifest has no trace to
    walk, so its metrics are explicit nulls rather than zeros -- but it stays in
    the table, because a completed no-op round is a real outcome and dropping
    it would silently shorten the series a reader plots.
    """
    rows: list[OptimizationRow] = []
    for record in discover_rounds(out_dir).rounds:
        environment = _environment_for_round(record.mining_round_path)
        sub_verifier = SUB_VERIFIERS.get(environment) if environment else None
        if not record.has_manifest:
            metrics = empty_group_metrics(sub_verifier_available=sub_verifier is not None)
        else:
            runs, verdicts, _envelope, _entries = load_round(
                record.mining_parent, record.round_index
            )
            metrics = compute_group_metrics(runs, verdicts, sub_verifier)
        rows.append(
            OptimizationRow(
                round_index=record.round_index,
                environment=environment,
                metrics=metrics,
                completeness=RoundCompleteness.from_record(record),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluation phase: grouped by condition x test set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRow:
    condition_id: str
    test_set_id: str
    environment: str | None
    length: str | None
    metrics: GroupMetrics
    completeness: EvalCompleteness

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "test_set_id": self.test_set_id,
            "environment": self.environment,
            "length": self.length,
            **self.metrics.to_dict(),
            **self.completeness.to_dict(),
        }


def eval_summary_path(out_dir: Path | str) -> Path:
    """``<out_dir>/eval/eval_summary.json`` -- ``evaluation.py``'s own ``eval_dir / EVAL_SUMMARY_FILENAME``."""
    return Path(out_dir) / EVAL_DIR / EVAL_SUMMARY_FILENAME


def collapse_and_attribution_evaluation(out_dir: Path | str) -> list[EvaluationRow]:
    """One row per condition x test set from the shared inventory, re-walked from traces.

    The grid comes from ``rounds.discover_rounds`` rather than from
    ``eval_summary.json`` directly (KTD1): the summary is one of two things
    discovery reconciles, and a set that exists on disk but never reached the
    summary -- an evaluation that crashed before it aggregated -- would be
    invisible to a table that enumerated the summary's own conditions. Such a
    set is emitted with ``complete`` unknown, which is the honest reading;
    ``outcome`` and the skipped-run count come from the same inventory, so this
    table and every other analysis agree on one verdict per set.
    """
    rows: list[EvaluationRow] = []
    for record in discover_rounds(out_dir).eval_sets:
        environment = record.environment
        sub_verifier = SUB_VERIFIERS.get(environment) if environment else None
        if not (record.round_path / MANIFEST_FILE).exists():
            # Entirely skipped -- e.g. the spend breaker tripped before this set
            # ran a single instance. Emitted with explicit nulls, not dropped.
            metrics = empty_group_metrics(sub_verifier_available=sub_verifier is not None)
        else:
            runs, verdicts, _envelope, _entries = load_round(record.set_path, EVAL_ROUND_INDEX)
            metrics = compute_group_metrics(runs, verdicts, sub_verifier)
        rows.append(
            EvaluationRow(
                condition_id=record.condition_id,
                test_set_id=record.set_id,
                environment=environment,
                length=record.length,
                metrics=metrics,
                completeness=EvalCompleteness.from_record(record),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _record_round_sources(snapshot: Snapshot, round_path: Path) -> None:
    """Every artifact ``load_round`` reads for one round, hashed for provenance.

    The per-run traces are recorded as one directory entry rather than one line
    per run: this analysis re-walks every trace, so the traces ARE its input
    and a changed one must change the manifest -- but a real experiment holds
    thousands of them, and a provenance record nobody can read is not a
    provenance record.
    """
    snapshot.record_source(round_path / MANIFEST_FILE)
    snapshot.record_source(round_path / INSTANCES_FILE)
    snapshot.record_source_dir(round_path / TRACES_DIR)


def write_collapse_and_attribution_optimization(
    snapshot: Snapshot, out_dir: Path | str, rows: Sequence[OptimizationRow]
) -> Path:
    """Write the optimization table into an allocated snapshot, with sources."""
    analyzed: list[int] = []
    for record in discover_rounds(out_dir).rounds:
        analyzed.append(record.round_index)
        _record_round_sources(snapshot, record.mining_round_path)
        snapshot.record_source(record.mining_round_path / BUNDLE_FILENAME)
    snapshot.record_rounds(analyzed)

    csv_path = snapshot.path / OPTIMIZATION_FILENAME
    write_csv(csv_path, rows, fieldnames=OPTIMIZATION_FIELDNAMES)
    return csv_path


def write_collapse_and_attribution_evaluation(
    snapshot: Snapshot, out_dir: Path | str, rows: Sequence[EvaluationRow]
) -> Path:
    """Write the evaluation table into an allocated snapshot, with sources."""
    out_dir = Path(out_dir)
    snapshot.record_source(eval_summary_path(out_dir))
    # Paths come from the inventory rather than being rebuilt here, so the
    # provenance manifest names exactly the files the rows were read from.
    for record in discover_rounds(out_dir).eval_sets:
        _record_round_sources(snapshot, record.round_path)
    snapshot.record_eval_sets(f"{row.condition_id}/{row.test_set_id}" for row in rows)

    csv_path = snapshot.path / EVALUATION_FILENAME
    write_csv(csv_path, rows, fieldnames=EVALUATION_FIELDNAMES)
    return csv_path


def run_collapse_and_attribution(out_dir: Path | str, snapshot: Snapshot, *, phase: str) -> Path:
    """Aggregate one phase into the caller's snapshot (KTD2).

    Takes an allocated snapshot rather than allocating one, so a batch that
    aggregates both phases -- or this phase alongside the other analyses --
    leaves one directory a reader can compare across, not four.
    """
    out_dir = Path(out_dir)
    if phase == PHASE_OPTIMIZATION:
        return write_collapse_and_attribution_optimization(
            snapshot, out_dir, collapse_and_attribution_optimization(out_dir)
        )
    return write_collapse_and_attribution_evaluation(
        snapshot, out_dir, collapse_and_attribution_evaluation(out_dir)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.collapse_and_attribution",
        description=(
            "Whole-input sub-call collapse rate and root/child failure-attribution share, "
            "re-walked from persisted traces (proposal sections 1.2 and 3.5)."
        ),
    )
    parser.add_argument("out_dir", help="the experiment directory to read")
    parser.add_argument(
        "--phase", choices=[PHASE_OPTIMIZATION, PHASE_EVALUATION], required=True, help="which phase to aggregate"
    )
    parser.add_argument(
        "--out",
        dest="snapshot_parent",
        metavar="DIR",
        default=None,
        help=(
            "parent directory for the timestamped analysis snapshot "
            "(default: <out_dir>/analysis)"
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)

    # The two phases are written out separately rather than dispatched through
    # one variable: their row types are different, and a shared call site would
    # only typecheck by widening both to `Any` -- which is how a row of the
    # wrong shape would reach the wrong writer without anything noticing.
    # The emptiness check comes before allocation in both, so a phase with
    # nothing to analyze leaves no empty unpublished directory behind.
    if args.phase == PHASE_OPTIMIZATION:
        optimization_rows = collapse_and_attribution_optimization(out_dir)
        if not optimization_rows:
            sys.stderr.write(f"no {args.phase} groups found under {out_dir}\n")
            return 1
        snapshot = allocate_snapshot(out_dir, parent=args.snapshot_parent)
        tool_name = TOOL_NAME_OPTIMIZATION
        filename = OPTIMIZATION_FILENAME
        ok = snapshot.run_tool(
            tool_name,
            lambda: write_collapse_and_attribution_optimization(
                snapshot, out_dir, optimization_rows
            ),
        )
    else:
        if not eval_summary_path(out_dir).exists():
            sys.stderr.write(f"{eval_summary_path(out_dir)} not found\n")
            return 1
        evaluation_rows = collapse_and_attribution_evaluation(out_dir)
        if not evaluation_rows:
            sys.stderr.write(f"no {args.phase} groups found under {out_dir}\n")
            return 1
        snapshot = allocate_snapshot(out_dir, parent=args.snapshot_parent)
        tool_name = TOOL_NAME_EVALUATION
        filename = EVALUATION_FILENAME
        ok = snapshot.run_tool(
            tool_name,
            lambda: write_collapse_and_attribution_evaluation(snapshot, out_dir, evaluation_rows),
        )

    snapshot.publish()
    if not ok:
        sys.stderr.write(f"{tool_name} failed; see {snapshot.path / PROVENANCE_FILENAME}\n")
        return 1

    sys.stdout.write(f"Wrote {snapshot.path / filename}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "COLLAPSE_THRESHOLD",
    "EVALUATION_FIELDNAMES",
    "EVALUATION_FILENAME",
    "EVAL_COMPLETENESS_FIELDNAMES",
    "OPTIMIZATION_FIELDNAMES",
    "OPTIMIZATION_FILENAME",
    "PHASE_EVALUATION",
    "PHASE_OPTIMIZATION",
    "ROUND_COMPLETENESS_FIELDNAMES",
    "SUB_VERIFIERS",
    "TOOL_NAME_EVALUATION",
    "TOOL_NAME_OPTIMIZATION",
    "EvalCompleteness",
    "EvaluationRow",
    "GroupMetrics",
    "OptimizationRow",
    "RoundCompleteness",
    "collapse_and_attribution_evaluation",
    "collapse_and_attribution_optimization",
    "compute_group_metrics",
    "empty_group_metrics",
    "eval_summary_path",
    "main",
    "run_collapse_and_attribution",
    "write_collapse_and_attribution_evaluation",
    "write_collapse_and_attribution_optimization",
]
