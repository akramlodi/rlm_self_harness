"""Script 2: incumbent quality over rounds -- feeds Graph 2 (quality over time).

``incumbent_quality_over_rounds`` walks every round's ``promotions.jsonl``
(via ``shrlm.experiment.rounds``, the shared discovery module) and tracks one
running "current incumbent" state -- held-in and held-out pass counts and run
counts -- that only moves on a ``promoted`` decision. Every round still emits
exactly one row, so the series is a clean step function even through rounds
that promoted nothing.

Seeding the baseline
    The state starts from the first round that actually scored a candidate:
    every scored row's ``rule.<split>.baseline_pass_count`` is the same
    incumbent-before-this-round figure (all candidates in a round are scored
    against the same evaluated baseline), so any one of them seeds it. In the
    ordinary case that is round 1, where baseline == H0, matching the
    proposal's framing; a round 1 that happened to be loader-rejection-only
    (no candidate survived the loader, so nothing was ever scored) is handled
    by carrying the seed forward to the first round that does score
    something, rather than crashing -- that round's row (and any before it)
    reports null pass rates with an annotation explaining why.

Non-promoting rounds
    A round that promotes nothing leaves the incumbent state untouched. When
    the reason is structural -- a loader-gate rejection, or an evaluation
    that never finished scoring because it went over budget (``rule`` is
    null on that row) -- the row's ``annotation`` names every such candidate
    and its reason, for plotting as a marker. An ordinary scored-but-rejected
    candidate (failed the promotion rule or band; ``rule`` is populated)
    contributes no annotation, per the spec this script implements: it is
    the expected, unremarkable outcome of a round that did not improve on
    the incumbent, not a structural anomaly worth flagging.

Completeness beside the series
    Each round row carries ``round_complete`` (the loop's own round marker,
    present and recording this round) and ``runs_complete`` (whether the
    round's mining stage persisted every run it planned), both read off the
    shared discovery rather than re-derived here. A step in this series taken
    from a round the loop never finished is still worth plotting, but only
    while it is visibly partial -- and ``runs_complete`` reads ``unknown``,
    never ``false``, when the experiment's configuration cannot be resolved to
    say what the round planned (KTD9).

The all-candidates table
    ``incumbent_quality_over_rounds`` tracks only the running incumbent -- one
    number per split per round. ``all_candidate_quality_over_rounds`` is the
    second table the plotting script's optional scatter overlay needs: one
    row per *scored* candidate record (``rule`` populated -- an unscored
    loader rejection or over-budget stop has no measured pass rate to plot),
    reading its own ``candidate_pass_count`` rather than the shared
    ``baseline_pass_count``. This includes the merged harness's own
    re-evaluation record (a genuine scored candidate, just with no single
    surface), and constituent records that lost the promotion despite being
    individually scored.

    Its ``surface`` is resolved through ``rounds.resolve_surface``, exactly as
    ``surface_activity`` resolves the same ledger rows, and every row carries
    the ``surface_source`` that placed it (KTD3). The two tables ship in one
    snapshot and a reader joins them by ``(round_index, subject_id)``, so a
    raw-ledger surface here beside a backfilled one there would have been two
    answers to one question with nothing to say which was which.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shrlm.experiment.analysis_io import (
    Snapshot,
    add_snapshot_parent_argument,
    allocate_snapshot,
    record_ledger_sources,
    record_surface_sources,
    tristate,
    write_csv,
)
from shrlm.experiment.rounds import (
    ExperimentInventory,
    RoundRecord,
    discover_rounds,
    iter_promotion_rounds,
    resolve_surface,
)
from shrlm.optimization.promotion import DECISION_PROMOTED

INCUMBENT_QUALITY_FILENAME = "incumbent_quality.csv"
INCUMBENT_QUALITY_CANDIDATES_FILENAME = "incumbent_quality_candidates.csv"

# The name this aggregation is recorded under in a snapshot's provenance.
TOOL_NAME = "incumbent_quality"

# Declared field order, so an experiment with no scored round still writes a
# header-only CSV rather than an empty file (KTD8).
INCUMBENT_QUALITY_FIELDNAMES = (
    "round_index",
    "heldin_pass_rate",
    "heldout_pass_rate",
    "incumbent_changed",
    "annotation",
    "round_complete",
    "runs_complete",
)
INCUMBENT_QUALITY_CANDIDATES_FIELDNAMES = (
    "round_index",
    "subject_id",
    "decision",
    "surface",
    "surface_source",
    "heldin_pass_rate",
    "heldout_pass_rate",
)

SPLIT_HELDIN = "heldin"
SPLIT_HELDOUT = "heldout"


@dataclass(frozen=True)
class _IncumbentState:
    """The running incumbent's pass counts on both splits."""

    heldin_pass_count: int
    heldin_n_runs: int
    heldout_pass_count: int
    heldout_n_runs: int

    @property
    def heldin_pass_rate(self) -> float | None:
        return self.heldin_pass_count / self.heldin_n_runs if self.heldin_n_runs else None

    @property
    def heldout_pass_rate(self) -> float | None:
        return self.heldout_pass_count / self.heldout_n_runs if self.heldout_n_runs else None

    @classmethod
    def from_rule(cls, rule: dict[str, dict], *, key: str) -> "_IncumbentState":
        """Build from a decision's ``rule`` dict, reading ``<key>_pass_count``."""
        return cls(
            heldin_pass_count=rule[SPLIT_HELDIN][f"{key}_pass_count"],
            heldin_n_runs=rule[SPLIT_HELDIN]["n_runs"],
            heldout_pass_count=rule[SPLIT_HELDOUT][f"{key}_pass_count"],
            heldout_n_runs=rule[SPLIT_HELDOUT]["n_runs"],
        )


@dataclass(frozen=True)
class IncumbentQualityRow:
    """One round's incumbent-quality reading, emitted regardless of outcome."""

    round_index: int
    heldin_pass_rate: float | None
    heldout_pass_rate: float | None
    incumbent_changed: bool
    annotation: str | None
    round_complete: bool
    runs_complete: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "heldin_pass_rate": self.heldin_pass_rate,
            "heldout_pass_rate": self.heldout_pass_rate,
            "incumbent_changed": self.incumbent_changed,
            "annotation": self.annotation,
            "round_complete": tristate(self.round_complete),
            "runs_complete": tristate(self.runs_complete),
        }


@dataclass(frozen=True)
class CandidateQualityRow:
    """One scored candidate's own pass rates in one round (the scatter table).

    ``surface`` is resolved through ``rounds.resolve_surface`` -- the same
    resolution ``surface_activity`` runs over the same ledger rows -- so the
    two tables in one snapshot cannot disagree about the same candidate.
    Reading ``record["surface"]`` raw here would have shown a blank surface for
    every row of every ledger written before the field existed, beside a
    backfilled one in ``surface_activity.csv``, with nothing in either table
    explaining the difference. ``surface_source`` is what keeps a recovered
    surface distinguishable from a recorded one (KTD3).
    """

    round_index: int
    subject_id: str
    decision: str
    surface: str | None
    surface_source: str
    heldin_pass_rate: float | None
    heldout_pass_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "subject_id": self.subject_id,
            "decision": self.decision,
            "surface": self.surface,
            "surface_source": self.surface_source,
            "heldin_pass_rate": self.heldin_pass_rate,
            "heldout_pass_rate": self.heldout_pass_rate,
        }


def _structural_rejection_annotation(records: list[dict]) -> str | None:
    """Name every non-promoted, unscored (``rule`` null) candidate and why."""
    parts = [
        f"{record['subject_id']}: {'; '.join(record.get('reasons', ())) or record['decision']}"
        for record in records
        if record["decision"] != DECISION_PROMOTED and record.get("rule") is None
    ]
    return "; ".join(parts) if parts else None


def incumbent_quality_over_rounds(
    out_dir: Path | str, *, inventory: ExperimentInventory | None = None
) -> list[IncumbentQualityRow]:
    """One row per validation round: the incumbent's pass rates and whether it changed.

    ``inventory`` is the caller's already-computed discovery (see
    ``run_incumbent_quality``); omitted, one is discovered here.
    """
    inventory = inventory if inventory is not None else discover_rounds(out_dir)
    rows: list[IncumbentQualityRow] = []
    state: _IncumbentState | None = None
    # The same inventory the ledger iteration below walks, kept by round index
    # so every emitted row carries that round's completeness (R2).
    discovered: dict[int, RoundRecord] = {record.round_index: record for record in inventory.rounds}

    for round_index, records, _decision in iter_promotion_rounds(out_dir, inventory=inventory):
        discovery = discovered.get(round_index)
        round_complete = discovery is not None and discovery.round_complete
        runs_complete = None if discovery is None else discovery.mining_runs.runs_complete
        scored = [record for record in records if record.get("rule") is not None]
        if state is None:
            if not scored:
                rows.append(
                    IncumbentQualityRow(
                        round_index=round_index,
                        heldin_pass_rate=None,
                        heldout_pass_rate=None,
                        incumbent_changed=False,
                        annotation=(
                            "no candidate was scored this round (loader-rejection-only or "
                            "over-budget); the H0 baseline has not been observed yet"
                        ),
                        round_complete=round_complete,
                        runs_complete=runs_complete,
                    )
                )
                continue
            state = _IncumbentState.from_rule(scored[0]["rule"], key="baseline")

        promoted_record = next(
            (record for record in records if record["decision"] == DECISION_PROMOTED), None
        )
        if promoted_record is not None:
            state = _IncumbentState.from_rule(promoted_record["rule"], key="candidate")
            incumbent_changed = True
            annotation = None
        else:
            incumbent_changed = False
            annotation = _structural_rejection_annotation(records)

        rows.append(
            IncumbentQualityRow(
                round_index=round_index,
                heldin_pass_rate=state.heldin_pass_rate,
                heldout_pass_rate=state.heldout_pass_rate,
                incumbent_changed=incumbent_changed,
                annotation=annotation,
                round_complete=round_complete,
                runs_complete=runs_complete,
            )
        )
    return rows


def all_candidate_quality_over_rounds(
    out_dir: Path | str, *, inventory: ExperimentInventory | None = None
) -> list[CandidateQualityRow]:
    """One row per scored candidate record, across every round (the scatter table).

    Each row's surface goes through the shared ``resolve_surface`` (KTD3, KTD7)
    rather than the raw ledger value, so this table and ``surface_activity.csv``
    place the same candidate on the same surface, and every row says whether
    that placement was recorded by the loop, recovered from the round's
    proposal artifact, the merged harness's own category, or unresolvable.

    ``inventory`` is the caller's already-computed discovery (see
    ``run_incumbent_quality``); omitted, one is discovered here.
    """
    inventory = inventory if inventory is not None else discover_rounds(out_dir)
    # The proposal artifacts the backfill joins against live at paths only
    # discovery knows, so the round's own record travels with its ledger rows.
    discovered: dict[int, RoundRecord] = {record.round_index: record for record in inventory.rounds}

    rows: list[CandidateQualityRow] = []
    for round_index, records, _decision in iter_promotion_rounds(out_dir, inventory=inventory):
        round_record = discovered[round_index]
        for record in records:
            rule = record.get("rule")
            if rule is None:
                continue
            state = _IncumbentState.from_rule(rule, key="candidate")
            attribution = resolve_surface(record, round_record)
            rows.append(
                CandidateQualityRow(
                    round_index=round_index,
                    subject_id=record["subject_id"],
                    decision=record["decision"],
                    surface=attribution.category,
                    surface_source=attribution.source,
                    heldin_pass_rate=state.heldin_pass_rate,
                    heldout_pass_rate=state.heldout_pass_rate,
                )
            )
    return rows


def write_incumbent_quality(
    snapshot: Snapshot,
    out_dir: Path | str,
    rows: Sequence[IncumbentQualityRow],
    candidates: Sequence[CandidateQualityRow],
    *,
    inventory: ExperimentInventory | None = None,
) -> list[Path]:
    """Write both tables into an already-allocated snapshot, recording sources.

    Sources are recorded before the outputs are written, which is the ordering
    the snapshot's identity resolution assumes for a legacy tree carrying no
    recorded identity of its own.

    The candidates table resolves each row's surface through the same backfill
    ``surface_activity`` uses, so the proposal artifacts that supplied a
    recovered surface are recorded too -- a snapshot that publishes a
    ``backfilled`` value must carry the artifact the value came from, or the
    attribution cannot be rechecked later.

    ``inventory`` is the caller's already-computed discovery (see
    ``run_incumbent_quality``); omitted, one is discovered here.
    """
    record_ledger_sources(snapshot, out_dir, inventory=inventory)
    record_surface_sources(snapshot, out_dir, inventory=inventory)
    quality_path = snapshot.path / INCUMBENT_QUALITY_FILENAME
    candidates_path = snapshot.path / INCUMBENT_QUALITY_CANDIDATES_FILENAME
    write_csv(quality_path, rows, fieldnames=INCUMBENT_QUALITY_FIELDNAMES)
    write_csv(candidates_path, candidates, fieldnames=INCUMBENT_QUALITY_CANDIDATES_FIELDNAMES)
    return [quality_path, candidates_path]


def run_incumbent_quality(out_dir: Path | str, snapshot: Snapshot) -> list[Path]:
    """Compute both tables and write them into the caller's snapshot (KTD2).

    Takes an allocated snapshot rather than allocating one: every aggregation
    in one invocation belongs in the same snapshot, so the quality series and
    the surface series a reader plots together are guaranteed to describe one
    pass over one tree.

    Discovery runs ONCE here and is threaded through both tables and the source
    recording. All three ask the same question of the same unchanging tree, so
    three separate walks would only re-derive the same answer at three times
    the IO -- and a tree that somehow did change under them would be worse than
    slow, it would put two different inventories in one snapshot.
    """
    out_dir = Path(out_dir)
    inventory = discover_rounds(out_dir)
    return write_incumbent_quality(
        snapshot,
        out_dir,
        incumbent_quality_over_rounds(out_dir, inventory=inventory),
        all_candidate_quality_over_rounds(out_dir, inventory=inventory),
        inventory=inventory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``incumbent_quality.csv`` and ``incumbent_quality_candidates.csv``."""
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.incumbent_quality",
        description="Per-round incumbent held-in/held-out pass rates (feeds Graph 2).",
    )
    parser.add_argument("out_dir", help="the experiment directory to read (holds opt/)")
    add_snapshot_parent_argument(parser)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    # Discovery and both aggregations run BEFORE a snapshot exists, deliberately:
    # an experiment with nothing to analyze must not leave a stamped empty
    # directory behind (the ``if not rows`` arm below). That places them outside
    # ``run_tool``'s guard, where an unreadable tree (an ``OSError``) or a
    # malformed ledger (a ``ValueError`` out of the JSON decode) would reach the
    # user as a raw traceback rather than as a diagnosis, so they get a guard of
    # their own: one stderr line naming what failed, and a non-zero exit.
    try:
        inventory = discover_rounds(out_dir)
        rows = incumbent_quality_over_rounds(out_dir, inventory=inventory)
        candidates = all_candidate_quality_over_rounds(out_dir, inventory=inventory)
    except Exception as error:  # noqa: BLE001 -- a CLI reports, it does not traceback
        sys.stderr.write(f"could not read {out_dir}: {type(error).__name__}: {error}\n")
        return 1
    if not rows:
        # Checked before a snapshot is allocated: an experiment with nothing to
        # analyze should not leave an empty unpublished directory behind.
        sys.stderr.write(f"no validation rounds with a promotion ledger found under {out_dir}\n")
        return 1

    snapshot = allocate_snapshot(out_dir, parent=args.snapshot_parent)
    snapshot.run_tool(
        TOOL_NAME,
        lambda: write_incumbent_quality(snapshot, out_dir, rows, candidates, inventory=inventory),
    )
    if not snapshot.publish():
        sys.stderr.write(snapshot.failure_message(TOOL_NAME))
        return 1

    sys.stdout.write(f"Wrote {snapshot.path / INCUMBENT_QUALITY_FILENAME}\n")
    sys.stdout.write(f"Wrote {snapshot.path / INCUMBENT_QUALITY_CANDIDATES_FILENAME}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "INCUMBENT_QUALITY_CANDIDATES_FIELDNAMES",
    "INCUMBENT_QUALITY_CANDIDATES_FILENAME",
    "INCUMBENT_QUALITY_FIELDNAMES",
    "INCUMBENT_QUALITY_FILENAME",
    "TOOL_NAME",
    "CandidateQualityRow",
    "IncumbentQualityRow",
    "all_candidate_quality_over_rounds",
    "incumbent_quality_over_rounds",
    "main",
    "run_incumbent_quality",
    "write_incumbent_quality",
]
