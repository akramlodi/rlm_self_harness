"""Script 1: surface activity over rounds -- feeds Graph 1 (lever-touch over time).

``surface_activity_over_rounds`` reads every round's ``promotions.jsonl`` (via
``shrlm.experiment.rounds``, the shared discovery module) and counts, per
``(round_index, surface)``, how many candidates targeted that surface
(``attempted_count``) and how many of those were promoted
(``promoted_count``). On top of that it
derives the running distinct-surface counts (``cumulative_surfaces_attempted``
/ ``cumulative_surfaces_promoted``) that drive the dashed-vs-solid line chart,
and the two grids (attempted, promoted) the heatmap needs -- both readable
straight off the tidy table by pivoting on ``surface``.

Not every ledger row states its surface, and the ones that do not are not all
the same case, so each row is placed through ``rounds.resolve_surface`` and
every output cell carries the ``surface_source`` that placed it (KTD3):

``ledger``
    The ledger recorded the surface; it is used verbatim.
``backfilled``
    The ledger's value was missing or null and the round's persisted proposal
    artifact supplied it. This covers every loader-gate rejection the running
    loop writes -- ``CandidateDecision.to_dict`` always emits the key, so those
    rows serialize as ``surface: null`` rather than as an absent key -- and
    every pre-``surface`` ledger, whose rows have no key at all.
``merged``
    The merged harness's own re-evaluation record, which spans more than one
    surface by construction. It is its own ``merged`` category row rather than
    a missing S1-S9 touch (KTD7), so it is never backfilled and never counted
    unattributed.
``none``
    Nothing on disk can place the row: no ledger value and no proposal to
    recover one from. Only these rows are unattributed, and they are not
    silently dropped -- they are tallied into a parallel
    ``unattributed_rows_by_round`` table so a caller can see, per round, how
    many rows the surface-level view genuinely could not place.

A cell whose count mixes recorded and recovered rows reports ``backfilled``:
the weaker claim is the true one for the cell as a whole. A cell nothing landed
in reports ``none``, because no row attributed it.

Every row also carries the round's completeness (R2, KTD9), read off the same
discovery that found the ledger: ``round_complete`` (the loop's own round
marker, present and recording this round) and ``runs_complete`` (whether the
round's mining stage persisted every run it planned -- the only per-round run
plan persisted state can know, and the one a partial round's counts come from).
``runs_complete`` is ``unknown`` rather than ``false`` when the experiment's
configuration cannot be resolved, so a round nobody can measure is never
plotted as one that failed.

A ``promoted_count`` row counts a decision of ``promoted`` -- the single
winner's own record, or the merged harness's own re-evaluation record (which
carries no single surface, so it never contributes here) -- *and* a decision
of ``accepted`` whose ``merge.role`` is ``constituent``: a surface that won
its own slot and was folded into a promoted merge, even though its own
record's ``decision`` field never flips to ``promoted`` (only the merged
subject's does). Without that second condition, every merge would show
``cumulative_surfaces_promoted`` undercounting the surfaces the incumbent
actually gained that round.
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
    MERGED_CATEGORY,
    SURFACE_SOURCE_BACKFILLED,
    SURFACE_SOURCE_LEDGER,
    SURFACE_SOURCE_MERGED,
    SURFACE_SOURCE_NONE,
    ExperimentInventory,
    RoundRecord,
    discover_rounds,
    iter_promotion_rounds,
    resolve_surface,
)
from shrlm.optimization.promotion import (
    DECISION_ACCEPTED,
    DECISION_PROMOTED,
    SURFACE_HARNESS_FIELDS,
)
from shrlm.optimization.validation import ROLE_CONSTITUENT

# S1..S9 in the codebase's canonical order (shrlm.optimization.promotion), so
# the output grid always has all nine columns even when a surface was never
# touched in this run.
CANONICAL_SURFACES: tuple[str, ...] = tuple(SURFACE_HARNESS_FIELDS)

# Every row category the table emits per round: the nine surfaces plus the
# merged harness's own category (KTD7). ``merged`` sits beside the surfaces
# rather than among them -- it is not one of the nine, so it never joins the
# distinct-surface cumulative counts and never appears in the S1-S9 heatmap.
ROW_CATEGORIES: tuple[str, ...] = (*CANONICAL_SURFACES, MERGED_CATEGORY)

SURFACE_ACTIVITY_FILENAME = "surface_activity.csv"
UNATTRIBUTED_FILENAME = "surface_activity_unattributed.csv"

# The name this aggregation is recorded under in a snapshot's provenance.
TOOL_NAME = "surface_activity"

# Field order is declared, not read off the first row: an experiment with no
# ledgered round still has to write a header-only CSV rather than an empty
# file, and an empty file cannot say which columns it would have had (KTD8).
SURFACE_ACTIVITY_FIELDNAMES = (
    "round_index",
    "surface",
    "surface_source",
    "attempted_count",
    "promoted_count",
    "cumulative_surfaces_attempted",
    "cumulative_surfaces_promoted",
    "round_complete",
    "runs_complete",
)
UNATTRIBUTED_FIELDNAMES = ("round_index", "unattributed_count", "total_rows")

# Fraction of a round's ledger rows landing as unattributed above which the
# CLI prints a warning -- a nudge to look, not a hard threshold.
UNATTRIBUTED_WARN_FRACTION = 0.25


@dataclass(frozen=True)
class SurfaceActivityRow:
    """One ``(round_index, surface)`` cell of the tidy output table.

    ``surface`` is one of the nine surface ids or the ``merged`` category;
    ``surface_source`` says what placed the rows counted here (KTD3).
    """

    round_index: int
    surface: str
    surface_source: str
    attempted_count: int
    promoted_count: int
    cumulative_surfaces_attempted: int
    cumulative_surfaces_promoted: int
    round_complete: bool
    runs_complete: bool | None

    def to_dict(self) -> dict[str, int | str]:
        return {
            "round_index": self.round_index,
            "surface": self.surface,
            "surface_source": self.surface_source,
            "attempted_count": self.attempted_count,
            "promoted_count": self.promoted_count,
            "cumulative_surfaces_attempted": self.cumulative_surfaces_attempted,
            "cumulative_surfaces_promoted": self.cumulative_surfaces_promoted,
            "round_complete": tristate(self.round_complete),
            "runs_complete": tristate(self.runs_complete),
        }


@dataclass(frozen=True)
class UnattributedRow:
    """One round's tally of ledger rows nothing on disk could place.

    ``unattributed_count`` counts only ``SURFACE_SOURCE_NONE`` rows -- rows a
    backfill could not recover and the merged category does not claim --
    against ``total_rows``, every line the round's ledger holds.
    """

    round_index: int
    unattributed_count: int
    total_rows: int

    def to_dict(self) -> dict[str, int]:
        return {
            "round_index": self.round_index,
            "unattributed_count": self.unattributed_count,
            "total_rows": self.total_rows,
        }


def _is_promoted(record: dict) -> bool:
    """A round's own ``promoted`` record, or a merge constituent folded into one.

    ``decision`` only reads ``"promoted"`` on the single winner's record, or
    the merged harness's own re-evaluation record. A constituent surface that
    won its slot and joined a *successful* merge stays ``"accepted"`` forever
    (``apply_merge_verdict`` never rewrites it) -- its promotion is visible
    only through ``merge.role == "constituent"`` on that same record.
    """
    if record["decision"] == DECISION_PROMOTED:
        return True
    return (
        record["decision"] == DECISION_ACCEPTED
        and record.get("merge", {}).get("role") == ROLE_CONSTITUENT
    )


def _cell_source(category: str, cell_sources: set[str] | None) -> str:
    """What placed the rows in one cell, as the cell reports it (KTD3).

    A cell nothing landed in reports ``none`` -- no row attributed it. A cell
    mixing recorded and recovered rows reports ``backfilled``, the weaker of
    the two claims: part of that count was reconstructed from a proposal, and
    a reader comparing it against a purely recorded cell must know that.
    """
    if category == MERGED_CATEGORY:
        return SURFACE_SOURCE_MERGED
    if not cell_sources:
        return SURFACE_SOURCE_NONE
    if SURFACE_SOURCE_BACKFILLED in cell_sources:
        return SURFACE_SOURCE_BACKFILLED
    return SURFACE_SOURCE_LEDGER


def surface_activity_over_rounds(
    out_dir: Path | str, *, inventory: ExperimentInventory | None = None
) -> tuple[list[SurfaceActivityRow], list[UnattributedRow]]:
    """Build the tidy surface-activity table plus the unattributed-rows tally.

    Args:
        out_dir: The experiment directory to read.
        inventory: The caller's already-computed discovery (see
            ``run_surface_activity``); omitted, one is discovered here.

    Returns:
        ``(rows, unattributed)``: ``rows`` has one entry per
        ``(round_index, category)`` pair, dense over every round found and
        every row category -- the nine canonical surfaces plus ``merged`` --
        zero-filled where nothing happened; ``unattributed`` has one entry per
        round with a ledger on disk, even when its unattributed count is zero.
    """
    # One discovery pass for the completeness every row carries and for the
    # proposals the backfill joins against; the ledger iteration below is built
    # on the same inventory, so the two agree on which rounds exist by
    # construction.
    inventory = inventory if inventory is not None else discover_rounds(out_dir)
    discovered: dict[int, RoundRecord] = {record.round_index: record for record in inventory.rounds}

    # counts[(round_index, category)] = {"attempted_count": n, "promoted_count": n}
    counts: dict[tuple[int, str], dict[str, int]] = {}
    # The distinct sources that placed each cell's rows, so a cell built partly
    # from recovered surfaces can say so.
    sources: dict[tuple[int, str], set[str]] = {}
    unattributed_by_round: dict[int, int] = {}
    total_by_round: dict[int, int] = {}
    round_indices: list[int] = []

    for round_index, records, _decision in iter_promotion_rounds(out_dir, inventory=inventory):
        round_indices.append(round_index)
        total_by_round[round_index] = len(records)
        unattributed_by_round[round_index] = 0
        round_record = discovered[round_index]
        for record in records:
            attribution = resolve_surface(record, round_record)
            if attribution.category is None:
                unattributed_by_round[round_index] += 1
                continue
            key = (round_index, attribution.category)
            entry = counts.setdefault(key, {"attempted_count": 0, "promoted_count": 0})
            sources.setdefault(key, set()).add(attribution.source)
            entry["attempted_count"] += 1
            if _is_promoted(record):
                entry["promoted_count"] += 1

    rows: list[SurfaceActivityRow] = []
    ever_attempted: set[str] = set()
    ever_promoted: set[str] = set()
    for round_index in round_indices:
        round_entries = {
            category: counts.get(
                (round_index, category), {"attempted_count": 0, "promoted_count": 0}
            )
            for category in ROW_CATEGORIES
        }
        # All of this round's surfaces join the running sets before any row is
        # emitted, so every row for this round reports the *same* cumulative
        # count -- "up through this round," independent of iteration order. The
        # merged category is deliberately not a member: it is not one of the
        # nine, and counting it would push the "distinct surfaces of 9" line
        # above what the experiment actually touched.
        for surface in CANONICAL_SURFACES:
            entry = round_entries[surface]
            if entry["attempted_count"] > 0:
                ever_attempted.add(surface)
            if entry["promoted_count"] > 0:
                ever_promoted.add(surface)
        cumulative_attempted = len(ever_attempted)
        cumulative_promoted = len(ever_promoted)
        record = discovered.get(round_index)
        for category in ROW_CATEGORIES:
            entry = round_entries[category]
            rows.append(
                SurfaceActivityRow(
                    round_index=round_index,
                    surface=category,
                    surface_source=_cell_source(category, sources.get((round_index, category))),
                    attempted_count=entry["attempted_count"],
                    promoted_count=entry["promoted_count"],
                    cumulative_surfaces_attempted=cumulative_attempted,
                    cumulative_surfaces_promoted=cumulative_promoted,
                    round_complete=record is not None and record.round_complete,
                    runs_complete=None if record is None else record.mining_runs.runs_complete,
                )
            )

    unattributed = [
        UnattributedRow(
            round_index=round_index,
            unattributed_count=unattributed_by_round[round_index],
            total_rows=total_by_round[round_index],
        )
        for round_index in round_indices
    ]
    return rows, unattributed


def write_surface_activity(
    snapshot: Snapshot,
    out_dir: Path | str,
    rows: Sequence[SurfaceActivityRow],
    unattributed: Sequence[UnattributedRow],
    *,
    inventory: ExperimentInventory | None = None,
) -> list[Path]:
    """Write both tables into an already-allocated snapshot, recording sources.

    Sources are recorded before the outputs are written, which is the ordering
    the snapshot's identity resolution assumes for a legacy tree carrying no
    recorded identity of its own.

    Three groups of them, because this table publishes three kinds of claim:
    the ledgers its counts come from, the inventory inputs its ``round_complete``
    / ``runs_complete`` columns come from, and (through
    ``record_surface_sources``) the proposal artifacts every ``backfilled``
    cell was recovered from -- a recovered surface nobody can trace back to the
    proposal that supplied it is not a reproducible number.

    ``inventory`` is the caller's already-computed discovery (see
    ``run_surface_activity``); omitted, one is discovered here.
    """
    record_ledger_sources(snapshot, out_dir, inventory=inventory)
    record_surface_sources(snapshot, out_dir, inventory=inventory)
    activity_path = snapshot.path / SURFACE_ACTIVITY_FILENAME
    unattributed_path = snapshot.path / UNATTRIBUTED_FILENAME
    write_csv(activity_path, rows, fieldnames=SURFACE_ACTIVITY_FIELDNAMES)
    write_csv(unattributed_path, unattributed, fieldnames=UNATTRIBUTED_FIELDNAMES)
    return [activity_path, unattributed_path]


def run_surface_activity(out_dir: Path | str, snapshot: Snapshot) -> list[Path]:
    """Compute both tables and write them into the caller's snapshot.

    The entry point a batch caller (a CLI run, or the post-round hook) invokes:
    it takes an already-allocated snapshot rather than allocating one, because
    every aggregation in one invocation belongs in ONE snapshot (KTD2) --
    tables a reader compares against each other must have come from a single
    pass over a single tree.

    Discovery likewise runs ONCE here and is threaded into both the table build
    and the source recording: the two ask the same question of the same
    unchanging tree, so a second walk would buy nothing but IO.
    """
    out_dir = Path(out_dir)
    inventory = discover_rounds(out_dir)
    rows, unattributed = surface_activity_over_rounds(out_dir, inventory=inventory)
    return write_surface_activity(snapshot, out_dir, rows, unattributed, inventory=inventory)


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``surface_activity.csv`` and ``surface_activity_unattributed.csv``."""
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.surface_activity",
        description="Per-round, per-surface attempted/promoted counts (feeds Graph 1).",
    )
    parser.add_argument("out_dir", help="the experiment directory to read (holds opt/)")
    add_snapshot_parent_argument(parser)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    # Discovery and the aggregation run BEFORE a snapshot exists, deliberately:
    # an experiment with nothing to analyze must not leave a stamped empty
    # directory behind (the ``if not rows`` arm below). That places them outside
    # ``run_tool``'s guard, where an unreadable tree (an ``OSError``) or a
    # malformed ledger (a ``ValueError`` out of the JSON decode) would reach the
    # user as a raw traceback rather than as a diagnosis, so they get a guard of
    # their own: one stderr line naming what failed, and a non-zero exit.
    try:
        inventory = discover_rounds(out_dir)
        rows, unattributed = surface_activity_over_rounds(out_dir, inventory=inventory)
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
        lambda: write_surface_activity(snapshot, out_dir, rows, unattributed, inventory=inventory),
    )
    if not snapshot.publish():
        sys.stderr.write(snapshot.failure_message(TOOL_NAME))
        return 1
    activity_path = snapshot.path / SURFACE_ACTIVITY_FILENAME
    unattributed_path = snapshot.path / UNATTRIBUTED_FILENAME

    for entry in unattributed:
        if (
            entry.total_rows
            and entry.unattributed_count / entry.total_rows > UNATTRIBUTED_WARN_FRACTION
        ):
            sys.stderr.write(
                f"warning: round {entry.round_index} has {entry.unattributed_count}/"
                f"{entry.total_rows} ledger rows no surface could be resolved for: "
                "no surface in the ledger and no proposal artifact left on disk to "
                "recover one from -- surface-level counts for this round are "
                "undercounting activity\n"
            )

    sys.stdout.write(f"Wrote {activity_path}\n")
    sys.stdout.write(f"Wrote {unattributed_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "CANONICAL_SURFACES",
    "ROW_CATEGORIES",
    "SURFACE_ACTIVITY_FIELDNAMES",
    "SURFACE_ACTIVITY_FILENAME",
    "TOOL_NAME",
    "UNATTRIBUTED_FIELDNAMES",
    "UNATTRIBUTED_FILENAME",
    "SurfaceActivityRow",
    "UnattributedRow",
    "main",
    "run_surface_activity",
    "surface_activity_over_rounds",
    "write_surface_activity",
]
