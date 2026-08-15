"""Script 1: surface activity over rounds -- feeds Graph 1 (lever-touch over time).

``surface_activity_over_rounds`` reads every round's ``promotions.jsonl`` (via
``shrlm.experiment.promotion_rounds``) and counts, per ``(round_index,
surface)``, how many candidates targeted that surface (``attempted_count``)
and how many of those were promoted (``promoted_count``). On top of that it
derives the running distinct-surface counts (``cumulative_surfaces_attempted``
/ ``cumulative_surfaces_promoted``) that drive the dashed-vs-solid line chart,
and the two grids (attempted, promoted) the heatmap needs -- both readable
straight off the tidy table by pivoting on ``surface``.

A record with ``surface`` null -- a loader-gate rejection whose surface was
never resolved, or the merged harness's own re-evaluation record, which spans
more than one surface -- is not a single-surface touch and is excluded from
every surface-level count. It is not silently dropped, though: it is tallied
into a parallel ``unattributed_rows_by_round`` table so a caller can see, per
round, how many ledger rows the surface-level view could not place.

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
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shrlm.experiment.promotion_rounds import iter_promotion_rounds
from shrlm.optimization.promotion import DECISION_ACCEPTED, DECISION_PROMOTED, SURFACE_HARNESS_FIELDS
from shrlm.optimization.validation import ROLE_CONSTITUENT

# S1..S9 in the codebase's canonical order (shrlm.optimization.promotion), so
# the output grid always has all nine columns even when a surface was never
# touched in this run.
CANONICAL_SURFACES: tuple[str, ...] = tuple(SURFACE_HARNESS_FIELDS)

SURFACE_ACTIVITY_FILENAME = "surface_activity.csv"
UNATTRIBUTED_FILENAME = "surface_activity_unattributed.csv"

# Fraction of a round's ledger rows landing as unattributed above which the
# CLI prints a warning -- a nudge to look, not a hard threshold.
UNATTRIBUTED_WARN_FRACTION = 0.25


@dataclass(frozen=True)
class SurfaceActivityRow:
    """One ``(round_index, surface)`` cell of the tidy output table."""

    round_index: int
    surface: str
    attempted_count: int
    promoted_count: int
    cumulative_surfaces_attempted: int
    cumulative_surfaces_promoted: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "round_index": self.round_index,
            "surface": self.surface,
            "attempted_count": self.attempted_count,
            "promoted_count": self.promoted_count,
            "cumulative_surfaces_attempted": self.cumulative_surfaces_attempted,
            "cumulative_surfaces_promoted": self.cumulative_surfaces_promoted,
        }


@dataclass(frozen=True)
class UnattributedRow:
    """One round's tally of surface-null ledger rows, for visibility."""

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
    return record["decision"] == DECISION_ACCEPTED and record.get("merge", {}).get(
        "role"
    ) == ROLE_CONSTITUENT


def surface_activity_over_rounds(
    out_dir: Path | str,
) -> tuple[list[SurfaceActivityRow], list[UnattributedRow]]:
    """Build the tidy surface-activity table plus the unattributed-rows tally.

    Returns:
        ``(rows, unattributed)``: ``rows`` has one entry per
        ``(round_index, surface)`` pair, dense over every round found and
        every canonical S1-S9 surface (zero-filled where nothing happened);
        ``unattributed`` has one entry per round with a nonzero row count
        found on disk, even when its unattributed count is zero.
    """
    # counts[(round_index, surface)] = {"attempted_count": n, "promoted_count": n}
    counts: dict[tuple[int, str], dict[str, int]] = {}
    unattributed_by_round: dict[int, int] = {}
    total_by_round: dict[int, int] = {}
    round_indices: list[int] = []

    for round_index, records, _decision in iter_promotion_rounds(out_dir):
        round_indices.append(round_index)
        total_by_round[round_index] = len(records)
        unattributed_by_round[round_index] = 0
        for record in records:
            surface = record.get("surface")
            if surface is None:
                unattributed_by_round[round_index] += 1
                continue
            entry = counts.setdefault(
                (round_index, surface), {"attempted_count": 0, "promoted_count": 0}
            )
            entry["attempted_count"] += 1
            if _is_promoted(record):
                entry["promoted_count"] += 1

    rows: list[SurfaceActivityRow] = []
    ever_attempted: set[str] = set()
    ever_promoted: set[str] = set()
    for round_index in round_indices:
        round_entries = {
            surface: counts.get((round_index, surface), {"attempted_count": 0, "promoted_count": 0})
            for surface in CANONICAL_SURFACES
        }
        # All of this round's surfaces join the running sets before any row is
        # emitted, so every row for this round reports the *same* cumulative
        # count -- "up through this round," independent of iteration order.
        for surface, entry in round_entries.items():
            if entry["attempted_count"] > 0:
                ever_attempted.add(surface)
            if entry["promoted_count"] > 0:
                ever_promoted.add(surface)
        cumulative_attempted = len(ever_attempted)
        cumulative_promoted = len(ever_promoted)
        for surface in CANONICAL_SURFACES:
            entry = round_entries[surface]
            rows.append(
                SurfaceActivityRow(
                    round_index=round_index,
                    surface=surface,
                    attempted_count=entry["attempted_count"],
                    promoted_count=entry["promoted_count"],
                    cumulative_surfaces_attempted=cumulative_attempted,
                    cumulative_surfaces_promoted=cumulative_promoted,
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


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    with path.open("w", newline="") as handle:
        if not rows:
            return
        fieldnames = list(rows[0].to_dict())  # type: ignore[attr-defined]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``surface_activity.csv`` and ``surface_activity_unattributed.csv``."""
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.surface_activity",
        description="Per-round, per-surface attempted/promoted counts (feeds Graph 1).",
    )
    parser.add_argument("out_dir", help="the experiment directory to read (holds opt/)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rows, unattributed = surface_activity_over_rounds(out_dir)
    if not rows:
        sys.stderr.write(f"no validation rounds with a promotion ledger found under {out_dir}\n")
        return 1

    activity_path = out_dir / SURFACE_ACTIVITY_FILENAME
    unattributed_path = out_dir / UNATTRIBUTED_FILENAME
    _write_csv(activity_path, rows)
    _write_csv(unattributed_path, unattributed)

    for entry in unattributed:
        if entry.total_rows and entry.unattributed_count / entry.total_rows > UNATTRIBUTED_WARN_FRACTION:
            sys.stderr.write(
                f"warning: round {entry.round_index} has {entry.unattributed_count}/"
                f"{entry.total_rows} ledger rows with no resolved surface (loader "
                "rejections and/or a merged-harness record) -- surface-level counts "
                "for this round are undercounting activity\n"
            )

    sys.stdout.write(f"Wrote {activity_path}\n")
    sys.stdout.write(f"Wrote {unattributed_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "CANONICAL_SURFACES",
    "SURFACE_ACTIVITY_FILENAME",
    "UNATTRIBUTED_FILENAME",
    "SurfaceActivityRow",
    "UnattributedRow",
    "main",
    "surface_activity_over_rounds",
]
