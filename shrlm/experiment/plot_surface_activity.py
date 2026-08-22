"""Static figure for Graph 1: harness lever-touch over time.

Reads the tidy CSVs ``surface_activity.py`` writes into an analysis snapshot
(``surface_activity.csv``, ``surface_activity_unattributed.csv``) and renders
one PNG with two panels into that same snapshot's ``plots/`` directory:

- **Panel A**: the running distinct-surface counts as two step functions --
  ``cumulative_surfaces_attempted`` (dashed) and ``cumulative_surfaces_promoted``
  (solid) -- against an "all declared surfaces" reference line that steps per
  round to the size of that round's own declared surface set (R12, KTD5).
  The y-limit, ticks, and axis label are sized by the largest declared count
  across the plotted rounds; none of them is a literal, and none is the
  current code's surface count, which is not the count a pre-S10 round had.
- **Panel B**: two surface x round heatmaps, one for ``attempted_count`` and
  one for ``promoted_count``, sharing one color scale and colorbar so the two
  are directly comparable (promoted is always <= attempted, and the shared
  scale makes that visible rather than each heatmap re-normalizing to its own
  max).

Only the canonical surfaces enter either panel. The table also carries a
``merged`` category row per round (KTD7) -- the merged harness spans several
surfaces by construction, so it is neither an extra heatmap row nor part of
the distinct-surface counts; the lookups here are keyed by
``CANONICAL_SURFACES``, which is what keeps it inert.

A zero-count heatmap cell is not one thing (R13). The table's
``surface_source`` column says which, per cell, and Panel B draws the three
states differently: a surface the round's harness declared and nothing touched
is the blank cell; a surface that round's harness never declared (a pre-S10
round's S10 cell) is a crosshatched grey cell; a cell whose round's harness
could not be read at all is a dotted-outline cell with a ``?`` -- unknown,
never folded into either of the other two. ``declared_surfaces_by_round``
reads the same column back into a per-round declared set, which is what a
per-round total has to be derived from rather than from the current code's
surface count. A round whose set is unknown gets no reference segment at
all -- no count is true of it, so none is drawn -- and if every plotted round
is unknown the axis label says so instead of stating a total.

Rounds the aggregation could not confirm complete (``round_complete`` or
``runs_complete`` not ``true``) are marked with a hatched band behind Panel A
and a starred tick label on every panel, and counted in a caption. The PNG is
what a reader draws conclusions from, so completeness that stopped at the CSV
would leave the honesty claim unmet exactly where it matters.

Palette: the repo's dataviz skill reference palette, shared through
``plot_style`` -- categorical slot 1 (blue) for both panels, since "attempted"
and "promoted" are the same underlying metric measured two ways (promoted is a
subset of attempted every round), not two independent identities; linestyle
carries that distinction instead of a second hue. The heatmaps reuse the
palette's blue sequential ramp (100->700), consistent with Panel A's hue.
This is a static, paper-bound PNG (not an interactive/themed artifact), so it
renders once against the palette's light chart surface -- there is no dark
variant to select here.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from shrlm.experiment.analysis_io import PLOTS_DIR
from shrlm.experiment.plot_style import (
    BASELINE,
    BLUE,
    DPI,
    GRIDLINE,
    MUTED_INK,
    PARTIAL_GID_PREFIX,
    PARTIAL_HATCH,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    PlotInputError,
    add_snapshot_arguments,
    apply_style,
    draw_footer,
    partial_caption,
    partial_rounds,
    partial_tick_labels,
    read_rows,
    resolve_snapshot,
    save_figure,
)
from shrlm.experiment.rounds import SURFACE_SOURCE_UNDECLARED, SURFACE_SOURCE_UNKNOWN
from shrlm.experiment.surface_activity import (
    CANONICAL_SURFACES,
    SURFACE_ACTIVITY_FILENAME,
    UNATTRIBUTED_FILENAME,
    UNATTRIBUTED_WARN_FRACTION,
)

OUTPUT_FILENAME = "surface_activity.png"

# How Panel B marks the two cell states a zero count cannot show (R13). The
# gid prefixes are what a test reads back; the hatch is deliberately not
# ``PARTIAL_HATCH`` (the partial-round band in Panel A), so a reader never sees
# the same texture mean two things on one figure.
UNDECLARED_GID_PREFIX = "undeclared-"
UNKNOWN_GID_PREFIX = "unknown-"
# Panel A's per-round "all declared surfaces" reference segments (R12), one
# Line2D per round with a known declared set, gid ``REFERENCE_GID_PREFIX<round>``.
REFERENCE_GID_PREFIX = "declared-total-"
UNDECLARED_HATCH = "xxx"
UNKNOWN_LINESTYLE = ":"
UNKNOWN_GLYPH = "?"

# The palette's blue sequential ramp, steps 100->700 (references/palette.md).
# Single-consumer, so it stays here rather than in ``plot_style``.
BLUE_SEQUENTIAL_STOPS = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    BLUE,  # the ramp's 700 step is the shared categorical blue, not a second value
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# A cell with exactly zero activity reads as *blank*, not "a pale blue value" --
# masked to NaN before imshow and painted the chart surface color, so an
# untouched surface (S1, ...) is visually empty rather than a faint tint of
# the same ramp used for real counts.
BLUE_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "shrlm_blue_seq", BLUE_SEQUENTIAL_STOPS
).with_extremes(bad=SURFACE)


def _read_activity_csv(snapshot_dir: Path) -> list[dict]:
    rows = read_rows(snapshot_dir, SURFACE_ACTIVITY_FILENAME)
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["attempted_count"] = int(row["attempted_count"])
        row["promoted_count"] = int(row["promoted_count"])
        row["cumulative_surfaces_attempted"] = int(row["cumulative_surfaces_attempted"])
        row["cumulative_surfaces_promoted"] = int(row["cumulative_surfaces_promoted"])
    return rows


def _read_unattributed_csv(snapshot_dir: Path) -> list[dict]:
    rows = read_rows(snapshot_dir, UNATTRIBUTED_FILENAME)
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["unattributed_count"] = int(row["unattributed_count"])
        row["total_rows"] = int(row["total_rows"])
    return rows


def _mark_partial_rounds(ax: "plt.Axes", rounds: Sequence[int], partial: set[int]) -> None:
    """Hatch the band behind each round that is not confirmed complete.

    A band rather than a per-point marker because Panel A already spends
    filled-vs-hollow circles on attempted-vs-promoted; a third marker state on
    the same points would collide with a distinction the reader needs more.
    """
    for index in sorted(partial):
        ax.axvspan(
            index - 0.5,
            index + 0.5,
            facecolor="none",
            edgecolor=BASELINE,
            hatch=PARTIAL_HATCH,
            linewidth=0,
            zorder=0,
            gid=f"{PARTIAL_GID_PREFIX}{index}",
        )


def _declared_counts(rounds: Sequence[int], activity_rows: list[dict]) -> dict[int, int | None]:
    """``round_index -> |declared set|`` for the plotted rounds; ``None`` when unknown."""
    declared = declared_surfaces_by_round(activity_rows)
    return {
        round_index: (None if declared.get(round_index) is None else len(declared[round_index]))
        for round_index in rounds
    }


def _draw_declared_total_reference(
    ax: "plt.Axes", rounds: Sequence[int], counts: dict[int, int | None]
) -> None:
    """The "all declared surfaces" reference, stepped per round (R12, KTD5).

    One dotted horizontal segment per round, spanning the same ``+-0.5`` band
    the partial-round hatch uses, at that round's declared count; where the
    count changes between adjacent rounds the incoming segment carries the
    riser, so a mixed-vintage tree reads as a step from nine to ten rather
    than two unrelated lines. A round with an unknown declared set gets no
    segment: no total is true of it, and drawing the neighbouring count
    across it would claim one. The label sits at the last round with a known
    count.
    """
    previous: int | None = None
    last_known: tuple[int, int] | None = None
    for round_index in rounds:
        count = counts[round_index]
        if count is None:
            previous = None
            continue
        xs = [round_index - 0.5, round_index + 0.5]
        ys = [count, count]
        if previous is not None and previous != count:
            xs.insert(0, round_index - 0.5)
            ys.insert(0, previous)
        ax.plot(
            xs,
            ys,
            color=BASELINE,
            linewidth=1,
            linestyle=(0, (1, 2)),
            zorder=1,
            gid=f"{REFERENCE_GID_PREFIX}{round_index}",
        )
        previous = count
        last_known = (round_index, count)
    if last_known is not None:
        ax.text(
            last_known[0] + 0.5,
            last_known[1],
            " all declared surfaces",
            color=MUTED_INK,
            fontsize=9,
            va="center",
            ha="left",
        )


def _plot_cumulative_panel(
    ax: "plt.Axes", rounds: list[int], activity_rows: list[dict], partial: set[int]
) -> None:
    """Panel A: dashed attempted vs solid promoted step lines.

    The y-axis is sized by the largest declared surface count across the
    plotted rounds (never a literal, never the current code's count), so a
    round that touched every declared surface plots inside the axes.
    """
    by_round: dict[int, dict] = {}
    for row in activity_rows:
        by_round.setdefault(
            row["round_index"],
            {
                "attempted": row["cumulative_surfaces_attempted"],
                "promoted": row["cumulative_surfaces_promoted"],
            },
        )
    attempted = [by_round[r]["attempted"] for r in rounds]
    promoted = [by_round[r]["promoted"] for r in rounds]

    counts = _declared_counts(rounds, activity_rows)
    known_counts = [count for count in counts.values() if count is not None]
    largest_declared = max(known_counts) if known_counts else None
    # The ceiling is the largest declared count; the plotted values are folded
    # in only so a table that disagrees with its own declarations still draws
    # unclipped rather than hiding the disagreement off-axis.
    ceiling = max(largest_declared or 0, *attempted, *promoted, 1)

    _mark_partial_rounds(ax, rounds, partial)
    _draw_declared_total_reference(ax, rounds, counts)

    ax.step(
        rounds, attempted, where="post", color=BLUE, linewidth=2, linestyle="--", label="attempted"
    )
    ax.step(
        rounds, promoted, where="post", color=BLUE, linewidth=2, linestyle="-", label="promoted"
    )
    ax.scatter(rounds, attempted, s=24, color=BLUE, marker="o", zorder=3, facecolors="none")
    ax.scatter(rounds, promoted, s=24, color=BLUE, marker="o", zorder=3)

    ax.set_ylim(0, ceiling + 0.6)
    ax.set_yticks(range(0, ceiling + 1))
    ax.set_xticks(rounds)
    ax.set_xticklabels(partial_tick_labels(rounds, partial))
    ax.set_xlabel("round")
    if largest_declared is None:
        ax.set_ylabel("distinct surfaces (declared total unknown)")
    else:
        ax.set_ylabel(f"distinct surfaces (of {largest_declared})")
    ax.set_title("Surfaces touched over time", color=PRIMARY_INK, fontsize=12, loc="left")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower right", labelcolor=SECONDARY_INK)

    # End-of-line direct labels (the one point per line worth calling out).
    ax.annotate(
        f"{attempted[-1]}",
        (rounds[-1], attempted[-1]),
        textcoords="offset points",
        xytext=(6, 4),
        color=SECONDARY_INK,
        fontsize=9,
    )
    ax.annotate(
        f"{promoted[-1]}",
        (rounds[-1], promoted[-1]),
        textcoords="offset points",
        xytext=(6, -10),
        color=SECONDARY_INK,
        fontsize=9,
    )


def _grid_for(activity_rows: list[dict], rounds: list[int], metric: str) -> list[list[int]]:
    """Dense surface x round matrix (S1 at the top row) for one count column.

    Keyed by ``CANONICAL_SURFACES`` only, so the table's ``merged`` category
    row (KTD7) never reaches the grid -- not as an extra surface row, and not
    through the shared color scale.
    """
    by_key = {(row["round_index"], row["surface"]): row[metric] for row in activity_rows}
    # imshow's default origin is "upper" (row 0 draws at the top), so building
    # row 0 = S1 here is what actually puts S1 at the top of the rendered plot.
    return [[by_key.get((r, s), 0) for r in rounds] for s in CANONICAL_SURFACES]


def _cell_sources(activity_rows: list[dict]) -> dict[tuple[int, str], str]:
    """``(round_index, surface) -> surface_source`` for every canonical-surface row."""
    return {
        (row["round_index"], row["surface"]): row["surface_source"]
        for row in activity_rows
        if row["surface"] in CANONICAL_SURFACES
    }


def declared_surfaces_by_round(activity_rows: list[dict]) -> dict[int, frozenset[str] | None]:
    """Each round's declared surface set, read back off the table's own cells (R13).

    The aggregation already resolved every cell against the round's persisted
    ``harness.json`` (``rounds.declared_surfaces``) and wrote the verdict into
    ``surface_source``, so the figure re-derives the per-round set from that
    column rather than walking the experiment tree again -- the snapshot's
    frozen CSV is the figure's only input (KTD2). A round with any ``unknown``
    cell is ``None``: its harness could not be read, so no count is true of
    it. Every other round's set is the canonical surfaces minus the cells it
    marked ``undeclared``. This is the per-round total a reference line or
    axis label must derive from, never the current code's surface count.
    """
    sources = _cell_sources(activity_rows)
    rounds = sorted({row["round_index"] for row in activity_rows})
    declared: dict[int, frozenset[str] | None] = {}
    for round_index in rounds:
        per_surface = {
            surface: sources.get((round_index, surface)) for surface in CANONICAL_SURFACES
        }
        if SURFACE_SOURCE_UNKNOWN in per_surface.values():
            declared[round_index] = None
            continue
        declared[round_index] = frozenset(
            surface
            for surface, source in per_surface.items()
            if source != SURFACE_SOURCE_UNDECLARED
        )
    return declared


def _mark_cell_states(
    ax: "plt.Axes",
    rounds: list[int],
    grid: list[list[int]],
    sources: dict[tuple[int, str], str],
) -> None:
    """Overlay the undeclared / unknown marks on one heatmap (R13).

    Drawn as patches over the image rather than as a third colour on the ramp,
    so the colour scale keeps meaning "count" and nothing else. An undeclared
    cell is filled grey and crosshatched; an unknown cell is a dotted outline
    with a ``?``. The fill is only laid down over a zero count -- a non-zero
    count under either mark (a ledger naming a surface the harness did not
    declare) would be evidence, and evidence is never painted over.
    """
    for row_index, surface in enumerate(CANONICAL_SURFACES):
        for col_index, round_index in enumerate(rounds):
            source = sources.get((round_index, surface))
            if source not in (SURFACE_SOURCE_UNDECLARED, SURFACE_SOURCE_UNKNOWN):
                continue
            value = grid[row_index][col_index]
            if source == SURFACE_SOURCE_UNDECLARED:
                ax.add_patch(
                    Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor=GRIDLINE if value == 0 else "none",
                        edgecolor=BASELINE,
                        hatch=UNDECLARED_HATCH,
                        linewidth=0,
                        zorder=2,
                        gid=f"{UNDECLARED_GID_PREFIX}{round_index}-{surface}",
                    )
                )
                continue
            ax.add_patch(
                Rectangle(
                    (col_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=MUTED_INK,
                    linestyle=UNKNOWN_LINESTYLE,
                    linewidth=1,
                    zorder=2,
                    gid=f"{UNKNOWN_GID_PREFIX}{round_index}-{surface}",
                )
            )
            if value == 0:
                ax.text(
                    col_index,
                    row_index,
                    UNKNOWN_GLYPH,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=MUTED_INK,
                )


def _declaration_caption(activity_rows: list[dict]) -> str | None:
    """The legend line for whichever of the two marks the figure actually drew."""
    present = set(_cell_sources(activity_rows).values())
    notes = []
    if SURFACE_SOURCE_UNDECLARED in present:
        notes.append("Grey crosshatched cell: surface not declared in that round's harness.")
    if SURFACE_SOURCE_UNKNOWN in present:
        notes.append(
            f"Dotted '{UNKNOWN_GLYPH}' cell: that round's harness could not be read, so its "
            "declared surfaces are unknown."
        )
    if not notes:
        return None
    return " ".join(notes)


def _plot_heatmaps(
    fig: "plt.Figure",
    axes: Sequence["plt.Axes"],
    rounds: list[int],
    activity_rows: list[dict],
    partial: set[int],
) -> None:
    """Panel B: attempted_count and promoted_count heatmaps, one shared color scale."""
    attempted_grid = _grid_for(activity_rows, rounds, "attempted_count")
    promoted_grid = _grid_for(activity_rows, rounds, "promoted_count")
    vmax = max(1, max(max(row) for row in attempted_grid))
    sources = _cell_sources(activity_rows)

    images = []
    titles = ["attempted_count", "promoted_count"]
    for ax, grid, title in zip(axes, (attempted_grid, promoted_grid), titles, strict=True):
        # NaN, not 0, so an untouched cell paints the surface color (set_bad)
        # instead of the ramp's palest step -- "never touched" reads as blank.
        masked = np.where(np.array(grid) == 0, np.nan, grid)
        image = ax.imshow(
            masked,
            cmap=BLUE_SEQUENTIAL_CMAP,
            vmin=0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        images.append(image)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(partial_tick_labels(rounds, partial))
        ax.set_yticks(range(len(CANONICAL_SURFACES)))
        ax.set_yticklabels(list(CANONICAL_SURFACES))
        ax.set_xlabel("round")
        ax.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _mark_cell_states(ax, rounds, grid, sources)
        # Sparing direct labels: only nonzero cells, so the grid stays legible.
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                if value:
                    luminance_dark = value > vmax * 0.6
                    ax.text(
                        col_index,
                        row_index,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#ffffff" if luminance_dark else PRIMARY_INK,
                    )
    axes[0].set_ylabel("surface")
    fig.colorbar(images[0], ax=list(axes), fraction=0.04, pad=0.02, label="count")


def _unattributed_caption(unattributed_rows: list[dict]) -> str | None:
    """The exclusion footnote, counting rounds that actually excluded something.

    The table holds one entry per round whether or not anything was excluded,
    so counting its length overstated the reach of the exclusions -- three
    rounds "affected" when one row in one round was dropped (R8).
    """
    excluded = [row for row in unattributed_rows if row["unattributed_count"]]
    if not excluded:
        return None
    total = sum(row["unattributed_count"] for row in excluded)
    note = (
        f"Note: {total} ledger row(s) across {len(excluded)} round(s) excluded from surface "
        "counts (no single surface: loader rejections with no proposal left on disk) -- see "
        f"{UNATTRIBUTED_FILENAME}."
    )
    flagged = [
        row
        for row in excluded
        if row["total_rows"]
        and row["unattributed_count"] / row["total_rows"] > UNATTRIBUTED_WARN_FRACTION
    ]
    if flagged:
        note += f" Rounds {', '.join(str(row['round_index']) for row in flagged)} exceed 25% unattributed."
    return note


def build_figure(snapshot_dir: Path | str, *, rendered_at: datetime | None = None) -> "plt.Figure":
    """Render the figure for one snapshot, without writing it anywhere.

    Separate from ``plot_surface_activity`` so the figure's structure -- which
    rounds carry the partial mark, what the captions say -- is assertable
    without going through a PNG.

    Raises:
        PlotInputError: A required table is missing, or the snapshot holds no
            round to plot (a header-only CSV, which the previous code walked
            straight into ``rounds[-1]`` on).
    """
    snapshot_dir = Path(snapshot_dir)
    activity_rows = _read_activity_csv(snapshot_dir)
    unattributed_rows = _read_unattributed_csv(snapshot_dir)
    rounds = sorted({row["round_index"] for row in activity_rows})
    if not rounds:
        raise PlotInputError(
            f"{snapshot_dir / SURFACE_ACTIVITY_FILENAME} holds no rounds -- nothing to plot"
        )
    partial = partial_rounds(activity_rows)

    apply_style()
    fig = plt.figure(figsize=(9, 9.5), dpi=DPI)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.15)
    ax_top = fig.add_subplot(grid_spec[0, :])
    ax_bottom_left = fig.add_subplot(grid_spec[1, 0])
    ax_bottom_right = fig.add_subplot(grid_spec[1, 1], sharey=ax_bottom_left)
    plt.setp(ax_bottom_right.get_yticklabels(), visible=False)
    ax_bottom_right.set_ylabel("")

    _plot_cumulative_panel(ax_top, rounds, activity_rows, partial)
    _plot_heatmaps(fig, (ax_bottom_left, ax_bottom_right), rounds, activity_rows, partial)

    captions = [
        partial_caption(
            partial,
            len(rounds),
            marker="Hatched band / starred round label",
            source=SURFACE_ACTIVITY_FILENAME,
        ),
        _unattributed_caption(unattributed_rows),
        _declaration_caption(activity_rows),
    ]
    text = "\n".join(caption for caption in captions if caption)
    if text:
        fig.text(
            0.5, 0.045, text, ha="center", va="bottom", fontsize=7.5, color=MUTED_INK, wrap=True
        )

    fig.suptitle(
        "Surface activity over optimization rounds",
        fontsize=14,
        color=PRIMARY_INK,
        x=0.02,
        ha="left",
    )
    draw_footer(fig, snapshot_dir, rendered_at=rendered_at)
    return fig


def plot_surface_activity(
    snapshot_dir: Path | str,
    output_path: Path | str | None = None,
    *,
    rendered_at: datetime | None = None,
) -> Path:
    """Render Graph 1 into the snapshot's re-renderable ``plots/`` directory.

    Writes ``<snapshot>/plots/surface_activity.png`` by default. ``plots/`` is
    the one part of a published snapshot that may be replaced (KTD2): the
    frozen CSVs underneath never move, and the footer's render time is what
    keeps a re-rendered PNG from misrepresenting when it was drawn.
    """
    snapshot_dir = Path(snapshot_dir)
    path = (
        Path(output_path) if output_path is not None else snapshot_dir / PLOTS_DIR / OUTPUT_FILENAME
    )
    fig = build_figure(snapshot_dir, rendered_at=rendered_at)
    try:
        return save_figure(fig, path)
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.plot_surface_activity",
        description=(
            "Render Graph 1 (surface activity over rounds) from an analysis snapshot's "
            "surface_activity.csv."
        ),
    )
    add_snapshot_arguments(parser)
    args = parser.parse_args(argv)

    try:
        snapshot_dir = resolve_snapshot(Path(args.out_dir), args.snapshot)
        output_path = plot_surface_activity(snapshot_dir)
    except PlotInputError as error:
        sys.stderr.write(f"{error}\n")
        return 1

    sys.stdout.write(f"Wrote {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "OUTPUT_FILENAME",
    "PLOTS_DIR",
    "REFERENCE_GID_PREFIX",
    "UNDECLARED_GID_PREFIX",
    "UNDECLARED_HATCH",
    "UNKNOWN_GID_PREFIX",
    "build_figure",
    "declared_surfaces_by_round",
    "main",
    "plot_surface_activity",
]
