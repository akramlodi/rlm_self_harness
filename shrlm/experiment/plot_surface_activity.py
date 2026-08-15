"""Static figure for Graph 1: harness lever-touch over time.

Reads the tidy CSV ``surface_activity.py`` writes (``surface_activity.csv``,
``surface_activity_unattributed.csv``) and renders one PNG with two panels:

- **Panel A**: the running distinct-surface counts as two step functions --
  ``cumulative_surfaces_attempted`` (dashed) and ``cumulative_surfaces_promoted``
  (solid) -- against a y=9 "all surfaces" reference line.
- **Panel B**: two S1-S9 x round heatmaps, one for ``attempted_count`` and one
  for ``promoted_count``, sharing one color scale and colorbar so the two are
  directly comparable (promoted is always <= attempted, and the shared scale
  makes that visible rather than each heatmap re-normalizing to its own max).

Palette: the repo's dataviz skill reference palette (``references/palette.md``)
-- categorical slot 1 (blue, `#2a78d6`) for both panels, since "attempted" and
"promoted" are the same underlying metric measured two ways (promoted is a
subset of attempted every round), not two independent identities; linestyle
carries that distinction instead of a second hue. The heatmaps reuse the
palette's blue sequential ramp (100->700), consistent with Panel A's hue.
This is a static, paper-bound PNG (not an interactive/themed artifact), so it
renders once against the palette's light chart surface -- there is no dark
variant to select here.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shrlm.experiment.surface_activity import (
    CANONICAL_SURFACES,
    SURFACE_ACTIVITY_FILENAME,
    UNATTRIBUTED_FILENAME,
    UNATTRIBUTED_WARN_FRACTION,
)

PLOTS_DIR = "plots"
OUTPUT_FILENAME = "surface_activity.png"

# shrlm's dataviz skill reference palette (references/palette.md), light mode.
BLUE = "#2a78d6"
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
PRIMARY_INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# The palette's blue sequential ramp, steps 100->700 (references/palette.md).
BLUE_SEQUENTIAL_STOPS = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
BLUE_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("shrlm_blue_seq", BLUE_SEQUENTIAL_STOPS)
# A cell with exactly zero activity reads as *blank*, not "a pale blue value" --
# masked to NaN before imshow and painted the chart surface color, so an
# untouched surface (S1, ...) is visually empty rather than a faint tint of
# the same ramp used for real counts.
BLUE_SEQUENTIAL_CMAP.set_bad(SURFACE)

DPI = 200

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = PRIMARY_INK
plt.rcParams["axes.edgecolor"] = BASELINE
plt.rcParams["axes.labelcolor"] = SECONDARY_INK
plt.rcParams["xtick.color"] = MUTED_INK
plt.rcParams["ytick.color"] = MUTED_INK
plt.rcParams["figure.facecolor"] = SURFACE
plt.rcParams["axes.facecolor"] = SURFACE
plt.rcParams["savefig.facecolor"] = SURFACE


def _read_activity_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["attempted_count"] = int(row["attempted_count"])
        row["promoted_count"] = int(row["promoted_count"])
        row["cumulative_surfaces_attempted"] = int(row["cumulative_surfaces_attempted"])
        row["cumulative_surfaces_promoted"] = int(row["cumulative_surfaces_promoted"])
    return rows


def _read_unattributed_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["unattributed_count"] = int(row["unattributed_count"])
        row["total_rows"] = int(row["total_rows"])
    return rows


def _plot_cumulative_panel(ax: "plt.Axes", rounds: list[int], activity_rows: list[dict]) -> None:
    """Panel A: dashed attempted vs solid promoted step lines, y capped at 9."""
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

    ax.axhline(9, color=BASELINE, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
    ax.text(
        rounds[-1],
        9,
        " all surfaces",
        color=MUTED_INK,
        fontsize=9,
        va="center",
        ha="left",
    )

    ax.step(
        rounds, attempted, where="post", color=BLUE, linewidth=2, linestyle="--", label="attempted"
    )
    ax.step(rounds, promoted, where="post", color=BLUE, linewidth=2, linestyle="-", label="promoted")
    ax.scatter(rounds, attempted, s=24, color=BLUE, marker="o", zorder=3, facecolors="none")
    ax.scatter(rounds, promoted, s=24, color=BLUE, marker="o", zorder=3)

    ax.set_ylim(0, 9.6)
    ax.set_yticks(range(0, 10))
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("distinct surfaces (of 9)")
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
    """Dense surface x round matrix (S1 at the top row) for one count column."""
    by_key = {(row["round_index"], row["surface"]): row[metric] for row in activity_rows}
    return [[by_key.get((r, s), 0) for r in rounds] for s in reversed(CANONICAL_SURFACES)]


def _plot_heatmaps(
    fig: "plt.Figure", axes: Sequence["plt.Axes"], rounds: list[int], activity_rows: list[dict]
) -> None:
    """Panel B: attempted_count and promoted_count heatmaps, one shared color scale."""
    attempted_grid = _grid_for(activity_rows, rounds, "attempted_count")
    promoted_grid = _grid_for(activity_rows, rounds, "promoted_count")
    vmax = max(1, max(max(row) for row in attempted_grid))

    images = []
    titles = ["attempted_count", "promoted_count"]
    for ax, grid, title in zip(axes, (attempted_grid, promoted_grid), titles, strict=True):
        # NaN, not 0, so an untouched cell paints the surface color (set_bad)
        # instead of the ramp's palest step -- "never touched" reads as blank.
        masked = np.where(np.array(grid) == 0, np.nan, grid)
        image = ax.imshow(
            masked, cmap=BLUE_SEQUENTIAL_CMAP, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest"
        )
        images.append(image)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_yticks(range(len(CANONICAL_SURFACES)))
        ax.set_yticklabels(list(reversed(CANONICAL_SURFACES)))
        ax.set_xlabel("round")
        ax.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
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


def plot_surface_activity(out_dir: Path, output_path: Path) -> None:
    activity_rows = _read_activity_csv(out_dir / SURFACE_ACTIVITY_FILENAME)
    unattributed_rows = _read_unattributed_csv(out_dir / UNATTRIBUTED_FILENAME)
    rounds = sorted({row["round_index"] for row in activity_rows})

    fig = plt.figure(figsize=(9, 9.5), dpi=DPI)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.15)
    ax_top = fig.add_subplot(grid_spec[0, :])
    ax_bottom_left = fig.add_subplot(grid_spec[1, 0])
    ax_bottom_right = fig.add_subplot(grid_spec[1, 1], sharey=ax_bottom_left)
    plt.setp(ax_bottom_right.get_yticklabels(), visible=False)
    ax_bottom_right.set_ylabel("")

    _plot_cumulative_panel(ax_top, rounds, activity_rows)
    _plot_heatmaps(fig, (ax_bottom_left, ax_bottom_right), rounds, activity_rows)

    flagged = [
        row
        for row in unattributed_rows
        if row["total_rows"] and row["unattributed_count"] / row["total_rows"] > UNATTRIBUTED_WARN_FRACTION
    ]
    total_unattributed = sum(row["unattributed_count"] for row in unattributed_rows)
    if total_unattributed:
        note = (
            f"Note: {total_unattributed} ledger row(s) across {len(unattributed_rows)} round(s) "
            "excluded from surface counts (no single surface: loader rejections, merged-harness "
            f"records) -- see {UNATTRIBUTED_FILENAME}."
        )
        if flagged:
            note += f" Rounds {', '.join(str(r['round_index']) for r in flagged)} exceed 25% unattributed."
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.5, color=MUTED_INK, wrap=True)

    fig.suptitle("Surface activity over optimization rounds", fontsize=14, color=PRIMARY_INK, x=0.02, ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.plot_surface_activity",
        description="Render Graph 1 (surface activity over rounds) from surface_activity.csv.",
    )
    parser.add_argument("out_dir", help="the experiment directory holding surface_activity.csv")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    activity_path = out_dir / SURFACE_ACTIVITY_FILENAME
    if not activity_path.exists():
        sys.stderr.write(
            f"{activity_path} not found -- run `python -m shrlm.experiment.surface_activity "
            f"{out_dir}` first\n"
        )
        return 1

    output_path = out_dir / PLOTS_DIR / OUTPUT_FILENAME
    plot_surface_activity(out_dir, output_path)
    sys.stdout.write(f"Wrote {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["OUTPUT_FILENAME", "PLOTS_DIR", "main", "plot_surface_activity"]
