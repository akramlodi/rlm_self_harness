"""Static figure for Graph 2: incumbent quality over time.

Reads ``incumbent_quality.csv`` (``incumbent_quality.py``'s per-round table)
out of an analysis snapshot and renders a step-function figure into that
snapshot's ``plots/`` directory: held-in and held-out pass rate as two series,
a marker at every round the incumbent actually changed, and a tick below the
axis at every round carrying a structural-rejection annotation -- keyed to a
numbered footnote list under the figure, since a static PNG has no hover to
carry that text.

Rounds the aggregation could not confirm complete (``round_complete`` or
``runs_complete`` not ``true``) are drawn hollow instead of filled, and
counted in a caption: the PNG is what a reader draws conclusions from, so a
pass rate computed over a round that may be missing runs has to *look*
different from one computed over a finished round.

``--show-candidates`` additionally reads ``incumbent_quality_candidates.csv``
and draws every scored candidate's own pass rate as a faint point behind the
incumbent lines -- off by default because it gets busy fast on a run with many
proposals per round.

Palette: the repo's dataviz skill reference palette, shared through
``plot_style`` -- held-in and held-out are two *independent* identities (not
one metric viewed two ways, unlike Graph 1's attempted/promoted), so they take
the first two categorical slots (blue, orange), validated for this pairing via
``scripts/validate_palette.js``. Distinct markers (circle / square) carry the
same distinction redundantly, per the skill's "identity is never color alone"
rule. Static, paper-bound PNG: renders once against the palette's light chart
surface.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrlm.experiment.analysis_io import PLOTS_DIR
from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    INCUMBENT_QUALITY_FILENAME,
)
from shrlm.experiment.plot_style import (
    BLUE,
    DPI,
    GRIDLINE,
    MUTED_INK,
    PARTIAL_EDGE_WIDTH,
    PARTIAL_GID_PREFIX,
    PRIMARY_INK,
    SECONDARY_INK,
    PlotInputError,
    add_snapshot_arguments,
    apply_style,
    draw_footer,
    partial_caption,
    partial_rounds,
    read_rows,
    resolve_snapshot,
    save_figure,
)

OUTPUT_FILENAME = "incumbent_quality.png"

# Single-consumer palette pieces stay local (the shared ones live in
# ``plot_style``): orange is Graph 2's second categorical identity, and the
# incumbent-change ring is the shared primary ink used as an outline.
ORANGE = "#eb6834"  # categorical slot 2 -- held-out
CHANGED_RING = PRIMARY_INK


def _to_float_or_nan(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def _read_quality_csv(snapshot_dir: Path) -> list[dict]:
    rows = read_rows(snapshot_dir, INCUMBENT_QUALITY_FILENAME)
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["heldin_pass_rate"] = _to_float_or_nan(row["heldin_pass_rate"])
        row["heldout_pass_rate"] = _to_float_or_nan(row["heldout_pass_rate"])
        row["incumbent_changed"] = row["incumbent_changed"] == "True"
        row["annotation"] = row["annotation"] or None
    return rows


def _read_candidates_csv(snapshot_dir: Path) -> list[dict]:
    rows = read_rows(snapshot_dir, INCUMBENT_QUALITY_CANDIDATES_FILENAME)
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["heldin_pass_rate"] = _to_float_or_nan(row["heldin_pass_rate"])
        row["heldout_pass_rate"] = _to_float_or_nan(row["heldout_pass_rate"])
    return rows


def _plot_candidate_scatter(ax: "plt.Axes", candidate_rows: list[dict]) -> None:
    heldin_x = [
        row["round_index"] for row in candidate_rows if not math.isnan(row["heldin_pass_rate"])
    ]
    heldin_y = [
        row["heldin_pass_rate"] for row in candidate_rows if not math.isnan(row["heldin_pass_rate"])
    ]
    heldout_x = [
        row["round_index"] for row in candidate_rows if not math.isnan(row["heldout_pass_rate"])
    ]
    heldout_y = [
        row["heldout_pass_rate"]
        for row in candidate_rows
        if not math.isnan(row["heldout_pass_rate"])
    ]
    ax.scatter(heldin_x, heldin_y, s=18, color=BLUE, alpha=0.18, zorder=1, linewidths=0)
    ax.scatter(heldout_x, heldout_y, s=18, color=ORANGE, alpha=0.18, zorder=1, linewidths=0)


def _plot_series(
    ax: "plt.Axes",
    rounds: list[int],
    values: list[float],
    color: str,
    marker: str,
    label: str,
    partial: set[int],
) -> None:
    """One split's step line, with confirmed-complete rounds filled and the rest hollow.

    The line is drawn over every round -- dropping the partial ones would
    silently redraw the history -- and only the point marker distinguishes
    them, so the reader sees the series as it is and can see which points are
    provisional.
    """
    ax.step(rounds, values, where="post", color=color, linewidth=2, zorder=3, label=label)
    complete = [(r, v) for r, v in zip(rounds, values, strict=True) if r not in partial]
    if complete:
        ax.scatter(
            [r for r, _ in complete],
            [v for _, v in complete],
            s=36,
            color=color,
            marker=marker,
            zorder=4,
        )
    incomplete = [(r, v) for r, v in zip(rounds, values, strict=True) if r in partial]
    if incomplete:
        ax.scatter(
            [r for r, _ in incomplete],
            [v for _, v in incomplete],
            s=44,
            facecolors="none",
            edgecolors=color,
            linewidths=PARTIAL_EDGE_WIDTH,
            marker=marker,
            zorder=4,
            gid=f"{PARTIAL_GID_PREFIX}{label}",
        )
    finite = [(r, v) for r, v in zip(rounds, values, strict=True) if not math.isnan(v)]
    if finite:
        last_round, last_value = finite[-1]
        ax.annotate(
            f"{last_value:.0%}",
            (last_round, last_value),
            textcoords="offset points",
            xytext=(6, 4 if color == BLUE else -12),
            color=SECONDARY_INK,
            fontsize=9,
        )


def _mark_incumbent_changes(ax: "plt.Axes", rows: list[dict]) -> None:
    changed = [row for row in rows if row["incumbent_changed"]]
    for split_key in ("heldin_pass_rate", "heldout_pass_rate"):
        xs = [row["round_index"] for row in changed if not math.isnan(row[split_key])]
        ys = [row[split_key] for row in changed if not math.isnan(row[split_key])]
        ax.scatter(
            xs,
            ys,
            s=140,
            facecolors="none",
            edgecolors=CHANGED_RING,
            linewidths=1.5,
            zorder=5,
            label="_nolegend_",
        )


def _mark_annotations(ax: "plt.Axes", rows: list[dict]) -> list[tuple[int, int, str]]:
    """Tick a small triangle below the axis at every annotated round.

    Returns ``(footnote_number, round_index, annotation_text)`` in plot order,
    for the numbered footnote list under the figure.
    """
    annotated = [row for row in rows if row["annotation"]]
    if not annotated:
        return []
    xs = [row["round_index"] for row in annotated]
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    ax.scatter(
        xs,
        [-0.04] * len(xs),
        transform=trans,
        marker="^",
        s=50,
        color=MUTED_INK,
        clip_on=False,
        zorder=4,
    )
    footnotes = []
    for number, row in enumerate(annotated, start=1):
        ax.annotate(
            str(number),
            (row["round_index"], -0.075),
            xycoords=trans,
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED_INK,
        )
        footnotes.append((number, row["round_index"], row["annotation"]))
    return footnotes


def build_figure(
    snapshot_dir: Path | str,
    *,
    show_candidates: bool = False,
    rendered_at: datetime | None = None,
) -> "plt.Figure":
    """Render the figure for one snapshot, without writing it anywhere.

    Separate from ``plot_incumbent_quality`` so the figure's structure -- which
    rounds carry the hollow partial marker, what the captions say -- is
    assertable without going through a PNG.

    Raises:
        PlotInputError: The quality table is missing, the requested candidates
            table is missing, or the snapshot holds no round to plot.
    """
    snapshot_dir = Path(snapshot_dir)
    rows = _read_quality_csv(snapshot_dir)
    if not rows:
        raise PlotInputError(
            f"{snapshot_dir / INCUMBENT_QUALITY_FILENAME} holds no rounds -- nothing to plot"
        )
    rounds = [row["round_index"] for row in rows]
    heldin = [row["heldin_pass_rate"] for row in rows]
    heldout = [row["heldout_pass_rate"] for row in rows]
    partial = partial_rounds(rows)

    apply_style()
    footnote_height = 0.22 if any(row["annotation"] for row in rows) else 0.02
    fig, ax = plt.subplots(figsize=(8.5, 5.5 + footnote_height * 5.5), dpi=DPI)
    fig.subplots_adjust(bottom=0.18 + footnote_height)

    if show_candidates:
        _plot_candidate_scatter(ax, _read_candidates_csv(snapshot_dir))

    _plot_series(ax, rounds, heldin, BLUE, "o", "held-in pass rate", partial)
    _plot_series(ax, rounds, heldout, ORANGE, "s", "held-out pass rate", partial)
    _mark_incumbent_changes(ax, rows)
    footnotes = _mark_annotations(ax, rows)

    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"{index}*" if index in partial else str(index) for index in rounds])
    ax.set_xlabel("round")
    ax.set_ylabel("pass rate")
    ax.set_title(
        "Incumbent quality over optimization rounds", color=PRIMARY_INK, fontsize=12, loc="left"
    )
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="lower right", labelcolor=SECONDARY_INK)

    captions = [
        partial_caption(
            partial,
            len(rounds),
            marker="Hollow marker / starred round label",
            source=INCUMBENT_QUALITY_FILENAME,
        )
    ]
    if footnotes:
        captions.append(
            "  ".join(
                f"{number}. round {round_index}: {annotation}"
                for number, round_index, annotation in footnotes
            )
        )
    text = "\n".join(caption for caption in captions if caption)
    if text:
        fig.text(
            0.02, 0.045, text, ha="left", va="bottom", fontsize=7.5, color=MUTED_INK, wrap=True
        )

    draw_footer(fig, snapshot_dir, rendered_at=rendered_at)
    return fig


def plot_incumbent_quality(
    snapshot_dir: Path | str,
    output_path: Path | str | None = None,
    *,
    show_candidates: bool = False,
    rendered_at: datetime | None = None,
) -> Path:
    """Render Graph 2 into the snapshot's re-renderable ``plots/`` directory.

    Writes ``<snapshot>/plots/incumbent_quality.png`` by default. ``plots/`` is
    the one part of a published snapshot that may be replaced (KTD2): the
    frozen CSVs underneath never move, and the footer's render time is what
    keeps a re-rendered PNG from misrepresenting when it was drawn.
    """
    snapshot_dir = Path(snapshot_dir)
    path = (
        Path(output_path) if output_path is not None else snapshot_dir / PLOTS_DIR / OUTPUT_FILENAME
    )
    fig = build_figure(snapshot_dir, show_candidates=show_candidates, rendered_at=rendered_at)
    try:
        return save_figure(fig, path)
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.plot_incumbent_quality",
        description=(
            "Render Graph 2 (incumbent quality over rounds) from an analysis snapshot's "
            "incumbent_quality.csv."
        ),
    )
    add_snapshot_arguments(parser)
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="overlay every scored candidate's own pass rate as faint background points",
    )
    args = parser.parse_args(argv)

    try:
        snapshot_dir = resolve_snapshot(Path(args.out_dir), args.snapshot)
        output_path = plot_incumbent_quality(snapshot_dir, show_candidates=args.show_candidates)
    except PlotInputError as error:
        sys.stderr.write(f"{error}\n")
        return 1

    sys.stdout.write(f"Wrote {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["OUTPUT_FILENAME", "PLOTS_DIR", "build_figure", "main", "plot_incumbent_quality"]
