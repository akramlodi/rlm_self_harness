"""Static figure for Graph 2: incumbent quality over time.

Reads ``incumbent_quality.csv`` (``incumbent_quality.py``'s per-round table)
and renders a step-function figure with held-in and held-out pass rate as two
series, a marker at every round the incumbent actually changed, and a tick
below the axis at every round carrying a structural-rejection annotation --
keyed to a numbered footnote list under the figure, since a static PNG has no
hover to carry that text.

``--show-candidates`` additionally reads ``incumbent_quality_candidates.csv``
and draws every scored candidate's own pass rate as a faint point behind the
incumbent lines -- off by default because it gets busy fast on a run with many
proposals per round.

Palette: the repo's dataviz skill reference palette (``references/palette.md``)
-- held-in and held-out are two *independent* identities (not one metric
viewed two ways, unlike Graph 1's attempted/promoted), so they take the first
two categorical slots (blue, orange), validated for this pairing via
``scripts/validate_palette.js``. Distinct markers (circle / square) carry the
same distinction redundantly, per the skill's "identity is never color alone"
rule. Static, paper-bound PNG: renders once against the palette's light chart
surface.
"""

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    INCUMBENT_QUALITY_FILENAME,
)

PLOTS_DIR = "plots"
OUTPUT_FILENAME = "incumbent_quality.png"

# shrlm's dataviz skill reference palette (references/palette.md), light mode.
BLUE = "#2a78d6"      # categorical slot 1 -- held-in
ORANGE = "#eb6834"    # categorical slot 2 -- held-out
CHANGED_RING = "#0b0b0b"
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
PRIMARY_INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

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


def _to_float_or_nan(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def _read_quality_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["heldin_pass_rate"] = _to_float_or_nan(row["heldin_pass_rate"])
        row["heldout_pass_rate"] = _to_float_or_nan(row["heldout_pass_rate"])
        row["incumbent_changed"] = row["incumbent_changed"] == "True"
        row["annotation"] = row["annotation"] or None
    return rows


def _read_candidates_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_index"] = int(row["round_index"])
        row["heldin_pass_rate"] = _to_float_or_nan(row["heldin_pass_rate"])
        row["heldout_pass_rate"] = _to_float_or_nan(row["heldout_pass_rate"])
    return rows


def _plot_candidate_scatter(ax: "plt.Axes", candidate_rows: list[dict]) -> None:
    heldin_x = [row["round_index"] for row in candidate_rows if not math.isnan(row["heldin_pass_rate"])]
    heldin_y = [row["heldin_pass_rate"] for row in candidate_rows if not math.isnan(row["heldin_pass_rate"])]
    heldout_x = [row["round_index"] for row in candidate_rows if not math.isnan(row["heldout_pass_rate"])]
    heldout_y = [
        row["heldout_pass_rate"] for row in candidate_rows if not math.isnan(row["heldout_pass_rate"])
    ]
    ax.scatter(heldin_x, heldin_y, s=18, color=BLUE, alpha=0.18, zorder=1, linewidths=0)
    ax.scatter(heldout_x, heldout_y, s=18, color=ORANGE, alpha=0.18, zorder=1, linewidths=0)


def _plot_series(
    ax: "plt.Axes", rounds: list[int], values: list[float], color: str, marker: str, label: str
) -> None:
    ax.step(rounds, values, where="post", color=color, linewidth=2, zorder=3, label=label)
    ax.scatter(rounds, values, s=36, color=color, marker=marker, zorder=4)
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
    for split_key, color in (("heldin_pass_rate", BLUE), ("heldout_pass_rate", ORANGE)):
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


def plot_incumbent_quality(out_dir: Path, output_path: Path, *, show_candidates: bool) -> None:
    rows = _read_quality_csv(out_dir / INCUMBENT_QUALITY_FILENAME)
    rounds = [row["round_index"] for row in rows]
    heldin = [row["heldin_pass_rate"] for row in rows]
    heldout = [row["heldout_pass_rate"] for row in rows]

    footnote_height = 0.22 if any(row["annotation"] for row in rows) else 0.02
    fig, ax = plt.subplots(figsize=(8.5, 5.5 + footnote_height * 5.5), dpi=DPI)
    fig.subplots_adjust(bottom=0.18 + footnote_height)

    if show_candidates:
        candidates_path = out_dir / INCUMBENT_QUALITY_CANDIDATES_FILENAME
        if candidates_path.exists():
            _plot_candidate_scatter(ax, _read_candidates_csv(candidates_path))
        else:
            sys.stderr.write(f"warning: --show-candidates requested but {candidates_path} not found\n")

    _plot_series(ax, rounds, heldin, BLUE, "o", "held-in pass rate")
    _plot_series(ax, rounds, heldout, ORANGE, "s", "held-out pass rate")
    _mark_incumbent_changes(ax, rows)
    footnotes = _mark_annotations(ax, rows)

    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(rounds)
    ax.set_xlabel("round")
    ax.set_ylabel("pass rate")
    ax.set_title("Incumbent quality over optimization rounds", color=PRIMARY_INK, fontsize=12, loc="left")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="lower right", labelcolor=SECONDARY_INK)

    if footnotes:
        text = "  ".join(
            f"{number}. round {round_index}: {annotation}" for number, round_index, annotation in footnotes
        )
        fig.text(0.02, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED_INK, wrap=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shrlm.experiment.plot_incumbent_quality",
        description="Render Graph 2 (incumbent quality over rounds) from incumbent_quality.csv.",
    )
    parser.add_argument("out_dir", help="the experiment directory holding incumbent_quality.csv")
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="overlay every scored candidate's own pass rate as faint background points",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    quality_path = out_dir / INCUMBENT_QUALITY_FILENAME
    if not quality_path.exists():
        sys.stderr.write(
            f"{quality_path} not found -- run `python -m shrlm.experiment.incumbent_quality "
            f"{out_dir}` first\n"
        )
        return 1

    output_path = out_dir / PLOTS_DIR / OUTPUT_FILENAME
    plot_incumbent_quality(out_dir, output_path, show_candidates=args.show_candidates)
    sys.stdout.write(f"Wrote {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["OUTPUT_FILENAME", "PLOTS_DIR", "main", "plot_incumbent_quality"]
