"""What both static figures share: palette, rcParams, and snapshot plumbing.

Two plot modules, one look, one way in and one way out. Before this module
each of them carried its own copy of the palette hex values and its own
``plt.rcParams`` block, so a palette change had to be made twice and a figure
could silently drift from its sibling (R7, KTD8). It also owns the parts of
the plot CLIs that are shared *because of where the data now lives* rather
than because of how it looks:

Reading a snapshot, not a directory
    Aggregations write into ``<out_dir>/analysis/<UTC-stamp>/`` (KTD2), so a
    plot CLI takes the experiment directory and resolves a snapshot inside it
    -- the latest published one by default, or the one ``--snapshot`` pins.
    Unpublished directories are never the default choice: a batch that crashed
    halfway is exactly the data a figure must not present as finished.

Saying when it was drawn
    ``plots/`` is the one deliberately re-renderable part of a published
    snapshot, so an improved plotting run may replace a figure over frozen
    data. That is only honest if the figure says when it was drawn, which is
    what the provenance footer carries alongside the experiment identity and
    the snapshot stamp: identity says which experiment, the stamp says which
    frozen numbers, the render time says which plotting run.

Marking what is not confirmed complete
    A round's ``round_complete`` / ``runs_complete`` cells are KTD9 tristates:
    ``true``, ``false``, or ``unknown``. Only ``true`` is complete -- but
    ``unknown`` (the experiment config was unavailable, so the expected run
    count could not be computed) is not a *failure*, and the caption wording
    here is deliberately "not confirmed complete" rather than "incomplete" so
    an unmeasurable round is never presented as a failed one. The marker is
    the same for both, because a reader's question at the figure is the same
    for both: can I trust this point as a finished round?
"""

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrlm.experiment.analysis_io import (
    PROVENANCE_FILENAME,
    SNAPSHOT_STAMP_FORMAT,
    TRISTATE_TRUE,
    is_published,
    latest_snapshot,
    snapshot_parent,
)
from shrlm.experiment.errors import ExperimentError

# shrlm's dataviz skill reference palette (references/palette.md), light mode.
# These are static, paper-bound PNGs, so there is one (light) surface to render
# against and no themed variant to select.
BLUE = "#2a78d6"  # categorical slot 1
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
PRIMARY_INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

DPI = 200

# How a round that is not confirmed complete is drawn: hollow where the figure
# marks points, hatched where it marks a whole round band. Both read as "this
# one is different" in grayscale and in print, which color alone would not.
PARTIAL_HATCH = "///"
PARTIAL_EDGE_WIDTH = 1.4

# Artist ids the completeness marks carry, so a reader of the SVG -- or a test
# -- can find them without guessing at draw order.
PARTIAL_GID_PREFIX = "partial-"


class PlotInputError(ExperimentError):
    """A figure's input snapshot is missing, unreadable, or has nothing to plot.

    Its own class, per the package's convention, because the recovery is
    always the same and always the caller's: run the aggregation that writes
    the missing table, or point at a snapshot that has it. The plot CLIs turn
    it into a one-line stderr message and a non-zero exit rather than a
    traceback -- a missing companion CSV is a user-facing mistake, not a bug.
    """


def apply_style() -> None:
    """Install the repository's one figure style (R7).

    Called at render time rather than at import: an import-time rcParams block
    mutates global state for anything that merely imports the module, and the
    two plot modules already import each other's neighbours.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.color"] = PRIMARY_INK
    plt.rcParams["axes.edgecolor"] = BASELINE
    plt.rcParams["axes.labelcolor"] = SECONDARY_INK
    plt.rcParams["xtick.color"] = MUTED_INK
    plt.rcParams["ytick.color"] = MUTED_INK
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["axes.facecolor"] = SURFACE
    plt.rcParams["savefig.facecolor"] = SURFACE


# ---------------------------------------------------------------------------
# Snapshot resolution
# ---------------------------------------------------------------------------


def resolve_snapshot(out_dir: Path | str, pinned: str | None = None) -> Path:
    """The snapshot directory a plot run reads (KTD2).

    Args:
        out_dir: The experiment directory (the one holding ``analysis/``).
        pinned: ``--snapshot``: either a stamp name inside ``analysis/`` or a
            path to a snapshot directory anywhere. Both spellings are accepted
            because both are things a person legitimately has in hand -- the
            name printed by an aggregation run, or the path they just ``ls``-ed.

    Returns:
        The snapshot directory to read.

    Raises:
        PlotInputError: The pinned snapshot does not exist, or nothing has
            been published yet.
    """
    out_dir = Path(out_dir)
    if pinned is not None:
        candidates = [Path(pinned), snapshot_parent(out_dir) / pinned]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise PlotInputError(
            f"snapshot {pinned!r} not found (looked for "
            f"{', '.join(str(candidate) for candidate in candidates)})"
        )
    latest = latest_snapshot(out_dir)
    if latest is None:
        raise PlotInputError(
            f"no published analysis snapshot under {snapshot_parent(out_dir)} -- run the "
            f"aggregation for this figure against {out_dir} first"
        )
    return latest


def add_snapshot_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``out_dir`` positional and ``--snapshot`` every plot CLI takes.

    Defined beside ``resolve_snapshot``, which is what consumes both: the pair
    of arguments and the resolution rule are one contract, and a CLI that
    described ``--snapshot`` differently from the function that resolves it
    would be documenting a behavior nothing implements.
    """
    parser.add_argument("out_dir", help="the experiment directory holding analysis/")
    parser.add_argument(
        "--snapshot",
        default=None,
        metavar="STAMP_OR_PATH",
        help="the analysis snapshot to read (default: the latest published one)",
    )


def read_rows(snapshot_dir: Path, filename: str) -> list[dict[str, Any]]:
    """Read one aggregation CSV out of a snapshot, or say plainly what is missing.

    Every companion table goes through here, so a figure whose second input is
    absent fails with the same one-line message as one whose first input is --
    the residual this closes was a plot CLI that checked one CSV and then let
    the other raise a bare ``FileNotFoundError``.

    Cells arrive as strings and each caller coerces the columns it plots in
    place (the value type is therefore ``Any``, not ``str``): every reader here
    wants a different subset as numbers, and a per-table row dataclass would
    duplicate the aggregations' own schemas one module further downstream.
    """
    path = Path(snapshot_dir) / filename
    if not path.is_file():
        raise PlotInputError(
            f"{path} not found -- this snapshot does not carry {filename}; re-run the "
            "aggregation that writes it, or pin a snapshot that has it"
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Completeness (KTD9)
# ---------------------------------------------------------------------------


def is_complete(value: str | None) -> bool:
    """Whether one KTD9 tristate cell reads as confirmed complete.

    Anything that is not the literal ``true`` -- ``false``, ``unknown``, or a
    column an older CSV never wrote -- is not confirmed complete. Erring this
    way is the point: an unmarked partial round would be read as a finished
    one, while a conservatively marked complete round only costs a footnote.
    """
    return value == TRISTATE_TRUE


def partial_rounds(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Round indices any of whose rows is not confirmed complete."""
    partial: set[int] = set()
    for row in rows:
        if not (is_complete(row.get("round_complete")) and is_complete(row.get("runs_complete"))):
            partial.add(int(row["round_index"]))
    return partial


def partial_caption(
    partial: Sequence[int] | set[int], n_rounds: int, *, marker: str, source: str
) -> str | None:
    """The caption line naming how many plotted rounds are not confirmed complete.

    ``None`` when every plotted round is confirmed complete -- a caption
    saying "0 partial" would be noise on the common case, and the reader who
    needs the count is the one looking at a marked point.
    """
    if not partial:
        return None
    ordered = ", ".join(str(index) for index in sorted(partial))
    return (
        f"{marker}: {len(partial)} of {n_rounds} plotted round(s) not confirmed complete "
        f"(round {ordered}) -- the round marker is missing, or the run count is short or "
        f"unknown (the experiment config was unavailable). See round_complete / "
        f"runs_complete in {source}."
    )


# ---------------------------------------------------------------------------
# The provenance footer
# ---------------------------------------------------------------------------


def snapshot_identity(snapshot_dir: Path | str) -> str | None:
    """The experiment identity the snapshot's provenance records, or None."""
    path = Path(snapshot_dir) / PROVENANCE_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    identity = payload.get("identity_hash") if isinstance(payload, dict) else None
    return None if identity is None else str(identity)


def footer_text(snapshot_dir: Path | str, *, rendered_at: datetime | None = None) -> str:
    """Identity, snapshot stamp, and render time, in that order.

    All three, because each answers a question the other two cannot: which
    experiment, which frozen numbers, and -- since ``plots/`` is re-renderable
    over an immutable snapshot -- which plotting run drew this PNG.
    """
    snapshot_dir = Path(snapshot_dir)
    identity = snapshot_identity(snapshot_dir)
    short = identity[:12] if identity else "identity unknown"
    unpublished = "" if is_published(snapshot_dir) else " | UNPUBLISHED SNAPSHOT"
    stamped = (rendered_at or datetime.now(UTC)).strftime(SNAPSHOT_STAMP_FORMAT)
    return f"experiment {short} | snapshot {snapshot_dir.name} | rendered {stamped}{unpublished}"


def draw_footer(
    fig: "plt.Figure", snapshot_dir: Path | str, *, rendered_at: datetime | None = None
) -> None:
    """Put the provenance footer in the figure's bottom-right corner."""
    fig.text(
        0.98,
        0.005,
        footer_text(snapshot_dir, rendered_at=rendered_at),
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MUTED_INK,
    )


def save_figure(fig: "plt.Figure", output_path: Path) -> Path:
    """Write one figure into a snapshot's re-renderable ``plots/`` directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    return output_path


__all__ = [
    "BASELINE",
    "BLUE",
    "DPI",
    "GRIDLINE",
    "MUTED_INK",
    "PARTIAL_EDGE_WIDTH",
    "PARTIAL_GID_PREFIX",
    "PARTIAL_HATCH",
    "PRIMARY_INK",
    "SECONDARY_INK",
    "SURFACE",
    "PlotInputError",
    "add_snapshot_arguments",
    "apply_style",
    "draw_footer",
    "footer_text",
    "is_complete",
    "partial_caption",
    "partial_rounds",
    "read_rows",
    "resolve_snapshot",
    "save_figure",
    "snapshot_identity",
]
