"""Tests for U7's snapshot-reading, completeness-marking plot layer.

The figures are what a reader draws conclusions from, so these tests assert on
the things a reader actually sees rather than on pixels: which rounds carry the
"not confirmed complete" marker, what the captions say, what the provenance
footer carries, and that a re-render never disturbs the frozen aggregation
artifacts underneath it (KTD2).

Fixtures are built by writing the aggregation CSVs into a real snapshot
allocated and published through ``analysis_io``, using each aggregation's own
declared fieldnames -- so a column rename in ``surface_activity`` or
``incumbent_quality`` moves these tests too, instead of leaving them asserting
against a hand-copied header that has quietly gone stale.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="rendering needs the optional `plotting` extra (matplotlib)"
)

import shrlm.experiment.plot_incumbent_quality as plot_iq  # noqa: E402
import shrlm.experiment.plot_surface_activity as plot_sa  # noqa: E402
from shrlm.experiment import plot_style  # noqa: E402
from shrlm.experiment.analysis_io import (  # noqa: E402 -- after the extra guard
    PLOTS_DIR,
    PROVENANCE_FILENAME,
    PUBLISHED_FILENAME,
    Snapshot,
    allocate_snapshot,
)
from shrlm.experiment.incumbent_quality import (  # noqa: E402
    INCUMBENT_QUALITY_CANDIDATES_FIELDNAMES,
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    INCUMBENT_QUALITY_FIELDNAMES,
    INCUMBENT_QUALITY_FILENAME,
)
from shrlm.experiment.rounds import (  # noqa: E402
    SURFACE_SOURCE_NONE,
    SURFACE_SOURCE_UNDECLARED,
    SURFACE_SOURCE_UNKNOWN,
)
from shrlm.experiment.surface_activity import (  # noqa: E402
    CANONICAL_SURFACES,
    SURFACE_ACTIVITY_FIELDNAMES,
    SURFACE_ACTIVITY_FILENAME,
    UNATTRIBUTED_FIELDNAMES,
    UNATTRIBUTED_FILENAME,
)

# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _publish(snapshot: Snapshot) -> Path:
    snapshot.record_tool("fixture", ok=True)
    snapshot.publish()
    return snapshot.path


def _activity_row(
    round_index: int,
    surface: str = "S1",
    *,
    source: str = "ledger",
    attempted: int = 1,
    promoted: int = 1,
    cumulative_attempted: int = 1,
    cumulative_promoted: int = 1,
    round_complete: str = "true",
    runs_complete: str = "true",
) -> dict[str, object]:
    return {
        "round_index": round_index,
        "surface": surface,
        "surface_source": source,
        "attempted_count": attempted,
        "promoted_count": promoted,
        "cumulative_surfaces_attempted": cumulative_attempted,
        "cumulative_surfaces_promoted": cumulative_promoted,
        "round_complete": round_complete,
        "runs_complete": runs_complete,
    }


def _quality_row(
    round_index: int,
    *,
    heldin: float = 0.5,
    heldout: float = 0.4,
    changed: bool = False,
    annotation: str = "",
    round_complete: str = "true",
    runs_complete: str = "true",
) -> dict[str, object]:
    return {
        "round_index": round_index,
        "heldin_pass_rate": heldin,
        "heldout_pass_rate": heldout,
        "incumbent_changed": changed,
        "annotation": annotation,
        "round_complete": round_complete,
        "runs_complete": runs_complete,
    }


def _surface_snapshot(
    out_dir: Path,
    activity_rows: Sequence[Mapping[str, object]],
    unattributed_rows: Sequence[Mapping[str, object]] | None = None,
    *,
    publish: bool = True,
    write_unattributed: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = allocate_snapshot(out_dir)
    _write_csv(
        snapshot.path / SURFACE_ACTIVITY_FILENAME, SURFACE_ACTIVITY_FIELDNAMES, activity_rows
    )
    if write_unattributed:
        rounds = sorted({int(row["round_index"]) for row in activity_rows})
        rows = unattributed_rows
        if rows is None:
            rows = [
                {"round_index": index, "unattributed_count": 0, "total_rows": 4} for index in rounds
            ]
        _write_csv(snapshot.path / UNATTRIBUTED_FILENAME, UNATTRIBUTED_FIELDNAMES, rows)
    if publish:
        return _publish(snapshot)
    return snapshot.path


def _quality_snapshot(
    out_dir: Path,
    quality_rows: Sequence[Mapping[str, object]],
    *,
    publish: bool = True,
    write_candidates: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = allocate_snapshot(out_dir)
    _write_csv(
        snapshot.path / INCUMBENT_QUALITY_FILENAME, INCUMBENT_QUALITY_FIELDNAMES, quality_rows
    )
    if write_candidates:
        _write_csv(
            snapshot.path / INCUMBENT_QUALITY_CANDIDATES_FILENAME,
            INCUMBENT_QUALITY_CANDIDATES_FIELDNAMES,
            [],
        )
    if publish:
        return _publish(snapshot)
    return snapshot.path


def _figure_text(fig) -> str:
    return "\n".join(text.get_text() for text in fig.texts)


def _artifact_bytes(snapshot_dir: Path) -> dict[str, bytes]:
    """Every file in the snapshot except the deliberately re-renderable plots."""
    return {
        str(path.relative_to(snapshot_dir)): path.read_bytes()
        for path in sorted(snapshot_dir.rglob("*"))
        if path.is_file() and PLOTS_DIR not in path.relative_to(snapshot_dir).parts
    }


# ---------------------------------------------------------------------------
# Completeness reaches the figure
# ---------------------------------------------------------------------------


def test_incumbent_partial_round_marked_and_counted_in_caption(tmp_path: Path) -> None:
    snapshot_dir = _quality_snapshot(
        tmp_path / "exp",
        [
            _quality_row(1),
            _quality_row(2, round_complete="false"),
            _quality_row(3),
        ],
    )

    fig = plot_iq.build_figure(snapshot_dir, show_candidates=False)
    try:
        partial = [
            collection
            for collection in fig.axes[0].collections
            if (collection.get_gid() or "").startswith(plot_style.PARTIAL_GID_PREFIX)
        ]
        assert partial, "no partial-marker collection was drawn"
        plotted = {
            round(offset[0]) for collection in partial for offset in collection.get_offsets()
        }
        assert plotted == {2}
        for collection in partial:
            # Hollow: no face, edge only -- the distinct marker U7 asks for.
            assert len(collection.get_facecolors()) == 0
        caption = _figure_text(fig)
        assert "1 of 3" in caption
        assert "not confirmed complete" in caption
    finally:
        plot_iq.plt.close(fig)


def test_incumbent_unknown_runs_complete_marks_partial_without_claiming_incomplete(
    tmp_path: Path,
) -> None:
    snapshot_dir = _quality_snapshot(
        tmp_path / "exp",
        [_quality_row(1), _quality_row(2, runs_complete="unknown")],
    )

    fig = plot_iq.build_figure(snapshot_dir, show_candidates=False)
    try:
        partial = [
            collection
            for collection in fig.axes[0].collections
            if (collection.get_gid() or "").startswith(plot_style.PARTIAL_GID_PREFIX)
        ]
        plotted = {
            round(offset[0]) for collection in partial for offset in collection.get_offsets()
        }
        assert plotted == {2}, "an unknown completeness flag is not `true`, so it marks as partial"
        caption = _figure_text(fig)
        assert "1 of 2" in caption
        # KTD9: unknown never collapses into false, so the caption may not
        # assert the round is known-incomplete.
        assert "unknown" in caption
        assert "known to be incomplete" not in caption
    finally:
        plot_iq.plt.close(fig)


def test_surface_partial_round_marked_and_counted_in_caption(tmp_path: Path) -> None:
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1),
            _activity_row(2, "S2", runs_complete="false"),
            _activity_row(3, "S3"),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        marks = [
            patch
            for ax in fig.axes
            for patch in ax.patches
            if (patch.get_gid() or "").startswith(plot_style.PARTIAL_GID_PREFIX)
        ]
        assert marks, "no partial-round mark was drawn"
        assert all(patch.get_hatch() for patch in marks)
        assert {patch.get_gid() for patch in marks} == {f"{plot_style.PARTIAL_GID_PREFIX}2"}
        caption = _figure_text(fig)
        assert "1 of 3" in caption
        assert "not confirmed complete" in caption
    finally:
        plot_sa.plt.close(fig)


def test_surface_merged_row_stays_inert_in_the_heatmaps(tmp_path: Path) -> None:
    """U4's extra ``merged`` category row must not enter the canonical-surface heatmaps."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S1", attempted=2, promoted=1),
            _activity_row(1, "merged", attempted=99, promoted=99),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        images = [image for ax in fig.axes for image in ax.images]
        assert images
        assert all(image.get_clim()[1] == 2 for image in images), (
            "the merged row's counts leaked into the heatmap color scale"
        )
        for ax in _heatmap_axes(fig):
            assert ax.images[0].get_array().shape[0] == len(CANONICAL_SURFACES)
            assert "merged" not in [label.get_text() for label in ax.get_yticklabels()]
    finally:
        plot_sa.plt.close(fig)


# ---------------------------------------------------------------------------
# Undeclared, unknown, and zero-activity-declared cells render differently (R13)
# ---------------------------------------------------------------------------


def _heatmap_axes(fig) -> list:
    """Panel B's two heatmap axes: the ones carrying an image."""
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 2, "expected exactly two heatmap axes"
    return axes


def _state_patches(ax, prefix: str) -> list:
    return [patch for patch in ax.patches if (patch.get_gid() or "").startswith(prefix)]


def test_surface_heatmap_renders_the_three_cell_states_distinctly(tmp_path: Path) -> None:
    """Undeclared, unknown, and declared-but-untouched are three different marks.

    Round 1 never declared S10 (a pre-S10 harness), round 2's harness could
    not be read, and round 3 declared S10 and left it alone. All three S10
    cells hold a zero count, so the count alone cannot tell them apart -- the
    mark has to.
    """
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S1"),
            _activity_row(1, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0, promoted=0),
            _activity_row(2, "S1", source=SURFACE_SOURCE_UNKNOWN),
            _activity_row(2, "S10", source=SURFACE_SOURCE_UNKNOWN, attempted=0, promoted=0),
            _activity_row(3, "S1"),
            _activity_row(3, "S10", source=SURFACE_SOURCE_NONE, attempted=0, promoted=0),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        for ax in _heatmap_axes(fig):
            undeclared = _state_patches(ax, plot_sa.UNDECLARED_GID_PREFIX)
            unknown = _state_patches(ax, plot_sa.UNKNOWN_GID_PREFIX)
            assert {patch.get_gid() for patch in undeclared} == {
                f"{plot_sa.UNDECLARED_GID_PREFIX}1-S10"
            }
            # Every cell of an unknown round is unknown, S1 included.
            assert {patch.get_gid() for patch in unknown} == {
                f"{plot_sa.UNKNOWN_GID_PREFIX}2-S1",
                f"{plot_sa.UNKNOWN_GID_PREFIX}2-S10",
            }
            # The declared-and-untouched cell carries no mark at all: it is the
            # blank cell the heatmap already draws for a zero count.
            assert not [patch for patch in ax.patches if (patch.get_gid() or "").endswith("3-S10")]
            # And the two marks are not the same mark.
            assert all(patch.get_hatch() == plot_sa.UNDECLARED_HATCH for patch in undeclared)
            assert all(patch.get_hatch() != plot_sa.UNDECLARED_HATCH for patch in unknown)
            assert all(patch.get_linestyle() != undeclared[0].get_linestyle() for patch in unknown)
        caption = _figure_text(fig)
        assert "not declared" in caption
        assert "could not be read" in caption
    finally:
        plot_sa.plt.close(fig)


def test_surface_mixed_vintage_tree_marks_only_the_pre_s10_cells(tmp_path: Path) -> None:
    """A pre-S10 round beside a post-S10 round: one undeclared mark, at (1, S10)."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S2"),
            _activity_row(1, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0, promoted=0),
            _activity_row(2, "S2"),
            _activity_row(2, "S10", attempted=1, promoted=0, cumulative_attempted=2),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        for ax in _heatmap_axes(fig):
            assert {
                patch.get_gid() for patch in _state_patches(ax, plot_sa.UNDECLARED_GID_PREFIX)
            } == {f"{plot_sa.UNDECLARED_GID_PREFIX}1-S10"}
            assert not _state_patches(ax, plot_sa.UNKNOWN_GID_PREFIX)
        caption = _figure_text(fig)
        assert "not declared" in caption
        assert "could not be read" not in caption
    finally:
        plot_sa.plt.close(fig)


def test_a_fully_declared_tree_draws_no_declaration_marks_or_caption(tmp_path: Path) -> None:
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [_activity_row(1, "S1"), _activity_row(1, "S10", source=SURFACE_SOURCE_NONE, attempted=0)],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        for ax in _heatmap_axes(fig):
            assert not _state_patches(ax, plot_sa.UNDECLARED_GID_PREFIX)
            assert not _state_patches(ax, plot_sa.UNKNOWN_GID_PREFIX)
        caption = _figure_text(fig)
        assert "not declared" not in caption
        assert "could not be read" not in caption
    finally:
        plot_sa.plt.close(fig)


def test_declared_surfaces_by_round_is_read_off_the_table() -> None:
    """The per-round declared set a figure may derive totals from (for U8)."""
    rows = [
        _activity_row(1, "S1"),
        _activity_row(1, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0),
        _activity_row(2, "S1", source=SURFACE_SOURCE_UNKNOWN),
        _activity_row(3, "S1"),
        _activity_row(3, "S10", source=SURFACE_SOURCE_NONE, attempted=0),
        _activity_row(3, "merged", source="merged", attempted=0),
    ]

    declared = plot_sa.declared_surfaces_by_round(rows)
    assert declared[1] == frozenset(CANONICAL_SURFACES) - {"S10"}
    assert declared[2] is None
    assert declared[3] == frozenset(CANONICAL_SURFACES)
    assert "merged" not in declared[3]


# ---------------------------------------------------------------------------
# Panel A derives its totals from each round's declared set (R12, KTD5)
# ---------------------------------------------------------------------------


def _panel_a(fig):
    """Panel A: the one axes carrying step lines and no image."""
    axes = [ax for ax in fig.axes if ax.lines and not ax.images]
    assert len(axes) == 1, "expected exactly one cumulative-count axes"
    return axes[0]


def _reference_segments(ax) -> dict[int, list[float]]:
    """``round_index -> y values`` of each per-round declared-total reference segment."""
    out: dict[int, list[float]] = {}
    for line in ax.lines:
        gid = line.get_gid() or ""
        if gid.startswith(plot_sa.REFERENCE_GID_PREFIX):
            out[int(gid[len(plot_sa.REFERENCE_GID_PREFIX) :])] = [
                float(y) for y in line.get_ydata()
            ]
    return out


def _all_surfaces_rows(round_index: int, **overrides) -> list[dict[str, object]]:
    """One declared-and-touched row per canonical surface for one round."""
    total = len(CANONICAL_SURFACES)
    return [
        _activity_row(
            round_index,
            surface,
            cumulative_attempted=total,
            cumulative_promoted=total,
            **overrides,
        )
        for surface in CANONICAL_SURFACES
    ]


def _assert_nothing_clipped(ax) -> None:
    """Every clip-on data artist on ``ax`` sits inside the axes' data limits."""
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    for line in ax.lines:
        if line.get_clip_on():
            assert all(y_low <= y <= y_high for y in line.get_ydata()), line.get_gid()
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if collection.get_clip_on() and len(offsets):
            assert all(y_low <= y <= y_high for _, y in offsets), collection.get_gid()
            assert all(x_low <= x <= x_high for x, _ in offsets), collection.get_gid()
    for image in ax.images:
        left, right, bottom, top = image.get_extent()
        assert min(x_low, x_high) <= min(left, right) and max(left, right) <= max(x_low, x_high)
        assert min(y_low, y_high) <= min(bottom, top) and max(bottom, top) <= max(y_low, y_high)
    for patch in ax.patches:
        if patch.get_clip_on():
            box = patch.get_extents().transformed(ax.transData.inverted())
            assert min(x_low, x_high) - 1e-6 <= box.x0 and box.x1 <= max(x_low, x_high) + 1e-6
            assert min(y_low, y_high) - 1e-6 <= box.y0 and box.y1 <= max(y_low, y_high) + 1e-6


def test_panel_a_ylim_exceeds_the_largest_declared_count(tmp_path: Path) -> None:
    """A round that touched every declared surface plots inside the axes, not on its edge."""
    snapshot_dir = _surface_snapshot(tmp_path / "exp", _all_surfaces_rows(1))

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        ax = _panel_a(fig)
        total = len(CANONICAL_SURFACES)
        assert ax.get_ylim()[1] > total
        assert total in list(ax.get_yticks())
        _assert_nothing_clipped(ax)
    finally:
        plot_sa.plt.close(fig)


def test_panel_a_reference_line_steps_from_nine_to_ten_across_a_mixed_vintage_tree(
    tmp_path: Path,
) -> None:
    """The "all surfaces" line is per round: nine over a pre-S10 round, ten after."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S2"),
            _activity_row(1, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0, promoted=0),
            _activity_row(2, "S2"),
            _activity_row(2, "S10", attempted=1, promoted=0, cumulative_attempted=2),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        ax = _panel_a(fig)
        segments = _reference_segments(ax)
        assert set(segments) == {1, 2}
        assert set(segments[1]) == {9}
        assert segments[2][-1] == 10
        assert 9 in segments[2], "the segment into round 2 rises from 9 to 10 -- a step, not a jump"
        assert ax.get_ylim()[1] > 10
    finally:
        plot_sa.plt.close(fig)


def test_panel_a_ylabel_states_the_largest_declared_count_not_the_codes_count(
    tmp_path: Path,
) -> None:
    """Every plotted round is pre-S10, so the axis says nine even though the code has ten."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S1"),
            _activity_row(1, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0, promoted=0),
            _activity_row(2, "S2"),
            _activity_row(2, "S10", source=SURFACE_SOURCE_UNDECLARED, attempted=0, promoted=0),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        ax = _panel_a(fig)
        assert len(CANONICAL_SURFACES) == 10, "this test's premise is a ten-surface code base"
        assert "(of 9)" in ax.get_ylabel()
        assert "10" not in ax.get_ylabel()
        assert set(_reference_segments(ax)) == {1, 2}
        assert all(set(ys) == {9} for ys in _reference_segments(ax).values())
    finally:
        plot_sa.plt.close(fig)


def test_panel_a_draws_no_reference_segment_over_an_unknown_round(tmp_path: Path) -> None:
    """A round whose harness could not be read has no true total, so no segment is drawn for it."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S1"),
            _activity_row(1, "S10", source=SURFACE_SOURCE_NONE, attempted=0, promoted=0),
            _activity_row(2, "S1", source=SURFACE_SOURCE_UNKNOWN),
            _activity_row(2, "S10", source=SURFACE_SOURCE_UNKNOWN, attempted=0, promoted=0),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        ax = _panel_a(fig)
        segments = _reference_segments(ax)
        assert set(segments) == {1}
        assert set(segments[1]) == {10}
        assert "(of 10)" in ax.get_ylabel()
    finally:
        plot_sa.plt.close(fig)


def test_panel_a_ylabel_states_declared_total_unknown_when_every_round_is_unknown(
    tmp_path: Path,
) -> None:
    """No plotted round has a known declared set, so the axis says so, not a total."""
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [
            _activity_row(1, "S1", source=SURFACE_SOURCE_UNKNOWN),
            _activity_row(1, "S10", source=SURFACE_SOURCE_UNKNOWN, attempted=0, promoted=0),
            _activity_row(2, "S1", source=SURFACE_SOURCE_UNKNOWN),
            _activity_row(2, "S10", source=SURFACE_SOURCE_UNKNOWN, attempted=0, promoted=0),
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        ax = _panel_a(fig)
        assert ax.get_ylabel() == "distinct surfaces (declared total unknown)"
        assert _reference_segments(ax) == {}
    finally:
        plot_sa.plt.close(fig)


def test_panel_b_has_one_row_per_canonical_surface_including_s10(tmp_path: Path) -> None:
    snapshot_dir = _surface_snapshot(tmp_path / "exp", _all_surfaces_rows(1))

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        for ax in _heatmap_axes(fig):
            assert ax.images[0].get_array().shape[0] == len(CANONICAL_SURFACES)
        labels = [label.get_text() for label in _heatmap_axes(fig)[0].get_yticklabels()]
        assert labels == list(CANONICAL_SURFACES)
        assert "S10" in labels
    finally:
        plot_sa.plt.close(fig)


def test_a_tree_touching_all_ten_surfaces_renders_with_no_clipped_artist(tmp_path: Path) -> None:
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [*_all_surfaces_rows(1), *_all_surfaces_rows(2, attempted=2, promoted=2)],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        for ax in fig.axes:
            _assert_nothing_clipped(ax)
        ax = _panel_a(fig)
        assert ax.get_ylim()[1] > len(CANONICAL_SURFACES)
    finally:
        plot_sa.plt.close(fig)


# ---------------------------------------------------------------------------
# The unattributed footnote counts rounds, not table entries (R8)
# ---------------------------------------------------------------------------


def test_unattributed_footnote_counts_only_rounds_with_exclusions(tmp_path: Path) -> None:
    snapshot_dir = _surface_snapshot(
        tmp_path / "exp",
        [_activity_row(1), _activity_row(2, "S2"), _activity_row(3, "S3")],
        [
            {"round_index": 1, "unattributed_count": 0, "total_rows": 4},
            {"round_index": 2, "unattributed_count": 1, "total_rows": 4},
            {"round_index": 3, "unattributed_count": 0, "total_rows": 4},
        ],
    )

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        caption = _figure_text(fig)
        assert "1 ledger row(s) across 1 round(s)" in caption
        assert "across 3 round(s)" not in caption
    finally:
        plot_sa.plt.close(fig)


# ---------------------------------------------------------------------------
# Snapshot resolution and the provenance footer
# ---------------------------------------------------------------------------


def test_plots_default_to_the_latest_published_snapshot(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp"
    published = _surface_snapshot(out_dir, [_activity_row(1)])
    unpublished = _surface_snapshot(out_dir, [_activity_row(1)], publish=False)
    assert unpublished.name > published.name or unpublished != published

    assert plot_sa.main([str(out_dir)]) == 0

    assert (published / PLOTS_DIR / plot_sa.OUTPUT_FILENAME).exists()
    assert not (unpublished / PLOTS_DIR).exists()


def test_pinned_snapshot_is_rendered_into_its_own_plots_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp"
    first = _surface_snapshot(out_dir, [_activity_row(1)])
    second = _surface_snapshot(out_dir, [_activity_row(1)])
    assert first != second

    assert plot_sa.main([str(out_dir), "--snapshot", first.name]) == 0

    assert (first / PLOTS_DIR / plot_sa.OUTPUT_FILENAME).exists()
    assert not (second / PLOTS_DIR).exists()


def test_pinned_unpublished_snapshot_says_so_on_the_footer(tmp_path: Path) -> None:
    """Pinning an unfinished batch is allowed, but the figure has to admit it."""
    out_dir = tmp_path / "exp"
    unpublished = _surface_snapshot(out_dir, [_activity_row(1)], publish=False)

    assert plot_sa.main([str(out_dir), "--snapshot", unpublished.name]) == 0

    fig = plot_sa.build_figure(unpublished)
    try:
        assert "UNPUBLISHED SNAPSHOT" in _figure_text(fig)
    finally:
        plot_sa.plt.close(fig)


def test_unknown_pinned_snapshot_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    _surface_snapshot(out_dir, [_activity_row(1)])

    assert plot_sa.main([str(out_dir), "--snapshot", "20200101T000000Z"]) == 1

    assert "20200101T000000Z" in capsys.readouterr().err


def test_footer_carries_identity_snapshot_stamp_and_render_time(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp"
    snapshot_dir = _surface_snapshot(out_dir, [_activity_row(1)])
    identity = plot_style.snapshot_identity(snapshot_dir)
    assert identity is not None

    fig = plot_sa.build_figure(snapshot_dir)
    try:
        footer = _figure_text(fig)
        assert identity[:12] in footer
        assert snapshot_dir.name in footer
        assert "rendered" in footer
    finally:
        plot_sa.plt.close(fig)


def test_no_published_snapshot_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    _surface_snapshot(out_dir, [_activity_row(1)], publish=False)

    assert plot_sa.main([str(out_dir)]) == 1

    assert "no published analysis snapshot" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Bad inputs fail loudly (review residual #6)
# ---------------------------------------------------------------------------


def test_missing_companion_csv_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    snapshot_dir = _surface_snapshot(out_dir, [_activity_row(1)], write_unattributed=False)

    assert plot_sa.main([str(out_dir)]) == 1

    error = capsys.readouterr().err
    assert UNATTRIBUTED_FILENAME in error
    assert not (snapshot_dir / PLOTS_DIR).exists()


def test_header_only_csv_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    _surface_snapshot(out_dir, [])

    assert plot_sa.main([str(out_dir)]) == 1

    assert "no rounds" in capsys.readouterr().err


def test_incumbent_missing_csv_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    out_dir.mkdir(parents=True)
    snapshot = allocate_snapshot(out_dir)
    _publish(snapshot)

    assert plot_iq.main([str(out_dir)]) == 1

    assert INCUMBENT_QUALITY_FILENAME in capsys.readouterr().err


def test_incumbent_header_only_csv_is_a_clear_cli_error(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "exp"
    _quality_snapshot(out_dir, [])

    assert plot_iq.main([str(out_dir)]) == 1

    assert "no rounds" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Re-rendering never disturbs the frozen aggregation artifacts (KTD2)
# ---------------------------------------------------------------------------


def test_rerender_replaces_only_plots(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp"
    snapshot_dir = _surface_snapshot(out_dir, [_activity_row(1), _activity_row(2, "S2")])
    _quality_snapshot(out_dir, [_quality_row(1)])

    assert plot_sa.main([str(out_dir), "--snapshot", snapshot_dir.name]) == 0
    before = _artifact_bytes(snapshot_dir)
    assert PROVENANCE_FILENAME in before
    assert PUBLISHED_FILENAME in before

    assert plot_sa.main([str(out_dir), "--snapshot", snapshot_dir.name]) == 0

    assert _artifact_bytes(snapshot_dir) == before
    assert (snapshot_dir / PLOTS_DIR / plot_sa.OUTPUT_FILENAME).exists()


# ---------------------------------------------------------------------------
# One style module (R7)
# ---------------------------------------------------------------------------


def test_apply_style_sets_the_shared_rcparams() -> None:
    matplotlib.rcParams["figure.facecolor"] = "#123456"
    plot_style.apply_style()
    assert matplotlib.rcParams["figure.facecolor"] == plot_style.SURFACE
    assert matplotlib.rcParams["axes.labelcolor"] == plot_style.SECONDARY_INK


def test_plot_modules_take_their_palette_from_plot_style() -> None:
    for module in (plot_sa, plot_iq):
        source = Path(module.__file__).read_text()
        assert "rcParams" not in source, f"{module.__name__} still sets rcParams itself"
        for name in ("PRIMARY_INK", "SECONDARY_INK", "MUTED_INK", "GRIDLINE", "BASELINE"):
            assert f'{name} = "#' not in source, f"{module.__name__} redefines {name}"
