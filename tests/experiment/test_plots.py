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
from shrlm.experiment.surface_activity import (  # noqa: E402
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
        "surface_source": "ledger",
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
    _write_csv(snapshot.path / SURFACE_ACTIVITY_FILENAME, SURFACE_ACTIVITY_FIELDNAMES, activity_rows)
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
    _write_csv(snapshot.path / INCUMBENT_QUALITY_FILENAME, INCUMBENT_QUALITY_FIELDNAMES, quality_rows)
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
        plotted = {round(offset[0]) for collection in partial for offset in collection.get_offsets()}
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
        plotted = {round(offset[0]) for collection in partial for offset in collection.get_offsets()}
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
    """U4's extra ``merged`` category row must not enter the S1-S9 heatmaps."""
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
