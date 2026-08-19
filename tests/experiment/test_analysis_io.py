"""Tests for U3's snapshot + provenance layer (KTD2, KTD8).

The claim this module has to make checkable is narrow and load-bearing: an
analysis rerun can never destroy a published snapshot, and every artifact a
snapshot holds can be traced back to the experiment identity, the analysis
revision, and the exact source bytes it was derived from.

Two of these tests carry most of that weight:

* Allocation under a *frozen* clock. Wall-clock timing would make the
  same-second collision a race the suite passes by luck; injecting the
  timestamp forces the collision on every run, which is the only way the
  monotonic-suffix retry is actually exercised.
* The source manifest across an edited artifact. Experiment identity plus
  round number does not distinguish two snapshots built from a re-mined
  round -- only the content hashes do, so the test edits one consumed file
  and asserts that exactly one manifest entry moved.

The CLI cases use the fabrication helpers from ``tests.experiment.test_rounds``
rather than hand-written path strings, so the trees here are built through the
loop's own layout functions and move with it.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.analysis_io import (
    ANALYSIS_DIR,
    IDENTITY_SOURCE_CONFIG,
    IDENTITY_SOURCE_DERIVED,
    PROVENANCE_FILENAME,
    PROVENANCE_FORMAT,
    PUBLISHED_FILENAME,
    AnalysisOutputError,
    allocate_snapshot,
    latest_snapshot,
    record_ledger_sources,
    write_csv,
)
from shrlm.experiment.collapse_and_attribution import EVALUATION_FILENAME
from shrlm.experiment.collapse_and_attribution import main as collapse_main
from shrlm.experiment.config import CONFIG_PATH
from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    INCUMBENT_QUALITY_FILENAME,
)
from shrlm.experiment.incumbent_quality import main as incumbent_main
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    EVIDENCE_MARKER_FILENAME,
    ROUND_MARKER_FILENAME,
)
from shrlm.experiment.pattern_frequency_diff import main as diff_main
from shrlm.experiment.surface_activity import (
    SURFACE_ACTIVITY_FILENAME,
    UNATTRIBUTED_FILENAME,
)
from shrlm.experiment.surface_activity import main as surface_main
from shrlm.optimization.driver import INSTANCES_FILE, MANIFEST_FILE
from tests.experiment.test_rounds import (
    complete_round,
    eval_set_payload,
    record_for,
    write_bundle,
    write_config,
    write_eval_summary,
)

FROZEN = datetime(2026, 8, 18, 16, 48, 0, tzinfo=UTC)
FROZEN_STAMP = "20260818T164800Z"


@dataclass(frozen=True)
class Row:
    """A minimal ``to_dict()`` row, standing in for any analysis dataclass."""

    a: int
    b: str

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b}


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An experiment directory with a persisted identity and one full round."""
    out_dir = tmp_path / "experiment"
    write_config(out_dir)
    complete_round(out_dir, 1)
    return out_dir


def sources_by_path(payload: dict[str, Any]) -> dict[str, Any]:
    return {entry["path"]: entry for entry in payload["sources"]}


def snapshot_dirs(parent: Path) -> list[Path]:
    return sorted(path for path in parent.iterdir() if path.is_dir())


def only_snapshot(parent: Path) -> Path:
    dirs = snapshot_dirs(parent)
    assert len(dirs) == 1, f"expected exactly one snapshot under {parent}, found {dirs}"
    return dirs[0]


# ---------------------------------------------------------------------------
# Allocation: never two batches in one directory
# ---------------------------------------------------------------------------


class TestAllocation:
    def test_a_snapshot_is_a_utc_stamped_directory_under_analysis(self, experiment: Path) -> None:
        snapshot = allocate_snapshot(experiment, now=FROZEN)

        assert snapshot.path == experiment / ANALYSIS_DIR / FROZEN_STAMP
        assert snapshot.path.is_dir()

    def test_two_allocations_in_the_same_utc_second_get_distinct_directories(
        self, experiment: Path
    ) -> None:
        first = allocate_snapshot(experiment, now=FROZEN)
        second = allocate_snapshot(experiment, now=FROZEN)

        assert first.path != second.path
        assert second.path.name == f"{FROZEN_STAMP}-2"

    def test_a_third_collision_takes_the_next_monotonic_suffix(self, experiment: Path) -> None:
        allocate_snapshot(experiment, now=FROZEN)
        allocate_snapshot(experiment, now=FROZEN)
        third = allocate_snapshot(experiment, now=FROZEN)

        assert third.path.name == f"{FROZEN_STAMP}-3"

    def test_the_first_snapshots_artifacts_are_untouched_by_a_second_allocation(
        self, experiment: Path
    ) -> None:
        first = allocate_snapshot(experiment, now=FROZEN)
        write_csv(first.path / "rows.csv", [Row(1, "x")], fieldnames=["a", "b"])
        first.publish()
        before = (first.path / "rows.csv").read_bytes()
        provenance_before = (first.path / PROVENANCE_FILENAME).read_bytes()

        second = allocate_snapshot(experiment, now=FROZEN)
        write_csv(second.path / "rows.csv", [Row(2, "y")], fieldnames=["a", "b"])
        second.publish()

        assert (first.path / "rows.csv").read_bytes() == before
        assert (first.path / PROVENANCE_FILENAME).read_bytes() == provenance_before

    def test_out_overrides_the_snapshot_parent_and_still_stamps(self, experiment: Path) -> None:
        parent = experiment.parent / "elsewhere"

        snapshot = allocate_snapshot(experiment, parent=parent, now=FROZEN)

        assert snapshot.path == parent / FROZEN_STAMP


# ---------------------------------------------------------------------------
# Publication: an unfinished batch is never "latest"
# ---------------------------------------------------------------------------


class TestPublication:
    def test_a_failing_tool_leaves_provenance_written_and_the_snapshot_unpublished(
        self, experiment: Path
    ) -> None:
        snapshot = allocate_snapshot(experiment, now=FROZEN)

        def first() -> None:
            write_csv(snapshot.path / "first.csv", [Row(1, "x")], fieldnames=["a", "b"])

        def second() -> None:
            raise RuntimeError("second tool blew up")

        assert snapshot.run_tool("first", first) is True
        assert snapshot.run_tool("second", second) is False
        published = snapshot.publish()

        assert published is False
        assert (snapshot.path / "first.csv").exists()
        assert not (snapshot.path / PUBLISHED_FILENAME).exists()
        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        tools = {entry["name"]: entry for entry in payload["tools"]}
        assert tools["first"]["ok"] is True
        assert tools["second"]["ok"] is False
        assert "second tool blew up" in tools["second"]["error"]

    def test_a_batch_where_every_tool_succeeded_is_published(self, experiment: Path) -> None:
        snapshot = allocate_snapshot(experiment, now=FROZEN)
        snapshot.run_tool("only", lambda: None)

        assert snapshot.publish() is True
        assert (snapshot.path / PUBLISHED_FILENAME).exists()
        assert snapshot.is_published

    def test_latest_skips_the_unpublished_directory_and_returns_the_newest_published(
        self, experiment: Path
    ) -> None:
        oldest = allocate_snapshot(experiment, now=datetime(2026, 8, 18, 10, 0, 0, tzinfo=UTC))
        oldest.publish()
        newest = allocate_snapshot(experiment, now=datetime(2026, 8, 18, 11, 0, 0, tzinfo=UTC))
        newest.publish()
        unpublished = allocate_snapshot(experiment, now=datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC))

        assert unpublished.path.is_dir()
        assert latest_snapshot(experiment) == newest.path

    def test_latest_is_none_when_no_snapshot_was_ever_published(self, experiment: Path) -> None:
        allocate_snapshot(experiment, now=FROZEN)

        assert latest_snapshot(experiment) is None


# ---------------------------------------------------------------------------
# Provenance: identity, revision, and the source manifest
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_identity_comes_from_config_json_with_the_full_record(self, experiment: Path) -> None:
        recorded = json.loads((experiment / CONFIG_FILENAME).read_text())
        ledger = experiment / "opt" / "round_01" / "validation" / "round_01" / "promotions.jsonl"

        snapshot = allocate_snapshot(experiment, now=FROZEN)
        snapshot.record_rounds([1])
        snapshot.record_eval_sets(["b1/graphwalks_short"])
        snapshot.record_source(ledger)
        snapshot.run_tool("surface_activity", lambda: None)
        snapshot.publish()

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        assert payload["format"] == PROVENANCE_FORMAT
        assert payload["identity_hash"] == recorded["identity_hash"]
        assert payload["identity_source"] == IDENTITY_SOURCE_CONFIG
        assert payload["legacy_derived"] is False
        assert payload["created_at"].startswith("2026-08-18T16:48:00")
        assert "analysis_revision" in payload
        assert payload["rounds"] == [1]
        assert payload["eval_sets"] == ["b1/graphwalks_short"]
        assert [entry["name"] for entry in payload["tools"]] == ["surface_activity"]
        entry = sources_by_path(payload)["opt/round_01/validation/round_01/promotions.jsonl"]
        assert entry["sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()

    def test_a_tree_with_no_config_derives_its_identity_from_the_source_hashes(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "legacy"
        out_dir.mkdir()
        artifact = out_dir / "artifact.jsonl"
        artifact.write_text('{"a": 1}\n')

        snapshot = allocate_snapshot(out_dir, now=FROZEN)
        snapshot.record_source(artifact)
        assert snapshot.publish() is True

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        assert payload["identity_source"] == IDENTITY_SOURCE_DERIVED
        assert payload["legacy_derived"] is True
        assert payload["identity_hash"]

    def test_the_derived_identity_is_deterministic_over_the_same_sources(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "legacy"
        out_dir.mkdir()
        artifact = out_dir / "artifact.jsonl"
        artifact.write_text('{"a": 1}\n')

        hashes = []
        for _ in range(2):
            snapshot = allocate_snapshot(out_dir, now=FROZEN)
            snapshot.record_source(artifact)
            snapshot.publish()
            hashes.append(
                json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())["identity_hash"]
            )

        assert hashes[0] == hashes[1]

    def test_editing_one_consumed_artifact_moves_only_its_own_source_hash(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "legacy"
        out_dir.mkdir()
        stable = out_dir / "stable.jsonl"
        stable.write_text('{"a": 1}\n')
        edited = out_dir / "edited.jsonl"
        edited.write_text('{"b": 1}\n')

        first = allocate_snapshot(out_dir, now=FROZEN)
        first.record_source(stable)
        first.record_source(edited)
        first.publish()

        edited.write_text('{"b": 2}\n')
        second = allocate_snapshot(out_dir, now=FROZEN)
        second.record_source(stable)
        second.record_source(edited)
        second.publish()

        before = sources_by_path(json.loads((first.path / PROVENANCE_FILENAME).read_text()))
        after = sources_by_path(json.loads((second.path / PROVENANCE_FILENAME).read_text()))
        assert before["stable.jsonl"]["sha256"] == after["stable.jsonl"]["sha256"]
        assert before["edited.jsonl"]["sha256"] != after["edited.jsonl"]["sha256"]

    def test_the_inputs_behind_the_completeness_columns_are_recorded(
        self, experiment: Path
    ) -> None:
        """``round_complete`` / ``runs_complete`` / ``evidence_complete`` /
        ``n_expected_runs`` are read off the stage markers, the run plan and the
        configuration -- none of which is a ledger. A manifest listing only the
        ledgers lets any of them be edited without a recorded hash moving, which
        is precisely the drift the manifest exists to detect."""
        snapshot = allocate_snapshot(experiment, now=FROZEN)
        record_ledger_sources(snapshot, experiment)
        snapshot.publish()

        sources = sources_by_path(json.loads((snapshot.path / PROVENANCE_FILENAME).read_text()))
        record = record_for(experiment, 1)
        expected = {
            CONFIG_FILENAME,
            (record.round_path / ROUND_MARKER_FILENAME).relative_to(experiment).as_posix(),
            (record.mining_round_path / EVIDENCE_MARKER_FILENAME)
            .relative_to(experiment)
            .as_posix(),
            (record.mining_round_path / MANIFEST_FILE).relative_to(experiment).as_posix(),
            (record.mining_round_path / INSTANCES_FILE).relative_to(experiment).as_posix(),
        }
        assert expected <= set(sources), f"missing from the manifest: {expected - set(sources)}"
        # The TOML lives outside the experiment directory, so it is recorded
        # repository-relative rather than dropped.
        assert any(name.endswith(CONFIG_PATH.name) for name in sources), sorted(sources)

    def test_editing_a_round_marker_moves_a_recorded_source_hash(self, experiment: Path) -> None:
        """The published ``round_complete`` flips when this file is edited; a
        provenance record that did not move would say the two snapshots were
        derived from the same bytes."""
        marker = record_for(experiment, 1).round_path / ROUND_MARKER_FILENAME
        name = marker.relative_to(experiment).as_posix()

        first = allocate_snapshot(experiment, now=FROZEN)
        record_ledger_sources(first, experiment)
        first.publish()

        payload = json.loads(marker.read_text())
        payload["round"] = 99
        marker.write_text(json.dumps(payload))

        second = allocate_snapshot(experiment, now=FROZEN)
        record_ledger_sources(second, experiment)
        second.publish()

        before = sources_by_path(json.loads((first.path / PROVENANCE_FILENAME).read_text()))
        after = sources_by_path(json.loads((second.path / PROVENANCE_FILENAME).read_text()))
        assert before[name]["sha256"] != after[name]["sha256"]

    def test_an_absent_source_is_recorded_with_a_null_hash_rather_than_dropped(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "legacy"
        out_dir.mkdir()

        snapshot = allocate_snapshot(out_dir, now=FROZEN)
        snapshot.record_source(out_dir / "never_written.jsonl")
        snapshot.publish()

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        assert sources_by_path(payload)["never_written.jsonl"]["sha256"] is None

    def test_a_source_directory_is_hashed_as_one_entry_over_its_files(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "legacy"
        traces = out_dir / "runs"
        traces.mkdir(parents=True)
        (traces / "a.json").write_text("{}\n")
        (traces / "b.json").write_text("{}\n")

        first = allocate_snapshot(out_dir, now=FROZEN)
        first.record_source_dir(traces)
        first.publish()
        (traces / "b.json").write_text('{"changed": true}\n')
        second = allocate_snapshot(out_dir, now=FROZEN)
        second.record_source_dir(traces)
        second.publish()

        before = sources_by_path(json.loads((first.path / PROVENANCE_FILENAME).read_text()))["runs"]
        after = sources_by_path(json.loads((second.path / PROVENANCE_FILENAME).read_text()))["runs"]
        assert before["n_files"] == 2
        assert before["sha256"] != after["sha256"]

    def test_an_unavailable_git_revision_records_null_rather_than_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> Any:
            raise OSError("git is not installed here")

        monkeypatch.setattr(subprocess, "run", explode)
        out_dir = tmp_path / "legacy"
        out_dir.mkdir()

        snapshot = allocate_snapshot(out_dir, now=FROZEN)
        assert snapshot.publish() is True

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        assert payload["analysis_revision"] is None


# ---------------------------------------------------------------------------
# The one CSV writer
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_an_empty_result_still_writes_the_header_from_the_explicit_fieldnames(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.csv"

        write_csv(path, [], fieldnames=["a", "b"])

        # Bytes, not text: the CRLF line ending is the ``csv`` module's own
        # default (and what the per-module writers this replaces produced), and
        # a text-mode read would translate it away before the assertion saw it.
        assert path.read_bytes() == b"a,b\r\n"

    def test_rows_are_written_in_the_declared_field_order(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.csv"

        write_csv(path, [Row(1, "x")], fieldnames=["b", "a"])

        assert path.read_text().splitlines() == ["b,a", "x,1"]

    def test_a_byte_identical_rewrite_is_a_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.csv"
        write_csv(path, [Row(1, "x")], fieldnames=["a", "b"])

        write_csv(path, [Row(1, "x")], fieldnames=["a", "b"])

        assert path.read_text().splitlines() == ["a,b", "1,x"]

    def test_overwriting_a_published_artifact_with_different_bytes_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "rows.csv"
        write_csv(path, [Row(1, "x")], fieldnames=["a", "b"])

        with pytest.raises(AnalysisOutputError):
            write_csv(path, [Row(2, "y")], fieldnames=["a", "b"])

    def test_the_plots_directory_stays_re_renderable(self, experiment: Path) -> None:
        snapshot = allocate_snapshot(experiment, now=FROZEN)
        snapshot.publish()

        figure = snapshot.plots_dir / "graph.png"
        figure.write_bytes(b"first render")
        figure.write_bytes(b"second render")

        assert figure.read_bytes() == b"second render"


# ---------------------------------------------------------------------------
# CLI plumbing: every aggregation writes into a snapshot
# ---------------------------------------------------------------------------


class TestSurfaceActivityCli:
    def test_a_standalone_run_allocates_its_own_snapshot_under_analysis(
        self, experiment: Path
    ) -> None:
        assert surface_main([str(experiment)]) == 0

        snapshot = only_snapshot(experiment / ANALYSIS_DIR)
        assert (snapshot / SURFACE_ACTIVITY_FILENAME).exists()
        assert (snapshot / UNATTRIBUTED_FILENAME).exists()
        assert (snapshot / PUBLISHED_FILENAME).exists()
        payload = json.loads((snapshot / PROVENANCE_FILENAME).read_text())
        assert [entry["name"] for entry in payload["tools"]] == ["surface_activity"]
        assert payload["rounds"] == [1]
        assert payload["sources"]

    def test_out_overrides_the_snapshot_parent(self, experiment: Path, tmp_path: Path) -> None:
        parent = tmp_path / "published"

        assert surface_main([str(experiment), "--out", str(parent)]) == 0

        snapshot = only_snapshot(parent)
        assert (snapshot / SURFACE_ACTIVITY_FILENAME).exists()
        assert not (experiment / ANALYSIS_DIR).exists()

    def test_two_runs_never_share_a_snapshot(self, experiment: Path) -> None:
        assert surface_main([str(experiment)]) == 0
        assert surface_main([str(experiment)]) == 0

        assert len(snapshot_dirs(experiment / ANALYSIS_DIR)) == 2


class TestIncumbentQualityCli:
    def test_both_tables_land_in_one_published_snapshot(self, experiment: Path) -> None:
        assert incumbent_main([str(experiment)]) == 0

        snapshot = only_snapshot(experiment / ANALYSIS_DIR)
        assert (snapshot / INCUMBENT_QUALITY_FILENAME).exists()
        assert (snapshot / INCUMBENT_QUALITY_CANDIDATES_FILENAME).exists()
        assert (snapshot / PUBLISHED_FILENAME).exists()

    def test_out_overrides_the_snapshot_parent(self, experiment: Path, tmp_path: Path) -> None:
        parent = tmp_path / "published"

        assert incumbent_main([str(experiment), "--out", str(parent)]) == 0

        assert (only_snapshot(parent) / INCUMBENT_QUALITY_FILENAME).exists()


class TestCollapseAndAttributionCli:
    def test_the_evaluation_phase_writes_into_a_published_snapshot(self, experiment: Path) -> None:
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        assert collapse_main([str(experiment), "--phase", "evaluation"]) == 0

        snapshot = only_snapshot(experiment / ANALYSIS_DIR)
        assert (snapshot / EVALUATION_FILENAME).exists()
        payload = json.loads((snapshot / PROVENANCE_FILENAME).read_text())
        assert payload["eval_sets"] == ["b1/graphwalks_short"]
        assert (snapshot / PUBLISHED_FILENAME).exists()


class TestPatternFrequencyDiffCli:
    def test_the_experiment_directory_anchors_the_snapshot(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        before = write_bundle(tmp_path / "before.json", 1, support=5, n_runs=10)
        after = write_bundle(tmp_path / "after.json", 2, support=2, n_runs=10)

        assert diff_main([str(experiment), str(before), str(after)]) == 0

        snapshot = only_snapshot(experiment / ANALYSIS_DIR)
        assert (snapshot / "pattern_frequency_diff_round_1_vs_round_2.csv").exists()
        assert (snapshot / "pattern_frequency_diff_round_1_vs_round_2.json").exists()
        assert (snapshot / PUBLISHED_FILENAME).exists()
        payload = json.loads((snapshot / PROVENANCE_FILENAME).read_text())
        assert payload["identity_source"] == IDENTITY_SOURCE_CONFIG
        # The two bundles the diff compared, plus the inventory inputs behind
        # the completeness table it also publishes (``round_complete`` /
        # ``evidence_complete`` are read off the markers and the config, not off
        # any bundle, so a manifest naming only the bundles would let a marker
        # edit move a published column with no recorded hash moving).
        sources = sources_by_path(payload)
        assert {before.name, after.name} <= {Path(name).name for name in sources}
        record = record_for(experiment, 1)
        assert (record.round_path / ROUND_MARKER_FILENAME).relative_to(
            experiment
        ).as_posix() in sources
        assert CONFIG_FILENAME in sources

    def test_out_overrides_the_snapshot_parent(self, experiment: Path, tmp_path: Path) -> None:
        before = write_bundle(tmp_path / "before.json", 1, support=5, n_runs=10)
        after = write_bundle(tmp_path / "after.json", 2, support=2, n_runs=10)
        parent = tmp_path / "published"

        assert diff_main([str(experiment), str(before), str(after), "--out", str(parent)]) == 0

        assert (only_snapshot(parent) / "pattern_frequency_diff_round_1_vs_round_2.json").exists()

    def test_the_json_summary_carries_the_experiment_identity_and_creation_time(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        before = write_bundle(tmp_path / "before.json", 1, support=5, n_runs=10)
        after = write_bundle(tmp_path / "after.json", 2, support=2, n_runs=10)
        recorded = json.loads((experiment / CONFIG_FILENAME).read_text())

        assert diff_main([str(experiment), str(before), str(after)]) == 0

        snapshot = only_snapshot(experiment / ANALYSIS_DIR)
        summary = json.loads(
            (snapshot / "pattern_frequency_diff_round_1_vs_round_2.json").read_text()
        )
        assert summary["identity_hash"] == recorded["identity_hash"]
        assert summary["created_at"]


# ---------------------------------------------------------------------------
# The duplicate writers are gone (R7)
# ---------------------------------------------------------------------------


def test_no_module_keeps_a_private_csv_writer() -> None:
    package = Path(__file__).resolve().parents[2] / "shrlm" / "experiment"
    offenders = [
        path.name for path in sorted(package.glob("*.py")) if "def _write_csv" in path.read_text()
    ]

    assert offenders == []
