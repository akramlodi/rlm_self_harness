"""Tests for the candidate table's surface attribution (R4, KTD3, KTD7).

``incumbent_quality_candidates.csv`` and ``surface_activity.csv`` ship in one
snapshot and describe the same ledger rows, so they have to place a candidate
on the same surface. They did not: the candidate table wrote
``record["surface"]`` raw while ``surface_activity`` resolved the same row
through ``rounds.resolve_surface``, which backfills a missing-or-null surface
from the round's persisted proposal. Since the ledger only gained the field on
this branch, every pre-existing ledger produced a blank surface in one table
and a recovered one in the other, in the same directory, with nothing saying
why.

These tests pin the agreement itself rather than the resolver's internals
(``tests.experiment.test_surface_activity`` owns those): the same candidate,
read out of both tables, lands on the same surface, and every candidate row
says where its surface came from.

Ledger rows are serialized through ``CandidateDecision.to_dict``, so the
"key present, value null" shape is the loop's own bytes rather than the test
author's model of them.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.analysis_io import PROVENANCE_FILENAME, Snapshot, allocate_snapshot
from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_CANDIDATES_FILENAME,
    all_candidate_quality_over_rounds,
    run_incumbent_quality,
    write_incumbent_quality,
)
from shrlm.experiment.orchestrator import ROUND_MARKER_FILENAME
from shrlm.experiment.rounds import (
    MERGED_CATEGORY,
    SURFACE_SOURCE_BACKFILLED,
    SURFACE_SOURCE_LEDGER,
    SURFACE_SOURCE_MERGED,
    SURFACE_SOURCE_NONE,
)
from shrlm.experiment.surface_activity import surface_activity_over_rounds
from shrlm.optimization.promotion import (
    DECISION_PROMOTED,
    DECISION_REJECTED,
    MERGED_SUBJECT_ID,
    CandidateDecision,
)
from shrlm.optimization.proposal import PROPOSAL_FILENAME, PROPOSAL_FORMAT
from shrlm.optimization.validation import LEDGER_RECORD_FORMAT, PROMOTIONS_FILENAME
from tests.experiment.test_rounds import (
    complete_round,
    read_csv,
    record_for,
    write_config,
    write_json,
    write_jsonl,
)

FROZEN = datetime(2026, 8, 18, 16, 48, 0, tzinfo=UTC)
PROFILE = "full"


def scored_rule(*, candidate_pass_count: int = 2, n_runs: int = 4) -> dict[str, Any]:
    """The ``rule`` block a scored candidate carries, on both splits.

    Only a scored row reaches the candidate table -- an unscored loader
    rejection has no measured pass rate to plot -- so every row here has one.
    """
    split = {
        "baseline_pass_count": 1,
        "candidate_pass_count": candidate_pass_count,
        "n_runs": n_runs,
    }
    return {"heldin": dict(split), "heldout": dict(split)}


def ledger_record(
    subject_id: str,
    *,
    decision: str = DECISION_REJECTED,
    surface: str | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """One scored ``promotions.jsonl`` line, in the loop's own serialization.

    ``legacy`` drops the ``surface`` key entirely: the shape of a ledger
    written before the field existed, which is every ledger this experiment
    already holds.
    """
    payload: dict[str, Any] = {
        "format": LEDGER_RECORD_FORMAT,
        **CandidateDecision(
            subject_id=subject_id,
            decision=decision,
            reasons=(),
            tau_regression=0.0,
            tau_improvement=0.0,
            surface=surface,
        ).to_dict(),
        "merge": {"role": None, "constituent_ids": None},
        "rule": scored_rule(),
    }
    if legacy:
        del payload["surface"]
    return payload


def write_round(
    out_dir: Path,
    round_index: int,
    records: list[dict[str, Any]],
    *,
    proposals: dict[str, str] | None = None,
) -> None:
    """A finished round whose ledger holds exactly ``records``.

    ``proposals`` maps ``candidate_id -> surface`` and is written where
    discovery says the round's proposals live, so the backfill join runs
    against the loop's layout rather than one assembled here.
    """
    complete_round(out_dir, round_index)
    record = record_for(out_dir, round_index)
    write_jsonl(record.validation_round_path / PROMOTIONS_FILENAME, records)
    for candidate_id, surface in (proposals or {}).items():
        write_json(
            record.proposals_dir / candidate_id / PROPOSAL_FILENAME,
            {
                "format": PROPOSAL_FORMAT,
                "candidate_id": candidate_id,
                "surface": surface,
                "harness": {},
            },
        )


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    out_dir = tmp_path / "experiment"
    write_config(out_dir, PROFILE)
    return out_dir


@pytest.fixture
def snapshot(experiment: Path) -> Snapshot:
    return allocate_snapshot(experiment, now=FROZEN)


def by_subject(rows: list[Any]) -> dict[str, Any]:
    return {row.subject_id: row for row in rows}


def recorded_sources(snapshot: Snapshot) -> dict[str, Any]:
    """The published manifest, keyed by the path each entry names."""
    payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
    return {entry["path"]: entry for entry in payload["sources"]}


def proposal_source_name(out_dir: Path, round_index: int, candidate_id: str) -> str:
    proposal = record_for(out_dir, round_index).proposals_dir / candidate_id / PROPOSAL_FILENAME
    return proposal.relative_to(out_dir).as_posix()


class TestCandidateSurfaceAttribution:
    def test_a_legacy_ledger_row_gets_the_same_surface_the_activity_table_gives_it(
        self, experiment: Path
    ) -> None:
        """The disagreement itself: two tables, one candidate, one surface."""
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s2", decision=DECISION_PROMOTED, legacy=True)],
            proposals={"r01-c01-s2": "S2"},
        )

        (candidate,) = all_candidate_quality_over_rounds(experiment)
        activity, _unattributed = surface_activity_over_rounds(experiment)
        touched = {row.surface for row in activity if row.attempted_count}

        assert candidate.surface == "S2"
        assert candidate.surface_source == SURFACE_SOURCE_BACKFILLED
        assert touched == {candidate.surface}

    def test_a_null_surface_with_a_persisted_proposal_is_backfilled(self, experiment: Path) -> None:
        """The shape the running loop writes for a loader rejection."""
        record = ledger_record("r01-c01-s7")
        assert "surface" in record and record["surface"] is None

        write_round(experiment, 1, [record], proposals={"r01-c01-s7": "S7"})

        (candidate,) = all_candidate_quality_over_rounds(experiment)
        assert candidate.surface == "S7"
        assert candidate.surface_source == SURFACE_SOURCE_BACKFILLED

    def test_a_recorded_surface_is_taken_verbatim_and_marked_as_recorded(
        self, experiment: Path
    ) -> None:
        """A proposal never overrides the ledger; the source says which spoke."""
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s4", surface="S4")],
            proposals={"r01-c01-s4": "S9"},
        )

        (candidate,) = all_candidate_quality_over_rounds(experiment)
        assert candidate.surface == "S4"
        assert candidate.surface_source == SURFACE_SOURCE_LEDGER

    def test_a_row_with_no_proposal_to_join_stays_honestly_unresolved(
        self, experiment: Path
    ) -> None:
        write_round(experiment, 1, [ledger_record("r01-c01-s3")])

        (candidate,) = all_candidate_quality_over_rounds(experiment)
        assert candidate.surface is None
        assert candidate.surface_source == SURFACE_SOURCE_NONE

    def test_the_merged_record_is_its_own_category_and_is_never_backfilled(
        self, experiment: Path
    ) -> None:
        """KTD7: the merged harness spans several surfaces, so null is correct."""
        write_round(
            experiment,
            1,
            [ledger_record(MERGED_SUBJECT_ID, decision=DECISION_PROMOTED)],
            proposals={MERGED_SUBJECT_ID: "S5"},
        )

        (candidate,) = all_candidate_quality_over_rounds(experiment)
        assert candidate.surface == MERGED_CATEGORY
        assert candidate.surface_source == SURFACE_SOURCE_MERGED

    def test_the_written_table_carries_the_source_beside_every_surface(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """The CSV is what a reader joins against; the source has to reach it."""
        write_round(
            experiment,
            1,
            [
                ledger_record("r01-c01-s2", legacy=True),
                ledger_record("r01-c02-s4", surface="S4"),
                ledger_record("r01-c03-s6"),
            ],
            proposals={"r01-c01-s2": "S2"},
        )

        write_incumbent_quality(
            snapshot, experiment, [], all_candidate_quality_over_rounds(experiment)
        )
        rows = {
            row["subject_id"]: row
            for row in read_csv(snapshot.path / INCUMBENT_QUALITY_CANDIDATES_FILENAME)
        }

        assert rows["r01-c01-s2"]["surface"] == "S2"
        assert rows["r01-c01-s2"]["surface_source"] == SURFACE_SOURCE_BACKFILLED
        assert rows["r01-c02-s4"]["surface"] == "S4"
        assert rows["r01-c02-s4"]["surface_source"] == SURFACE_SOURCE_LEDGER
        assert rows["r01-c03-s6"]["surface"] == ""
        assert rows["r01-c03-s6"]["surface_source"] == SURFACE_SOURCE_NONE


# ---------------------------------------------------------------------------
# Provenance: the candidates table's backfill has to be reproducible too
# ---------------------------------------------------------------------------


class TestSurfaceSourcesAreRecorded:
    """A backfilled surface in the candidates table is an attribution no ledger
    states, exactly as in ``surface_activity`` -- so the proposal that supplied
    it belongs in the manifest here as well. The two tables resolve surfaces the
    same way; a manifest that covered only one of them would let the same
    recovered value be checkable in one table and unverifiable in the other."""

    def test_the_proposal_a_backfilled_surface_came_from_is_recorded(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s2", legacy=True)],
            proposals={"r01-c01-s2": "S2"},
        )

        snapshot.run_tool("incumbent_quality", lambda: run_incumbent_quality(experiment, snapshot))
        snapshot.publish()

        sources = recorded_sources(snapshot)
        name = proposal_source_name(experiment, 1, "r01-c01-s2")
        assert name in sources, f"the backfilled proposal is missing from {sorted(sources)}"
        assert sources[name]["sha256"] is not None

    def test_the_completeness_inputs_are_recorded(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """``round_complete`` and ``runs_complete`` are published columns read
        off the round marker, so the marker is an input: editing it moves a
        published verdict, and a manifest that omitted it would not show that."""
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s2", legacy=True)],
            proposals={"r01-c01-s2": "S2"},
        )

        snapshot.run_tool("incumbent_quality", lambda: run_incumbent_quality(experiment, snapshot))
        snapshot.publish()

        sources = recorded_sources(snapshot)
        marker = f"opt/round_01/{ROUND_MARKER_FILENAME}"
        assert marker in sources, f"the round marker is missing from {sorted(sources)}"
