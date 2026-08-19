"""Tests for U4: a ledger row's surface, recovered honestly (R4, KTD3, KTD7).

The claim under test is that a surface-level count is never silently missing
history that is still on disk, and never claims history it invented. Four kinds
of ledger row exist, and each must land in exactly one place:

* a row whose ``surface`` the ledger recorded -- taken verbatim, ``ledger``;
* a row whose ``surface`` is *null or absent* but whose proposal artifact
  survives -- recovered by joining ``subject_id`` to that proposal,
  ``backfilled``;
* a row with no proposal to join -- genuinely unresolvable, ``none``, and
  tallied unattributed;
* the merged harness's own re-evaluation row -- its ``merged`` category,
  ``merged``, never backfilled and never unattributed (KTD7: it spans several
  surfaces by construction, so a null surface is correct for it, not missing).

The first scenario is the one that motivates the whole unit and is asserted
first here. ``CandidateDecision.to_dict`` ALWAYS emits the ``surface`` key, so
the current loop serializes a loader rejection as ``surface: null`` -- a
backfill triggered only on an absent key would never fire for anything the
running experiment writes, and every loader rejection would stay permanently
unattributed while its proposal sat on disk. The ledger rows in these tests are
therefore built through ``CandidateDecision.to_dict`` itself rather than by
hand, so the "key present, value null" shape is the loop's own bytes and not
the test author's model of them.

Trees are fabricated through ``tests.experiment.test_rounds``' helpers, which
build them with the loop's own layout functions; proposal artifacts are written
at the path discovery reports for the round, never at a locally assembled one.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.analysis_io import Snapshot, allocate_snapshot
from shrlm.experiment.rounds import (
    MERGED_CATEGORY,
    SURFACE_SOURCE_BACKFILLED,
    SURFACE_SOURCE_LEDGER,
    SURFACE_SOURCE_MERGED,
    SURFACE_SOURCE_NONE,
    discover_rounds,
    resolve_surface,
)
from shrlm.experiment.surface_activity import (
    SURFACE_ACTIVITY_FIELDNAMES,
    SURFACE_ACTIVITY_FILENAME,
    UNATTRIBUTED_FILENAME,
    main,
    surface_activity_over_rounds,
    write_surface_activity,
)
from shrlm.optimization.promotion import (
    DECISION_ACCEPTED,
    DECISION_PROMOTED,
    DECISION_REJECTED,
    MERGED_SUBJECT_ID,
    CandidateDecision,
)
from shrlm.optimization.proposal import PROPOSAL_FILENAME, PROPOSAL_FORMAT
from shrlm.optimization.validation import (
    LEDGER_RECORD_FORMAT,
    PROMOTIONS_FILENAME,
    ROLE_MERGED,
)
from tests.experiment.test_rounds import (
    complete_round,
    record_for,
    write_config,
    write_json,
    write_jsonl,
)

FROZEN = datetime(2026, 8, 18, 16, 48, 0, tzinfo=UTC)
PROFILE = "full"


# ---------------------------------------------------------------------------
# Fabrication: ledger rows in the loop's own serialization, proposals on disk
# ---------------------------------------------------------------------------


def ledger_record(
    subject_id: str,
    *,
    decision: str = DECISION_REJECTED,
    surface: str | None = None,
    role: str | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """One ``promotions.jsonl`` line, serialized the way the loop serializes it.

    ``legacy`` drops the ``surface`` key entirely -- the shape a ledger written
    before the field existed has, and the only shape a key-absence backfill
    would ever have caught.
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
        "merge": {"role": role, "constituent_ids": None},
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

    ``proposals`` maps ``candidate_id -> surface`` and is persisted where
    discovery says the round's proposals live, so the join under test runs
    against the layout the loop writes rather than one assembled here.
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def written_activity(snapshot: Snapshot, out_dir: Path) -> list[dict[str, str]]:
    rows, unattributed = surface_activity_over_rounds(out_dir)
    write_surface_activity(snapshot, out_dir, rows, unattributed)
    return read_csv(snapshot.path / SURFACE_ACTIVITY_FILENAME)


def written_unattributed(snapshot: Snapshot, out_dir: Path) -> list[dict[str, str]]:
    rows, unattributed = surface_activity_over_rounds(out_dir)
    write_surface_activity(snapshot, out_dir, rows, unattributed)
    return read_csv(snapshot.path / UNATTRIBUTED_FILENAME)


def cell(rows: list[dict[str, str]], round_index: int, surface: str) -> dict[str, str]:
    (match,) = [
        row for row in rows if row["round_index"] == str(round_index) and row["surface"] == surface
    ]
    return match


def round_record(out_dir: Path, round_index: int) -> Any:
    return record_for(out_dir, round_index)


# ---------------------------------------------------------------------------
# The resolver (U4.1)
# ---------------------------------------------------------------------------


class TestResolveSurface:
    def test_a_null_surface_with_a_persisted_proposal_is_backfilled(self, experiment: Path) -> None:
        """The case the running loop actually writes: key present, value null.

        A backfill that fired only on a missing key would never reach this row,
        so it is asserted first and asserted on the loop's own serialization.
        """
        record = ledger_record("r01-c01-s2")
        assert "surface" in record and record["surface"] is None

        write_round(experiment, 1, [record], proposals={"r01-c01-s2": "S2"})

        attribution = resolve_surface(record, round_record(experiment, 1))
        assert attribution.category == "S2"
        assert attribution.source == SURFACE_SOURCE_BACKFILLED

    def test_a_legacy_record_with_no_surface_key_is_backfilled(self, experiment: Path) -> None:
        record = ledger_record("r01-c01-s3", legacy=True)
        assert "surface" not in record

        write_round(experiment, 1, [record], proposals={"r01-c01-s3": "S3"})

        attribution = resolve_surface(record, round_record(experiment, 1))
        assert attribution.category == "S3"
        assert attribution.source == SURFACE_SOURCE_BACKFILLED

    def test_a_record_with_no_proposal_on_disk_is_unresolvable(self, experiment: Path) -> None:
        record = ledger_record("r01-c01-s5")
        write_round(experiment, 1, [record])

        attribution = resolve_surface(record, round_record(experiment, 1))
        assert attribution.category is None
        assert attribution.source == SURFACE_SOURCE_NONE

    def test_the_merged_record_is_its_own_category_and_never_backfilled(
        self, experiment: Path
    ) -> None:
        """KTD7: null is correct for it, so a proposal must not override it."""
        record = ledger_record(MERGED_SUBJECT_ID, decision=DECISION_PROMOTED, role=ROLE_MERGED)
        write_round(experiment, 1, [record], proposals={MERGED_SUBJECT_ID: "S9"})

        attribution = resolve_surface(record, round_record(experiment, 1))
        assert attribution.category == MERGED_CATEGORY
        assert attribution.source == SURFACE_SOURCE_MERGED

    def test_a_recorded_surface_is_taken_verbatim(self, experiment: Path) -> None:
        """A ledger value wins over any proposal; the ledger is the evidence."""
        record = ledger_record("r01-c01-s4", decision=DECISION_ACCEPTED, surface="S4")
        write_round(experiment, 1, [record], proposals={"r01-c01-s4": "S7"})

        attribution = resolve_surface(record, round_record(experiment, 1))
        assert attribution.category == "S4"
        assert attribution.source == SURFACE_SOURCE_LEDGER

    def test_a_proposal_of_another_format_is_not_trusted(self, experiment: Path) -> None:
        record = ledger_record("r01-c01-s2")
        write_round(experiment, 1, [record])
        discovered = round_record(experiment, 1)
        write_json(
            discovered.proposals_dir / "r01-c01-s2" / PROPOSAL_FILENAME,
            {"format": "something-else/v1", "surface": "S2"},
        )

        attribution = resolve_surface(record, discovered)
        assert attribution.category is None
        assert attribution.source == SURFACE_SOURCE_NONE


# ---------------------------------------------------------------------------
# The tables (U4.2)
# ---------------------------------------------------------------------------


class TestSurfaceActivityBackfill:
    def test_a_null_surface_row_is_counted_in_its_surface_and_marked_backfilled(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s2")],
            proposals={"r01-c01-s2": "S2"},
        )

        rows = written_activity(snapshot, experiment)
        assert cell(rows, 1, "S2")["attempted_count"] == "1"
        assert cell(rows, 1, "S2")["surface_source"] == SURFACE_SOURCE_BACKFILLED
        (tally,) = written_unattributed(snapshot, experiment)
        assert tally["unattributed_count"] == "0"
        assert tally["total_rows"] == "1"

    def test_a_legacy_row_is_counted_in_its_surfaces_tallies(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s3", decision=DECISION_PROMOTED, legacy=True)],
            proposals={"r01-c01-s3": "S3"},
        )

        rows = written_activity(snapshot, experiment)
        s3 = cell(rows, 1, "S3")
        assert s3["attempted_count"] == "1"
        assert s3["promoted_count"] == "1"
        assert s3["surface_source"] == SURFACE_SOURCE_BACKFILLED
        assert s3["cumulative_surfaces_attempted"] == "1"
        assert s3["cumulative_surfaces_promoted"] == "1"

    def test_a_row_with_no_proposal_is_counted_unattributed(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(experiment, 1, [ledger_record("r01-c01-s5")])

        rows = written_activity(snapshot, experiment)
        assert {row["attempted_count"] for row in rows} == {"0"}
        assert cell(rows, 1, "S5")["surface_source"] == SURFACE_SOURCE_NONE
        (tally,) = written_unattributed(snapshot, experiment)
        assert tally["unattributed_count"] == "1"
        assert tally["total_rows"] == "1"

    def test_the_merged_record_is_its_own_row_and_never_unattributed(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(
            experiment,
            1,
            [
                ledger_record("r01-c01-s4", decision=DECISION_ACCEPTED, surface="S4"),
                ledger_record(MERGED_SUBJECT_ID, decision=DECISION_PROMOTED, role=ROLE_MERGED),
            ],
        )

        rows = written_activity(snapshot, experiment)
        merged = cell(rows, 1, MERGED_CATEGORY)
        assert merged["attempted_count"] == "1"
        assert merged["promoted_count"] == "1"
        assert merged["surface_source"] == SURFACE_SOURCE_MERGED
        # A category of its own, so it never inflates the nine-surface counts.
        assert merged["cumulative_surfaces_attempted"] == "1"
        (tally,) = written_unattributed(snapshot, experiment)
        assert tally["unattributed_count"] == "0"
        assert tally["total_rows"] == "2"

    def test_a_recorded_surface_is_written_verbatim_and_marked_ledger(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_round(
            experiment,
            1,
            [ledger_record("r01-c01-s4", decision=DECISION_PROMOTED, surface="S4")],
        )

        rows = written_activity(snapshot, experiment)
        s4 = cell(rows, 1, "S4")
        assert s4["attempted_count"] == "1"
        assert s4["promoted_count"] == "1"
        assert s4["surface_source"] == SURFACE_SOURCE_LEDGER

    def test_a_cell_mixing_a_recorded_and_a_recovered_row_reports_backfilled(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """The weaker claim wins: part of this cell was reconstructed."""
        write_round(
            experiment,
            1,
            [
                ledger_record("r01-c01-s4", decision=DECISION_ACCEPTED, surface="S4"),
                ledger_record("r01-c02-s4"),
            ],
            proposals={"r01-c02-s4": "S4"},
        )

        rows = written_activity(snapshot, experiment)
        assert cell(rows, 1, "S4")["attempted_count"] == "2"
        assert cell(rows, 1, "S4")["surface_source"] == SURFACE_SOURCE_BACKFILLED

    def test_the_table_declares_the_surface_source_column(self) -> None:
        assert "surface_source" in SURFACE_ACTIVITY_FIELDNAMES


# ---------------------------------------------------------------------------
# The CLI warning names only genuinely unresolvable rows
# ---------------------------------------------------------------------------


class TestUnattributedWarning:
    def test_backfilled_and_merged_rows_do_not_trigger_the_warning(
        self, experiment: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_round(
            experiment,
            1,
            [
                ledger_record("r01-c01-s2"),
                ledger_record(MERGED_SUBJECT_ID, decision=DECISION_PROMOTED, role=ROLE_MERGED),
            ],
            proposals={"r01-c01-s2": "S2"},
        )

        assert main([str(experiment)]) == 0
        assert "no resolved surface" not in capsys.readouterr().err

    def test_an_unresolvable_row_is_warned_about_without_blaming_the_loader(
        self, experiment: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_round(experiment, 1, [ledger_record("r01-c01-s5")])

        assert main([str(experiment)]) == 0
        stderr = capsys.readouterr().err
        assert "round 1" in stderr
        assert "proposal" in stderr
        assert "loader rejections" not in stderr


# ---------------------------------------------------------------------------
# The whole pass over a tree that mixes all four kinds of row
# ---------------------------------------------------------------------------


def test_a_legacy_experiment_regains_its_surface_history(
    experiment: Path, snapshot: Snapshot
) -> None:
    """Every recoverable row is recovered; only the truly unresolvable is not."""
    write_round(
        experiment,
        1,
        [
            ledger_record("r01-c01-s2", legacy=True),
            ledger_record("r01-c02-s7"),
            ledger_record("r01-c03-s1"),
            ledger_record(MERGED_SUBJECT_ID, decision=DECISION_PROMOTED, role=ROLE_MERGED),
        ],
        proposals={"r01-c01-s2": "S2", "r01-c02-s7": "S7"},
    )

    rows = written_activity(snapshot, experiment)
    attributed = {
        row["surface"]: row["surface_source"] for row in rows if row["attempted_count"] != "0"
    }
    assert attributed == {
        "S2": SURFACE_SOURCE_BACKFILLED,
        "S7": SURFACE_SOURCE_BACKFILLED,
        MERGED_CATEGORY: SURFACE_SOURCE_MERGED,
    }
    (tally,) = written_unattributed(snapshot, experiment)
    assert tally["unattributed_count"] == "1"
    assert tally["total_rows"] == "4"
    # The round the loop actually wrote is still discovered as one round: the
    # backfill reads the tree, it never rewrites it.
    assert [record.round_index for record in discover_rounds(experiment).rounds] == [1]
