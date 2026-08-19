"""Tests for U2: completeness reaching every analysis output (R2, KTD5, KTD9).

Named for its largest consumer, but the claim under test spans all four
aggregations: a partial round, a partial test set, or a partial bundle is
EMITTED, FLAGGED, and never presented as complete -- and never dropped, which
is the failure mode the PR review found (a truncated round looked exactly like
a finished one once it reached a CSV, and a bundle the diff could not pair
simply vanished).

Three things these tests pin that a looser reading would let slip:

* ``unknown`` is its own serialized value. A tree with no resolvable config
  cannot know how many runs a round planned, and a CSV that rendered that as
  ``false`` would report an unmeasurable round as a failed one. The assertions
  therefore read the written CSV text, not just the row objects -- the
  serialization is the part a reader actually sees.
* A completed no-op round (marker present, nothing mined) is ``round_complete``
  true with zero runs, not missing data.
* The evaluation phase's verdict comes from the shared inventory, so a test set
  with run directories but no ``eval_summary.json`` entry is ``unknown`` rather
  than absent from the table -- which is only possible once the phase stops
  enumerating test sets out of the summary itself.

Trees are fabricated through ``tests.experiment.test_rounds``' helpers (which
build them with the loop's own layout functions), plus one class grounded on
the mock-LM end-to-end run's real output, where the metric numbers are real
rather than stubbed.

``load_round`` is stubbed for the fabricated trees on purpose: rehydrating a
round needs sha-verified traces and real completions, and none of that is what
this unit changed. The completeness numbers stay real throughout -- they come
from the persisted manifest and ``instances.jsonl`` through discovery, not from
the stub.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shrlm.experiment.analysis_io import (
    TRISTATE_FALSE,
    TRISTATE_TRUE,
    TRISTATE_UNKNOWN,
    Snapshot,
    allocate_snapshot,
)
from shrlm.experiment.collapse_and_attribution import (
    EVALUATION_FILENAME,
    OPTIMIZATION_FILENAME,
    collapse_and_attribution_evaluation,
    collapse_and_attribution_optimization,
    write_collapse_and_attribution_evaluation,
    write_collapse_and_attribution_optimization,
)
from shrlm.experiment.config import load_config
from shrlm.experiment.evaluation import EVAL_DIR, EVAL_SUMMARY_FILENAME
from shrlm.experiment.incumbent_quality import (
    INCUMBENT_QUALITY_FILENAME,
    incumbent_quality_over_rounds,
    write_incumbent_quality,
)
from shrlm.experiment.pattern_frequency_diff import (
    BUNDLES_FILENAME,
    run_pattern_frequency_diff,
)
from shrlm.experiment.rounds import discover_rounds
from shrlm.experiment.surface_activity import (
    SURFACE_ACTIVITY_FILENAME,
    surface_activity_over_rounds,
    write_surface_activity,
)
from shrlm.optimization.bundle import BUNDLE_FILENAME
from shrlm.optimization.costs import OUTCOME_COMPLETED, OUTCOME_OVER_BUDGET
from tests.experiment.test_rounds import (
    complete_round,
    eval_set_payload,
    write_config,
    write_eval_set,
    write_eval_summary,
    write_evidence,
    write_json,
    write_ledger,
    write_mining_round,
    write_round_marker,
)
from tests.experiment.test_smoke_mock import (
    SmokeRun,
    smoke,  # noqa: F401 -- the module-scoped fixture
)

FROZEN = datetime(2026, 8, 18, 16, 48, 0, tzinfo=UTC)

# The fabricated trees run on the full profile, whose three stage repetition
# counts differ (m=2, v=4, eval_repetitions=3), so a row that quietly used the
# wrong stage's multiplier shows up in the arithmetic.
PROFILE = "full"


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An experiment directory with a persisted identity and nothing else."""
    out_dir = tmp_path / "experiment"
    write_config(out_dir, PROFILE)
    return out_dir


@pytest.fixture
def snapshot(experiment: Path) -> Snapshot:
    return allocate_snapshot(experiment, now=FROZEN)


@pytest.fixture
def no_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub round rehydration; completeness comes from the manifest regardless.

    Every fabricated tree here persists a manifest and an instance list -- the
    two files the expected/present run counts are computed from -- but not the
    sha-verified traces ``load_round`` insists on. The stub keeps the metric
    columns empty and leaves the completeness columns entirely real, which is
    the split this unit is about.
    """

    def load_nothing(parent: Path | str, round_index: int) -> tuple[list, list, dict, list]:
        return [], [], {}, []

    monkeypatch.setattr("shrlm.experiment.collapse_and_attribution.load_round", load_nothing)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def written_optimization(snapshot: Snapshot, out_dir: Path) -> list[dict[str, str]]:
    write_collapse_and_attribution_optimization(
        snapshot, out_dir, collapse_and_attribution_optimization(out_dir)
    )
    return read_csv(snapshot.path / OPTIMIZATION_FILENAME)


def written_evaluation(snapshot: Snapshot, out_dir: Path) -> list[dict[str, str]]:
    write_collapse_and_attribution_evaluation(
        snapshot, out_dir, collapse_and_attribution_evaluation(out_dir)
    )
    return read_csv(snapshot.path / EVALUATION_FILENAME)


def write_bundle(
    path: Path, round_index: int, *, support: int, n_runs: int = 10, environment: str = "graphwalks"
) -> Path:
    """A ``bundle.json`` carrying one mined pattern, in mining's own shape."""
    write_json(
        path,
        {
            "config": {"round_index": round_index, "verifier_config": {"environment": environment}},
            "totals": {"n_runs": n_runs},
            "patterns": [
                {
                    "signature": {
                        "verifier_cause": "wrong_answer",
                        "failing_level": "child",
                        "causal_status": "causal",
                        "agent_mechanism": "routed_whole_input",
                    },
                    "instance_support": support,
                    "grounded_fraction": 1.0,
                    "below_support_floor": False,
                }
            ],
        },
    )
    return path


def bundle_path_for(out_dir: Path, round_index: int) -> Path:
    """Where the loop writes a round's bundle, asked of discovery rather than built."""
    (record,) = [
        record for record in discover_rounds(out_dir).rounds if record.round_index == round_index
    ]
    return record.mining_round_path / BUNDLE_FILENAME


# ---------------------------------------------------------------------------
# Optimization phase (U2.1)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_traces")
class TestOptimizationCompleteness:
    def test_a_partial_mining_manifest_is_emitted_flagged_with_its_missing_count(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """A budget stop truncates the round; the row says so instead of hiding it."""
        config = load_config(PROFILE)
        write_mining_round(experiment, 1, n_instances=3, attempts=config.loop.m, n_runs=2)
        write_evidence(experiment, 1)

        (row,) = written_optimization(snapshot, experiment)
        assert row["round_index"] == "1"
        assert row["n_expected_runs"] == str(3 * config.loop.m)
        assert row["n_missing_runs"] == str(3 * config.loop.m - 2)
        assert row["runs_complete"] == TRISTATE_FALSE
        assert row["round_complete"] == TRISTATE_FALSE

    def test_a_fully_complete_round_carries_true_flags_and_no_missing_runs(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)

        (row,) = written_optimization(snapshot, experiment)
        assert row["n_missing_runs"] == "0"
        assert row["runs_complete"] == TRISTATE_TRUE
        assert row["evidence_complete"] == TRISTATE_TRUE
        assert row["round_complete"] == TRISTATE_TRUE

    def test_an_unknown_expected_count_is_serialized_as_unknown_never_as_false(
        self, tmp_path: Path
    ) -> None:
        """KTD9: a tree that cannot say what it planned is unmeasurable, not failed."""
        legacy = tmp_path / "legacy"
        write_mining_round(legacy, 1, n_instances=3, attempts=2, n_runs=2)
        write_evidence(legacy, 1)
        unknown_snapshot = allocate_snapshot(legacy, now=FROZEN)

        (row,) = written_optimization(unknown_snapshot, legacy)
        assert row["n_expected_runs"] == ""
        assert row["n_missing_runs"] == ""
        assert row["runs_complete"] == TRISTATE_UNKNOWN
        assert row["runs_complete"] != TRISTATE_FALSE

    def test_an_incomplete_round_and_an_unmeasurable_one_are_distinguishable(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """The two rows must not collapse onto the same serialization."""
        config = load_config(PROFILE)
        write_mining_round(experiment, 1, n_instances=3, attempts=config.loop.m, n_runs=2)
        known = written_optimization(snapshot, experiment)[0]["runs_complete"]

        legacy = experiment.parent / "legacy"
        write_mining_round(legacy, 1, n_instances=3, attempts=2, n_runs=2)
        unknown = written_optimization(allocate_snapshot(legacy, now=FROZEN), legacy)[0][
            "runs_complete"
        ]

        assert known == TRISTATE_FALSE
        assert unknown == TRISTATE_UNKNOWN
        assert known != unknown

    def test_a_completed_no_op_round_is_complete_with_no_runs_not_missing_data(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """Marker present, nothing mined: a real outcome, and it stays in the table."""
        write_round_marker(experiment, 1, has_ledger=False)

        (row,) = written_optimization(snapshot, experiment)
        assert row["round_complete"] == TRISTATE_TRUE
        assert row["n_runs"] == "0"
        assert row["collapse_rate"] == ""

    def test_an_evidence_marker_disagreeing_with_its_audit_files_is_flagged(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, recorded_records=5)

        (row,) = written_optimization(snapshot, experiment)
        assert row["evidence_complete"] == TRISTATE_FALSE
        assert row["round_complete"] == TRISTATE_TRUE

    def test_every_discovered_round_is_emitted_in_order(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)
        write_round_marker(experiment, 2, has_ledger=False)
        complete_round(experiment, 3)

        rows = written_optimization(snapshot, experiment)
        assert [row["round_index"] for row in rows] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Evaluation phase (U2.2): the verdict comes from the shared inventory
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_traces")
class TestEvaluationCompleteness:
    def test_a_set_with_skipped_runs_is_flagged_not_dropped(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_eval_set(experiment, "b1", "graphwalks_short", n_runs=4)
        write_eval_summary(
            experiment,
            {
                "b1": {
                    "test_sets": {
                        "graphwalks_short": eval_set_payload(skipped=("inst01__a02", "inst01__a03"))
                    }
                }
            },
        )

        (row,) = written_evaluation(snapshot, experiment)
        assert row["condition_id"] == "b1"
        assert row["test_set_id"] == "graphwalks_short"
        assert row["outcome"] == OUTCOME_COMPLETED
        assert row["n_skipped"] == "2"
        assert row["complete"] == TRISTATE_FALSE

    def test_an_over_budget_set_is_flagged_incomplete(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_summary(
            experiment,
            {
                "b1": {
                    "test_sets": {"graphwalks_short": eval_set_payload(outcome=OUTCOME_OVER_BUDGET)}
                }
            },
        )

        (row,) = written_evaluation(snapshot, experiment)
        assert row["outcome"] == OUTCOME_OVER_BUDGET
        assert row["complete"] == TRISTATE_FALSE

    def test_a_completed_set_with_no_skips_is_complete(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        (row,) = written_evaluation(snapshot, experiment)
        assert row["complete"] == TRISTATE_TRUE
        assert row["n_skipped"] == "0"
        assert row["environment"] == "graphwalks"
        assert row["length"] == "short"

    def test_a_set_on_disk_but_absent_from_the_summary_is_unknown_and_still_emitted(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """Only possible once the phase reads the inventory, not the summary's grid."""
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_set(experiment, "b1", "oolong_pairs_long")
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        by_set = {row["test_set_id"]: row for row in written_evaluation(snapshot, experiment)}
        assert set(by_set) == {"graphwalks_short", "oolong_pairs_long"}
        orphan = by_set["oolong_pairs_long"]
        assert orphan["outcome"] == ""
        assert orphan["complete"] == TRISTATE_UNKNOWN
        assert orphan["complete"] != TRISTATE_FALSE

    def test_a_set_whose_runs_never_landed_is_emitted_with_explicit_nulls(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """The spend breaker tripped before the set ran: nulls, not a missing row."""
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_summary(
            experiment,
            {
                "b1": {
                    "test_sets": {"graphwalks_short": eval_set_payload(outcome=OUTCOME_OVER_BUDGET)}
                },
                "sh_rlm": {
                    "test_sets": {"graphwalks_short": eval_set_payload(outcome=OUTCOME_OVER_BUDGET)}
                },
            },
        )

        by_condition = {
            row["condition_id"]: row for row in written_evaluation(snapshot, experiment)
        }
        never_ran = by_condition["sh_rlm"]
        assert never_ran["n_runs"] == "0"
        assert never_ran["collapse_rate"] == ""
        assert never_ran["complete"] == TRISTATE_FALSE


# ---------------------------------------------------------------------------
# The two round-series tables (U2.3)
# ---------------------------------------------------------------------------


class TestSurfaceActivityCompleteness:
    def test_every_row_of_a_complete_round_carries_true_flags(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)

        rows, unattributed = surface_activity_over_rounds(experiment)
        write_surface_activity(snapshot, experiment, rows, unattributed)

        written = read_csv(snapshot.path / SURFACE_ACTIVITY_FILENAME)
        assert written
        assert {row["round_complete"] for row in written} == {TRISTATE_TRUE}
        assert {row["runs_complete"] for row in written} == {TRISTATE_TRUE}

    def test_a_round_with_a_ledger_but_no_marker_is_flagged_incomplete(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        config = load_config(PROFILE)
        write_mining_round(experiment, 1, n_instances=3, attempts=config.loop.m, n_runs=2)
        write_ledger(experiment, 1)

        rows, unattributed = surface_activity_over_rounds(experiment)
        write_surface_activity(snapshot, experiment, rows, unattributed)

        written = read_csv(snapshot.path / SURFACE_ACTIVITY_FILENAME)
        assert {row["round_complete"] for row in written} == {TRISTATE_FALSE}
        assert {row["runs_complete"] for row in written} == {TRISTATE_FALSE}

    def test_an_unmeasurable_round_reports_unknown_runs_not_false(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy"
        write_mining_round(legacy, 1, n_instances=3, attempts=2, n_runs=2)
        write_ledger(legacy, 1)
        write_round_marker(legacy, 1, has_ledger=True)
        legacy_snapshot = allocate_snapshot(legacy, now=FROZEN)

        rows, unattributed = surface_activity_over_rounds(legacy)
        write_surface_activity(legacy_snapshot, legacy, rows, unattributed)

        written = read_csv(legacy_snapshot.path / SURFACE_ACTIVITY_FILENAME)
        assert {row["runs_complete"] for row in written} == {TRISTATE_UNKNOWN}
        assert {row["round_complete"] for row in written} == {TRISTATE_TRUE}


class TestIncumbentQualityCompleteness:
    def test_the_round_row_carries_the_rounds_completeness(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)
        write_ledger(experiment, 2, promoted=False)
        write_mining_round(experiment, 2, n_instances=3, n_runs=1)

        rows = incumbent_quality_over_rounds(experiment)
        write_incumbent_quality(snapshot, experiment, rows, [])

        written = {
            row["round_index"]: row for row in read_csv(snapshot.path / INCUMBENT_QUALITY_FILENAME)
        }
        assert written["1"]["round_complete"] == TRISTATE_TRUE
        assert written["1"]["runs_complete"] == TRISTATE_TRUE
        assert written["2"]["round_complete"] == TRISTATE_FALSE
        assert written["2"]["runs_complete"] == TRISTATE_FALSE


# ---------------------------------------------------------------------------
# The bundle diff (U2.4): flag, don't drop
# ---------------------------------------------------------------------------


class TestPatternFrequencyDiffCompleteness:
    def test_a_partial_bundle_is_emitted_with_a_reason_and_never_compared(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        write_evidence(experiment, 2, n_records=3, recorded_records=5)
        first = write_bundle(bundle_path_for(experiment, 1), 1, support=5)
        second = write_bundle(bundle_path_for(experiment, 2), 2, support=2)

        run_pattern_frequency_diff([first, second], snapshot)

        rows = {row["round_index"]: row for row in read_csv(snapshot.path / BUNDLES_FILENAME)}
        assert set(rows) == {"1", "2"}
        assert rows["1"]["evidence_complete"] == TRISTATE_TRUE
        assert rows["1"]["included"] == TRISTATE_TRUE
        assert rows["2"]["evidence_complete"] == TRISTATE_FALSE
        assert rows["2"]["included"] == TRISTATE_FALSE
        assert rows["2"]["exclusion_reason"]
        assert list(snapshot.path.glob("pattern_frequency_diff_round_*_vs_*.csv")) == []

    def test_two_evidence_complete_bundles_are_compared_as_before(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        first = write_bundle(bundle_path_for(experiment, 1), 1, support=5)
        second = write_bundle(bundle_path_for(experiment, 2), 2, support=2)

        written = run_pattern_frequency_diff([first, second], snapshot)

        diff_path = snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv"
        assert diff_path in written
        (diff_row,) = read_csv(diff_path)
        assert diff_row["status"] == "persisted_improved"
        rows = read_csv(snapshot.path / BUNDLES_FILENAME)
        assert {row["included"] for row in rows} == {TRISTATE_TRUE}

    def test_a_bundle_outside_the_experiment_tree_is_unknown_and_still_compared(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """Unknown never collapses into false, so it is never a reason to exclude."""
        first = write_bundle(tmp_path / "before.json", 1, support=5)
        second = write_bundle(tmp_path / "after.json", 2, support=2)

        run_pattern_frequency_diff([first, second], snapshot)

        rows = read_csv(snapshot.path / BUNDLES_FILENAME)
        assert {row["evidence_complete"] for row in rows} == {TRISTATE_UNKNOWN}
        assert {row["included"] for row in rows} == {TRISTATE_TRUE}
        assert {row["round_index"] for row in rows} == {""}
        assert (snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv").exists()


# ---------------------------------------------------------------------------
# The grounding scenario: real numbers, from the tree the loop wrote
# ---------------------------------------------------------------------------


class TestAgainstTheMockSmokeTree:
    """Completeness over the mock-LM end-to-end run's own output.

    The fabricated trees above stub trace rehydration; this class does not, so
    the metric columns are the real pre-change numbers and the flags sit beside
    them on the same rows.

    The ``smoke`` parameter is the imported module-scoped fixture, which ruff
    reads as a redefinition of the imported name (hence ``F811``).
    """

    def test_the_completed_round_is_flagged_complete_beside_real_metrics(
        self,
        smoke: SmokeRun,  # noqa: F811
    ) -> None:
        (row,) = collapse_and_attribution_optimization(smoke.out_dir)

        assert row.round_index == 1
        assert row.completeness.round_complete is True
        assert row.completeness.evidence_complete is True
        assert row.completeness.runs_complete is True
        assert row.completeness.n_missing_runs == 0
        assert row.metrics.n_runs == row.completeness.n_expected_runs
        assert row.metrics.n_runs > 0
        assert row.metrics.n_walked + row.metrics.n_unwalkable == row.metrics.n_runs

    def test_every_summarized_test_set_is_emitted_complete_with_its_outcome(
        self,
        smoke: SmokeRun,  # noqa: F811
    ) -> None:
        summary = json.loads((smoke.out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_text())
        expected = {
            (condition_id, set_id)
            for condition_id, condition in summary["conditions"].items()
            for set_id in condition["test_sets"]
        }

        rows = collapse_and_attribution_evaluation(smoke.out_dir)

        assert {(row.condition_id, row.test_set_id) for row in rows} == expected
        for row in rows:
            assert row.completeness.outcome == OUTCOME_COMPLETED
            assert row.completeness.n_skipped == 0
            assert row.completeness.complete is True
            assert row.metrics.n_runs > 0


def test_no_analysis_row_type_omits_its_completeness() -> None:
    """Every emitted table declares its completeness columns (R2's surface)."""
    from shrlm.experiment.collapse_and_attribution import (
        EVALUATION_FIELDNAMES,
        OPTIMIZATION_FIELDNAMES,
    )
    from shrlm.experiment.incumbent_quality import INCUMBENT_QUALITY_FIELDNAMES
    from shrlm.experiment.pattern_frequency_diff import BUNDLE_FIELDNAMES
    from shrlm.experiment.surface_activity import SURFACE_ACTIVITY_FIELDNAMES

    assert {"round_complete", "runs_complete"} <= set(OPTIMIZATION_FIELDNAMES)
    assert {"n_expected_runs", "n_missing_runs", "evidence_complete"} <= set(
        OPTIMIZATION_FIELDNAMES
    )
    assert {"outcome", "n_skipped", "complete"} <= set(EVALUATION_FIELDNAMES)
    assert {"round_complete", "runs_complete"} <= set(SURFACE_ACTIVITY_FIELDNAMES)
    assert {"round_complete", "runs_complete"} <= set(INCUMBENT_QUALITY_FIELDNAMES)
    assert {"evidence_complete", "included", "exclusion_reason"} <= set(BUNDLE_FIELDNAMES)
