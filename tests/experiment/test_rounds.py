"""Tests for U1's shared round discovery: what exists, and how complete it is.

Two kinds of tree are exercised, deliberately:

* The tree the mock-LM end-to-end run actually produces
  (``tests.experiment.test_smoke_mock``'s module-scoped ``smoke`` fixture, run
  once more here). That scenario is the one that grounds the whole premise --
  a fabricated tree only ever tests the test author's model of the layout, so
  discovery is pinned against bytes the real loop wrote at least once.
* Fabricated trees for the cases the happy path cannot produce on demand: a
  truncated round, a round with no marker, a completed no-op round, a
  malformed directory name, an experiment with no config. These are built
  through the loop's own layout functions and payload builders
  (``experiment_round_dir``, ``round_dir``, ``RoundOutcome.to_payload``,
  ``check_identity``), never through hand-written path strings, so a layout
  change moves the fixtures and the module together.

The counting assertions carry the plan's sharpest correctness point (KTD6):
the expected run count comes from ``instances.jsonl`` times the configured
per-stage repetition count, and is ``None`` -- explicitly unknown -- when the
config cannot be resolved. Inferring it from the observed run ids would let a
round truncated at attempt 2 of 3 report itself complete, which is exactly the
false completeness this module exists to remove, so the unknown case asserts
``is None`` and ``is not False`` rather than merely a falsy value.
"""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.config import ExperimentConfig, load_config
from shrlm.experiment.evaluation import EVAL_DIR, EVAL_SUMMARY_FILENAME
from shrlm.experiment.orchestrator import (
    CONFIG_FILENAME,
    EVIDENCE_MARKER_FILENAME,
    EVIDENCE_MARKER_FORMAT,
    MINING_DIR,
    OPT_DIR,
    PROPOSALS_DIR,
    ROUND_MARKER_FILENAME,
    VALIDATION_DIR,
    RoundOutcome,
    check_identity,
    experiment_round_dir,
)
from shrlm.experiment.rounds import (
    STAGE_EVALUATION,
    STAGE_MINING,
    STAGE_VALIDATION,
    discover_rounds,
    iter_promotion_rounds,
    stage_repetitions,
)
from shrlm.optimization.bundle import (
    ATTRIBUTIONS_FILENAME,
    BUNDLE_FILENAME,
    RECORDS_FILENAME,
    round_dir,
)
from shrlm.optimization.costs import OUTCOME_COMPLETED, OUTCOME_OVER_BUDGET
from shrlm.optimization.driver import INSTANCES_FILE, MANIFEST_FILE, run_id_for
from shrlm.optimization.validation import (
    DECISION_FORMAT,
    EVAL_ROUND_INDEX,
    LEDGER_RECORD_FORMAT,
    PROMOTIONS_FILENAME,
)
from tests.experiment.test_smoke_mock import (
    SmokeRun,
    smoke,  # noqa: F401 -- the module-scoped fixture
)

# The fabricated trees run on the full profile (m=2, v=4, eval_repetitions=3),
# so a per-stage repetition count that silently used the wrong stage's number
# is visible in the arithmetic; the smoke profile's counts are all 1.
PROFILE = "full"


# ---------------------------------------------------------------------------
# Fabrication: one experiment tree, built through the loop's own layout code
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def instance_ids(n_instances: int) -> list[str]:
    return [f"inst{index:02d}" for index in range(n_instances)]


def write_mining_round(
    out_dir: Path,
    round_index: int,
    *,
    n_instances: int = 3,
    n_runs: int | None = None,
    attempts: int = 2,
) -> Path:
    """A mining stage the driver could have written: instances + manifest.

    ``n_runs`` defaults to the full plan (``n_instances * attempts``); a
    smaller number is the truncated round a tripped budget breaker leaves.
    """
    mining_round_path = round_dir(
        experiment_round_dir(out_dir, round_index) / MINING_DIR, round_index
    )
    ids = instance_ids(n_instances)
    write_jsonl(mining_round_path / INSTANCES_FILE, [{"id": id_} for id_ in ids])
    planned = [
        {"run_id": run_id_for(id_, attempt), "cost": 0.001}
        for id_ in ids
        for attempt in range(1, attempts + 1)
    ]
    write_jsonl(
        mining_round_path / MANIFEST_FILE, planned[: n_runs if n_runs is not None else len(planned)]
    )
    return mining_round_path


def write_evidence(
    out_dir: Path,
    round_index: int,
    *,
    n_records: int = 3,
    recorded_records: int | None = None,
    recorded_round: int | None = None,
    bundle_id: str | None = "",
) -> None:
    """The evidence triplet plus the orchestrator's completion marker.

    ``recorded_records`` diverging from ``n_records`` is one crash the marker
    exists to catch: a marker claiming more records than the audit file holds.
    ``recorded_round`` diverging from ``round_index`` is the other: a marker
    that certifies some other round's evidence. ``bundle_id`` defaults to this
    round's own id and can be set to ``None`` to omit the key entirely, which
    the loop's writer never does.
    """
    mining_round_path = round_dir(
        experiment_round_dir(out_dir, round_index) / MINING_DIR, round_index
    )
    write_jsonl(
        mining_round_path / RECORDS_FILENAME, [{"run_id": f"r{i}"} for i in range(n_records)]
    )
    write_jsonl(
        mining_round_path / ATTRIBUTIONS_FILENAME, [{"run_id": f"r{i}"} for i in range(n_records)]
    )
    payload: dict[str, Any] = {
        "format": EVIDENCE_MARKER_FORMAT,
        "round": round_index if recorded_round is None else recorded_round,
        "n_records": n_records if recorded_records is None else recorded_records,
        "n_attributions": n_records,
    }
    if bundle_id is not None:
        payload["bundle_id"] = f"bundle-{round_index}" if bundle_id == "" else bundle_id
    write_json(mining_round_path / EVIDENCE_MARKER_FILENAME, payload)


def write_bundle_identity(out_dir: Path, round_index: int, *, bundle_id: str) -> Path:
    """A ``bundle.json`` that names itself, as ``write_bundle`` in mining does.

    The evidence marker records the id it read out of this file, so the pair is
    what a re-check compares. Kept separate from ``write_bundle`` (which the
    pattern-frequency tests use for its pattern list) because the identity is
    all these cases care about.
    """
    mining_round_path = round_dir(
        experiment_round_dir(out_dir, round_index) / MINING_DIR, round_index
    )
    bundle_path = mining_round_path / BUNDLE_FILENAME
    write_json(bundle_path, {"bundle_id": bundle_id, "config": {}, "patterns": []})
    return bundle_path


def write_ledger(out_dir: Path, round_index: int, *, promoted: bool = True) -> Path:
    """One validation round's promotion ledger and decision, in the loop's layout."""
    validation_round_path = round_dir(
        experiment_round_dir(out_dir, round_index) / VALIDATION_DIR, round_index
    )
    write_jsonl(
        validation_round_path / PROMOTIONS_FILENAME,
        [
            {
                "format": LEDGER_RECORD_FORMAT,
                "subject_id": f"r{round_index:02d}-c01-s4",
                "decision": "promoted" if promoted else "rejected",
                "surface": "S4",
            }
        ],
    )
    write_json(
        validation_round_path / "decision.json",
        {"format": DECISION_FORMAT, "round": round_index, "promoted": promoted},
    )
    return validation_round_path


def write_round_marker(
    out_dir: Path,
    round_index: int,
    *,
    has_ledger: bool = True,
    promoted: bool = False,
    recorded_round: int | None = None,
) -> None:
    """The round's completion marker, built by the loop's own payload builder."""
    payload = RoundOutcome(
        round_index=round_index if recorded_round is None else recorded_round,
        promoted=promoted,
        promoted_harness_hash=None,
        has_ledger=has_ledger,
    ).to_payload()
    write_json(experiment_round_dir(out_dir, round_index) / ROUND_MARKER_FILENAME, payload)


def write_config(out_dir: Path, profile: str = PROFILE) -> ExperimentConfig:
    """Persist the experiment's identity exactly as the orchestrator does."""
    config = load_config(profile)
    check_identity(config, out_dir)
    return config


def complete_round(out_dir: Path, round_index: int, *, n_instances: int = 3) -> None:
    """A whole round as a finished loop leaves it: runs, evidence, ledger, marker."""
    write_mining_round(out_dir, round_index, n_instances=n_instances)
    write_evidence(out_dir, round_index)
    write_ledger(out_dir, round_index)
    write_round_marker(out_dir, round_index, has_ledger=True)


def write_eval_set(
    out_dir: Path,
    condition_id: str,
    set_id: str,
    *,
    n_instances: int = 2,
    attempts: int = 3,
    n_runs: int | None = None,
) -> Path:
    """One condition x test set round directory, in ``evaluation.py``'s layout."""
    set_path = out_dir / EVAL_DIR / condition_id / set_id
    eval_round_path = round_dir(set_path, EVAL_ROUND_INDEX)
    ids = instance_ids(n_instances)
    write_jsonl(eval_round_path / INSTANCES_FILE, [{"id": id_} for id_ in ids])
    planned = [
        {"run_id": run_id_for(id_, attempt), "cost": 0.001}
        for id_ in ids
        for attempt in range(1, attempts + 1)
    ]
    write_jsonl(
        eval_round_path / MANIFEST_FILE, planned[: n_runs if n_runs is not None else len(planned)]
    )
    return set_path


def write_eval_summary(out_dir: Path, conditions: dict[str, Any]) -> None:
    write_json(out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME, {"conditions": conditions})


def eval_set_payload(
    environment: str = "graphwalks",
    length: str = "short",
    *,
    outcome: str = OUTCOME_COMPLETED,
    skipped: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "environment": environment,
        "length": length,
        "outcome": outcome,
        "skipped_run_ids": list(skipped),
    }


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An experiment directory with a persisted identity and nothing else."""
    out_dir = tmp_path / "experiment"
    write_config(out_dir)
    return out_dir


def indices(out_dir: Path) -> list[int]:
    return [record.round_index for record in discover_rounds(out_dir).rounds]


def record_for(out_dir: Path, round_index: int) -> Any:
    matches = [
        record for record in discover_rounds(out_dir).rounds if record.round_index == round_index
    ]
    assert len(matches) == 1, f"round {round_index} not discovered"
    return matches[0]


# ---------------------------------------------------------------------------
# Which rounds exist
# ---------------------------------------------------------------------------


class TestRoundDiscovery:
    def test_rounds_one_through_three_are_discovered_in_order_at_the_loops_paths(
        self, experiment: Path
    ):
        for round_index in (1, 2, 3):
            complete_round(experiment, round_index)

        inventory = discover_rounds(experiment)
        assert [record.round_index for record in inventory.rounds] == [1, 2, 3]
        for record in inventory.rounds:
            expected_round_path = experiment_round_dir(experiment, record.round_index)
            assert record.round_path == expected_round_path
            assert record.mining_parent == expected_round_path / MINING_DIR
            assert record.mining_round_path == round_dir(
                expected_round_path / MINING_DIR, record.round_index
            )
            assert record.validation_round_path == round_dir(
                expected_round_path / VALIDATION_DIR, record.round_index
            )
            assert record.proposals_dir == expected_round_path / PROPOSALS_DIR
            assert record.round_complete is True
            assert record.has_ledger is True
            assert record.has_manifest is True

    def test_non_contiguous_rounds_are_both_returned_and_no_round_is_invented(
        self, experiment: Path
    ):
        complete_round(experiment, 1)
        complete_round(experiment, 3)

        assert indices(experiment) == [1, 3]

    def test_a_round_with_a_manifest_but_no_marker_is_discovered_incomplete(self, experiment: Path):
        write_mining_round(experiment, 1)

        record = record_for(experiment, 1)
        assert record.round_complete is False
        assert record.has_manifest is True

    def test_a_completed_no_op_round_is_discovered_with_has_ledger_false(self, experiment: Path):
        """A round that completed without writing a ledger is a real outcome."""
        write_round_marker(experiment, 1, has_ledger=False)

        record = record_for(experiment, 1)
        assert record.round_complete is True
        assert record.has_ledger is False
        assert record.has_manifest is False

    def test_a_marker_recording_a_different_round_index_is_not_complete(self, experiment: Path):
        complete_round(experiment, 1)
        write_round_marker(experiment, 1, recorded_round=7)

        assert record_for(experiment, 1).round_complete is False

    def test_a_malformed_round_directory_name_is_skipped_without_crashing(self, experiment: Path):
        complete_round(experiment, 1)
        (experiment / OPT_DIR / "round_x").mkdir(parents=True)
        (experiment / OPT_DIR / "round_").mkdir(parents=True)

        assert indices(experiment) == [1]

    def test_a_name_the_loop_would_never_write_is_not_reported_as_a_round(self, experiment: Path):
        """``round_2`` is not ``round_02``; reporting it would invent a path."""
        complete_round(experiment, 1)
        (experiment / OPT_DIR / "round_2").mkdir(parents=True)

        assert indices(experiment) == [1]

    def test_an_experiment_with_no_optimization_directory_is_empty_not_an_error(
        self, experiment: Path
    ):
        inventory = discover_rounds(experiment)
        assert inventory.rounds == ()
        assert inventory.eval_sets == ()


# ---------------------------------------------------------------------------
# Evidence completeness (KTD9 row 2)
# ---------------------------------------------------------------------------


class TestEvidenceCompleteness:
    def test_a_marker_matching_the_persisted_line_counts_is_complete(self, experiment: Path):
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3)

        record = record_for(experiment, 1)
        assert record.has_evidence_marker is True
        assert record.evidence_complete is True

    def test_a_marker_disagreeing_with_the_persisted_counts_is_not_complete(self, experiment: Path):
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, recorded_records=5)

        record = record_for(experiment, 1)
        assert record.has_evidence_marker is True
        assert record.evidence_complete is False

    def test_an_absent_marker_is_not_complete(self, experiment: Path):
        write_mining_round(experiment, 1)

        record = record_for(experiment, 1)
        assert record.has_evidence_marker is False
        assert record.evidence_complete is False

    def test_a_marker_naming_another_round_certifies_nothing_here(self, experiment: Path):
        """A marker that travelled describes another round's evidence, not this one's.

        The counts still match -- the audit files beside it are intact -- which
        is exactly why the count check alone is not enough: only the recorded
        round index distinguishes a marker written for these files from one
        copied in from a different round.
        """
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, recorded_round=7)

        record = record_for(experiment, 1)
        assert record.has_evidence_marker is True
        assert record.evidence_complete is False

    def test_a_marker_naming_a_different_bundle_than_the_one_on_disk_is_not_complete(
        self, experiment: Path
    ):
        """The marker names the evidence it certifies; a swapped bundle is not it."""
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, bundle_id="bundle-A")
        write_bundle_identity(experiment, 1, bundle_id="bundle-B")

        assert record_for(experiment, 1).evidence_complete is False

    def test_a_marker_over_the_bundle_it_names_stays_complete(self, experiment: Path):
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, bundle_id="bundle-A")
        write_bundle_identity(experiment, 1, bundle_id="bundle-A")

        assert record_for(experiment, 1).evidence_complete is True

    def test_a_marker_carrying_no_bundle_id_is_not_complete(self, experiment: Path):
        """The writer always records one; a marker without it is not that writer's."""
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3, bundle_id=None)

        assert record_for(experiment, 1).evidence_complete is False

    def test_an_audit_file_that_cannot_be_parsed_is_not_complete(self, experiment: Path):
        """The writer parses the records; a line count would pass a torn file.

        Three lines are on disk and the marker says three, so a non-blank line
        count agrees -- but the last line is a truncated JSON object, which is
        the crash the marker exists to catch and which ``read_jsonl`` sees.
        """
        write_mining_round(experiment, 1)
        write_evidence(experiment, 1, n_records=3)
        records_path = record_for(experiment, 1).mining_round_path / RECORDS_FILENAME
        lines = records_path.read_text().splitlines()
        records_path.write_text("\n".join([*lines[:-1], '{"run_id": "r2"']) + "\n")

        assert len(records_path.read_text().splitlines()) == 3
        assert record_for(experiment, 1).evidence_complete is False


# ---------------------------------------------------------------------------
# Expected run counts (KTD6) and the unknown case (KTD9 row 3)
# ---------------------------------------------------------------------------


class TestExpectedRunCounts:
    def test_expected_runs_are_instances_times_the_mining_repetition_count(self, experiment: Path):
        config = load_config(PROFILE)
        write_mining_round(experiment, 1, n_instances=3, attempts=config.loop.m)

        runs = record_for(experiment, 1).mining_runs
        assert runs.n_expected == 3 * config.loop.m
        assert runs.n_present == 3 * config.loop.m
        assert runs.n_missing == 0
        assert runs.runs_complete is True
        assert runs.count_unknown is False

    def test_a_truncated_round_reports_missing_runs_rather_than_completeness(
        self, experiment: Path
    ):
        config = load_config(PROFILE)
        write_mining_round(experiment, 1, n_instances=3, attempts=config.loop.m, n_runs=2)

        runs = record_for(experiment, 1).mining_runs
        assert runs.n_present == 2
        assert runs.n_expected == 3 * config.loop.m
        assert runs.n_missing == 3 * config.loop.m - 2
        assert runs.runs_complete is False

    def test_without_a_config_the_expected_count_is_unknown_and_never_inferred(
        self, tmp_path: Path
    ):
        """A pre-orchestrator tree reports unknown, not a count read off the runs."""
        out_dir = tmp_path / "legacy"
        write_mining_round(out_dir, 1, n_instances=3, attempts=2, n_runs=2)
        assert not (out_dir / CONFIG_FILENAME).exists()

        inventory = discover_rounds(out_dir)
        assert inventory.repetitions is None
        runs = inventory.rounds[0].mining_runs
        assert runs.n_present == 2
        assert runs.n_expected is None
        assert runs.n_missing is None
        assert runs.count_unknown is True
        # KTD9: unknown is its own value; a truncated round must never read as complete,
        # and an unmeasurable one must never read as failed.
        assert runs.runs_complete is None
        assert runs.runs_complete is not False

    def test_a_config_whose_identity_no_longer_matches_is_unavailable(self, experiment: Path):
        """A drifted identity cannot supply repetition counts for this tree."""
        write_mining_round(experiment, 1)
        payload = json.loads((experiment / CONFIG_FILENAME).read_text())
        payload["identity_hash"] = "0" * 64
        write_json(experiment / CONFIG_FILENAME, payload)

        inventory = discover_rounds(experiment)
        assert inventory.repetitions is None
        assert inventory.rounds[0].mining_runs.count_unknown is True

    def test_a_caller_held_config_is_used_without_reading_the_persisted_one(self, tmp_path: Path):
        """The loop holds its config in memory; it need not re-resolve it from disk."""
        out_dir = tmp_path / "legacy"
        config = load_config(PROFILE)
        write_mining_round(out_dir, 1, n_instances=3, attempts=config.loop.m)

        inventory = discover_rounds(out_dir, config=config)
        assert inventory.rounds[0].mining_runs.n_expected == 3 * config.loop.m

    def test_repetition_counts_are_read_per_stage(self):
        config = load_config(PROFILE)
        repetitions = stage_repetitions(config)

        assert repetitions.for_stage(STAGE_MINING) == config.loop.m
        assert repetitions.for_stage(STAGE_VALIDATION) == config.loop.v
        assert repetitions.for_stage(STAGE_EVALUATION) == config.operational.eval_repetitions
        # The three differ in the full profile, so a stage confusion is visible.
        assert len({config.loop.m, config.loop.v, config.operational.eval_repetitions}) == 3


# ---------------------------------------------------------------------------
# The evaluation test-set inventory
# ---------------------------------------------------------------------------


class TestEvaluationInventory:
    def test_every_condition_and_test_set_in_the_summary_is_returned(self, experiment: Path):
        for condition_id in ("b1", "sh_rlm"):
            for set_id in ("graphwalks_short", "oolong_pairs_long"):
                write_eval_set(experiment, condition_id, set_id)
        write_eval_summary(
            experiment,
            {
                condition_id: {
                    "test_sets": {
                        "graphwalks_short": eval_set_payload("graphwalks", "short"),
                        "oolong_pairs_long": eval_set_payload("oolong_pairs", "long"),
                    }
                }
                for condition_id in ("b1", "sh_rlm")
            },
        )

        sets = discover_rounds(experiment).eval_sets
        assert [(record.condition_id, record.set_id) for record in sets] == [
            ("b1", "graphwalks_short"),
            ("b1", "oolong_pairs_long"),
            ("sh_rlm", "graphwalks_short"),
            ("sh_rlm", "oolong_pairs_long"),
        ]
        for record in sets:
            assert record.round_path == round_dir(
                experiment / EVAL_DIR / record.condition_id / record.set_id, EVAL_ROUND_INDEX
            )
            assert record.outcome == OUTCOME_COMPLETED
            assert record.n_skipped == 0
            assert record.complete is True

    def test_a_set_with_skipped_runs_is_flagged_incomplete_not_dropped(self, experiment: Path):
        write_eval_set(experiment, "b1", "graphwalks_short", n_runs=4)
        write_eval_summary(
            experiment,
            {
                "b1": {
                    "test_sets": {
                        "graphwalks_short": eval_set_payload(skipped=("inst01__a03", "inst01__a02"))
                    }
                }
            },
        )

        (record,) = discover_rounds(experiment).eval_sets
        assert record.n_skipped == 2
        assert record.skipped_run_ids == ("inst01__a03", "inst01__a02")
        assert record.complete is False

    def test_an_over_budget_set_is_incomplete(self, experiment: Path):
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_summary(
            experiment,
            {
                "b1": {
                    "test_sets": {"graphwalks_short": eval_set_payload(outcome=OUTCOME_OVER_BUDGET)}
                }
            },
        )

        (record,) = discover_rounds(experiment).eval_sets
        assert record.outcome == OUTCOME_OVER_BUDGET
        assert record.complete is False

    def test_a_set_on_disk_but_absent_from_the_summary_is_unknown(self, experiment: Path):
        write_eval_set(experiment, "b1", "graphwalks_short")
        write_eval_set(experiment, "b1", "oolong_pairs_long")
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        by_set = {record.set_id: record for record in discover_rounds(experiment).eval_sets}
        assert set(by_set) == {"graphwalks_short", "oolong_pairs_long"}
        orphan = by_set["oolong_pairs_long"]
        assert orphan.in_summary is False
        assert orphan.outcome is None
        assert orphan.complete is None
        assert orphan.complete is not False

    def test_a_set_with_no_summary_at_all_is_still_discovered(self, experiment: Path):
        write_eval_set(experiment, "b1", "graphwalks_short")

        (record,) = discover_rounds(experiment).eval_sets
        assert record.set_id == "graphwalks_short"
        assert record.complete is None

    def test_expected_eval_runs_use_the_evaluation_repetition_count(self, experiment: Path):
        config = load_config(PROFILE)
        write_eval_set(
            experiment,
            "b1",
            "graphwalks_short",
            n_instances=2,
            attempts=config.operational.eval_repetitions,
        )
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        (record,) = discover_rounds(experiment).eval_sets
        assert record.runs.n_expected == 2 * config.operational.eval_repetitions
        assert record.runs.n_present == 2 * config.operational.eval_repetitions
        assert record.runs.runs_complete is True

    def test_a_set_short_on_runs_is_not_complete_however_the_summary_reports_itself(
        self, experiment: Path
    ):
        """The summary is a self-report; the persisted runs are the evidence.

        ``eval_summary.json`` records the outcome the evaluation held in memory
        when it aggregated, so a set truncated on disk can still say
        ``completed`` with no skipped runs. Publishing that as complete is the
        false completeness the round rows have always guarded against with
        ``runs_complete``, so the eval verdict folds the same evidence in.
        """
        config = load_config(PROFILE)
        write_eval_set(
            experiment,
            "b1",
            "graphwalks_short",
            n_instances=2,
            attempts=config.operational.eval_repetitions,
            n_runs=2,
        )
        write_eval_summary(
            experiment, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}}
        )

        (record,) = discover_rounds(experiment).eval_sets
        assert record.outcome == OUTCOME_COMPLETED
        assert record.n_skipped == 0
        assert record.summary_complete is True
        assert record.runs.n_present == 2
        assert record.runs.n_expected == 2 * config.operational.eval_repetitions
        assert record.runs.runs_complete is False
        assert record.complete is False

    def test_a_set_whose_planned_count_is_unknowable_is_unknown_never_complete(
        self, tmp_path: Path
    ):
        """KTD9: no config means unmeasurable, which is neither complete nor failed."""
        legacy = tmp_path / "legacy"
        write_eval_set(legacy, "b1", "graphwalks_short", n_instances=2, attempts=3)
        write_eval_summary(legacy, {"b1": {"test_sets": {"graphwalks_short": eval_set_payload()}}})
        assert not (legacy / CONFIG_FILENAME).exists()

        (record,) = discover_rounds(legacy).eval_sets
        assert record.summary_complete is True
        assert record.runs.count_unknown is True
        assert record.complete is None
        assert record.complete is not False

    def test_a_condition_work_directory_is_not_a_test_set(self, experiment: Path):
        write_eval_set(experiment, "sh_rlm", "graphwalks_short")
        (experiment / EVAL_DIR / "sh_rlm" / "work").mkdir(parents=True)

        assert [record.set_id for record in discover_rounds(experiment).eval_sets] == [
            "graphwalks_short"
        ]


# ---------------------------------------------------------------------------
# The promotion-ledger iteration the graph scripts consume
# ---------------------------------------------------------------------------


class TestPromotionLedgerIteration:
    def test_only_ledgered_rounds_are_yielded_in_round_order(self, experiment: Path):
        complete_round(experiment, 1)
        write_round_marker(experiment, 2, has_ledger=False)  # a completed no-op round
        complete_round(experiment, 3)

        yielded = list(iter_promotion_rounds(experiment))
        assert [round_index for round_index, _records, _decision in yielded] == [1, 3]
        for round_index, records, decision in yielded:
            assert records[0]["subject_id"].startswith(f"r{round_index:02d}-")
            assert decision["round"] == round_index


# ---------------------------------------------------------------------------
# The grounding scenario: the tree the real loop writes
# ---------------------------------------------------------------------------


class TestDiscoveryAgainstTheMockSmokeTree:
    """Discovery over the mock-LM end-to-end run's own output (KTD1's premise).

    The ``smoke`` parameter is the imported module-scoped fixture; ruff reads
    the parameter as a redefinition of the imported name, which is what the
    ``F811`` suppressions below are for.
    """

    def test_the_single_smoke_round_is_discovered_exactly_as_the_loop_wrote_it(
        self,
        smoke: SmokeRun,  # noqa: F811
    ):
        inventory = discover_rounds(smoke.out_dir)

        assert [record.round_index for record in inventory.rounds] == [1]
        (record,) = inventory.rounds
        assert record.round_path == experiment_round_dir(smoke.out_dir, 1)
        assert (record.round_path / ROUND_MARKER_FILENAME).exists()
        assert (record.mining_round_path / MANIFEST_FILE).exists()
        assert (record.validation_round_path / PROMOTIONS_FILENAME).exists()
        assert record.proposals_dir.is_dir()

        assert record.round_complete is True
        assert record.has_ledger is True
        assert record.has_manifest is True
        assert record.has_evidence_marker is True
        assert record.evidence_complete is True

    def test_the_smoke_rounds_run_counts_match_the_configured_plan(
        self,
        smoke: SmokeRun,  # noqa: F811
    ):
        """m=1 attempt over the three held-in instances, all of them persisted."""
        (record,) = discover_rounds(smoke.out_dir).rounds
        n_instances = len((record.mining_round_path / INSTANCES_FILE).read_text().splitlines())

        assert record.mining_runs.n_expected == n_instances * smoke.config.loop.m
        assert record.mining_runs.n_present == record.mining_runs.n_expected
        assert record.mining_runs.runs_complete is True

    def test_the_smoke_eval_inventory_matches_its_summary(
        self,
        smoke: SmokeRun,  # noqa: F811
    ):
        summary = json.loads((smoke.out_dir / EVAL_DIR / EVAL_SUMMARY_FILENAME).read_text())
        expected = {
            (condition_id, set_id)
            for condition_id, condition in summary["conditions"].items()
            for set_id in condition["test_sets"]
        }

        sets = discover_rounds(smoke.out_dir).eval_sets
        assert {(record.condition_id, record.set_id) for record in sets} == expected
        for record in sets:
            assert record.in_summary is True
            assert record.complete is True
            assert record.runs.runs_complete is True
            assert record.round_path.is_dir()

    def test_promotion_ledger_iteration_reads_the_smoke_ledger(
        self,
        smoke: SmokeRun,  # noqa: F811
    ):
        yielded = list(iter_promotion_rounds(smoke.out_dir))

        assert [round_index for round_index, _records, _decision in yielded] == [1]
        _round_index, records, decision = yielded[0]
        assert records
        assert decision["format"] == DECISION_FORMAT
