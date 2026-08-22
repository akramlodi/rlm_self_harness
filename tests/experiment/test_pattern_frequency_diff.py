"""Tests for the mechanism-frequency diff (proposal section 3.5).

The module full-outer-joins two ``bundle.json``s' mined ``patterns[]`` on the
four-tuple failure signature and says, per signature, what happened to it
between the rounds. Four things carry most of the weight here, and each has its
own class below:

* **The join key is the whole four-tuple.** Two patterns that agree on three
  fields and differ on the fourth are different failures, and must not be
  reported as one signature that moved -- so the disagreement cases are
  parametrized over every field of ``SIGNATURE_FIELDS`` rather than spot-checked
  on one of them.
* **Presence is rate-based, not key-based.** A signature whose support fell to
  zero is ``resolved``, not ``persisted_improved``: the key is still in the
  after-bundle, but the failure is not. The classification tests therefore pin
  the boundary between "key present" and "failure present".
* **A rate difference smaller than ``RATE_EPSILON`` is unchanged.** Two rounds
  mined at different run counts produce rates that differ in the last bits of a
  float; flipping those to ``persisted_worsened`` would report floating-point
  noise as a regression. The epsilon tests use denominators that put the
  difference genuinely below and genuinely above the threshold, rather than
  comparing a float to itself, which would pass with no epsilon at all.
* **A bundle is matched to a round by its path, never by the ``round_index``
  it carries.** A bundle copied out of the tree records a round index just as
  convincingly as an in-tree one, so the matching test deliberately writes a
  bundle claiming round 1 into round 2's mining directory and asserts the
  completeness row says 2.

Bundles are fabricated through ``tests.experiment.test_rounds``' shared
``write_bundle`` where one default-signature pattern is enough, and through the
local ``write_patterns`` generalization where the case needs several patterns or
a non-default signature. Trees come from the same module's layout helpers, so a
layout change moves the fixtures with it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shrlm.experiment.analysis_io import (
    ANALYSIS_DIR,
    PROVENANCE_FILENAME,
    PUBLISHED_FILENAME,
    TRISTATE_FALSE,
    TRISTATE_TRUE,
    TRISTATE_UNKNOWN,
    Snapshot,
    allocate_snapshot,
)
from shrlm.experiment.orchestrator import CONFIG_FILENAME, ROUND_MARKER_FILENAME
from shrlm.experiment.pattern_frequency_diff import (
    BUNDLES_FILENAME,
    DIFF_FIELDNAMES,
    RATE_EPSILON,
    SIGNATURE_FIELDS,
    STATUS_NEW,
    STATUS_PERSISTED_IMPROVED,
    STATUS_PERSISTED_UNCHANGED,
    STATUS_PERSISTED_WORSENED,
    STATUS_RESOLVED,
    TOOL_NAME,
    UNVERSIONED_TAXONOMY,
    bundle_completeness,
    diff_bundle_pair,
    load_bundle_summary,
    main,
    run_pattern_frequency_diff,
    summarize,
    taxonomy_versions_seen,
)
from shrlm.experiment.rounds import discover_rounds
from shrlm.optimization.bundle import BUNDLE_FILENAME
from shrlm.optimization.taxonomy import TAXONOMY_VERSION
from tests.experiment.test_rounds import (
    complete_round,
    read_csv,
    record_for,
    write_bundle,
    write_config,
    write_evidence,
    write_json,
)

FROZEN = datetime(2026, 8, 18, 16, 48, 0, tzinfo=UTC)
PROFILE = "full"


# ---------------------------------------------------------------------------
# Fabrication
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An experiment directory with a persisted identity and nothing else."""
    out_dir = tmp_path / "experiment"
    write_config(out_dir, PROFILE)
    return out_dir


@pytest.fixture
def snapshot(experiment: Path) -> Snapshot:
    return allocate_snapshot(experiment, now=FROZEN)


def signature_fields(**overrides: str) -> dict[str, str]:
    """The default four-tuple, with any subset of its fields replaced."""
    fields = {
        "verifier_cause": "wrong_answer",
        "failing_level": "child",
        "causal_status": "causal",
        "agent_mechanism": "routed_whole_input",
    }
    fields.update(overrides)
    return fields


def pattern(
    *,
    support: int,
    grounded_fraction: float = 1.0,
    below_support_floor: bool = False,
    **signature_overrides: str,
) -> dict[str, Any]:
    """One mined ``patterns[]`` entry, in mining's own shape."""
    return {
        "signature": signature_fields(**signature_overrides),
        "instance_support": support,
        "grounded_fraction": grounded_fraction,
        "below_support_floor": below_support_floor,
    }


def write_patterns(
    path: Path,
    patterns: list[dict[str, Any]],
    *,
    n_runs: int = 10,
    round_index: int | None = None,
    taxonomy_version: str | None = None,
) -> Path:
    """``write_bundle``'s generalization: any pattern list, optional round index.

    The shared ``write_bundle`` carries exactly one default-signature pattern,
    which is what the completeness and CLI tests need. The join, duplicate,
    empty-bundle, and rate-epsilon cases here need several patterns, non-default
    signatures, arbitrary denominators, and a bundle with no ``round_index`` at
    all, so those are built from the same shape with those parts left open.
    ``taxonomy_version`` stamps ``config.taxonomy_version`` the way mining does;
    left None, the bundle is unversioned, as every fixture bundle was before the
    version gate existed.
    """
    config: dict[str, Any] = {}
    if round_index is not None:
        config["round_index"] = round_index
    if taxonomy_version is not None:
        config["taxonomy_version"] = taxonomy_version
    write_json(path, {"config": config, "totals": {"n_runs": n_runs}, "patterns": patterns})
    return path


def summary_for(
    path: Path,
    patterns: list[dict[str, Any]],
    *,
    n_runs: int = 10,
    round_index: int | None = None,
    label: str | None = None,
) -> Any:
    """A loaded ``_BundleSummary`` over a freshly written bundle."""
    written = write_patterns(path, patterns, n_runs=n_runs, round_index=round_index)
    return load_bundle_summary(written, explicit_label=label)


def diff(
    tmp_path: Path,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    n_runs_before: int = 10,
    n_runs_after: int = 10,
) -> list[Any]:
    return diff_bundle_pair(
        summary_for(tmp_path / "before.json", before, n_runs=n_runs_before),
        summary_for(tmp_path / "after.json", after, n_runs=n_runs_after),
    )


def statuses(rows: list[Any]) -> list[str]:
    return [row.status for row in rows]


def diff_names(snapshot: Snapshot) -> list[str]:
    return sorted(path.name for path in snapshot.path.glob("pattern_frequency_diff_*_vs_*.csv"))


# ---------------------------------------------------------------------------
# The join key: the whole four-tuple, and exactly one pattern per signature
# ---------------------------------------------------------------------------


class TestSignatureJoin:
    def test_two_bundles_agreeing_on_all_four_fields_join_to_one_row(self, tmp_path: Path) -> None:
        rows = diff(tmp_path, [pattern(support=5)], [pattern(support=2)])

        assert len(rows) == 1
        assert rows[0].signature_key == (
            "wrong_answer",
            "child",
            "causal",
            "routed_whole_input",
        )

    @pytest.mark.parametrize("field_name", SIGNATURE_FIELDS)
    def test_a_disagreement_in_any_single_field_makes_two_distinct_signatures(
        self, tmp_path: Path, field_name: str
    ) -> None:
        """Three fields matching is not a match; the key is the whole tuple."""
        rows = diff(
            tmp_path,
            [pattern(support=5)],
            [pattern(support=5, **{field_name: "something_else"})],
        )

        assert len(rows) == 2
        assert sorted(statuses(rows)) == sorted([STATUS_NEW, STATUS_RESOLVED])

    def test_the_signature_key_is_ordered_by_the_declared_field_order(self, tmp_path: Path) -> None:
        (row,) = diff(
            tmp_path,
            [
                pattern(
                    support=5,
                    verifier_cause="a",
                    failing_level="b",
                    causal_status="c",
                    agent_mechanism="d",
                )
            ],
            [],
        )

        assert row.signature_key == ("a", "b", "c", "d")
        assert row.signature_str == (
            "verifier_cause=a|failing_level=b|causal_status=c|agent_mechanism=d"
        )

    def test_a_bundle_carrying_two_patterns_with_one_signature_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Mining clusters each signature into exactly one pattern; two is corrupt."""
        path = write_patterns(tmp_path / "corrupt.json", [pattern(support=5), pattern(support=9)])

        with pytest.raises(ValueError) as raised:
            load_bundle_summary(path)

        assert str(path) in str(raised.value)
        assert "two patterns sharing signature" in str(raised.value)

    def test_two_patterns_differing_only_outside_the_signature_still_collide(
        self, tmp_path: Path
    ) -> None:
        """The guard is on the signature, not on the whole pattern payload."""
        path = write_patterns(
            tmp_path / "corrupt.json",
            [
                pattern(support=5, grounded_fraction=1.0),
                pattern(support=5, grounded_fraction=0.25, below_support_floor=True),
            ],
        )

        with pytest.raises(ValueError):
            load_bundle_summary(path)


# ---------------------------------------------------------------------------
# The five-way status classification
# ---------------------------------------------------------------------------


class TestStatusClassification:
    def test_a_signature_present_before_and_absent_after_is_resolved(self, tmp_path: Path) -> None:
        (row,) = diff(tmp_path, [pattern(support=5)], [])

        assert row.status == STATUS_RESOLVED
        assert row.support_rate_before == 0.5
        assert row.support_rate_after == 0.0
        assert row.delta == -0.5

    def test_a_signature_absent_before_and_present_after_is_new(self, tmp_path: Path) -> None:
        (row,) = diff(tmp_path, [], [pattern(support=3)])

        assert row.status == STATUS_NEW
        assert row.support_rate_before == 0.0
        assert row.support_rate_after == 0.3

    def test_a_falling_rate_on_a_surviving_signature_is_persisted_improved(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(tmp_path, [pattern(support=8)], [pattern(support=2)])

        assert row.status == STATUS_PERSISTED_IMPROVED
        assert row.delta == pytest.approx(-0.6)

    def test_a_rising_rate_on_a_surviving_signature_is_persisted_worsened(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(tmp_path, [pattern(support=2)], [pattern(support=8)])

        assert row.status == STATUS_PERSISTED_WORSENED
        assert row.delta == pytest.approx(0.6)

    def test_an_identical_rate_on_both_sides_is_persisted_unchanged(self, tmp_path: Path) -> None:
        (row,) = diff(tmp_path, [pattern(support=4)], [pattern(support=4)])

        assert row.status == STATUS_PERSISTED_UNCHANGED
        assert row.delta == 0.0

    def test_an_equal_rate_reached_from_different_denominators_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """The reading is the rate, not the raw support count."""
        (row,) = diff(
            tmp_path, [pattern(support=4)], [pattern(support=8)], n_runs_before=10, n_runs_after=20
        )

        assert row.status == STATUS_PERSISTED_UNCHANGED
        assert row.support_rate_before == row.support_rate_after == 0.4

    def test_a_signature_whose_support_fell_to_zero_is_resolved_not_improved(
        self, tmp_path: Path
    ) -> None:
        """The key survives into the after-bundle; the failure does not."""
        (row,) = diff(tmp_path, [pattern(support=5)], [pattern(support=0)])

        assert row.status == STATUS_RESOLVED
        assert row.support_rate_after == 0.0
        # The after-side columns are still populated: the pattern was there to
        # read, which is a different statement from "no pattern was found".
        assert row.grounded_fraction_after == 1.0
        assert row.below_support_floor_after is False

    def test_a_signature_absent_on_both_sides_of_a_zero_denominator_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """A mined-nothing round reads as no change, never as a resolution."""
        (row,) = diff(
            tmp_path, [pattern(support=0)], [pattern(support=0)], n_runs_before=0, n_runs_after=0
        )

        assert row.status == STATUS_PERSISTED_UNCHANGED

    def test_the_absent_side_reports_null_metadata_rather_than_a_default(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(tmp_path, [], [pattern(support=3, grounded_fraction=0.5)])

        assert row.grounded_fraction_before is None
        assert row.below_support_floor_before is None
        assert row.grounded_fraction_after == 0.5

    def test_summarize_counts_every_status_and_the_signature_total(self, tmp_path: Path) -> None:
        rows = diff(
            tmp_path,
            [
                pattern(support=5, verifier_cause="gone"),
                pattern(support=8, verifier_cause="better"),
                pattern(support=2, verifier_cause="worse"),
                pattern(support=4, verifier_cause="same"),
            ],
            [
                pattern(support=2, verifier_cause="better"),
                pattern(support=8, verifier_cause="worse"),
                pattern(support=4, verifier_cause="same"),
                pattern(support=1, verifier_cause="arrived"),
            ],
        )

        assert summarize(rows) == {
            STATUS_RESOLVED: 1,
            STATUS_NEW: 1,
            STATUS_PERSISTED_IMPROVED: 1,
            STATUS_PERSISTED_WORSENED: 1,
            STATUS_PERSISTED_UNCHANGED: 1,
            "total_signatures": 5,
        }


# ---------------------------------------------------------------------------
# The rate epsilon: float noise is not a regression
# ---------------------------------------------------------------------------


# 5e-10 apart -- below RATE_EPSILON, the difference two rounds mined at slightly
# different run counts produce; 2e-9 apart -- above it, a real move.
EQUAL_DENOMINATOR = 1_000_000_000
BELOW_EPSILON_DENOMINATOR = 999_999_999
ABOVE_EPSILON_DENOMINATOR = 999_999_996
HALF_SUPPORT = 500_000_000


class TestRateEpsilon:
    def test_a_rise_smaller_than_the_epsilon_is_unchanged_not_worsened(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(
            tmp_path,
            [pattern(support=HALF_SUPPORT)],
            [pattern(support=HALF_SUPPORT)],
            n_runs_before=EQUAL_DENOMINATOR,
            n_runs_after=BELOW_EPSILON_DENOMINATOR,
        )

        assert row.support_rate_after != row.support_rate_before
        assert 0 < row.delta < RATE_EPSILON
        assert row.status == STATUS_PERSISTED_UNCHANGED

    def test_a_fall_smaller_than_the_epsilon_is_unchanged_not_improved(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(
            tmp_path,
            [pattern(support=HALF_SUPPORT)],
            [pattern(support=HALF_SUPPORT)],
            n_runs_before=BELOW_EPSILON_DENOMINATOR,
            n_runs_after=EQUAL_DENOMINATOR,
        )

        assert row.support_rate_after != row.support_rate_before
        assert -RATE_EPSILON < row.delta < 0
        assert row.status == STATUS_PERSISTED_UNCHANGED

    def test_a_rise_just_larger_than_the_epsilon_is_still_worsened(self, tmp_path: Path) -> None:
        """The epsilon suppresses noise, not small real moves."""
        (row,) = diff(
            tmp_path,
            [pattern(support=HALF_SUPPORT)],
            [pattern(support=HALF_SUPPORT)],
            n_runs_before=EQUAL_DENOMINATOR,
            n_runs_after=ABOVE_EPSILON_DENOMINATOR,
        )

        assert row.delta > RATE_EPSILON
        assert row.status == STATUS_PERSISTED_WORSENED


# ---------------------------------------------------------------------------
# The percentage delta and its zero denominator
# ---------------------------------------------------------------------------


class TestPercentageDelta:
    def test_a_halved_rate_reports_a_minus_one_half_fraction(self, tmp_path: Path) -> None:
        (row,) = diff(tmp_path, [pattern(support=8)], [pattern(support=4)])

        assert row.delta_pct == pytest.approx(-0.5)

    def test_a_newly_introduced_signature_has_no_percentage_rather_than_a_division(
        self, tmp_path: Path
    ) -> None:
        (row,) = diff(tmp_path, [], [pattern(support=3)])

        assert row.support_rate_before == 0.0
        assert row.delta_pct is None

    def test_a_pattern_whose_round_mined_nothing_has_no_percentage(self, tmp_path: Path) -> None:
        """The pattern is on the before side, but its denominator is zero."""
        (row,) = diff(
            tmp_path,
            [pattern(support=5)],
            [pattern(support=3)],
            n_runs_before=0,
            n_runs_after=10,
        )

        assert row.support_rate_before == 0.0
        assert row.delta_pct is None
        assert row.status == STATUS_NEW
        # The before-side pattern metadata is still reported: the guard is on
        # the arithmetic, not on whether the pattern existed.
        assert row.grounded_fraction_before == 1.0

    def test_the_empty_percentage_is_written_as_a_blank_cell(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        before = write_patterns(tmp_path / "before.json", [], round_index=1)
        after = write_patterns(tmp_path / "after.json", [pattern(support=3)], round_index=2)

        run_pattern_frequency_diff([before, after], snapshot)

        (row,) = read_csv(snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv")
        assert row["status"] == STATUS_NEW
        assert row["delta_pct"] == ""
        assert row["grounded_fraction_before"] == ""
        assert row["support_rate_before"] == "0.0"


# ---------------------------------------------------------------------------
# Labels: where a bundle's name comes from, and what happens when two collide
# ---------------------------------------------------------------------------


class TestLabelDerivation:
    def test_a_round_index_in_the_config_names_the_bundle(self, tmp_path: Path) -> None:
        summary = summary_for(tmp_path / "bundle.json", [], round_index=3)

        assert summary.label == "round_3"

    def test_the_round_index_wins_over_an_explicit_label(self, tmp_path: Path) -> None:
        """``--labels`` is the fallback for a bundle that never stamped a round."""
        summary = summary_for(tmp_path / "bundle.json", [], round_index=3, label="baseline")

        assert summary.label == "round_3"

    def test_an_explicit_label_is_used_when_the_config_has_no_round_index(
        self, tmp_path: Path
    ) -> None:
        summary = summary_for(tmp_path / "bundle.json", [], label="baseline")

        assert summary.label == "baseline"

    def test_the_parent_directory_names_a_bundle_with_neither(self, tmp_path: Path) -> None:
        """Every bundle shares the filename ``bundle.json``; the directory does not."""
        summary = summary_for(tmp_path / "round_07" / BUNDLE_FILENAME, [])

        assert summary.label == "round_07"

    def test_a_label_with_path_unsafe_characters_is_made_filesystem_safe(
        self, tmp_path: Path
    ) -> None:
        summary = summary_for(tmp_path / "bundle.json", [], label="before opt/run #1")

        assert summary.label == "before_opt_run_1"

    def test_a_label_with_nothing_safe_left_falls_back_to_a_constant(self, tmp_path: Path) -> None:
        summary = summary_for(tmp_path / "bundle.json", [], label="///")

        assert summary.label == "bundle"

    def test_two_bundles_claiming_one_round_index_get_distinct_output_names(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """A re-mined round would otherwise write both diffs to one filename."""
        first = write_bundle(tmp_path / "first.json", 1, support=5)
        second = write_bundle(tmp_path / "second.json", 1, support=2)

        run_pattern_frequency_diff([first, second], snapshot)

        assert diff_names(snapshot) == ["pattern_frequency_diff_round_1_vs_round_1_1.csv"]
        assert [row["label"] for row in read_csv(snapshot.path / BUNDLES_FILENAME)] == [
            "round_1",
            "round_1_1",
        ]

    def test_the_dedup_suffix_is_positional_so_a_third_collision_is_distinct(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        paths = [write_bundle(tmp_path / f"b{i}.json", 1, support=i + 1) for i in range(3)]

        run_pattern_frequency_diff(paths, snapshot)

        assert [row["label"] for row in read_csv(snapshot.path / BUNDLES_FILENAME)] == [
            "round_1",
            "round_1_1",
            "round_1_2",
        ]


# ---------------------------------------------------------------------------
# Which pairs get diffed
# ---------------------------------------------------------------------------


class TestPairing:
    def test_two_bundles_produce_the_single_before_after_pair(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        first = write_bundle(tmp_path / "b1.json", 1, support=5)
        second = write_bundle(tmp_path / "b2.json", 2, support=2)

        written = run_pattern_frequency_diff([first, second], snapshot)

        assert diff_names(snapshot) == ["pattern_frequency_diff_round_1_vs_round_2.csv"]
        assert written[0] == snapshot.path / BUNDLES_FILENAME
        assert snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv" in written

    def test_three_bundles_add_the_first_versus_last_pair_to_the_consecutive_ones(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """The whole-experiment reading is not the sum of the round-to-round ones."""
        paths = [write_bundle(tmp_path / f"b{i}.json", i, support=i) for i in (1, 2, 3)]

        run_pattern_frequency_diff(paths, snapshot)

        assert diff_names(snapshot) == [
            "pattern_frequency_diff_round_1_vs_round_2.csv",
            "pattern_frequency_diff_round_1_vs_round_3.csv",
            "pattern_frequency_diff_round_2_vs_round_3.csv",
        ]

    def test_four_bundles_produce_three_consecutive_pairs_plus_first_versus_last(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        paths = [write_bundle(tmp_path / f"b{i}.json", i, support=i) for i in (1, 2, 3, 4)]

        run_pattern_frequency_diff(paths, snapshot)

        assert diff_names(snapshot) == [
            "pattern_frequency_diff_round_1_vs_round_2.csv",
            "pattern_frequency_diff_round_1_vs_round_4.csv",
            "pattern_frequency_diff_round_2_vs_round_3.csv",
            "pattern_frequency_diff_round_3_vs_round_4.csv",
        ]

    def test_the_first_versus_last_pair_reads_across_the_whole_experiment(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """Round 1 -> 2 worsens and 2 -> 3 improves; 1 -> 3 is the net."""
        paths = [
            write_bundle(tmp_path / "b1.json", 1, support=4),
            write_bundle(tmp_path / "b2.json", 2, support=9),
            write_bundle(tmp_path / "b3.json", 3, support=1),
        ]

        run_pattern_frequency_diff(paths, snapshot)

        def status(name: str) -> str:
            (row,) = read_csv(snapshot.path / f"pattern_frequency_diff_{name}.csv")
            return row["status"]

        assert status("round_1_vs_round_2") == STATUS_PERSISTED_WORSENED
        assert status("round_2_vs_round_3") == STATUS_PERSISTED_IMPROVED
        assert status("round_1_vs_round_3") == STATUS_PERSISTED_IMPROVED

    def test_a_single_bundle_is_refused_rather_than_diffed_against_itself(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        only = write_bundle(tmp_path / "only.json", 1, support=5)

        with pytest.raises(ValueError, match="at least 2 bundle paths"):
            run_pattern_frequency_diff([only], snapshot)

    def test_a_label_list_that_does_not_cover_every_bundle_is_refused(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """Silently zipping short would name the wrong bundle."""
        first = write_bundle(tmp_path / "b1.json", 1, support=5)
        second = write_bundle(tmp_path / "b2.json", 2, support=2)

        with pytest.raises(ValueError, match="--labels has 1 entries but 2 bundles"):
            run_pattern_frequency_diff([first, second], snapshot, labels=["only"])


# ---------------------------------------------------------------------------
# Matching a bundle to a round, and what completeness does with the answer
# ---------------------------------------------------------------------------


class TestBundleToRoundMatching:
    def test_a_bundle_is_matched_to_the_round_whose_path_holds_it(self, experiment: Path) -> None:
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        bundle_path = record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME
        write_bundle(bundle_path, 2, support=5)

        (row,) = bundle_completeness(
            [load_bundle_summary(bundle_path)], discover_rounds(experiment)
        )

        assert row.round_index == 2
        assert row.round_complete is True
        assert row.evidence_complete is True
        assert row.included is True

    def test_the_config_round_index_never_overrides_the_path_the_bundle_sits_at(
        self, experiment: Path
    ) -> None:
        """A copied bundle records a round index just as convincingly as an in-tree one."""
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        bundle_path = record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME
        write_bundle(bundle_path, 1, support=5)

        summary = load_bundle_summary(bundle_path)
        (row,) = bundle_completeness([summary], discover_rounds(experiment))

        assert summary.label == "round_1"
        assert row.round_index == 2

    def test_a_bundle_claiming_a_complete_round_is_still_excluded_by_the_round_it_sits_in(
        self, experiment: Path
    ) -> None:
        """The whole point of matching by path: the claimed round is not evidence."""
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        write_evidence(experiment, 2, n_records=3, recorded_records=5)
        bundle_path = record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME
        write_bundle(bundle_path, 1, support=5)

        (row,) = bundle_completeness(
            [load_bundle_summary(bundle_path)], discover_rounds(experiment)
        )

        assert row.round_index == 2
        assert row.evidence_complete is False
        assert row.included is False
        assert "round 2 is not evidence-complete" in (row.exclusion_reason or "")

    def test_a_bundle_no_round_holds_is_unknown_and_still_included(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        """Unknown never collapses into false, so it is never a reason to exclude."""
        complete_round(experiment, 1)
        outside = write_bundle(tmp_path / "lifted.json", 1, support=5)

        (row,) = bundle_completeness([load_bundle_summary(outside)], discover_rounds(experiment))

        assert row.round_index is None
        assert row.round_complete is None
        assert row.evidence_complete is None
        assert row.included is True
        assert row.exclusion_reason is None

    def test_the_completeness_row_serializes_unknown_as_its_own_value(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        (row,) = bundle_completeness(
            [load_bundle_summary(write_bundle(tmp_path / "lifted.json", 1, support=5))],
            discover_rounds(experiment),
        )
        serialized = row.to_dict()

        assert serialized["round_complete"] == TRISTATE_UNKNOWN
        assert serialized["evidence_complete"] == TRISTATE_UNKNOWN
        assert serialized["evidence_complete"] != TRISTATE_FALSE
        assert serialized["included"] == TRISTATE_TRUE
        assert serialized["round_index"] is None


OLD_TAXONOMY = "2.0.0"


class TestTaxonomyVersionGate:
    """A bundle written under another taxonomy version is not diffed across the boundary.

    The mechanism vocabulary is the join key, so two bundles under different
    taxonomy versions would be joined on labels that mean different things.
    The gate is against the running code's ``TAXONOMY_VERSION`` (not a
    majority: a pairwise diff has none, and a tree where post-bump rounds are
    the minority would exclude the new bundles); an unversioned bundle is
    unknown, not different, and stays in -- as every fixture bundle here does.
    """

    def test_an_old_version_bundle_is_excluded_naming_both_versions(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        assert OLD_TAXONOMY != TAXONOMY_VERSION
        old = write_patterns(
            tmp_path / "old.json", [pattern(support=3)], taxonomy_version=OLD_TAXONOMY
        )

        (row,) = bundle_completeness([load_bundle_summary(old)], discover_rounds(experiment))

        assert row.included is False
        assert row.taxonomy_version == OLD_TAXONOMY
        assert OLD_TAXONOMY in (row.exclusion_reason or "")
        assert TAXONOMY_VERSION in (row.exclusion_reason or "")
        assert row.to_dict()["taxonomy_version"] == OLD_TAXONOMY

    def test_a_current_version_bundle_is_included(self, experiment: Path, tmp_path: Path) -> None:
        new = write_patterns(
            tmp_path / "new.json", [pattern(support=3)], taxonomy_version=TAXONOMY_VERSION
        )
        (row,) = bundle_completeness([load_bundle_summary(new)], discover_rounds(experiment))
        assert row.included is True
        assert row.exclusion_reason is None
        assert row.taxonomy_version == TAXONOMY_VERSION

    def test_an_unversioned_bundle_is_unknown_and_still_included(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        """Unknown never collapses into different, exactly as for completeness."""
        bare = write_patterns(tmp_path / "bare.json", [pattern(support=3)])
        (row,) = bundle_completeness([load_bundle_summary(bare)], discover_rounds(experiment))
        assert row.included is True
        assert row.taxonomy_version is None
        assert row.to_dict()["taxonomy_version"] == UNVERSIONED_TAXONOMY

    def test_a_pairwise_diff_of_one_old_and_one_new_bundle_excludes_only_the_old(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        old = write_patterns(
            tmp_path / "b1.json", [pattern(support=5)], round_index=1, taxonomy_version=OLD_TAXONOMY
        )
        new = write_patterns(
            tmp_path / "b2.json",
            [pattern(support=2)],
            round_index=2,
            taxonomy_version=TAXONOMY_VERSION,
        )

        written = run_pattern_frequency_diff([old, new], snapshot)

        rows = {row["label"]: row for row in read_csv(snapshot.path / BUNDLES_FILENAME)}
        assert rows["round_1"]["included"] == TRISTATE_FALSE
        assert rows["round_2"]["included"] == TRISTATE_TRUE
        assert rows["round_1"]["taxonomy_version"] == OLD_TAXONOMY
        assert rows["round_2"]["taxonomy_version"] == TAXONOMY_VERSION
        assert OLD_TAXONOMY in rows["round_1"]["exclusion_reason"]
        assert TAXONOMY_VERSION in rows["round_1"]["exclusion_reason"]
        # One comparable bundle is below the pair minimum: nothing was diffed,
        # and the new bundle was not dragged out with the old one.
        assert written == [snapshot.path / BUNDLES_FILENAME]
        assert diff_names(snapshot) == []

    def test_every_version_seen_is_reported_with_its_bundle_count(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        paths = [
            write_patterns(
                tmp_path / "b1.json",
                [pattern(support=5)],
                round_index=1,
                taxonomy_version=OLD_TAXONOMY,
            ),
            write_patterns(
                tmp_path / "b2.json",
                [pattern(support=4)],
                round_index=2,
                taxonomy_version=OLD_TAXONOMY,
            ),
            write_patterns(
                tmp_path / "b3.json",
                [pattern(support=3)],
                round_index=3,
                taxonomy_version=TAXONOMY_VERSION,
            ),
            write_patterns(
                tmp_path / "b4.json",
                [pattern(support=2)],
                round_index=4,
                taxonomy_version=TAXONOMY_VERSION,
            ),
            write_patterns(tmp_path / "b5.json", [pattern(support=1)], round_index=5),
        ]
        summaries = [load_bundle_summary(path) for path in paths]
        assert taxonomy_versions_seen(summaries) == {
            OLD_TAXONOMY: 2,
            TAXONOMY_VERSION: 2,
            UNVERSIONED_TAXONOMY: 1,
        }

        run_pattern_frequency_diff(paths, snapshot)

        # Three comparable bundles (two current, one unversioned) -> the
        # consecutive pairs plus first-vs-last; the old pair never appears.
        assert diff_names(snapshot) == [
            "pattern_frequency_diff_round_3_vs_round_4.csv",
            "pattern_frequency_diff_round_3_vs_round_5.csv",
            "pattern_frequency_diff_round_4_vs_round_5.csv",
        ]
        payload = json.loads(
            (snapshot.path / "pattern_frequency_diff_round_3_vs_round_4.json").read_text()
        )
        assert payload["taxonomy_version_expected"] == TAXONOMY_VERSION
        assert payload["taxonomy_versions_seen"] == {
            OLD_TAXONOMY: 2,
            TAXONOMY_VERSION: 2,
            UNVERSIONED_TAXONOMY: 1,
        }

    def test_a_mismatch_and_an_incomplete_round_are_both_named_in_the_reason(
        self, experiment: Path
    ) -> None:
        complete_round(experiment, 2)
        write_evidence(experiment, 2, n_records=3, recorded_records=5)
        bundle_path = record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME
        write_patterns(
            bundle_path, [pattern(support=5)], round_index=2, taxonomy_version=OLD_TAXONOMY
        )

        (row,) = bundle_completeness(
            [load_bundle_summary(bundle_path)], discover_rounds(experiment)
        )

        assert row.included is False
        assert "round 2 is not evidence-complete" in (row.exclusion_reason or "")
        assert OLD_TAXONOMY in (row.exclusion_reason or "")

    def test_diffing_old_version_bundles_is_an_explicit_opt_in(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """Two old bundles diff only when the caller names that version."""
        first = write_patterns(
            tmp_path / "b1.json", [pattern(support=5)], round_index=1, taxonomy_version=OLD_TAXONOMY
        )
        second = write_patterns(
            tmp_path / "b2.json", [pattern(support=2)], round_index=2, taxonomy_version=OLD_TAXONOMY
        )

        # By default: both excluded, nothing diffed, never by accident.
        run_pattern_frequency_diff([first, second], snapshot)
        rows = read_csv(snapshot.path / BUNDLES_FILENAME)
        assert {row["included"] for row in rows} == {TRISTATE_FALSE}
        assert diff_names(snapshot) == []

        # Opted in: the version named, the pair diffs, and the current version
        # is now the odd one out.
        opted = allocate_snapshot(experiment, now=FROZEN.replace(minute=49))
        run_pattern_frequency_diff([first, second], opted, taxonomy_version=OLD_TAXONOMY)
        rows = read_csv(opted.path / BUNDLES_FILENAME)
        assert {row["included"] for row in rows} == {TRISTATE_TRUE}
        assert diff_names(opted) == ["pattern_frequency_diff_round_1_vs_round_2.csv"]
        payload = json.loads(
            (opted.path / "pattern_frequency_diff_round_1_vs_round_2.json").read_text()
        )
        assert payload["taxonomy_version_expected"] == OLD_TAXONOMY

    def test_the_cli_exposes_the_opt_in_as_a_named_flag(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        first = write_patterns(
            tmp_path / "b1.json", [pattern(support=5)], round_index=1, taxonomy_version=OLD_TAXONOMY
        )
        second = write_patterns(
            tmp_path / "b2.json", [pattern(support=2)], round_index=2, taxonomy_version=OLD_TAXONOMY
        )

        assert (
            main([str(experiment), str(first), str(second), "--taxonomy-version", OLD_TAXONOMY])
            == 0
        )

        (snapshot_path,) = sorted(
            path for path in (experiment / ANALYSIS_DIR).iterdir() if path.is_dir()
        )
        assert (snapshot_path / PUBLISHED_FILENAME).exists()
        assert (snapshot_path / "pattern_frequency_diff_round_1_vs_round_2.csv").exists()


class TestCompletenessDrivenComparison:
    def test_every_bundle_gets_a_row_whether_or_not_it_was_compared(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        write_evidence(experiment, 2, n_records=3, recorded_records=5)
        first = write_bundle(
            record_for(experiment, 1).mining_round_path / BUNDLE_FILENAME, 1, support=5
        )
        second = write_bundle(
            record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME, 2, support=2
        )
        outside = write_bundle(tmp_path / "lifted.json", 9, support=1)

        run_pattern_frequency_diff([first, second, outside], snapshot)

        rows = {row["label"]: row for row in read_csv(snapshot.path / BUNDLES_FILENAME)}
        assert set(rows) == {"round_1", "round_2", "round_9"}
        assert rows["round_1"]["included"] == TRISTATE_TRUE
        assert rows["round_2"]["included"] == TRISTATE_FALSE
        assert rows["round_9"]["included"] == TRISTATE_TRUE
        assert rows["round_9"]["evidence_complete"] == TRISTATE_UNKNOWN
        # Only the two includable bundles were paired -- the excluded one is in
        # the table, never in a comparison.
        assert diff_names(snapshot) == ["pattern_frequency_diff_round_1_vs_round_9.csv"]

    def test_fewer_than_two_comparable_bundles_still_writes_the_completeness_table(
        self, experiment: Path, snapshot: Snapshot
    ) -> None:
        """An empty snapshot would say only that the analysis ran."""
        complete_round(experiment, 1)
        complete_round(experiment, 2)
        write_evidence(experiment, 1, n_records=3, recorded_records=5)
        write_evidence(experiment, 2, n_records=3, recorded_records=5)
        first = write_bundle(
            record_for(experiment, 1).mining_round_path / BUNDLE_FILENAME, 1, support=5
        )
        second = write_bundle(
            record_for(experiment, 2).mining_round_path / BUNDLE_FILENAME, 2, support=2
        )

        written = run_pattern_frequency_diff([first, second], snapshot)

        assert written == [snapshot.path / BUNDLES_FILENAME]
        assert diff_names(snapshot) == []
        rows = read_csv(snapshot.path / BUNDLES_FILENAME)
        assert {row["included"] for row in rows} == {TRISTATE_FALSE}
        assert all(row["exclusion_reason"] for row in rows)

    def test_the_compared_bundles_are_recorded_as_provenance_sources(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """A diff is only interpretable against the exact bundle bytes it read."""
        first = write_bundle(tmp_path / "b1.json", 1, support=5)
        second = write_bundle(tmp_path / "b2.json", 2, support=2)

        run_pattern_frequency_diff([first, second], snapshot)
        snapshot.publish()

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        bundles = [entry for entry in payload["sources"] if entry["path"].endswith(".json")]
        assert {Path(entry["path"]).name for entry in bundles} >= {"b1.json", "b2.json"}
        assert all(
            entry["sha256"]
            for entry in payload["sources"]
            if Path(entry["path"]).name in {"b1.json", "b2.json"}
        )

    def test_the_inputs_behind_the_completeness_table_are_recorded_too(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """``round_complete`` / ``evidence_complete`` come off the loop's stage
        markers and the config, not off any bundle, so a manifest naming only
        the bundles would let a marker edit move a published column without
        moving a recorded hash."""
        complete_round(experiment, 1)
        first = write_bundle(tmp_path / "b1.json", 1, support=5)
        second = write_bundle(tmp_path / "b2.json", 2, support=2)

        run_pattern_frequency_diff([first, second], snapshot)
        snapshot.publish()

        payload = json.loads((snapshot.path / PROVENANCE_FILENAME).read_text())
        recorded = {entry["path"] for entry in payload["sources"]}
        record = record_for(experiment, 1)
        assert CONFIG_FILENAME in recorded
        assert (record.round_path / ROUND_MARKER_FILENAME).relative_to(
            experiment
        ).as_posix() in recorded


# ---------------------------------------------------------------------------
# What lands on disk
# ---------------------------------------------------------------------------


class TestWrittenArtifacts:
    def test_a_pair_sharing_no_signature_writes_a_header_only_csv(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        """Header-only says "no shared failure patterns"; empty says "never ran"."""
        first = write_patterns(tmp_path / "b1.json", [], round_index=1)
        second = write_patterns(tmp_path / "b2.json", [], round_index=2)

        run_pattern_frequency_diff([first, second], snapshot)

        csv_path = snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv"
        assert csv_path.read_text().splitlines() == [",".join(DIFF_FIELDNAMES)]

    def test_the_json_summary_names_both_sides_and_counts_every_status(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        first = write_bundle(tmp_path / "b1.json", 1, support=8, n_runs=16)
        second = write_bundle(tmp_path / "b2.json", 2, support=2, n_runs=20)

        run_pattern_frequency_diff([first, second], snapshot)

        payload = json.loads(
            (snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.json").read_text()
        )
        assert payload["before"] == {"path": str(first), "label": "round_1", "n_runs": 16}
        assert payload["after"] == {"path": str(second), "label": "round_2", "n_runs": 20}
        assert payload[STATUS_PERSISTED_IMPROVED] == 1
        assert payload["total_signatures"] == 1
        assert payload["identity_hash"]
        assert payload["created_at"]

    def test_the_csv_carries_the_signature_and_both_sides_readings(
        self, experiment: Path, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        first = write_bundle(tmp_path / "b1.json", 1, support=8)
        second = write_bundle(tmp_path / "b2.json", 2, support=4)

        run_pattern_frequency_diff([first, second], snapshot)

        (row,) = read_csv(snapshot.path / "pattern_frequency_diff_round_1_vs_round_2.csv")
        assert list(row) == list(DIFF_FIELDNAMES)
        assert row["signature"].startswith("verifier_cause=wrong_answer|")
        assert row["support_rate_before"] == "0.8"
        assert row["support_rate_after"] == "0.4"
        assert row["delta_pct"] == "-0.5"
        assert row["below_support_floor_before"] == "False"


# ---------------------------------------------------------------------------
# The CLI's failure path
# ---------------------------------------------------------------------------


class TestCliFailurePath:
    def test_a_corrupt_bundle_leaves_the_snapshot_unpublished_with_the_error_recorded(
        self, experiment: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The directory stays as the audit trail; nothing will select it as latest."""
        good = write_bundle(tmp_path / "good.json", 1, support=5)
        corrupt = write_patterns(
            tmp_path / "corrupt.json", [pattern(support=5), pattern(support=9)], round_index=2
        )

        assert main([str(experiment), str(good), str(corrupt)]) == 1

        (snapshot_path,) = sorted(
            path for path in (experiment / ANALYSIS_DIR).iterdir() if path.is_dir()
        )
        assert not (snapshot_path / PUBLISHED_FILENAME).exists()
        payload = json.loads((snapshot_path / PROVENANCE_FILENAME).read_text())
        (tool,) = payload["tools"]
        assert tool["name"] == TOOL_NAME
        assert tool["ok"] is False
        assert "two patterns sharing signature" in tool["error"]
        assert PROVENANCE_FILENAME in capsys.readouterr().err

    def test_a_single_bundle_path_is_rejected_before_a_snapshot_is_allocated(
        self, experiment: Path, tmp_path: Path
    ) -> None:
        only = write_bundle(tmp_path / "only.json", 1, support=5)

        assert main([str(experiment), str(only)]) == 1
        assert not (experiment / ANALYSIS_DIR).exists()
