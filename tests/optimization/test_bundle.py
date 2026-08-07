"""Tests for evidence-bundle assembly in shrlm/optimization/bundle.py.

The bundle is the round's artifact: it must land on disk in the documented
layout, round-trip to an equal payload, keep its identity independent of when
it was made, and refuse to carry a prescription -- describing weaknesses is
stage 1's job, proposing edits is stage 2's.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from shrlm.optimization.bundle import (
    assert_no_prescription,
    build_evidence_bundle,
    build_integrity_report,
    bundle_without_timestamp,
    compute_bundle_id,
    write_bundle,
)
from shrlm.optimization.clustering import cluster_failures, compute_marginals
from shrlm.optimization.types import EvidenceBundle, FailureRecord, MiningTotals
from tests.optimization.fixtures import make_config, make_record, make_stats


def make_records() -> list[FailureRecord]:
    return [make_record("run-a"), make_record("run-b")]


def make_totals(records: list[FailureRecord]) -> MiningTotals:
    return MiningTotals(
        n_runs=len(records) + 1,
        n_failures=len(records),
        n_attributed=len(records),
        n_unattributed=0,
        n_grounded=len(records),
        n_degraded_trees=0,
    )


def make_bundle(created_at: str | None = None) -> tuple[EvidenceBundle, list[FailureRecord]]:
    records = make_records()
    bundle = build_evidence_bundle(
        config=make_config(),
        records=records,
        patterns=cluster_failures(records),
        marginals=compute_marginals(records),
        totals=make_totals(records),
        digest_coverages=[0.8, 1.0],
        created_at=created_at,
    )
    return bundle, records


class TestWriteBundle:
    def test_writes_the_documented_round_layout(self, tmp_path: Path):
        bundle, records = make_bundle()
        bundle_path = write_bundle(
            bundle, records, str(tmp_path), raw_attributions=[{"attempt": 1}]
        )
        round_dir = tmp_path / "round_03"
        assert Path(bundle_path) == round_dir / "bundle.json"
        assert (round_dir / "bundle.json").is_file()
        assert (round_dir / "records.jsonl").is_file()
        assert (round_dir / "attributions.jsonl").is_file()

    def test_bundle_json_round_trips_to_an_equal_payload(self, tmp_path: Path):
        bundle, records = make_bundle()
        bundle_path = write_bundle(bundle, records, str(tmp_path))
        with open(bundle_path) as handle:
            assert json.load(handle) == bundle.to_dict()

    def test_records_jsonl_holds_one_line_per_record(self, tmp_path: Path):
        bundle, records = make_bundle()
        write_bundle(bundle, records, str(tmp_path))
        lines = (tmp_path / "round_03" / "records.jsonl").read_text().splitlines()
        assert len(lines) == len(records)
        assert [json.loads(line)["instance_id"] for line in lines] == ["run-a", "run-b"]

    def test_attributions_file_is_omitted_when_no_raw_responses_given(self, tmp_path: Path):
        bundle, records = make_bundle()
        write_bundle(bundle, records, str(tmp_path))
        assert not (tmp_path / "round_03" / "attributions.jsonl").exists()


class TestBundleIdentity:
    def test_bundle_id_excludes_the_timestamp(self):
        early, _ = make_bundle(created_at="2026-01-01T00:00:00")
        late, _ = make_bundle(created_at="2026-06-30T23:59:59")
        assert early.created_at != late.created_at
        assert early.bundle_id == late.bundle_id
        assert bundle_without_timestamp(early) == bundle_without_timestamp(late)

    def test_bundle_without_timestamp_drops_only_created_at(self):
        bundle, _ = make_bundle()
        payload = bundle_without_timestamp(bundle)
        assert "created_at" not in payload
        assert set(bundle.to_dict()) - set(payload) == {"created_at"}

    def test_config_change_changes_the_id(self):
        records = make_records()
        assert compute_bundle_id(make_config(), records) != compute_bundle_id(
            make_config(round_index=4), records
        )

    def test_record_membership_changes_the_id(self):
        records = make_records()
        assert compute_bundle_id(make_config(), records) != compute_bundle_id(
            make_config(), records[:1]
        )


class TestPrescriptionLint:
    def test_clean_bundle_passes(self):
        bundle, _ = make_bundle()
        assert_no_prescription(bundle)

    def test_prescriptive_symptom_is_rejected(self):
        bundle, _ = make_bundle()
        pattern = bundle.patterns[0]
        tainted = dataclasses.replace(
            pattern, shared_symptoms=[*pattern.shared_symptoms, "the fix is to batch sub-calls"]
        )
        bundle.patterns[0] = tainted
        with pytest.raises(ValueError, match="prescribes a harness edit"):
            assert_no_prescription(bundle)

    def test_lint_is_case_insensitive(self):
        bundle, _ = make_bundle()
        pattern = bundle.patterns[0]
        bundle.patterns[0] = dataclasses.replace(
            pattern, verifier_evidence=["We RECOMMEND retrying the sub-call"]
        )
        with pytest.raises(ValueError, match="prescribes a harness edit"):
            assert_no_prescription(bundle)

    def test_build_evidence_bundle_runs_the_lint(self):
        records = make_records()
        patterns = cluster_failures(records)
        patterns[0] = dataclasses.replace(
            patterns[0], shared_symptoms=["to fix this, raise max_depth"]
        )
        with pytest.raises(ValueError, match="prescribes a harness edit"):
            build_evidence_bundle(
                config=make_config(),
                records=records,
                patterns=patterns,
                marginals=compute_marginals(records),
                totals=make_totals(records),
                digest_coverages=[1.0],
            )


class TestIntegrityReport:
    def test_counts_are_sums_over_the_records(self):
        records = [
            make_record("run-a", stats=make_stats(suspected_lost_subcalls=2)),
            make_record("run-b", stats=make_stats(block_attribution_reliable=False)),
            make_record("run-c", stats=make_stats(n_indeterminate=3)),
        ]
        report = build_integrity_report(records, digest_coverages=[0.5, 1.0])
        assert report.total_suspected_lost_subcalls == 2
        assert report.n_records_with_lost_subcalls == 1
        assert report.n_records_unreliable_block_attribution == 1
        # make_stats defaults contribute one indeterminate node per record.
        assert report.n_indeterminate_nodes == 1 + 1 + 3
        assert report.mean_digest_coverage == pytest.approx(0.75)

    def test_no_digests_means_full_coverage_by_convention(self):
        report = build_integrity_report([], digest_coverages=[])
        assert report.mean_digest_coverage == 1.0
