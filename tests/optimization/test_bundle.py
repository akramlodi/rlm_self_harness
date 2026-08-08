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
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import (
    AttributionErrorKind,
    EvidenceBundle,
    FailureRecord,
    MiningTotals,
)
from tests.optimization.fixtures import make_config, make_record, make_stats, make_verdict


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

    def test_sub_verifier_flip_alone_changes_the_id(self):
        """R4: the ablation switch is identity-bearing. Two mines of the same
        round that differ only in ``sub_verifier_enabled`` must mint different
        bundle ids, or the two modes' evidence could silently collide."""
        records = make_records()
        grounded = compute_bundle_id(make_config(sub_verifier_enabled=True), records)
        ablated = compute_bundle_id(make_config(sub_verifier_enabled=False), records)
        assert grounded != ablated

    def test_verifier_config_changes_the_id(self):
        records = make_records()
        assert compute_bundle_id(make_config(), records) != compute_bundle_id(
            make_config(verifier_config={"environment": "graphwalks", "pass_f1_threshold": 1.0}),
            records,
        )

    def test_sampling_seed_lands_in_the_bundle(self):
        bundle = build_evidence_bundle(
            config=make_config(sampling_seed=7),
            records=make_records(),
            patterns=cluster_failures(make_records()),
            marginals=compute_marginals(make_records()),
            totals=make_totals(make_records()),
            digest_coverages=[1.0],
        )
        assert bundle.to_dict()["config"]["sampling_seed"] == 7


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
            pattern, shared_symptoms=["We RECOMMEND retrying the sub-call"]
        )
        with pytest.raises(ValueError, match="prescribes a harness edit"):
            assert_no_prescription(bundle)

    def test_quoted_model_output_is_exempt_from_the_lint(self):
        """A wrong answer that happens to say "instead of" is evidence, not a
        recommendation: verifier_evidence embeds verdict.produced verbatim and
        must never crash bundle emission."""
        records = [
            make_record(
                "run-a",
                verdict=make_verdict(produced="use depth 3 instead of 2; the fix is obvious"),
            ),
            make_record("run-b"),
        ]
        bundle = build_evidence_bundle(
            config=make_config(),
            records=records,
            patterns=cluster_failures(records),
            marginals=compute_marginals(records),
            totals=make_totals(records),
            digest_coverages=[1.0, 1.0],
        )
        assert any("instead of" in line for line in bundle.patterns[0].verifier_evidence)

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


class TestNonClobberingWrites:
    def test_byte_identical_rewrite_succeeds(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        first_path = Path(write_bundle(bundle, records, str(tmp_path)))
        before = first_path.read_bytes()
        second_path = Path(write_bundle(bundle, records, str(tmp_path)))
        assert second_path == first_path
        assert second_path.read_bytes() == before

    def test_different_bundle_id_is_refused_naming_both_ids(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        write_bundle(bundle, records, str(tmp_path))

        other_records = [make_record("run-a")]
        other = build_evidence_bundle(
            config=make_config(),
            records=other_records,
            patterns=cluster_failures(other_records),
            marginals=compute_marginals(other_records),
            totals=make_totals(other_records),
            digest_coverages=[1.0],
            created_at="2026-01-01T00:00:00",
        )
        assert other.bundle_id != bundle.bundle_id
        with pytest.raises(ValueError) as excinfo:
            write_bundle(other, other_records, str(tmp_path))
        assert bundle.bundle_id in str(excinfo.value)
        assert other.bundle_id in str(excinfo.value)
        # The audited artifact is untouched.
        round_path = tmp_path / "round_03"
        assert json.loads((round_path / "bundle.json").read_text()) == bundle.to_dict()

    def test_same_id_with_divergent_content_is_refused(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        write_bundle(bundle, records, str(tmp_path))

        divergent, _ = make_bundle(created_at="2026-01-01T00:00:00")
        divergent.patterns[0] = dataclasses.replace(
            divergent.patterns[0], shared_symptoms=["median iterations: 9"]
        )
        assert divergent.bundle_id == bundle.bundle_id
        with pytest.raises(ValueError, match="diverge"):
            write_bundle(divergent, records, str(tmp_path))
        round_path = tmp_path / "round_03"
        assert json.loads((round_path / "bundle.json").read_text()) == bundle.to_dict()

    @pytest.mark.parametrize(
        "corruption",
        ["truncated", "not-a-bundle"],
        ids=["truncated json", "parseable but no bundle_id"],
    )
    def test_corrupt_existing_bundle_names_the_corruption_and_the_recovery(
        self, tmp_path: Path, corruption: str
    ):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        bundle_path = Path(write_bundle(bundle, records, str(tmp_path)))
        text = bundle_path.read_text()
        bundle_path.write_text(text[: len(text) // 2] if corruption == "truncated" else "{}")

        with pytest.raises(ValueError, match="corrupt") as excinfo:
            write_bundle(bundle, records, str(tmp_path))
        # The error names the recovery step instead of a bare traceback.
        assert "Delete" in str(excinfo.value)

    def test_overwrite_replaces_divergent_content_deliberately(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        write_bundle(bundle, records, str(tmp_path))

        divergent, _ = make_bundle(created_at="2026-01-01T00:00:00")
        divergent.patterns[0] = dataclasses.replace(
            divergent.patterns[0], shared_symptoms=["median iterations: 9"]
        )
        with pytest.raises(ValueError, match="diverge"):
            write_bundle(divergent, records, str(tmp_path))

        bundle_path = Path(write_bundle(divergent, records, str(tmp_path), overwrite=True))
        assert json.loads(bundle_path.read_text()) == divergent.to_dict()

    def test_overwrite_also_replaces_a_corrupt_bundle(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        bundle_path = Path(write_bundle(bundle, records, str(tmp_path)))
        bundle_path.write_text("{ truncated")

        write_bundle(bundle, records, str(tmp_path), overwrite=True)
        assert json.loads(bundle_path.read_text()) == bundle.to_dict()

    def test_atomic_write_leaves_no_tmp_file_on_success(self, tmp_path: Path):
        bundle, records = make_bundle()
        bundle_path = Path(write_bundle(bundle, records, str(tmp_path)))
        assert bundle_path.is_file()
        assert list(bundle_path.parent.glob("*.tmp")) == []


class TestBundleDestination:
    def make_other_bundle(self) -> tuple[EvidenceBundle, list[FailureRecord]]:
        """A second bundle over the same round with a different config, as the
        sub-verification ablation produces (only the mode flag differs;
        ``make_config`` now defaults to the ablated ``False``, so the other
        mode here is the grounded one)."""
        records = make_records()
        bundle = build_evidence_bundle(
            config=make_config(sub_verifier_enabled=True),
            records=records,
            patterns=cluster_failures(records),
            marginals=compute_marginals(records),
            totals=make_totals(records),
            digest_coverages=[0.8, 1.0],
            created_at="2026-01-01T00:00:00",
        )
        return bundle, records

    def test_bundle_dir_names_the_triplet_directory_directly(self, tmp_path: Path):
        bundle, records = make_bundle()
        destination = tmp_path / "round_03" / "bundles" / "ablated"
        bundle_path = write_bundle(
            bundle,
            records,
            str(tmp_path),
            raw_attributions=[{"attempt": 1}],
            bundle_dir=str(destination),
        )
        assert Path(bundle_path) == destination / "bundle.json"
        assert (destination / "bundle.json").is_file()
        assert (destination / "records.jsonl").is_file()
        assert (destination / "attributions.jsonl").is_file()
        # Nothing lands at the round root: the parameter bypasses the
        # round_NN join entirely.
        assert not (tmp_path / "round_03" / "bundle.json").exists()

    def test_two_bundles_coexist_in_one_round(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        other, other_records = self.make_other_bundle()
        assert other.bundle_id != bundle.bundle_id

        write_bundle(bundle, records, str(tmp_path))
        write_bundle(
            other,
            other_records,
            str(tmp_path),
            bundle_dir=str(tmp_path / "round_03" / "bundles" / "ablated"),
        )

        round_path = tmp_path / "round_03"
        assert json.loads((round_path / "bundle.json").read_text()) == bundle.to_dict()
        ablated = round_path / "bundles" / "ablated"
        assert json.loads((ablated / "bundle.json").read_text()) == other.to_dict()

    def test_non_clobber_guard_is_independent_per_destination(self, tmp_path: Path):
        bundle, records = make_bundle(created_at="2026-01-01T00:00:00")
        other, other_records = self.make_other_bundle()
        ablated_dir = str(tmp_path / "round_03" / "bundles" / "ablated")
        write_bundle(bundle, records, str(tmp_path))
        write_bundle(other, other_records, str(tmp_path), bundle_dir=ablated_dir)

        # The round root still refuses a different bundle...
        with pytest.raises(ValueError, match="refusing to overwrite"):
            write_bundle(other, other_records, str(tmp_path))
        # ...and the subdirectory refuses one too, independently.
        with pytest.raises(ValueError, match="refusing to overwrite"):
            write_bundle(bundle, records, str(tmp_path), bundle_dir=ablated_dir)
        # Byte-identical rewrites stay idempotent within each destination.
        write_bundle(bundle, records, str(tmp_path))
        write_bundle(other, other_records, str(tmp_path), bundle_dir=ablated_dir)


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

    def test_known_substrate_biases_name_a1_and_a2(self):
        report = build_integrity_report([], digest_coverages=[])
        assert [bias.defect_id for bias in report.known_substrate_biases] == ["A1", "A2"]
        payload = report.to_dict()
        assert [entry["defect_id"] for entry in payload["known_substrate_biases"]] == ["A1", "A2"]
        for entry in payload["known_substrate_biases"]:
            assert entry["summary"]
            assert entry["effect"]

    def test_operational_counts_are_sums_over_the_records(self):
        # A legacy record: no typed kind, recognized by the message prefix.
        legacy_transport = make_record(
            "run-d", signature=None, detail=None, attribution_failed=True
        )
        legacy_transport.attribution_error = "transport failure: connection reset"
        # A current record: typed kind, message carries no prefix.
        typed_transport = make_record("run-e", signature=None, detail=None, attribution_failed=True)
        typed_transport.attribution_error = "connection reset"
        typed_transport.attribution_error_kind = AttributionErrorKind.TRANSPORT
        # A typed rejection whose message happens to carry the prefix: the
        # typed field wins over the legacy prefix heuristic.
        typed_rejection = make_record("run-f", signature=None, detail=None, attribution_failed=True)
        typed_rejection.attribution_error = "transport failure mentioned in a rejection"
        typed_rejection.attribution_error_kind = AttributionErrorKind.REJECTION
        records = [
            make_record("run-a", signature=None, detail=None, attribution_failed=True),
            make_record("run-b", level_grounded=False),
            make_record("run-c", verdict=make_verdict(cause=VerifierCause.RESOURCE_TERMINATED)),
            legacy_transport,
            typed_transport,
            typed_rejection,
        ]
        report = build_integrity_report(records, digest_coverages=[1.0])
        assert report.n_unattributed == 4
        assert report.n_ungrounded == 1
        assert report.n_resource_terminated == 1
        assert report.n_transport_errors == 2

    def test_zero_failure_round_builds_a_valid_bundle(self):
        bundle = build_evidence_bundle(
            config=make_config(),
            records=[],
            patterns=[],
            marginals=compute_marginals([]),
            totals=MiningTotals(
                n_runs=3,
                n_failures=0,
                n_attributed=0,
                n_unattributed=0,
                n_grounded=0,
                n_degraded_trees=0,
            ),
            digest_coverages=[],
        )
        assert bundle.patterns == []
        integrity = bundle.to_dict()["integrity"]
        assert integrity["n_unattributed"] == 0
        assert integrity["n_ungrounded"] == 0
        assert integrity["n_resource_terminated"] == 0
        assert integrity["n_transport_errors"] == 0
        assert integrity["known_substrate_biases"]
