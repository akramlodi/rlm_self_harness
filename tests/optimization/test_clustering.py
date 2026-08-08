"""Tests for the mining step in shrlm/optimization/clustering.py.

Clustering is a GROUP BY on the four signature strings, nothing fuzzier: two
records cluster together iff every component matches exactly, and free-text
detail can never split or merge a cluster. Below-support patterns are flagged
rather than dropped, and every piece of cluster-level evidence is arithmetic
over TreeStats.
"""

import pytest

from shrlm.optimization.clustering import (
    DEFAULT_MIN_SUPPORT,
    ClusteringConfig,
    actionability,
    cluster_failures,
    compute_marginals,
    rank_patterns,
    shared_symptoms,
)
from shrlm.optimization.taxonomy import (
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    VerifierCause,
)
from shrlm.optimization.types import AttributionDetail
from tests.optimization.fixtures import make_record, make_signature, make_stats


class TestExactSignatureGrouping:
    def test_identical_signatures_cluster_together(self):
        records = [make_record("run-a"), make_record("run-b")]
        (pattern,) = cluster_failures(records)
        assert pattern.support == 2
        assert pattern.instance_ids == ["run-a", "run-b"]

    def test_any_single_component_difference_splits_the_cluster(self):
        base = make_record("run-a")
        variants = [
            make_record("run-b", signature=make_signature(cause=VerifierCause.INCOMPLETE)),
            make_record("run-c", signature=make_signature(level=FailingLevel.ROOT)),
            make_record("run-d", signature=make_signature(status=CausalStatus.CORRELATED)),
            make_record(
                "run-e", signature=make_signature(mechanism=AgentMechanism.PREMATURE_TERMINATION)
            ),
        ]
        patterns = cluster_failures([base, *variants])
        assert len(patterns) == 5
        assert all(pattern.support == 1 for pattern in patterns)

    def test_free_text_detail_never_splits_a_cluster(self):
        records = [
            make_record(
                "run-a",
                detail=AttributionDetail(
                    symptom_summary="dropped the second half at merge time",
                    evidence_node_ids=["r/i0/b0/c0"],
                    agent_mechanism_detail="merge dropped a slice",
                ),
            ),
            make_record(
                "run-b",
                detail=AttributionDetail(
                    symptom_summary="one sub-answer vanished from the final join",
                    evidence_node_ids=["r/i1/b0/c1"],
                    agent_mechanism_detail="join lost an element",
                ),
            ),
        ]
        (pattern,) = cluster_failures(records)
        assert pattern.support == 2

    def test_unattributed_records_are_excluded_from_clusters(self):
        records = [
            make_record("run-a"),
            make_record("run-b"),
            make_record("run-x", signature=None, detail=None, attribution_failed=True),
        ]
        (pattern,) = cluster_failures(records)
        assert "run-x" not in pattern.instance_ids


class TestSupportFloor:
    def test_below_support_patterns_are_flagged_never_dropped(self):
        assert DEFAULT_MIN_SUPPORT == 2
        singleton = make_record(
            "run-solo", signature=make_signature(mechanism=AgentMechanism.MISALIGNED_UNIT)
        )
        patterns = cluster_failures([make_record("run-a"), make_record("run-b"), singleton])
        by_support = {pattern.support: pattern for pattern in patterns}
        assert by_support[1].below_support_floor is True
        assert by_support[2].below_support_floor is False

    def test_min_support_is_configurable(self):
        records = [make_record("run-a"), make_record("run-b")]
        (pattern,) = cluster_failures(records, ClusteringConfig(min_support=3))
        assert pattern.below_support_floor is True


class TestDistinctInstanceSupport:
    def test_two_attempts_of_one_instance_count_as_one_instance(self):
        """KTD7/R2: repeated failing attempts of one instance are two runs of
        evidence but only one distinct instance."""
        records = [make_record("inst-1"), make_record("inst-1")]
        (pattern,) = cluster_failures(records)
        assert pattern.support == 2
        assert pattern.instance_support == 1

    def test_ordering_is_instance_support_desc_then_actionability(self):
        # Breadth across instances beats repeated attempts of one instance,
        # even when run-level support ties; instance_support ties fall back to
        # actionability.
        repeated_strong = [make_record("dup-1"), make_record("dup-1")]
        broad_weak = [
            make_record(
                f"broad-{i}",
                signature=make_signature(
                    status=CausalStatus.CORRELATED,
                    mechanism=AgentMechanism.PREMATURE_TERMINATION,
                ),
                level_grounded=False,
            )
            for i in range(2)
        ]
        singleton_weakest = [
            make_record("solo-1", signature=make_signature(mechanism=AgentMechanism.OTHER))
        ]
        patterns = cluster_failures([*repeated_strong, *broad_weak, *singleton_weakest])
        assert [(p.instance_support, p.support) for p in patterns] == [(2, 2), (1, 2), (1, 1)]
        assert patterns[0].instance_ids == ["broad-0", "broad-1"]
        assert patterns[1].instance_ids == ["dup-1", "dup-1"]
        # The instance_support tie is broken by actionability, not run support.
        assert patterns[1].actionability > patterns[2].actionability

    def test_instance_support_serializes_into_the_pattern_payload(self):
        (pattern,) = cluster_failures([make_record("inst-1"), make_record("inst-1")])
        payload = pattern.to_dict()
        assert payload["support"] == 2
        assert payload["instance_support"] == 1


class TestRanking:
    def test_instance_support_dominates_then_actionability(self):
        weak_pair = [
            make_record(
                f"run-{i}",
                signature=make_signature(status=CausalStatus.CORRELATED),
                level_grounded=False,
            )
            for i in range(2)
        ]
        strong_singleton = [
            make_record(
                "run-z", signature=make_signature(mechanism=AgentMechanism.INCOMPLETE_COVERAGE)
            )
        ]
        patterns = cluster_failures([*weak_pair, *strong_singleton])
        assert [pattern.support for pattern in patterns] == [2, 1]

        reranked = rank_patterns(list(reversed(patterns)))
        assert [pattern.support for pattern in reranked] == [2, 1]

    def test_actionability_is_one_for_a_perfect_grounded_causal_cluster(self):
        records = [make_record("run-a"), make_record("run-b")]
        assert actionability(records) == pytest.approx(1.0)

    def test_other_mechanism_earns_no_surface_credit(self):
        with_surface = [make_record("run-a")]
        without_surface = [
            make_record("run-b", signature=make_signature(mechanism=AgentMechanism.OTHER))
        ]
        assert actionability(without_surface) < actionability(with_surface)


class TestSharedSymptoms:
    def test_medians_are_computed_from_tree_stats(self):
        records = [
            make_record("run-a", stats=make_stats(n_nodes=3, n_iterations=1, max_observed_depth=1)),
            make_record("run-b", stats=make_stats(n_nodes=5, n_iterations=3, max_observed_depth=2)),
            make_record("run-c", stats=make_stats(n_nodes=9, n_iterations=7, max_observed_depth=3)),
        ]
        lines = shared_symptoms(records)
        assert "median sub-calls per run: 4" in lines
        assert "median iterations: 3" in lines
        assert "median observed depth: 2" in lines

    def test_collapse_and_fallback_counts_are_arithmetic_over_stats(self):
        records = [
            make_record("run-a", stats=make_stats(collapse_ratio=0.95)),
            make_record("run-b", stats=make_stats(terminated_by_fallback=True)),
        ]
        lines = shared_symptoms(records)
        assert "largest sub-call took >80% of the root context in 1/2 runs" in lines
        assert "answer synthesized by the iteration fallback in 1/2 runs" in lines

    def test_lost_subcall_evidence_is_reported_as_a_lower_bound(self):
        records = [make_record("run-a", stats=make_stats(suspected_lost_subcalls=2))]
        assert any("missing sub-calls" in line for line in shared_symptoms(records))


class TestMarginals:
    def test_backoff_views_are_present_and_count_correctly(self):
        records = [
            make_record("run-a"),
            make_record("run-b"),
            make_record("run-c", signature=make_signature(mechanism=AgentMechanism.OTHER)),
            make_record("run-x", signature=None, detail=None, attribution_failed=True),
        ]
        marginals = compute_marginals(records)
        assert set(marginals) == {
            "by_cause",
            "by_level",
            "by_causal_status",
            "by_mechanism",
            "by_surface",
        }
        assert marginals["by_cause"] == {"wrong_value": 3}
        assert marginals["by_mechanism"] == {"lossy_aggregation": 2, "other": 1}

    def test_other_mechanism_lands_in_the_unmapped_surface_bucket(self):
        records = [
            make_record("run-a"),
            make_record("run-b", signature=make_signature(mechanism=AgentMechanism.OTHER)),
        ]
        surfaces = compute_marginals(records)["by_surface"]
        assert surfaces == {EditableSurface.ANSWER_MIDDLEWARE.value: 1, "unmapped": 1}
