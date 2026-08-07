"""Tests for the data model in shrlm/optimization/types.py.

The load-bearing decision is that FailureSignature holds only the four enum
values while all free text lives on AttributionDetail. These tests pin the
signature key shape, the separation from free text, and the Verdict
consistency contract.
"""

import pytest

from shrlm.optimization.taxonomy import (
    AgentMechanism,
    CausalStatus,
    EditableSurface,
    FailingLevel,
    VerifierCause,
)
from shrlm.optimization.types import (
    AttributionDetail,
    CallNode,
    FailureSignature,
    NodeKind,
    Verdict,
    iter_nodes,
)
from tests.optimization.fixtures import make_record, make_signature


class TestFailureSignature:
    def test_key_is_the_four_tuple_of_strings_in_signature_order(self):
        signature = FailureSignature(
            verifier_cause=VerifierCause.INCOMPLETE,
            failing_level=FailingLevel.ROOT,
            causal_status=CausalStatus.CONTRIBUTING,
            agent_mechanism=AgentMechanism.LOSSY_AGGREGATION,
        )
        key = signature.key()
        assert key == ("incomplete", "root", "contributing", "lossy_aggregation")
        assert all(isinstance(part, str) for part in key)

    def test_identical_enums_with_different_free_text_produce_equal_keys(self):
        # The whole reason AttributionDetail is a separate class: wording must
        # not fragment clusters.
        first = make_record(
            "run-a",
            detail=AttributionDetail(
                symptom_summary="the merge dropped the second half",
                evidence_node_ids=["r/i0/b0/c0"],
            ),
        )
        second = make_record(
            "run-b",
            detail=AttributionDetail(
                symptom_summary="aggregation lost one sub-answer entirely",
                evidence_node_ids=["r/i1/b0/c2"],
            ),
        )
        assert first.signature is not None and second.signature is not None
        assert first.signature.key() == second.signature.key()
        assert first.signature == second.signature

    def test_signature_is_hashable_and_usable_as_a_dict_key(self):
        counts = {make_signature(): 1}
        counts[make_signature()] = counts.get(make_signature(), 0) + 1
        assert counts == {make_signature(): 2}

    def test_surface_is_the_taxonomy_mapping(self):
        swallowed = make_signature(mechanism=AgentMechanism.SWALLOWED_SUBCALL_ERROR)
        assert swallowed.surface() is EditableSurface.ERROR_POLICY

    def test_other_mechanism_has_no_surface(self):
        assert make_signature(mechanism=AgentMechanism.OTHER).surface() is None

    def test_to_dict_carries_exactly_the_four_components_as_strings(self):
        payload = make_signature().to_dict()
        assert set(payload) == {
            "verifier_cause",
            "failing_level",
            "causal_status",
            "agent_mechanism",
        }
        assert all(isinstance(value, str) for value in payload.values())


class TestVerdict:
    def test_passing_verdict_with_a_cause_is_inconsistent(self):
        with pytest.raises(ValueError, match="must not carry"):
            Verdict(passed=True, cause=VerifierCause.WRONG_VALUE, gold="B", produced="B")

    def test_failing_verdict_without_a_cause_is_inconsistent(self):
        with pytest.raises(ValueError, match="must carry"):
            Verdict(passed=False, cause=None, gold="B", produced="C")

    def test_consistent_verdicts_construct(self):
        passing = Verdict(passed=True, cause=None, gold="B", produced="B")
        failing = Verdict(passed=False, cause=VerifierCause.SPURIOUS, gold="B", produced="B,C")
        assert passing.to_dict()["cause"] is None
        assert failing.to_dict()["cause"] == "spurious"


def leaf(node_id: str, parent_id: str, depth: int) -> CallNode:
    return CallNode(
        node_id=node_id,
        parent_id=parent_id,
        kind=NodeKind.LLM_LEAF,
        depth=depth,
        model="mock-model",
        prompt="p",
        response="r",
        prompt_chars=1,
        response_chars=1,
        execution_time=0.1,
    )


class TestIterNodes:
    def test_depth_first_root_first_in_trajectory_order(self):
        root = leaf("r", parent_id="", depth=0)
        root.parent_id = None
        first = leaf("r/i0/b0/c0", "r", 1)
        first.children.append(leaf("r/i0/b0/c0/i0/b0/c0", "r/i0/b0/c0", 2))
        root.children = [first, leaf("r/i0/b0/c1", "r", 1)]
        assert [node.node_id for node in iter_nodes(root)] == [
            "r",
            "r/i0/b0/c0",
            "r/i0/b0/c0/i0/b0/c0",
            "r/i0/b0/c1",
        ]
