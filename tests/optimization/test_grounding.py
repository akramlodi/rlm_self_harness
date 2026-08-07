"""Tests for the sub-verification switch in shrlm/optimization/grounding.py.

derive_failing_level is the distinction the project rests on: any wrong child
means the error signal first appeared below the root, all-correct children
implicate the root's aggregation, and no descendants at all means
decomposition never happened. apply_sub_verifier is the ablation switch that
decides whether that level is a checkable fact or a model's opinion.
"""

from typing import Any

from shrlm.optimization.grounding import apply_sub_verifier, derive_failing_level
from shrlm.optimization.taxonomy import FailingLevel
from shrlm.optimization.types import CallNode, iter_nodes
from shrlm.optimization.walker import build_call_tree_from_dict
from tests.optimization.fixtures import nested_run, shallow_run


def nested_tree() -> CallNode:
    return build_call_tree_from_dict(nested_run())


def descendant_ids(root: CallNode) -> list[str]:
    return [node.node_id for node in iter_nodes(root) if node.node_id != root.node_id]


class TestDeriveFailingLevel:
    def test_any_wrong_child_means_child(self):
        root = nested_tree()
        verdicts: dict[str, bool | None] = dict.fromkeys(descendant_ids(root), True)
        verdicts["r/i0/b0/c0"] = False
        assert derive_failing_level(root, verdicts) is FailingLevel.CHILD

    def test_a_wrong_child_wins_even_when_others_are_uncheckable(self):
        # "First appears" makes the mixed case deterministic.
        root = nested_tree()
        verdicts: dict[str, bool | None] = dict.fromkeys(descendant_ids(root), None)
        verdicts["r/i0/b0/c0/i0/b0/c0"] = False
        assert derive_failing_level(root, verdicts) is FailingLevel.CHILD

    def test_all_checkable_children_correct_means_root(self):
        root = nested_tree()
        verdicts: dict[str, bool | None] = dict.fromkeys(descendant_ids(root), True)
        assert derive_failing_level(root, verdicts) is FailingLevel.ROOT

    def test_no_descendants_means_no_recursion(self):
        root = build_call_tree_from_dict(shallow_run())
        assert derive_failing_level(root, {}) is FailingLevel.NO_RECURSION

    def test_all_none_verdicts_means_undetermined(self):
        root = nested_tree()
        verdicts: dict[str, bool | None] = dict.fromkeys(descendant_ids(root), None)
        assert derive_failing_level(root, verdicts) is FailingLevel.UNDETERMINED


class TestApplySubVerifier:
    def test_no_sub_verifier_means_ungrounded_and_no_level(self):
        result = apply_sub_verifier({}, nested_tree(), None)
        assert result.grounded is False
        assert result.failing_level is None
        assert result.verdicts == {}

    def test_sub_verifier_scores_every_descendant_but_not_the_root(self):
        root = nested_tree()
        seen: list[str] = []

        def sub_verifier(instance: dict[str, Any], node: CallNode) -> bool | None:
            seen.append(node.node_id)
            return node.node_id != "r/i0/b0/c0"

        result = apply_sub_verifier({"id": "x"}, root, sub_verifier)
        assert result.grounded is True
        assert result.failing_level is FailingLevel.CHILD
        assert sorted(seen) == sorted(descendant_ids(root))
        assert "r" not in result.verdicts

    def test_verdicts_are_written_onto_the_nodes(self):
        # The digest reads node.sub_verdict, so grounding must annotate in place.
        root = nested_tree()

        def sub_verifier(instance: dict[str, Any], node: CallNode) -> bool | None:
            return True

        result = apply_sub_verifier({}, root, sub_verifier)
        for node in iter_nodes(root):
            if node.node_id == root.node_id:
                assert node.sub_verdict is None
            else:
                assert node.sub_verdict is True
                assert result.verdicts[node.node_id] is True

    def test_all_uncheckable_children_is_currently_grounded_undetermined(self):
        # Current behavior: running a sub-verifier counts as grounded even when
        # every verdict comes back None and the level is UNDETERMINED. A later
        # unit changes this case to grounded=False; this test pins the present
        # semantics so that change is visible.
        def sub_verifier(instance: dict[str, Any], node: CallNode) -> bool | None:
            return None

        result = apply_sub_verifier({}, nested_tree(), sub_verifier)
        assert result.grounded is True
        assert result.failing_level is FailingLevel.UNDETERMINED
