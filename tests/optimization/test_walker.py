"""Tests for the trajectory walker in shrlm/optimization/walker.py.

The walker's job is to turn nested logging dicts into an addressable tree and
to report what the log could not show. These tests exercise the three
documented defects of the logged format -- depth is derivable only from
nesting, error sub-calls vanish leaving only a printed marker, and calls at
max depth are genuinely indeterminate -- against the four canonical fixtures.
"""

import pytest

from shrlm.optimization.types import NodeKind, TraceIntegrity
from shrlm.optimization.walker import (
    build_call_tree,
    build_call_tree_from_dict,
    classify_error_kind,
    count_lost_subcalls,
    iter_skill_loads,
    walk,
)
from tests.optimization.fixtures import (
    NESTED_CHILD_PROMPT,
    NESTED_ROOT_CONTEXT,
    ROOT_MODEL,
    SKILL_INDEX,
    as_completion,
    code_block,
    completion_dict,
    fallback_run,
    iteration_entry,
    nested_run,
    no_metadata_completion,
    run_metadata,
    shallow_run,
    skilled_run,
    swallowed_error_run,
    usage,
)


class TestNestedTree:
    def test_depth_comes_from_nesting_not_from_run_metadata(self):
        # max_depth is the constant 2 in root and child alike; only nesting can
        # place the grandchild at depth 2.
        root, stats = walk(as_completion(nested_run()))
        depths = {node.node_id: node.depth for node in [root, *root.children]}
        assert depths["r"] == 0
        assert depths["r/i0/b0/c0"] == 1
        grandchild = root.children[0].children[0]
        assert grandchild.depth == 2
        assert stats.max_observed_depth == 2

    def test_node_ids_are_positional_and_stable(self):
        root = build_call_tree_from_dict(nested_run())
        assert root.node_id == "r"
        assert [child.node_id for child in root.children] == ["r/i0/b0/c0", "r/i0/b0/c1"]
        assert root.children[0].children[0].node_id == "r/i0/b0/c0/i0/b0/c0"

    def test_metadata_presence_discriminates_rlm_children_from_llm_leaves(self):
        root = build_call_tree_from_dict(nested_run())
        child, leaf = root.children
        assert child.kind is NodeKind.RLM_CHILD
        assert leaf.kind is NodeKind.LLM_LEAF

    def test_call_at_max_depth_without_metadata_is_indeterminate(self):
        # At depth == max_depth an rlm_query falls back to a plain completion,
        # byte-identical to an llm_query; the walker declines to invent a kind.
        root = build_call_tree_from_dict(nested_run())
        grandchild = root.children[0].children[0]
        assert grandchild.kind is NodeKind.INDETERMINATE
        assert grandchild.ambiguous is True

    def test_stats_count_each_kind_once(self):
        _root, stats = walk(as_completion(nested_run()))
        assert stats.n_nodes == 4
        assert stats.n_rlm_children == 1
        assert stats.n_llm_leaves == 1
        assert stats.n_indeterminate == 1
        assert stats.n_errored == 0
        assert stats.n_iterations == 2
        assert stats.recursion_available is True
        assert stats.terminated_by_fallback is False
        assert stats.trace_integrity is TraceIntegrity.COMPLETE

    def test_collapse_ratio_is_largest_child_prompt_over_root_context(self):
        _root, stats = walk(as_completion(nested_run()))
        assert stats.root_context_chars == len(NESTED_ROOT_CONTEXT)
        assert stats.max_child_prompt_chars == len(NESTED_CHILD_PROMPT)
        assert stats.collapse_ratio == pytest.approx(0.75)


class TestSwallowedErrors:
    def test_printed_marker_counts_as_a_lost_subcall(self):
        root, stats = walk(as_completion(swallowed_error_run()))
        assert count_lost_subcalls(root) == 1
        assert stats.suspected_lost_subcalls == 1
        assert stats.trace_integrity is TraceIntegrity.DEGRADED

    def test_recorded_error_completion_is_classified_errored(self):
        # Error-string response plus empty usage: the shape every error
        # completion built in rlm/core/rlm.py has.
        root, stats = walk(as_completion(swallowed_error_run()))
        (errored,) = root.children
        assert errored.kind is NodeKind.ERRORED
        assert errored.error_kind == "lm_query_failed"
        assert stats.n_errored == 1

    def test_error_kind_prefixes_are_matched_longest_first(self):
        assert classify_error_kind("Error: Child RLM completion failed - x") == (
            "child_completion_failed"
        )
        assert classify_error_kind("Error: RLM query failed - x") == "rlm_query_failed"
        assert classify_error_kind("Error: something novel") == "unknown"


class TestFallbackTermination:
    def test_trailing_codeless_iteration_with_answer_marks_fallback(self):
        _root, stats = walk(as_completion(fallback_run()))
        assert stats.terminated_by_fallback is True

    def test_committing_via_code_is_not_fallback(self):
        _root, stats = walk(as_completion(nested_run()))
        assert stats.terminated_by_fallback is False


class TestShallowTree:
    def test_root_only_run_has_sane_stats(self):
        root, stats = walk(as_completion(shallow_run()))
        assert root.node_id == "r"
        assert root.children == []
        assert stats.n_nodes == 1
        assert stats.n_rlm_children == 0
        assert stats.n_llm_leaves == 0
        assert stats.max_observed_depth == 0
        assert stats.max_child_prompt_chars == 0
        # No children means no collapse, not a division error.
        assert stats.collapse_ratio == 0.0
        assert stats.suspected_lost_subcalls == 0

    def test_max_depth_one_means_recursion_was_never_available(self):
        _root, stats = walk(as_completion(shallow_run()))
        assert stats.recursion_available is False


class TestSkillFacts:
    """The walker lifts the persisted loader events and run-start index onto the tree.

    These are the trace facts U12 records -- ``run_metadata.skill_index`` and
    ``code_blocks[].result.skill_loads`` -- and the only source the digest's
    available_skills / loaded_skills pair reads from.
    """

    def test_root_carries_the_run_start_skill_index_when_present(self):
        root = build_call_tree_from_dict(skilled_run([], skill_index=SKILL_INDEX))
        assert root.skill_index == SKILL_INDEX
        assert root.skill_index is not SKILL_INDEX  # copied, not aliased

    def test_an_index_free_trace_carries_none_not_an_empty_list(self):
        # None is "no loader was installed" (empty S10, or pre-S10); an empty
        # list would claim a loader with nothing in it.
        assert build_call_tree_from_dict(shallow_run()).skill_index is None
        assert build_call_tree_from_dict(nested_run()).skill_index is None

    def test_code_blocks_carry_their_loader_events(self):
        loads = [{"skill": "merge_slice_totals", "depth": 0}]
        root = build_call_tree_from_dict(skilled_run(loads, skill_index=SKILL_INDEX))
        (first, _commit) = root.iterations
        assert first.code_blocks[0].skill_loads == loads
        assert first.skill_loads == loads
        assert root.iterations[1].code_blocks[0].skill_loads == []

    def test_blocks_without_the_key_read_as_no_loads(self):
        # Pre-U12 traces never wrote ``skill_loads``; absence means none.
        data = shallow_run()
        del data["metadata"]["iterations"][0]["code_blocks"][0]["result"]["skill_loads"]
        root = build_call_tree_from_dict(data)
        assert root.iterations[0].code_blocks[0].skill_loads == []

    def test_iter_skill_loads_walks_the_whole_tree_in_trajectory_order(self):
        child_block = code_block(
            code="proc = load_skill('check_slice_coverage')",
            skill_loads=[{"skill": "check_slice_coverage", "depth": 1}],
        )
        child = {
            "root_model": ROOT_MODEL,
            "prompt": "check coverage",
            "response": "covered",
            "usage_summary": usage(),
            "execution_time": 0.3,
            "metadata": {
                "run_metadata": run_metadata(max_depth=2, skill_index=SKILL_INDEX),
                "iterations": [
                    iteration_entry(1, "...", [child_block], final_answer="covered"),
                ],
            },
        }
        root_block = code_block(
            code="proc = load_skill('merge_slice_totals')\nrlm_query('check coverage')",
            rlm_calls=[child],
            skill_loads=[{"skill": "merge_slice_totals", "depth": 0}],
        )
        root = build_call_tree_from_dict(
            completion_dict(
                prompt="total?",
                response="41",
                iterations=[iteration_entry(1, "...", [root_block], final_answer="41")],
                max_depth=2,
                skill_index=SKILL_INDEX,
            )
        )
        assert list(iter_skill_loads(root)) == [
            ("merge_slice_totals", 0),
            ("check_slice_coverage", 1),
        ]
        # The child's own run-start record is lifted too.
        assert root.children[0].skill_index == SKILL_INDEX

    def test_skill_facts_survive_to_dict(self):
        loads = [{"skill": "merge_slice_totals", "depth": 0}]
        root = build_call_tree_from_dict(skilled_run(loads, skill_index=SKILL_INDEX))
        payload = root.to_dict()
        assert payload["skill_index"] == SKILL_INDEX
        assert payload["iterations"][0]["code_blocks"][0]["skill_loads"] == loads


class TestMissingTrajectory:
    def test_walk_raises_the_documented_error_without_metadata(self):
        with pytest.raises(ValueError, match="no trajectory metadata"):
            walk(no_metadata_completion())

    def test_build_call_tree_raises_the_documented_error_without_metadata(self):
        with pytest.raises(ValueError, match="logger=RLMLogger"):
            build_call_tree(no_metadata_completion())

    def test_build_from_dict_raises_without_metadata(self):
        data = shallow_run()
        del data["metadata"]
        with pytest.raises(ValueError, match="no trajectory metadata"):
            build_call_tree_from_dict(data)
