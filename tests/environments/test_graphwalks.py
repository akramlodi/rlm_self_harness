"""Tests for the GraphWalks environment: loader transform, Verifier, SubVerifier.

Everything runs offline against small fixture graphs. The only network-touching
seam, ``fetch_rows``, is monkeypatched with fixture rows so ``load_graphwalks``
is exercised end to end -- file routing, char-window filtering, and revision
plumbing -- without pyarrow, huggingface_hub, or the network.
"""

from typing import Any

import pytest

import shrlm.environments.graphwalks as graphwalks
from shrlm.environments.graphwalks import (
    DATASET_FILE,
    DATASET_REPO,
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    GraphWalksSubVerifier,
    GraphWalksVerifier,
    extract_answer_nodes,
    fetch_rows,
    load_graphwalks,
    parse_subproblem,
    row_to_instance,
    sample_rows,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind

# A ~7-node fixture graph: a -> {b, c}, b -> d, c -> d, d -> e, f -> a.
FIXTURE_EDGES = "a -> b\na -> c\nb -> d\nc -> d\nd -> e\nf -> a"

BFS_PROMPT = (
    "You are given a directed graph as an edge list:\n"
    f"{FIXTURE_EDGES}\n"
    "Operation:\n"
    "Perform a BFS from node a with depth 1.\n"
    "End your response with a line 'Final Answer: [n1, n2, ...]'."
)

PARENTS_PROMPT = (
    "You are given a directed graph as an edge list:\n"
    f"{FIXTURE_EDGES}\n"
    "Operation:\n"
    "Find the parents of node d.\n"
    "End your response with a line 'Final Answer: [n1, n2, ...]'."
)


def make_row(
    prompt: str = BFS_PROMPT,
    problem_type: str = "bfs",
    answer_nodes: tuple[str, ...] = ("b", "c"),
    prompt_chars: int | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "problem_type": problem_type,
        "answer_nodes": list(answer_nodes),
        "prompt_chars": len(prompt) if prompt_chars is None else prompt_chars,
    }


def make_node(
    prompt: str | dict[str, Any],
    response: str,
    kind: NodeKind = NodeKind.RLM_CHILD,
    depth: int = 1,
    error_kind: str | None = None,
) -> CallNode:
    return CallNode(
        node_id="r/i0/b0/c0",
        parent_id="r",
        kind=kind,
        depth=depth,
        model="mock-model",
        prompt=prompt,
        response=response,
        prompt_chars=len(str(prompt)),
        response_chars=len(response),
        execution_time=0.1,
        error_kind=error_kind,
    )


class TestExtractAnswerNodes:
    def test_no_final_answer_line_is_none(self):
        assert extract_answer_nodes("I believe the answer is b and c.") is None

    def test_final_answer_without_brackets_is_none(self):
        assert extract_answer_nodes("Final Answer: b, c") is None

    def test_parsed_empty_list_is_empty_not_none(self):
        assert extract_answer_nodes("reasoning...\nFinal Answer: []") == []

    def test_parses_trailing_line_only(self):
        assert extract_answer_nodes("Final Answer: [x]\njust kidding") is None
        assert extract_answer_nodes("some text\nFinal Answer: [b, c]") == ["b", "c"]

    def test_first_bracket_pair_wins_over_trailing_commentary(self):
        # A greedy parse would swallow "] (excluding [c" into the item list
        # and grade a correct answer wrong.
        assert extract_answer_nodes("Final Answer: [a, b] (excluding [c])") == ["a", "b"]

    def test_redundantly_nested_brackets_parse_to_the_inner_items(self):
        # Stray bracket characters are stripped per item: node ids are \w+
        # and can never legitimately contain brackets.
        assert extract_answer_nodes("Final Answer: [[a, b]]") == ["a", "b"]


class TestRowToInstance:
    def test_id_is_content_derived_and_seed_independent(self):
        row = make_row()
        first = row_to_instance(row, sample_seed=0, sample_index=3)
        second = row_to_instance(row, sample_seed=7, sample_index=1)
        assert first["id"] == second["id"]
        assert first["id"].startswith("bfs-")
        assert first["sample_seed"] == 0 and first["sample_index"] == 3
        assert second["sample_seed"] == 7 and second["sample_index"] == 1

    def test_ids_differ_across_prompts(self):
        bfs = row_to_instance(make_row(), sample_seed=0, sample_index=0)
        parents = row_to_instance(
            make_row(prompt=PARENTS_PROMPT, problem_type="parents", answer_nodes=("b", "c")),
            sample_seed=0,
            sample_index=1,
        )
        assert bfs["id"] != parents["id"]

    def test_question_is_the_operation_line(self):
        instance = row_to_instance(make_row(), sample_seed=0, sample_index=0)
        assert instance["question"] == "Perform a BFS from node a with depth 1."
        parents = row_to_instance(
            make_row(prompt=PARENTS_PROMPT, problem_type="parents"), sample_seed=0, sample_index=0
        )
        assert parents["question"] == "Find the parents of node d."

    def test_question_falls_back_to_first_nonempty_line(self):
        row = make_row(prompt="\nSolve this puzzle.\nno operation line here.")
        instance = row_to_instance(row, sample_seed=0, sample_index=0)
        assert instance["question"] == "Solve this puzzle."


class TestSampleRows:
    def make_pool(self) -> list[dict[str, Any]]:
        rows = []
        for index in range(10):
            rows.append(make_row(prompt=f"{BFS_PROMPT}\nvariant {index}", problem_type="bfs"))
            rows.append(
                make_row(prompt=f"{PARENTS_PROMPT}\nvariant {index}", problem_type="parents")
            )
        return rows

    def test_same_seed_same_sample(self):
        pool = self.make_pool()
        first = sample_rows(pool, ("bfs", "parents"), limit=4, seed=0)
        second = sample_rows(pool, ("bfs", "parents"), limit=4, seed=0)
        assert [row["prompt"] for row in first] == [row["prompt"] for row in second]

    def test_different_seed_different_sample_but_stable_ids(self):
        pool = self.make_pool()
        first = sample_rows(pool, ("bfs", "parents"), limit=6, seed=0)
        second = sample_rows(pool, ("bfs", "parents"), limit=6, seed=1)
        assert [row["prompt"] for row in first] != [row["prompt"] for row in second]
        # Content-derived ids: any row landing in both samples keeps its id.
        first_ids = {
            row["prompt"]: row_to_instance(row, 0, index)["id"] for index, row in enumerate(first)
        }
        second_ids = {
            row["prompt"]: row_to_instance(row, 1, index)["id"] for index, row in enumerate(second)
        }
        overlap = set(first_ids) & set(second_ids)
        assert overlap, "fixture pools are small enough that samples must overlap"
        for prompt in overlap:
            assert first_ids[prompt] == second_ids[prompt]

    def test_balanced_across_problem_types(self):
        picked = sample_rows(self.make_pool(), ("bfs", "parents"), limit=4, seed=3)
        types = [row["problem_type"] for row in picked]
        assert types.count("bfs") == 2 and types.count("parents") == 2

    def test_no_limit_returns_all_rows(self):
        pool = self.make_pool()
        assert len(sample_rows(pool, ("bfs", "parents"), limit=None, seed=0)) == len(pool)

    def test_odd_limit_returns_exactly_limit_rows(self):
        # limit=5 over 2 types: the balanced draw yields 2 per type and the
        # remainder is topped up, never silently dropped.
        picked = sample_rows(self.make_pool(), ("bfs", "parents"), limit=5, seed=0)
        assert len(picked) == 5
        types = [row["problem_type"] for row in picked]
        assert sorted([types.count("bfs"), types.count("parents")]) == [2, 3]

    def test_limit_beyond_availability_returns_all_rows(self):
        pool = self.make_pool()
        picked = sample_rows(pool, ("bfs", "parents"), limit=len(pool) + 5, seed=0)
        assert len(picked) == len(pool)

    def test_exhausted_type_is_topped_up_from_the_other(self):
        pool = [
            make_row(prompt=f"{BFS_PROMPT}\nvariant {i}", problem_type="bfs") for i in range(10)
        ]
        pool.append(make_row(prompt=PARENTS_PROMPT, problem_type="parents"))
        picked = sample_rows(pool, ("bfs", "parents"), limit=6, seed=0)
        types = [row["problem_type"] for row in picked]
        assert len(picked) == 6
        assert types.count("parents") == 1 and types.count("bfs") == 5

    def test_topped_up_sample_is_deterministic(self):
        pool = self.make_pool()
        first = sample_rows(pool, ("bfs", "parents"), limit=5, seed=2)
        second = sample_rows(pool, ("bfs", "parents"), limit=5, seed=2)
        assert [row["prompt"] for row in first] == [row["prompt"] for row in second]


class TestFetchRowsDownloadTimeout:
    """``fetch_rows``'s HTTP bound, exercised one level below the
    ``TestLoadGraphwalks`` tests above: the real ``huggingface_hub`` and
    ``pyarrow`` imports run (skipped if the ``graphwalks`` extra is not
    installed), but ``hf_hub_download`` and ``pq.read_table`` are stubbed so
    nothing ever touches the network."""

    def _patch(self, monkeypatch):
        huggingface_hub = pytest.importorskip("huggingface_hub")
        hf_constants = pytest.importorskip("huggingface_hub.constants")
        pq = pytest.importorskip("pyarrow.parquet")

        seen: dict[str, Any] = {}

        def fake_hf_hub_download(**kwargs: Any) -> str:
            seen["kwargs"] = kwargs
            seen["timeout_during_call"] = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
            return "/fake/path.parquet"

        class FakeTable:
            def to_pylist(self) -> list[dict[str, Any]]:
                return [{"prompt": "x"}]

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
        monkeypatch.setattr(pq, "read_table", lambda path: FakeTable())
        return hf_constants, seen

    def test_default_timeout_bounds_both_the_etag_request_and_the_transfer(self, monkeypatch):
        hf_constants, seen = self._patch(monkeypatch)
        original_timeout = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT

        rows = fetch_rows("some/repo", "some_file.parquet", revision="abc")

        assert rows == [{"prompt": "x"}]
        assert seen["kwargs"]["etag_timeout"] == DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
        assert seen["timeout_during_call"] == DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
        # Process-wide state is restored, not left mutated for unrelated
        # huggingface_hub calls elsewhere in the process.
        assert hf_constants.HF_HUB_DOWNLOAD_TIMEOUT == original_timeout

    def test_custom_timeout_overrides_the_default(self, monkeypatch):
        hf_constants, seen = self._patch(monkeypatch)

        fetch_rows("some/repo", "some_file.parquet", revision=None, download_timeout_seconds=5.0)

        assert seen["kwargs"]["etag_timeout"] == 5.0
        assert seen["timeout_during_call"] == 5.0

    def test_timeout_is_restored_even_if_the_download_raises(self, monkeypatch):
        huggingface_hub = pytest.importorskip("huggingface_hub")
        hf_constants = pytest.importorskip("huggingface_hub.constants")
        pytest.importorskip("pyarrow.parquet")
        original_timeout = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT

        def failing_hf_hub_download(**kwargs: Any) -> str:
            raise TimeoutError("simulated stalled download")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", failing_hf_hub_download)

        with pytest.raises(TimeoutError):
            fetch_rows(
                "some/repo", "some_file.parquet", revision=None, download_timeout_seconds=5.0
            )

        assert hf_constants.HF_HUB_DOWNLOAD_TIMEOUT == original_timeout


class TestLoadGraphwalks:
    """``load_graphwalks`` over a monkeypatched ``fetch_rows`` seam: dataset
    file routing, the [min_chars, max_chars] window, and revision plumbing."""

    def make_pool(self) -> list[dict[str, Any]]:
        rows = []
        for index in range(4):
            rows.append(
                make_row(
                    prompt=f"{BFS_PROMPT}\nshort variant {index}",
                    problem_type="bfs",
                    prompt_chars=1_000 + index,
                )
            )
            rows.append(
                make_row(
                    prompt=f"{PARENTS_PROMPT}\nlong variant {index}",
                    problem_type="parents",
                    prompt_chars=300_000 + index,
                )
            )
        return rows

    def patch_fetch(self, monkeypatch, rows: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
        calls: list[tuple[str, str, Any]] = []

        def fake_fetch(dataset_repo: str, dataset_file: str, revision: str | None):
            calls.append((dataset_repo, dataset_file, revision))
            return [dict(row) for row in rows]

        monkeypatch.setattr(graphwalks, "fetch_rows", fake_fetch)
        return calls

    def test_defaults_preserve_today_s_behavior(self, monkeypatch):
        # Default call: 128k-and-shorter file, unpinned revision, <= 128k cap,
        # no lower bound -- exactly what the hardcoded loader did.
        calls = self.patch_fetch(monkeypatch, self.make_pool())
        instances = load_graphwalks()
        assert calls == [(DATASET_REPO, DATASET_FILE, None)]
        assert len(instances) == 4
        assert {instance["problem_type"] for instance in instances} == {"bfs"}

    def test_long_split_routes_to_the_long_file_and_respects_the_floor(self, monkeypatch):
        calls = self.patch_fetch(monkeypatch, self.make_pool())
        instances = load_graphwalks(
            dataset_file="graphwalks_256k_to_1mil.parquet",
            min_chars=300_000,
            max_chars=None,
            revision="pinned-sha",
        )
        assert calls == [(DATASET_REPO, "graphwalks_256k_to_1mil.parquet", "pinned-sha")]
        assert len(instances) == 4
        assert {instance["problem_type"] for instance in instances} == {"parents"}

    def test_min_chars_floor_is_inclusive(self, monkeypatch):
        self.patch_fetch(monkeypatch, self.make_pool())
        instances = load_graphwalks(min_chars=1_003, max_chars=128_000)
        assert len(instances) == 1
        assert instances[0]["problem_type"] == "bfs"

    def test_dataset_repo_is_parameterized(self, monkeypatch):
        calls = self.patch_fetch(monkeypatch, self.make_pool())
        load_graphwalks(dataset_repo="someone-else/graphwalks")
        assert calls == [("someone-else/graphwalks", DATASET_FILE, None)]

    def test_instances_carry_seed_provenance(self, monkeypatch):
        self.patch_fetch(monkeypatch, self.make_pool())
        instances = load_graphwalks(seed=7, limit=2)
        assert len(instances) == 2
        assert all(instance["sample_seed"] == 7 for instance in instances)
        assert sorted(instance["sample_index"] for instance in instances) == [0, 1]


class TestGraphWalksVerifier:
    def instance(self, answer_nodes: tuple[str, ...] = ("b", "c")) -> dict[str, Any]:
        return row_to_instance(make_row(answer_nodes=answer_nodes), sample_seed=0, sample_index=0)

    def test_exact_match_passes(self):
        verdict = GraphWalksVerifier()(self.instance(), "reasoning\nFinal Answer: [c, b]")
        assert verdict.passed is True
        assert verdict.cause is None

    def test_trailing_bracketed_commentary_does_not_corrupt_the_answer(self):
        verdict = GraphWalksVerifier()(self.instance(), "Final Answer: [b, c] (visited [a])")
        assert verdict.passed is True

    def test_gold_and_produced_are_sorted(self):
        verdict = GraphWalksVerifier()(
            self.instance(answer_nodes=("c", "b")), "Final Answer: [c, b]"
        )
        assert verdict.gold == "[b, c]"
        assert verdict.produced == "[b, c]"

    def test_prose_fallback_answer_is_wrong_format(self):
        # A fallback-synthesized root answer has no Final Answer line at all:
        # the answer never entered the environment's channel, so the cause is
        # WRONG_FORMAT, not NO_ANSWER.
        verdict = GraphWalksVerifier()(self.instance(), "The nodes reachable are b and c.")
        assert verdict.passed is False
        assert verdict.cause is VerifierCause.WRONG_FORMAT
        assert verdict.produced == "The nodes reachable are b and c."

    def test_parsed_empty_against_nonempty_gold_is_no_answer(self):
        verdict = GraphWalksVerifier()(self.instance(), "Final Answer: []")
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_missing_only_is_incomplete(self):
        verdict = GraphWalksVerifier()(self.instance(), "Final Answer: [b]")
        assert verdict.cause is VerifierCause.INCOMPLETE

    def test_extra_only_is_spurious(self):
        verdict = GraphWalksVerifier()(self.instance(), "Final Answer: [b, c, z]")
        assert verdict.cause is VerifierCause.SPURIOUS

    def test_missing_and_extra_is_mixed_set_error(self):
        verdict = GraphWalksVerifier()(self.instance(), "Final Answer: [b, z]")
        assert verdict.cause is VerifierCause.MIXED_SET_ERROR

    def test_both_empty_sets_pass(self):
        verdict = GraphWalksVerifier()(self.instance(answer_nodes=()), "Final Answer: []")
        assert verdict.passed is True
        assert verdict.gold == "[]"

    def test_config_names_the_grading_facts(self):
        config = GraphWalksVerifier().config()
        assert config["environment"] == "graphwalks"
        assert config["pass_f1_threshold"] == 1.0
        assert config["extraction_rule"] == "trailing-final-answer-line"
        assert config["gold_ordering"] == "sorted"


CHILD_FORWARD_PROMPT = (
    "You are given this edge slice:\n"
    "a -> b\n"
    "a -> c\n"
    "Compute the children of the frontier [a] over just this edge list.\n"
    "End with a line 'Final Answer: [n1, n2, ...]'."
)

CHILD_REVERSE_PROMPT = (
    "You are given this edge slice:\n"
    "b -> d\n"
    "c -> d\n"
    "Find the parents of node d over just these edges.\n"
    "End with a line 'Final Answer: [n1, n2, ...]'."
)

CHILD_EXCLUDED_PROMPT = (
    "Edge slice:\n"
    "b -> d\n"
    "c -> d\n"
    "d -> e\n"
    "Compute the children of the frontier [b, c], excluding already-visited nodes [a, d].\n"
    "End with a line 'Final Answer: [...]'."
)


class TestParseSubproblem:
    def test_forward_frontier_list(self):
        edges, frontier, excluded, reverse = parse_subproblem(CHILD_FORWARD_PROMPT)
        assert ("a", "b") in edges and ("a", "c") in edges
        assert frontier == {"a"}
        assert excluded == set()
        assert reverse is False

    def test_reverse_target(self):
        _, frontier, excluded, reverse = parse_subproblem(CHILD_REVERSE_PROMPT)
        assert frontier == {"d"}
        assert excluded == set()
        assert reverse is True

    def test_excluded_list_parses(self):
        _, frontier, excluded, _ = parse_subproblem(CHILD_EXCLUDED_PROMPT)
        assert frontier == {"b", "c"}
        assert excluded == {"a", "d"}

    def test_exclusion_mentioned_without_list_is_unparseable(self):
        prompt = (
            "Edges:\na -> b\nCompute the children of the frontier [a], "
            "excluding any nodes already visited.\nFinish appropriately."
        )
        assert parse_subproblem(prompt) is None

    def test_no_edges_is_unparseable(self):
        assert parse_subproblem("Please summarize this document for me.") is None

    def test_edges_without_a_query_is_unparseable(self):
        assert parse_subproblem("Some edges:\na -> b\nb -> c\nHave a look.") is None


class TestGraphWalksSubVerifier:
    def test_correct_forward_child_passes(self):
        node = make_node(CHILD_FORWARD_PROMPT, "sure\nFinal Answer: [b, c]")
        assert GraphWalksSubVerifier()({}, node) is True

    def test_correct_reverse_child_passes(self):
        node = make_node(CHILD_REVERSE_PROMPT, "Final Answer: [c, b]")
        assert GraphWalksSubVerifier()({}, node) is True

    def test_excluded_nodes_are_removed_from_the_expected_hop(self):
        # Forward hop from {b, c} over the slice reaches only d, which is
        # excluded, so the correct answer is the empty set.
        node = make_node(CHILD_EXCLUDED_PROMPT, "Final Answer: []")
        assert GraphWalksSubVerifier()({}, node) is True

    def test_wrong_child_fails(self):
        node = make_node(CHILD_FORWARD_PROMPT, "Final Answer: [b]")
        assert GraphWalksSubVerifier()({}, node) is False

    def test_depth_two_node_is_graded_against_its_own_slice(self):
        # The grandchild's slice mentions edges absent from any fixture
        # instance; only its own prompt defines its sub-problem.
        prompt = "Slice:\nx -> y\nx -> z\nCompute the children of the frontier [x]."
        node = make_node(prompt, "Final Answer: [y, z]", depth=2)
        assert GraphWalksSubVerifier()({"answer_nodes": ["b", "c"]}, node) is True

    def test_unparseable_prompt_is_uncheckable(self):
        node = make_node("Summarize the middle slice of the report.", "Final Answer: [b]")
        assert GraphWalksSubVerifier()({}, node) is None

    def test_dict_prompt_is_uncheckable(self):
        node = make_node({"role": "user", "content": CHILD_FORWARD_PROMPT}, "Final Answer: [b, c]")
        assert GraphWalksSubVerifier()({}, node) is None

    def test_errored_node_is_uncheckable(self):
        node = make_node(
            CHILD_FORWARD_PROMPT,
            "Error: LM query failed - connection reset",
            kind=NodeKind.ERRORED,
            error_kind="lm_error",
        )
        assert GraphWalksSubVerifier()({}, node) is None

    def test_response_without_final_answer_line_is_uncheckable_not_wrong(self):
        # The child may even be right; a format mismatch must not masquerade
        # as a wrong child answer and flip the failing level to CHILD.
        node = make_node(CHILD_FORWARD_PROMPT, "The children of a are b and c.")
        assert GraphWalksSubVerifier()({}, node) is None

    def test_pathological_prompt_never_raises(self):
        binary_garbage = ("\x00\x01\xff�" * 50_000) + "]["
        node = make_node(binary_garbage, "Final Answer: [b]")
        assert GraphWalksSubVerifier()({}, node) is None
