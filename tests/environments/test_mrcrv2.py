"""Tests for the MRCRv2 environment: the generator's determinism/sizing, answer
extraction tolerance, scoring, the Verifier's cause mapping, and the
SubVerifier's grounding (including the uncheckable cases).

Everything runs offline -- the generator has no network seam at all (it is a
pure synthetic generator), so there is nothing to monkeypatch.
"""

from typing import Any

from shrlm.environments.mrcrv2 import (
    ANSWER_FORMAT_CONTRACT,
    Mrcrv2SubVerifier,
    Mrcrv2Verifier,
    extract_mrcrv2_answer,
    generate_mrcrv2_instance,
    generate_mrcrv2_instances,
    needles_in_slice,
    parse_local_finding,
    score_mrcrv2,
)
from shrlm.optimization.bundle import FILESYSTEM_SAFE_ID_PATTERN
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_instance(
    *,
    target_tokens: int = 500,
    n_needles: int = 2,
    seed: int = 0,
    index: int = 0,
) -> dict[str, Any]:
    return generate_mrcrv2_instance(target_tokens, n_needles, seed, index)


def make_node(
    prompt: str | dict[str, Any],
    response: str,
    kind: NodeKind = NodeKind.RLM_CHILD,
    error_kind: str | None = None,
) -> CallNode:
    return CallNode(
        node_id="r/i0/b0/c0",
        parent_id="r",
        kind=kind,
        depth=1,
        model="mock",
        prompt=prompt,
        response=response,
        prompt_chars=len(str(prompt)),
        response_chars=len(response),
        execution_time=0.1,
        error_kind=error_kind,
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerateMrcrv2Instance:
    def test_deterministic_for_same_arguments(self):
        a = make_instance(seed=5, index=3)
        b = make_instance(seed=5, index=3)
        assert a == b

    def test_different_index_differs(self):
        a = make_instance(seed=5, index=0)
        b = make_instance(seed=5, index=1)
        assert a["id"] != b["id"]

    def test_different_seed_differs(self):
        a = make_instance(seed=1, index=0)
        b = make_instance(seed=2, index=0)
        assert a["id"] != b["id"]

    def test_needle_count_matches_request(self):
        inst = make_instance(n_needles=5)
        assert len(inst["needles"]) == 5
        assert {n["instance_index"] for n in inst["needles"]} == {1, 2, 3, 4, 5}

    def test_target_instance_index_in_bounds(self):
        for index in range(10):
            inst = make_instance(n_needles=3, index=index)
            assert 1 <= inst["target_instance_index"] <= 3

    def test_gold_content_matches_target_needle(self):
        inst = make_instance(n_needles=3)
        target = inst["target_instance_index"]
        expected = next(
            n["content"] for n in inst["needles"] if n["instance_index"] == target
        )
        assert inst["gold_content"] == expected

    def test_approx_token_sizing_within_tolerance(self):
        target_tokens = 2000
        inst = generate_mrcrv2_instance(target_tokens, 2, seed=0, index=0)
        # chars_per_token defaults to 4.0; the build only ever overshoots (it
        # keeps adding turns until the budget is met), never undershoots.
        assert inst["prompt_chars"] >= target_tokens * 4.0
        assert inst["prompt_chars"] < target_tokens * 4.0 * 1.5

    def test_needle_offsets_are_exact(self):
        inst = make_instance(n_needles=3)
        for needle in inst["needles"]:
            span = inst["prompt"][needle["char_start"] : needle["char_end"]]
            assert span == f"user: {needle['query']}\nassistant: {needle['content']}"

    def test_needle_queries_all_match_instance_query(self):
        inst = make_instance(n_needles=4)
        assert all(n["query"] == inst["query"] for n in inst["needles"])

    def test_id_is_filesystem_safe(self):
        inst = make_instance()
        assert FILESYSTEM_SAFE_ID_PATTERN.fullmatch(inst["id"])

    def test_prompt_carries_both_contracts(self):
        inst = make_instance()
        assert "final line" in inst["prompt"].lower()
        assert "LOCAL FINDING" in inst["prompt"]

    def test_rejects_zero_needles(self):
        import pytest

        with pytest.raises(ValueError):
            generate_mrcrv2_instance(500, 0, seed=0, index=0)

    def test_rejects_zero_target_tokens(self):
        import pytest

        with pytest.raises(ValueError):
            generate_mrcrv2_instance(0, 2, seed=0, index=0)


class TestGenerateMrcrv2Instances:
    def test_unique_ids(self):
        instances = generate_mrcrv2_instances(500, 2, limit=20, seed=0)
        ids = [inst["id"] for inst in instances]
        assert len(set(ids)) == len(ids) == 20

    def test_limit_respected(self):
        instances = generate_mrcrv2_instances(500, 2, limit=7, seed=0)
        assert len(instances) == 7


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------


class TestExtractMrcrv2Answer:
    def test_marker_line_wins_over_trailing_prose(self):
        parsed = extract_mrcrv2_answer("some reasoning\nFINAL: amber wren delta")
        assert parsed == ("amber wren delta", False)

    def test_last_nonempty_line_when_no_marker(self):
        parsed = extract_mrcrv2_answer("reasoning here\namber wren delta")
        assert parsed == ("amber wren delta", False)

    def test_none_marker_is_empty(self):
        assert extract_mrcrv2_answer("FINAL: none") == ("", True)
        assert extract_mrcrv2_answer("FINAL: not found") == ("", True)
        assert extract_mrcrv2_answer("FINAL: no match") == ("", True)

    def test_all_whitespace_is_none(self):
        assert extract_mrcrv2_answer("   \n  \n") is None

    def test_whitespace_normalized(self):
        parsed = extract_mrcrv2_answer("FINAL:   amber   wren\tdelta  ")
        assert parsed == ("amber wren delta", False)


class TestScoreMrcrv2:
    def test_identical_is_one(self):
        assert score_mrcrv2("amber wren delta", "amber wren delta") == 1.0

    def test_disjoint_is_low(self):
        assert score_mrcrv2("amber wren delta", "zephyr yarrow talon") < 0.5

    def test_near_miss_is_partial(self):
        ratio = score_mrcrv2("amber wren delta fern", "amber wren delta")
        assert 0.5 < ratio < 1.0


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TestMrcrv2Verifier:
    def test_exact_match_passes(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, f"FINAL: {inst['gold_content']}")
        assert verdict.passed
        assert verdict.cause is None

    def test_last_line_fallback_also_passes(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, inst["gold_content"])
        assert verdict.passed

    def test_no_candidate_line_is_wrong_format(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, "   \n  ")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.WRONG_FORMAT

    def test_explicit_none_against_nonempty_gold_is_no_answer(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, "FINAL: none")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_wrong_content_is_wrong_value(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, "FINAL: totally unrelated garbage text")
        assert not verdict.passed
        assert verdict.cause is VerifierCause.WRONG_VALUE

    def test_passing_verdict_carries_no_cause(self):
        inst = make_instance()
        verdict = Mrcrv2Verifier()(inst, f"FINAL: {inst['gold_content']}")
        assert verdict.cause is None  # Verdict.__post_init__ enforces this too

    def test_config_exposes_answer_contract(self):
        config = Mrcrv2Verifier().config()
        assert config["environment"] == "mrcrv2"
        assert config["pass_score_threshold"] == 1.0
        assert "extraction_rule" in config


# ---------------------------------------------------------------------------
# SubVerifier
# ---------------------------------------------------------------------------


class TestParseLocalFinding:
    def test_found_parses_query_and_count(self):
        parsed = parse_local_finding(
            'LOCAL FINDING: found instance 2 of query "What is your favorite tea?" in this slice'
        )
        assert parsed == ("What is your favorite tea?", 2)

    def test_no_match_parses_query_with_none_count(self):
        parsed = parse_local_finding(
            'LOCAL FINDING: no match for query "What is your favorite tea?" in this slice'
        )
        assert parsed == ("What is your favorite tea?", None)

    def test_no_finding_line_is_none(self):
        assert parse_local_finding("I just dumped some text back") is None


class TestNeedlesInSlice:
    def test_finds_needle_present_by_literal_content(self):
        inst = make_instance(n_needles=3)
        needle = inst["needles"][0]
        prompt = f"Here is your slice:\n{needle['content']}\nGo."
        found = needles_in_slice(inst["needles"], prompt)
        assert [n["instance_index"] for n in found] == [needle["instance_index"]]

    def test_empty_slice_finds_nothing(self):
        inst = make_instance(n_needles=3)
        assert needles_in_slice(inst["needles"], "nothing relevant here") == []


class TestMrcrv2SubVerifier:
    def test_correct_found_claim(self):
        inst = make_instance(n_needles=2)
        needle = inst["needles"][0]
        prompt = f"slice:\n{needle['content']}"
        response = (
            f'LOCAL FINDING: found instance 1 of query "{needle["query"]}" in this slice'
        )
        assert Mrcrv2SubVerifier()(inst, make_node(prompt, response)) is True

    def test_wrong_count_is_false(self):
        inst = make_instance(n_needles=2)
        needle = inst["needles"][0]
        prompt = f"slice:\n{needle['content']}"
        response = (
            f'LOCAL FINDING: found instance 9 of query "{needle["query"]}" in this slice'
        )
        assert Mrcrv2SubVerifier()(inst, make_node(prompt, response)) is False

    def test_false_no_match_claim_when_needle_present_is_false(self):
        inst = make_instance(n_needles=2)
        needle = inst["needles"][0]
        prompt = f"slice:\n{needle['content']}"
        response = f'LOCAL FINDING: no match for query "{needle["query"]}" in this slice'
        assert Mrcrv2SubVerifier()(inst, make_node(prompt, response)) is False

    def test_correct_no_match_claim_when_needle_absent(self):
        inst = make_instance(n_needles=2)
        needle = inst["needles"][0]
        response = f'LOCAL FINDING: no match for query "{needle["query"]}" in this slice'
        assert Mrcrv2SubVerifier()(inst, make_node("nothing relevant", response)) is True

    def test_no_finding_line_is_uncheckable(self):
        inst = make_instance(n_needles=2)
        node = make_node("slice text", "just some prose, no marker")
        assert Mrcrv2SubVerifier()(inst, node) is None

    def test_unknown_query_is_uncheckable(self):
        inst = make_instance(n_needles=2)
        response = 'LOCAL FINDING: no match for query "a totally unrelated query" in this slice'
        node = make_node("slice text", response)
        assert Mrcrv2SubVerifier()(inst, node) is None

    def test_errored_node_is_uncheckable(self):
        inst = make_instance(n_needles=2)
        node = make_node("slice", "LOCAL FINDING: no match", error_kind="timeout_exhausted")
        assert Mrcrv2SubVerifier()(inst, node) is None

    def test_dict_prompt_is_uncheckable(self):
        inst = make_instance(n_needles=2)
        node = make_node({"role": "user", "content": "x"}, "LOCAL FINDING: no match")
        assert Mrcrv2SubVerifier()(inst, node) is None

    def test_pathological_prompt_never_raises(self):
        inst = make_instance(n_needles=2)
        pathological = [
            "",
            "\x00\x01\x02",
            "LOCAL FINDING: found instance abc of query \"x\" in this slice",
            "LOCAL FINDING: " * 1000,
            '"' * 500,
        ]
        for text in pathological:
            result = Mrcrv2SubVerifier()(inst, make_node(text, text))
            assert result in (True, False, None)


def test_answer_format_contract_is_nonempty_string():
    assert isinstance(ANSWER_FORMAT_CONTRACT, str) and ANSWER_FORMAT_CONTRACT.strip()
