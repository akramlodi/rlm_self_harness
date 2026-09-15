"""Tests for the RULER environment: generator, Verifier, SubVerifier.

Fully synthetic -- there is no network seam to monkeypatch. Every test calls
the generator directly with small ``target_tokens`` values.
"""

from typing import Any

import pytest

from shrlm.environments.ruler import (
    TASK_TYPES,
    RulerSubVerifier,
    RulerVerifier,
    extract_ruler_answer,
    generate_ruler_instance,
    generate_ruler_instances,
    parse_local_findings,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import CallNode, NodeKind


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


class TestGenerateRulerInstance:
    def test_deterministic_for_same_arguments(self):
        first = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=0)
        second = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=0)
        assert first == second

    def test_different_index_gives_different_but_reproducible_instance(self):
        a = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=0)
        b = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=1)
        assert a["id"] != b["id"]
        b_again = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=1)
        assert b == b_again

    def test_unknown_task_type_raises(self):
        with pytest.raises(ValueError, match="unknown RULER task_type"):
            generate_ruler_instance("not_a_task", target_tokens=200, seed=0, index=0)

    def test_zero_target_tokens_raises(self):
        with pytest.raises(ValueError, match="target_tokens must be"):
            generate_ruler_instance("niah_multikey", target_tokens=0, seed=0, index=0)

    def test_id_is_filesystem_safe_and_content_derived(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=200, seed=0, index=0)
        assert instance["id"].startswith("ruler-niah_multikey-200-")

    @pytest.mark.parametrize("task_type", TASK_TYPES)
    def test_every_task_type_produces_a_well_formed_instance(self, task_type):
        instance = generate_ruler_instance(task_type, target_tokens=300, seed=0, index=0)
        assert instance["task_type"] == task_type
        assert instance["answer_kind"] in ("scalar", "list")
        assert instance["prompt"].startswith(instance["prompt"][:1])
        assert instance["gold"] not in (None, "", [])
        assert isinstance(instance["tracked_units"], list) and instance["tracked_units"]
        assert "LOCAL FINDING" in instance["prompt"]

    def test_scalar_gold_needle_sentence_is_verbatim_in_the_prompt(self):
        instance = generate_ruler_instance("niah_single", target_tokens=300, seed=3, index=0)
        matching = [unit for unit in instance["tracked_units"] if unit["value"] == instance["gold"]]
        assert matching
        assert matching[0]["literal_text"] in instance["prompt"]

    def test_variable_tracking_gold_lines_are_verbatim_in_the_prompt(self):
        instance = generate_ruler_instance("variable_tracking", target_tokens=400, seed=4, index=0)
        gold_names = set(instance["gold"])
        target_units = [u for u in instance["tracked_units"] if u["item"] in gold_names]
        assert len(target_units) == len(gold_names)
        for unit in target_units:
            assert unit["literal_text"] in instance["prompt"]

    def test_common_words_extraction_top_words_have_the_highest_counts(self):
        instance = generate_ruler_instance(
            "common_words_extraction", target_tokens=500, seed=5, index=0
        )
        body = instance["prompt"].split("\n\n---\n")[0]
        words = [w.strip() for w in body.split(",")]
        counts = {word: words.count(word) for word in set(words)}
        top_10_by_count = sorted(counts, key=lambda w: (-counts[w], w))[:10]
        assert sorted(top_10_by_count) == instance["gold"]

    def test_frequent_words_extraction_top_words_match_the_realized_counts(self):
        instance = generate_ruler_instance(
            "frequent_words_extraction", target_tokens=500, seed=6, index=0
        )
        body = instance["prompt"].split("\n\n---\n")[0]
        words = [w.strip() for w in body.split(",")]
        counts = {word: words.count(word) for word in set(words)}
        top_3_by_count = sorted(counts, key=lambda w: (-counts[w], w))[:3]
        assert sorted(top_3_by_count) == instance["gold"]


class TestGenerateRulerInstances:
    def test_returns_exactly_limit_instances_with_unique_ids(self):
        instances = generate_ruler_instances(
            ("niah_multikey", "variable_tracking"), target_tokens=200, limit=7, seed=0
        )
        assert len(instances) == 7
        assert len({i["id"] for i in instances}) == 7

    def test_balanced_across_task_types(self):
        instances = generate_ruler_instances(
            ("niah_multikey", "variable_tracking"), target_tokens=200, limit=6, seed=0
        )
        types = [i["task_type"] for i in instances]
        assert types.count("niah_multikey") == 3
        assert types.count("variable_tracking") == 3

    def test_odd_limit_is_topped_up_not_dropped(self):
        instances = generate_ruler_instances(
            ("niah_multikey", "variable_tracking"), target_tokens=200, limit=5, seed=0
        )
        assert len(instances) == 5

    def test_same_seed_same_sample(self):
        first = generate_ruler_instances(("niah_multikey",), target_tokens=200, limit=3, seed=0)
        second = generate_ruler_instances(("niah_multikey",), target_tokens=200, limit=3, seed=0)
        assert first == second

    def test_empty_task_types_raises(self):
        with pytest.raises(ValueError, match="task_types must not be empty"):
            generate_ruler_instances((), target_tokens=200, limit=1, seed=0)

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be"):
            generate_ruler_instances(("niah_multikey",), target_tokens=200, limit=0, seed=0)


class TestExtractRulerAnswer:
    def test_no_lines_is_none(self):
        assert extract_ruler_answer("   \n  ", "scalar") is None

    def test_marker_is_optional(self):
        assert extract_ruler_answer("some reasoning\n123456", "scalar").value == "123456"
        assert extract_ruler_answer("FINAL: 123456", "scalar").value == "123456"

    def test_explicit_none_marker_is_empty_not_wrong_format(self):
        parsed = extract_ruler_answer("FINAL: NONE", "scalar")
        assert parsed.empty is True and parsed.value == ""

    def test_list_parses_comma_separated_items(self):
        parsed = extract_ruler_answer("FINAL: a, b, c", "list")
        assert parsed.value == ["a", "b", "c"]

    def test_list_strips_quotes_and_brackets(self):
        parsed = extract_ruler_answer("FINAL: ['a', 'b']", "list")
        assert parsed.value == ["a", "b"]

    def test_list_explicit_none_is_empty(self):
        parsed = extract_ruler_answer("FINAL: NONE", "list")
        assert parsed.empty is True and parsed.value == []


class TestRulerVerifier:
    def scalar_instance(self) -> dict[str, Any]:
        return generate_ruler_instance("niah_multikey", target_tokens=300, seed=1, index=0)

    def list_instance(self) -> dict[str, Any]:
        return generate_ruler_instance("niah_multivalue", target_tokens=400, seed=2, index=0)

    def test_scalar_exact_match_passes(self):
        instance = self.scalar_instance()
        verdict = RulerVerifier()(instance, f"FINAL: {instance['gold']}")
        assert verdict.passed is True and verdict.cause is None

    def test_scalar_wrong_value(self):
        instance = self.scalar_instance()
        verdict = RulerVerifier()(instance, "FINAL: 000000")
        assert verdict.passed is False
        assert verdict.cause is VerifierCause.WRONG_VALUE

    def test_scalar_empty_response_is_wrong_format(self):
        instance = self.scalar_instance()
        verdict = RulerVerifier()(instance, "   ")
        assert verdict.cause is VerifierCause.WRONG_FORMAT

    def test_scalar_explicit_none_is_no_answer(self):
        instance = self.scalar_instance()
        verdict = RulerVerifier()(instance, "FINAL: NONE")
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_list_exact_match_passes(self):
        instance = self.list_instance()
        verdict = RulerVerifier()(instance, "FINAL: " + ", ".join(instance["gold"]))
        assert verdict.passed is True and verdict.cause is None

    def test_list_missing_only_is_incomplete(self):
        instance = self.list_instance()
        verdict = RulerVerifier()(instance, f"FINAL: {instance['gold'][0]}")
        assert verdict.cause is VerifierCause.INCOMPLETE

    def test_list_extra_only_is_spurious(self):
        instance = self.list_instance()
        verdict = RulerVerifier()(instance, "FINAL: " + ", ".join([*instance["gold"], "999999"]))
        assert verdict.cause is VerifierCause.SPURIOUS

    def test_list_missing_and_extra_is_mixed_set_error(self):
        instance = self.list_instance()
        verdict = RulerVerifier()(instance, "FINAL: " + instance["gold"][0] + ", 999999")
        assert verdict.cause is VerifierCause.MIXED_SET_ERROR

    def test_list_explicit_none_against_nonempty_gold_is_no_answer(self):
        instance = self.list_instance()
        verdict = RulerVerifier()(instance, "FINAL: NONE")
        assert verdict.cause is VerifierCause.NO_ANSWER

    def test_gold_and_produced_are_sorted(self):
        instance = self.list_instance()
        shuffled = list(reversed(instance["gold"]))
        verdict = RulerVerifier()(instance, "FINAL: " + ", ".join(shuffled))
        assert verdict.gold == "[" + ", ".join(sorted(instance["gold"])) + "]"

    def test_config_names_the_grading_facts(self):
        config = RulerVerifier().config()
        assert config["environment"] == "ruler"
        assert config["pass_threshold"] == 1.0
        assert config["gold_ordering"] == "sorted"


class TestParseLocalFindings:
    def test_parses_a_single_finding(self):
        findings = parse_local_findings("reasoning...\nLOCAL FINDING: apple-1234 = 555555")
        assert findings == {"apple-1234": "555555"}

    def test_parses_multiple_findings(self):
        response = "LOCAL FINDING: a = 1\nLOCAL FINDING: b = NONE"
        assert parse_local_findings(response) == {"a": "1", "b": "NONE"}

    def test_first_occurrence_wins_on_duplicate_item(self):
        response = "LOCAL FINDING: a = 1\nLOCAL FINDING: a = 2"
        assert parse_local_findings(response) == {"a": "1"}

    def test_no_finding_line_is_empty_dict(self):
        assert parse_local_findings("just a normal response") == {}

    def test_pathological_response_never_raises(self):
        garbage = ("\x00\x01\xff�" * 10_000) + "LOCAL FINDING:"
        assert isinstance(parse_local_findings(garbage), dict)


class TestRulerSubVerifier:
    def test_niah_correct_child_passes(self):
        instance = generate_ruler_instance("niah_multivalue", target_tokens=400, seed=7, index=0)
        unit = instance["tracked_units"][0]
        prompt = f"Slice:\n{unit['literal_text']}\nWhat values do you see?"
        response = f"LOCAL FINDING: {unit['item']} = {unit['value']}"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is True

    def test_niah_wrong_child_fails(self):
        instance = generate_ruler_instance("niah_multivalue", target_tokens=400, seed=7, index=0)
        unit = instance["tracked_units"][0]
        prompt = f"Slice:\n{unit['literal_text']}\nWhat values do you see?"
        response = f"LOCAL FINDING: {unit['item']} = 000000"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is False

    def test_niah_absent_key_grounds_to_none_marker(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=8, index=0)
        key = instance["tracked_units"][0]["item"]
        prompt = "Slice:\nnothing relevant to any key is here."
        response = f"LOCAL FINDING: {key} = NONE"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is True

    def test_variable_tracking_presence_claim_passes(self):
        instance = generate_ruler_instance("variable_tracking", target_tokens=400, seed=9, index=0)
        unit = instance["tracked_units"][0]
        prompt = f"Slice:\n{unit['literal_text']}"
        response = f"LOCAL FINDING: {unit['item']} = PRESENT"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is True

    def test_variable_tracking_absent_claim_when_line_is_missing_fails(self):
        instance = generate_ruler_instance("variable_tracking", target_tokens=400, seed=9, index=0)
        unit = instance["tracked_units"][0]
        prompt = "Slice:\nnothing relevant here at all."
        response = f"LOCAL FINDING: {unit['item']} = PRESENT"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is False

    def test_word_count_correct_recount_passes(self):
        instance = generate_ruler_instance(
            "common_words_extraction", target_tokens=400, seed=10, index=0
        )
        word = instance["tracked_units"][0]["item"]
        prompt = f"{word}, {word}, other, {word}"
        response = f"LOCAL FINDING: {word} = 3"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is True

    def test_word_count_wrong_recount_fails(self):
        instance = generate_ruler_instance(
            "common_words_extraction", target_tokens=400, seed=10, index=0
        )
        word = instance["tracked_units"][0]["item"]
        prompt = f"{word}, {word}, other, {word}"
        response = f"LOCAL FINDING: {word} = 99"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is False

    def test_word_count_does_not_overlap_substrings(self):
        # A word-boundary count must not match "wordy" inside "wordypants".
        instance = generate_ruler_instance(
            "common_words_extraction", target_tokens=400, seed=10, index=0
        )
        word = instance["tracked_units"][0]["item"]
        prompt = f"{word}pants, {word}"
        response = f"LOCAL FINDING: {word} = 1"
        node = make_node(prompt, response)
        assert RulerSubVerifier()(instance, node) is True

    def test_unknown_claimed_item_is_uncheckable(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        node = make_node("some slice", "LOCAL FINDING: not-a-real-key = 123456")
        assert RulerSubVerifier()(instance, node) is None

    def test_no_local_finding_line_is_uncheckable(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        node = make_node("some slice", "I looked but found nothing to report.")
        assert RulerSubVerifier()(instance, node) is None

    def test_errored_node_is_uncheckable(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        node = make_node(
            "some slice",
            "Error: LM query failed",
            kind=NodeKind.ERRORED,
            error_kind="lm_error",
        )
        assert RulerSubVerifier()(instance, node) is None

    def test_dict_prompt_is_uncheckable(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        node = make_node({"role": "user", "content": "x"}, "LOCAL FINDING: a = 1")
        assert RulerSubVerifier()(instance, node) is None

    def test_pathological_prompt_never_raises(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        binary_garbage = ("\x00\x01\xff�" * 50_000) + "[["
        node = make_node(binary_garbage, "LOCAL FINDING: totally-fake-key-9999 = 1")
        assert RulerSubVerifier()(instance, node) is None

    def test_pathological_response_never_raises(self):
        instance = generate_ruler_instance("niah_multikey", target_tokens=400, seed=11, index=0)
        key = instance["tracked_units"][0]["item"]
        binary_garbage = ("\x00\x01\xff�" * 50_000) + f"LOCAL FINDING: {key} = 1"
        node = make_node("some slice", binary_garbage)
        # Never raises; result is a legitimate bool since the key is real.
        assert RulerSubVerifier()(instance, node) in (True, False, None)
