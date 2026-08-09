"""Tests for the stage-2 proposer (``shrlm-proposal/v1`` output).

The proposer is validated the way the interface doc recommends: build a
candidate, write it, then run it through the *real* stage-3 loader
(``shrlm.optimization.candidates.load_candidates``) and assert zero
rejections. That is the strongest guarantee available -- it proves the
proposer's output clears the actual gate, not a re-description of it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from shrlm.harness_identity import serialize_harness
from shrlm.optimization.candidates import LoadedCandidate, changed_surfaces, load_candidates
from shrlm.optimization.proposal import (
    CandidateSpec,
    MaterializationFailure,
    ProposalCache,
    ProposalRejection,
    ProposerConfig,
    build_candidate,
    extract_json_array,
    load_passing_behaviors,
    materialize_candidate_harness,
    propose_round,
    render_prompt,
    validate_candidate_spec,
    write_proposal,
)
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Synthetic patterns, one per edit kind the proposer supports, plus one
# unaddressable (OTHER) pattern. Same shape as FailurePattern.to_dict().
# ---------------------------------------------------------------------------


def make_pattern(mechanism: str, verifier_cause: str = "wrong_value") -> dict[str, Any]:
    return {
        "signature": {
            "verifier_cause": verifier_cause,
            "failing_level": "root",
            "causal_status": "causal",
            "agent_mechanism": mechanism,
        },
        "support": 3,
        "instance_support": 3,
        "instance_ids": ["a", "b", "c"],
        "representatives": ["a"],
        "shared_symptoms": ["median sub-calls per run: 2"],
        "verifier_evidence": ["a: expected 'x', produced 'y'"],
        "grounded_fraction": 1.0,
        "surface": None,
        "actionability": 0.9,
        "below_support_floor": False,
    }


PATTERN_TEXT = make_pattern("skipped_verification")  # -> S4
PATTERN_POLICY = make_pattern("iteration_budget_exhaustion")  # -> S6
PATTERN_CODE_S7 = make_pattern("unparsed_child_output")  # -> S7
PATTERN_CODE_S9 = make_pattern("lossy_aggregation")  # -> S9
PATTERN_REPL_HELPER = make_pattern("repl_execution_fault")  # -> S8
PATTERN_OTHER = make_pattern("other")  # unaddressable

ALL_PATTERNS = [
    PATTERN_TEXT,
    PATTERN_POLICY,
    PATTERN_CODE_S7,
    PATTERN_CODE_S9,
    PATTERN_REPL_HELPER,
    PATTERN_OTHER,
]

BUNDLE = {"bundle_id": "test-bundle-0001", "patterns": ALL_PATTERNS}

# Known-good source, reusing H0's own known-good S7 builder body so the
# generated-code path is exercised without inventing new invariant risk.
S7_SOURCE = (
    "def build_metadata(stdout, repl_inventory, max_character_length=20000):\n"
    "    return default_metadata_builder(stdout, repl_inventory, max_character_length)\n"
)
S9_SOURCE = (
    "def redirect_empty(answer, repl_inventory):\n"
    "    if not answer.strip():\n"
    "        return AnswerDecision.redirect('aggregate more evidence first')\n"
    "    return AnswerDecision.accept(answer)\n"
)
S8_SOURCE = (
    "def safe_index(seq, i):\n"
    "    if 0 <= i < len(seq):\n"
    "        return seq[i]\n"
    "    return None\n"
)


def edit_item(pattern_index: int, edit: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    item = {
        "pattern_index": pattern_index,
        "edit": edit,
        "predicted_effect": "the root double-checks before answering",
        "regression_risks": ["one extra turn per run"],
    }
    item.update(overrides)
    return item


TEXT_ITEM = edit_item(0, {"kind": "text", "new_text": "Restate the answer before submitting it."})
POLICY_ITEM = edit_item(1, {"kind": "policy", "runtime_policy": {"enabled": True, "max_depth": 3}})
CODE_S7_ITEM = edit_item(2, {"kind": "code", "source": S7_SOURCE})
CODE_S9_ITEM = edit_item(3, {"kind": "code", "source": S9_SOURCE})
REPL_HELPER_ITEM = edit_item(
    4, {"kind": "repl_helper", "dict": "repl_helpers", "name": "safe_index", "source": S8_SOURCE}
)


def canned_batch(*items: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(list(items)) + "\n```"


# ---------------------------------------------------------------------------
# extract_json_array
# ---------------------------------------------------------------------------


def test_extract_json_array_fenced():
    assert extract_json_array(canned_batch(TEXT_ITEM)) == [TEXT_ITEM]


def test_extract_json_array_unfenced_falls_back_to_bracket_span():
    text = "here is my answer: " + json.dumps([TEXT_ITEM]) + " done"
    assert extract_json_array(text) == [TEXT_ITEM]


def test_extract_json_array_rejects_non_array():
    with pytest.raises(ProposalRejection, match="array"):
        extract_json_array("```json\n{}\n```")


def test_extract_json_array_rejects_bad_json():
    with pytest.raises(ProposalRejection, match="not valid JSON"):
        extract_json_array("```json\n[1, 2,\n```")


# ---------------------------------------------------------------------------
# validate_candidate_spec: one accepted spec per edit kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item,expected_surface",
    [
        (TEXT_ITEM, "S4"),
        (POLICY_ITEM, "S6"),
        (CODE_S7_ITEM, "S7"),
        (CODE_S9_ITEM, "S9"),
        (REPL_HELPER_ITEM, "S8"),
    ],
)
def test_validate_candidate_spec_accepts_each_kind(item, expected_surface):
    spec = validate_candidate_spec(item, ALL_PATTERNS)
    assert spec.surface == expected_surface
    assert spec.pattern_index == item["pattern_index"]


def test_validate_candidate_spec_rejects_out_of_range_index():
    with pytest.raises(ProposalRejection, match="pattern_index"):
        validate_candidate_spec(edit_item(99, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_other_mechanism():
    with pytest.raises(ProposalRejection, match="no editable surface"):
        validate_candidate_spec(edit_item(5, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_wrong_edit_kind():
    with pytest.raises(ProposalRejection, match="edit.kind"):
        validate_candidate_spec(
            edit_item(0, {"kind": "policy", "runtime_policy": {}}), ALL_PATTERNS
        )


def test_validate_candidate_spec_rejects_empty_text():
    with pytest.raises(ProposalRejection, match="new_text"):
        validate_candidate_spec(edit_item(0, {"kind": "text", "new_text": "  "}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_unknown_s6_key():
    with pytest.raises(ProposalRejection, match="unknown keys"):
        validate_candidate_spec(
            edit_item(1, {"kind": "policy", "runtime_policy": {"other_backends": {}}}), ALL_PATTERNS
        )


def test_validate_candidate_spec_rejects_two_defs():
    bad = "def a():\n    pass\ndef b():\n    pass\n"
    with pytest.raises(ProposalRejection, match="exactly one"):
        validate_candidate_spec(edit_item(2, {"kind": "code", "source": bad}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_decorated_def():
    bad = "@staticmethod\ndef build_metadata(stdout, repl_inventory, max_character_length=1):\n    return stdout\n"
    with pytest.raises(ProposalRejection, match="exactly one"):
        validate_candidate_spec(edit_item(2, {"kind": "code", "source": bad}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_syntax_error():
    with pytest.raises(ProposalRejection, match="does not parse"):
        validate_candidate_spec(edit_item(2, {"kind": "code", "source": "def build_metadata(:\n"}), ALL_PATTERNS)


def test_validate_candidate_spec_rejects_repl_helper_name_mismatch():
    item = edit_item(
        4, {"kind": "repl_helper", "dict": "repl_helpers", "name": "wrong_name", "source": S8_SOURCE}
    )
    with pytest.raises(ProposalRejection, match="must match the function name"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_validate_candidate_spec_rejects_reserved_repl_helper_name():
    source = "def llm_query(prompt):\n    return prompt\n"
    item = edit_item(4, {"kind": "repl_helper", "dict": "repl_helpers", "name": "llm_query", "source": source})
    with pytest.raises(ProposalRejection, match="reserved"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_validate_candidate_spec_rejects_empty_predicted_effect():
    with pytest.raises(ProposalRejection, match="predicted_effect"):
        validate_candidate_spec(
            edit_item(0, {"kind": "text", "new_text": "x"}, predicted_effect=""), ALL_PATTERNS
        )


# ---------------------------------------------------------------------------
# materialize_candidate_harness: one surface changes, per kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item,expected_surface",
    [
        (TEXT_ITEM, "S4"),
        (POLICY_ITEM, "S6"),
        (CODE_S7_ITEM, "S7"),
        (CODE_S9_ITEM, "S9"),
        (REPL_HELPER_ITEM, "S8"),
    ],
)
def test_materialize_candidate_harness_changes_one_surface(item, expected_surface, tmp_path):
    spec = validate_candidate_spec(item, ALL_PATTERNS)
    harness = materialize_candidate_harness(H0, spec, tmp_path)
    changed = changed_surfaces(serialize_harness(H0), serialize_harness(harness))
    assert changed == [expected_surface]


def test_build_candidate_detects_no_op_edit_as_materialization_failure(tmp_path):
    # A policy edit that never sets enabled=True changes nothing (build_runtime_policy()
    # is the all-None/disabled base already), so the merged dict is byte-identical to H0's.
    item = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    spec = validate_candidate_spec(item, ALL_PATTERNS)
    incumbent_serialization = serialize_harness(H0)
    with pytest.raises(MaterializationFailure):
        build_candidate(H0, incumbent_serialization, spec, tmp_path)


# ---------------------------------------------------------------------------
# End-to-end: written proposals clear the real stage-3 loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item,expected_surface",
    [
        (TEXT_ITEM, "S4"),
        (POLICY_ITEM, "S6"),
        (CODE_S7_ITEM, "S7"),
        (CODE_S9_ITEM, "S9"),
        (REPL_HELPER_ITEM, "S8"),
    ],
)
def test_written_proposal_loads_cleanly(item, expected_surface, tmp_path):
    spec = validate_candidate_spec(item, ALL_PATTERNS)
    incumbent_serialization = serialize_harness(H0)
    harness, serialization = build_candidate(H0, incumbent_serialization, spec, tmp_path / "work")
    candidate_id = f"cand-{expected_surface.lower()}"
    write_proposal(
        tmp_path / "proposals",
        candidate_id,
        incumbent_serialization,
        spec,
        serialization,
        "mock-model",
        "a" * 64,
    )
    loaded, rejections = load_candidates(tmp_path / "proposals", H0)
    assert rejections == [], rejections
    assert len(loaded) == 1
    assert isinstance(loaded[0], LoadedCandidate)
    assert loaded[0].surface == expected_surface


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_includes_surfaces_patterns_and_fallbacks():
    incumbent_serialization = serialize_harness(H0)
    rendered, addressable = render_prompt(ALL_PATTERNS, incumbent_serialization, (), (), k=4)
    assert "S1" in rendered and "S9" in rendered  # render_surface_block()
    assert "skipped_verification" in rendered
    assert "no passing runs" in rendered.lower()
    assert "no prior validation rounds" in rendered.lower()
    # OTHER mechanism must never be offered as an addressable index.
    assert [index for index, _ in addressable] == [0, 1, 2, 3, 4]


def test_render_prompt_passing_and_history_blocks():
    incumbent_serialization = serialize_harness(H0)
    passing = [{"instance_id": "bfs-1", "passed": True}]
    history = [([{"subject_id": "r00-c01-s4", "decision": "rejected", "reasons": ["cost too high"]}],
                {"promoted": False, "promoted_harness_hash": None})]
    rendered, _ = render_prompt(ALL_PATTERNS, incumbent_serialization, passing, history, k=4)
    assert "bfs-1" in rendered
    assert "r00-c01-s4" in rendered
    assert "cost too high" in rendered


# ---------------------------------------------------------------------------
# load_passing_behaviors
# ---------------------------------------------------------------------------


def test_load_passing_behaviors_filters_passed(tmp_path):
    runs = [
        {"instance_id": "a", "passed": True},
        {"instance_id": "b", "passed": False},
    ]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    result = load_passing_behaviors(tmp_path)
    assert result == [{"instance_id": "a", "passed": True}]


def test_load_passing_behaviors_missing_file_returns_empty(tmp_path):
    assert load_passing_behaviors(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# propose_round: the full loop against MockLM
# ---------------------------------------------------------------------------


def test_propose_round_writes_every_addressable_kind(tmp_path):
    response = canned_batch(TEXT_ITEM, POLICY_ITEM, CODE_S7_ITEM, CODE_S9_ITEM)
    lm = MockLM(model_name="mock-proposer", responses=[response])
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        round_index=0,
        config=ProposerConfig(k=4),
        workdir=tmp_path / "work",
    )
    assert result.materialization_failures == []
    assert {w.surface for w in result.written} == {"S4", "S6", "S7", "S9"}
    assert result.skipped_patterns == [4]  # the S8 pattern was never proposed for

    loaded, rejections = load_candidates(tmp_path / "proposals", H0)
    assert rejections == [], rejections
    assert len(loaded) == 4


def test_propose_round_reask_loop_records_both_attempts(tmp_path):
    bad_response = "not json at all"
    good_response = canned_batch(TEXT_ITEM)
    lm = MockLM(model_name="mock-proposer", responses=[bad_response, good_response])
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=3),
        workdir=tmp_path / "work",
    )
    assert len(result.attempts) == 2
    assert result.attempts[0].accepted is False
    assert result.attempts[0].violation
    assert result.attempts[1].accepted is True
    assert len(result.written) == 1


def test_propose_round_cache_replays_with_zero_additional_calls(tmp_path):
    response = canned_batch(TEXT_ITEM)
    lm = MockLM(model_name="mock-proposer", responses=[response])
    cache = ProposalCache()
    propose_round(
        BUNDLE, H0, lm, tmp_path / "proposals", cache=cache, workdir=tmp_path / "work"
    )
    assert lm._call_count == 1
    # Second round with an empty responses list: a cache miss would raise IndexError.
    lm2 = MockLM(model_name="mock-proposer", responses=[])
    propose_round(
        BUNDLE, H0, lm2, tmp_path / "proposals2", cache=cache, workdir=tmp_path / "work2"
    )
    assert lm2._call_count == 0


def test_propose_round_materialization_failure_does_not_drop_the_rest(tmp_path):
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    response = canned_batch(TEXT_ITEM, no_op_policy)
    lm = MockLM(model_name="mock-proposer", responses=[response])
    result = propose_round(
        BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work"
    )
    assert len(result.written) == 1
    assert result.written[0].surface == "S4"
    assert len(result.materialization_failures) == 1
    assert result.materialization_failures[0].pattern_index == 1


def test_propose_round_empty_bundle_no_crash(tmp_path):
    empty_bundle = {"bundle_id": "empty", "patterns": []}
    lm = MockLM(model_name="mock-proposer", responses=["```json\n[]\n```"])
    result = propose_round(empty_bundle, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert result.written == []
    assert result.skipped_patterns == []


def test_propose_round_raises_after_exhausting_attempts(tmp_path):
    lm = MockLM(model_name="mock-proposer", responses=["garbage", "garbage", "garbage"])
    with pytest.raises(ProposalRejection):
        propose_round(
            BUNDLE, H0, lm, tmp_path / "proposals",
            config=ProposerConfig(max_attempts=3), workdir=tmp_path / "work",
        )
