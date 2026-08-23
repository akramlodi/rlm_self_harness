"""Tests for the stage-2 proposer (``shrlm-proposal/v1`` output).

The proposer is validated the way the interface doc recommends: build a
candidate, write it, then run it through the *real* stage-3 loader
(``shrlm.optimization.candidates.load_candidates``) and assert zero
rejections. That is the strongest guarantee available -- it proves the
proposer's output clears the actual gate, not a re-description of it.
"""

import json
from dataclasses import replace
from typing import Any

import pytest

from shrlm.harness_identity import serialize_harness
from shrlm.optimization.candidates import LoadedCandidate, changed_surfaces, load_candidates
from shrlm.optimization.proposal import (
    SKILL_BODY_MAX_CHARS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_MAX_ENTRIES,
    SKILL_NAME_MAX_CHARS,
    SKILL_TOTAL_MAX_CHARS,
    MaterializationFailure,
    ProposalCache,
    ProposalRejection,
    ProposerConfig,
    _candidate_id,
    build_candidate,
    extract_json_array,
    load_passing_behaviors,
    materialize_candidate_harness,
    propose_round,
    render_prompt,
    validate_candidate_spec,
    write_proposal,
)
from shrlm.rlm_harness import H0, SKILL_LOADER_NAME, SkillEntry
from shrlm.runner import build_skill_loader
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
PATTERN_SKILLS = make_pattern("unconsulted_procedure")  # -> S10
PATTERN_OTHER = make_pattern("other")  # unaddressable

ALL_PATTERNS = [
    PATTERN_TEXT,
    PATTERN_POLICY,
    PATTERN_CODE_S7,
    PATTERN_CODE_S9,
    PATTERN_REPL_HELPER,
    PATTERN_SKILLS,
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
# One well-formed S10 record: identifier name, one-line brace-free description,
# body of ordered steps (same fixture shape as tests/optimization/test_candidates.py).
SKILL_RECORD = {
    "name": "verify_aggregate",
    "description": "Consult before committing an aggregated answer.",
    "body": "1. Re-read each partial result.\n2. Recompute the aggregate.\n3. Compare.",
}


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


def skill(**fields: Any) -> dict[str, Any]:
    """SKILL_RECORD with some fields overridden."""
    return {**SKILL_RECORD, **fields}


def skills_item(*records: Any, **overrides: Any) -> dict[str, Any]:
    """An S10 edit over the S10 pattern (index 5) carrying ``records`` as its whole value."""
    return edit_item(5, {"kind": "skills", "skills": list(records)}, **overrides)


SKILLS_ITEM = skills_item(SKILL_RECORD)


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
        (SKILLS_ITEM, "S10"),
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
        validate_candidate_spec(edit_item(6, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)


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
        (SKILLS_ITEM, "S10"),
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
        (SKILLS_ITEM, "S10"),
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
    assert "S1" in rendered and "S9" in rendered and "S10" in rendered  # render_surface_block()
    assert "skipped_verification" in rendered
    assert "no passing runs" in rendered.lower()
    assert "no prior validation rounds" in rendered.lower()
    # OTHER mechanism must never be offered as an addressable index; the S10
    # pattern (unconsulted_procedure) must be.
    assert [index for index, _ in addressable] == [0, 1, 2, 3, 4, 5]


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
    assert result.skipped_patterns == [4, 5]  # the S8 and S10 patterns were never proposed for

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


# ---------------------------------------------------------------------------
# S10: the skills edit kind (R3, R5, R6, R7, R14, R15). Rejection cases first --
# the value of this branch is what it refuses.
# ---------------------------------------------------------------------------


def _ok_skill_names(count: int) -> list[str]:
    return [f"skill_{index}" for index in range(count)]


@pytest.mark.parametrize(
    "description",
    [
        "Use when the answer is {x}.",
        "Use when { is needed.",
        "Use when } is needed.",
        "Consult the tools in {custom_tools_section} first.",
        "Doubled {{braces}} are still braces.",
    ],
)
def test_s10_rejects_brace_in_description(description):
    with pytest.raises(ProposalRejection, match=r"S10.*brace"):
        validate_candidate_spec(skills_item(skill(description=description)), ALL_PATTERNS)


def test_s10_rejects_custom_tools_slot_in_description():
    item = skills_item(skill(description="See {custom_tools_section} before answering."))
    with pytest.raises(ProposalRejection, match="S10"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_s10_accepts_undoubled_braces_in_body_and_loader_returns_it_verbatim(tmp_path):
    body = (
        "1. Build the batch as a dict: {\"ids\": ids}.\n"
        "2. Call llm_query_batched(prompts) and keep {k: v for k, v in zip(ids, outs)}.\n"
    )
    spec = validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)
    harness = materialize_candidate_harness(H0, spec, tmp_path)
    assert harness.skills == [SkillEntry(SKILL_RECORD["name"], SKILL_RECORD["description"], body)]
    assert build_skill_loader(harness.skills)(SKILL_RECORD["name"]) == body


@pytest.mark.parametrize(
    "body",
    [
        "Carefully re-read every partial result and recompute before answering.",
        "1. Only one step is not an ordered procedure.",
        "Step one: re-read.\nStep two: recompute.",
        "re-read\nrecompute\ncompare",
    ],
)
def test_s10_rejects_body_without_ordered_steps(body):
    with pytest.raises(ProposalRejection, match=r"S10.*ordered steps"):
        validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)


@pytest.mark.parametrize(
    "body",
    [
        "1. Re-read.\n2. Recompute.",
        "1) Re-read.\n2) Recompute.\n3) Compare.",
        "- Re-read.\n- Recompute.",
        "Before committing an aggregate:\n\n1. Re-read.\n2. Recompute.\n\nThen answer.",
    ],
)
def test_s10_accepts_ordered_step_bodies(body):
    spec = validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)
    assert spec.edit["skills"][0]["body"] == body


@pytest.mark.parametrize("description", ["Line one.\nLine two.", "Line one.\r\nLine two.", "Trailing\n"])
def test_s10_rejects_multiline_description(description):
    with pytest.raises(ProposalRejection, match=r"S10.*single line"):
        validate_candidate_spec(skills_item(skill(description=description)), ALL_PATTERNS)


def test_s10_rejects_duplicate_names():
    item = skills_item(SKILL_RECORD, skill(description="A second entry with the same name."))
    with pytest.raises(ProposalRejection, match=r"S10.*unique"):
        validate_candidate_spec(item, ALL_PATTERNS)


@pytest.mark.parametrize(
    "name",
    ["verify aggregate", "1st_step", "for", "verify-aggregate", "", " verify", "vérifier", "a.b"],
)
def test_s10_rejects_non_identifier_name(name):
    with pytest.raises(ProposalRejection, match=r"S10.*identifier"):
        validate_candidate_spec(skills_item(skill(name=name)), ALL_PATTERNS)


def test_s10_description_over_index_cap_rejected_but_same_length_body_accepted():
    long_text = "x" * (SKILL_DESCRIPTION_MAX_CHARS + 1)
    with pytest.raises(ProposalRejection, match=rf"S10.*description.*{SKILL_DESCRIPTION_MAX_CHARS}"):
        validate_candidate_spec(skills_item(skill(description=long_text)), ALL_PATTERNS)
    # Same byte count in a body is fine: the body is paid only when loaded.
    body = "1. " + "x" * (SKILL_DESCRIPTION_MAX_CHARS - 7) + "\n2. y"
    assert len(body) == len(long_text)
    assert len(body) <= SKILL_BODY_MAX_CHARS
    spec = validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)
    assert spec.edit["skills"][0]["body"] == body


def test_s10_candidate_ids_distinct_from_s1():
    assert _candidate_id(0, 1, "S1") != _candidate_id(0, 1, "S10")
    assert _candidate_id(0, 1, "S10").endswith("-s10")


@pytest.mark.parametrize(
    "stated_limit",
    [
        "REPL outputs over 20000 characters are truncated",
        "REPL outputs over 1k characters are truncated",
        "each sub-call takes ~8K characters per prompt",
        "send ~16 prompts per batch",
    ],
)
def test_s10_rejects_body_stating_a_runtime_limit(stated_limit):
    body = f"1. Remember: {stated_limit}.\n2. Split the input accordingly."
    with pytest.raises(ProposalRejection, match=r"S10.*runtime limit"):
        validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)


@pytest.mark.parametrize(
    "stated_limit",
    [
        "REPL outputs over 20000 characters are truncated",
        "each sub-call takes ~8K characters per prompt",
        "send ~16 prompts per batch",
    ],
)
def test_s10_rejects_description_stating_a_runtime_limit(stated_limit):
    description = f"Consult when {stated_limit}."
    with pytest.raises(ProposalRejection, match=r"S10.*runtime limit"):
        validate_candidate_spec(skills_item(skill(description=description)), ALL_PATTERNS)


def test_s10_rejects_entry_count_over_cap():
    at_cap = [skill(name=name) for name in _ok_skill_names(SKILL_MAX_ENTRIES)]
    validate_candidate_spec(skills_item(*at_cap), ALL_PATTERNS)
    over = at_cap + [skill(name="one_too_many")]
    with pytest.raises(ProposalRejection, match=rf"S10.*{SKILL_MAX_ENTRIES}"):
        validate_candidate_spec(skills_item(*over), ALL_PATTERNS)


def test_s10_rejects_name_over_cap():
    name = "n" * (SKILL_NAME_MAX_CHARS + 1)
    assert name.isidentifier()
    with pytest.raises(ProposalRejection, match=rf"S10.*name.*{SKILL_NAME_MAX_CHARS}"):
        validate_candidate_spec(skills_item(skill(name=name)), ALL_PATTERNS)


def test_s10_rejects_body_over_cap():
    body = "1. Re-read.\n2. " + "x" * SKILL_BODY_MAX_CHARS
    with pytest.raises(ProposalRejection, match=rf"S10.*body.*{SKILL_BODY_MAX_CHARS}"):
        validate_candidate_spec(skills_item(skill(body=body)), ALL_PATTERNS)


def test_s10_rejects_total_length_over_cap():
    # Every entry is individually within the per-field caps and the entry count
    # is within its cap, yet the sum of all fields exceeds the total cap.
    per_body = SKILL_BODY_MAX_CHARS - 100
    count = SKILL_TOTAL_MAX_CHARS // per_body + 1
    assert count <= SKILL_MAX_ENTRIES
    body = "1. Re-read.\n2. " + "x" * (per_body - 15)
    assert len(body) <= SKILL_BODY_MAX_CHARS
    records = [skill(name=name, body=body) for name in _ok_skill_names(count)]
    with pytest.raises(ProposalRejection, match=rf"S10.*total.*{SKILL_TOTAL_MAX_CHARS}"):
        validate_candidate_spec(skills_item(*records), ALL_PATTERNS)


@pytest.mark.parametrize(
    "value", ["1. Re-read.\n2. Recompute.", {"name": "x", "description": "y", "body": "1. a\n2. b"}, None]
)
def test_s10_rejects_non_list_value(value):
    with pytest.raises(ProposalRejection, match=r"S10.*list"):
        validate_candidate_spec(edit_item(5, {"kind": "skills", "skills": value}), ALL_PATTERNS)


@pytest.mark.parametrize("entry", ["", "verify_aggregate", 3, ["verify_aggregate"], None])
def test_s10_rejects_non_record_entry(entry):
    with pytest.raises(ProposalRejection, match=r"S10.*record"):
        validate_candidate_spec(skills_item(entry), ALL_PATTERNS)


@pytest.mark.parametrize("field_name", ["name", "description", "body"])
def test_s10_rejects_empty_string_field(field_name):
    with pytest.raises(ProposalRejection, match=rf"S10.*{field_name}"):
        validate_candidate_spec(skills_item(skill(**{field_name: ""})), ALL_PATTERNS)
    with pytest.raises(ProposalRejection, match=rf"S10.*{field_name}"):
        validate_candidate_spec(skills_item(skill(**{field_name: "   "})), ALL_PATTERNS)


@pytest.mark.parametrize(
    "record",
    [
        {"name": "verify_aggregate", "description": "d"},  # missing body
        {**SKILL_RECORD, "steps": "extra"},  # unknown key
        skill(body=["1. a", "2. b"]),  # non-string field
    ],
)
def test_s10_rejects_malformed_record(record):
    with pytest.raises(ProposalRejection, match="S10"):
        validate_candidate_spec(skills_item(record), ALL_PATTERNS)


def test_s10_rejects_wrong_edit_kind_for_the_surface():
    with pytest.raises(ProposalRejection, match="edit.kind='skills'"):
        validate_candidate_spec(edit_item(5, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)
    with pytest.raises(ProposalRejection, match="edit.kind='text'"):
        validate_candidate_spec(edit_item(0, {"kind": "skills", "skills": [SKILL_RECORD]}), ALL_PATTERNS)


def test_s10_well_formed_edit_materializes_changing_only_s10(tmp_path):
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    incumbent_serialization = serialize_harness(H0)
    harness, serialization = build_candidate(H0, incumbent_serialization, spec, tmp_path)
    assert changed_surfaces(incumbent_serialization, serialization) == ["S10"]
    assert harness.skills == [SkillEntry(**SKILL_RECORD)]
    assert serialization["surfaces"]["S10_skills"] == [SKILL_RECORD]


def test_s10_edit_replaces_the_whole_list(tmp_path):
    # KD2: S10 is edited whole -- the proposed list is the new value, not an append.
    incumbent = replace(H0, skills=[SkillEntry("old_skill", "Old.", "1. a\n2. b")])
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    harness = materialize_candidate_harness(incumbent, spec, tmp_path)
    assert [entry.name for entry in harness.skills] == [SKILL_RECORD["name"]]


def test_rendered_prompt_names_s10_and_its_edit_format():
    rendered, _ = render_prompt(ALL_PATTERNS, serialize_harness(H0), (), (), k=4)
    assert "S10" in rendered
    assert '"kind": "skills"' in rendered
    assert "ten editable surfaces" in rendered
    assert "nine editable surfaces" not in rendered
    # The brace rule is stated to the proposer: index fields brace-free, bodies raw.
    assert "brace" in rendered.lower()
    assert "unconsulted_procedure" in rendered


def test_s8_helper_binding_the_skill_loader_name_is_rejected_naming_s10():
    source = f"def {SKILL_LOADER_NAME}(name):\n    return name\n"
    item = edit_item(
        4, {"kind": "repl_helper", "dict": "repl_helpers", "name": SKILL_LOADER_NAME, "source": source}
    )
    with pytest.raises(ProposalRejection, match=r"S10"):
        validate_candidate_spec(item, ALL_PATTERNS)
    item = edit_item(
        4,
        {"kind": "repl_helper", "dict": "sub_repl_helpers", "name": SKILL_LOADER_NAME, "source": source},
    )
    with pytest.raises(ProposalRejection, match=rf"{SKILL_LOADER_NAME}.*S10"):
        validate_candidate_spec(item, ALL_PATTERNS)
