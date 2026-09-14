"""Tests for the stage-2 proposer (``shrlm-proposal/v1`` output).

The proposer is validated the way the interface doc recommends: build a
candidate, write it, then run it through the *real* stage-3 loader
(``shrlm.optimization.candidates.load_candidates``) and assert zero
rejections. That is the strongest guarantee available -- it proves the
proposer's output clears the actual gate, not a re-description of it.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from shrlm.harness_identity import serialize_harness
from shrlm.optimization.candidates import LoadedCandidate, changed_surfaces, load_candidates
from shrlm.optimization.driver import RoundPersistenceError
from shrlm.optimization.proposal import (
    OOLONG_RECORD_GUIDANCE,
    SKILL_BODY_MAX_CHARS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_MAX_ENTRIES,
    SKILL_NAME_MAX_CHARS,
    SKILL_TOTAL_MAX_CHARS,
    MaterializationFailure,
    ProposalBudgetExhausted,
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
from shrlm.rlm_harness import H0, SKILL_LOADER_NAME, Harness, SkillEntry
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
PATTERN_CODE_S9 = make_pattern("premature_termination")  # explicit S9 below
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
    "def safe_index(seq, i):\n    if 0 <= i < len(seq):\n        return seq[i]\n    return None\n"
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
        "incumbent_behavior": "Submits the computed result.",
        "observed_failure": "The trace does not compare the aggregate with its inputs.",
        "behavioral_change": "Compare the aggregate with its inputs before committing.",
        "edit": edit,
        "predicted_effect": "the root double-checks before answering",
        "regression_risks": ["one extra turn per run"],
    }
    if pattern_index == 3:
        item["surface"] = "S9"
    item.update(overrides)
    return item


TEXT_ITEM = edit_item(0, {"kind": "text", "new_text": "Restate the answer before submitting it."})
POLICY_ITEM = edit_item(1, {"kind": "policy", "runtime_policy": {"enabled": True, "max_depth": 3}})
CODE_S7_ITEM = edit_item(2, {"kind": "code", "source": S7_SOURCE})
CODE_S9_ITEM = edit_item(3, {"kind": "code", "source": S9_SOURCE})
REPL_HELPER_ITEM = edit_item(
    4, {"kind": "repl_helper", "dict": "repl_helpers", "name": "safe_index", "source": S8_SOURCE}
)


def skills_item(**fields: Any) -> dict[str, Any]:
    """An S10 edit over the S10 pattern (index 5): one skill, added or replaced by name."""
    return edit_item(5, {"kind": "skills", **SKILL_RECORD, **fields})


SKILLS_ITEM = skills_item()


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


def test_validate_candidate_spec_accepts_other_on_any_surface():
    # Taxonomy 3.1.0: OTHER is addressable; the proposer names the surface.
    spec = validate_candidate_spec(
        edit_item(6, {"kind": "text", "new_text": "x"}, surface="S3"), ALL_PATTERNS
    )
    assert spec.surface == "S3"


def test_validate_candidate_spec_defaults_to_the_primary_surface():
    spec = validate_candidate_spec(edit_item(0, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)
    assert spec.surface == "S4"  # skipped_verification's primary


def test_validate_candidate_spec_accepts_an_eligible_alternate_surface():
    # skipped_verification may also be addressed on S9 (answer middleware).
    source = (
        "def accept_answer(answer, repl_inventory):\n    return AnswerDecision.accept(answer)\n"
    )
    spec = validate_candidate_spec(
        edit_item(0, {"kind": "code", "source": source}, surface="S9"), ALL_PATTERNS
    )
    assert spec.surface == "S9"
    assert spec.edit["kind"] == "code"


def test_validate_candidate_spec_rejects_a_surface_outside_the_eligible_set():
    with pytest.raises(ProposalRejection, match="not eligible"):
        validate_candidate_spec(
            edit_item(0, {"kind": "text", "new_text": "x"}, surface="S2"), ALL_PATTERNS
        )


def test_validate_candidate_spec_edit_shape_follows_the_chosen_surface():
    # S9 chosen but a text edit supplied: rejected on shape, not on surface.
    with pytest.raises(ProposalRejection, match="needs edit.kind='code'"):
        validate_candidate_spec(
            edit_item(0, {"kind": "text", "new_text": "x"}, surface="S9"), ALL_PATTERNS
        )


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
        validate_candidate_spec(
            edit_item(2, {"kind": "code", "source": "def build_metadata(:\n"}), ALL_PATTERNS
        )


def test_validate_candidate_spec_rejects_repl_helper_name_mismatch():
    item = edit_item(
        4,
        {"kind": "repl_helper", "dict": "repl_helpers", "name": "wrong_name", "source": S8_SOURCE},
    )
    with pytest.raises(ProposalRejection, match="must match the function name"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_validate_candidate_spec_rejects_reserved_repl_helper_name():
    source = "def llm_query(prompt):\n    return prompt\n"
    item = edit_item(
        4, {"kind": "repl_helper", "dict": "repl_helpers", "name": "llm_query", "source": source}
    )
    with pytest.raises(ProposalRejection, match="reserved"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_validate_candidate_spec_rejects_empty_predicted_effect():
    with pytest.raises(ProposalRejection, match="predicted_effect"):
        validate_candidate_spec(
            edit_item(0, {"kind": "text", "new_text": "x"}, predicted_effect=""), ALL_PATTERNS
        )


@pytest.mark.parametrize("field", ["incumbent_behavior", "observed_failure", "behavioral_change"])
def test_new_proposals_require_a_bounded_behavioral_difference(field):
    item = {
        **TEXT_ITEM,
        "incumbent_behavior": "Counts only.",
        "observed_failure": "The predicate needs per-user dates.",
        "behavioral_change": "Preserve record IDs and join labels to user/date metadata.",
    }
    item.pop(field)
    with pytest.raises(ProposalRejection, match=field):
        validate_candidate_spec(item, ALL_PATTERNS)
    item[field] = "x" * 601
    with pytest.raises(ProposalRejection, match=field):
        validate_candidate_spec(item, ALL_PATTERNS)


@pytest.mark.parametrize("change", ["none", "NO CHANGE", "no-op", "unchanged"])
def test_explicit_absent_behavioral_change_is_rejected(change):
    item = {
        **TEXT_ITEM,
        "incumbent_behavior": "Checks coverage.",
        "observed_failure": "Some pairs are missing.",
        "behavioral_change": change,
    }
    with pytest.raises(ProposalRejection, match="behavioral_change"):
        validate_candidate_spec(item, ALL_PATTERNS)


@pytest.mark.parametrize(
    "rows,valid",
    [
        ([(2, "b"), (1, "a")], True),
        ([(1, "a"), (1, "b")], False),
        ([(1, "a")], False),
        ([(1, "a"), (3, "b")], False),
        ([(1, "a"), (2, "unknown")], False),
    ],
)
def test_record_guidance_example_checks_rows_before_mapping(rows, valid):
    code = OOLONG_RECORD_GUIDANCE.split("```python\n")[1].split("```")[0]
    namespace = {"returned_rows": rows, "record_ids": [1, 2], "task_labels": {"a", "b"}}
    if valid:
        exec(code, namespace)
        assert namespace["labels_by_id"] == {1: "a", 2: "b"}
    else:
        with pytest.raises(AssertionError):
            exec(code, namespace)
        assert "labels_by_id" not in namespace


def test_behavioral_fields_persist_and_legacy_loader_remains_compatible(tmp_path):
    spec = validate_candidate_spec(TEXT_ITEM, ALL_PATTERNS)
    incumbent = serialize_harness(H0)
    _, serialization = build_candidate(H0, incumbent, spec, tmp_path / "work")
    path = write_proposal(
        tmp_path / "proposals", "r01-c01-s4", incumbent, spec, serialization, "mock", "prompt"
    )
    payload = json.loads(path.read_text())
    for name in ("incumbent_behavior", "observed_failure", "behavioral_change"):
        assert payload.pop(name) == TEXT_ITEM[name]
    loaded, rejections = load_candidates(tmp_path / "proposals", H0)
    assert len(loaded) == 1 and not rejections
    path.write_text(json.dumps(payload))
    loaded, rejections = load_candidates(tmp_path / "proposals", H0)
    assert len(loaded) == 1 and not rejections


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
    assert "no prior rounds" in rendered.lower()
    # Taxonomy 3.1.0: every recognized mechanism is addressable, OTHER included,
    # and each pattern advertises its eligible surfaces with the primary first.
    assert [index for index, _ in addressable] == [0, 1, 2, 3, 4, 5, 6]
    assert "eligible surfaces: S4" in rendered
    assert "-- primary" in rendered


def test_render_prompt_passing_and_history_blocks():
    incumbent_serialization = serialize_harness(H0)
    passing = [{"instance_id": "bfs-1", "passed": True}]
    history = [
        (
            [{"subject_id": "r00-c01-s4", "decision": "rejected", "reasons": ["cost too high"]}],
            {"promoted": False, "promoted_harness_hash": None},
        )
    ]
    rendered, _ = render_prompt(ALL_PATTERNS, incumbent_serialization, passing, history, k=4)
    assert "bfs-1" in rendered
    assert "r00-c01-s4" in rendered
    assert "cost too high" in rendered


def test_render_prompt_truncates_huge_verifier_evidence_but_never_the_pattern():
    """Verifier evidence quotes unbounded model output; the PROMPT bounds it
    at render time (the $5 proof's ungoverned char cap depends on it) while
    the pattern dict -- what a persisted bundle holds -- keeps the full text."""
    import copy

    huge = "x" * 50_000
    pattern = copy.deepcopy(PATTERN_TEXT)
    pattern["verifier_evidence"] = [f"a: produced {huge}"]

    rendered, addressable = render_prompt([pattern], serialize_harness(H0), (), (), k=4)
    assert [index for index, _ in addressable] == [0]
    assert huge not in rendered
    assert "[truncated" in rendered
    assert "x" * 2001 not in rendered
    # Persisted evidence stays complete: only the prompt string was bounded.
    assert huge in pattern["verifier_evidence"][0]


# ---------------------------------------------------------------------------
# load_passing_behaviors
# ---------------------------------------------------------------------------


def test_load_passing_behaviors_filters_passed(tmp_path):
    instances = [{"id": "a"}, {"id": "b"}]
    runs = [
        {"run_id": "b__a02", "instance_id": "b", "attempt": 2, "passed": True},
        {"run_id": "b__a01", "instance_id": "b", "attempt": 1, "passed": False},
        {"run_id": "a__a02", "instance_id": "a", "attempt": 2, "passed": True},
        {"run_id": "a__a01", "instance_id": "a", "attempt": 1, "passed": True},
    ]
    (tmp_path / "instances.jsonl").write_text(
        "".join(json.dumps(instance) + "\n" for instance in instances)
    )
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    result = load_passing_behaviors(tmp_path)
    assert [run["run_id"] for run in result] == ["a__a01", "a__a02", "b__a02"]


def test_load_passing_behaviors_stabilizes_the_proposal_prompt_hash(tmp_path):
    instances = [{"id": "a"}, {"id": "b"}]
    runs = [
        {"run_id": "a__a01", "instance_id": "a", "attempt": 1, "passed": True},
        {"run_id": "a__a02", "instance_id": "a", "attempt": 2, "passed": True},
        {"run_id": "b__a01", "instance_id": "b", "attempt": 1, "passed": True},
        {"run_id": "b__a02", "instance_id": "b", "attempt": 2, "passed": True},
    ]
    (tmp_path / "instances.jsonl").write_text(
        "".join(json.dumps(instance) + "\n" for instance in instances)
    )

    def proposed_hash(entries, label):
        (tmp_path / "runs.jsonl").write_text("".join(json.dumps(run) + "\n" for run in entries))
        result = propose_round(
            BUNDLE,
            H0,
            MockLM(responses=[canned_batch(TEXT_ITEM)]),
            tmp_path / label,
            passing_behaviors=load_passing_behaviors(tmp_path),
            workdir=tmp_path / f"{label}-work",
        )
        return result.prompt_sha256

    assert proposed_hash(list(reversed(runs)), "reversed") == proposed_hash(runs, "canonical")


def test_load_passing_behaviors_rejects_an_unknown_instance_id(tmp_path):
    (tmp_path / "instances.jsonl").write_text(json.dumps({"id": "known"}) + "\n")
    (tmp_path / "runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "unknown__a01",
                "instance_id": "unknown",
                "attempt": 1,
                "passed": True,
            }
        )
        + "\n"
    )

    with pytest.raises(RoundPersistenceError, match="unknown"):
        load_passing_behaviors(tmp_path)


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
    # The S8, S10, and OTHER patterns were addressable and never proposed for.
    assert result.skipped_patterns == [4, 5, 6]

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
    propose_round(BUNDLE, H0, lm, tmp_path / "proposals", cache=cache, workdir=tmp_path / "work")
    assert lm._call_count == 1
    # Second round with an empty responses list: a cache miss would raise IndexError.
    lm2 = MockLM(model_name="mock-proposer", responses=[])
    propose_round(BUNDLE, H0, lm2, tmp_path / "proposals2", cache=cache, workdir=tmp_path / "work2")
    assert lm2._call_count == 0


def test_propose_round_materialization_failure_does_not_drop_the_rest(tmp_path):
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    response = canned_batch(TEXT_ITEM, no_op_policy)
    lm = MockLM(model_name="mock-proposer", responses=[response])
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        workdir=tmp_path / "work",
        config=ProposerConfig(max_attempts=1),
    )
    assert len(result.written) == 1
    assert result.written[0].surface == "S4"
    assert len(result.materialization_failures) == 1
    assert result.materialization_failures[0].pattern_index == 1


def test_preflight_repairs_only_failed_member_and_replays(tmp_path):
    broken = edit_item(
        3,
        {
            "kind": "code",
            "def_name": "broken",
            "source": "def broken(answer, repl_inventory):\n"
            "    if answer.startswith('['):\n"
            "        return AnswerDecision.accept()\n"
            "    return AnswerDecision.accept(answer)\n",
        },
    )
    bundle = {**BUNDLE, "config": {"verifier_config": {"environment": "oolong_pairs"}}}
    lm = MockLM(responses=[canned_batch(TEXT_ITEM, broken), canned_batch(CODE_S9_ITEM)])
    cache = ProposalCache()
    kwargs = dict(cache=cache, workdir=tmp_path / "work")
    result = propose_round(bundle, H0, lm, tmp_path / "proposals", **kwargs)
    assert [w.candidate_id for w in result.written] == ["r00-c01-s4", "r00-c02-s9"]
    assert len(result.attempts) == 2
    assert "bracketed_pair" in result.attempts[0].violation
    assert result.preflight_failures == []
    replay = propose_round(bundle, H0, lm, tmp_path / "proposals", **kwargs)
    assert all(a.cached for a in replay.attempts)


def test_propose_round_survivor_keeps_its_batch_position_after_a_failure(tmp_path):
    """A refused candidate still consumes its position: the survivor behind a
    failed first candidate is written as c02, not renumbered to c01, because
    the candidate id is hash material and a ledger subject id."""
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    response = canned_batch(no_op_policy, TEXT_ITEM)
    lm = MockLM(model_name="mock-proposer", responses=[response])
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        workdir=tmp_path / "work",
        config=ProposerConfig(max_attempts=1),
    )
    assert [w.candidate_id for w in result.written] == ["r00-c02-s4"]
    assert len(result.materialization_failures) == 1
    assert result.materialization_failures[0].pattern_index == 1


def test_propose_round_reasks_when_the_whole_batch_fails_to_materialize(tmp_path):
    """A batch whose every candidate is a no-op is rejected with the reason and
    re-asked, exactly like a malformed batch (KTD1)."""
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(
        model_name="mock-proposer",
        responses=[canned_batch(no_op_policy), canned_batch(POLICY_ITEM)],
    )
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=3),
        workdir=tmp_path / "work",
    )
    assert len(result.attempts) == 2
    first, second = result.attempts
    assert first.accepted is False
    assert "pattern 1" in first.violation and "S6" in first.violation
    assert "already" in first.violation  # the surface already holds this text
    assert second.accepted is True
    assert [w.surface for w in result.written] == ["S6"]
    # The failure that was re-asked away lives in the attempt trail, not in the
    # round's failure records: those describe the accepted attempt only.
    assert result.materialization_failures == []


def test_propose_round_exhaustion_on_no_ops_returns_an_empty_result(tmp_path):
    """Exhaustion whose final attempt validated but did not materialize is a
    proposer-quality outcome, not an exception (KTD2): the round closes with
    zero candidates and its failure records, and nothing escapes to crash a
    resumable run."""
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(
        model_name="mock-proposer",
        responses=[canned_batch(no_op_policy), canned_batch(no_op_policy)],
    )
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=2),
        workdir=tmp_path / "work",
    )
    assert result.written == []
    assert len(result.attempts) == 2
    assert all(a.accepted is False for a in result.attempts)
    assert len(result.materialization_failures) == 1
    record = result.materialization_failures[0]
    assert (record.pattern_index, record.surface) == (1, "S6")
    assert "no surface" in record.reason
    # The refused pattern was proposed, so it is not "skipped".
    assert 1 not in result.skipped_patterns
    assert list((tmp_path / "proposals").glob("r00-*")) == []


def test_propose_round_mixed_malformed_then_no_op_exhaustion_returns_empty(tmp_path):
    """The exhaustion branch is classified by the FINAL attempt (KTD2): a
    malformed first response followed by a no-op does not fall through to the
    ProposalRejection raise."""
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(
        model_name="mock-proposer",
        responses=["not json at all", canned_batch(no_op_policy)],
    )
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=2),
        workdir=tmp_path / "work",
    )
    assert result.written == []
    assert len(result.materialization_failures) == 1
    assert [a.accepted for a in result.attempts] == [False, False]


def test_propose_round_malformed_repair_seals_original_failures(tmp_path):
    no_op_policy = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(responses=[canned_batch(no_op_policy), "not json at all"])
    result = propose_round(
        BUNDLE,
        H0,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=8),
        workdir=tmp_path / "work",
    )
    assert result.written == []
    assert len(result.attempts) == 2
    assert len(result.materialization_failures) == 1


def test_propose_round_empty_bundle_no_crash(tmp_path):
    empty_bundle = {"bundle_id": "empty", "patterns": []}
    lm = MockLM(model_name="mock-proposer", responses=["```json\n[]\n```"])
    result = propose_round(empty_bundle, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert result.written == []
    assert result.skipped_patterns == []


def test_propose_round_budget_exhaustion_is_raised_once_without_re_asking(tmp_path, monkeypatch):
    """A reasoning-exhausted proposer response (the client's
    ``TokenLimitExceededError``, R6/KTD3) is deterministic for the prompt:
    it is neither a transport glitch to re-send nor a rejected attempt to
    re-ask, so it surfaces once with the attempts so far."""
    from rlm.utils.exceptions import TokenLimitExceededError

    monkeypatch.setattr("shrlm.optimization.proposal.time.sleep", lambda _s: None)

    def boom(prompt: Any) -> str:
        raise TokenLimitExceededError(
            tokens_used=16384, token_limit=16384, message="output budget exhausted by reasoning"
        )

    lm = MockLM(model_name="mock-proposer", response_fn=boom)
    with pytest.raises(ProposalBudgetExhausted, match="reasoning") as excinfo:
        propose_round(
            BUNDLE,
            H0,
            lm,
            tmp_path / "proposals",
            config=ProposerConfig(max_attempts=3, transport_retries=3),
            workdir=tmp_path / "work",
        )
    assert lm._call_count == 1
    assert excinfo.value.attempts == []
    assert not any((tmp_path / "proposals").glob("*/proposal.json"))


def test_propose_round_raises_after_exhausting_attempts(tmp_path):
    lm = MockLM(model_name="mock-proposer", responses=["garbage", "garbage", "garbage"])
    with pytest.raises(ProposalRejection):
        propose_round(
            BUNDLE,
            H0,
            lm,
            tmp_path / "proposals",
            config=ProposerConfig(max_attempts=3),
            workdir=tmp_path / "work",
        )


# ---------------------------------------------------------------------------
# S10: the skills edit kind (R3, R5, R6, R7, R14, R15). One named skill,
# added or replacing the same name (like S8). Rejection cases first.
# ---------------------------------------------------------------------------


def _ok_skill_names(count: int) -> list[str]:
    return [f"skill_{index}" for index in range(count)]


def _incumbent_with_skills(*names: str, body: str = "1. a\n2. b") -> Harness:
    return replace(
        H0,
        skills=[SkillEntry(name, "Consult for x.", body) for name in names],
    )


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
        validate_candidate_spec(skills_item(description=description), ALL_PATTERNS)


def test_s10_rejects_custom_tools_slot_in_description():
    item = skills_item(description="See {custom_tools_section} before answering.")
    with pytest.raises(ProposalRejection, match="S10"):
        validate_candidate_spec(item, ALL_PATTERNS)


def test_s10_accepts_undoubled_braces_in_body_and_loader_returns_it_verbatim(tmp_path):
    body = (
        '1. Build the batch as a dict: {"ids": ids}.\n'
        "2. Call llm_query_batched(prompts) and keep {k: v for k, v in zip(ids, outs)}.\n"
    )
    spec = validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)
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
        validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)


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
    spec = validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)
    assert spec.edit["body"] == body


@pytest.mark.parametrize(
    "description", ["Line one.\nLine two.", "Line one.\r\nLine two.", "Trailing\n"]
)
def test_s10_rejects_multiline_description(description):
    with pytest.raises(ProposalRejection, match=r"S10.*single line"):
        validate_candidate_spec(skills_item(description=description), ALL_PATTERNS)


@pytest.mark.parametrize(
    "name",
    ["verify aggregate", "1st_step", "for", "verify-aggregate", "", " verify", "vérifier", "a.b"],
)
def test_s10_rejects_non_identifier_name(name):
    with pytest.raises(ProposalRejection, match=r"S10.*identifier"):
        validate_candidate_spec(skills_item(name=name), ALL_PATTERNS)


def test_s10_description_over_index_cap_rejected_but_same_length_body_accepted():
    long_text = "x" * (SKILL_DESCRIPTION_MAX_CHARS + 1)
    with pytest.raises(
        ProposalRejection, match=rf"S10.*description.*{SKILL_DESCRIPTION_MAX_CHARS}"
    ):
        validate_candidate_spec(skills_item(description=long_text), ALL_PATTERNS)
    # Same byte count in a body is fine: the body is paid only when loaded.
    body = "1. " + "x" * (SKILL_DESCRIPTION_MAX_CHARS - 7) + "\n2. y"
    assert len(body) == len(long_text)
    assert len(body) <= SKILL_BODY_MAX_CHARS
    spec = validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)
    assert spec.edit["body"] == body


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
        validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)


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
        validate_candidate_spec(skills_item(description=description), ALL_PATTERNS)


def test_s10_rejects_name_over_cap():
    name = "n" * (SKILL_NAME_MAX_CHARS + 1)
    assert name.isidentifier()
    with pytest.raises(ProposalRejection, match=rf"S10.*name.*{SKILL_NAME_MAX_CHARS}"):
        validate_candidate_spec(skills_item(name=name), ALL_PATTERNS)


def test_s10_rejects_body_over_cap():
    body = "1. Re-read.\n2. " + "x" * SKILL_BODY_MAX_CHARS
    with pytest.raises(ProposalRejection, match=rf"S10.*body.*{SKILL_BODY_MAX_CHARS}"):
        validate_candidate_spec(skills_item(body=body), ALL_PATTERNS)


def test_s10_rejects_whole_list_edit():
    with pytest.raises(ProposalRejection, match=r"S10.*one skill"):
        validate_candidate_spec(
            edit_item(5, {"kind": "skills", "skills": [SKILL_RECORD]}), ALL_PATTERNS
        )


@pytest.mark.parametrize("field_name", ["name", "description", "body"])
def test_s10_rejects_empty_string_field(field_name):
    with pytest.raises(ProposalRejection, match=rf"S10.*{field_name}"):
        validate_candidate_spec(skills_item(**{field_name: ""}), ALL_PATTERNS)
    with pytest.raises(ProposalRejection, match=rf"S10.*{field_name}"):
        validate_candidate_spec(skills_item(**{field_name: "   "}), ALL_PATTERNS)


@pytest.mark.parametrize(
    "edit",
    [
        {"kind": "skills", "name": "verify_aggregate", "description": "d"},  # missing body
        {**SKILL_RECORD, "kind": "skills", "steps": "extra"},  # unknown key
        {**SKILL_RECORD, "kind": "skills", "body": ["1. a", "2. b"]},  # non-string field
    ],
)
def test_s10_rejects_malformed_edit(edit):
    with pytest.raises(ProposalRejection, match="S10"):
        validate_candidate_spec(edit_item(5, edit), ALL_PATTERNS)


def test_s10_rejects_wrong_edit_kind_for_the_surface():
    with pytest.raises(ProposalRejection, match="edit.kind='skills'"):
        validate_candidate_spec(edit_item(5, {"kind": "text", "new_text": "x"}), ALL_PATTERNS)
    with pytest.raises(ProposalRejection, match="edit.kind='text'"):
        validate_candidate_spec(edit_item(0, {"kind": "skills", **SKILL_RECORD}), ALL_PATTERNS)


def test_s10_well_formed_edit_materializes_changing_only_s10(tmp_path):
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    incumbent_serialization = serialize_harness(H0)
    harness, serialization = build_candidate(H0, incumbent_serialization, spec, tmp_path)
    assert changed_surfaces(incumbent_serialization, serialization) == ["S10"]
    assert harness.skills == [SkillEntry(**SKILL_RECORD)]
    assert serialization["surfaces"]["S10_skills"] == [SKILL_RECORD]


def test_s10_edit_adds_one_skill_keeping_existing(tmp_path):
    incumbent = _incumbent_with_skills("old_skill")
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    harness = materialize_candidate_harness(incumbent, spec, tmp_path)
    assert [entry.name for entry in harness.skills] == ["old_skill", SKILL_RECORD["name"]]


def test_s10_edit_replaces_existing_skill_of_the_same_name(tmp_path):
    incumbent = replace(
        H0,
        skills=[SkillEntry(SKILL_RECORD["name"], "Old description.", "1. old\n2. steps")],
    )
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    harness = materialize_candidate_harness(incumbent, spec, tmp_path)
    assert harness.skills == [SkillEntry(**SKILL_RECORD)]


def test_s10_ninth_distinct_name_fails_materialization(tmp_path):
    incumbent = _incumbent_with_skills(*_ok_skill_names(SKILL_MAX_ENTRIES))
    spec = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    with pytest.raises(MaterializationFailure, match=rf"S10.*{SKILL_MAX_ENTRIES}"):
        build_candidate(incumbent, serialize_harness(incumbent), spec, tmp_path)


def test_s10_replace_at_entry_cap_succeeds(tmp_path):
    names = _ok_skill_names(SKILL_MAX_ENTRIES)
    incumbent = _incumbent_with_skills(*names)
    spec = validate_candidate_spec(
        skills_item(name=names[0], description="Consult before replacing at cap."),
        ALL_PATTERNS,
    )
    harness = materialize_candidate_harness(incumbent, spec, tmp_path)
    assert len(harness.skills) == SKILL_MAX_ENTRIES
    assert harness.skills[0].description == "Consult before replacing at cap."
    assert [entry.name for entry in harness.skills] == names


def test_s10_merged_total_length_over_cap_fails_materialization(tmp_path):
    per_body = SKILL_BODY_MAX_CHARS - 100
    count = SKILL_TOTAL_MAX_CHARS // per_body
    assert 1 < count <= SKILL_MAX_ENTRIES
    large_body = "1. Re-read.\n2. " + "x" * (per_body - 15)
    assert len(large_body) <= SKILL_BODY_MAX_CHARS
    incumbent = _incumbent_with_skills(*_ok_skill_names(count), body=large_body)
    spec = validate_candidate_spec(skills_item(body=large_body), ALL_PATTERNS)
    with pytest.raises(MaterializationFailure, match=rf"S10.*total.*{SKILL_TOTAL_MAX_CHARS}"):
        build_candidate(incumbent, serialize_harness(incumbent), spec, tmp_path)


def test_rendered_prompt_names_s10_and_its_edit_format():
    rendered, _ = render_prompt(ALL_PATTERNS, serialize_harness(H0), (), (), k=4)
    assert "S10" in rendered
    assert '"kind": "skills"' in rendered
    assert "ten editable surfaces" in rendered
    assert "nine editable surfaces" not in rendered
    # The brace rule is stated to the proposer: index fields brace-free, bodies raw.
    assert "brace" in rendered.lower()
    assert "unconsulted_procedure" in rendered
    # One skill, merged by name -- not a whole-list rewrite.
    assert "WHOLE list" not in rendered
    assert "added or replacing" in rendered
    # Skill-writing guidance: distill traces into a procedure, not a YAML dump
    # or a kebab-case SKILL.md. The body template and the identifier rule are
    # what stop the proposer inventing a format the validator will reject.
    assert "procedural anchor" in rendered
    assert "## Use When" in rendered
    assert "## Don't Use When" in rendered
    assert "## Pitfalls" in rendered
    assert "## Verify" in rendered
    assert "encode the process" in rendered
    assert "kebab-case" not in rendered
    assert "REPL-safe identifier" in rendered


def test_s8_helper_binding_the_skill_loader_name_is_rejected_naming_s10():
    source = f"def {SKILL_LOADER_NAME}(name):\n    return name\n"
    item = edit_item(
        4,
        {
            "kind": "repl_helper",
            "dict": "repl_helpers",
            "name": SKILL_LOADER_NAME,
            "source": source,
        },
    )
    with pytest.raises(ProposalRejection, match=r"S10"):
        validate_candidate_spec(item, ALL_PATTERNS)
    item = edit_item(
        4,
        {
            "kind": "repl_helper",
            "dict": "sub_repl_helpers",
            "name": SKILL_LOADER_NAME,
            "source": source,
        },
    )
    with pytest.raises(ProposalRejection, match=rf"{SKILL_LOADER_NAME}.*S10"):
        validate_candidate_spec(item, ALL_PATTERNS)


# ---------------------------------------------------------------------------
# S10: proposal-time merge dry-run (over-cap edits are re-asked, not silently
# dropped at materialization), the removal form, the prompt inventory line,
# missing-field rejections, and the conditional pedagogy block.
# ---------------------------------------------------------------------------


def test_s10_over_entry_cap_edit_is_reasked_with_coaching(tmp_path):
    """A ninth distinct name is a re-askable ProposalRejection during candidate
    validation, not a silent MaterializationFailure: the re-ask names the caps,
    the current totals, and the existing entry names."""
    incumbent = _incumbent_with_skills(*_ok_skill_names(SKILL_MAX_ENTRIES))
    over_cap = canned_batch(SKILLS_ITEM)  # a new ninth name
    corrected = canned_batch(skills_item(name="skill_0"))  # replaces an existing name
    lm = MockLM(model_name="mock-proposer", responses=[over_cap, corrected])
    result = propose_round(
        BUNDLE,
        incumbent,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=3),
        workdir=tmp_path / "work",
    )
    assert result.materialization_failures == []
    assert len(result.written) == 1
    assert result.written[0].surface == "S10"
    assert len(result.attempts) == 2
    violation = result.attempts[0].violation
    assert str(SKILL_MAX_ENTRIES) in violation
    assert f"{SKILL_TOTAL_MAX_CHARS:,}" in violation
    for name in _ok_skill_names(SKILL_MAX_ENTRIES):
        assert name in violation


def test_s10_over_total_cap_edit_is_reasked(tmp_path):
    per_body = SKILL_BODY_MAX_CHARS - 100
    count = SKILL_TOTAL_MAX_CHARS // per_body
    large_body = "1. Re-read.\n2. " + "x" * (per_body - 15)
    incumbent = _incumbent_with_skills(*_ok_skill_names(count), body=large_body)
    over_cap = canned_batch(skills_item(body=large_body))
    corrected = canned_batch(SKILLS_ITEM)
    lm = MockLM(model_name="mock-proposer", responses=[over_cap, corrected])
    result = propose_round(
        BUNDLE,
        incumbent,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=3),
        workdir=tmp_path / "work",
    )
    assert result.materialization_failures == []
    assert len(result.written) == 1
    assert result.attempts[0].accepted is False
    assert str(SKILL_TOTAL_MAX_CHARS) in result.attempts[0].violation


def removal_item(name: str) -> dict[str, Any]:
    """The S10 removal form: both description and body empty strings."""
    return edit_item(5, {"kind": "skills", "name": name, "description": "", "body": ""})


def test_s10_removal_deletes_the_named_entry(tmp_path):
    incumbent = _incumbent_with_skills("keep_me", "drop_me")
    spec = validate_candidate_spec(removal_item("drop_me"), ALL_PATTERNS)
    incumbent_serialization = serialize_harness(incumbent)
    harness, serialization = build_candidate(incumbent, incumbent_serialization, spec, tmp_path)
    assert [entry.name for entry in harness.skills] == ["keep_me"]
    assert changed_surfaces(incumbent_serialization, serialization) == ["S10"]


def test_s10_removal_of_unknown_name_is_rejected_naming_existing_entries(tmp_path):
    incumbent = _incumbent_with_skills("real_one", "real_two")
    spec = validate_candidate_spec(removal_item("ghost"), ALL_PATTERNS)
    with pytest.raises(MaterializationFailure, match=r"real_one.*real_two"):
        build_candidate(incumbent, serialize_harness(incumbent), spec, tmp_path)


def test_s10_removal_of_unknown_name_is_reasked_at_proposal_time(tmp_path):
    incumbent = _incumbent_with_skills("real_one")
    bad = canned_batch(removal_item("ghost"))
    corrected = canned_batch(removal_item("real_one"))
    lm = MockLM(model_name="mock-proposer", responses=[bad, corrected])
    result = propose_round(
        BUNDLE,
        incumbent,
        lm,
        tmp_path / "proposals",
        config=ProposerConfig(max_attempts=3),
        workdir=tmp_path / "work",
    )
    assert result.materialization_failures == []
    assert len(result.written) == 1
    assert "real_one" in result.attempts[0].violation


def test_s10_removal_frees_cap_budget(tmp_path):
    """An add at the entry cap succeeds once a removal has shrunk the library."""
    incumbent = _incumbent_with_skills(*_ok_skill_names(SKILL_MAX_ENTRIES))
    removal = validate_candidate_spec(removal_item("skill_0"), ALL_PATTERNS)
    shrunk = materialize_candidate_harness(incumbent, removal, tmp_path)
    assert len(shrunk.skills) == SKILL_MAX_ENTRIES - 1
    add = validate_candidate_spec(SKILLS_ITEM, ALL_PATTERNS)
    harness = materialize_candidate_harness(shrunk, add, tmp_path)
    assert len(harness.skills) == SKILL_MAX_ENTRIES
    assert harness.skills[-1].name == SKILL_RECORD["name"]


def test_s10_removal_form_is_documented_in_the_prompt():
    rendered, _ = render_prompt(ALL_PATTERNS, serialize_harness(H0), (), (), k=4)
    assert "remove" in rendered.lower()


def test_s10_missing_field_names_the_missing_field():
    edit = {"kind": "skills", "name": "verify_aggregate", "description": "d"}
    with pytest.raises(ProposalRejection, match=r"missing.*body"):
        validate_candidate_spec(edit_item(5, edit), ALL_PATTERNS)
    edit = {"kind": "skills", "name": "verify_aggregate"}
    with pytest.raises(ProposalRejection, match=r"missing.*\['body', 'description'\]"):
        validate_candidate_spec(edit_item(5, edit), ALL_PATTERNS)


def test_s10_inventory_line_names_all_entries_past_the_truncation_point():
    """A library whose serialized value exceeds the 4000-char display truncation
    still gets a complete name inventory with per-entry char counts."""
    body = "1. Re-read.\n2. " + "y" * 1500
    incumbent = _incumbent_with_skills("alpha_skill", "beta_skill", "gamma_skill", body=body)
    serialization = serialize_harness(incumbent)
    value_text = json.dumps(
        {"S10_skills": serialization["surfaces"]["S10_skills"]}, indent=2, sort_keys=True
    )
    assert len(value_text) > 4000  # the display value really is truncated
    rendered, _ = render_prompt(ALL_PATTERNS, serialization, (), (), k=4)
    assert "S10 inventory:" in rendered
    assert f"3/{SKILL_MAX_ENTRIES} entries" in rendered
    for name in ("alpha_skill", "beta_skill", "gamma_skill"):
        assert f"{name} (" in rendered
    per_entry = len(body) + len("gamma_skill") + len("Consult for x.")
    assert f"{per_entry:,}" in rendered


def test_skills_pedagogy_only_rendered_when_an_s10_pattern_is_addressable():
    # PATTERN_TEXT (skipped_verification -> S4, S9, S10) and PATTERN_OTHER both
    # list S10 under 3.1.0; only lossy_aggregation's set (S9, S3, S4) does not.
    no_s10 = [make_pattern("lossy_aggregation")]
    rendered, _ = render_prompt(no_s10, serialize_harness(H0), (), (), k=4)
    assert "procedural anchor" not in rendered
    assert '"kind": "skills"' in rendered  # the compact format bullet stays

    with_s10 = [PATTERN_TEXT, PATTERN_SKILLS]
    rendered, _ = render_prompt(with_s10, serialize_harness(H0), (), (), k=4)
    assert rendered.count("procedural anchor") == 1
    assert rendered.count("## Use When") == 1


def test_render_prompt_names_the_verifier_contract_when_the_bundle_carries_it():
    contract = {
        "environment": "graphwalks",
        "extraction_rule": "trailing-bracket-list;marker-optional;quotes-stripped",
        "gold_ordering": "sorted",
        "pass_f1_threshold": 1.0,
    }
    rendered, _ = render_prompt(
        [PATTERN_TEXT], serialize_harness(H0), (), (), k=4, verifier_config=contract
    )
    assert "Verifier contract" in rendered
    assert "extraction_rule=trailing-bracket-list;marker-optional;quotes-stripped" in rendered
    absent, _ = render_prompt([PATTERN_TEXT], serialize_harness(H0), (), (), k=4)
    assert "not recorded for this bundle" in absent


def test_render_prompt_bounds_each_evidence_entry_separately():
    """A huge gold list in one entry must not hide another entry's produced string."""
    pattern = copy.deepcopy(PATTERN_TEXT)
    huge = "x" * 50_000
    pattern["verifier_evidence"] = [
        f"a: produced '[1f0e3dad99]', expected '[{huge}]'",
        "b: produced '[2c7f9ccb5a]', expected '[2c7f9ccb5a]'",
    ]
    rendered, _ = render_prompt([pattern], serialize_harness(H0), (), (), k=4)
    assert "a: produced '[1f0e3dad99]'" in rendered
    assert "b: produced '[2c7f9ccb5a]'" in rendered
    assert huge not in rendered


def test_duplicate_surfaces_reask_before_materialization(tmp_path):
    bundle = {"bundle_id": "same-surface", "patterns": [PATTERN_TEXT, PATTERN_TEXT]}
    duplicate = {**TEXT_ITEM, "pattern_index": 1}
    lm = MockLM(responses=[canned_batch(TEXT_ITEM, duplicate), canned_batch(TEXT_ITEM)])
    result = propose_round(bundle, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert len(result.attempts) == 2
    assert "surface S4" in result.attempts[0].violation
    assert len(result.written) == 1
    assert result.skipped_patterns == [1]


@pytest.mark.parametrize("failure", ["unchanged", "malformed", "duplicate"])
def test_repair_retargets_failed_pattern_and_preserves_independent_member(tmp_path, failure):
    patterns = [make_pattern("incomplete_coverage"), make_pattern("lossy_aggregation")]
    retained = edit_item(0, {"kind": "text", "new_text": "Check record IDs before mapping."})
    failed = edit_item(1, {"kind": "text", "new_text": H0.execution_instruction}, surface="S3")
    batch = [retained, failed]
    if failure == "malformed":
        del failed["behavioral_change"]
    elif failure == "duplicate":
        patterns.append(make_pattern("lossy_aggregation"))
        batch.append({**failed, "pattern_index": 2})
    repaired = edit_item(
        1,
        {"kind": "text", "new_text": "Recompute the predicate from joined records."},
        surface="S4",
        behavioral_change="Verify the predicate against the joined records.",
    )
    lm = MockLM(responses=[canned_batch(*batch), canned_batch(repaired)])
    work = tmp_path / "work"
    result = propose_round({"patterns": patterns}, H0, lm, tmp_path / "proposals", workdir=work)
    assert lm._call_count == 2
    assert [w.candidate_id for w in result.written] == ["r00-c01-s2", "r00-c02-s4"]
    assert (
        result.written[0].path.read_bytes()
        == (work / "attempt_01/proposals/r00-c01-s2/proposal.json").read_bytes()
    )
    assert not result.materialization_failures and not result.preflight_failures
    saved = [w.path.read_bytes() for w in result.written]
    replay = propose_round(
        {"patterns": patterns}, H0, MockLM(responses=[]), tmp_path / "proposals", workdir=work
    )
    assert [w.path.read_bytes() for w in replay.written] == saved


def test_prompt_explains_batch_surface_limit():
    prompt, _ = render_prompt([PATTERN_TEXT], serialize_harness(H0), [], [], 4)
    assert "at most one edit per surface" in prompt
    assert "one combined candidate" in prompt
    assert "not a quota" in prompt


def test_complete_current_surface_is_shared_and_contract_is_environment_specific():
    serialization = serialize_harness(H0)
    source = "x" * 1600 + "distinctive middleware tail"
    serialization["surfaces"]["S9_answer_middleware"] = source
    prompt, _ = render_prompt(
        [PATTERN_CODE_S9, PATTERN_CODE_S9],
        serialization,
        [],
        [],
        4,
        verifier_config={"environment": "oolong_pairs"},
    )
    assert prompt.count(source) == 1
    assert "AnswerDecision.accept(answer)" in prompt
    assert "No valid pairs found." in prompt
    assert "newline" in prompt
    assert "('str', 19006)" in prompt
    other, _ = render_prompt([PATTERN_CODE_S9], serialization, [], [], 4)
    assert "No valid pairs found." not in other


def test_history_reports_one_shared_verdict_for_bundled_edits():
    from shrlm.optimization.proposal import _render_history_block

    records = [
        {"subject_id": "a", "decision": "bundled", "surface": "S2"},
        {"subject_id": "b", "decision": "bundled", "surface": "S3"},
        {
            "subject_id": "merged",
            "decision": "rejected",
            "reasons": ["heldout did not improve"],
            "merge": {"role": "merged", "constituent_ids": ["a", "b"]},
        },
    ]
    history = _render_history_block([(records, {"promoted": False})])
    assert history.count("rejected") == 1
    assert "S2, S3" in history
    assert "bundled" not in history


# ---------------------------------------------------------------------------
# Cross-round history (R5-R7): every prior round renders, refused edits carry
# their direction and reason, and the proposer is told a no-op is refused.
# ---------------------------------------------------------------------------


def test_history_renders_not_materialized_records_with_effect_and_reason():
    from shrlm.optimization.proposal import HISTORY_NOT_MATERIALIZED, _render_history_block

    records = [
        {
            "subject_id": "pattern 0 on S9",
            "surface": "S9",
            "decision": HISTORY_NOT_MATERIALIZED,
            "reasons": ["declared surface S9 but the materialized harness changes no surface"],
            "predicted_effect": "redirect on an incomplete pair set",
        }
    ]
    history = _render_history_block([(records, {"round": 4, "promoted": False})])
    assert "Round 4:" in history
    [line] = [line for line in history.splitlines() if "pattern 0 on S9" in line]
    assert "not_materialized" in line
    assert "redirect on an incomplete pair set" in line
    assert "changes no surface" in line


def test_history_labels_rounds_by_position_when_the_decision_has_no_round():
    from shrlm.optimization.proposal import _render_history_block

    history = _render_history_block(
        [
            ([], {"promoted": False, "promoted_harness_hash": None}),
            ([], {"round": 7, "promoted": True, "promoted_harness_hash": "abc"}),
        ]
    )
    assert "Round 0:" in history and "Round 7:" in history


def test_history_renders_predicted_effect_on_ledger_records_and_tolerates_absence():
    from shrlm.optimization.proposal import _render_history_block

    records = [
        {
            "subject_id": "r01-c01-s4",
            "surface": "S4",
            "decision": "rejected",
            "reasons": ["heldout did not improve"],
            "predicted_effect": "the root verifies before answering",
        },
        {"subject_id": "baseline", "decision": "rejected", "reasons": []},
    ]
    history = _render_history_block([(records, {"round": 1, "promoted": False})])
    assert "the root verifies before answering" in history
    assert "baseline: rejected" in history


def test_history_renders_a_round_without_records_as_no_record_persisted():
    from shrlm.optimization.proposal import _render_history_block

    history = _render_history_block([([], {"round": 2, "promoted": False})])
    assert "Round 2:" in history
    assert "no per-edit record persisted" in history


def test_render_prompt_history_preamble_names_the_no_op_refusal():
    rendered, _ = render_prompt(ALL_PATTERNS, serialize_harness(H0), (), (), k=4)
    assert "identical to the current surface" in rendered


@pytest.mark.parametrize(
    "repair", ["not json", canned_batch(TEXT_ITEM), canned_batch(POLICY_ITEM, POLICY_ITEM)]
)
def test_invalid_repair_preserves_survivor_and_stops(tmp_path, repair):
    no_op = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(responses=[canned_batch(TEXT_ITEM, no_op), repair])
    result = propose_round(BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert len(result.attempts) == 2
    assert not result.attempts[-1].accepted
    assert [w.surface for w in result.written] == ["S4"]
    assert len(result.materialization_failures) == 1


def test_prompt_brace_error_repairs_locally(tmp_path):
    bad = edit_item(0, {"kind": "text", "new_text": 'Return {"record_id": 1}'})
    fixed = edit_item(0, {"kind": "text", "new_text": 'Return {{"record_id": 1}}'})
    lm = MockLM(responses=[canned_batch(bad), canned_batch(fixed)])
    result = propose_round(BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert len(result.attempts) == 2 and len(result.written) == 1
    assert "record_id" in result.attempts[0].violation


def test_changed_profile_refused_before_proposal_calls(tmp_path):
    lm = MockLM(responses=[canned_batch(TEXT_ITEM)])
    propose_round(BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    idle = MockLM(responses=[])
    with pytest.raises(ValueError, match="contract changed"):
        propose_round(
            BUNDLE,
            H0,
            idle,
            tmp_path / "proposals",
            workdir=tmp_path / "work",
            preflight_profile="oolong-pairs/v1",
        )
    assert idle._call_count == 0


@pytest.mark.parametrize("version", ["EVIDENCE_SELECTOR_VERSION", "DIAGNOSTIC_HISTORY_VERSION"])
def test_changed_evidence_contract_refuses_paid_replay(tmp_path, monkeypatch, version):
    import shrlm.optimization.proposal_evidence as evidence_module

    propose_round(
        BUNDLE, H0, MockLM(responses=["[]"]), tmp_path / "proposals", workdir=tmp_path / "work"
    )
    monkeypatch.setattr(evidence_module, version, "changed")
    idle = MockLM(responses=[])
    with pytest.raises(ValueError, match="contract changed"):
        propose_round(BUNDLE, H0, idle, tmp_path / "proposals", workdir=tmp_path / "work")
    assert idle._call_count == 0


@pytest.mark.parametrize(
    "surface,index,change",
    [
        ("S2", 1, "Use IDs"),
        ("S9", 1, "Inspect labels"),
        ("S4", 2, "Verify"),
        ("S4", 1, TEXT_ITEM["behavioral_change"]),
    ],
)
def test_repair_rejects_occupied_ineligible_unrelated_or_unexplained_target(
    tmp_path, surface, index, change
):
    patterns = [
        make_pattern("incomplete_coverage"),
        make_pattern("lossy_aggregation"),
        PATTERN_TEXT,
    ]
    keep = edit_item(0, {"kind": "text", "new_text": "Check record IDs."})
    failed = edit_item(1, {"kind": "text", "new_text": H0.execution_instruction}, surface="S3")
    repair = edit_item(
        index,
        {"kind": "text", "new_text": "Verify the predicate."},
        surface=surface,
        behavioral_change=change,
    )
    lm = MockLM(responses=[canned_batch(keep, failed), canned_batch(repair)])
    result = propose_round(
        {"patterns": patterns}, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work"
    )
    assert [w.surface for w in result.written] == ["S2"]
    assert not result.attempts[-1].accepted and lm._call_count == 2


def test_repair_output_budget_exhaustion_keeps_survivor_and_replays(tmp_path):
    from rlm.utils.exceptions import TokenLimitExceededError

    initial = canned_batch(TEXT_ITEM, edit_item(1, {"kind": "policy", "runtime_policy": {}}))
    calls = []

    def respond(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return initial
        raise TokenLimitExceededError(tokens_used=512, token_limit=512, message="repair exhausted")

    cache = ProposalCache(path=str(tmp_path / "cache.jsonl"))
    lm = MockLM(response_fn=respond)
    first = propose_round(
        BUNDLE, H0, lm, tmp_path / "proposals", cache=cache, workdir=tmp_path / "work"
    )
    assert len(calls) == 2
    assert [w.candidate_id for w in first.written] == ["r00-c01-s4"]
    assert "repair exhausted" in first.attempts[-1].violation
    before = first.written[0].path.read_bytes()
    idle = MockLM(responses=[])
    replay = propose_round(
        BUNDLE,
        H0,
        idle,
        tmp_path / "proposals",
        cache=ProposalCache(path=str(tmp_path / "cache.jsonl")),
        workdir=tmp_path / "work",
    )
    assert idle._call_count == 0
    assert replay.written[0].path.read_bytes() == before
    assert replay.materialization_failures == first.materialization_failures


def test_empty_repair_withdraws_failed_slot_without_replacing_survivor(tmp_path):
    no_op = edit_item(1, {"kind": "policy", "runtime_policy": {}})
    lm = MockLM(responses=[canned_batch(no_op, TEXT_ITEM), "[]"])
    result = propose_round(BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert len(result.attempts) == 2
    assert [w.candidate_id for w in result.written] == ["r00-c02-s4"]
    assert not result.materialization_failures
    assert not result.preflight_failures


def test_interrupted_publication_replays_frozen_survivors_without_gates(tmp_path, monkeypatch):
    import shrlm.optimization.proposal as proposal_module

    proposals = tmp_path / "proposals"
    work = tmp_path / "work"
    original_writer = proposal_module.write_proposal

    def interrupted_writer(directory, candidate_id, *args, **kwargs):
        if directory == proposals and candidate_id == "r00-c02-s9":
            raise OSError("interrupted publication")
        return original_writer(directory, candidate_id, *args, **kwargs)

    monkeypatch.setattr(proposal_module, "write_proposal", interrupted_writer)
    lm = MockLM(responses=[canned_batch(TEXT_ITEM, CODE_S9_ITEM)])
    with pytest.raises(OSError, match="interrupted publication"):
        propose_round(BUNDLE, H0, lm, proposals, workdir=work)
    assert lm._call_count == 1
    assert (work / "proposal_result.json").exists()
    first = proposals / "r00-c01-s4" / "proposal.json"
    before = first.read_bytes()
    monkeypatch.setattr(proposal_module, "write_proposal", original_writer)

    def forbidden_gate(*args, **kwargs):
        raise AssertionError("a frozen proposal outcome must not rerun preflight")

    monkeypatch.setattr(proposal_module, "load_candidate", forbidden_gate)
    idle = MockLM(responses=[])
    replay = propose_round(BUNDLE, H0, idle, proposals, workdir=work)
    assert idle._call_count == 0
    assert [w.candidate_id for w in replay.written] == ["r00-c01-s4", "r00-c02-s9"]
    assert first.read_bytes() == before
    assert {p.name for p in proposals.iterdir()} == {w.candidate_id for w in replay.written}
    (proposals / "unrecorded-candidate").mkdir()
    with pytest.raises(ValueError, match="outside the frozen survivor set"):
        propose_round(BUNDLE, H0, idle, proposals, workdir=work)


def test_proposal_gate_spawn_failure_does_not_spend_repair(tmp_path, monkeypatch):
    import shrlm.optimization.candidates as candidates_module

    def cannot_spawn(*args, **kwargs):
        raise OSError("process resources exhausted")

    monkeypatch.setattr(candidates_module.subprocess, "run", cannot_spawn)
    lm = MockLM(responses=[canned_batch(TEXT_ITEM)])
    with pytest.raises(OSError, match="process resources exhausted"):
        propose_round(BUNDLE, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work")
    assert lm._call_count == 1
    assert not list((tmp_path / "proposals").iterdir())
    assert not (tmp_path / "work" / "proposal_result.json").exists()


def test_partial_final_proposal_write_replays_from_checkpoint(tmp_path, monkeypatch):
    proposals = tmp_path / "proposals"
    work = tmp_path / "work"
    original_write = Path.write_text

    def partial_write(path, data, *args, **kwargs):
        if path.parent.parent == proposals:
            original_write(path, data[:20], *args, **kwargs)
            raise OSError("disk write interrupted")
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial_write)
    with pytest.raises(OSError, match="disk write interrupted"):
        propose_round(
            BUNDLE, H0, MockLM(responses=[canned_batch(TEXT_ITEM)]), proposals, workdir=work
        )
    assert not list(proposals.glob("*/proposal.json"))
    monkeypatch.setattr(Path, "write_text", original_write)
    idle = MockLM(responses=[])
    replay = propose_round(BUNDLE, H0, idle, proposals, workdir=work)
    assert idle._call_count == 0
    assert [w.candidate_id for w in replay.written] == ["r00-c01-s4"]
    assert json.loads(replay.written[0].path.read_text())["surface"] == "S4"
