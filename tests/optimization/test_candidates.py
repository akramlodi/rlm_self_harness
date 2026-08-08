"""Tests for the stage-2 candidate contract loader (``shrlm-proposal/v1``).

The loader is the stage boundary: everything stage 2 hands over arrives as one
``proposal.json`` whose harness payload is a full ``shrlm-harness/v1`` envelope,
and every gate failure must come back as a structured ``CandidateRejection``,
never an exception. The gates run in KTD2's order: text-level first (schema,
envelope hash recompute, base-hash match, one-surface diff, cap comparison),
none of which execute candidate code; only then materialization plus
``check_harness`` inside a subprocess under a wall-clock timeout. A successful
load hands back a live ``Harness`` whose unchanged surfaces are the incumbent's
own objects and whose serialization is byte-identical to the envelope's.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import shrlm.optimization.candidates as candidates_module
from rlm.core.types import AnswerDecision
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH, default_metadata_builder
from shrlm.harness_identity import harness_hash, hash_of_serialization, serialize_harness
from shrlm.optimization.candidates import (
    MODULE_FILENAME,
    SURFACE_SERIALIZATION_KEYS,
    CandidateRejection,
    LoadedCandidate,
    load_candidate,
    load_candidates,
)
from shrlm.optimization.driver import RoundConfig, _prepare_round_dir
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0, SURFACES, build_runtime_policy

# ---------------------------------------------------------------------------
# Candidate surface values. Top-level so ``inspect.getsource`` can serialize
# them, exactly as a stage-2 proposer's envelope would carry their source.
# ---------------------------------------------------------------------------


def reject_empty_answer(answer, repl_inventory):
    """S9 candidate: redirect empty answers instead of accepting them."""
    if not answer.strip():
        return AnswerDecision.redirect("The answer was empty; aggregate your results first.")
    return AnswerDecision.accept(answer)


def exploding_middleware(answer, repl_inventory):
    """S9 candidate whose probe call raises."""
    raise RuntimeError("boom at probe time")


def hanging_middleware(answer, repl_inventory):
    """S9 candidate whose probe call never returns."""
    while True:
        pass


def unbounded_metadata(stdout, repl_inventory):
    """S7 candidate violating I1: output grows with the REPL state."""
    return stdout


unbounded_metadata.declared_bound = 20_000


def tail_metadata(stdout, repl_inventory):
    """S7 candidate: the shipped truncation, re-expressed under the same bound."""
    return default_metadata_builder(stdout, repl_inventory, DEFAULT_MAX_CHARACTER_LENGTH)


tail_metadata.declared_bound = DEFAULT_MAX_CHARACTER_LENGTH


def count_lines(text):
    """S8 candidate helper."""
    return len(text.splitlines())


def s2_candidate():
    return replace(
        H0,
        name="cand-s2",
        decomposition_instruction=(
            "Probe `context`, list the sub-tasks explicitly, then split by sub-task."
        ),
    )


def s6_candidate(max_depth=2):
    return replace(
        H0,
        name="cand-s6",
        runtime_policy={**build_runtime_policy(), "enabled": True, "max_depth": max_depth},
    )


def s7_candidate(builder=tail_metadata):
    return replace(H0, name="cand-s7", metadata=builder)


def s8_candidate():
    return replace(
        H0,
        name="cand-s8",
        repl_helpers={"count_lines": count_lines},
        sub_repl_helpers={"count_lines": count_lines},
    )


def s9_candidate(middleware=reject_empty_answer):
    return replace(H0, name="cand-s9", answer_middleware=middleware)


# ---------------------------------------------------------------------------
# Proposal builders
# ---------------------------------------------------------------------------

TARGET_SIGNATURE = {
    "verifier_cause": "wrong_value",
    "failing_level": "root",
    "causal_status": "causal",
    "agent_mechanism": "lossy_aggregation",
}


def envelope_for(harness) -> dict[str, Any]:
    serialization = serialize_harness(harness)
    return {
        "format": "shrlm-harness/v1",
        "name": serialization["name"],
        "hash": hash_of_serialization(serialization),
        "harness": serialization,
    }


def proposal_payload(harness, surface, candidate_id="cand-01", **overrides) -> dict[str, Any]:
    payload = {
        "format": "shrlm-proposal/v1",
        "candidate_id": candidate_id,
        "base_harness_hash": harness_hash(H0),
        "target_signature": dict(TARGET_SIGNATURE),
        "surface": surface,
        "harness": envelope_for(harness),
        "predicted_effect": "The root verifies accumulated results before committing.",
        "regression_risks": ["May cost one extra turn per run."],
        "provenance": {"model": "mock-model", "prompt_sha256": "a" * 64},
    }
    payload.update(overrides)
    return payload


def write_payload(tmp_path: Path, payload: dict[str, Any], candidate_id: str | None = None) -> Path:
    candidate_id = candidate_id or str(payload.get("candidate_id", "cand-01"))
    directory = tmp_path / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "proposal.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def passing_verifier(instance, produced):
    return Verdict(passed=True, cause=None, gold="", produced=produced)


# ---------------------------------------------------------------------------
# The surface grouping itself
# ---------------------------------------------------------------------------


def test_surface_groups_cover_the_serialization_exactly():
    grouped = {key for keys in SURFACE_SERIALIZATION_KEYS.values() for key in keys}
    assert grouped == set(serialize_harness(H0)["surfaces"])
    assert set(SURFACE_SERIALIZATION_KEYS) == set(SURFACES)


# ---------------------------------------------------------------------------
# Round trips: envelope in, byte-identical serialization out
# ---------------------------------------------------------------------------


def test_string_surface_round_trip(tmp_path):
    payload = proposal_payload(s2_candidate(), "S2")
    loaded = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    assert loaded.surface == "S2"
    assert canonical(serialize_harness(loaded.harness)) == canonical(payload["harness"]["harness"])
    # KTD2: unchanged surfaces reuse the incumbent's live objects.
    assert loaded.harness.metadata is H0.metadata
    assert loaded.harness.answer_middleware is H0.answer_middleware


def test_policy_surface_round_trip(tmp_path):
    payload = proposal_payload(s6_candidate(max_depth=2), "S6")
    loaded = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    assert loaded.surface == "S6"
    assert canonical(serialize_harness(loaded.harness)) == canonical(payload["harness"]["harness"])
    assert loaded.harness.runtime_policy["max_depth"] == 2
    assert loaded.harness.metadata is H0.metadata


def test_callable_surface_round_trip(tmp_path):
    payload = proposal_payload(s9_candidate(), "S9")
    loaded = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    assert loaded.surface == "S9"
    assert canonical(serialize_harness(loaded.harness)) == canonical(payload["harness"]["harness"])
    assert loaded.harness_hash == payload["harness"]["hash"]
    assert harness_hash(loaded.harness) == payload["harness"]["hash"]
    assert loaded.harness.answer_middleware is not H0.answer_middleware
    assert (loaded.path.parent / MODULE_FILENAME).exists()


def test_metadata_surface_round_trip_carries_declared_bound(tmp_path):
    payload = proposal_payload(s7_candidate(), "S7")
    loaded = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    assert canonical(serialize_harness(loaded.harness)) == canonical(payload["harness"]["harness"])
    assert loaded.harness.metadata.declared_bound == DEFAULT_MAX_CHARACTER_LENGTH


def test_helper_surface_round_trip_yields_working_helpers(tmp_path):
    payload = proposal_payload(s8_candidate(), "S8")
    loaded = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    assert loaded.surface == "S8"
    assert canonical(serialize_harness(loaded.harness)) == canonical(payload["harness"]["harness"])
    assert loaded.harness.repl_helpers["count_lines"]("a\nb") == 2


# ---------------------------------------------------------------------------
# Text-level gates
# ---------------------------------------------------------------------------


def test_zero_surface_diff_rejected(tmp_path):
    payload = proposal_payload(replace(H0, name="cand-noop"), "S2", candidate_id="cand-noop")
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "surface_diff"
    assert "modifies no surface" in result.reason


def test_two_surface_diff_rejected_naming_both(tmp_path):
    two = replace(
        H0,
        name="cand-two",
        decomposition_instruction="Split it.",
        execution_instruction="Run it.",
    )
    result = load_candidate(write_payload(tmp_path, proposal_payload(two, "S2")), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "surface_diff"
    assert "S2" in result.reason and "S3" in result.reason


def test_declared_surface_must_match_the_modified_one(tmp_path):
    edited = replace(H0, name="cand-s3", execution_instruction="Run one block per turn.")
    result = load_candidate(write_payload(tmp_path, proposal_payload(edited, "S2")), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "surface_diff"
    assert "S2" in result.reason and "S3" in result.reason


def test_orchestrator_flip_is_not_an_editable_surface(tmp_path):
    flipped = replace(H0, name="cand-orch", orchestrator=True)
    result = load_candidate(write_payload(tmp_path, proposal_payload(flipped, "S2")), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "surface_diff"
    assert "orchestrator" in result.reason


def test_tampered_envelope_rejected(tmp_path):
    payload = proposal_payload(s2_candidate(), "S2")
    payload["harness"]["harness"]["surfaces"]["S2_decomposition_instruction"] = "tampered"
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "envelope_hash"


def test_wrong_base_hash_rejected(tmp_path):
    payload = proposal_payload(s2_candidate(), "S2", base_harness_hash="0" * 64)
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "base_hash"
    assert harness_hash(H0) in result.reason


def test_policy_above_caps_rejected_below_accepted(tmp_path):
    over = proposal_payload(s6_candidate(max_depth=5), "S6", candidate_id="cand-over")
    result = load_candidate(write_payload(tmp_path, over), H0, caps={"max_depth": 2})
    assert isinstance(result, CandidateRejection)
    assert result.gate == "caps"
    assert "max_depth" in result.reason

    under = proposal_payload(s6_candidate(max_depth=2), "S6", candidate_id="cand-under")
    loaded = load_candidate(write_payload(tmp_path, under), H0, caps={"max_depth": 4})
    assert isinstance(loaded, LoadedCandidate), loaded


def test_text_gates_execute_no_candidate_code(tmp_path, monkeypatch):
    payload = proposal_payload(s9_candidate(), "S9")
    payload["harness"]["hash"] = "0" * 64  # tampered

    def forbid_subprocess(*args, **kwargs):
        raise AssertionError("the gate subprocess must not run for a text-level rejection")

    monkeypatch.setattr(candidates_module, "_run_gate_subprocess", forbid_subprocess)
    path = write_payload(tmp_path, payload)
    result = load_candidate(path, H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "envelope_hash"
    assert not (path.parent / MODULE_FILENAME).exists()


# ---------------------------------------------------------------------------
# Schema gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "expected_fragment"),
    [
        ({"format": "shrlm-proposal/v2"}, "format"),
        ({"surface": "S10"}, "surface"),
        ({"predicted_effect": ""}, "predicted_effect"),
        (
            {"target_signature": {**TARGET_SIGNATURE, "verifier_cause": "not_a_cause"}},
            "verifier_cause",
        ),
        ({"regression_risks": "oops"}, "regression_risks"),
        ({"provenance": {"model": "mock-model"}}, "prompt_sha256"),
    ],
)
def test_schema_gate_names_the_violation(tmp_path, override, expected_fragment):
    payload = proposal_payload(s2_candidate(), "S2")
    payload.update(override)
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "schema"
    assert expected_fragment in result.reason


def test_unparseable_proposal_is_a_schema_rejection(tmp_path):
    directory = tmp_path / "cand-bad"
    directory.mkdir()
    path = directory / "proposal.json"
    path.write_text("{not json")
    result = load_candidate(path, H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "schema"
    assert result.candidate_id == "cand-bad"


# ---------------------------------------------------------------------------
# The subprocess boundary: materialization, check_harness, timeout
# ---------------------------------------------------------------------------


def test_source_raising_at_probe_is_a_structured_rejection(tmp_path):
    payload = proposal_payload(s9_candidate(exploding_middleware), "S9")
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "harness_check"
    assert "boom at probe time" in result.reason


def test_source_failing_at_import_is_a_structured_rejection(tmp_path):
    payload = proposal_payload(s9_candidate(), "S9")
    serialization = payload["harness"]["harness"]
    serialization["surfaces"]["S9_answer_middleware"]["source"] = (
        "def broken_middleware(answer, repl_inventory, _marker=UNDEFINED_NAME):\n"
        "    return AnswerDecision.accept(answer)\n"
    )
    payload["harness"]["hash"] = hash_of_serialization(serialization)
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "materialization"
    assert "UNDEFINED_NAME" in result.reason


def test_hanging_candidate_source_times_out(tmp_path):
    payload = proposal_payload(s9_candidate(hanging_middleware), "S9")
    result = load_candidate(write_payload(tmp_path, payload), H0, timeout_seconds=5.0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "materialization"
    assert "timed out" in result.reason


def test_check_harness_failure_carries_the_runner_message(tmp_path):
    payload = proposal_payload(s7_candidate(unbounded_metadata), "S7")
    result = load_candidate(write_payload(tmp_path, payload), H0)
    assert isinstance(result, CandidateRejection)
    assert result.gate == "harness_check"
    assert "violates I1" in result.reason


# ---------------------------------------------------------------------------
# Downstream compatibility: the evaluation driver's harness-hash step
# ---------------------------------------------------------------------------


def test_materialized_candidate_passes_the_driver_hash_step(tmp_path):
    payload = proposal_payload(s9_candidate(), "S9")
    loaded = load_candidate(write_payload(tmp_path / "proposals", payload), H0)
    assert isinstance(loaded, LoadedCandidate), loaded
    config = RoundConfig(
        round_index=0,
        harness=loaded.harness,
        instances=[{"id": "i1", "prompt": "p"}],
        verifier=passing_verifier,
        out_dir=tmp_path / "rounds",
    )
    round_path = _prepare_round_dir(config)
    envelope = json.loads((round_path / "harness.json").read_text())
    assert envelope["hash"] == loaded.harness_hash


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


def test_load_candidates_partitions_and_enforces_directory_names(tmp_path):
    write_payload(tmp_path, proposal_payload(s2_candidate(), "S2", candidate_id="cand-good"))
    tampered = proposal_payload(s6_candidate(), "S6", candidate_id="cand-tampered")
    tampered["harness"]["hash"] = "f" * 64
    write_payload(tmp_path, tampered)
    mismatch = proposal_payload(s9_candidate(), "S9", candidate_id="cand-elsewhere")
    write_payload(tmp_path, mismatch, candidate_id="cand-dir")
    (tmp_path / "cand-empty").mkdir()

    loaded, rejections = load_candidates(tmp_path, H0)
    assert [candidate.candidate_id for candidate in loaded] == ["cand-good"]
    gates = {rejection.candidate_id: rejection.gate for rejection in rejections}
    assert gates["cand-tampered"] == "envelope_hash"
    assert gates["cand-dir"] == "schema"
    assert gates["cand-empty"] == "schema"


def test_load_candidates_on_an_empty_directory(tmp_path):
    loaded, rejections = load_candidates(tmp_path, H0)
    assert loaded == []
    assert rejections == []
