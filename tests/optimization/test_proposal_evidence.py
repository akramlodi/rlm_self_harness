"""Diagnostics use saved outcomes and keep held-out payloads out of proposals."""

import hashlib
import json
from dataclasses import replace

import pytest

import rlm.core.rlm as rlm_module
from shrlm.environments.oolong_pairs import OolongPairsVerifier, recorded_pair_metrics
from shrlm.harness_identity import serialize_harness
from shrlm.optimization.driver import RoundPersistenceError, load_round, run_round
from shrlm.optimization.proposal import render_prompt
from shrlm.optimization.proposal_evidence import (
    aggregate_pair_diagnostics,
    load_proposal_evidence,
    pair_diagnostics,
    structural_execution_error,
    validation_history_diagnostics,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0
from tests.optimization.test_driver import ClientFactory, final, make_round_config
from tests.optimization.test_proposal import PATTERN_CODE_S9, PATTERN_TEXT


def test_saved_metrics_and_all_attempt_denominators():
    verifier = OolongPairsVerifier()
    near = verifier({"gold_pairs": [(1, 2), (1, 3)]}, "[(1, 2)]")
    empty = verifier({"gold_pairs": []}, "No valid pairs found.")
    invalid = verifier({"gold_pairs": [(1, 2)]}, "[]")
    runtime = Verdict(False, VerifierCause.RUNTIME_ERROR, "[]", "(1, 2)", "crashed")
    assert recorded_pair_metrics(near) == dict(
        precision=1.0, recall=0.5, f1=0.667, missing=1, extra=0
    )
    assert pair_diagnostics(runtime)["metrics"] == "unavailable"
    metrics = aggregate_pair_diagnostics([near, empty, invalid, runtime])
    assert metrics["mean_f1_all_attempts"] == pytest.approx(1.667 / 4)
    assert metrics["counts_denominator"] == 2
    assert metrics["missing_total_measured"] == 1
    assert metrics["extra_total_measured"] == 0
    assert metrics["n_unscored"] == 2
    legacy = replace(near, detail="")
    assert aggregate_pair_diagnostics([legacy])["mean_f1_all_attempts"] is None
    assert aggregate_pair_diagnostics([])["mean_f1_all_attempts"] is None
    assert aggregate_pair_diagnostics([runtime])["missing_total_measured"] is None
    filtered = replace(runtime, cause=VerifierCause.CONTENT_FILTERED)
    aggregate = aggregate_pair_diagnostics([near, filtered])
    assert aggregate["mean_f1_all_attempts"] == 0.667 / 2
    assert aggregate["n_missing_legacy_metrics"] == 0
    assert (
        aggregate_pair_diagnostics([replace(runtime, cause=VerifierCause.OTHER)])[
            "mean_f1_all_attempts"
        ]
        is None
    )


def mining_fixture(tmp_path, monkeypatch, *, attempts=2):
    instance = {
        "id": "held-in",
        "question": "ACTUAL PREDICATE and LABEL VOCABULARY",
        "prompt": "classify rows",
        "gold_pairs": [(1, 2), (1, 3)],
    }
    factory = ClientFactory([final("[(1, 2)]")] * attempts)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_round_config(
        tmp_path, instances=[instance], attempts=attempts, verifier=OolongPairsVerifier()
    )
    run_round(config)
    runs, verdicts, _, entries = load_round(tmp_path, 1)
    path = tmp_path / "round_01"
    pattern = dict(PATTERN_TEXT, representatives=["held-in"])
    bundle = {
        "bundle_id": "synthetic",
        "patterns": [pattern],
        "config": {"verifier_config": OolongPairsVerifier().config()},
    }
    records = [
        {
            "instance_id": "held-in",
            "signature": pattern["signature"],
            "verdict": verdict.to_dict(),
            "run_id": entry["run_id"],
            "trace_path": entry["trace_path"],
            "trace_sha256": entry["trace_sha256"],
            "detail": {"symptom_summary": f"diagnosis {i}", "evidence_node_ids": ["r"]},
        }
        for i, (verdict, entry) in enumerate(zip(verdicts, entries, strict=True))
    ]
    (path / "bundle.json").write_text(json.dumps(bundle))
    (path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in reversed(records)))
    return path, bundle, records


def test_evidence_joins_exact_attempt_and_does_not_rewrite_artifacts(tmp_path, monkeypatch):
    path, bundle, records = mining_fixture(tmp_path, monkeypatch)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}
    evidence = load_proposal_evidence(path, bundle)
    context = evidence["patterns"][0]
    assert context["symptom_summary"] == "diagnosis 1"
    assert context["missing_examples"] == [(1, 3)]
    assert context["pair_diagnostics"]["metrics"]["f1"] == 0.667
    assert context["task_question"] == "ACTUAL PREDICATE and LABEL VOCABULARY"
    assert "root_code" in context["trace"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    prompt, _ = render_prompt(
        bundle["patterns"],
        serialize_harness(H0),
        [],
        [],
        4,
        verifier_config=OolongPairsVerifier().config(),
        evidence=evidence,
    )
    assert "diagnosis 1" in prompt and "ACTUAL PREDICATE" in prompt
    assert "original row ordinal" in prompt
    assert "never response position" in prompt
    assert evidence["passing"] == []
    record = dict(records[0], run_id=None, trace_path=None, trace_sha256=None)
    (path / "records.jsonl").write_text(json.dumps(record))
    ambiguous = load_proposal_evidence(path, bundle)["patterns"][0]
    assert "ambiguous" in ambiguous["trace"]["observation"]
    assert ambiguous["symptom_summary"] == "diagnosis 0"


def test_trace_integrity_failure_is_not_an_optional_evidence_fallback(tmp_path, monkeypatch):
    path, bundle, records = mining_fixture(tmp_path, monkeypatch, attempts=1)
    (path / records[0]["trace_path"]).write_text("{}")
    with pytest.raises(RoundPersistenceError, match="sha|hash"):
        load_proposal_evidence(path, bundle)


def test_history_keeps_heldout_payloads_out_and_preserves_saved_bytes(tmp_path, monkeypatch):
    path, _, _ = mining_fixture(tmp_path / "merged" / "heldout", monkeypatch, attempts=1)
    contract_path = tmp_path / "merged" / "evaluation.json"
    contract_path.write_text(json.dumps({"verifier_config": OolongPairsVerifier().config()}))
    record = {
        "subject_id": "merged",
        "links": {"splits": {"heldout": {"round_dir": str(path.relative_to(tmp_path))}}},
    }
    before = (path / "runs.jsonl").read_bytes()
    diagnostics = validation_history_diagnostics(tmp_path, record)
    text = json.dumps(diagnostics)
    assert "ACTUAL PREDICATE" not in text
    assert "held-in" not in text
    assert "(1, 3)" not in text
    assert diagnostics["pairs"]["mean_f1_all_attempts"] == 0.667
    assert before == (path / "runs.jsonl").read_bytes()
    for trace in (path / "runs").glob("*.json"):
        trace.unlink()
    assert validation_history_diagnostics(tmp_path, record) == diagnostics
    # The same text format is also emitted by GraphWalks. Never infer a domain
    # from numeric fields, including when old evaluation contracts are absent.
    contract_path.write_text(json.dumps({"verifier_config": {"environment": "graphwalks"}}))
    assert "pairs" not in validation_history_diagnostics(tmp_path, record)
    contract_path.unlink()
    assert "pairs" not in validation_history_diagnostics(tmp_path, record)


def test_evidence_rejects_cross_instance_run_links(tmp_path, monkeypatch):
    path, bundle, records = mining_fixture(tmp_path, monkeypatch, attempts=1)
    bundle["patterns"][0]["representatives"] = ["another-instance"]
    (path / "bundle.json").write_text(json.dumps(bundle))
    records[0]["instance_id"] = "another-instance"
    (path / "records.jsonl").write_text(json.dumps(records[0]))
    with pytest.raises(ValueError, match="instance_id disagrees"):
        load_proposal_evidence(path, bundle)


def test_evidence_does_not_hide_missing_required_harness(tmp_path, monkeypatch):
    path, bundle, _ = mining_fixture(tmp_path, monkeypatch, attempts=1)
    (path / "harness.json").unlink()
    with pytest.raises(FileNotFoundError, match="harness.json"):
        load_proposal_evidence(path, bundle)


def test_runtime_error_messages_are_structural_only():
    message = "AnswerDecision.accept() missing 1 required positional argument: 'answer'"
    verdict = Verdict(
        False,
        VerifierCause.RUNTIME_ERROR,
        "HELDOUT_GOLD",
        "HELDOUT_ANSWER",
        f"TypeError: {message}",
    )
    assert structural_execution_error(verdict)["message"] == message
    verdict = replace(verdict, detail=verdict.detail + " HELDOUT_SECRET (101, 203)")
    result = structural_execution_error(verdict)
    assert "HELDOUT_SECRET" not in json.dumps(result)
    assert "omitted" in result["message"]


def test_record_recipe_is_conditional_on_environment_and_eligible_surface(monkeypatch):
    from shrlm.optimization.proposal import MECHANISM_SURFACES
    from shrlm.optimization.taxonomy import AgentMechanism, EditableSurface

    monkeypatch.setitem(
        MECHANISM_SURFACES, AgentMechanism.LOSSY_AGGREGATION, (EditableSurface.ANSWER_MIDDLEWARE,)
    )
    prompt, _ = render_prompt(
        [PATTERN_CODE_S9],
        serialize_harness(H0),
        [],
        [],
        4,
        verifier_config=OolongPairsVerifier().config(),
    )
    assert "original row ordinal" not in prompt
    prompt, _ = render_prompt([PATTERN_TEXT], serialize_harness(H0), [], [], 4)
    assert "original row ordinal" not in prompt
