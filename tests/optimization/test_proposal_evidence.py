"""Diagnostics use saved outcomes and keep held-out payloads out of proposals."""

import copy
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
    aggregate_quality_diagnostics,
    compare_quality_diagnostics,
    load_proposal_evidence,
    pair_diagnostics,
    structural_execution_error,
    trace_excerpt,
    validation_history_diagnostics,
    validation_history_progress,
)
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0
from tests.optimization.fixtures import as_completion, code_block, completion_dict, iteration_entry
from tests.optimization.test_driver import ClientFactory, final, make_round_config
from tests.optimization.test_proposal import PATTERN_CODE_S9, PATTERN_TEXT


def test_core_packets_precede_optional_context_and_repeat_support():
    from shrlm.optimization.proposal_evidence import pack_evidence

    inventory = [
        {
            "index": i,
            "signature": {"agent_mechanism": mechanism},
            "support": 100 if i == 0 else 2,
            "instance_support": i + 1,
        }
        for i, mechanism in enumerate(
            ["incomplete_coverage", "lossy_aggregation", "repl_execution_fault"]
        )
    ]
    evidence = {
        "patterns": {
            i: {
                "task_question": "derive the result",
                "trace": {
                    "snippets": [
                        {
                            "node_id": "root",
                            "iteration_index": 0,
                            "code_block_index": 0,
                            "code": f"combine_{i}(inputs)",
                            "code_complete": True,
                            "reason": "cited operation",
                        },
                        {
                            "node_id": "root",
                            "iteration_index": 1,
                            "code_block_index": 0,
                            "code": "optional = '" + "x" * 5000 + "'",
                            "code_complete": True,
                            "reason": "following consumer context",
                        },
                    ]
                },
            }
            for i in range(3)
        }
    }
    rendered, audit = pack_evidence(inventory, evidence, k=3, budget=5000)
    assert audit["expanded_patterns"] == [2, 1, 0]
    assert len(rendered) <= 5000
    section = json.loads(rendered.split("\n", 1)[1])
    assert len(section["operations"]) == 3
    assert section["inventory"][1]["eligible_surfaces"] == ["S3", "S4", "S8"]
    assert audit["route_support"]["1"]["S8"]


@pytest.mark.parametrize("mechanism", ["iteration_budget_exhaustion", "repl_execution_fault"])
def test_recovery_route_requires_admitted_error_and_subsequent_complete_operation(mechanism):
    from shrlm.optimization.proposal_evidence import pack_evidence

    inventory = [{"index": 0, "signature": {"agent_mechanism": mechanism}}]
    snippets = [
        {
            "node_id": "r",
            "iteration_index": 0,
            "code_block_index": 0,
            "code": "child_result = child()",
            "code_complete": True,
            "reason": "cited operation",
            "error_observed": True,
        },
        {
            "node_id": "r",
            "iteration_index": 1,
            "code_block_index": 0,
            "code": "consume(child_result)",
            "code_complete": True,
            "reason": "following consumer context",
            "follows_error": True,
        },
    ]
    evidence = {"patterns": {0: {"task_question": "task", "trace": {"snippets": snippets}}}}
    _, audit = pack_evidence(inventory, evidence, k=1, budget=4000)
    assert len(audit["route_support"]["0"]["S5"]) == 2
    # If the required consumer cannot fit, the host cannot admit the route.
    snippets[1]["code"] = "oversized" * 4000
    _, audit = pack_evidence(inventory, evidence, k=1, budget=4000)
    assert "S5" not in audit["route_support"]["0"]
    snippets.clear()
    _, audit = pack_evidence(inventory, evidence, k=1, budget=4000)
    assert "S5" not in audit["route_support"]["0"]


def test_heldin_evidence_uses_custom_verifier_definition(tmp_path):
    from shrlm.optimization.types import QualityDefinition, QualityMeasurement

    definition = QualityDefinition("distance", "v1", "lower")
    verdict = Verdict(
        False,
        VerifierCause.WRONG_VALUE,
        "",
        "",
        quality=QualityMeasurement(definition.identifier, 3.0),
    )
    signature = {"agent_mechanism": "lossy_aggregation"}
    bundle = {
        "config": {
            "verifier_config": {"environment": "custom", "primary_quality": definition.to_dict()}
        },
        "patterns": [{"signature": signature, "representatives": ["a"]}],
    }
    (tmp_path / "records.jsonl").write_text(
        json.dumps({"instance_id": "a", "signature": signature, "verdict": verdict.to_dict()})
        + "\n"
    )
    evidence = load_proposal_evidence(tmp_path, bundle)
    assert evidence["patterns"][0]["quality_diagnostics"]["mean"] == 3.0


@pytest.mark.parametrize(
    "environment,detail,name",
    [
        ("oolong_pairs", "precision=0.800 recall=0.800 f1=0.800 missing=2 extra=1", "f1"),
        ("graphwalks", "precision=0.800 recall=0.800 f1=0.800 missing=2 extra=1", "f1"),
        ("oolong", "score=0.800 exact=False kind=numeric", "score"),
    ],
)
def test_quality_uses_trusted_verifier_detail_with_all_attempt_denominator(
    environment, detail, name
):
    scored = Verdict(False, VerifierCause.WRONG_VALUE, gold="", produced="", detail=detail)
    failed = Verdict(False, VerifierCause.WRONG_FORMAT, gold="", produced="", detail="CANARY 0.99")
    quality = aggregate_quality_diagnostics([scored, failed], environment)
    assert quality["definition"]["name"] == name
    assert quality["mean"] == 0.4
    assert compare_quality_diagnostics({**quality, "mean": 0.3}, quality) == "potentially_promising"
    assert quality["n_measured"] == 1 and quality["n_known_zero"] == 1
    legacy = replace(scored, detail="legacy 0.99")
    assert aggregate_quality_diagnostics([scored, legacy], environment)["mean"] is None
    assert aggregate_quality_diagnostics([failed], environment)["mean"] == 0


def test_explicit_empty_oolong_is_zero_but_unknown_detail_is_not_guessed():
    empty = Verdict(
        False,
        VerifierCause.NO_ANSWER,
        gold="",
        produced="",
        detail="final line carried an explicit empty marker",
    )
    assert aggregate_quality_diagnostics([empty], "oolong")["mean"] == 0
    assert (
        aggregate_quality_diagnostics([replace(empty, detail="old empty")], "oolong")["mean"]
        is None
    )
    assert aggregate_quality_diagnostics([empty], "unsupported")["mean"] is None
    for detail in (
        "score=nan exact=False kind=numeric",
        "score=1.100 exact=False kind=numeric",
        "score=0.900 exact=False kind=CANARY",
        "score=0.900 exact=False kind=numeric CANARY",
    ):
        assert (
            aggregate_quality_diagnostics([replace(empty, detail=detail)], "oolong")["mean"] is None
        )


@pytest.mark.parametrize("task_set", ["synth", "real"])
def test_actual_oolong_contracts_produce_structured_quality_and_support_legacy(task_set):
    from shrlm.environments.oolong import OolongVerifier

    verifier = OolongVerifier(task_set)
    verdict = verifier({"answer_kind": "numeric", "answer_raw": "4"}, "4")
    config = verifier.config()
    assert aggregate_quality_diagnostics([verdict], config)["mean"] == 1.0
    legacy = {key: value for key, value in config.items() if key != "primary_quality"}
    assert aggregate_quality_diagnostics([replace(verdict, quality=None)], legacy)["mean"] == 1.0
    assert aggregate_quality_diagnostics([replace(verdict, quality=None)], config)["mean"] is None


def test_progress_obeys_declared_direction_and_requires_matching_definition():
    baseline = {"definition": {"name": "error", "direction": "lower"}, "mean": 0.8}
    candidate = {**baseline, "mean": 0.6}
    assert compare_quality_diagnostics(baseline, candidate) == "potentially_promising"
    assert compare_quality_diagnostics(candidate, baseline) == "no_measured_improvement"
    assert compare_quality_diagnostics(candidate, candidate) == "no_measured_improvement"
    assert compare_quality_diagnostics(baseline, {**candidate, "definition": {}}) == "not_assessed"


def test_verifier_owned_lower_metric_and_unknown_values():
    from shrlm.optimization.types import QualityDefinition, QualityMeasurement

    definition = QualityDefinition(
        "distance", "v1", "lower", precision=3, terminal_values={"wrong_format": 100.0}
    )
    config = {"environment": "unfamiliar", "primary_quality": definition.to_dict()}
    measured = Verdict(
        False,
        VerifierCause.WRONG_VALUE,
        "CANARY",
        "CANARY",
        quality=QualityMeasurement(definition.identifier, 25.0),
    )
    terminal = Verdict(False, VerifierCause.WRONG_FORMAT, "", "")
    quality = aggregate_quality_diagnostics([measured, terminal], config)
    assert quality["mean"] == 62.5
    assert quality["n_known_zero"] == 0
    assert quality["n_declared_terminal"] == 1
    unknown = replace(measured, quality=None)
    assert aggregate_quality_diagnostics([measured, unknown], config)["mean"] is None
    assert Verdict.from_dict(measured.to_dict()) == measured
    assert "quality" not in unknown.to_dict()
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            QualityMeasurement(definition.identifier, value)
    with pytest.raises(ValueError, match="definition"):
        aggregate_quality_diagnostics(
            [replace(measured, quality=QualityMeasurement("other/v1", 1))], config
        )


@pytest.mark.parametrize(
    "old_values,new_values,old_mean,new_mean,old_exact,new_exact",
    [
        (
            [0.5, 0.6, 0.7, 0.3, 0.2, 0.8, 0.4, 0.888, 1, 1],
            [0.8, 0.85, 0.9, 0.7, 0.7, 0.8, 0.911, 0.9, 1, None],
            0.6388,
            0.7561,
            2,
            1,
        ),
        ([1, *([0.577] * 8), 0.578], [*([0.719] * 9), 0.722], 0.6194, 0.7193, 1, 0),
        ([1, 1, 1, *([0.447] * 6), 0.446], [1, *([0.661] * 8), 0.663], 0.6128, 0.6951, 3, 1),
    ],
)
def test_rejected_history_preserves_partial_gain_and_rejection_without_payloads(
    tmp_path, monkeypatch, old_values, new_values, old_mean, new_mean, old_exact, new_exact
):
    records = []
    saved_paths = []
    for subject, values in (
        ("baseline", old_values),
        ("merged", new_values),
    ):
        path, _, _ = mining_fixture(tmp_path / subject / "heldout", monkeypatch, attempts=10)
        entries = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
        for entry, value in zip(entries, values, strict=True):
            verdict = Verdict(
                value == 1,
                None if value == 1 else VerifierCause.WRONG_VALUE,
                gold="GOLD CANARY",
                produced="ANSWER CANARY",
                detail=f"precision=0.800 recall=0.800 f1={value:.3f} missing=2 extra=3"
                if value is not None
                else "ERROR CANARY",
            )
            if value is None:
                verdict = replace(verdict, cause=VerifierCause.WRONG_FORMAT)
            entry.update(
                verdict=verdict.to_dict(),
                passed=verdict.passed,
                cause=verdict.cause.value if verdict.cause else None,
            )
        (path / "runs.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
        contract = {
            "verifier_config": {
                key: value
                for key, value in OolongPairsVerifier().config().items()
                if key != "primary_quality"
            },
            "verifier_type": "pairs",
            "repetitions": 10,
            "validation_protocol": "heldout-batch/v1",
        }
        (path.parents[1] / "evaluation.json").write_text(json.dumps(contract))
        records.append(
            {
                "subject_id": subject,
                "decision": "rejected",
                "links": {"splits": {"heldout": {"round_dir": str(path.relative_to(tmp_path))}}},
            }
        )
        saved_paths.append(path / "runs.jsonl")
    before = [path.read_bytes() for path in saved_paths]
    progress = validation_history_progress(tmp_path, records[1], records[0])
    assert progress["status"] == "potentially_promising"
    assert progress["baseline"]["quality"]["mean"] == pytest.approx(old_mean)
    assert progress["candidate"]["quality"]["mean"] == pytest.approx(new_mean)
    assert progress["baseline"]["exact_passes"] == old_exact
    assert progress["candidate"]["exact_passes"] == new_exact
    assert progress["candidate"]["pairs"]["counts_denominator"] == sum(
        value is not None for value in new_values
    )
    assert "CANARY" not in json.dumps(progress) and "held-in" not in json.dumps(progress)
    assert before == [path.read_bytes() for path in saved_paths]
    assert (
        validation_history_progress(tmp_path, {**records[1], "decision": "bundled"}, records[0])[
            "status"
        ]
        == "not_assessed"
    )
    entries[0]["attempt"] = 11
    saved_paths[1].write_text("\n".join(json.dumps(e) for e in entries))
    assert validation_history_progress(tmp_path, records[1], records[0])["status"] == "not_assessed"
    saved_paths[1].write_bytes(before[1])
    contract["verifier_config"] = {"environment": "graphwalks"}
    (tmp_path / "merged/evaluation.json").write_text(json.dumps(contract))
    assert validation_history_progress(tmp_path, records[1], records[0])["status"] == "not_assessed"


def coverage_trace():
    child = completion_dict(
        prompt="Classify record IDs.", response='[[0, "entity"]]', iterations=[], max_depth=2
    )
    codes = [
        code_block(code="print(context[:100])", stdout="INPUT PREVIEW"),
        code_block(code="records = parse(context)"),
        code_block(code="results = rlm_query_batched(chunks)", rlm_calls=[child]),
        code_block(code="labels = parse_returns(results)"),
        code_block(
            code="missing = set(record_ids) - set(labels)",
            stdout="Total classified: 188\nMissing indices: []",
        ),
        code_block(code="answer['ready'] = True"),
    ]
    return as_completion(
        completion_dict(
            prompt="Compute qualifying pairs.",
            response="[(1, 2)]",
            iterations=[
                iteration_entry(index=i, response="", code_blocks=[b])
                for i, b in enumerate(codes, 1)
            ],
            max_depth=2,
        )
    )


def test_cited_call_reveals_later_coverage_check_and_child_contract():
    excerpt = trace_excerpt(coverage_trace(), ["r/i2/b0/c0"])
    rendered = json.dumps(excerpt)
    assert "Missing indices: []" in rendered
    assert "Classify record IDs" in rendered
    assert "INPUT PREVIEW" not in rendered
    assert "answer['ready']" not in rendered
    assert excerpt == trace_excerpt(coverage_trace(), ["r/i2/b0/c0"] * 5)


def test_evidence_packing_bounds_rendered_text_and_deduplicates_operations():
    from shrlm.optimization.proposal_evidence import pack_evidence

    context = {
        "run_id": "held-in-a",
        "task_question": "Preserve counts",
        "trace": trace_excerpt(coverage_trace(), ["r/i2/b0/c0"]),
    }
    patterns = [
        {
            "index": i,
            "signature": {
                "agent_mechanism": [
                    "incomplete_coverage",
                    "lossy_aggregation",
                    "repl_execution_fault",
                    "unparsed_child_output",
                    "iteration_budget_exhaustion",
                    "skipped_verification",
                ][i]
            },
            "eligible_surfaces": ["S3"],
            "support": 3,
        }
        for i in range(6)
    ]
    evidence = {"patterns": {i: context for i in range(6)}, "passing": []}
    rendered, audit = pack_evidence(patterns, evidence, k=4, budget=6000)
    assert len(rendered) <= 6000
    assert audit["evidence_chars"] == len(rendered)
    assert len(audit["expanded_patterns"]) <= 4
    assert rendered.count("results = rlm_query_batched(chunks)") == 1
    assert len(json.loads(rendered.split("\n", 1)[1])["inventory"]) == 6
    assert pack_evidence(patterns, evidence, k=4, budget=6000) == (rendered, audit)


def test_oversized_operation_uses_alternative_without_cutting_code():
    from shrlm.optimization.proposal_evidence import pack_evidence

    huge = {
        "task_question": "task",
        "trace": {"snippets": [{"code": "HUGE" * 20000, "node_id": "r"}]},
    }
    small = {
        "task_question": "task",
        "trace": {"snippets": [{"code": "counts = count(records)", "node_id": "r"}]},
    }
    inventory = [
        {
            "index": 0,
            "signature": {"agent_mechanism": "lossy_aggregation"},
            "eligible_surfaces": ["S3"],
            "support": 2,
        }
    ]
    text, audit = pack_evidence(
        inventory, {"patterns": {0: huge}, "alternatives": {0: [huge, small]}}, k=4, budget=2000
    )
    assert "counts = count(records)" in text
    assert "HUGE" not in text
    assert audit["expanded_patterns"] == [0]


def test_compact_representative_crosses_signatures_before_repeating_mechanism():
    from shrlm.optimization.proposal_evidence import pack_evidence

    inventory = [
        {
            "index": i,
            "signature": {"agent_mechanism": mechanism},
            "instance_support": support,
        }
        for i, (mechanism, support) in enumerate(
            [
                ("incomplete_coverage", 10),
                ("incomplete_coverage", 9),
                ("lossy_aggregation", 8),
            ]
        )
    ]
    contexts = {
        i: {
            "run_id": f"run-{i}",
            "task_question": "Apply the predicate to all documents.",
            "trace": {"snippets": [{"node_id": "r", "code": "#" * size}]},
        }
        for i, size in enumerate([15000, 4000, 15000])
    }
    text, audit = pack_evidence(inventory, {"patterns": contexts}, k=4, budget=23000)
    assert audit["expanded_patterns"] == [1, 2]
    assert audit["expanded_mechanisms"] == ["incomplete_coverage", "lossy_aggregation"]
    assert audit["distinct_actionable_mechanisms"] == 2
    packet = json.loads(text.split("\n", 1)[1])
    assert packet["expanded"]["1"]["run_id"] == "run-1"
    assert len(packet["inventory"]) == 3
    assert len(text) <= 23000
    assert pack_evidence(inventory, {"patterns": contexts}, k=4, budget=23000) == (text, audit)


def test_only_actionable_grounded_patterns_spend_expansion_slots():
    from shrlm.optimization.proposal_evidence import pack_evidence

    inventory = [
        {
            "index": i,
            "signature": {"agent_mechanism": "other", "causal_status": status},
            "actionability": actionability,
            "instance_support": 10 - i,
        }
        for i, (status, actionability) in enumerate(
            [
                ("unattributed", 0.5),
                ("contributing", 0),
                ("contributing", None),
                ("contributing", 0.5),
            ]
        )
    ]
    context = {
        "run_id": "grounded",
        "trace": {"snippets": [{"node_id": "r", "code": "merge(values)"}]},
    }
    text, audit = pack_evidence(inventory, {"patterns": {0: context, 1: context, 2: context}}, k=4)
    assert audit["expanded_patterns"] == [2]
    packet = json.loads(text.split("\n", 1)[1])
    assert packet["inventory"][0]["eligible_surfaces"] == []
    assert "unattributed" in audit["omitted_patterns"]["0"]
    assert "non-actionable" in audit["omitted_patterns"]["1"]
    assert "operation" in audit["omitted_patterns"]["3"]


@pytest.mark.parametrize(
    "snippet",
    [
        {"node_id": "r", "code": "partial", "code_complete": False},
        {"node_id": "r", "code": "combine(values)", "reason": "following consumer context"},
        {"node_id": "r", "observation": "no code or child return"},
    ],
)
def test_no_expansion_for_uncited_or_incomplete_core(snippet):
    from shrlm.optimization.proposal_evidence import pack_evidence

    _, audit = pack_evidence(
        [{"index": 0, "signature": {"agent_mechanism": "other"}}],
        {"patterns": {0: {"trace": {"snippets": [snippet]}}}},
        k=4,
    )
    assert audit["expanded_patterns"] == []
    assert "no resolvable operation" in audit["omitted_patterns"]["0"]


def test_legacy_coverage_basis_stays_unassessed_without_rewriting_records(tmp_path, monkeypatch):
    path, bundle, records = mining_fixture(tmp_path, monkeypatch, attempts=1)
    bundle["patterns"][0]["signature"]["agent_mechanism"] = "incomplete_coverage"
    records[0]["signature"] = bundle["patterns"][0]["signature"]
    (path / "bundle.json").write_text(json.dumps(bundle))
    (path / "records.jsonl").write_text(json.dumps(records[0]))
    original = (path / "records.jsonl").read_bytes()
    context = load_proposal_evidence(path, bundle)["patterns"][0]
    assert context["coverage_basis"] == "coverage basis not assessed"
    assert (path / "records.jsonl").read_bytes() == original


def test_explicit_operation_wins_and_unresolvable_citation_is_labelled():
    excerpt = trace_excerpt(
        coverage_trace(),
        [],
        [
            {
                "node_id": "r",
                "iteration_index": 5,
                "code_block_index": 0,
                "observation": "All parsed record IDs covered.",
            }
        ],
    )
    assert excerpt["snippets"][0]["iteration_index"] == 5
    assert "Missing indices: []" in json.dumps(excerpt)
    fallback = trace_excerpt(coverage_trace(), ["unknown"])
    assert "fallback" in fallback["selection"]
    assert "unresolved" in fallback["selection"]


def test_nested_operation_keeps_its_node_and_payload_budget():
    child = coverage_trace().to_dict()
    for iteration in child["metadata"]["iterations"]:
        for block in iteration["code_blocks"]:
            block["code"] += "# large\n" * 1000
            block["result"]["stdout"] = "large output\n" * 1000
    outer = as_completion(
        completion_dict(
            prompt="outer",
            response="answer",
            max_depth=3,
            iterations=[
                iteration_entry(
                    index=5,
                    response="",
                    code_blocks=[
                        code_block(code="outer_result = rlm_query(context)", rlm_calls=[child])
                    ],
                )
            ],
        )
    )
    excerpt = trace_excerpt(
        outer,
        [],
        [
            {
                "node_id": "r/i0/b0/c0",
                "iteration_index": 5,
                "code_block_index": 0,
                "observation": "coverage check inside child",
            }
        ],
    )
    assert excerpt["snippets"][0]["node_id"] == "r/i0/b0/c0"
    assert "missing =" in excerpt["snippets"][0]["code"]
    # Code remains complete; payload fields can be bounded. The final rendered
    # evidence pack, rather than equal per-field slices, owns the total budget.
    original = child["metadata"]["iterations"][4]["code_blocks"][0]["code"]
    assert excerpt["snippets"][0]["code"] == original
    assert "truncated" in json.dumps(excerpt)


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


def mining_fixture(tmp_path, monkeypatch, *, attempts=2, produced="[(1, 2)]"):
    instance = {
        "id": "held-in",
        "question": "ACTUAL PREDICATE and LABEL VOCABULARY",
        "prompt": "classify rows",
        "gold_pairs": [(1, 2), (1, 3)],
    }
    factory = ClientFactory([final(produced)] * attempts)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_round_config(
        tmp_path, instances=[instance], attempts=attempts, verifier=OolongPairsVerifier()
    )
    run_round(config)
    runs, verdicts, _, entries = load_round(tmp_path, 1)
    path = tmp_path / "round_01"
    pattern = {**copy.deepcopy(PATTERN_TEXT), "representatives": ["held-in"]}
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
    expected = next(record for record in records if record["run_id"] == context["run_id"])
    assert context["symptom_summary"] == expected["detail"]["symptom_summary"]
    assert len(evidence["alternatives"][0]) == 2
    assert context["missing_examples"] == [(1, 3)]
    assert context["pair_diagnostics"]["metrics"]["f1"] == 0.667
    assert context["task_question"] == "ACTUAL PREDICATE and LABEL VOCABULARY"
    assert "snippets" in context["trace"]
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
    assert "no resolvable operation" in prompt
    assert "ACTUAL PREDICATE" not in prompt
    assert "original row ordinal" not in prompt
    assert "unparsed:" not in prompt
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


def test_known_terminal_zero_retains_unparsed_failure_detail(tmp_path, monkeypatch):
    path, bundle, records = mining_fixture(tmp_path, monkeypatch, attempts=1, produced="[]")
    context = load_proposal_evidence(path, bundle)["patterns"][0]
    assert context["quality_diagnostics"]["n_known_zero"] == 1
    assert context["quality_diagnostics"]["mean"] == 0
    assert context["verifier_outcome"] == {"passed": False, "cause": "wrong_format"}
    assert records[0]["verdict"]["detail"] in context["verifier_detail"]


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
    # Legacy ledgers can retain subject hashes after an old evaluation contract
    # becomes unavailable. Missing context is not evidence of identity corruption.
    legacy = validation_history_diagnostics(tmp_path, {**record, "harness_hash": "legacy-hash"})
    assert legacy["quality"]["definition"] is None


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


def test_task_reasoning_replaces_recipe_on_every_surface(monkeypatch):
    from shrlm.optimization.taxonomy import MECHANISM_SURFACES, AgentMechanism, EditableSurface

    monkeypatch.setitem(
        MECHANISM_SURFACES,
        AgentMechanism.PREMATURE_TERMINATION,
        (EditableSurface.ANSWER_MIDDLEWARE,),
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
    assert "Which information must survive each step" in prompt
    prompt, _ = render_prompt([PATTERN_TEXT], serialize_harness(H0), [], [], 4)
    assert "original row ordinal" not in prompt


def test_evidence_budget_counts_escaping_and_keeps_relevant_contrast():
    from shrlm.optimization.proposal_evidence import EVIDENCE_HEADING, pack_evidence

    def context(run_id, code):
        return {
            "run_id": run_id,
            "task_question": "Keep each task condition intact.",
            "trace": {
                "snippets": [
                    {
                        "node_id": "r",
                        "iteration_index": 0,
                        "code_block_index": 0,
                        "code": code,
                        "code_complete": True,
                    }
                ]
            },
        }

    code = 'result = classify("\\\\quoted")\n' * 25
    failed = context("failed", code)
    passed = context("passed", "result = classify(inputs)\nverify(result)\n")
    unrelated = context("unrelated", "x = unrelated_operation()")
    inventory = [{"index": 7, "signature": {"agent_mechanism": "lossy_aggregation"}, "support": 3}]
    evidence = {"patterns": {7: failed}, "passing": [unrelated, passed]}
    rendered, audit = pack_evidence(inventory, evidence, k=4, budget=3200)
    section = json.loads(rendered.removeprefix(EVIDENCE_HEADING))
    assert audit["evidence_chars"] == len(rendered) <= 3200
    assert section["passing"][0]["run_id"] == "passed"
    assert "unrelated_operation" not in rendered
    assert "unverified" in section["contrast_status"]
    assert code in [operation["code"] for operation in section["operations"].values()]
    exact, _ = pack_evidence(inventory, evidence, k=4, budget=len(rendered))
    assert exact == rendered


def test_oversized_question_is_omitted_whole_with_inventory_preserved():
    from shrlm.optimization.proposal_evidence import EVIDENCE_HEADING, pack_evidence

    inventory = [{"index": 9, "signature": {"agent_mechanism": "lossy_aggregation"}}]
    question = "QUESTION_SENTINEL" * 1000
    rendered, audit = pack_evidence(
        inventory,
        {
            "patterns": {
                9: {
                    "task_question": question,
                    "trace": {"snippets": [{"node_id": "r", "code": "combine(values)"}]},
                }
            }
        },
        k=4,
        budget=2000,
    )
    section = json.loads(rendered.removeprefix(EVIDENCE_HEADING))
    assert section["inventory"][0]["index"] == inventory[0]["index"]
    assert section["inventory"][0]["eligible_surfaces"] == []
    assert not section["inventory"][0]["selectable"]
    assert not section["expanded"]
    assert "QUESTION_SENTINEL" not in rendered
    assert "exceeds remaining budget" in audit["omitted_patterns"]["9"]


def consumer_chain_trace(stderr=""):
    children = [
        completion_dict(
            prompt=f"Classify segment {i}",
            response=json.dumps({str(j): [1, 2] for j in range(26)}),
            iterations=[],
            max_depth=2,
        )
        for i in range(5)
    ]
    blocks = [
        code_block(code="replies = rlm_query_batched(parts)", rlm_calls=children, stderr=stderr),
        code_block(code="print(replies[:1])", stdout="PREVIEW"),
        code_block(code="print(len(replies))", stdout="5"),
        code_block(code="merged = merge(replies)", stdout="merge complete"),
        code_block(code="filtered = apply_predicate(merged)"),
        code_block(code="print(len(filtered))", stdout="remaining: 26"),
    ]
    return as_completion(
        completion_dict(
            prompt="Combine segments",
            response="wrong",
            max_depth=2,
            iterations=[
                iteration_entry(i, "", code_blocks=[block]) for i, block in enumerate(blocks)
            ],
        )
    )


def test_computational_consumers_survive_previews_and_sibling_returns():
    from shrlm.optimization.proposal_evidence import pack_evidence

    trace = trace_excerpt(
        consumer_chain_trace(), [], [{"node_id": "r", "iteration_index": 0, "code_block_index": 0}]
    )
    codes = [op.get("code", "") for op in trace["snippets"]]
    assert "merged = merge(replies)" in codes
    assert "filtered = apply_predicate(merged)" in codes
    assert "print(len(filtered))" in codes
    assert len(codes) <= 6
    assert "PREVIEW" not in json.dumps(trace)
    child = next(op for op in trace["snippets"] if "response" in op)
    assert child["response_structure"]["item_count"] == 26
    contexts = {
        0: {"trace": trace},
        1: {
            "trace": {
                "snippets": [
                    {
                        "node_id": "r",
                        "code": "check(result)",
                        "code_complete": True,
                        "reason": "cited operation",
                    }
                ]
            }
        },
    }
    _, audit = pack_evidence(
        [
            {"index": i, "signature": {"agent_mechanism": m}}
            for i, m in enumerate(["lossy_aggregation", "skipped_verification"])
        ],
        {"patterns": contexts},
        k=2,
        budget=6500,
    )
    assert set(audit["expanded_patterns"]) == {0, 1}


def test_retry_notices_alone_do_not_authorize_recovery_route():
    from shrlm.optimization.proposal_evidence import operation_support

    notice = "Transient API error (RateLimitError); retrying (1/6)...\n"
    for suffix, expected in (("", False), ("ValueError: terminal", True)):
        trace = trace_excerpt(
            consumer_chain_trace(notice * 10 + suffix),
            [],
            [{"node_id": "r", "iteration_index": 0, "code_block_index": 0}],
        )
        assert trace["snippets"][0]["error_observed"] is expected
        support = operation_support(
            "repl_execution_fault", {str(i): op for i, op in enumerate(trace["snippets"])}
        )
        assert ("S5" in support) is expected
        assert trace["retry_notices"][0]["count"] == 10


def test_required_sibling_comparison_is_not_partially_admitted():
    trace = trace_excerpt(consumer_chain_trace(), [f"r/i0/b0/c{i}" for i in range(5)])
    from shrlm.optimization.proposal_evidence import pack_evidence

    _, audit = pack_evidence(
        [{"index": 0, "signature": {"agent_mechanism": "lossy_aggregation"}}],
        {"patterns": {0: {"trace": trace}}},
        k=1,
    )
    assert not audit["expanded_patterns"]
