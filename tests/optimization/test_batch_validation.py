"""The promotion gate evaluates one fixed batch on held-out instances only."""

import json
from dataclasses import replace

import pytest

import rlm.core.rlm as rlm_module
from shrlm.optimization.validation import validate_round
from shrlm.rlm_harness import H0
from tests.optimization.test_validation import ClientFactory, final
from tests.optimization.test_validation_e2e import edited, make_config, write_candidate


@pytest.mark.parametrize("merged", [False, True])
def test_rejected_evaluated_harness_is_not_repeated(tmp_path, monkeypatch, merged):
    from shrlm.harness_identity import harness_hash
    from shrlm.optimization.candidates import load_candidates
    from shrlm.optimization.promotion import plan_batch

    proposals = tmp_path / "proposals"
    write_candidate(proposals, edited("a", "execution_instruction", "a"), "S3", "a")
    if merged:
        write_candidate(proposals, edited("b", "decomposition_instruction", "b"), "S2", "b")
    admitted, _ = load_candidates(proposals, H0)
    fingerprint = plan_batch(H0, admitted).harness_hash
    prior = [
        {
            "round": 1,
            "subject_id": "old",
            "incumbent_hash": harness_hash(H0),
            "harness_hash": fingerprint,
        }
    ]
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    result = validate_round(H0, proposals, make_config(tmp_path), prior_evaluations=prior)
    assert result.evaluation is None
    assert idle.total_calls == 0
    assert all(r["upstream"]["gate"] == "duplicate_evaluation" for r in result.ledger.records)
    assert all(r["rule"] is None for r in result.ledger.records)
    replay = validate_round(H0, proposals, make_config(tmp_path), prior_evaluations=prior)
    assert replay.ledger.records == result.ledger.records
    assert idle.total_calls == 0


def test_batch_runs_only_baseline_and_combined_candidate(tmp_path, monkeypatch):
    proposals = tmp_path / "proposals"
    write_candidate(proposals, edited("a", "decomposition_instruction", "a"), "S2", "a")
    write_candidate(proposals, edited("b", "execution_instruction", "b"), "S3", "b")
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    result = validate_round(H0, proposals, make_config(tmp_path))
    assert factory.total_calls == 4
    assert result.promoted
    assert result.plan.constituent_ids == ("a", "b")
    assert not (result.round_path / "a").exists()
    assert not list(result.round_path.glob("*/heldin"))
    records = {r["subject_id"]: r for r in result.ledger.records}
    assert records["a"]["decision"] == "bundled"
    assert records["a"]["rule"] is None
    assert records["a"]["links"] is None
    assert records["merged"]["decision"] == "promoted"
    assert set(records["merged"]["rule"]) == {"heldout"}
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    replay = validate_round(H0, proposals, make_config(tmp_path))
    assert idle.total_calls == 0
    assert replay.ledger.records == result.ledger.records


@pytest.mark.parametrize("valid_sibling", [False, True])
def test_duplicate_guard_uses_admitted_batch_not_requested_members(
    tmp_path, monkeypatch, valid_sibling
):
    from shrlm.harness_identity import harness_hash

    proposals = tmp_path / "proposals"
    candidate = edited("a", "execution_instruction", "a")
    write_candidate(proposals, candidate, "S3", "a")
    # An unchanged sibling is refused locally; a valid sibling changes the
    # evaluated subject and must not inherit the standalone rejection.
    sibling = edited("b", "decomposition_instruction", "b") if valid_sibling else H0
    write_candidate(proposals, sibling, "S2", "b")
    prior = [
        {
            "round": 1,
            "subject_id": "prior-a",
            "incumbent_hash": harness_hash(H0),
            "harness_hash": harness_hash(candidate),
        }
    ]
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2 if valid_sibling else [])
    monkeypatch.setattr(rlm_module, "get_client", factory)
    result = validate_round(H0, proposals, make_config(tmp_path), prior_evaluations=prior)
    if valid_sibling:
        assert result.promoted and factory.total_calls == 4
        assert result.plan.constituent_ids == ("a", "b")
    else:
        assert result.evaluation is None and factory.total_calls == 0
        gates = {record["upstream"]["gate"] for record in result.ledger.records}
        assert "duplicate_evaluation" in gates and len(gates) == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_duplicate_surface_fails_before_evaluation(tmp_path, monkeypatch, workers):
    from tests.optimization.test_validation import GOLD_VERIFIER_FACTORY

    proposals = tmp_path / "proposals"
    for name in ("a", "b"):
        write_candidate(proposals, edited(name, "execution_instruction", name), "S3", name)
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    config = make_config(tmp_path, workers=workers, verifier_factory=GOLD_VERIFIER_FACTORY)
    with pytest.raises(ValueError, match="surface S3"):
        validate_round(H0, proposals, config)
    assert idle.total_calls == 0
    assert not config.out_dir.exists()


@pytest.mark.parametrize("change", ["sample", "repetitions", "batch", "threshold", "empty"])
def test_changed_round_inputs_fail_before_calls(tmp_path, monkeypatch, change):
    from shrlm.optimization.promotion import PromotionConfig

    proposals = tmp_path / "proposals"
    proposal = write_candidate(proposals, edited("a", "execution_instruction", "a"), "S3", "a")
    config = make_config(tmp_path)
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    validate_round(H0, proposals, config)
    promotion = None
    if change == "sample":
        config = replace(
            config,
            splits=replace(
                config.splits,
                heldout=[dict(instance, prompt="changed") for instance in config.splits.heldout],
            ),
        )
    elif change == "repetitions":
        config = replace(config, repetitions=2)
    elif change == "batch":
        write_candidate(proposals, edited("b", "decomposition_instruction", "b"), "S2", "b")
    elif change == "threshold":
        promotion = PromotionConfig(tau_improvement=1)
    else:
        proposal.unlink()
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    with pytest.raises(ValueError, match="different validation contract"):
        validate_round(H0, proposals, config, promotion)
    assert idle.total_calls == 0


@pytest.mark.parametrize("artifact", ["heldin/round_00/harness.json", "summary.json"])
def test_legacy_subject_evidence_fails_before_calls(tmp_path, monkeypatch, artifact):
    from shrlm.optimization.validation import evaluate_subject, subject_dir

    config = make_config(tmp_path)
    path = subject_dir(config.out_dir, config.round_index, "baseline") / artifact
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    with pytest.raises(ValueError, match="legacy validation evidence"):
        evaluate_subject("baseline", H0, config)
    assert idle.total_calls == 0
    assert path.read_text() == "{}"


def test_current_round_can_resume_with_subject_workers(tmp_path, monkeypatch):
    from tests.optimization.test_validation import GOLD_VERIFIER_FACTORY, parallel_client_factory

    proposals = tmp_path / "proposals"
    write_candidate(proposals, edited("a", "execution_instruction", "a"), "S3", "a")
    config = make_config(tmp_path)
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    first = validate_round(H0, proposals, config)
    replay = validate_round(
        H0,
        proposals,
        replace(
            config,
            workers=2,
            verifier_factory=GOLD_VERIFIER_FACTORY,
            client_factory=parallel_client_factory(tmp_path, {"baseline": [], "a": []}),
        ),
    )
    assert replay.ledger.records == first.ledger.records


def test_legacy_summaries_are_readable_but_cannot_be_scored(tmp_path):
    from shrlm.optimization.promotion import PromotionConfig, score_candidate
    from shrlm.optimization.validation import load_summary
    from tests.optimization.test_promotion import make_summary

    summary = make_summary("baseline", 2, 2)
    summary["format"] = "shrlm-validation-summary/v1"
    del summary["validation_protocol"]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert load_summary(tmp_path) == summary
    with pytest.raises(ValueError, match="legacy evidence"):
        score_candidate(summary, make_summary("candidate", 4, 4), PromotionConfig())


@pytest.mark.parametrize("multiple", [False, True])
def test_over_budget_batch_promotes_nothing(tmp_path, monkeypatch, multiple):
    proposals = tmp_path / "proposals"
    write_candidate(proposals, edited("a", "execution_instruction", "a"), "S3", "a")
    if multiple:
        write_candidate(proposals, edited("b", "decomposition_instruction", "b"), "S2", "b")
    factory = ClientFactory([final("WRONG")] * 2 + ["Still working."] * 4)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_config(tmp_path)
    config = replace(config, caps=replace(config.caps, candidate_budget=0.0025))
    result = validate_round(H0, proposals, config)
    assert factory.total_calls == 6
    assert not result.promoted
    measured = result.ledger.records[-1]
    assert measured["decision"] == "over_budget"
    assert measured["rule"] is None
    if multiple:
        assert all(record["decision"] == "bundled" for record in result.ledger.records[:-1])


def test_changed_verifier_settings_refuse_resume(tmp_path, monkeypatch):
    from shrlm.optimization.validation import evaluate_subject
    from tests.optimization.test_validation import GoldVerifier

    class ConfigurableVerifier(GoldVerifier):
        def __init__(self, tolerance):
            self.tolerance = tolerance

        def config(self):
            return {"tolerance": self.tolerance}

    config = make_config(tmp_path, verifier=ConfigurableVerifier(0))
    factory = ClientFactory([final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    evaluate_subject("baseline", H0, config)
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    with pytest.raises(ValueError, match="different validation contract"):
        evaluate_subject("baseline", H0, replace(config, verifier=ConfigurableVerifier(1)))
    assert idle.total_calls == 0


def test_writer_refuses_missing_batch_evaluation_or_scored_constituent(tmp_path, monkeypatch):
    from shrlm.optimization.candidates import load_candidates
    from shrlm.optimization.validation import write_promotion_ledger

    proposals = tmp_path / "proposals"
    for name, field, surface in [
        ("a", "execution_instruction", "S3"),
        ("b", "decomposition_instruction", "S2"),
    ]:
        write_candidate(proposals, edited(name, field, name), surface, name)
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    result = validate_round(H0, proposals, make_config(tmp_path))
    constituents, _ = load_candidates(proposals, H0)
    assert result.ledger.decision["n_candidates"] == 2
    with pytest.raises(ValueError, match="exactly one evaluated"):
        write_promotion_ledger(
            replace(result.evaluation, candidates=[]),
            result.decisions[:-1],
            result.plan,
            constituents=constituents,
        )
    scored = [replace(result.decisions[0], decision="accepted"), *result.decisions[1:]]
    with pytest.raises(ValueError, match="without individual scores"):
        write_promotion_ledger(result.evaluation, scored, result.plan, constituents=constituents)


def test_standalone_oolong_preflight_rejects_before_evaluation(tmp_path, monkeypatch):
    from shrlm.environments.oolong_pairs import OolongPairsVerifier
    from tests.optimization.test_candidates import missing_accept_argument

    proposals = tmp_path / "proposals"
    write_candidate(
        proposals, replace(H0, answer_middleware=missing_accept_argument), "S9", "broken"
    )
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    config = make_config(tmp_path, verifier=OolongPairsVerifier())
    result = validate_round(H0, proposals, config)
    assert idle.total_calls == 0
    assert not result.promoted
    assert "bracketed_pair" in result.loader_rejections[0].reason
    assert not (result.round_path / "baseline").exists()
    with pytest.raises(ValueError, match="preflight profile changed"):
        validate_round(H0, proposals, config, preflight_profile="generic/v1")
    assert idle.total_calls == 0


def test_prechange_validation_contract_replays_without_rewriting(tmp_path, monkeypatch):
    proposals = tmp_path / "proposals"
    write_candidate(proposals, edited("a", "execution_instruction", "a"), "S3", "a")
    factory = ClientFactory([final("WRONG")] * 2 + [final("RIGHT")] * 2)
    monkeypatch.setattr(rlm_module, "get_client", factory)
    config = make_config(tmp_path)
    result = validate_round(H0, proposals, config)
    contract_path = result.round_path / "validation.json"
    contract = json.loads(contract_path.read_text())
    del contract["preflight_profile"]
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    before = contract_path.read_bytes()
    ledger = result.ledger.ledger_path.read_bytes()
    idle = ClientFactory([])
    monkeypatch.setattr(rlm_module, "get_client", idle)
    replay = validate_round(H0, proposals, config)
    assert replay.promoted
    assert idle.total_calls == 0
    assert contract_path.read_bytes() == before
    assert result.ledger.ledger_path.read_bytes() == ledger
