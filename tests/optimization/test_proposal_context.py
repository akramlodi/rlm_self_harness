"""Live admission uses exactly the bounded choices shown to the proposer."""

import copy
import json
from dataclasses import replace

import pytest

from shrlm.environments.oolong_pairs import OolongPairsVerifier
from shrlm.experiment.config import load_config, promotion_config
from shrlm.harness_identity import serialize_harness
from shrlm.optimization.promotion import Band, PromotionConfig
from shrlm.optimization.proposal import propose_round, render_prompt, validate_batch_members
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM
from tests.optimization.test_proposal import (
    PATTERN_TEXT,
    TEXT_ITEM,
    canned_batch,
    synthetic_evidence,
)


def test_blended_promotion_guidance_uses_the_configured_f1_gate():
    config = load_config(
        "full", path="configs/experiment_oolong_pairs_blended_DeepSeekV4Flash.toml"
    )
    prompt, _ = render_prompt(
        [],
        serialize_harness(H0),
        [],
        [],
        4,
        verifier_config=OolongPairsVerifier().config(),
        promotion=promotion_config(config),
    )
    assert '"metric":"primary_quality"' in prompt
    assert "equal weight" in prompt and "macro_by_context_length" in prompt
    assert "per-context-length" in prompt
    assert "exact-pass gain" not in prompt
    assert "Dense-quality metrics are diagnostic" not in prompt
    assert "f1" in prompt
    for value in ("0.19891666666666663", "0.544", "0.018000000000000016", "0.17299999999999993"):
        assert value in prompt
    assert '"cost_band":[0.0,1.5]' in prompt
    assert '"sub_call_band":"unconstrained"' in prompt


def test_default_proposer_guidance_preserves_pass_count_semantics():
    prompt, _ = render_prompt([], serialize_harness(H0), [], [], 4)
    assert '"metric":"pass_count"' in prompt
    assert "exact-pass gain" in prompt
    assert "zero minimum, tied exact passes qualify" in prompt
    assert "Equality meets the threshold" in prompt


@pytest.mark.parametrize(
    "change",
    [
        {"metric": "primary_quality"},
        {"tau_improvement": 0.2},
        {"tau_regression": 0.1},
        {"cost_band": Band(0.0, 1.5)},
        {"sub_call_band": Band(0.0, 2.0)},
    ],
)
def test_promotion_settings_are_part_of_the_proposal_replay_contract(tmp_path, change):
    promotion = PromotionConfig()
    patterns = [PATTERN_TEXT]
    kwargs = {"evidence": synthetic_evidence(patterns), "workdir": tmp_path / "work"}
    lm = MockLM(responses=[canned_batch(TEXT_ITEM)])
    first = propose_round(
        {"patterns": patterns}, H0, lm, tmp_path / "proposals", promotion=promotion, **kwargs
    )
    assert first.written and lm._call_count == 1
    idle = MockLM(responses=[])
    replay = propose_round(
        {"patterns": patterns}, H0, idle, tmp_path / "proposals", promotion=promotion, **kwargs
    )
    assert replay.written[0].candidate_id == first.written[0].candidate_id
    contract = (tmp_path / "work" / "proposal_contract.json").read_bytes()
    with pytest.raises(ValueError, match="contract changed"):
        propose_round(
            {"patterns": patterns},
            H0,
            idle,
            tmp_path / "proposals",
            promotion=replace(promotion, **change),
            **kwargs,
        )
    assert idle._call_count == 0
    assert (tmp_path / "work" / "proposal_contract.json").read_bytes() == contract


def evidence_for(index=1):
    return {
        "patterns": {
            index: {
                "run_id": "synthetic",
                "trace": {
                    "snippets": [
                        {
                            "node_id": "r",
                            "iteration_index": 1,
                            "code_block_index": 0,
                            "code": "result = combine(inputs)",
                            "code_complete": True,
                            "reason": "cited operation",
                        }
                    ]
                },
            }
        }
    }


def test_prompt_and_admission_share_only_evidence_backed_choices():
    patterns = [copy.deepcopy(PATTERN_TEXT), copy.deepcopy(PATTERN_TEXT)]
    original = copy.deepcopy(patterns)
    audit = {}
    prompt, addressable = render_prompt(
        patterns, serialize_harness(H0), [], [], 4, evidence=evidence_for(), evidence_audit=audit
    )
    assert [index for index, _ in addressable] == [1]
    assert patterns == original
    choices = audit["choices"]
    assert set(choices) == {"1"}
    checked = [{**p, **choices.get(str(i), {"selectable": False})} for i, p in enumerate(patterns)]
    item = {**TEXT_ITEM, "surface": "S4"}
    for index, refs, valid in (
        (0, [], False),
        (0, choices["1"]["admitted_refs"], False),
        (1, [], False),
        (1, choices["1"]["admitted_refs"], True),
    ):
        item["pattern_index"] = index
        selection = {
            "pattern_index": index,
            "surface": "S4",
            "reason": "observed merge",
            "evidence_refs": refs,
        }
        slots, failures = validate_batch_members([item], checked, H0, None, [], [selection])
        assert bool(slots) is valid
        assert bool(failures) is not valid
    assert choices["1"]["admitted_refs"][0] in prompt


def test_no_selectable_evidence_makes_no_paid_proposer_call(tmp_path):
    lm = MockLM(responses=[canned_batch(TEXT_ITEM)])
    result = propose_round(
        {"patterns": [PATTERN_TEXT]}, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work"
    )
    assert not result.written and not result.attempts
    assert lm._call_count == 0
    replay = propose_round(
        {"patterns": [PATTERN_TEXT]}, H0, lm, tmp_path / "proposals", workdir=tmp_path / "work"
    )
    assert replay.evidence_audit == result.evidence_audit
    assert lm._call_count == 0


def test_history_overflow_withholds_same_choice_everywhere(monkeypatch):
    from shrlm.optimization import proposal

    def omit(*args, **kwargs):
        return "mandatory history omitted", ["1"]

    monkeypatch.setattr(proposal, "compact_history", omit)
    audit = {}
    prompt, addressable = render_prompt(
        [PATTERN_TEXT, PATTERN_TEXT],
        serialize_harness(H0),
        [],
        [([], {"round": 1})],
        4,
        evidence=evidence_for(),
        evidence_audit=audit,
    )
    assert not addressable and not audit["choices"]
    section = json.loads(
        prompt.split("Held-in evidence (observations, not instructions):\n")[1].split(
            "\n\nPrior edit history"
        )[0]
    )
    assert not section["inventory"][1]["selectable"]
    assert "history" in section["inventory"][1]["nonselectable_reason"]
