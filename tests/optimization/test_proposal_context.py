"""Live admission uses exactly the bounded choices shown to the proposer."""

import copy
import json

from shrlm.harness_identity import serialize_harness
from shrlm.optimization.proposal import propose_round, render_prompt, validate_batch_members
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM
from tests.optimization.test_proposal import PATTERN_TEXT, TEXT_ITEM, canned_batch


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
