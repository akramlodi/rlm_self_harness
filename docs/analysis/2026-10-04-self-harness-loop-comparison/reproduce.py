"""Offline probes of optimizer differences; no model calls or experiment launches.

Run from the repository root:
    .venv/bin/python docs/analysis/2026-10-04-self-harness-loop-comparison/reproduce.py
"""

# Imports below require the repository and cloned package paths set first.
# ruff: noqa: E402

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Self-Harness/proposer/src"))

from self_harness_proposer.hooks import apply_candidate_values

from shrlm.optimization.proposal import propose_round
from shrlm.optimization.proposal_evidence import load_proposal_evidence
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict
from shrlm.rlm_harness import H0
from tests.mock_lm import MockLM
from tests.optimization.test_proposal import (
    canned_batch,
    edit_item,
    make_pattern,
    synthetic_evidence,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    gate = load_module(
        "review_upstream_acceptance",
        ROOT / "Self-Harness/acceptance/scripts/run_acceptance_gate.py",
    )

    def result(train, heldout):
        return {
            "splits": {
                split: [{"repeat": repeat, "passed": count, "total": 10} for repeat in (1, 2)]
                for split, count in (("train", train), ("heldout", heldout))
            }
        }

    decisions = {}
    for name, candidate in (
        ("both_tied", result(5, 5)),
        ("train_gain_heldout_tie", result(6, 5)),
        ("train_loss_heldout_gain", result(4, 6)),
    ):
        decision = gate.run_acceptance_gate(
            baseline_result=result(5, 5), candidate_result=candidate
        )
        decisions[name] = decision["accepted"]
    assert decisions == {
        "both_tied": False,
        "train_gain_heldout_tie": True,
        "train_loss_heldout_gain": False,
    }

    source = (
        "def build_subagents():\n    return []\n\n"
        "def build_verification_instruction():\n    return 'Check the result.'\n"
    )
    updated = apply_candidate_values(
        current_values={"baseline": source},
        candidate_values={
            "subagent_call_policy": {
                "subagents": [
                    {
                        "name": "checker",
                        "description": "Check the requested artifact.",
                        "system_prompt": "Inspect the provided evidence.",
                    }
                ],
                "call_when": ["the required artifact has been written"],
            }
        },
    )["baseline"]
    before = {node.name: ast.dump(node) for node in ast.parse(source).body}
    changed = [node.name for node in ast.parse(updated).body if ast.dump(node) != before[node.name]]
    assert changed == ["build_subagents", "build_verification_instruction"]

    with tempfile.TemporaryDirectory(prefix="self-harness-review-") as scratch:
        scratch = Path(scratch)
        patterns = [make_pattern("lossy_aggregation")]
        response = canned_batch(
            edit_item(
                0,
                {"kind": "text", "new_text": "Use the declared aggregation procedure."},
                surface="S3",
            ),
            edit_item(
                0,
                {"kind": "text", "new_text": "Check the aggregation procedure's output."},
                surface="S4",
            ),
        )
        proposed = propose_round(
            {"patterns": patterns},
            H0,
            MockLM(responses=[response, canned_batch()]),
            scratch / "proposals",
            workdir=scratch / "work",
            evidence=synthetic_evidence(patterns, preferred=()),
        )
        surfaces = [item.surface for item in proposed.written]
        occupied = [
            admission
            for attempt in proposed.attempts
            for admission in attempt.admissions
            if admission["status"] == "occupied"
        ]
        assert surfaces == ["S3"]
        assert any(item["surface"] == "S4" for item in occupied)

        signature = make_pattern("other")["signature"]
        detail = {
            "symptom_summary": "The result is incorrect.",
            "agent_mechanism_detail": "Dropped multiplicity while converting a list to a set.",
            "causal_status_detail": "The final count uses the deduplicated collection.",
            "failing_level_detail": "The final calculation is visible in the root.",
        }
        record = {
            "instance_id": "case-a",
            "signature": signature,
            "detail": detail,
            "verdict": Verdict(False, VerifierCause.WRONG_VALUE, "2", "1").to_dict(),
        }
        (scratch / "records.jsonl").write_text(json.dumps(record) + "\n")
        bundle = {"patterns": [{"signature": signature, "representatives": ["case-a"]}]}
        context = load_proposal_evidence(scratch, bundle)["patterns"][0]
        dropped = sorted(set(detail) - set(context))
        assert dropped == ["agent_mechanism_detail", "causal_status_detail", "failing_level_detail"]
        assert detail["agent_mechanism_detail"] not in json.dumps(context)

    print(
        json.dumps(
            {
                "upstream_acceptance": decisions,
                "upstream_single_alias_changed_functions": changed,
                "ours_same_pattern_distinct_surfaces_admitted": surfaces,
                "ours_same_pattern_occupancy": occupied,
                "ours_diagnosis_fields_dropped_before_proposer": dropped,
                "model_calls": 0,
                "note": "Synthetic probes establish implementation behavior, not performance benefit.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
