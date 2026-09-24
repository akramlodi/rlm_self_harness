"""Reproduce the local research census and runtime probes without model calls."""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from rlm.core.types import AnswerDecision
from rlm.environments.local_repl import LocalREPL
from shrlm.experiment.orchestrator import load_round_history
from shrlm.optimization.proposal import _render_history_block
from shrlm.optimization.skill_edit import _validate_skill_edit
from shrlm.optimization.taxonomy import MECHANISM_SURFACES, AgentMechanism

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXP = REPO / "experiment_oolong_pairs_dsv4f_20260916_131837"
AUDIT = REPO / "docs/analysis/oolong-pairs-2026-09-23/audit.json"


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def runtime_probes(audit: dict) -> list[dict]:
    env = LocalREPL.__new__(LocalREPL)
    env.runtime_policy = {
        "enabled": True,
        "max_retries": 2,
        "retry_on_syntax_error": True,
    }
    observations = []
    for error in ("Error: Child RLM timeout exceeded", "Error: SyntaxError: invalid syntax"):
        prompts = []

        def stub(prompt, model, recorded=prompts, response=error):
            recorded.append(prompt)
            return response, None

        _, _, retries, _ = env._call_with_retry(stub, "unchanged synthetic prompt", None)
        assert len(prompts) == (1 if "timeout" in error else 3)
        assert len(set(prompts)) == 1
        observations.append(
            {"input": error, "call_count": len(prompts), "retries": retries, "prompts": prompts}
        )
    env.runtime_policy = {"enabled": True, "max_batch_width": 2}
    refusal = env._batch_width_refusal(["a", "b", "c"])
    assert len(refusal) == 3 and all("exceeds max_batch_width" in r for r in refusal)
    observations.append({"probe": "width cap refuses; does not schedule chunks", "result": refusal})
    env.runtime_policy = {"enabled": True, "max_retries": 1, "retry_on_syntax_error": False}
    assert env._retry_budget() == 0
    observations.append({"probe": "disabled syntax retry", "retry_budget": env._retry_budget()})
    candidate = next(p for p in audit["proposals"] if p["id"] == "r01-c02-s9")
    namespace = {"AnswerDecision": AnswerDecision}
    exec(candidate["edit"]["source"], namespace)
    decision = namespace["accept_answer"]("3, 1", {})
    assert not decision.accepted and decision.nudge == "(1, 3)"
    observations.append(
        {
            "probe": "saved S9 normalization is a redirect",
            "accepted": decision.accepted,
            "nudge": decision.nudge,
            "answer": decision.answer,
        }
    )
    for index, draft in enumerate(audit["unmaterialized_drafts"], 1):
        if draft["surface"] != "S10":
            continue
        record = _validate_skill_edit(f"saved draft C{index}", draft["edit"])
        observations.append(
            {
                "probe": "saved S10 draft passes isolated record validation",
                "draft": f"C{index}",
                "name": record["name"],
                "valid_record": True,
                "performance_tested": False,
            }
        )
    return observations


def main() -> None:
    audit = read_json(AUDIT)
    history, rounds, record_inventory, candidate_inventory = [], [], [], []
    mechanism_counts, status_counts = Counter(), Counter()
    surfaces = {
        f"S{i}": {
            "eligible_rounds": [],
            "expanded_rounds": [],
            "materialized": [],
            "promoted": [],
            "unmaterialized_drafts": [],
        }
        for i in range(1, 11)
    }
    for rd in sorted((EXP / "opt").glob("round_*")):
        number = int(rd.name.split("_")[1])
        mining = rd / "mining" / rd.name
        bundle = read_json(mining / "bundle.json")
        marker = read_json(rd / "proposals_complete.json")
        ea = marker["evidence_audit"]
        records = read_jsonl(mining / "records.jsonl")
        eligible, expanded = set(), set()
        for index, pattern in enumerate(bundle["patterns"]):
            mechanism = AgentMechanism(pattern["signature"]["agent_mechanism"])
            route = {s.value for s in MECHANISM_SURFACES[mechanism]}
            eligible |= route
            if index in ea["expanded_patterns"]:
                expanded |= route
        for surface in eligible:
            surfaces[surface]["eligible_rounds"].append(number)
        for surface in expanded:
            surfaces[surface]["expanded_rounds"].append(number)
        for record in records:
            signature = record.get("signature") or {}
            mechanism_counts[signature.get("agent_mechanism", "unattributed")] += 1
            status_counts[signature.get("causal_status", "absent")] += 1
            record_inventory.append(
                {
                    "round": number,
                    **{
                        k: record.get(k)
                        for k in (
                            "run_id",
                            "instance_id",
                            "signature",
                            "detail",
                            "level_grounded",
                            "attribution_error_kind",
                            "trace_path",
                            "trace_sha256",
                        )
                    },
                }
            )
        outside = []
        for proposal in (p for p in audit["proposals"] if p["round"] == number):
            spec = proposal["spec"]
            full = spec["pattern_index"] in ea["expanded_patterns"]
            if not full:
                outside.append(proposal["id"])
            item = {k: proposal[k] for k in ("round", "id", "surface", "promoted", "source")}
            item.update(
                pattern_index=spec["pattern_index"],
                signature=spec["pattern"]["signature"],
                expanded_evidence=full,
            )
            item.update(
                {
                    k: spec[k]
                    for k in ("incumbent_behavior", "observed_failure", "behavioral_change")
                }
            )
            candidate_inventory.append(item)
            surfaces[proposal["surface"]]["materialized"].append(proposal["id"])
            if proposal["promoted"]:
                surfaces[proposal["surface"]]["promoted"].append(proposal["id"])
        rounds.append(
            {
                "round": number,
                "failed_runs": len(records),
                "pattern_count": len(bundle["patterns"]),
                "expanded_pattern_indices": ea["expanded_patterns"],
                "expanded_mechanisms": [
                    bundle["patterns"][i]["signature"]["agent_mechanism"]
                    for i in ea["expanded_patterns"]
                ],
                "expanded_pattern_count": len(ea["expanded_patterns"]),
                "system_prompt_chars": ea["system_prompt_chars"],
                "evidence_chars": ea["evidence_chars"],
                "history_chars": len(_render_history_block(history)),
                "operation_count": ea["operation_count"],
                "candidates_without_expanded_evidence": outside,
            }
        )
        history.append(load_round_history(rd, number, has_ledger=True))
    for i, draft in enumerate(audit["unmaterialized_drafts"], 1):
        surfaces[draft["surface"]]["unmaterialized_drafts"].append(f"C{i}")
    files = [
        "shrlm/optimization/proposal.py",
        "shrlm/optimization/proposal_evidence.py",
        "shrlm/optimization/attribution.py",
        "shrlm/optimization/clustering.py",
        "shrlm/optimization/taxonomy.py",
        "shrlm/optimization/digest.py",
        "shrlm/rlm_harness.py",
        "shrlm/runner.py",
        "rlm/core/rlm.py",
        "rlm/environments/local_repl.py",
    ]
    frozen = EXP / "eval/.launch/20260922T202236Z/source"
    provenance = []
    for name in files:
        current = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        old = hashlib.sha256((frozen / name).read_bytes()).hexdigest()
        provenance.append(
            {
                "path": name,
                "current_sha256": current,
                "frozen_sha256": old,
                "matches_frozen": current == old,
            }
        )
    result = {
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "experiment": str(EXP.relative_to(REPO)),
        "rounds": rounds,
        "surfaces": surfaces,
        "failed_runs": len(record_inventory),
        "mechanisms": dict(mechanism_counts),
        "causal_statuses": dict(status_counts),
        "grounded_records": sum(bool(r["level_grounded"]) for r in record_inventory),
        "candidates": candidate_inventory,
        "source_provenance": provenance,
        "notes": [
            "Eligible/expanded surface counts use the inspected taxonomy, verified against the frozen source.",
            "A candidate without expanded evidence can still see the compact inventory; it does not necessarily have no evidence.",
            "Repeated attempts are not distinct tasks. Counts describe this experiment, not all domains.",
        ],
    }
    assert result["failed_runs"] == 239
    assert len(candidate_inventory) == 16
    assert sum(p["promoted"] for p in candidate_inventory) == 5
    assert sum(not p["expanded_evidence"] for p in candidate_inventory) == 7
    assert all(p["matches_frozen"] for p in provenance)
    (OUT / "census.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "mining-records.json").write_text(json.dumps(record_inventory, indent=2) + "\n")
    (OUT / "runtime-probes.json").write_text(json.dumps(runtime_probes(audit), indent=2) + "\n")
    with (OUT / "surface-funnel.csv").open("w") as stream:
        fields = [
            "surface",
            "eligible_rounds",
            "expanded_rounds",
            "materialized",
            "promoted",
            "unmaterialized_drafts",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for surface, values in surfaces.items():
            writer.writerow(
                {"surface": surface, **{key: len(value) for key, value in values.items()}}
            )
    print(
        json.dumps(
            {
                "failed_runs": result["failed_runs"],
                "statuses": dict(status_counts),
                "mechanisms": dict(mechanism_counts),
                "surfaces": surfaces,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
