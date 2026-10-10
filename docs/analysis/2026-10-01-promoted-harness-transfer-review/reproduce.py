"""Recompute this review's measurements offline; never execute generated trace code.

Run from the repository root: .venv/bin/python docs/analysis/
2026-10-01-promoted-harness-transfer-review/reproduce.py
"""

import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[2]
OUT = ROOT / "experiment_oolong_pairs_dsv4f_20260928_174735_3rounds"


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def walk_tree(trace, node="r"):
    # Canonical child edges only: llm_observations can duplicate these records.
    for i, iteration in enumerate((trace.get("metadata") or {}).get("iterations", [])):
        for b, block in enumerate(iteration.get("code_blocks", [])):
            yield node, i, b, block, iteration
            for c, child in enumerate(block["result"].get("rlm_calls", [])):
                yield from walk_tree(child, f"{node}/i{i}b{b}c{c}")


def pair_score(pairs, gold):
    return {
        "pairs": len(pairs),
        "missing": len(gold - pairs),
        "extra": len(pairs - gold),
        "f1_unrounded": 2 * len(pairs & gold) / (len(pairs) + len(gold)) if pairs or gold else 1.0,
    }


def pairs_from_printed_output(output):
    return {tuple(sorted(map(int, match))) for match in re.findall(r"\((\d+),\s*(\d+)\)", output)}


summary = {"conditions": {}, "selected_intermediate_answers": [], "validation": []}
traces = {}
instances = {}
for condition in ("initial", "sh_rlm"):
    directory = OUT / "eval" / condition / "oolong_pairs_short/round_00"
    rows = read_rows(directory / "runs.jsonl")
    instances[condition] = {r["id"]: r for r in read_rows(directory / "instances.jsonl")}
    assert len(rows) == 30
    assert len({(r["instance_id"], r["attempt"]) for r in rows}) == 30
    iterations, loads, redirects = [], [], []
    for row in rows:
        path = directory / row["trace_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["trace_sha256"]
        trace = read_json(path)
        traces[condition, row["run_id"]] = trace
        iterations.append(len(trace["metadata"]["iterations"]))
        for node, i, b, block, iteration in walk_tree(trace):
            for event in block["result"].get("skill_loads", []):
                loads.append(
                    {
                        "run": row["run_id"],
                        "node": node,
                        "iteration": i,
                        "block": b,
                        "event": event,
                        "code": block["code"],
                    }
                )
            if (
                b == 0
                and iteration.get("trace_metrics", {}).get("answer_event") == "answer_redirected"
            ):
                redirects.append({"run": row["run_id"], "node": node, "iteration": i})
    summary["conditions"][condition] = {
        "attempts": len(rows),
        "exact": sum(r["passed"] for r in rows),
        "mean_f1": statistics.mean(r["verdict"].get("quality", {}).get("value", 0) for r in rows),
        "total_cost": sum(r["cost"] for r in rows),
        "total_run_seconds": sum(r["execution_time"] for r in rows),
        "mean_root_iterations_including_fallback": statistics.mean(iterations),
        "median_root_iterations_including_fallback": statistics.median(iterations),
        "exact_attempt_ids": sorted(r["run_id"] for r in rows if r["passed"]),
        "exact_task_ids": sorted({r["instance_id"] for r in rows if r["passed"]}),
        "wrong_format": sorted(r["run_id"] for r in rows if r["cause"] == "wrong_format"),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "skill_loads": loads,
        "answer_redirects": redirects,
    }
assert instances["initial"] == instances["sh_rlm"]

# These operations print the entire intermediate pair set; manually verified,
# unlike arbitrary stdout snippets that might only show a preview.
for run, iteration in [
    ("oolong-t12-w9-7fd1c6c04d9c0437__a02", 17),
    ("oolong-t11-w10-68f90f0d94d504da__a02", 25),
    ("oolong-t13-w10-a393091893c98516__a03", 10),
    ("oolong-t13-w10-a393091893c98516__a03", 21),
]:
    trace = traces["sh_rlm", run]
    output = trace["metadata"]["iterations"][iteration]["code_blocks"][0]["result"]["stdout"]
    gold = {tuple(p) for p in instances["sh_rlm"][run.split("__")[0]]["gold_pairs"]}
    summary["selected_intermediate_answers"].append(
        {
            "run": run,
            "iteration": iteration,
            "block": 0,
            **pair_score(pairs_from_printed_output(output), gold),
        }
    )

# Controlled offline reconstruction: retain the exact four child outputs and
# only restore the task's actual predicate with multiplicity intact.
run = "oolong-t17-w9-cc7785997a8db97f__a01"
labels = defaultdict(list)
for child in traces["sh_rlm", run]["metadata"]["iterations"][5]["code_blocks"][0]["result"][
    "rlm_calls"
]:
    for uid, values in json.loads(child["response"]).items():
        labels[int(uid)].extend(values)
a = {uid for uid, values in labels.items() if values.count("numeric value") == 1}
b = {
    uid
    for uid, values in labels.items()
    if "location" in values and "description and abstract concept" in values
}
pairs = {tuple(sorted((u, v))) for u in a for v in b if u != v}
gold = {tuple(p) for p in instances["sh_rlm"][run.split("__")[0]]["gold_pairs"]}
summary["predicate_only_reconstruction"] = {
    "run": run,
    "labels": sum(map(len, labels.values())),
    "role_a": sorted(a),
    "role_b": sorted(b),
    **pair_score(pairs, gold),
}

# Exhaustive tiny-universe verification of the pair-count defect, plus a
# witness that the reverse product adds nothing after canonicalization.
universe = range(4)
sets = [{i for i in universe if mask & (1 << i)} for mask in range(16)]
wrong_counts = 0
for a in sets:
    for b in sets:
        pairs = {tuple(sorted((u, v))) for u in a for v in b if u != v}
        reverse = {tuple(sorted((u, v))) for u in b for v in a if u != v}
        overlap = len(a & b)
        assert pairs == reverse
        assert len(pairs) == len(a) * len(b) - overlap - overlap * (overlap - 1) // 2
        wrong_counts += len(pairs) != len(a) * len(b) - overlap
summary["formula_check"] = {
    "set_pairs_checked": len(sets) ** 2,
    "promoted_formula_failures": wrong_counts,
    "minimal_witness": {"a": [1, 2], "b": [1, 2], "actual_pairs": [[1, 2]], "promoted_expected": 2},
}

for round_data in read_json(OUT / "latest-audit.json")["rounds"]:
    if not round_data["promoted"]:
        continue
    candidate = round_data["measured_candidates"][0]
    baseline = round_data["validation_baseline"]
    summary["validation"].append(
        {
            "round": round_data["round"],
            "candidate": candidate["subject_id"],
            "exact": candidate["exact"],
            "baseline_f1": baseline["quality"]["mean"],
            "candidate_f1": candidate["quality_and_cost"]["quality"]["mean"],
            "cost": candidate["resource_bands"]["mean_cost"],
        }
    )

summary["r7_validation_skill_loads"] = []
directory = OUT / "opt/round_07/validation/round_07/r07-c01-s10/heldout/round_00"
for row in read_rows(directory / "runs.jsonl"):
    for node, i, b, block, _ in walk_tree(read_json(directory / row["trace_path"])):
        for event in block["result"].get("skill_loads", []):
            summary["r7_validation_skill_loads"].append(
                {"run": row["run_id"], "node": node, "iteration": i, "block": b, "event": event}
            )

(ART / "verified-measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            "hash_verified_traces": len(traces),
            "conditions": {
                c: {k: v for k, v in s.items() if k in ("exact", "mean_f1", "total_cost")}
                for c, s in summary["conditions"].items()
            },
            "predicate_only_reconstruction": summary["predicate_only_reconstruction"],
            "formula_check": summary["formula_check"],
            "r7_validation_skill_loads": len(summary["r7_validation_skill_loads"]),
        },
        indent=2,
    )
)
