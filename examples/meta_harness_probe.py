"""Small, opt-in execution probes; no benchmark downloads or optimization runs.

Default: uv run python -m examples.meta_harness_probe --out-dir /tmp/harness-probe
Azure: add --live (eight executions, one free-choice proposal, $2 total ceiling).
Scripted controls demonstrate wiring, not model discovery or benchmark improvement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from shrlm.harness_identity import harness_hash
from shrlm.optimization.candidates import load_candidates
from shrlm.optimization.promotion import plan_batch
from shrlm.optimization.proposal import ProposerConfig, propose_round
from shrlm.optimization.types import iter_nodes
from shrlm.optimization.walker import build_call_tree
from shrlm.rlm_harness import H0, Harness
from shrlm.runner import build_harnessed_rlm
from tests.mock_lm import MockLM

HELPER_SOURCE = '''def count_events(records):
    """Count occurrences by key without discarding repeated events."""
    counts = {}
    for record in records:
        key = record["key"]
        counts[key] = counts.get(key, 0) + 1
    print("PROBE_HELPER_INVOKED")
    return counts
'''
SKILL = {
    "name": "normalize_measurements",
    "description": "Consult before combining measurements in mixed units.",
    "body": (
        "1. Preserve each value and its unit until conversion.\n"
        "2. Convert to meters using scales = {'m': 1, 'cm': 0.01}.\n"
        "3. Sum the converted values; verify all units have a scale."
    ),
}


def cases_for(family: str) -> dict[str, dict[str, Any]]:
    if family == "counts":
        return {
            "target": {
                "context": {
                    "records": [{"key": "alpha"}, {"key": "alpha"}, {"key": "beta"}],
                    "count": 2,
                },
                "question": "Return a JSON list of sorted keys occurring exactly context['count'] times. Count every record.",
                "expected": ["alpha"],
            },
            "protected": {
                "context": {"records": [{"key": "alpha"}, {"key": "beta"}], "count": 1},
                "question": "Return a JSON list of sorted keys occurring exactly context['count'] times. Count every record.",
                "expected": ["alpha", "beta"],
            },
        }
    if family != "units":
        raise ValueError(f"Unknown probe family: {family}")
    return {
        name: {
            "context": {
                "records": [
                    {"key": "alpha", "value": value, "unit": unit},
                    {"key": "beta", "value": 0.5, "unit": "m"},
                ]
            },
            "question": "Return the total length in meters as a JSON number. A centimeter is 0.01 meters.",
            "expected": 2.0,
        }
        for name, value, unit in [("target", 150, "cm"), ("protected", 1.5, "m")]
    }


def proposal_inputs(family: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Author two edits and a complete synthetic failure operation, never gold answers."""
    counts = family == "counts"
    operation = (
        "keys = set(record['key'] for record in context['records'])\n"
        "counts = {key: 1 for key in keys}"
        if counts
        else "total = sum(record['value'] for record in context['records'])"
    )
    pattern = {
        "signature": {
            "verifier_cause": "wrong_value",
            "failing_level": "root",
            "causal_status": "causal",
            "agent_mechanism": "other",
        },
        "support": 1,
        "instance_support": 1,
        "actionability": 1.0,
    }
    evidence = {
        "patterns": {
            0: {
                "run_id": "constructed-failure",
                "resolution": "unresolved",
                "agent_mechanism_detail": "Multiplicity is discarded before the occurrence predicate."
                if counts
                else "Values are combined without preserving their unit semantics.",
                "verification_limits": "Constructed operation; intermediate semantics supplied by fixture author.",
                "task_question": cases_for(family)["target"]["question"],
                "trace": {
                    "snippets": [
                        {
                            "node_id": "r",
                            "iteration_index": 1,
                            "code_block_index": 0,
                            "code": operation,
                            "code_complete": True,
                            "reason": "cited operation",
                        }
                    ]
                },
            }
        }
    }
    reference = hashlib.sha256(
        json.dumps(["constructed-failure", "r", 1, 0, "code"]).encode()
    ).hexdigest()[:16]
    capability = (
        {
            "kind": "repl_helper",
            "dict": "repl_helpers",
            "name": "count_events",
            "source": HELPER_SOURCE,
        }
        if counts
        else {"kind": "skills", **SKILL}
    )
    caller = (
        "For occurrence predicates, call count_events(context['records']) before filtering. Preserve multiplicity and apply the task's requested count."
        if counts
        else "Before aggregating measurements, use load_skill('normalize_measurements') and apply its unit conversion to the current records."
    )
    candidates = [
        {
            "pattern_index": 0,
            "surface": surface,
            "activation_pair": "invoke",
            "revision": None,
            "incumbent_behavior": "The demonstrated computation discards information required by the task.",
            "observed_failure": evidence["patterns"][0]["agent_mechanism_detail"],
            "behavioral_change": change,
            "edit": edit,
            "predicted_effect": "When this predicate needs retained information, compute with it before aggregating.",
            "regression_risks": [
                "Preserve inputs already handled correctly; additional invocation cost is unmeasured."
            ],
        }
        for surface, edit, change in [
            (
                "S8" if counts else "S10",
                capability,
                "Supply a reusable computation preserving the relevant information.",
            ),
            (
                "S3",
                {"kind": "text", "new_text": H0.execution_instruction + "\n" + caller},
                "Invoke that capability at the demonstrated aggregation operation.",
            ),
        ]
    ]
    response = {
        "format": "proposal-selection/v3",
        "selections": [
            {
                "pattern_index": 0,
                "surface": item["surface"],
                "activation_pair": "invoke",
                "reason": "Install the capability and explicitly invoke it on the task's records at aggregation.",
                "evidence_refs": [reference],
            }
            for item in candidates
        ],
        "candidates": candidates,
    }
    return {"patterns": [pattern]}, evidence, json.dumps(response)


def build_variants(out: Path, family: str) -> dict[str, Harness]:
    bundle, evidence, response = proposal_inputs(family)
    propose_round(
        bundle,
        H0,
        MockLM(responses=[response]),
        out / "proposals",
        workdir=out / "work",
        evidence=evidence,
        config=ProposerConfig(k=2, max_attempts=1),
    )
    candidates, rejected = load_candidates(out / "proposals", H0)
    if rejected or len(candidates) != 2:
        raise RuntimeError(f"Constructed pair failed admission: {rejected}")
    batch = plan_batch(H0, candidates)
    assert batch.harness is not None, batch.kind
    return {
        "baseline": H0,
        "capability": next(c.harness for c in candidates if c.surface != "S3"),
        "caller": next(c.harness for c in candidates if c.surface == "S3"),
        "paired": batch.harness,
    }


def scripted_client(family: str) -> MockLM:
    """Interpret a narrow instruction cue, then execute real code on current input."""
    calls = 0

    def response(prompt: Any) -> str:
        nonlocal calls
        calls += 1
        text = json.dumps(prompt)
        if "NameError" in text:
            code = "answer['content'] = 'UNAVAILABLE'"
        elif family == "counts":
            code = (
                "counts = count_events(context['records'])"
                if "call count_events(" in text
                else "counts = {record['key']: 1 for record in context['records']}"
            )
            code += "\nimport json\nanswer['content'] = json.dumps(sorted(key for key, count in counts.items() if count == context['count']))"
        elif "load_skill('normalize_measurements')" in text:
            if calls == 1:
                return "```repl\nprint(load_skill('normalize_measurements'))\n```"
            code = "scales = {'m': 1, 'cm': 0.01}\nanswer['content'] = str(sum(record['value'] * scales[record['unit']] for record in context['records']))"
        else:
            code = "answer['content'] = str(sum(record['value'] for record in context['records']))"
        return "```repl\n" + code + "\nanswer['ready'] = True\n```"

    return MockLM(response_fn=response)


def run_case(
    harness: Harness, family: str, case: dict[str, Any], out: Path, client_factory: Any = None
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    factory = client_factory or (lambda: scripted_client(family))
    client = factory()
    with patch("rlm.core.rlm.get_client", lambda *args, **kwargs: client):
        runner = build_harnessed_rlm(
            harness,
            backend="openai",
            backend_kwargs={"model_name": client.model_name},
            max_iterations=3,
            max_depth=1,
            max_timeout=90,
            log_dir=str(out / "logs"),
        )
        run = runner.completion(case["context"], root_prompt=case["question"])
    trace = run.completion.to_dict()
    (out / "trace.json").write_text(json.dumps(trace, indent=2, default=str))
    stdout = "\n".join(
        block.stdout
        for node in iter_nodes(build_call_tree(run.completion))
        for iteration in node.iterations
        for block in iteration.code_blocks
    )
    try:
        actual = json.loads(run.completion.response)
    except (ValueError, TypeError):
        actual = run.completion.response
    result = {
        "harness_hash": harness_hash(harness),
        "expected": case["expected"],
        "actual": actual,
        "passed": actual == case["expected"],
        "activated": "PROBE_HELPER_INVOKED" in stdout
        if family == "counts"
        else run.metrics["skill_load_count"] > 0,
        "trace": str(out / "trace.json"),
        "metrics": run.metrics,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@dataclass
class ProbeBudget:
    """Reserve each SDK request, including retries and nested calls, before dispatch."""

    input_price: float
    output_price: float
    max_requests: int = 32
    requests: int = 0
    reserved_usd: float = 0.0
    lock: Any = field(default_factory=threading.Lock)

    def reserve(self, request: dict[str, Any]) -> None:
        messages = request["messages"]
        size = len(json.dumps(messages).encode())
        output = request.get("max_completion_tokens", request.get("max_tokens"))
        if (
            size > 262144
            or len(messages) > 64
            or not isinstance(output, int)
            or not 0 < output <= 4096
        ):
            raise RuntimeError("Probe request exceeds its bounded text/output contract")
        # UTF-8 byte count overestimates text tokens; reserve extra chat framing too.
        ceiling = (
            (size + 1024 + 128 * len(messages)) * self.input_price + output * self.output_price
        ) / 1e6
        with self.lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("Probe request limit reached")
            if self.reserved_usd + ceiling > 2.0:
                raise RuntimeError("Probe spend limit would be exceeded")
            self.requests += 1
            self.reserved_usd += ceiling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "live": args.live,
        "supplied_pairs": True,
        "results": [],
        "interpretation": "Constructed comparisons; no benchmark or autonomous-discovery claim.",
    }
    clients = []
    budget = None
    factory = None
    if args.live:
        from rlm.clients.azure_foundry import AzureFoundryClient
        from shrlm.experiment.config import backend_kwargs_for, load_config

        config = load_config(path="configs/experiment_oolong_pairs_DeepSeekV4Flash.toml")
        kwargs = backend_kwargs_for(config, "runner")
        budget = ProbeBudget(
            kwargs["pricing"]["input_per_million"], kwargs["pricing"]["output_per_million"]
        )
        kwargs["sampling_args"]["max_tokens"] = 4096

        def factory():
            client = AzureFoundryClient(**kwargs, max_retries=0)
            create = client.client.chat.completions.create
            async_create = client.async_client.chat.completions.create

            def guarded(**request):
                budget.reserve(request)
                return create(**request)

            async def guarded_async(**request):
                budget.reserve(request)
                return await async_create(**request)

            client.client.chat.completions.create = guarded
            client.async_client.chat.completions.create = guarded_async
            clients.append(client)
            return client

    try:
        for family in ["counts"] if args.live else ["counts", "units"]:
            variants = build_variants(args.out_dir / family, family)
            for name, harness in variants.items():
                for case_name, case in cases_for(family).items():
                    try:
                        result = run_case(
                            harness, family, case, args.out_dir / family / name / case_name, factory
                        )
                    except Exception as error:
                        result = {"passed": False, "activated": None, "error": type(error).__name__}
                    report["results"].append(
                        {"family": family, "variant": name, "case": case_name, **result}
                    )
        if args.live:
            assert factory is not None
            bundle, evidence, _ = proposal_inputs("counts")
            try:
                result = propose_round(
                    bundle,
                    H0,
                    factory(),
                    args.out_dir / "free-choice" / "proposals",
                    workdir=args.out_dir / "free-choice" / "work",
                    evidence=evidence,
                    config=ProposerConfig(k=2, max_attempts=1),
                )
                report["free_choice"] = {
                    "surfaces": [p.surface for p in result.written],
                    "caveat": "One synthetic opportunity; pair generation was not required.",
                }
            except Exception as error:
                report["free_choice"] = {"error": type(error).__name__}
    finally:
        report["sdk_requests"] = budget.requests if budget else 0
        report["reserved_cost_ceiling_usd"] = budget.reserved_usd if budget else 0.0
        report["reported_usage"] = [client.get_usage_summary().to_dict() for client in clients]
        for client in clients:
            client.client.close()
            asyncio.run(client.async_client.close())
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                "report": str(args.out_dir / "report.json"),
                "sdk_requests": report["sdk_requests"],
                "reserved_cost_ceiling_usd": report["reserved_cost_ceiling_usd"],
            }
        )
    )


if __name__ == "__main__":
    main()
