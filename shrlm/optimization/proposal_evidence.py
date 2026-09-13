"""Read-only proposal context from held-in evidence and aggregate validation outcomes.

Never write derived metrics into persisted verifier results, bundles, or ledgers.
The two entry points deliberately keep held-in trace excerpts separate from
held-out aggregates; history exports no task, answer, or trace text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlm.core.types import RLMChatCompletion
from shrlm.optimization.digest import head_tail
from shrlm.optimization.driver import canonical_manifest_entries, load_manifest, load_round
from shrlm.optimization.taxonomy import VerifierCause
from shrlm.optimization.types import Verdict

UNSCORED_CAUSES = {
    VerifierCause.RUNTIME_ERROR,
    VerifierCause.RESOURCE_TERMINATED,
    VerifierCause.WRONG_FORMAT,
    VerifierCause.CONTENT_FILTERED,
}


def pair_diagnostics(verdict: Verdict) -> dict[str, Any]:
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    metrics = recorded_pair_metrics(verdict)
    return {"scored": metrics is not None, "metrics": metrics or "unavailable"}


def aggregate_pair_diagnostics(verdicts: Sequence[Verdict]) -> dict[str, Any]:
    from shrlm.environments.oolong_pairs import recorded_pair_metrics

    measured = [recorded_pair_metrics(verdict) for verdict in verdicts]
    known = [metrics for metrics in measured if metrics is not None]
    unaccounted = sum(
        metrics is None and verdict.cause not in UNSCORED_CAUSES
        for metrics, verdict in zip(measured, verdicts, strict=True)
    )
    return {
        "n_attempts": len(verdicts),
        "n_scored": len(known),
        "n_unscored": len(verdicts) - len(known),
        "n_missing_legacy_metrics": unaccounted,
        "mean_f1_all_attempts": (
            sum(float(m["f1"]) for m in known) / len(verdicts)
            if verdicts and not unaccounted
            else None
        ),
        "missing_total_measured": sum(int(m["missing"]) for m in known) if known else None,
        "extra_total_measured": sum(int(m["extra"]) for m in known) if known else None,
        "counts_denominator": len(known),
    }


def read_persisted_round(
    path: Path,
) -> tuple[
    list[tuple[dict[str, Any], RLMChatCompletion]],
    list[Verdict],
    dict[str, Any],
    list[dict[str, Any]],
]:
    match = re.fullmatch(r"round_(\d+)", path.name)
    if match is None:
        raise ValueError(f"not a persisted round directory: {path}")
    return load_round(path.parent, int(match[1]))


def trace_excerpt(completion: RLMChatCompletion) -> dict[str, Any]:
    iterations = (completion.metadata or {}).get("iterations", [])
    blocks = [block for iteration in iterations for block in iteration.get("code_blocks", [])]
    if not blocks:
        return {"observation": "trace excerpt unavailable"}
    # Two bounded snippets, explicitly observations rather than a causal claim.
    selected = [blocks[0]] if len(blocks) == 1 else [blocks[0], blocks[-1]]
    return {
        "observation": "partial observed behavior; labels and causal effectiveness are not verified",
        "root_code": [head_tail(str(block.get("code", "")), 1800) for block in selected],
        "coverage_observations": [
            head_tail(str((block.get("result") or {}).get("stdout", "")), 600) for block in selected
        ],
        "observed_child_calls": sum(
            len((block.get("result") or {}).get("rlm_calls", [])) for block in blocks
        ),
    }


def load_proposal_evidence(
    mining_round_path: Path, bundle: dict[str, Any], *, bundle_path: Path | None = None
) -> dict[str, Any]:
    """Join one selected bundle's records to SHA-verified held-in runs only."""
    bundle_path = bundle_path or mining_round_path / "bundle.json"
    if bundle_path.exists() and json.loads(bundle_path.read_text()) != bundle:
        raise ValueError("selected proposal bundle differs from its persisted evidence")
    records_path = bundle_path.parent / "records.jsonl"
    records = (
        [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        if records_path.exists()
        else []
    )
    instances_path = mining_round_path / "instances.jsonl"
    instances = (
        [json.loads(line) for line in instances_path.read_text().splitlines() if line.strip()]
        if instances_path.exists()
        else []
    )
    by_instance = {str(instance["id"]): instance for instance in instances}
    runs, verdicts, entries = [], [], []
    if (mining_round_path / "runs.jsonl").exists():
        runs, verdicts, _, entries = read_persisted_round(mining_round_path)
    contexts = {}
    is_pairs = (bundle.get("config", {}).get("verifier_config") or {}).get(
        "environment"
    ) == "oolong_pairs"
    for index, pattern in enumerate(bundle.get("patterns", [])):
        representative_ids = pattern.get("representatives") or []
        candidates = [
            record
            for instance_id in representative_ids
            for record in records
            if str(record["instance_id"]) == str(instance_id)
            and record.get("signature") == pattern.get("signature")
        ]
        if not candidates:
            contexts[index] = {"diagnosis": "representative evidence unavailable"}
            continue
        record = candidates[0]
        detail = record.get("detail") or {}
        instance = by_instance.get(str(record["instance_id"]), {})
        verdict = Verdict.from_dict(record["verdict"])
        context = {
            "symptom_summary": head_tail(detail.get("symptom_summary", "unavailable"), 2000),
            "evidence_node_ids": detail.get("evidence_node_ids", []),
            "task_question": head_tail(str(instance.get("question", "unavailable")), 8000),
        }
        if is_pairs:
            context["pair_diagnostics"] = pair_diagnostics(verdict)
            if context["pair_diagnostics"]["scored"]:
                from shrlm.environments.oolong_pairs import extract_answer_pairs

                gold = set(extract_answer_pairs(verdict.gold) or [])
                produced = set(extract_answer_pairs(verdict.produced) or [])
                context["missing_examples"] = sorted(gold - produced)[:3]
                context["extra_examples"] = sorted(produced - gold)[:3]
        if record.get("run_id"):
            matches = [i for i, entry in enumerate(entries) if entry["run_id"] == record["run_id"]]
        else:
            matches = [
                i
                for i, entry in enumerate(entries)
                if str(entry["instance_id"]) == str(record["instance_id"])
                and entry["verdict"] == record["verdict"]
            ]
        context["trace"] = {"observation": "trace unavailable: missing or ambiguous run linkage"}
        if len(matches) == 1:
            selected = matches[0]
            entry = entries[selected]
            if str(entry["instance_id"]) != str(record["instance_id"]):
                raise ValueError("record instance_id disagrees with linked run")
            for key in ("trace_path", "trace_sha256"):
                if record.get(key) and record[key] != entry.get(key):
                    raise ValueError(f"record {key} disagrees with manifest")
            if verdict.to_dict() != verdicts[selected].to_dict():
                raise ValueError("record verdict disagrees with linked run")
            context["trace"] = trace_excerpt(runs[selected][1])
        contexts[index] = context
    passing = []
    seen = set()
    for (instance, completion), verdict in zip(runs, verdicts, strict=True):
        if verdict.passed and str(instance["id"]) not in seen:
            seen.add(str(instance["id"]))
            passing.append(
                {
                    "instance_id": str(instance["id"]),
                    "task_question": head_tail(str(instance.get("question", "unavailable")), 8000),
                    "trace": trace_excerpt(completion),
                }
            )
            if len(passing) == 2:
                break
    return {
        "patterns": contexts,
        "passing": passing,
        "passing_status": "observed held-in examples"
        if passing
        else "no passing trace evidence available",
    }


def structural_execution_error(verdict: Verdict) -> dict[str, str] | None:
    if verdict.cause is not VerifierCause.RUNTIME_ERROR:
        return None
    exception_type, _, message = verdict.detail.partition(": ")
    # Fixed vocabulary only: even the suffix of a familiar error can contain
    # a provider response or task payload. Never echo arbitrary exception text.
    messages = {
        "AnswerDecision.accept() missing 1 required positional argument: 'answer'",
        "type object 'AnswerDecision' has no attribute 'reject'",
        "expected string or bytes-like object, got 'tuple'",
        "'tuple' object has no attribute 'strip'",
        "'tuple' object has no attribute 'splitlines'",
    }
    return {
        "exception_type": (
            exception_type
            if exception_type
            in {
                "TypeError",
                "AttributeError",
                "ValueError",
                "RuntimeError",
                "TimeoutError",
                "KeyError",
                "IndexError",
                "NameError",
            }
            else "other exception"
        ),
        "message": message if message in messages else "message omitted: may contain task data",
    }


def validation_history_diagnostics(validation_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Export only aggregate numbers and a sanitized structural runtime error."""
    links = record.get("links") or {}
    split = (links.get("splits") or {}).get("heldout") or {}
    if not split.get("round_dir"):
        return {"status": "validation diagnostics unavailable"}
    path = (validation_path / split["round_dir"]).resolve()
    if not path.is_relative_to(validation_path.resolve()):
        raise ValueError("validation history link escapes its round")
    if not (path / "runs.jsonl").exists():
        return {"status": "validation diagnostics unavailable"}
    # Durable verdicts contain all exported fields, including structural errors.
    # History must not open large/transient trace bodies just to report numbers.
    instances = [
        json.loads(line)
        for line in (path / "instances.jsonl").read_text().splitlines()
        if line.strip()
    ]
    entries = canonical_manifest_entries(
        load_manifest(path.parent, int(path.name.removeprefix("round_"))), instances
    )
    verdicts = [Verdict.from_dict(entry["verdict"]) for entry in entries]
    diagnostics = {
        "n_attempts": len(verdicts),
        "exact_passes": sum(v.passed for v in verdicts),
        "n_runtime_errors": sum(v.cause is VerifierCause.RUNTIME_ERROR for v in verdicts),
        "total_cost": sum(float(e["cost"]) for e in entries if e.get("cost") is not None),
    }
    # Environment identity comes from the persisted subject contract, not task data.
    contract_path = validation_path / str(record["subject_id"]) / "evaluation.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    if (contract.get("verifier_config") or {}).get("environment") == "oolong_pairs":
        diagnostics["pairs"] = aggregate_pair_diagnostics(verdicts)
    diagnostics["first_runtime_error"] = next(
        (
            structural_execution_error(verdict)
            for verdict in verdicts
            if verdict.cause is VerifierCause.RUNTIME_ERROR
        ),
        None,
    )
    return diagnostics
